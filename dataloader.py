"""
通用耕地变化检测数据加载器 (适配 Pure Numpy Transforms)
支持PX-CLCD等双时相遥感变化检测数据集
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Callable
import cv2
import logging
from utils.transforms import get_train_transforms, get_val_transforms

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ChangeDetectionDataset(Dataset):
    """变化检测数据集基类"""

    def __init__(
            self,
            root_dir: str,
            split: str = 'train',
            transform: Optional[Callable] = None,
            return_name: bool = False,
            img_size: int = 256,
            use_cache: bool = False
    ):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.return_name = return_name
        self.img_size = img_size
        self.use_cache = use_cache

        # 检查目录
        if not self.root_dir.exists():
            raise FileNotFoundError(f"数据集根目录不存在: {root_dir}")

        self.image1_dir = self.root_dir / split / 'A'
        self.image2_dir = self.root_dir / split / 'B'
        self.label_dir = self.root_dir / split / 'label'

        self._validate_dirs()
        self.samples = self._load_samples()

        # 内存缓存 (存储 uint8 numpy array 以节省内存)
        self.cache = {} if use_cache else None

        logger.info(f"加载 {split} 数据集: {len(self.samples)} 个样本")

    def _validate_dirs(self):
        """验证目录结构"""
        for dir_path in [self.image1_dir, self.image2_dir, self.label_dir]:
            if not dir_path.exists():
                raise FileNotFoundError(f"目录不存在: {dir_path}")

    def _load_samples(self) -> List[Dict]:
        """加载样本列表"""
        samples = []
        # 支持常见格式
        valid_exts = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}

        # 扫描 A 目录
        image_files = sorted([
            f for f in self.image1_dir.iterdir()
            if f.is_file() and f.suffix.lower() in valid_exts
        ], key=lambda x: x.name)

        for img1_path in image_files:
            filename = img1_path.stem
            suffix = img1_path.suffix

            # 假设 A/B/label 同名
            img2_path = self.image2_dir / f"{filename}{suffix}"
            label_path = self.label_dir / f"{filename}{suffix}"

            if not img2_path.exists() or not label_path.exists():
                logger.warning(f"样本不完整，跳过: {filename}")
                continue

            samples.append({
                'image1_path': str(img1_path),
                'image2_path': str(img2_path),
                'label_path': str(label_path),
                'name': filename
            })

        if not samples:
            logger.error(f"在 {self.split} 中未找到有效样本")

        return samples

    def _read_image_opencv(self, path: str) -> np.ndarray:
        """使用 OpenCV 读取图像 (输出 HWC RGB/BGR)"""
        # 1. 尝试读取
        img = cv2.imread(path, cv2.IMREAD_COLOR) # 默认读取为 BGR
        if img is None:
            # 尝试处理中文路径或特殊情况
            img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError(f"无法读取图像: {path}")

        # 2. BGR -> RGB (PyTorch 模型通常需要 RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img # 返回 uint8 [H, W, 3]

    def _load_data(self, sample: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """读取数据核心逻辑"""
        path1 = sample['image1_path']
        path2 = sample['image2_path']
        path_lbl = sample['label_path']

        # === 缓存逻辑 ===
        if self.use_cache and path1 in self.cache:
            return self.cache[path1], self.cache[path2], self.cache[path_lbl]

        # === 读取图像 (OpenCV) ===
        img1 = self._read_image_opencv(path1)
        img2 = self._read_image_opencv(path2)

        # === 读取标签 (灰度) ===
        label = cv2.imread(path_lbl, cv2.IMREAD_GRAYSCALE)
        if label is None:
            label = cv2.imdecode(np.fromfile(path_lbl, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

        # === 标签数值处理 ===
        # 变化检测通常要求标签为 0, 1。如果是 0, 255，需要转换。
        # 这是一个常见坑：255 输入 CrossEntropyLoss 会报错
        if label.max() > 1:
            label = (label > 127).astype(np.uint8) # 阈值化为 0 和 1

        # === 存入缓存 ===
        if self.use_cache:
            self.cache[path1] = img1
            self.cache[path2] = img2
            self.cache[path_lbl] = label

        return img1, img2, label

    def __getitem__(self, idx: int) -> Dict:
        """
        获取一个样本
        逻辑：读取 Numpy -> 增强 (包含转 Tensor) -> 返回字典
        """
        sample = self.samples[idx]

        try:
            # 1. 读取数据 (全部为 Numpy uint8)
            img1, img2, label = self._load_data(sample)

            # 2. 应用增强 pipeline
            # 注意：transform 的最后一步现在是 ToTensorTemporal，
            # 它会负责将 Numpy 转为 Tensor，并进行 HWC->CHW 转换
            if self.transform:
                img1, img2, label = self.transform(img1, img2, label)
            else:
                # 如果没有定义 transform（极少情况），需要手动转 Tensor 以防报错
                # 这种 fallback 逻辑保持基本兼容性
                img1 = torch.from_numpy(img1).float().permute(2, 0, 1) / 255.0
                img2 = torch.from_numpy(img2).float().permute(2, 0, 1) / 255.0
                label = torch.from_numpy(label).long()

            # 3. 此时 img1/img2 是 FloatTensor [C, H, W], label 是 LongTensor [H, W]

            # 4. 包装返回
            result = {
                'image1': img1,
                'image2': img2,
                'label': label
            }
            if self.return_name:
                result['name'] = sample['name']

            return result

        except Exception as e:
            logger.error(f"加载样本 {sample['name']} 失败: {e}")
            return self._get_empty_sample()

    def _get_empty_sample(self) -> Dict:
        """返回全零样本防止DataLoader崩溃"""
        return {
            'image1': torch.zeros((3, self.img_size, self.img_size)),
            'image2': torch.zeros((3, self.img_size, self.img_size)),
            'label': torch.zeros((self.img_size, self.img_size)).long(),
            'name': 'error_sample',
            'is_empty': True
        }

    def __len__(self) -> int:
        return len(self.samples)

    def visualize_sample(self, idx: int):
        """可视化辅助函数"""
        import matplotlib.pyplot as plt

        # 获取 Tensor 数据
        data = self.__getitem__(idx)
        img1 = data['image1'] # [C, H, W]
        img2 = data['image2']
        label = data['label'] # [H, W]

        # 反归一化 (简单的 clip 用于显示)
        # 注意：如果 Transform 里用了 Normalize(mean, std)，这里显示颜色可能会怪
        # 这里假设已经归一化到了 0-1 或标准化附近

        def to_numpy_img(tensor):
            img = tensor.permute(1, 2, 0).cpu().numpy() # CHW -> HWC
            # 简单的 min-max 归一化用于显示
            img = (img - img.min()) / (img.max() - img.min() + 1e-6)
            return (img * 255).astype(np.uint8)

        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        ax[0].imshow(to_numpy_img(img1))
        ax[0].set_title("Image T1")
        ax[1].imshow(to_numpy_img(img2))
        ax[1].set_title("Image T2")
        ax[2].imshow(label.cpu().numpy(), cmap='gray')
        ax[2].set_title("Label (Change)")
        plt.show()


class DataLoaderFactory:
    """工厂类保持不变，逻辑是通用的"""

    @staticmethod
    def create_dataloader(
            root_dir: str,
            split: str = 'train',
            batch_size: int = 8,
            num_workers: int = 4,
            img_size: int = 256,
            shuffle: Optional[bool] = None,
            use_cache: bool = False,
            distributed: bool = False,
            transform: Optional[Callable] = None, # 允许外部传入
            **kwargs
    ) -> DataLoader:

        if shuffle is None:
            shuffle = (split == 'train')

        # 如果外部没传 transform，使用默认的
        if transform is None:
            if split == 'train':
                transform = get_train_transforms(img_size=img_size)
            else:
                transform = get_val_transforms(img_size=img_size)

        dataset = ChangeDetectionDataset(
            root_dir=root_dir,
            split=split,
            transform=transform,
            img_size=img_size,
            use_cache=use_cache,
            return_name=True # 默认返回文件名方便调试
        )

        sampler = None
        if distributed:
            sampler = DistributedSampler(dataset, shuffle=shuffle)
            shuffle = False

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            sampler=sampler,
            pin_memory=True,
            drop_last=(split == 'train'),
            collate_fn=collate_fn
        )


def collate_fn(batch: List[Dict]) -> Dict:
    valid_batch = [x for x in batch if 'is_empty' not in x]
    if not valid_batch:
        return {}

    return {
        'image1': torch.stack([x['image1'] for x in valid_batch]),
        'image2': torch.stack([x['image2'] for x in valid_batch]),
        'label': torch.stack([x['label'] for x in valid_batch]), # Stack会自动处理 [H,W] -> [B,H,W]
        'name': [x['name'] for x in valid_batch]
    }

# 快捷入口函数保持不变...
def create_dataloader(root_dir, **kwargs):
    return DataLoaderFactory.create_dataloader(root_dir, **kwargs)



