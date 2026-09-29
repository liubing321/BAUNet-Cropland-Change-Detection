# predict.py
# DS+Edge 推理脚本（micro/global 评估）：累计全数据 TP/FP/FN 计算 IoU/F1/Precision/Recall

import os
import argparse
import cv2
import numpy as np
import torch
from tqdm import tqdm

from models.BCDNet import ChangeDetectionModel as Net
from Dataset.dataloader import create_dataloader


def parse_args():
    parser = argparse.ArgumentParser("Predict Change Map (DS+Edge, micro metrics)")

    parser.add_argument("--data_root", type=str, default="./Dataset/BCD_256")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--ckpt", type=str, required=True, help="best.pth/last.pth 路径")
    parser.add_argument("--out_dir", type=str, default="./predictions")
    parser.add_argument("--encoder", type=str, default="resnet50")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threshold", type=float, default=0.5, help="sigmoid 后阈值")

    # DS+Edge 模型参数（务必与训练一致）
    parser.add_argument("--msam_dim", type=int, default=256)
    parser.add_argument("--use_transformer", action="store_true",
                        help="训练若 use_transformer=True，这里也要加这个 flag")
    parser.add_argument("--no_umformer", action="store_true", help="关闭 Umformer deep（一般不建议）")

    # 保存开关
    parser.add_argument("--save_prob", action="store_true", help="保存概率图(0-255)")
    parser.add_argument("--save_edge", action="store_true", help="保存 edge 分支输出(0-255)")

    # 评估开关（micro/global）
    parser.add_argument("--eval", action="store_true", help="如果 label 存在，计算 micro IoU/F1/Pre/Rec")

    return parser.parse_args()


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def save_gray_png(path: str, img01: np.ndarray):
    """img01: [H,W] in {0,1} or [0,1] float"""
    img255 = (np.clip(img01, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(path, img255)


def load_checkpoint(model: torch.nn.Module, ckpt_path: str, device: torch.device):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"ckpt not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        print(f"[!] Unexpected keys: {len(unexpected)} (show first 10)\n  {unexpected[:10]}")
    if missing:
        print(f"[!] Missing keys: {len(missing)} (show first 10)\n  {missing[:10]}")

    print(f"[*] Loaded checkpoint: {ckpt_path}")
    model.to(device)
    model.eval()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(args.device)

    loader = create_dataloader(
        root_dir=args.data_root,
        split=args.split,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers,
        shuffle=False
    )
    in_channels = loader.dataset.num_channels if hasattr(loader.dataset, "num_channels") else 3

    # build model
    model = Net(
        encoder_name=args.encoder,
        in_channels=in_channels,
        out_classes=1,
        encoder_weights=None,  # 推理无需 imagenet 权重
        msam_dim=args.msam_dim,
        use_transformer=bool(args.use_transformer),
        use_umformer=not bool(args.no_umformer),
    )
    load_checkpoint(model, args.ckpt, device)

    # output dirs
    pred_dir = os.path.join(args.out_dir, args.split, "mask")
    prob_dir = os.path.join(args.out_dir, args.split, "prob")
    edge_dir = os.path.join(args.out_dir, args.split, "edge")
    ensure_dir(pred_dir)
    if args.save_prob:
        ensure_dir(prob_dir)
    if args.save_edge:
        ensure_dir(edge_dir)

    # micro/global accumulators
    tp_sum, fp_sum, fn_sum = 0.0, 0.0, 0.0
    has_labels = False

    pbar = tqdm(loader, desc=f"Predict[{args.split}]", ncols=110)
    for batch in pbar:
        image1 = batch["image1"].to(device)
        image2 = batch["image2"].to(device)
        names = batch.get("name", None)

        targets = None
        if args.eval and "label" in batch:
            has_labels = True
            lbl = batch["label"].to(device)           # [B,H,W]
            targets = lbl.unsqueeze(1).float()        # [B,1,H,W]

        out = model(image1, image2)

        # 兼容：多输出 or 单输出
        if isinstance(out, (tuple, list)):
            main_logits = out[0]
            edge_logits = out[3] if (len(out) >= 4) else None
        else:
            main_logits = out
            edge_logits = None

        probs = torch.sigmoid(main_logits)                     # [B,1,H,W]
        pred01 = (probs >= args.threshold).float()             # [B,1,H,W]

        # save
        pred_np = pred01.squeeze(1).cpu().numpy()
        prob_np = probs.squeeze(1).cpu().numpy()

        edge_prob_np = None
        if edge_logits is not None:
            edge_prob_np = torch.sigmoid(edge_logits).squeeze(1).cpu().numpy()

        bs = pred_np.shape[0]
        for i in range(bs):
            name = names[i] if names is not None else f"{i:06d}"
            # 有些 dataloader 的 name 可能是带后缀的路径，这里简单净化一下
            if isinstance(name, (list, tuple)):
                name = name[0]
            name = str(name)
            name = os.path.splitext(os.path.basename(name))[0]

            save_gray_png(os.path.join(pred_dir, f"{name}.png"), pred_np[i])
            if args.save_prob:
                save_gray_png(os.path.join(prob_dir, f"{name}.png"), prob_np[i])
            if args.save_edge and edge_prob_np is not None:
                save_gray_png(os.path.join(edge_dir, f"{name}.png"), edge_prob_np[i])

        # micro eval accumulate
        if args.eval and targets is not None:
            # pred01/targets: [B,1,H,W] float 0/1
            tp_sum += (pred01 * targets).sum().item()
            fp_sum += (pred01 * (1 - targets)).sum().item()
            fn_sum += ((1 - pred01) * targets).sum().item()

            # 实时显示当前 micro IoU
            micro_iou = tp_sum / (tp_sum + fp_sum + fn_sum + 1e-8)
            pbar.set_postfix({"micro_iou": f"{micro_iou:.4f}"})

    print("\n[*] Prediction saved to:", os.path.join(args.out_dir, args.split))

    if args.eval:
        if not has_labels:
            print("[!] --eval 开启但 batch 中没有 label 字段，跳过评估。")
            return

        iou = tp_sum / (tp_sum + fp_sum + fn_sum + 1e-8)
        precision = tp_sum / (tp_sum + fp_sum + 1e-8)
        recall = tp_sum / (tp_sum + fn_sum + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        print("[*] Eval metrics (micro/global over ALL pixels):")
        print(f"    IoU      : {iou:.6f}")
        print(f"    F1       : {f1:.6f}")
        print(f"    Precision: {precision:.6f}")
        print(f"    Recall   : {recall:.6f}")


if __name__ == "__main__":
    main()
