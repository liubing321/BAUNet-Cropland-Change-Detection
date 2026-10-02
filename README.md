# BAUNet: Fine-Grained Cropland Change Detection

This repository provides selected code, dataset resources, and research materials associated with the manuscript:

**Fine-Grained Cropland Change Detection from Sub-Meter Henan-1 Imagery Using a Cross-Scale Attention Network**

## Overview

BAUNet is developed for fine-grained cropland change detection in very-high-resolution remote sensing imagery, with particular attention to small, fragmented, and boundary-sensitive change regions under cross-season observation conditions.

The proposed framework contains a shared-weight Siamese encoder, cross-scale feature enhancement, and progressive spatial reconstruction for parcel-level cropland change detection.

## Available Resources

The current repository provides selected research resources, including:

- Data loading utilities
- Prediction and inference utilities
- Dataset organization information
- Experimental information
- Complete BCD dataset release

Core implementation details of the proposed model are not included in the current public version.

## BCD Dataset

BCD (Bi-temporal Cropland Change Detection Dataset) was constructed using 0.75 m Henan-1 satellite imagery over Kaifeng, Henan Province, China.

The complete dataset contains:

- 4,747 bi-temporal image pairs
- Image size: 256 × 256 pixels
- Pixel-level binary change annotations
- Training set: 2,848 pairs
- Validation set: 950 pairs
- Test set: 949 pairs

Each sample consists of:

- `A`: pre-change image
- `B`: post-change image
- `label`: binary change mask

The complete BCD dataset is publicly available through the **BCD Full Dataset v1.0.0** release:

https://github.com/liubing321/BAUNet-Cropland-Change-Detection/releases/tag/bcd-full-v1.0.0

## Public Benchmark Datasets

The experiments also use the following public cropland change detection datasets:

- CLCD: https://github.com/liumency/CropLand-CD
- PX-CLCD: https://github.com/lixint5/Peixian-Cultivated-land-Change-detection-dataset

Please refer to their original repositories for data access and licensing information.

## Code

Currently available:

- `dataloader.py`: dataset loading utilities
- `predict.py`: prediction and inference utilities

Additional research resources may be released in future updates.

## Data Availability

The complete BCD dataset, comprising 4,747 bi-temporal image pairs with corresponding pixel-level binary change annotations, is publicly available through the **BCD Full Dataset v1.0.0** release:

https://github.com/liubing321/BAUNet-Cropland-Change-Detection/releases/tag/bcd-full-v1.0.0

The released BCD dataset is licensed under the **Creative Commons Attribution 4.0 International License (CC BY 4.0)**.

License details:

https://creativecommons.org/licenses/by/4.0/

The CC BY 4.0 license applies to the released BCD dataset and does not apply to the original Henan-1 satellite imagery or to any third-party datasets referenced in this project.

For detailed licensing information, please see `DATA_LICENSE.md`.

## Citation

If you use the BCD dataset in your research, please cite:

Liu, B. (2026). *BCD Full Dataset v1.0.0* [dataset]. GitHub.  
https://github.com/liubing321/BAUNet-Cropland-Change-Detection/releases/tag/bcd-full-v1.0.0

Paper citation information will be added after publication.

## Contact

For academic inquiries regarding BAUNet or the BCD dataset, please contact the authors of the manuscript.
