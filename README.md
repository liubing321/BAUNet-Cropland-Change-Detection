# BAUNet: Fine-Grained Cropland Change Detection

This repository provides selected code and research resources associated with the manuscript:

**Fine-Grained Cropland Change Detection from Sub-Meter Henan-1 Imagery Using a Cross-Scale Attention Network**

## Overview

BAUNet is developed for fine-grained cropland change detection in very-high-resolution remote sensing imagery, with particular attention to small, fragmented, and boundary-sensitive change regions under cross-season observation conditions.

The proposed framework contains a shared-weight Siamese encoder, cross-scale feature enhancement, and progressive spatial reconstruction for parcel-level cropland change detection.

## Available Resources

The current repository provides selected research resources, including:

- Data loading utilities
- Prediction and inference utilities
- Dataset organization examples
- Experimental information
- Selected BCD dataset samples

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

A representative subset of BCD will be made publicly available for academic research.

## Public Benchmark Datasets

The experiments also use the following public cropland change detection datasets:

- CLCD
- PX-CLCD

Please refer to their original repositories for data access.

## Code

Currently available:

- `dataloader.py`: dataset loading utilities
- `predict.py`: prediction utilities

Additional research resources may be released in future updates.

## Data Availability

A representative subset of the BCD dataset will be provided in this repository.

The complete BCD dataset may be available from the corresponding author for academic research purposes upon reasonable request and subject to applicable data-use restrictions.

## Citation

Citation information will be added after publication.

## Contact

For academic inquiries regarding BAUNet or the BCD dataset, please contact the authors of the manuscript.
