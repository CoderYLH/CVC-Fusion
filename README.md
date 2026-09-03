# CVC-Fusion
A lightweight Chicken Vocalization Classification-Fusion (CVC-Fusion) model based on multi-domain feature fusion to recognize chicken vocalizations under heat stress and thermoneutral conditions.

CVC-Fusion training code
This repository contains the minimal code required to train the CVC-Fusion model. CVC-Fusion combines a one-dimensional MobileNetV2 encoder for acoustic signals with a MobileViTv3 image encoder for spectrograms. Hybrid Feature Selection (HFS) blocks adaptively combine local and global image features before the two encoder outputs are projected and fused by summation.

The repository intentionally excludes the dataset, dataset-partitioning scripts, offline data-augmentation scripts, trained weights, evaluation and visualization programs, logs, and unrelated single-path baseline models.

Environment
Python 3.10 or later is recommended. Install a PyTorch build suitable for the available CPU or CUDA system, and then install the remaining requirements:

pip install -r requirements.txt
Prepared data
The code expects data that have already been split into training and validation subsets. Offline-augmented samples should be placed only in the training subset. Each one-dimensional signal is stored in the first column of an XLSX file, and its paired spectrogram is stored as a PNG file with the same stem. The default expected signal length is 100, in agreement with the model configuration reported in the manuscript. Class folder names are configured in configs/train.yaml.

data/
├── train/
│   ├── audio/
│   │   ├── 0/*.xlsx
│   │   └── 1/*.xlsx
│   └── images/
│       ├── 0/*.png
│       └── 1/*.png
└── validation/
    ├── audio/
    │   ├── 0/*.xlsx
    │   └── 1/*.xlsx
    └── images/
        ├── 0/*.png
        └── 1/*.png
The loader checks that every XLSX file has a paired PNG file. Labels are assigned in the order given by class_names; the default folders 0 and 1 therefore map to labels 0 and 1, respectively.

Preprocessing
No on-the-fly data augmentation is applied. Spectrograms are resized with bicubic interpolation so that the shorter side is 224 pixels, center-cropped to 224 × 224 pixels, converted to RGB tensors, and scaled to [0, 1]. These operations follow the source MobileViT resize, center-crop, and tensor- conversion pipeline. The loader checks that each one-dimensional input has the configured length of 100 values.

Training
Review the four data paths in configs/train.yaml, then run:

python train.py --config configs/train.yaml
The default configuration contains the settings used for the CVC-Fusion training pipeline: batch size 12, 100 epochs, SGD with momentum 0.9 and weight decay 0.0001, cosine learning-rate decay from 0.01 to 0.00001, 500 warm-up iterations, cross-entropy loss with label smoothing 0.1, gradient clipping, mixed-precision training, and exponential moving average updates.

Training writes the copied configuration, epoch metrics, the latest checkpoint, and the checkpoint with the highest validation accuracy to runs/cvc_fusion. Training can be resumed with:

python train.py --config configs/train.yaml --resume runs/cvc_fusion/checkpoint_last.pt
Source acknowledgement
The image encoder and supporting operations are adapted from Apple's CVNets and MobileViT implementations. The repository retains the applicable Apple license notice in LICENSE.
