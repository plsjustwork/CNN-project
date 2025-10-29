# CNN-Project: CIFAR-10 Classification

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## Table of Contents
- [Overview](#overview)
- [Dataset](#data)
- [Training](#training)
- [Results](#results)
- [Models](#models)
- [Usage](#usage)
- [Saved Models & Outputs](#saved-models--outputs)
- [Dependencies](#dependencies)

## Overview

- This project explores various neural network architectures on the CIFAR-10 dataset, comparing performance from simple fully connected models to deep convolutional networks.

## Data

- Dataset: CIFAR-10: 60,000 32x32 color images in 10 classes (50,000 train / 10,000 test).
- A sample of the dataset can be visualized with inspect_data.py, which generates a grid of one example per class:
  ![CIFAR-10 Sample Grid](data/outputs/figures/cifar10_samples.png)

    
## Training

- Loss: CrossEntropyLoss
- Optimizer: Adam
- Scheduler: CosineAnnealingLR (optional)
- Batch size: 128 (default)
- Epoches: 30 (default)
- Device: CUDA if available

## Example Training Command:
```bash
# Train TinyCNN (default)
python train_model.py --model TinyCNN

# Train DeepCNN with scheduler
python train_model.py --model DeepCNN --use_scheduler

# Custom batch size and learning rate
python train_model.py --model WideCNN --batch_size 64 --lr 1e-4

```

## Results:
```
| Model       | Epochs | Data Aug | Scheduler | Best Validation Accuracy |
| ----------- | ------ | -------- | --------- | ------------------------ |
| FlatNet     | 3      | ❌        | ❌         | 0.377                 |
| FlatNet     | 10     | ❌        | ❌         | 0.392                 |
| Tiny CNN    | 3      | ❌        | ❌         | 0.581                 |
| Tiny CNN    | 10     | ❌        | ❌         | 0.647                 |
| Tiny CNN    | 10     | ✅        | ❌         | 0.641                 |
| Tiny CNN    | 30     | ✅        | ✅         | 0.688                 |
| Wide CNN    | 30     | ✅        | ✅         | 0.728                 |
| Deep CNN    | 30     | ✅        | ✅         | 0.784                 |
| 4-block CNN | 30     | ✅        | ✅         | 0.815                 |
| 5-block CNN | 30     | ✅        | ✅         | 0.829                 |
```
- Observation: CNN architectures outperform fully connected networks significantly. Increasing width and depth improves accuracy, but returns diminish after 4-5 blocks.
## Saved Models
- All trained models are saved in:
  ```bash
  data/outputs/models/
  ```
- Default naming: <ModelName>_cifar10.pth
- Best validation model: best_model_<ModelName>.pt

## Future Work

- Modularize training and evaluation for easier experimentation
- Add more data augmentations
- Explore more architectures (ResNet, VGG, etc.)
- Add inference scripts and visualizations

## Usage

- 1.Install requirements
```bash
pip install -r requirements.txt
```
- 2.Download CIFAR-10 and visualize samples
```bash
python inspect_data.py
```
- 3.Train a model
```bash
python train_model.py --model DeepCNN --use_scheduler
```
- 4.Load a trained model for inference
```bash
from models.deep_cnn import DeepCNN
import torch

model = DeepCNN()
model.load_state_dict(torch.load('data/outputs/models/DeepCNN_cifar10.pth'))
model.eval()
```
## Project Structure
```
CNN-project/
├── checkpoints/
│   └── log.txt/ 
├── data/                  # CIFAR-10 dataset and outputs
│   ├── outputs/           # Important results folder
│   └── models/
├── src/
│   ├── models/            # CNN and baseline models
│   │   ├── flat_model.py
│   │   ├── cnn_model.py
│   │   ├── wide_cnn.py
│   │   ├── deep_cnn.py
│   │   ├── cnn_4block.py
│   │   └── cnn_5block.py
│   ├── dataloader.py      # Data loading and augmentation
│   ├── train_utils.py     # Training and evaluation functions
│   ├── train_model.py
│   └── inspect_data.py    # Visualize sample CIFAR-10 images
├── .gitignore             # Ignored files/folders
├── .requirements.txt
└── README.md              # Project documentation
```

## Observations:

- CNN architectures outperform fully connected networks by a large margin.
- Increasing depth and width improves accuracy, but gains diminish beyond 4–5 convolutional blocks.
- Data augmentation and scheduler improve final validation accuracy.

  
