# CNN-Project: CIFAR-10 Classification

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
```
# Train DeepCNN with scheduler
python train_model.py --model DeepCNN --use_scheduler
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
- All trained models are saved in: data/outputs/models/
- Default naming: <ModelName>_cifar10.pth
- Best validation model: best_model_<ModelName>.pt

## Future Work

- Modularize training and evaluation for easier experimentation
- Add more data augmentations
- Explore more architectures (ResNet, VGG, etc.)
- Add inference scripts and visualizations

## Project Structure
```
CNN-project/
├── data/                  # CIFAR-10 dataset and outputs
│   └── outputs/           # Important results folder
├── src/
│   ├── models/            # CNN and baseline models
│   │   ├── flat.py
│   │   ├── tiny_cnn.py
│   │   ├── wide_cnn.py
│   │   ├── deep_cnn.py
│   │   └── ...
│   ├── dataloader.py      # Data loading and augmentation
│   ├── train_utils.py     # Training and evaluation functions
│   └── inspect_data.py    # Visualize sample CIFAR-10 images
├── logs/
│   └── log.txt            # Training results
├── .gitignore             # Ignored files/folders
└── README.md              # Project documentation
```
## Training Function:
- All models use the same run_one_epoch() function for training and evaluation, with proper handling for GPU and gradient updates.


## Observations:

- CNN architectures outperform fully connected networks by a large margin.
- Increasing depth and width improves accuracy, but gains diminish beyond 4–5 convolutional blocks.
- Data augmentation and scheduler improve final validation accuracy.

  
