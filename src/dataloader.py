import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
BATCH_SIZE = 128

train_tf = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),                      
])

test_tf = transforms.Compose([
    transforms.ToTensor(),
])

train_set = datasets.CIFAR10(root=DATA_DIR, train=True,
                             download=False,   
                             transform=train_tf)

test_set  = datasets.CIFAR10(root=DATA_DIR, train=False,
                             download=False,
                             transform=test_tf)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                          shuffle=True,  num_workers=0)

test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=0)

if __name__ == '__main__':
    for images, labels in train_loader:
        print('Batch shape:', images.shape)
        print('Labels shape:', labels.shape)
        break                                