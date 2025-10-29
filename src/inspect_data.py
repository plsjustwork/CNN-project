from torchvision import datasets
import os
import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')   
dataset = datasets.CIFAR10(root=DATA_DIR, train=True, download=True)
print('Download complete →', len(dataset), 'images')

print(dataset.classes)  
print(len(dataset.classes))


fig, axes = plt.subplots(2, 5, figsize=(10, 4)) 
axes = axes.ravel()                             

got = set()                                       
for image, label_idx in dataset:                  
    word = dataset.classes[label_idx]
    if word not in got:                           
        got.add(word)
        pos = dataset.classes.index(word)
        axes[pos].imshow(image)
        axes[pos].set_title(word, size=9)
        axes[pos].axis('off')
    if len(got) == 10:
        break

plt.tight_layout()
plt.savefig('data/outputs/figures/cifar10_samples.png', dpi=120)
print('Saved picture grid → data/outputs/figures/cifar10_samples.png')