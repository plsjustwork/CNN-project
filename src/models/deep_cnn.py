import torch, torch.nn as nn

class DeepCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),   # 32×32 → 32×32
            nn.ReLU(),
            nn.MaxPool2d(2),                  # 16×16

            nn.Conv2d(32, 64, 3, padding=1),  # 16×16 → 16×16
            nn.ReLU(),
            nn.MaxPool2d(2),                  # 8×8

            nn.Conv2d(64, 128, 3, padding=1), # 8×8 → 8×8
            nn.ReLU(),
            nn.MaxPool2d(2)                   # 8×8 → 4×4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 10)   # 2048 → 10
        )

    def forward(self, x):
        x = self.features(x)      # [B, 128, 4, 4]
        x= self.classifier(x)
        return x
    