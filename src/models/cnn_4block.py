import torch, torch.nn as nn

class Deep4bCNN(nn.Module):
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
            nn.MaxPool2d(2),                   # 8×8 → 4×4

            nn.Conv2d(128, 256, 3, padding=1), # 4x4 -> 4x4
            nn.ReLU(),
            nn.MaxPool2d(2)                   # 4x4 → 2×2
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 2 * 2, 10)   # 1024 → 10
        )

    def forward(self, x):
        x = self.features(x)      # [B, 256, 2, 2]
        x=self.classifier(x)
        return x