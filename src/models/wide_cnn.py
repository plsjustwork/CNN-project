import torch, torch.nn as nn

class WideCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),   # 16→32
            nn.ReLU(),
            nn.MaxPool2d(2),                  # 32×32 → 16×16
            nn.Conv2d(32, 64, 3, padding=1),  # 32→64
            nn.ReLU(),
            nn.MaxPool2d(2)                   # 16×16 → 8×8
        )
        self.classifier =nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*8*8, 10)   # 4096 → 10
        )
    def forward(self, x):
        x = self.features(x)      # [B, 64, 8, 8]
        x= self.classifier(x)
        return x