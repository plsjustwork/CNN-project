import torch, torch.nn as nn

class FlatNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*32*3, 10)   # 3072 numbers → 10 scores
        )

    def forward(self, x):        # x is [B, 3, 32, 32]
        return self.fc(x)                # [B, 10]
    