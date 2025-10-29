import torch, torch.nn as nn
from dataloader import train_loader, test_loader

class Deep5bCNN(nn.Module):
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
            nn.MaxPool2d(2),                   # 4x4 → 2×2

            nn.Conv2d(256, 512, 3, padding=1), # 2x2 -> 2x2
            nn.ReLU(),
            nn.MaxPool2d(2)                   # 2x2 → 1x1
        )
        self.classifier = nn.Linear(512 * 1 * 1, 10)   # 512 → 10


    def forward(self, x):
        x = self.features(x)      # [B, 512, 1, 1]
        x = x.view(x.size(0), -1) # flatten
        return self.classifier(x)
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Deep5bCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

def run_one_epoch(loader, training):
    if training:
        model.train()
    else:
        model.eval()
    total, correct, loss_sum = 0, 0, 0.
    with torch.no_grad() if not training else torch.enable_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if training:
                optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            if training:
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * x.size(0)
            total   += y.size(0)
            correct += (out.argmax(1) == y).sum().item()
    return loss_sum/total, correct/total

for epoch in range(1, 31):              # 10 epochs
    tr_loss, tr_acc = run_one_epoch(train_loader, training=True)
    val_loss, val_acc = run_one_epoch(test_loader,  training=False)
    print(f'Epoch {epoch}: train acc {tr_acc:.3f}  val acc {val_acc:.3f}') 
    scheduler.step()