import torch, torch.nn as nn
from dataloader import train_loader, test_loader

class FlatNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(32*32*3, 10)   # 3072 numbers → 10 scores

    def forward(self, x):        # x is [B, 3, 32, 32]
        x = x.view(x.size(0), -1)        # flatten to [B, 3072]
        return self.fc(x)                # [B, 10]
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FlatNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

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

for epoch in range(1, 4):               # 3 epochs only
    tr_loss, tr_acc = run_one_epoch(train_loader, training=True)
    val_loss, val_acc = run_one_epoch(test_loader,  training=False)
    print(f'Epoch {epoch}: train acc {tr_acc:.3f}  val acc {val_acc:.3f}')