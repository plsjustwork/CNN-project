import torch
import os

save_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'outputs', 'figures')
os.makedirs(save_dir, exist_ok=True)

def run_one_epoch(loader, model, criterion, optimizer=None, device='cpu', training=True):
    if training:
        model.train()
    else:
        model.eval()

    total, correct, loss_sum = 0, 0, 0.0
    with torch.set_grad_enabled(training):
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
    return loss_sum / total, correct / total

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler=None, epochs=30, device='cpu'):
    best_val_acc = 0.0
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = run_one_epoch(train_loader, model, criterion, optimizer, device, training=True)
        val_loss, val_acc = run_one_epoch(val_loader, model, criterion, device=device, training=False)

        if scheduler:
            scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(save_dir, f'best_model_{model.__class__.__name__}.pt')
            torch.save(model.state_dict(), save_path)

        print(f"Epoch {epoch}: Train Acc={tr_acc:.3f}, Val Acc={val_acc:.3f}, Train Loss={tr_loss:.3f}, Val Loss={val_loss:.3f}")

    print(f"Best validation accuracy: {best_val_acc:.3f}")