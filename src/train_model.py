import torch
from dataloader import get_cifar10_loaders
from train_utils import train_model
import argparse
import os

from models.flat_model import FlatNet
from models.cnn_model import TinyCNN
from models.wide_cnn import WideCNN
from models.deep_cnn import DeepCNN
from models.cnn_4block import Deep4bCNN
from models.cnn_5block import Deep5bCNN

model_dict = {
    'FlatNet': FlatNet,
    'TinyCNN': TinyCNN,
    'WideCNN': WideCNN,
    'DeepCNN': DeepCNN,
    'Deep4bCNN': Deep4bCNN,
    'Deep5bCNN': Deep5bCNN
}

def main():
    parser = argparse.ArgumentParser(description="Train a CNN on CIFAR-10")
    parser.add_argument('--model', type=str, default='TinyCNN', choices=model_dict.keys(), help='Model to train')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=0, help='DataLoader workers')
    parser.add_argument('--use_scheduler', action='store_true', help='Use learning rate scheduler')
    parser.add_argument('--save_path', type=str, default=None, help='Path to save trained model (default: model name + _cifar10.pth)')
    args = parser.parse_args()

    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = get_cifar10_loaders(batch_size=args.batch_size, num_workers=args.num_workers)
    model = model_dict[args.model]().to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs) if args.use_scheduler else None
    print("=" * 60)
    print(f"Training Model: {args.model}")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs} | Batch Size: {args.batch_size} | LR: {args.lr}")
    print(f"Scheduler: {'Enabled' if args.use_scheduler else 'Disabled'}")
    print("=" * 60)

    train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, epochs=args.epochs, device=device)

    save_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'outputs', 'models')
    os.makedirs(save_dir, exist_ok=True)
    save_path = args.save_path or os.path.join(save_dir, f"{args.model}_cifar10.pth")
    torch.save(model.state_dict(), save_path)
    print(f"\n✅ Model saved successfully at: {save_path}")

if __name__ == '__main__':
    main()