"""Training script for the model."""

import yaml
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms 

from ..data import HairSwatchDataset
from ..models import CNNRNNModel
from .losses import compute_loss
from .metrics import tolerance_accuracy


def seed_all(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pretty_print_params(params: dict):
    print("Training parameters:")
    print("  " + "-" * 31)
    for k, v in params.items():
        print(f"  {k:<15}|{v:>15}")
    print("  " + "-" * 31)


def train_one_epoch(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(vocab=torch.tensor(range(10), dtype=torch.long).to(device), input_img=images)
        loss = compute_loss(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    avg_loss = total_loss / len(train_loader.dataset)
    return avg_loss


def evaluate(model, val_loader, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(vocab=torch.tensor(range(10), dtype=torch.long).to(device), input_img=images)
            loss = compute_loss(logits, labels)
            total_loss += loss.item() * images.size(0)
            preds = torch.argmax(logits, dim=-1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    avg_loss = total_loss / len(val_loader.dataset)
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    accuracy = tolerance_accuracy(all_preds, all_labels)
    return avg_loss, accuracy


def main():
    seed_all(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    tsfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Load dataset
    dataset = HairSwatchDataset(transform=tsfm)
    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    with open("config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    params = config['parameters']
    pretty_print_params(params)

    # Model, optimizer
    model = CNNRNNModel(
        vocab_size=params['vocab_size'],
        embed_dim=params['embed_dim'],
        hidden_dim=params['hidden_dim'],
        output_dim=params['output_dim'],
        feat_dim=params['feat_dim']
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])

    # Training loop
    for epoch in range(1, params['num_epochs'] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_accuracy = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:02d}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")


if __name__ == "__main__":
    main()
