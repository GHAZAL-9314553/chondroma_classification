# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ---------- Dataset ----------
class SlideBagDataset(Dataset):
    def __init__(self, csv_path, feature_dir, split):
        df = pd.read_csv(csv_path)
        df = df[df['dataset'] == split]
        self.feature_dir = feature_dir

        self.paths = []
        self.labels = []

        for slide, label in zip(df['slide'], df['category']):
            path = os.path.join(self.feature_dir, f"{slide}.pt")
            if os.path.exists(path):
                self.paths.append(path)
                self.labels.append(label)
            else:
                print(f"Missing: {path} — Skipping.")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        feats = torch.load(self.paths[idx])  # [N, D]
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return feats, label

# ---------- MIL Attention Model ----------
class MILAttention(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=128, num_classes=3):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x):  # x: [N, D]
        A = self.attention(x)  # [N, 1]
        A = torch.softmax(A, dim=0)  # attention weights
        M = torch.sum(A * x, dim=0)  # weighted sum -> [D]
        out = self.classifier(M)     # [num_classes]
        return out

# ---------- Training ----------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for feats, label in loader:
        feats = feats.squeeze(0).to(device)   # [N, D]
        label = torch.tensor([label.item()], dtype=torch.long).to(device)

        optimizer.zero_grad()
        output = model(feats)
        loss = criterion(output.unsqueeze(0), label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_preds.append(output.argmax().item())
        all_labels.append(label.item())

    return total_loss / len(loader), accuracy_score(all_labels, all_preds), f1_score(all_labels, all_preds, average='macro')

# ---------- Evaluation ----------
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for feats, label in loader:
            feats = feats.squeeze(0).to(device)
            label = label.to(device)
            output = model(feats)
            all_preds.append(output.argmax().item())
            all_labels.append(label.item())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    cm = confusion_matrix(all_labels, all_preds)
    return acc, f1, cm

# ---------- Main ----------
def main():
    csv_path = "path_bench_data.csv"
    feature_dir = "/gpfs/work1/0/prjs1420/features/all"
    input_dim = 384
    num_classes = 3
    batch_size = 1
    num_epochs = 30
    learning_rate = 1e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = SlideBagDataset(csv_path, feature_dir, split="training")
    val_dataset = SlideBagDataset(csv_path, feature_dir, split="testing")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    model = MILAttention(input_dim=input_dim, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_f1 = 0.0
    best_cm = None

    for epoch in range(1, num_epochs + 1):
        loss, acc, f1 = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_acc, val_f1, val_cm = evaluate(model, val_loader, device)

        print(f"Epoch {epoch}: Train Loss={loss:.4f} Acc={acc:.2f} F1={f1:.2f} | Val Acc={val_acc:.2f} F1={val_f1:.2f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_cm = val_cm
            torch.save(model.state_dict(), "best_mil_model.pt")
            print("Best model saved.")

    print("Confusion Matrix (Val):")
    print(best_cm)
    np.save("val_confusion_matrix.npy", best_cm)

if __name__ == "__main__":
    main()
