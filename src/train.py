"""
Prototype: CNN backbone with either (A) RNN decoder (auto-regressive over tone digits)
or (B) parallel branch heads with optional conditioning — for hierarchical tone output.

Context from "!!! - Project Scoping":
- Predict Base first; condition subsequent digits on earlier ones.
- Try RNN only at the output layer to reuse information of earlier digits.
- Start simple (small CNN), avoid heavy backbones initially.

Label conventions (updated):
- CSV columns: filename, Base, Primary, Secondary, Tertiary
- Base ∈ {1..10}; others ∈ {0..9}.
- **Missing digits are encoded as -1 in the CSV.**
- For PyTorch CE: map Base {1..10} → {0..9}; others keep {0..9}. Missing stay as -1 and are masked from loss/metrics.

Run:
- Fill DATA_CSV and IMG_DIR.
- Choose ARCH in {"rnn", "branch"} and (optionally) enable conditioning for branch.
- python cnn_rnn_hierarchical_tone_prototype.py

This is a self-contained prototype with: dataset, models, training loop, masking, and metrics.
"""
from __future__ import annotations
import os
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from pathlib import Path

import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T

# -----------------------
# Config
# -----------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_CSV = ROOT_DIR / "data/ColourCorrectedImages/hair_swatch_labels.csv"   # <-- set me
IMG_DIR  = ROOT_DIR / "data/ColourCorrectedImages"                    # <-- set me
SAVE_DIR = ROOT_DIR / "runs"
SEED = 42
BATCH_SIZE = 32
LR = 3e-4
EPOCHS = 50
IMG_SIZE = 224
ARCH = "rnn"          # "rnn" or "branch"
BRANCH_CONDITION_ON_BASE = True  # used when ARCH == "branch"
TEACHER_FORCING = 0.6            # used when ARCH == "rnn" (probability during training)
VAL_SPLIT = 0.15

# Class counts per time step (digits). We standardize to 10 here for simplicity.
NUM_CLASSES_PER_STEP = [10, 10, 10, 10]  # Base mapped to 0..9; others 0..9

# -----------------------
# Utils
# -----------------------

def seed_all(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    # Prefer Apple Silicon MPS if present
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# -----------------------
# Dataset
# -----------------------
class HairSwatchCsvDataset(Dataset):
    def __init__(self, csv_path: str | Path, img_dir: str | Path, transform: Optional[nn.Module] = None):
        self.df = pd.read_csv(csv_path)
        # Normalize column names
        self.df.columns = [c.strip().capitalize() for c in self.df.columns]
        required = ["Filename", "Base", "Primary", "Secondary", "Tertiary"]
        for c in required:
            if c not in self.df.columns:
                raise ValueError(f"CSV must contain column '{c}'")
        self.img_dir = img_dir
        self.transform = transform if transform is not None else T.ToTensor()
        # Prepare labels and mask (-1 for missing)
        labels = []
        mask = []
        for _, row in self.df.iterrows():
            # Ensure ints (handles strings like "-1")
            def as_int(x):
                try:
                    return int(x)
                except Exception:
                    return -1

            base_raw = as_int(row["Base"])  # 1..10 or -1
            if base_raw == -1:
                base = -1
                m0 = 0
            else:
                if not (1 <= base_raw <= 10):
                    raise ValueError(f"Base out of range [1..10]: found {base_raw}")
                base = base_raw - 1  # map to 0..9
                m0 = 1

            steps = []
            msk = [m0]
            for col in ["Primary", "Secondary", "Tertiary"]:
                v = as_int(row[col])
                if v == -1:
                    steps.append(-1)
                    msk.append(0)
                else:
                    if not (0 <= v <= 9):
                        raise ValueError(f"{col} out of range [0..9]: found {v}")
                    steps.append(v)
                    msk.append(1)
            labels.append([base] + steps)
            mask.append(msk)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.mask = torch.tensor(mask, dtype=torch.bool)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, str(row["Filename"]))
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        targets = self.labels[idx]          # shape (4,)
        mask = self.mask[idx]               # shape (4,)
        return img, targets, mask


def make_transforms(img_size: int = 224):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        # Add Normalize(mean,std) once you compute dataset stats
    ])


# -----------------------
# Models
# -----------------------
class SmallCNN(nn.Module):
    """A compact CNN backbone with global average pooling → feature vector."""
    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.features = nn.Sequential(
            # in: 3 x 224 x 224
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),  # 32 x 112 x 112
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),  # 64 x 56 x 56
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),  # 128 x 28 x 28
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.AdaptiveAvgPool2d((1, 1))     # 256 x 1 x 1
        )
        self.proj = nn.Linear(256, out_dim)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        x = self.proj(x)
        return x  # (B, out_dim)


class RNNDecoder(nn.Module):
    """Auto-regressive decoder over 4 digits using LSTMCell.

    - At t=0, input is [feat, BOS_emb].
    - At t>0, input is [feat, emb(y_{t-1})].
    - Hidden state is initialized from features.
    - Produces logits_t for each step t.
    - All steps emit 10-way logits by default.
    """
    def __init__(self, feat_dim: int, emb_dim: int = 32, hidden_dim: int = 256, num_steps: int = 4,
                 num_classes_per_step: List[int] = NUM_CLASSES_PER_STEP):
        super().__init__()
        self.num_steps = num_steps
        self.emb_dim = emb_dim
        self.num_classes_per_step = num_classes_per_step
        self.total_vocab = max(num_classes_per_step)  # 10 here

        self.bos = nn.Parameter(torch.zeros(emb_dim))
        self.emb = nn.Embedding(self.total_vocab, emb_dim)

        self.h0_proj = nn.Linear(feat_dim, hidden_dim)
        self.c0_proj = nn.Linear(feat_dim, hidden_dim)
        self.lstm = nn.LSTMCell(feat_dim + emb_dim, hidden_dim)
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, c) for c in num_classes_per_step])

    def forward(self, feat, targets: Optional[torch.Tensor] = None, teacher_forcing: float = 0.0,
                mask: Optional[torch.Tensor] = None):
        """Forward with optional teacher forcing.
        Args:
            feat: (B, F)
            targets: (B, T) long or None (may contain -1 for missing)
            teacher_forcing: prob of feeding ground-truth at t>0
            mask: (B, T) bool or None — not used inside, but kept for API symmetry
        Returns:
            logits: List[T] of (B, C_t)
            preds: (B, T) greedy predictions
        """
        B = feat.size(0)
        h = torch.tanh(self.h0_proj(feat))
        c = torch.tanh(self.c0_proj(feat))
        inp = torch.cat([feat, self.bos.expand(B, -1)], dim=1)

        logits = []
        preds = []
        for t in range(self.num_steps):
            h, c = self.lstm(inp, (h, c))
            logit_t = self.heads[t](h)
            logits.append(logit_t)
            y_t = logit_t.argmax(dim=1)
            preds.append(y_t)
            # Next input token embedding (except after last step)
            if t < self.num_steps - 1:
                if (targets is not None) and (random.random() < teacher_forcing):
                    token = targets[:, t]
                    token = torch.where(token < 0, torch.zeros_like(token), token)  # replace -1 with 0 for embedding
                else:
                    token = y_t
                emb_t = self.emb(token)
                inp = torch.cat([feat, emb_t], dim=1)
        preds = torch.stack(preds, dim=1)  # (B, T)
        return logits, preds


class BranchHeads(nn.Module):
    """Parallel heads with optional conditioning on predicted/true base.

    If condition_on_base=True, we append an embedding of the base digit to
    the feature vector for Primary/Secondary/Tertiary heads.
    """
    def __init__(self, feat_dim: int, hidden: int = 256, num_classes_per_step: List[int] = NUM_CLASSES_PER_STEP,
                 condition_on_base: bool = True, cond_emb_dim: int = 16):
        super().__init__()
        self.condition_on_base = condition_on_base
        self.cond_emb = nn.Embedding(num_classes_per_step[0], cond_emb_dim) if condition_on_base else None
        # shared MLP head builder
        def mlp(in_dim, out_dim):
            return nn.Sequential(
                nn.Linear(in_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, out_dim)
            )
        self.base_head = mlp(feat_dim, num_classes_per_step[0])
        in_dim = feat_dim + (cond_emb_dim if condition_on_base else 0)
        self.prim_head = mlp(in_dim, num_classes_per_step[1])
        self.sec_head  = mlp(in_dim, num_classes_per_step[2])
        self.ter_head  = mlp(in_dim, num_classes_per_step[3])

    def forward(self, feat, targets: Optional[torch.Tensor] = None, use_teacher_forcing: bool = True):
        logits_base = self.base_head(feat)
        base_pred = logits_base.argmax(dim=1)
        if self.condition_on_base:
            if (targets is not None) and use_teacher_forcing:
                base_token = torch.where(targets[:, 0] < 0, torch.zeros_like(targets[:, 0]), targets[:, 0])
            else:
                base_token = base_pred
            base_emb = self.cond_emb(base_token)
            z = torch.cat([feat, base_emb], dim=1)
        else:
            z = feat
        logits_prim = self.prim_head(z)
        logits_sec  = self.sec_head(z)
        logits_ter  = self.ter_head(z)
        logits = [logits_base, logits_prim, logits_sec, logits_ter]
        preds = torch.stack([l.argmax(dim=1) for l in logits], dim=1)
        return logits, preds


class ToneNet(nn.Module):
    """Wrapper: CNN backbone + (A) RNN decoder OR (B) Branch heads."""
    def __init__(self, arch: str = "rnn", feat_dim: int = 256, **kwargs):
        super().__init__()
        self.backbone = SmallCNN(out_dim=feat_dim)
        self.arch = arch
        if arch == "rnn":
            self.decoder = RNNDecoder(feat_dim=feat_dim, **kwargs)
        elif arch == "branch":
            self.decoder = BranchHeads(feat_dim=feat_dim, **kwargs)
        else:
            raise ValueError("arch must be 'rnn' or 'branch'")

    def forward(self, x, targets: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None, **kwargs):
        feat = self.backbone(x)
        if self.arch == "rnn":
            logits, preds = self.decoder(feat, targets=targets, mask=mask, **kwargs)
        else:
            logits, preds = self.decoder(feat, targets=targets, **kwargs)
        return logits, preds


# -----------------------
# Loss & Metrics
# -----------------------

def masked_ce_loss(logits_list: List[torch.Tensor], targets: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, List[float]]:
    """Sum CE over steps using mask (-1 labels are ignored by mask).
    Returns total loss and per-step accuracies.
    """
    total_loss = torch.tensor(0.0, device=targets.device)
    per_step_acc = []
    for t, logits_t in enumerate(logits_list):
        m = mask[:, t]
        if m.any():
            tgt = targets[m, t]
            loss_t = F.cross_entropy(logits_t[m], tgt)
            total_loss = total_loss + loss_t
            preds_t = logits_t[m].argmax(dim=1)
            acc_t = (preds_t == tgt).float().mean().item()
        else:
            acc_t = float("nan")
        per_step_acc.append(acc_t)
    return total_loss, per_step_acc


def hierarchical_exact_match(logits_list: List[torch.Tensor], targets: torch.Tensor, mask: torch.Tensor) -> float:
    """Percentage of samples for which all available digits are correct simultaneously.
    
    Modified to allow ±1 tolerance for base values 1-4 (mapped to 0-3 in zero-indexed),
    and exact match for base values 5-10 (mapped to 4-9 in zero-indexed).
    """
    # Original implementation (commented out):
    # with torch.no_grad():
    #     preds = torch.stack([l.argmax(dim=1) for l in logits_list], dim=1)  # (B, T)
    #     ok = (preds == targets)
    #     ok = torch.where(mask, ok, torch.ones_like(ok, dtype=torch.bool))  # ignore missing
    #     per_sample = ok.all(dim=1).float()
    #     return per_sample.mean().item()
    
    with torch.no_grad():
        preds = torch.stack([l.argmax(dim=1) for l in logits_list], dim=1)  # (B, T)
        
        # Start with exact match for all positions
        ok = (preds == targets)
        
        # For base digit (position 0), apply ±1 tolerance for values 0-3 (original 1-4)
        base_preds = preds[:, 0]  # (B,)
        base_targets = targets[:, 0]  # (B,)
        base_mask = mask[:, 0]  # (B,)
        
        # Check if target base is in range 0-3 (original 1-4) and mask is valid
        tolerance_range = (base_targets >= 0) & (base_targets <= 3) & base_mask
        
        # For positions in tolerance range, check if prediction is within ±1
        base_tolerance_match = torch.abs(base_preds - base_targets) <= 1
        
        # Combine: use tolerance match for range 0-3, exact match for range 4-9
        base_ok = torch.where(tolerance_range, base_tolerance_match, ok[:, 0])
        
        # Replace the base column with the new tolerance-aware matching
        ok[:, 0] = base_ok
        
        # Apply mask (ignore missing values by setting them to True)
        ok = torch.where(mask, ok, torch.ones_like(ok, dtype=torch.bool))
        
        # Check if all digits in each sample are correct
        per_sample = ok.all(dim=1).float()
        return per_sample.mean().item()


# -----------------------
# Training Loop
# -----------------------
@dataclass
class Batch:
    x: torch.Tensor
    y: torch.Tensor
    m: torch.Tensor


def collate(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Batch:
    imgs, ys, ms = zip(*batch)
    x = torch.stack(imgs)
    y = torch.stack(ys)
    m = torch.stack(ms)
    return Batch(x, y, m)


def split_datasets(ds: Dataset, val_split: float, seed: int) -> Tuple[Dataset, Dataset]:
    n = len(ds)
    n_val = max(1, int(n * val_split))
    n_train = n - n_val
    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=g)
    return train_ds, val_ds


def train_one_epoch(model: ToneNet, loader: DataLoader, opt: torch.optim.Optimizer, device, arch: str,
                    teacher_forcing: float) -> Dict[str, float]:
    model.train()
    total, n_batches = 0.0, 0
    acc_accum = [0.0, 0.0, 0.0, 0.0]
    em_accum = 0.0
    for batch in loader:
        x, y, m = batch.x.to(device), batch.y.to(device), batch.m.to(device)
        opt.zero_grad()
        if arch == "rnn":
            logits, _ = model(x, targets=y, mask=m, teacher_forcing=teacher_forcing)
        else:
            logits, _ = model(x, targets=y, mask=m, use_teacher_forcing=True)
        loss, per_step_acc = masked_ce_loss(logits, y, m)
        em = hierarchical_exact_match(logits, y, m)
        loss.backward()
        opt.step()
        total += loss.item()
        n_batches += 1
        for i, a in enumerate(per_step_acc):
            if not math.isnan(a):
                acc_accum[i] += a
        em_accum += em
    stats = {
        "loss": total / max(1, n_batches),
        "acc_base": acc_accum[0] / max(1, n_batches),
        "acc_prim": acc_accum[1] / max(1, n_batches),
        "acc_sec":  acc_accum[2] / max(1, n_batches),
        "acc_ter":  acc_accum[3] / max(1, n_batches),
        "acc_exact": em_accum / max(1, n_batches),
    }
    return stats


def evaluate(model: ToneNet, loader: DataLoader, device, arch: str) -> Dict[str, float]:
    model.eval()
    total, n_batches = 0.0, 0
    acc_accum = [0.0, 0.0, 0.0, 0.0]
    em_accum = 0.0
    with torch.no_grad():
        for batch in loader:
            x, y, m = batch.x.to(device), batch.y.to(device), batch.m.to(device)
            if arch == "rnn":
                logits, _ = model(x, targets=None, mask=m, teacher_forcing=0.0)
            else:
                logits, _ = model(x, targets=None, mask=m, use_teacher_forcing=False)
            loss, per_step_acc = masked_ce_loss(logits, y, m)
            em = hierarchical_exact_match(logits, y, m)
            total += loss.item()
            n_batches += 1
            for i, a in enumerate(per_step_acc):
                if not math.isnan(a):
                    acc_accum[i] += a
            em_accum += em
    stats = {
        "val_loss": total / max(1, n_batches),
        "val_acc_base": acc_accum[0] / max(1, n_batches),
        "val_acc_prim": acc_accum[1] / max(1, n_batches),
        "val_acc_sec":  acc_accum[2] / max(1, n_batches),
        "val_acc_ter":  acc_accum[3] / max(1, n_batches),
        "val_acc_exact": em_accum / max(1, n_batches),
    }
    return stats


# -----------------------
# Main
# -----------------------

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    seed_all(SEED)
    device = get_device()
    print(f"Using device: {device}")

    transform = make_transforms(IMG_SIZE)
    ds = HairSwatchCsvDataset(DATA_CSV, IMG_DIR, transform)
    train_ds, val_ds = split_datasets(ds, VAL_SPLIT, SEED)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=collate)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate)

    if ARCH == "rnn":
        model = ToneNet(arch="rnn", feat_dim=256, emb_dim=32, hidden_dim=256,
                        num_steps=4, num_classes_per_step=NUM_CLASSES_PER_STEP)
    else:
        model = ToneNet(arch="branch", feat_dim=256, hidden=256,
                        num_classes_per_step=NUM_CLASSES_PER_STEP,
                        condition_on_base=BRANCH_CONDITION_ON_BASE, cond_emb_dim=16)

    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    best_val = float("inf")
    for epoch in range(1, EPOCHS + 1):
        tr = train_one_epoch(model, train_loader, opt, device, ARCH, TEACHER_FORCING)
        va = evaluate(model, val_loader, device, ARCH)
        log = {"epoch": epoch, **tr, **va}
        print(" | ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in log.items()]))
        # basic checkpointing
        if va["val_loss"] < best_val:
            best_val = va["val_loss"]
            ckpt = {
                "model": model.state_dict(),
                "config": {
                    "arch": ARCH,
                    "img_size": IMG_SIZE,
                    "num_classes": NUM_CLASSES_PER_STEP,
                }
            }
            torch.save(ckpt, os.path.join(SAVE_DIR, f"best_{ARCH}.pt"))


if __name__ == "__main__":
    main()
