"""
3_train.py
Trains a U-Net convolutional neural network to predict
24-hour ahead 2m temperature fields from 4 atmospheric input channels.

Architecture:
  Input:  (batch, 4, lat, lon)   — t2m, tp, z500, t850 at day T
  Output: (batch, 1, lat, lon)   — predicted t2m at day T+1

Handles arbitrary grid sizes by using bilinear interpolation in the decoder
instead of transposed convolutions, avoiding size mismatch issues.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os

os.makedirs("models", exist_ok=True)

EPOCHS     = 50
BATCH_SIZE = 8
LR         = 1e-3
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {DEVICE}")


# ── U-Net building blocks ─────────────────────────────────────────────────────

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """
    U-Net using bilinear upsampling instead of transposed convolutions.
    This avoids spatial size mismatch errors on arbitrary grid sizes.
    """
    def __init__(self, in_channels=4, out_channels=1, base_ch=32):
        super().__init__()
        # Encoder
        self.enc1 = DoubleConv(in_channels, base_ch)
        self.enc2 = DoubleConv(base_ch, base_ch * 2)
        self.pool = nn.MaxPool2d(2)
        # Bottleneck
        self.bottleneck = DoubleConv(base_ch * 2, base_ch * 4)
        # Decoder
        self.dec2 = DoubleConv(base_ch * 4 + base_ch * 2, base_ch * 2)
        self.dec1 = DoubleConv(base_ch * 2 + base_ch, base_ch)
        # Output
        self.out = nn.Conv2d(base_ch, out_channels, kernel_size=1)

    def forward(self, x):
        # Encode
        e1 = self.enc1(x)                    # (B, 32, H, W)
        e2 = self.enc2(self.pool(e1))         # (B, 64, H/2, W/2)
        b  = self.bottleneck(self.pool(e2))   # (B, 128, H/4, W/4)

        # Decode — upsample to match skip connection size exactly
        up2 = F.interpolate(b,  size=e2.shape[2:], mode="bilinear", align_corners=False)
        d2  = self.dec2(torch.cat([up2, e2], dim=1))

        up1 = F.interpolate(d2, size=e1.shape[2:], mode="bilinear", align_corners=False)
        d1  = self.dec1(torch.cat([up1, e1], dim=1))

        return self.out(d1)   # same spatial size as input


# ── load data ─────────────────────────────────────────────────────────────────

print("Loading data...")
X_train = torch.load("data/processed/X_train.pt")
y_train = torch.load("data/processed/y_train.pt")
X_val   = torch.load("data/processed/X_val.pt")
y_val   = torch.load("data/processed/y_val.pt")

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val,   y_val),   batch_size=BATCH_SIZE)

print(f"  Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

# ── model, loss, optimiser ────────────────────────────────────────────────────

model     = UNet(in_channels=4, out_channels=1).to(DEVICE)
criterion = nn.MSELoss()
optimiser = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, patience=5, factor=0.5)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Model parameters: {n_params:,}")

# ── training loop ─────────────────────────────────────────────────────────────

best_val_loss = float("inf")
history = {"train_loss": [], "val_loss": []}

for epoch in range(1, EPOCHS + 1):
    # train
    model.train()
    train_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimiser.zero_grad()
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        optimiser.step()
        train_loss += loss.item() * X_batch.size(0)
    train_loss /= len(train_loader.dataset)

    # validate
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            pred = model(X_batch)
            val_loss += criterion(pred, y_batch).item() * X_batch.size(0)
    val_loss /= len(val_loader.dataset)

    scheduler.step(val_loss)
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)

    print(f"  Epoch {epoch:3d}/{EPOCHS} | Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "models/unet_best.pt")

np.save("models/training_history.npy", history)
print(f"\nTraining complete. Best val MSE: {best_val_loss:.4f}")
print("Model saved to models/unet_best.pt")
