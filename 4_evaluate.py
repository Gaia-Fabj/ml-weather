"""
4_evaluate.py
Evaluates the trained U-Net against the persistence baseline.

Outputs:
  figures/training_curves.png
  figures/spatial_rmse.png
  figures/example_forecast.png
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

os.makedirs("figures", exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── U-Net (must match 3_train.py exactly) ─────────────────────────────────────

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
    def __init__(self, in_channels=4, out_channels=1, base_ch=32):
        super().__init__()
        self.enc1      = DoubleConv(in_channels, base_ch)
        self.enc2      = DoubleConv(base_ch, base_ch * 2)
        self.pool      = nn.MaxPool2d(2)
        self.bottleneck= DoubleConv(base_ch * 2, base_ch * 4)
        self.dec2      = DoubleConv(base_ch * 4 + base_ch * 2, base_ch * 2)
        self.dec1      = DoubleConv(base_ch * 2 + base_ch, base_ch)
        self.out       = nn.Conv2d(base_ch, out_channels, kernel_size=1)

    def forward(self, x):
        e1  = self.enc1(x)
        e2  = self.enc2(self.pool(e1))
        b   = self.bottleneck(self.pool(e2))
        up2 = F.interpolate(b,  size=e2.shape[2:], mode="bilinear", align_corners=False)
        d2  = self.dec2(torch.cat([up2, e2], dim=1))
        up1 = F.interpolate(d2, size=e1.shape[2:], mode="bilinear", align_corners=False)
        d1  = self.dec1(torch.cat([up1, e1], dim=1))
        return self.out(d1)


# ── load data and model ───────────────────────────────────────────────────────

print("Loading test data...")
X_test = torch.load("data/processed/X_test.pt")
y_test = torch.load("data/processed/y_test.pt")
stats  = np.load("data/processed/stats.npz")
t2m_mu  = float(stats["t2m_mu"])
t2m_std = float(stats["t2m_std"])

model = UNet(in_channels=4, out_channels=1).to(DEVICE)
model.load_state_dict(torch.load("models/unet_best.pt", map_location=DEVICE))
model.eval()
print(f"  Test samples: {X_test.shape[0]}")

# ── run inference ─────────────────────────────────────────────────────────────

with torch.no_grad():
    pred_norm = model(X_test.to(DEVICE)).cpu().numpy()[:, 0]

truth_norm   = y_test.numpy()[:, 0]
persist_norm = X_test.numpy()[:, 0]

# denormalise to Celsius
pred_C    = pred_norm    * t2m_std + t2m_mu - 273.15
truth_C   = truth_norm   * t2m_std + t2m_mu - 273.15
persist_C = persist_norm * t2m_std + t2m_mu - 273.15

# ── skill metrics ─────────────────────────────────────────────────────────────

rmse_model   = np.sqrt(((pred_C    - truth_C) ** 2).mean())
rmse_persist = np.sqrt(((persist_C - truth_C) ** 2).mean())
mae_model    = np.abs(pred_C - truth_C).mean()
skill_score  = 1 - rmse_model / rmse_persist

print(f"\n{'='*45}")
print(f"  RMSE (model):       {rmse_model:.2f} °C")
print(f"  RMSE (persistence): {rmse_persist:.2f} °C")
print(f"  MAE  (model):       {mae_model:.2f} °C")
print(f"  Skill score:        {skill_score:.3f}  (>0 beats persistence)")
print(f"{'='*45}\n")

# ── spatial RMSE map ──────────────────────────────────────────────────────────

spatial_rmse = np.sqrt(((pred_C - truth_C) ** 2).mean(axis=0))

fig, ax = plt.subplots(figsize=(10, 5))
im = ax.imshow(spatial_rmse, cmap="hot_r", vmin=0, vmax=4, origin="upper")
plt.colorbar(im, ax=ax, label="RMSE (°C)")
ax.set_title("Spatial RMSE — U-Net 24h Temperature Forecast over Europe")
ax.set_xlabel("Longitude index")
ax.set_ylabel("Latitude index")
plt.tight_layout()
plt.savefig("figures/spatial_rmse.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: figures/spatial_rmse.png")

# ── example forecast ──────────────────────────────────────────────────────────

errors = np.abs(pred_C - truth_C).mean(axis=(1, 2))
idx    = np.argsort(errors)[len(errors) // 2]

vmin = min(truth_C[idx].min(), pred_C[idx].min())
vmax = max(truth_C[idx].max(), pred_C[idx].max())

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
data = [
    (truth_C[idx],              "Ground truth (T+1)",    "RdBu_r", vmin,  vmax),
    (pred_C[idx],               "U-Net prediction",      "RdBu_r", vmin,  vmax),
    (pred_C[idx]-truth_C[idx],  "Error (pred − truth)",  "bwr",    -3,    3),
]
for ax, (arr, title, cmap, vn, vx) in zip(axes, data):
    im = ax.imshow(arr, cmap=cmap, vmin=vn, vmax=vx, origin="upper")
    ax.set_title(title)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="°C")

plt.suptitle(f"Example forecast (test sample #{idx})", fontsize=13)
plt.tight_layout()
plt.savefig("figures/example_forecast.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: figures/example_forecast.png")

# ── training curves ───────────────────────────────────────────────────────────

history = np.load("models/training_history.npy", allow_pickle=True).item()
epochs  = range(1, len(history["train_loss"]) + 1)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(epochs, history["train_loss"], label="Train MSE", color="#2563eb")
ax.plot(epochs, history["val_loss"],   label="Val MSE",   color="#dc2626")
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE (normalised)")
ax.set_title("Training curves — U-Net temperature forecast")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures/training_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: figures/training_curves.png")

print("\nEvaluation complete. All figures saved to figures/")
