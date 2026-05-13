"""
2_preprocess.py — rewritten for reliability
Loads ERA5 files, normalises, creates (day T → day T+1) pairs,
saves train/val/test tensors.
"""

import numpy as np
import torch
import os

os.makedirs("data/processed", exist_ok=True)

def load_nc(path):
    """Load a NetCDF file trying multiple engines."""
    import xarray as xr
    for engine in ["netcdf4", "h5netcdf", "scipy"]:
        try:
            ds = xr.open_dataset(path, engine=engine)
            print(f"  Loaded {path} with engine={engine}")
            return ds
        except Exception:
            continue
    raise RuntimeError(f"Could not open {path} with any engine.")

print("Loading ERA5 data...")
t2m_ds  = load_nc("data/raw/t2m.nc")
tp_ds   = load_nc("data/raw/tp.nc")
z500_ds = load_nc("data/raw/z500.nc")
t850_ds = load_nc("data/raw/t850.nc")

# print available variable names to help debug
print("  t2m variables:",  list(t2m_ds.data_vars))
print("  tp variables:",   list(tp_ds.data_vars))
print("  z500 variables:", list(z500_ds.data_vars))
print("  t850 variables:", list(t850_ds.data_vars))

# extract the first data variable from each dataset automatically
def get_array(ds):
    skip = {"number", "expver", "time", "latitude", "longitude", "level"}
    varname = [v for v in ds.data_vars if v not in skip][0]
    arr = ds[varname].squeeze().values.astype(np.float32)
    if arr.ndim == 4:
        arr = arr[:, 0]
    return arr

t2m  = get_array(t2m_ds)
tp   = get_array(tp_ds)
z500 = get_array(z500_ds)
t850 = get_array(t850_ds)

# align lengths in case time axes differ slightly
n = min(len(t2m), len(tp), len(z500), len(t850))
t2m, tp, z500, t850 = t2m[:n], tp[:n], z500[:n], t850[:n]

print(f"  Timesteps: {n}, Grid: {t2m.shape[1]}x{t2m.shape[2]}")

# normalise
def normalise(arr):
    mu  = arr.mean()
    std = arr.std() + 1e-8
    return (arr - mu) / std, mu, std

t2m_n,  t2m_mu,  t2m_std  = normalise(t2m)
tp_n,   tp_mu,   tp_std   = normalise(tp)
z500_n, z500_mu, z500_std = normalise(z500)
t850_n, t850_mu, t850_std = normalise(t850)

np.savez("data/processed/stats.npz",
    t2m_mu=t2m_mu,   t2m_std=t2m_std,
    tp_mu=tp_mu,     tp_std=tp_std,
    z500_mu=z500_mu, z500_std=z500_std,
    t850_mu=t850_mu, t850_std=t850_std,
)

# build (X, y) pairs: X = 4 channels at day T, y = t2m at day T+1
X = np.stack([t2m_n[:-1], tp_n[:-1], z500_n[:-1], t850_n[:-1]], axis=1)
y = t2m_n[1:][:, np.newaxis]

print(f"  Input shape:  {X.shape}")
print(f"  Target shape: {y.shape}")

# train / val / test split (70/15/15)
n      = X.shape[0]
n_train = int(n * 0.70)
n_val   = int(n * 0.15)

splits = {
    "X_train": X[:n_train],
    "y_train": y[:n_train],
    "X_val":   X[n_train:n_train+n_val],
    "y_val":   y[n_train:n_train+n_val],
    "X_test":  X[n_train+n_val:],
    "y_test":  y[n_train+n_val:],
}

for name, arr in splits.items():
    torch.save(torch.tensor(arr), f"data/processed/{name}.pt")
    print(f"  Saved data/processed/{name}.pt  shape={arr.shape}")

print("\nPreprocessing complete.")
