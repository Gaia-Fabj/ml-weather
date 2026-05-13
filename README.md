# ML Weather Forecasting — U-Net Temperature Prediction

A machine learning project for short-range weather forecasting using ERA5 reanalysis data.
A U-Net convolutional neural network is trained to predict 24-hour ahead 2m temperature
fields over Europe, combining data-driven methods with physically meaningful atmospheric variables.

Built as a hands-on introduction to ML applied to numerical weather prediction,
motivated by work at institutes such as DMI and ECMWF on next-generation forecasting systems.

---

## What this project does

Given today's atmospheric state over Europe, the model predicts tomorrow's surface temperature field.

**Input (4 channels at day T):**
- 2m temperature — current surface temperature
- Total precipitation — surface moisture forcing
- 500 hPa geopotential — large-scale atmospheric dynamics (captures weather systems)
- 850 hPa temperature — lower troposphere thermal state

**Output:**
- Predicted 2m temperature field at day T+1

**Baseline comparison:**
The model is evaluated against *persistence* — the naive forecast that tomorrow's temperature
equals today's. Beating persistence is the standard first benchmark in meteorology.

---

## Data

**Source:** ERA5 reanalysis, produced by ECMWF and distributed freely via the
[Copernicus Climate Data Store](https://cds.climate.copernicus.eu).

**Domain:** Europe (30N-75N, 25W-45E) at 1 degree horizontal resolution (~100 km)

**Period:** 2023-2024 (731 daily snapshots at 12:00 UTC)

**Split:** 70% train / 15% validation / 15% test (chronological, no shuffling)

ERA5 is the gold standard reanalysis dataset used by meteorological institutes,
universities, and operational forecasting centres worldwide.

---

## Model architecture: U-Net

The U-Net was originally developed for biomedical image segmentation but has become
widely used in weather and climate ML due to its ability to capture spatial patterns
at multiple scales simultaneously.

```
Input (4, 46, 71)
    |
    |-- Encoder block 1 --> (32, 46, 71)  --------------------> skip connection
    |        | MaxPool                                                |
    |-- Encoder block 2 --> (64, 23, 35)  -----------> skip          |
    |        | MaxPool                                    |           |
    |-- Bottleneck       --> (128, 11, 17)               |           |
    |        | Upsample                                   |           |
    |-- Decoder block 2 --> (64, 23, 35)  <-- concat ----           |
    |        | Upsample                                               |
    |-- Decoder block 1 --> (32, 46, 71)  <-- concat ---------------
    |
Output (1, 46, 71)
```

**Key design choices:**
- **Skip connections** pass high-resolution spatial detail from encoder to decoder,
  preserving fine-grained temperature structure that would otherwise be lost during downsampling
- **Bilinear upsampling** in the decoder handles arbitrary grid sizes cleanly
- **BatchNorm + ReLU** after each convolution stabilises training
- **ReduceLROnPlateau** scheduler reduces learning rate when validation loss stalls

---

## Results

Output figures are saved in `figures/` after running `4_evaluate.py`:

- `training_curves.png` — train and validation MSE over 50 epochs
- `spatial_rmse.png` — map of forecast error across the European domain
- `example_forecast.png` — side-by-side: ground truth, prediction, and error field

The model is evaluated using RMSE and a skill score relative to persistence:

```
Skill score = 1 - (RMSE_model / RMSE_persistence)
```

A skill score above 0 means the model beats the persistence baseline.

---

## Project structure

```
ml-weather/
├── README.md
├── requirements.txt
├── 1_download_era5.py       <- Download ERA5 from Copernicus CDS
├── 2_preprocess.py          <- Normalise, split, save tensors
├── 3_train.py               <- U-Net definition and training loop
├── 4_evaluate.py            <- Skill scores and output figures
├── data/
│   ├── raw/                 <- Downloaded NetCDF files (not tracked by git)
│   └── processed/           <- Normalised PyTorch tensors (not tracked by git)
├── models/                  <- Saved checkpoints (not tracked by git)
└── figures/                 <- Output plots
```

---

## How to reproduce

### 1. Clone and set up

```bash
git clone https://github.com/yourusername/ml-weather.git
cd ml-weather
pip install -r requirements.txt
```

### 2. Register for ERA5 access (free)

Create an account at cds.climate.copernicus.eu, accept the ERA5 licence,
then create the file ~/.cdsapirc with your credentials:

```
url: https://cds.climate.copernicus.eu/api/v2
key: YOUR-UID:YOUR-API-KEY
```

### 3. Run the pipeline

```bash
python 1_download_era5.py    # downloads ERA5 data (~30 min)
python 2_preprocess.py       # normalises and splits data
python 3_train.py            # trains the U-Net (~20 min on CPU)
python 4_evaluate.py         # computes skill scores and saves figures
```

---

## Dependencies

- `cdsapi` — Copernicus CDS API client
- `xarray` + `netCDF4` — NetCDF file handling
- `numpy` — array operations
- `torch` — neural network training
- `matplotlib` — plotting

---

## Background

Numerical weather prediction has traditionally relied on physics-based models solving
the equations of atmospheric dynamics. Recent years have seen a rapid rise in
data-driven approaches: models like GraphCast (Google DeepMind) and Pangu-Weather
(Huawei) have demonstrated that neural networks can match or exceed operational NWP
systems at a fraction of the computational cost.

This project is a minimal, educational implementation of the core idea: learn a mapping
from today's atmospheric state to tomorrow's directly from observational data,
without explicitly encoding the physical equations.

---

## Possible extensions

- Probabilistic forecasting with MC dropout ensembles for uncertainty estimation
- Multi-step forecasting to show how error accumulates over time
- Additional input variables: wind fields (u, v), sea surface temperature
- Higher resolution using 0.25 degree ERA5 data
- Seasonal breakdown of RMSE to understand model limitations by time of year
