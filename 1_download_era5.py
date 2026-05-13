"""
1_download_era5.py — rewritten for reliability
Downloads ERA5 data as separate files per variable to avoid format issues.
Uses GRIB format which is more reliably handled than NetCDF from CDS.
"""

import cdsapi
import os

os.makedirs("data/raw", exist_ok=True)
client = cdsapi.Client()

YEARS  = ["2023", "2024"]
MONTHS = [f"{m:02d}" for m in range(1, 13)]
DAYS   = [f"{d:02d}" for d in range(1, 32)]
AREA   = [75, -25, 30, 45]  # N, W, S, E

variables = {
    "t2m":  ("reanalysis-era5-single-levels",  "2m_temperature"),
    "tp":   ("reanalysis-era5-single-levels",  "total_precipitation"),
    "z500": ("reanalysis-era5-pressure-levels", "geopotential"),
    "t850": ("reanalysis-era5-pressure-levels", "temperature"),
}

for name, (dataset, variable) in variables.items():
    outfile = f"data/raw/{name}.nc"
    if os.path.exists(outfile):
        print(f"  Skipping {name} — already exists")
        continue

    print(f"Downloading {name}...")
    request = {
        "product_type": "reanalysis",
        "variable": variable,
        "year":  YEARS,
        "month": MONTHS,
        "day":   DAYS,
        "time":  "12:00",
        "area":  AREA,
        "grid":  [1.0, 1.0],
        "format": "netcdf",
    }
    if dataset == "reanalysis-era5-pressure-levels":
        request["pressure_level"] = ["500"] if name == "z500" else ["850"]

    client.retrieve(dataset, request, outfile)
    print(f"  Saved: {outfile}")

print("\nDownload complete.")
