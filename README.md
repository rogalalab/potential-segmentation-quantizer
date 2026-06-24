# potential-segmentation-quantizer

## Overview

This repository contains a single analysis script for mTOR–lysosome colocalization in the Rogala Lab MiaPaca2 nutrient perturbation dataset. The script:

- loads multi-plane TIFF images for DAPI, LAMP1, and mTOR channels
- selects the best focal plane for each channel
- segments nuclei and expands them into cell regions
- detects puncta in LAMP1 and mTOR channels
- computes per-cell mTOR/LAMP1 colocalization metrics
- saves per-condition visualization panels and a 2×2 summary figure
- writes a CSV summary table of colocalization statistics

## Input data

Place the source TIFF files in the repository root, then run the script from that folder. The script expects filenames matching the pattern:

```
MiaPaca2_{COND}_{GLUC}..._C{ch}.tif
```

Where:

- `{COND}` is `FED` or `ST`
- `{GLUC}` is `HG` or `LG`
- `C0` is the DAPI channel
- `C1` is the LAMP1 channel
- `C3` is the mTOR channel

## Requirements

Install the required Python packages:

```bash
pip install tifffile scikit-image scipy matplotlib numpy pandas
```

## Usage

Run the analysis script from the repo folder containing the TIFFs:

```bash
cd '/Users/vaishnavinagesh/Desktop/Rogala Lab Images Olga/potential-segmentation-quantizer'
python analyze_coloc.py
```

## Outputs

The script writes results into `./coloc_results/`:

- `coloc_summary.csv` — per-cell colocalization measurements for all processed conditions
- `coloc_summary.png` — 2×2 summary figure comparing FED vs STARVED and high vs low glucose
- `<COND>_<GLUC>_panel.png` — individual condition visualization panels

## What it measures

For each segmented cell, the script computes:

- total mTOR puncta pixels inside the cell region
- total LAMP1 puncta pixels inside the cell region
- overlapping pixels where mTOR and LAMP1 puncta coincide
- fraction of mTOR pixels that overlap with LAMP1 pixels (`coloc_frac`)

Cells with very low mTOR signal are excluded from the per-cell results.

## Notes

- The script automatically selects the z-plane with the highest variance for each channel.
- Segmentation and puncta detection use tunable parameters at the top of `analyze_coloc.py`.
- If a condition is missing required channels, it will be skipped and a warning printed.

## Contact

For questions about the analysis pipeline or dataset naming conventions, refer to the Rogala Lab workflow or ask the original dataset owner.