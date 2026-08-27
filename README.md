# Rogala Lab — Colocalization Analysis Pipeline

**Author:** Vaishnavi Nagesh  
**Lab:** Rogala Lab, Stanford University  
**Last updated:** August 2026  

Quantitative fluorescence microscopy pipeline for colocalization analysis of confocal images. Measures how much of a marker protein (Raptor, SHMT1, mTOR) is physically co-localised with a reference organelle or protein (LAMP1/lysosomes, SAMTOR).

---

## File Overview

```
potential-segmentation-quantizer/
├── utils.py                  ← Shared functions (import by all scripts)
├── ingest.py                 ← File discovery → manifest.tsv
├── param_sweep.py            ← Parameter calibration sweep
├── object_coloc.py           ← Primary analysis (object-based)
├── manders_coloc.py          ← Secondary analysis (intensity-based)
├── residual_factor_sweep.py  ← Background subtraction diagnostic
├── analyze_coloc_v3.py       ← Legacy pixel overlap (comparison only)
└── verify_channels.ipynb     ← Visual channel verification notebook
```

---

## Standard Workflow — Run In This Order

### Step 0 — Verify channels in FIJI

Before running anything, open one TIFF per condition in FIJI. Confirm:
- Which channel number is DAPI, LAMP1, and marker
- Whether images are z-stacks (3D) or single planes (2D)
- That signal looks reasonable — visible puncta, no saturation

Use `verify_channels.ipynb` if unsure.

---

### Step 1 — Ingest: discover files and write manifest

```bash
# Dry run first — no files written
python ingest.py \
  --image_dir "/path/to/images" \
  --layout flat \
  --ch_dapi 0 --ch_lamp1 1 --ch_marker 2 \
  --ch1_name LAMP1 \
  --marker_name Raptor \
  --cell_line HEK293 \
  --dry_run

# Once output looks correct, write the manifest
python ingest.py \
  --image_dir "/path/to/images" \
  --layout flat \
  --ch_dapi 0 --ch_lamp1 1 --ch_marker 2 \
  --ch1_name LAMP1 \
  --marker_name Raptor \
  --cell_line HEK293 \
  --output ~/Desktop/manifest.tsv \
  --integrity_check
```

**Open the manifest TSV** in Excel/Numbers and verify:
- Label names make biological sense
- File paths in `ch_dapi`, `ch_lamp1`, `ch_marker` columns point to the right files
- Edit anything wrong directly before proceeding

**Supported layouts:**

| Layout | Description | Example datasets |
|---|---|---|
| `subdir` | One subdirectory per condition | 071526 MiaPaca2 |
| `flat` | All files in one folder, condition in filename | 080326, HBSS timeline |

**Supported flat filename patterns (auto-detected):**

{cell_line}_{date}_{experiment_id}_{condition}_{dose}{unit}_{treatment}_{well}_{rep}_{channel}.tif

| Field | Description | Example values |
|-------|-------------|----------------|
|cellline |	Cell line used | MiaPaca2, HEK293 |
| yymmdd | Acquisition date | 071526 |
| expid | Experiment/project identifier | Exp109, 6698, HBSStimeline |
| nutrient | Nutrient/media condition | AAplus, AAminus, FED, ST, HBSS |
| treatment-dose | Drug + dose, or vehicle if none | AZD8055-10uM, DMSO-0uM, vehicle |
| timepoint | Duration/timepoint, NA if not applicable | t0, t2.5min, t120min, NA |
| well | Well identifier | well3, A1 |
| rep | Replicate number | rep1 |
| channel | Imaging channel — use marker name, not just ch1/ch2 | LAMP1, Raptor, SAMTOR, SHMT1, DAPI |

*Rules:*

- Use underscores _ between fields only — never within a field name
- Use hyphens - within the treatment-dose field only (e.g. AZD8055-10uM)
- Use NA for timepoint when the experiment is not a time course
Channel must be the marker protein name — never C0, C1, ch1 etc.
- Decimal timepoints use p as decimal separator: t2p5min not t2.5min
---

### Step 2 — Calibrate parameters

Run on one representative image — use your control condition (DMSO, Basal, FullMedia):

```bash
python object_coloc_sweep.py \
  --manifest ~/Desktop/manifest.tsv \
  --output ~/Desktop/sweep
```

Or with individual files:
```bash
python object_coloc_sweep.py \
  --dapi  "/path/C0.tif" \
  --lamp1 "/path/C1.tif" \
  --marker "/path/C2.tif" \
  --output ~/Desktop/sweep
```

**Three output files to inspect:**

| File | What to look for |
|---|---|
| `phase1_heatmaps.png` | Pick lamp1_percentile (rows) and marker_percentile (columns). Want ~20-50 marker puncta/cell, most cells included. Red box = auto-selected. |
| `phase2_proximity.png` | Pick proximity_px where the curve flattens. Linear rise = too many false positives. |
| `phase3_visual_comparison.png` | Visually confirm puncta look biologically real at recommended parameters. |

**Narrow the sweep if needed:**
```bash
python object_coloc_sweep.py \
  --manifest ~/Desktop/manifest.tsv \
  --residual_factors 0.5 1.0 1.5 \
  --lamp1_percentiles 78 82 85 \
  --marker_percentiles 88 90 93 \
  --proximities 2 3 5 \
  --output ~/Desktop/sweep_narrow
```

---

### Step 3 — Primary analysis: object-based colocalization

```bash
python object_coloc.py \
  --manifest ~/Desktop/manifest.tsv \
  --output_dir ~/Desktop/results \
  --projection max \
  --lamp1_percentile 75 \
  --marker_percentile 93 \
  --residual_factor 1.0 \
  --proximity_px 3
```

**Optional flags:**
```bash
--nuclear_cyto    # Also compute nuclear/cytoplasmic ratio for both channels
--spatial         # Also compute perinuclear vs peripheral puncta distribution
```

**Outputs:**
- `object_coloc_summary.csv` — per-cell metrics
- `object_coloc_summary.png` — summary bar chart
- `{condition}_{collection}_panel.png` — per-collection visual panels

**Always check panel PNGs first** before interpreting the CSV numbers.

---

### Step 4 — Secondary validation: Manders colocalization

```bash
python manders_coloc.py \
  --manifest ~/Desktop/manifest.tsv \
  --output_dir ~/Desktop/manders_results \
  --projection max \
  --skip_costes \
  --residual_factor 1.0
```

**When to use `--skip_costes`:**  
Look at the scatterplots from param_sweep. If channels show a flat horizontal cloud (r < 0.2), channels are not proportionally correlated — use `--skip_costes`. If there is a linear relationship (r > 0.4), run without it.

**Outputs:**
- `manders_summary.csv` — per-cell M1, M2, Pearson r
- `manders_summary.png` — three-panel bar chart (M2, M1, PCC)
- `scatterplots.png` — per-condition pixel intensity scatter with regression

---

### Step 5 — Interpret results

**Decision rule — which metric to use:**

| Situation | Recommended metric |
|---|---|
| Marker is punctate, reference is an organelle (LAMP1) | Object-based colocalization fraction |
| Channels are linearly correlated (r > 0.4, elongated scatter) | Manders M2 with Costes |
| Channels flat/uncorrelated (r < 0.2, horizontal scatter) | Object-based OR Manders with skip_costes |
| Protein redistribution between nucleus and cytoplasm | Nuclear/cytoplasmic ratio (`--nuclear_cyto`) |
| Protein condensation / puncta size changes | Puncta morphology columns in CSV |
| Density-driven colocalization concern | Enrichment ratio (observed / expected by chance) |

---

## Thresholding Options Reference

| Script | Method | Parameters |
|---|---|---|
| object_coloc.py | Per-channel intensity percentile | `--lamp1_percentile`, `--marker_percentile` |
| manders_coloc.py | Costes automatic thresholding | None — automatic per cell |
| manders_coloc.py | Skip Costes (threshold = 0) | `--skip_costes` |
| Both | Background subtraction aggressiveness | `--residual_factor`, `--window` |
| object_coloc_sweep.py | Sweeps all of the above | `--lamp1_percentiles`, `--marker_percentiles`, `--residual_factors`, `--proximities` |

---

## CSV Output Columns

**`object_coloc_summary.csv`**

| Column | Description |
|---|---|
| `label` | Condition name |
| `collection` | Field of view identifier |
| `cell_id` | Per-cell ID within collection |
| `n_ref_puncta` | Reference channel (LAMP1/SAMTOR) puncta count per cell |
| `n_marker_puncta` | Marker channel puncta count per cell |
| `n_coloc_puncta` | Marker puncta colocalized with reference |
| `coloc_frac` | Primary metric: n_coloc / n_marker |
| `ref_mean_area` | Mean reference punctum area (px²) |
| `marker_mean_area` | Mean marker punctum area (px²) — proxy for complex size |
| `ref_mean_intensity` | Mean intensity per reference punctum |
| `marker_mean_intensity` | Mean intensity per marker punctum |
| `ref_elongation` | Mean major/minor axis ratio (1=round, >1=elongated) |
| `marker_elongation` | Same for marker channel |
| `ref_nc_ratio` | Nuclear/cytoplasmic ratio for reference channel (if --nuclear_cyto) |
| `marker_nc_ratio` | Nuclear/cytoplasmic ratio for marker channel (if --nuclear_cyto) |
| `ref_mean_dist` | Mean normalised distance from nucleus (0=perinuclear, 1=peripheral) (if --spatial) |
| `marker_mean_dist` | Same for marker channel (if --spatial) |
| `marker_frac_perinuc` | Fraction of marker puncta in perinuclear zone (dist < 0.33) |
| `marker_frac_periph` | Fraction of marker puncta in peripheral zone (dist > 0.67) |
| `residual_factor` | Background subtraction rf used |
| `proximity_px` | Colocalization distance threshold used |
| `lamp1_percentile` | Reference channel threshold used |
| `marker_percentile` | Marker channel threshold used |
| `projection` | Z-stack projection method used |

**`manders_summary.csv`**

| Column | Description |
|---|---|
| `M1_lamp1_on_marker` | Fraction of reference intensity where marker > threshold |
| `M2_marker_on_lamp1` | Primary metric: fraction of marker intensity where reference > threshold |
| `pearson_r` | Pearson correlation (cytoplasm only, nucleus excluded) |
| `costes_thr_lamp1` | Costes threshold applied to reference channel |
| `costes_thr_marker` | Costes threshold applied to marker channel |

---

## Parameters Explained

**`--residual_factor`** (default 1.0)  
After local median subtraction, subtracts `residual_factor × noise_std` as a final offset to zero out background. Lower = more signal survives. Use `residual_factor_sweep.py` to calibrate.
- 0.0 = pure median subtraction, most permissive
- 1.0 = remove 1σ noise (default, moderate)
- 2.0 = conservative, may remove weak real signal

**`--lamp1_percentile` / `--marker_percentile`** (defaults 82 / 93)  
Within each cell's cytoplasm, pixels above this percentile of background-subtracted signal are called as puncta. Marker threshold is higher than reference because marker channels typically have more diffuse background.

**`--proximity_px`** (default 3)  
LAMP1 mask is dilated by this many pixels before checking whether a marker punctum centroid falls within it. Accounts for slight channel registration offset and the fact that mTOR docks adjacent to the lysosomal surface rather than pixel-identical.

**`--projection`** (default max)  
How to collapse z-stacks. `max` = maximum intensity projection, captures all puncta across z but slightly overestimates colocalization. `best_z` = single sharpest plane, faster. `mean` = average projection.

**`--window`** (default 32)  
Local median background estimation window in pixels. Must be large enough to measure background between puncta (puncta should occupy < 50% of window). 24-36px recommended for 100x images.

---

## Methods Summary (for papers/grants)

> Confocal z-stacks were collapsed using maximum intensity projection. Local median background subtraction (32×32 pixel window) was applied to each channel to remove spatially varying illumination gradients. Nuclei were segmented from DAPI using Otsu thresholding, watershed separation, and morphological operations; the nucleus was excluded from all colocalization measurements. Discrete puncta were detected as connected pixel regions above the 82nd (reference) and 93rd (marker) percentile of background-subtracted cytoplasmic signal, with a minimum size of 5 pixels. Object-based colocalization was scored per cell as the fraction of marker puncta whose centroids fell within the reference channel mask dilated by 3 pixels. All metrics were computed per cell and averaged across cells per condition.

---

## Dependencies

```bash
pip install tifffile scikit-image scipy matplotlib numpy pandas --break-system-packages
```

Python 3.10+ required. All scripts tested on Python 3.12.

---

## Troubleshooting

**"No valid triplets found" from ingest.py**  
Filename pattern not recognised. Run `ls` on the folder and check which pattern applies. Use `--flat_pattern` to specify manually.

**Corrupt file error mid-analysis**  
Add `--integrity_check` to ingest.py run to detect bad files before analysis. Both object_coloc.py and manders_coloc.py skip corrupt files automatically with a warning.

**M2 values near zero / most cells have M2 = 0**  
Costes thresholds are too high. Add `--skip_costes` to manders_coloc.py. Check `costes_thr_lamp1` column in CSV — if values are near the maximum intensity, Costes is failing.

**Colocalization fraction inflated in high-expression conditions**  
Density bias — more reference puncta means higher geometric probability of overlap. Use per-cell matched normalisation (see notebook) or enrichment ratio: `coloc_frac / (dilated_ref_mask_area / cyto_area)`.

**Puncta look noisy in panel images**  
Lower `--residual_factor` to preserve more signal, or increase `--lamp1_percentile` / `--marker_percentile` to be more selective. Run `param_sweep.py` to calibrate visually.

**"ch_dapi not found" AttributeError**  
Download the latest version of the script — this happens when `--ch_dapi` argument was accidentally dropped during an edit.

---

## Contact

Vaishnavi Nagesh — nageshv@stanford.edu (or current email)  
Rogala Lab — Stanford University