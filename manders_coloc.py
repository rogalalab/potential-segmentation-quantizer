"""
Manders Colocalization Analysis — Rogala Lab
Computes M1 and M2 Manders coefficients per cell with Costes automatic thresholding

M1 = fraction of LAMP1 intensity overlapping with Raptor (how much lysosome is covered by Raptor)
M2 = fraction of Raptor intensity overlapping with LAMP1 (how much Raptor is on lysosomes)

M2 is the primary biological readout: is Raptor recruited to the lysosome?

Supports same directory layouts as analyze_coloc_v3.py:
  subdir   — one subdirectory per condition, multiple collections pooled
  miapaca2 — flat folder, condition in filename
  hek293   — flat folder, condition in filename

Usage:
  python manders_coloc.py --image_dir ~/Desktop/071526 --dataset subdir
  python manders_coloc.py --image_dir ~/Desktop/071526 --dataset subdir --output_dir ~/Desktop/manders_results

Requirements:
  pip install tifffile scikit-image scipy matplotlib numpy pandas
"""

import os, re, argparse, warnings
from collections import defaultdict
import numpy as np
import pandas as pd
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage import filters, morphology, measure, segmentation, feature
from skimage.filters import gaussian
from scipy import ndimage as ndi
from scipy.stats import pearsonr

warnings.filterwarnings('ignore')

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Manders colocalization analysis')
parser.add_argument('--image_dir',  type=str, required=True)
parser.add_argument('--output_dir', type=str, default=None)
parser.add_argument('--dataset',    type=str, default='subdir',
                    choices=['miapaca2', 'hek293', 'subdir'])
parser.add_argument('--ch_dapi',   type=int, default=None)
parser.add_argument('--ch_lamp1',  type=int, default=None)
parser.add_argument('--ch_marker', type=int, default=None)
args = parser.parse_args()

IMAGE_DIR  = args.image_dir
OUTPUT_DIR = args.output_dir or os.path.join(IMAGE_DIR, 'manders_results')
DATASET    = args.dataset
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Dataset config ────────────────────────────────────────────────────────────
if DATASET == 'miapaca2':
    CH_DAPI, CH_LAMP1, CH_MARKER = 0, 1, 3
    MARKER_NAME = 'mTOR';   CELL_LINE = 'MiaPaca2'
elif DATASET in ('hek293', 'subdir'):
    CH_DAPI, CH_LAMP1, CH_MARKER = 0, 1, 2
    MARKER_NAME = 'Raptor'; CELL_LINE = 'HEK293'

if args.ch_dapi   is not None: CH_DAPI   = args.ch_dapi
if args.ch_lamp1  is not None: CH_LAMP1  = args.ch_lamp1
if args.ch_marker is not None: CH_MARKER = args.ch_marker

print(f"Dataset:   {DATASET}  |  Marker: {MARKER_NAME} (C{CH_MARKER})")
print(f"Image dir: {IMAGE_DIR}")
print(f"Output:    {OUTPUT_DIR}\n")

# ── Segmentation parameters ───────────────────────────────────────────────────
NUC_SIGMA     = 3
NUC_MIN_SIZE  = 2000
NUC_ERODE     = 4
NUC_EXPAND_PX = 30
NUC_PEAK_DIST = 40

# ── Core helpers ──────────────────────────────────────────────────────────────
def norm(img, plow=1, phigh=99.5):
    lo, hi = np.percentile(img, plow), np.percentile(img, phigh)
    return np.clip((img - lo) / (hi - lo + 1e-6), 0, 1)

def best_z(stack):
    return int(np.argmax([np.var(stack[z]) for z in range(stack.shape[0])]))

def load_plane(path):
    print(f"      loading {os.path.basename(path)}...", end=' ', flush=True)
    stack = tifffile.imread(path).astype(np.float32)
    plane = stack[best_z(stack)] if stack.ndim == 3 else stack
    print(f"done {plane.shape}", flush=True)
    return plane

def segment_nuclei(dapi):
    smooth      = gaussian(dapi, sigma=NUC_SIGMA)
    thresh      = filters.threshold_otsu(smooth)
    mask        = smooth > thresh
    mask        = morphology.remove_small_objects(mask, min_size=NUC_MIN_SIZE)
    mask        = ndi.binary_fill_holes(mask)
    mask        = morphology.binary_erosion(mask, morphology.disk(NUC_ERODE))
    dist        = ndi.distance_transform_edt(mask)
    peaks       = feature.peak_local_max(dist, min_distance=NUC_PEAK_DIST, labels=mask)
    pm          = np.zeros(dist.shape, dtype=bool)
    pm[tuple(peaks.T)] = True
    markers     = measure.label(pm)
    nuc_labels  = segmentation.watershed(-dist, markers, mask=mask)
    cell_mask   = morphology.dilation(mask, morphology.disk(NUC_EXPAND_PX))
    cell_labels = segmentation.watershed(-dist, markers, mask=cell_mask)
    return nuc_labels, cell_labels

# ── Costes automatic thresholding ─────────────────────────────────────────────
def costes_threshold(ch1, ch2, n_steps=100):
    """
    Find the threshold for each channel such that pixels below threshold
    in either channel have zero or negative Pearson correlation.

    Scans thresholds from max down to min in n_steps.
    Returns (thr1, thr2) — threshold for ch1 and ch2 respectively.

    Reference: Costes et al., Biophysical Journal 2004
    """
    max1, min1 = ch1.max(), ch1.min()
    max2, min2 = ch2.max(), ch2.min()

    # Scan threshold as fraction of max intensity
    thresholds1 = np.linspace(max1, min1, n_steps)

    thr1_final, thr2_final = max1 * 0.15, max2 * 0.15  # fallback

    for thr1 in thresholds1:
        # Corresponding threshold in ch2 estimated by linear regression
        # Use pixels below current threshold to estimate ch2 threshold
        mask_below = (ch1 < thr1) & (ch2 > 0)
        if mask_below.sum() < 10:
            continue
        # Linear fit: ch2 = a * ch1 + b for pixels below threshold
        x = ch1[mask_below].flatten()
        y = ch2[mask_below].flatten()
        if x.std() < 1e-6:
            continue
        a = np.cov(x, y)[0, 1] / np.var(x)
        b = y.mean() - a * x.mean()
        thr2 = a * thr1 + b

        # Check Pearson correlation for pixels below both thresholds
        mask_sub = (ch1 < thr1) & (ch2 < thr2)
        if mask_sub.sum() < 10:
            continue
        r, _ = pearsonr(ch1[mask_sub].flatten(), ch2[mask_sub].flatten())
        if r <= 0:
            thr1_final = thr1
            thr2_final = max(thr2, 0)
            break

    return thr1_final, thr2_final

# ── Manders coefficients ──────────────────────────────────────────────────────
def manders_coefficients(ch1, ch2, thr1, thr2, cell_roi):
    """
    Compute M1 and M2 within a cell ROI using Costes thresholds.

    M1 = sum(ch1 where ch2 > thr2) / sum(ch1)   — fraction of ch1 on ch2
    M2 = sum(ch2 where ch1 > thr1) / sum(ch2)   — fraction of ch2 on ch1

    For our experiment:
      ch1 = LAMP1,  ch2 = Raptor
      M1 = fraction of LAMP1 intensity covered by Raptor
      M2 = fraction of Raptor intensity on lysosomes  ← primary readout
    """
    # Restrict to cell ROI
    c1_roi = ch1 * cell_roi
    c2_roi = ch2 * cell_roi

    total_c1 = c1_roi.sum()
    total_c2 = c2_roi.sum()

    if total_c1 < 1 or total_c2 < 1:
        return np.nan, np.nan, np.nan

    # M1: LAMP1 intensity where Raptor is above threshold
    M1 = c1_roi[c2_roi > thr2].sum() / total_c1

    # M2: Raptor intensity where LAMP1 is above threshold
    M2 = c2_roi[c1_roi > thr1].sum() / total_c2

    # Also compute Pearson r within ROI (useful diagnostic)
    flat1 = c1_roi[cell_roi].flatten()
    flat2 = c2_roi[cell_roi].flatten()
    if flat1.std() > 0 and flat2.std() > 0:
        r, _ = pearsonr(flat1, flat2)
    else:
        r = np.nan

    return float(M1), float(M2), float(r)

def analyze_collection(c0, c1, cm, label, collection_id):
    """Segment cells and compute Manders per cell."""
    print(f"    [2/3] Segmenting nuclei...", end=' ', flush=True)
    nuc_labels, cell_labels = segment_nuclei(c0)
    print(f"done ({nuc_labels.max()} nuclei)", flush=True)

    print(f"    [3/3] Computing Manders per cell...", end=' ', flush=True)

    # Compute Costes thresholds for the full image
    thr_lamp1, thr_marker = costes_threshold(c1, cm)

    rows = []
    for cid in range(1, cell_labels.max() + 1):
        roi = (cell_labels == cid).astype(float)

        # Skip cells with very little signal
        if cm[roi.astype(bool)].sum() < 500:
            continue

        M1, M2, r = manders_coefficients(c1, cm, thr_lamp1, thr_marker, roi.astype(bool))

        if np.isnan(M2):
            continue

        rows.append({
            'label'       : label,
            'collection'  : collection_id,
            'cell_id'     : cid,
            'M1_lamp1_on_marker' : M1,   # fraction LAMP1 covered by Raptor
            'M2_marker_on_lamp1' : M2,   # fraction Raptor on lysosome ← key metric
            'pearson_r'   : r,
            'thr_lamp1'   : thr_lamp1,
            'thr_marker'  : thr_marker,
        })

    print(f"done — {len(rows)} cells", flush=True)
    return rows

# ── File discovery (same logic as v3) ────────────────────────────────────────
COLLECTION_FILE_RE = re.compile(r'.*_C(\d)\.tif$', re.IGNORECASE)

def find_triplet(file_list, folder):
    stems = defaultdict(dict)
    for f in file_list:
        m = COLLECTION_FILE_RE.match(f)
        if m:
            ch   = int(m.group(1))
            stem = f[:m.start(1)-1]
            stems[stem][ch] = os.path.join(folder, f)
    triplets = []
    for stem, channels in stems.items():
        if all(ch in channels for ch in [CH_DAPI, CH_LAMP1, CH_MARKER]):
            triplets.append((channels[CH_DAPI], channels[CH_LAMP1],
                             channels[CH_MARKER], os.path.basename(stem)))
    return sorted(triplets)

groups = defaultdict(list)

if DATASET == 'subdir':
    for entry in sorted(os.listdir(IMAGE_DIR)):
        full = os.path.join(IMAGE_DIR, entry)
        if not os.path.isdir(full) or entry.startswith('.') or 'result' in entry.lower():
            continue
        triplets = find_triplet(sorted(os.listdir(full)), full)
        if triplets:
            groups[entry].extend(triplets)
            print(f"  {entry}: {len(triplets)} collection(s)")

elif DATASET == 'miapaca2':
    COND_RE = re.compile(r'MiaPaca2_(FED|ST)_(HG|LG).*_C(\d)\.tif$', re.IGNORECASE)
    stems = defaultdict(dict)
    for f in sorted(os.listdir(IMAGE_DIR)):
        m = COND_RE.search(f)
        if m:
            cond, gluc, ch = m.group(1).upper(), m.group(2).upper(), int(m.group(3))
            label = {'FED':'FED','ST':'STARVED'}[cond]+' / '+{'HG':'High Glucose','LG':'Low Glucose'}[gluc]
            stem  = f[:f.rfind('_C')]
            stems[(label,stem)][ch] = os.path.join(IMAGE_DIR,f)
    for (label,stem),channels in stems.items():
        if all(ch in channels for ch in [CH_DAPI,CH_LAMP1,CH_MARKER]):
            groups[label].append((channels[CH_DAPI],channels[CH_LAMP1],channels[CH_MARKER],stem))

elif DATASET == 'hek293':
    COND_RE = re.compile(
        r'(?:Slide\s*\d+_)?(DMSO|M6659|AZD8055)_(No_Refed|Refed).*_C(\d)\.tif$', re.IGNORECASE)
    DRUG_L  = {'DMSO':'DMSO','M6659':'25µM M6659','AZD8055':'100nM AZD8055'}
    REFED_L = {'NO_REFED':'No Refed','REFED':'Refed'}
    stems = defaultdict(dict)
    for f in sorted(os.listdir(IMAGE_DIR)):
        m = COND_RE.search(f)
        if m:
            drug=m.group(1).upper(); refed=m.group(2).upper().replace(' ','_'); ch=int(m.group(3))
            label=DRUG_L.get(drug,drug)+' / '+REFED_L.get(refed,refed)
            stem=f[:f.rfind('_C')]
            stems[(label,stem)][ch]=os.path.join(IMAGE_DIR,f)
    for (label,stem),channels in stems.items():
        if all(ch in channels for ch in [CH_DAPI,CH_LAMP1,CH_MARKER]):
            groups[label].append((channels[CH_DAPI],channels[CH_LAMP1],channels[CH_MARKER],stem))

label_list = sorted(groups.keys())
print(f"\nConditions: {len(groups)}")
for lbl, trips in sorted(groups.items()):
    print(f"  {lbl}: {len(trips)} field(s)")

# ── Main loop ─────────────────────────────────────────────────────────────────
all_rows = []
cond_summary = {}  # label -> {'M1': [], 'M2': [], 'r': []}

PALETTE = ['#D4845A','#4A9BAF','#7B9E6B','#A07BC4','#C4A07B','#6B9EA0','#C47B7B','#7B7BC4']

for idx, label in enumerate(label_list):
    triplets = groups[label]
    print(f"\n══ {label} ({len(triplets)} field(s)) ══", flush=True)

    cond_M1, cond_M2, cond_r = [], [], []

    for c0_path, c1_path, cm_path, collection_id in triplets:
        print(f"  Collection: {collection_id}", flush=True)
        print(f"    [1/3] Loading...", flush=True)
        c0 = load_plane(c0_path)
        c1 = load_plane(c1_path)
        cm = load_plane(cm_path)

        rows = analyze_collection(c0, c1, cm, label, collection_id)
        all_rows.extend(rows)

        for r in rows:
            cond_M1.append(r['M1_lamp1_on_marker'])
            cond_M2.append(r['M2_marker_on_lamp1'])
            cond_r.append(r['pearson_r'])

    cond_summary[label] = {
        'M1': cond_M1, 'M2': cond_M2, 'r': cond_r,
        'color': PALETTE[idx % len(PALETTE)]
    }

    if cond_M2:
        print(f"\n  ► {label}")
        print(f"    M2 (Raptor on lysosome): mean={np.mean(cond_M2):.3f}  "
              f"median={np.median(cond_M2):.3f}  std={np.std(cond_M2):.3f}  n={len(cond_M2)}")
        print(f"    M1 (LAMP1 covered by Raptor): mean={np.mean(cond_M1):.3f}")
        print(f"    Pearson r: mean={np.mean(cond_r):.3f}", flush=True)

# ── CSV ───────────────────────────────────────────────────────────────────────
df = pd.DataFrame(all_rows)
csv_path = os.path.join(OUTPUT_DIR, 'manders_summary.csv')
df.to_csv(csv_path, index=False)
print(f"\nCSV: {csv_path}")

if not df.empty:
    print("\nM2 summary by condition:")
    print(df.groupby('label')['M2_marker_on_lamp1']
            .agg(['count','mean','median','std'])
            .round(3)
            .sort_values('mean', ascending=False))

# ── Summary figure ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(20, 7), facecolor='#F8F7F4')
fig.suptitle(f'{CELL_LINE} — Manders Colocalization: {MARKER_NAME} ↔ LAMP1',
             color='#1E2D3A', fontsize=14, fontweight='bold')

metrics = [
    ('M2', 'M2: Fraction of Raptor on Lysosome\n(primary readout — higher = more mTORC1 at lysosome)', 'M2_marker_on_lamp1'),
    ('M1', 'M1: Fraction of LAMP1 covered by Raptor\n(how much lysosome surface has Raptor)', 'M1_lamp1_on_marker'),
    ('r',  'Pearson r: Channel correlation within cell\n(overall colocalization tendency)', 'pearson_r'),
]

for ax, (key, ylabel, col_name) in zip(axes, metrics):
    ax.set_facecolor('#F0F4F6')

    means, sems, xlabels, colors = [], [], [], []
    for label in label_list:
        vals = cond_summary[label][key]
        if not vals: continue
        vals = [v for v in vals if not np.isnan(v)]
        if not vals: continue
        means.append(np.mean(vals))
        sems.append(np.std(vals) / np.sqrt(len(vals)))
        xlabels.append(label.replace('_', '\n').replace(' / ', '\n'))
        colors.append(cond_summary[label]['color'])

    x = np.arange(len(means))
    ax.bar(x, means, yerr=sems, color=colors, edgecolor='none', width=0.6,
           capsize=5, error_kw={'ecolor':'#1E2D3A','linewidth':1.5})
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, color='#1E2D3A', fontsize=7, rotation=25, ha='right')
    ax.set_ylabel(ylabel, color='#1E2D3A', fontsize=9)
    ax.set_ylim(0, max(means) * 1.25 if means else 1)
    ax.tick_params(colors='#1E2D3A', labelsize=7)
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    for sp in ['bottom','left']: ax.spines[sp].set_color('#A0B8C4')
    for xi, (m, s) in enumerate(zip(means, sems)):
        ax.text(xi, m + s + 0.01, f'{m:.3f}', ha='center', va='bottom',
                color='#1E2D3A', fontsize=7, fontweight='bold')

    # Highlight M2 as primary
    if key == 'M2':
        ax.set_title('PRIMARY METRIC', color='#D4845A', fontsize=9, fontweight='bold')

plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, 'manders_summary.png')
plt.savefig(out_path, dpi=130, bbox_inches='tight', facecolor='#F8F7F4')
plt.close()
print(f"\nFigure: {out_path}")
print(f"Done. All outputs in: {OUTPUT_DIR}")