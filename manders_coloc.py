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
parser.add_argument('--image_dir',  type=str, default=None,
                    help='Root folder containing images. Not required if --manifest is provided.')
parser.add_argument('--output_dir', type=str, default=None)
parser.add_argument('--manifest',   type=str, default=None,
                    help='Path to manifest TSV from ingest.py. '
                         'If provided, --image_dir and --dataset are ignored for file discovery.')
parser.add_argument('--dataset',    type=str, default='subdir',
                    choices=['miapaca2', 'hek293', 'subdir'])
parser.add_argument('--projection', type=str, default='best_z',
                    choices=['best_z', 'max', 'mean', 'sum'],
                    help='How to collapse z-stack: '
                         'best_z = sharpest single plane (default, fast); '
                         'max = maximum intensity projection (recommended, captures all puncta); '
                         'mean = average projection; '
                         'sum = sum projection')
parser.add_argument('--residual_factor', type=float, default=2.0,
                    help='Multiplier on noise std for final background offset removal '
                         'after local median subtraction (default 2.0). '
                         'Decrease to 0.5 or 1.0 if too much real signal is being zeroed out. '
                         'Increase to 3.0 for noisier images.')
parser.add_argument('--skip_costes', action='store_true',
                    help='Skip Costes thresholding and use threshold=0 on background-subtracted '
                         'images. Recommended when channels are not proportionally correlated '
                         '(confirmed by flat scatterplots). Per Dunn et al. 2011: after local '
                         'median subtraction, pixels > 0 are signal by definition.')
parser.add_argument('--ch_dapi',   type=int, default=None)
parser.add_argument('--ch_lamp1',  type=int, default=None)
parser.add_argument('--ch_marker', type=int, default=None)
args = parser.parse_args()

IMAGE_DIR  = args.image_dir
OUTPUT_DIR = args.output_dir or (
    os.path.join(IMAGE_DIR, 'manders_results') if IMAGE_DIR
    else os.path.join(os.path.dirname(args.manifest), 'manders_results')
)
DATASET    = args.dataset
os.makedirs(OUTPUT_DIR, exist_ok=True)

if not args.manifest and not IMAGE_DIR:
    print("❌ Must provide either --image_dir or --manifest")
    exit(1)

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

PROJECTION      = args.projection
SKIP_COSTES     = args.skip_costes
RESIDUAL_FACTOR = args.residual_factor

print(f"Dataset:          {DATASET}  |  Marker: {MARKER_NAME} (C{CH_MARKER})")
print(f"Projection:       {PROJECTION}")
print(f"Costes:           {'SKIPPED — threshold=0 on background-subtracted images' if SKIP_COSTES else 'enabled'}")
print(f"Residual factor:  {RESIDUAL_FACTOR}")
print(f"Image dir:        {IMAGE_DIR}")
print(f"Output:           {OUTPUT_DIR}\n")

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
    """
    Load a TIFF and collapse z-stack according to PROJECTION mode.

    best_z: single sharpest plane (highest variance) — fast, misses out-of-focus puncta
    max:    maximum intensity projection — captures all puncta across z,
            slight MCC overestimate due to superimposition (Dunn et al. 2011)
    mean:   average projection — reduces noise but may dilute puncta signal
    sum:    sum projection — preserves total intensity, similar to mean * n_planes
    """
    print(f"      loading {os.path.basename(path)}...", end=' ', flush=True)
    stack = tifffile.imread(path).astype(np.float32)

    if stack.ndim == 2:
        print(f"done {stack.shape} (2D)", flush=True)
        return stack

    n_z = stack.shape[0]

    if PROJECTION == 'best_z':
        plane = stack[best_z(stack)]
        print(f"done {plane.shape} (best z/{n_z})", flush=True)
    elif PROJECTION == 'max':
        plane = stack.max(axis=0)
        print(f"done {plane.shape} (max projection, {n_z} planes)", flush=True)
    elif PROJECTION == 'mean':
        plane = stack.mean(axis=0)
        print(f"done {plane.shape} (mean projection, {n_z} planes)", flush=True)
    elif PROJECTION == 'sum':
        plane = stack.sum(axis=0)
        print(f"done {plane.shape} (sum projection, {n_z} planes)", flush=True)

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

# ── Local median background subtraction ──────────────────────────────────────
def local_median_subtract(img, window=32, residual_factor=2.0):
    """
    Spatially adaptive background subtraction using local median.

    For each pixel, background = median intensity in (window x window) neighborhood.
    This tracks slowly-varying background from uneven illumination or out-of-focus
    fluorescence, while puncta appear as sharp spikes above it.

    Steps:
      1. Compute local median image (sliding window)
      2. Subtract local median from original
      3. Estimate residual noise from near-zero pixels
      4. Subtract residual_factor * noise_std to zero out background
      5. Clip negatives to zero

    Window size (Dunn et al. 2011):
      Must be large enough to measure background between puncta but small enough
      to track spatial variation. 24-36 px recommended for 100x images.
      Critical assumption: puncta occupy < 50% of the window area.

    residual_factor: multiplier on noise std for final offset subtraction.
      2.0 is conservative; increase to 3.0 for noisier images.

    Reference: Dunn KW et al., AJP Cell Physiol 2011 (Fig. 7)
    """
    from skimage.filters import rank
    from skimage.morphology import disk

    img_norm = img - img.min()
    scale    = 65535.0 / (img_norm.max() + 1e-6)
    img_u16  = (img_norm * scale).astype(np.uint16)

    radius   = window // 2
    local_bg = rank.median(img_u16, disk(radius)).astype(np.float32) / scale

    subtracted = img - local_bg

    # Estimate noise from background pixels (lower half of distribution)
    near_zero = subtracted[subtracted < np.percentile(subtracted, 50)]
    noise_std  = near_zero.std() if len(near_zero) > 10 else 1.0

    subtracted = subtracted - residual_factor * noise_std
    return np.clip(subtracted, 0, None)


# ── Costes automatic thresholding ────────────────────────────────────────────
def costes_threshold(ch1_sub, ch2_sub, n_steps=200):
    """
    Costes et al. (2004) thresholding on background-subtracted images.

    Because images are already locally background-subtracted, many pixels
    are zero — this avoids the Costes failure mode where thresholds are
    too low to discriminate signal from background (Dunn et al. 2011, Fig 7).

    Step 1: Global linear regression on nonzero pixels: ch2 = a*ch1 + b
    Step 2: Scan T1 from max down to 0.
            T2 = a*T1 + b (from regression line).
            PCC of pixels below both thresholds.
            Stop when PCC <= 0.

    Reference: Costes SV et al., Biophys J 2004; Dunn KW et al., AJP Cell 2011
    """
    ch1_flat = ch1_sub.flatten()
    ch2_flat = ch2_sub.flatten()

    nonzero = (ch1_flat > 0) | (ch2_flat > 0)
    x = ch1_flat[nonzero]
    y = ch2_flat[nonzero]

    if len(x) < 20 or x.std() < 1e-6:
        return float(ch1_sub.max() * 0.1), float(ch2_sub.max() * 0.1)

    # Step 1: single global regression
    cov = np.cov(x, y)
    a   = cov[0, 1] / (np.var(x) + 1e-9)
    b   = y.mean() - a * x.mean()

    thr1_final = float(ch1_sub.max() * 0.1)
    thr2_final = float(ch2_sub.max() * 0.1)

    for thr1 in np.linspace(ch1_sub.max(), 0, n_steps):
        thr2 = max(a * thr1 + b, 0)

        sub = (ch1_flat <= thr1) & (ch2_flat <= thr2) & nonzero
        if sub.sum() < 10:
            continue

        x_sub, y_sub = ch1_flat[sub], ch2_flat[sub]
        if x_sub.std() < 1e-6 or y_sub.std() < 1e-6:
            continue

        r, _ = pearsonr(x_sub, y_sub)
        if r <= 0:
            thr1_final = float(thr1)
            thr2_final = float(thr2)
            break

    return thr1_final, thr2_final


# ── PCC ───────────────────────────────────────────────────────────────────────
def pearson_in_roi(ch1_sub, ch2_sub, cyto_roi):
    """
    Pearson r within cytoplasmic ROI on background-subtracted images.
    Only signal pixels (at least one channel > 0) are used.
    Nucleus excluded. No thresholding needed for PCC.
    Per Dunn et al.: measure per cell not per image.
    """
    x = ch1_sub[cyto_roi].flatten()
    y = ch2_sub[cyto_roi].flatten()

    has_signal = (x > 0) | (y > 0)
    x, y = x[has_signal], y[has_signal]

    if len(x) < 10 or x.std() < 1e-6 or y.std() < 1e-6:
        return np.nan

    r, _ = pearsonr(x, y)
    return float(r)


# ── Manders M1 and M2 ────────────────────────────────────────────────────────
def manders_coefficients(ch1_sub, ch2_sub, thr1, thr2, cyto_roi):
    """
    Manders Colocalization Coefficients on background-subtracted images.
    Nucleus excluded from ROI.

    ch1 = LAMP1 (background subtracted)
    ch2 = Raptor / mTOR (background subtracted)

    M1 = fraction of LAMP1 intensity where Raptor > thr2
    M2 = fraction of Raptor intensity where LAMP1 > thr1  <- PRIMARY

    Reference: Manders et al. J Microsc 1993; Dunn et al. AJP Cell 2011
    """
    c1 = ch1_sub * cyto_roi
    c2 = ch2_sub * cyto_roi

    total_c1 = c1.sum()
    total_c2 = c2.sum()

    if total_c1 < 1 or total_c2 < 1:
        return np.nan, np.nan

    M1 = c1[c2 > thr2].sum() / total_c1
    M2 = c2[c1 > thr1].sum() / total_c2

    return float(M1), float(M2)


# ── Per-collection analysis ───────────────────────────────────────────────────
def analyze_collection(c0, c1, cm, label, collection_id,
                       median_window=32, residual_factor=2.0):
    """
    Full pipeline per collection:
      1. Segment nuclei + cells from DAPI
      2. Local median background subtraction on LAMP1 and Raptor/mTOR
      3. Per cell (cytoplasm only, nucleus excluded):
           - PCC on background-subtracted signal pixels
           - Costes thresholds on per-cell background-subtracted pixels
           - M1 and M2 using those thresholds

    median_window:   background estimation window (default 32px)
    residual_factor: noise multiplier for offset removal (default 2.0)
    """
    print(f"    [2/4] Segmenting nuclei...", end=' ', flush=True)
    nuc_labels, cell_labels = segment_nuclei(c0)
    print(f"done ({nuc_labels.max()} nuclei)", flush=True)

    print(f"    [3/4] Local median background subtraction...", end=' ', flush=True)
    c1_sub = local_median_subtract(c1, window=median_window,
                                   residual_factor=residual_factor)
    cm_sub = local_median_subtract(cm, window=median_window,
                                   residual_factor=residual_factor)
    print(f"done", flush=True)

    print(f"    [4/4] PCC + Costes + Manders per cell...", end=' ', flush=True)

    rows = []
    for cid in range(1, cell_labels.max() + 1):
        cell_roi = (cell_labels == cid)
        nuc_roi  = (nuc_labels  == cid)
        cyto_roi = cell_roi & ~nuc_roi

        if cyto_roi.sum() < 100:
            continue
        if cm_sub[cyto_roi].sum() < 50:
            continue

        pcc = pearson_in_roi(c1_sub, cm_sub, cyto_roi)

        c1_cyto = c1_sub * cyto_roi
        cm_cyto = cm_sub * cyto_roi

        if SKIP_COSTES:
            # After local median subtraction, pixels > 0 are signal by definition.
            # Use threshold = 0 — appropriate when channels are not proportionally
            # correlated (flat scatterplots), where Costes finds pathologically
            # high thresholds. Per Dunn et al. 2011.
            thr_lamp1  = 0.0
            thr_marker = 0.0
        else:
            thr_lamp1, thr_marker = costes_threshold(c1_cyto, cm_cyto)

        M1, M2 = manders_coefficients(c1_sub, cm_sub,
                                       thr_lamp1, thr_marker, cyto_roi)
        if np.isnan(M2):
            continue

        rows.append({
            'label'              : label,
            'collection'         : collection_id,
            'cell_id'            : cid,
            'projection'         : PROJECTION,
            'skip_costes'        : SKIP_COSTES,
            'pearson_r'          : pcc,
            'costes_thr_lamp1'   : thr_lamp1,
            'costes_thr_marker'  : thr_marker,
            'M1_lamp1_on_marker' : M1,
            'M2_marker_on_lamp1' : M2,
            'cyto_px'            : int(cyto_roi.sum()),
            'median_window'      : median_window,
            'residual_factor'    : residual_factor,
        })

    print(f"done — {len(rows)} cells", flush=True)
    return rows


# ── File discovery — manifest or auto-detect ──────────────────────────────────
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

if args.manifest:
    # ── Load from manifest ────────────────────────────────────────────────────
    print(f"Loading manifest: {args.manifest}")
    mdf = pd.read_csv(args.manifest, sep='\t')
    required = ['label','collection','ch_dapi','ch_lamp1','ch_marker']
    missing_cols = [c for c in required if c not in mdf.columns]
    if missing_cols:
        print(f"❌ Manifest missing columns: {missing_cols}")
        exit(1)
    for _, row in mdf.iterrows():
        groups[row['label']].append((
            row['ch_dapi'],
            row['ch_lamp1'],
            row['ch_marker'],
            row['collection'],
        ))
    if 'cell_line' in mdf.columns:
        CELL_LINE   = mdf['cell_line'].iloc[0]
    if 'marker_name' in mdf.columns:
        MARKER_NAME = mdf['marker_name'].iloc[0]
    print(f"  {len(mdf)} collection(s) across {len(groups)} condition(s)")

elif DATASET == 'subdir':
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
cond_summary = {}  # label -> {'M1': [], 'M2': [], 'r': [], 'pixels': (x,y)}

PALETTE = ['#D4845A','#4A9BAF','#7B9E6B','#A07BC4','#C4A07B','#6B9EA0','#C47B7B','#7B7BC4']

for idx, label in enumerate(label_list):
    triplets = groups[label]
    print(f"\n══ {label} ({len(triplets)} field(s)) ══", flush=True)

    cond_M1, cond_M2, cond_r = [], [], []
    cond_px_lamp1, cond_px_marker = [], []   # pooled signal pixels for scatterplot

    for c0_path, c1_path, cm_path, collection_id in triplets:
        print(f"  Collection: {collection_id}", flush=True)
        print(f"    [1/3] Loading...", flush=True)
        c0 = load_plane(c0_path)
        c1 = load_plane(c1_path)
        cm = load_plane(cm_path)

        rows = analyze_collection(c0, c1, cm, label, collection_id,
                                   residual_factor=RESIDUAL_FACTOR)
        all_rows.extend(rows)

        for r in rows:
            cond_M1.append(r['M1_lamp1_on_marker'])
            cond_M2.append(r['M2_marker_on_lamp1'])
            cond_r.append(r['pearson_r'])

        # Collect background-subtracted pixel values for scatterplot
        # Re-use the subtracted images from this collection
        from skimage.filters import rank as skrank
        from skimage.morphology import disk as skdisk
        c1_sub  = local_median_subtract(c1,  window=32, residual_factor=RESIDUAL_FACTOR)
        cm_sub  = local_median_subtract(cm,  window=32, residual_factor=RESIDUAL_FACTOR)
        nuc_lbl, cell_lbl = segment_nuclei(c0)

        for cid in range(1, cell_lbl.max() + 1):
            cyto = (cell_lbl == cid) & ~(nuc_lbl == cid)
            if cyto.sum() < 100:
                continue
            x_pix = c1_sub[cyto].flatten()
            y_pix = cm_sub[cyto].flatten()
            # Keep only signal pixels (at least one channel > 0)
            sig = (x_pix > 0) | (y_pix > 0)
            # Subsample to keep memory manageable (max 5000 px per cell)
            idx_sig = np.where(sig)[0]
            if len(idx_sig) > 5000:
                idx_sig = np.random.choice(idx_sig, 5000, replace=False)
            cond_px_lamp1.extend(x_pix[idx_sig].tolist())
            cond_px_marker.extend(y_pix[idx_sig].tolist())

    cond_summary[label] = {
        'M1'    : cond_M1,
        'M2'    : cond_M2,
        'r'     : cond_r,
        'color' : PALETTE[idx % len(PALETTE)],
        'px_lamp1'  : np.array(cond_px_lamp1,  dtype=np.float32),
        'px_marker' : np.array(cond_px_marker, dtype=np.float32),
    }

    if cond_M2:
        print(f"\n  ► {label}")
        print(f"    M2 (Raptor on lysosome): mean={np.mean(cond_M2):.3f}  "
              f"median={np.median(cond_M2):.3f}  std={np.std(cond_M2):.3f}  n={len(cond_M2)}")
        print(f"    M1 (LAMP1 covered by Raptor): mean={np.mean(cond_M1):.3f}")
        print(f"    Pearson r: mean={np.mean([v for v in cond_r if not np.isnan(v)]):.3f}",
              flush=True)

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

# ── Scatterplot figure: LAMP1 vs Raptor per condition ────────────────────────
# Per Dunn et al.: inspect scatterplots to verify linearity before trusting PCC
n_cond  = len(label_list)
n_cols  = min(n_cond, 3)
n_rows  = (n_cond + n_cols - 1) // n_cols

fig_sc, axes_sc = plt.subplots(n_rows, n_cols,
                                figsize=(6 * n_cols, 5 * n_rows),
                                facecolor='#F8F7F4')
axes_sc = np.array(axes_sc).flatten()
fig_sc.suptitle(f'{CELL_LINE} — LAMP1 vs {MARKER_NAME} pixel intensity scatterplots\n'
                f'(background-subtracted, cytoplasm only, nucleus excluded, signal pixels only)',
                color='#1E2D3A', fontsize=12, fontweight='bold')

for ax_idx, label in enumerate(label_list):
    ax  = axes_sc[ax_idx]
    ax.set_facecolor('#F0F4F6')
    p   = cond_summary[label]
    col = p['color']

    x = p['px_lamp1']
    y = p['px_marker']

    if len(x) < 10:
        ax.set_title(label, color='#1E2D3A', fontsize=8)
        ax.axis('off')
        continue

    # Subsample for plotting (max 8000 points total)
    n_plot = min(len(x), 8000)
    idx_plot = np.random.choice(len(x), n_plot, replace=False)
    xp, yp = x[idx_plot], y[idx_plot]

    # Scatter
    ax.scatter(xp, yp, c=col, alpha=0.25, s=3, linewidths=0, rasterized=True)

    # Clip to 99th percentile for axis limits
    xlim = np.percentile(x[x > 0], 99) if (x > 0).any() else 1
    ylim = np.percentile(y[y > 0], 99) if (y > 0).any() else 1
    ax.set_xlim(0, xlim * 1.05)
    ax.set_ylim(0, ylim * 1.05)

    # Regression line on signal pixels
    sig = (x > 0) | (y > 0)
    if sig.sum() > 20 and x[sig].std() > 0:
        a_reg = np.cov(x[sig], y[sig])[0,1] / (np.var(x[sig]) + 1e-9)
        b_reg = y[sig].mean() - a_reg * x[sig].mean()
        x_line = np.array([0, xlim])
        ax.plot(x_line, a_reg * x_line + b_reg,
                color='#1E2D3A', linewidth=1.5, linestyle='--',
                label=f'fit: y={a_reg:.2f}x+{b_reg:.1f}')

    # Mean Pearson r
    r_vals = [v for v in p['r'] if not np.isnan(v)]
    r_mean = np.mean(r_vals) if r_vals else np.nan
    ax.text(0.97, 0.97,
            f'r = {r_mean:.3f}\nn = {len(p["M2"])} cells\n{n_plot} px shown',
            transform=ax.transAxes, color='#1E2D3A', fontsize=8,
            ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      alpha=0.8, linewidth=0))

    ax.set_xlabel('LAMP1 intensity (background subtracted)', color='#1E2D3A', fontsize=8)
    ax.set_ylabel(f'{MARKER_NAME} intensity (background subtracted)', color='#1E2D3A', fontsize=8)
    ax.set_title(label.replace('_', ' '), color='#1E2D3A', fontsize=9, fontweight='bold')
    ax.tick_params(colors='#1E2D3A', labelsize=7)
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
    for sp in ['bottom', 'left']: ax.spines[sp].set_color('#A0B8C4')
    if sig.sum() > 20:
        ax.legend(fontsize=7, facecolor='white', labelcolor='#1E2D3A')

# Hide unused axes
for ax_idx in range(n_cond, len(axes_sc)):
    axes_sc[ax_idx].set_visible(False)

plt.tight_layout()
scatter_path = os.path.join(OUTPUT_DIR, 'scatterplots.png')
plt.savefig(scatter_path, dpi=130, bbox_inches='tight', facecolor='#F8F7F4')
plt.close()
print(f"Scatterplots: {scatter_path}")


fig, axes = plt.subplots(1, 3, figsize=(20, 7), facecolor='#F8F7F4')
fig.suptitle(f'{CELL_LINE} — Manders Colocalization: {MARKER_NAME} ↔ LAMP1',
             color='#1E2D3A', fontsize=14, fontweight='bold')

metrics = [
    ('M2', 'M2: Fraction of Raptor on Lysosome\n(Manders — primary readout)', 'M2_marker_on_lamp1'),
    ('M1', 'M1: Fraction of LAMP1 covered by Raptor\n(Manders)', 'M1_lamp1_on_marker'),
    ('r',  'Pearson r: Channel correlation\n(cytoplasm only, nucleus excluded)', 'pearson_r'),
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