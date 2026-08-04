"""
Residual Factor Sweep — Diagnostic Tool
Rogala Lab

Runs local median background subtraction at multiple residual_factor values
on a single image and shows:
  1. How many pixels survive (are > 0) at each level
  2. What the background-subtracted image looks like visually
  3. The signal-to-noise characteristics

Use this to pick an appropriate residual_factor before running manders_coloc.py.

Usage:
  python residual_factor_sweep.py \
    --lamp1  "/path/to/C1.tif" \
    --marker "/path/to/C2.tif" \
    --dapi   "/path/to/C0.tif" \
    --output ~/Desktop/residual_sweep
"""

import os, argparse, warnings
import numpy as np
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.filters import rank
from skimage.morphology import disk
from skimage import filters, morphology, measure, segmentation, feature
from skimage.filters import gaussian
from scipy import ndimage as ndi

warnings.filterwarnings('ignore')

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Residual factor sweep diagnostic')
parser.add_argument('--lamp1',  type=str, required=True, help='Path to LAMP1 channel TIFF (C1)')
parser.add_argument('--marker', type=str, required=True, help='Path to Raptor/mTOR channel TIFF (C2/C3)')
parser.add_argument('--dapi',   type=str, required=True, help='Path to DAPI channel TIFF (C0)')
parser.add_argument('--output', type=str, default='./residual_sweep', help='Output folder')
parser.add_argument('--window', type=int, default=32, help='Local median window size (default 32)')
parser.add_argument('--projection', type=str, default='max',
                    choices=['max','best_z','mean'],
                    help='Z-stack projection method (default: max)')
args = parser.parse_args()

os.makedirs(args.output, exist_ok=True)

FACTORS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]

# ── Helpers ───────────────────────────────────────────────────────────────────
def best_z(stack):
    return int(np.argmax([np.var(stack[z]) for z in range(stack.shape[0])]))

def load(path, projection='max'):
    print(f"  Loading {os.path.basename(path)}...", end=' ', flush=True)
    stack = tifffile.imread(path).astype(np.float32)
    if stack.ndim == 2:
        print(f"done {stack.shape}", flush=True)
        return stack
    n_z = stack.shape[0]
    if projection == 'max':
        img = stack.max(axis=0)
    elif projection == 'best_z':
        img = stack[best_z(stack)]
    else:
        img = stack.mean(axis=0)
    print(f"done {img.shape} ({projection}, {n_z} planes)", flush=True)
    return img

def norm(img, plow=1, phigh=99.5):
    lo, hi = np.percentile(img, plow), np.percentile(img, phigh)
    return np.clip((img - lo) / (hi - lo + 1e-6), 0, 1)

def local_median_subtract(img, window=32, residual_factor=2.0):
    img_norm = img - img.min()
    scale    = 65535.0 / (img_norm.max() + 1e-6)
    img_u16  = (img_norm * scale).astype(np.uint16)
    radius   = window // 2
    local_bg = rank.median(img_u16, disk(radius)).astype(np.float32) / scale
    subtracted = img - local_bg
    near_zero  = subtracted[subtracted < np.percentile(subtracted, 50)]
    noise_std  = near_zero.std() if len(near_zero) > 10 else 1.0
    subtracted = subtracted - residual_factor * noise_std
    return np.clip(subtracted, 0, None), noise_std, local_bg

def segment_nuclei_simple(dapi):
    smooth = gaussian(dapi, sigma=3)
    thresh = filters.threshold_otsu(smooth)
    mask   = smooth > thresh
    mask   = morphology.remove_small_objects(mask, min_size=2000)
    mask   = ndi.binary_fill_holes(mask)
    mask   = morphology.binary_erosion(mask, morphology.disk(4))
    dist   = ndi.distance_transform_edt(mask)
    peaks  = feature.peak_local_max(dist, min_distance=40, labels=mask)
    pm     = np.zeros(dist.shape, dtype=bool)
    pm[tuple(peaks.T)] = True
    markers     = measure.label(pm)
    nuc_labels  = segmentation.watershed(-dist, markers, mask=mask)
    cell_mask   = morphology.dilation(mask, morphology.disk(30))
    cell_labels = segmentation.watershed(-dist, markers, mask=cell_mask)
    return nuc_labels, cell_labels

# ── Load images ───────────────────────────────────────────────────────────────
print("Loading images...")
c0 = load(args.dapi,   args.projection)
c1 = load(args.lamp1,  args.projection)
cm = load(args.marker, args.projection)

# Crop to center 1024x1024 for speed
H, W = c0.shape
S = 512
sl = np.s_[H//2-S:H//2+S, W//2-S:W//2+S]
c0c, c1c, cmc = c0[sl], c1[sl], cm[sl]

print("Segmenting nuclei...", end=' ', flush=True)
nuc_lbl, cell_lbl = segment_nuclei_simple(c0c)
n_total_cells = nuc_lbl.max()
print(f"done ({n_total_cells} nuclei in crop)")

# Compute background once
_, noise_std_c1, bg_c1 = local_median_subtract(c1c, window=args.window, residual_factor=0.0)
_, noise_std_cm, bg_cm = local_median_subtract(cmc, window=args.window, residual_factor=0.0)
print(f"\nNoise std — LAMP1: {noise_std_c1:.2f}  |  Marker: {noise_std_cm:.2f}")

# ── Sweep ─────────────────────────────────────────────────────────────────────
print(f"\nRunning sweep across residual_factor = {FACTORS}...")

results = []

for rf in FACTORS:
    c1_sub, _, _ = local_median_subtract(c1c, window=args.window, residual_factor=rf)
    cm_sub, _, _ = local_median_subtract(cmc, window=args.window, residual_factor=rf)

    # Count cells that pass the signal filter (cm_sub sum > 50 in cytoplasm)
    cells_passing = 0
    total_c1_pct  = []
    total_cm_pct  = []

    for cid in range(1, cell_lbl.max() + 1):
        cell_roi = (cell_lbl == cid)
        nuc_roi  = (nuc_lbl  == cid)
        cyto_roi = cell_roi & ~nuc_roi

        if cyto_roi.sum() < 100:
            continue

        cm_cyto = cm_sub[cyto_roi]
        c1_cyto = c1_sub[cyto_roi]

        if cm_cyto.sum() >= 50:
            cells_passing += 1

        # Fraction of cytoplasm pixels with signal > 0
        total_cm_pct.append((cm_cyto > 0).mean() * 100)
        total_c1_pct.append((c1_cyto > 0).mean() * 100)

    # Pixel-level stats
    c1_nonzero_pct = (c1_sub > 0).mean() * 100
    cm_nonzero_pct = (cm_sub > 0).mean() * 100

    results.append({
        'rf'            : rf,
        'c1_nonzero_pct': c1_nonzero_pct,
        'cm_nonzero_pct': cm_nonzero_pct,
        'cells_passing' : cells_passing,
        'c1_sub'        : c1_sub,
        'cm_sub'        : cm_sub,
        'mean_c1_cyto_pct': np.mean(total_c1_pct) if total_c1_pct else 0,
        'mean_cm_cyto_pct': np.mean(total_cm_pct) if total_cm_pct else 0,
    })

    print(f"  rf={rf:.1f}  LAMP1 signal px: {c1_nonzero_pct:.1f}%  "
          f"Marker signal px: {cm_nonzero_pct:.1f}%  "
          f"Cells passing filter: {cells_passing}/{n_total_cells}")

# ── Figure 1: Visual comparison ───────────────────────────────────────────────
print("\nGenerating figures...")

fig, axes = plt.subplots(3, len(FACTORS), figsize=(4*len(FACTORS), 10),
                          facecolor='#F8F7F4')
fig.suptitle(f'Residual Factor Sweep — window={args.window}px\n'
             f'LAMP1 noise std={noise_std_c1:.1f}  |  Marker noise std={noise_std_cm:.1f}',
             color='#1E2D3A', fontsize=12, fontweight='bold')

for col, res in enumerate(results):
    rf = res['rf']

    # Row 0: LAMP1 subtracted
    axes[0, col].imshow(norm(res['c1_sub']), cmap='Greens', vmin=0, vmax=1)
    axes[0, col].set_title(f'rf={rf}', color='#1E2D3A', fontsize=10, fontweight='bold')
    axes[0, col].set_ylabel('LAMP1' if col == 0 else '', color='#1E2D3A', fontsize=9)
    axes[0, col].text(0.02, 0.02, f'{res["c1_nonzero_pct"]:.1f}% px > 0',
                      transform=axes[0,col].transAxes, color='white',
                      fontsize=8, fontweight='bold',
                      bbox=dict(facecolor='#2C7A8C', alpha=0.8, linewidth=0, pad=2))
    axes[0, col].axis('off')

    # Row 1: Marker subtracted
    axes[1, col].imshow(norm(res['cm_sub']), cmap='Reds', vmin=0, vmax=1)
    axes[1, col].set_ylabel('Marker' if col == 0 else '', color='#1E2D3A', fontsize=9)
    axes[1, col].text(0.02, 0.02, f'{res["cm_nonzero_pct"]:.1f}% px > 0',
                      transform=axes[1,col].transAxes, color='white',
                      fontsize=8, fontweight='bold',
                      bbox=dict(facecolor='#C04020', alpha=0.8, linewidth=0, pad=2))
    axes[1, col].axis('off')

    # Row 2: Overlay
    rgb = np.zeros((*c1c.shape, 3))
    rgb[:,:,1] = norm(res['c1_sub']) * 0.9
    rgb[:,:,0] = norm(res['cm_sub']) * 0.9
    overlap = (res['c1_sub'] > 0) & (res['cm_sub'] > 0)
    rgb[overlap, 0] = 1; rgb[overlap, 1] = 1; rgb[overlap, 2] = 0
    axes[2, col].imshow(np.clip(rgb, 0, 1))
    axes[2, col].set_ylabel('Overlay' if col == 0 else '', color='#1E2D3A', fontsize=9)
    axes[2, col].text(0.02, 0.02, f'{res["cells_passing"]}/{n_total_cells} cells',
                      transform=axes[2,col].transAxes, color='white',
                      fontsize=8, fontweight='bold',
                      bbox=dict(facecolor='#1E2D3A', alpha=0.8, linewidth=0, pad=2))
    axes[2, col].axis('off')

for ax in axes.flat:
    ax.set_facecolor('#F8F7F4')

plt.tight_layout()
fig1_path = os.path.join(args.output, 'sweep_images.png')
plt.savefig(fig1_path, dpi=120, bbox_inches='tight', facecolor='#F8F7F4')
plt.close()
print(f"  Image sweep: {fig1_path}")

# ── Figure 2: Summary stats ───────────────────────────────────────────────────
fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5), facecolor='#F8F7F4')
fig2.suptitle('Residual Factor — Effect on Signal Retention',
              color='#1E2D3A', fontsize=13, fontweight='bold')

rfs      = [r['rf'] for r in results]
c1_pcts  = [r['c1_nonzero_pct']  for r in results]
cm_pcts  = [r['cm_nonzero_pct']  for r in results]
n_cells  = [r['cells_passing']   for r in results]
c1_cyto  = [r['mean_c1_cyto_pct'] for r in results]
cm_cyto  = [r['mean_cm_cyto_pct'] for r in results]

# Panel 1: % pixels with signal
ax = axes2[0]; ax.set_facecolor('#F0F4F6')
ax.plot(rfs, c1_pcts, 'o-', color='#2C7A8C', linewidth=2, markersize=8, label='LAMP1')
ax.plot(rfs, cm_pcts, 's-', color='#D4845A', linewidth=2, markersize=8, label='Marker')
ax.set_xlabel('residual_factor', color='#1E2D3A')
ax.set_ylabel('% pixels with signal > 0\n(whole crop)', color='#1E2D3A')
ax.set_title('Signal pixel retention\n(whole image)', color='#1E2D3A', fontweight='bold')
ax.legend(fontsize=9, facecolor='white')
ax.tick_params(colors='#1E2D3A')
for sp in ['top','right']: ax.spines[sp].set_visible(False)
for sp in ['bottom','left']: ax.spines[sp].set_color('#A0B8C4')

# Panel 2: % pixels with signal in cytoplasm per cell
ax = axes2[1]; ax.set_facecolor('#F0F4F6')
ax.plot(rfs, c1_cyto, 'o-', color='#2C7A8C', linewidth=2, markersize=8, label='LAMP1')
ax.plot(rfs, cm_cyto, 's-', color='#D4845A', linewidth=2, markersize=8, label='Marker')
ax.set_xlabel('residual_factor', color='#1E2D3A')
ax.set_ylabel('Mean % cytoplasm pixels\nwith signal > 0 per cell', color='#1E2D3A')
ax.set_title('Signal retention\n(cytoplasm per cell)', color='#1E2D3A', fontweight='bold')
ax.legend(fontsize=9, facecolor='white')
ax.tick_params(colors='#1E2D3A')
for sp in ['top','right']: ax.spines[sp].set_visible(False)
for sp in ['bottom','left']: ax.spines[sp].set_color('#A0B8C4')

# Panel 3: cells passing signal filter
ax = axes2[2]; ax.set_facecolor('#F0F4F6')
ax.bar(rfs, n_cells, color='#4A9BAF', width=0.3, edgecolor='none')
ax.axhline(n_total_cells, color='#D4845A', linestyle='--', linewidth=1.5,
           label=f'Total nuclei ({n_total_cells})')
ax.set_xlabel('residual_factor', color='#1E2D3A')
ax.set_ylabel('Cells passing signal filter', color='#1E2D3A')
ax.set_title('Cells included in analysis\n(marker signal sum > 50)', color='#1E2D3A',
             fontweight='bold')
ax.legend(fontsize=9, facecolor='white')
ax.tick_params(colors='#1E2D3A')
for sp in ['top','right']: ax.spines[sp].set_visible(False)
for sp in ['bottom','left']: ax.spines[sp].set_color('#A0B8C4')
for xi, (rf, n) in enumerate(zip(rfs, n_cells)):
    ax.text(rf, n + 0.3, str(n), ha='center', va='bottom',
            color='#1E2D3A', fontsize=10, fontweight='bold')

for ax in axes2:
    ax.set_facecolor('#F0F4F6')

plt.tight_layout()
fig2_path = os.path.join(args.output, 'sweep_stats.png')
plt.savefig(fig2_path, dpi=130, bbox_inches='tight', facecolor='#F8F7F4')
plt.close()
print(f"  Stats plot:  {fig2_path}")
print(f"\nDone. Use these plots to choose your residual_factor.")
print(f"Look for the value where:")
print(f"  - The image panels still show distinct puncta (not noise)")
print(f"  - The % signal pixels drops to a sensible level (~5-30% for sparse puncta)")
print(f"  - Most cells pass the signal filter")