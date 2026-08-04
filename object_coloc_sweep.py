"""
Parameter Sweep for Object-Based Colocalization — Rogala Lab

Tests combinations of:
  residual_factor   — background subtraction aggressiveness
  puncta_percentile — intensity threshold for puncta detection
  proximity_px      — colocalization distance threshold

Runs on a SINGLE representative image (fast) and produces:
  sweep_heatmaps.png  — heatmaps of key metrics across parameter space
  sweep_panels.png    — visual panels showing detected puncta at each setting
  sweep_results.csv   — all numbers for further analysis

Use results to pick optimal parameters for object_coloc.py.

Strategy:
  1. Find residual_factor where puncta look clean (not noisy, not over-zeroed)
  2. Find puncta_percentile where discrete puncta are detected (not diffuse signal)
  3. Set proximity_px based on visual channel offset in your images

Usage:
  python object_coloc_sweep.py \\
    --dapi   "/path/to/C0.tif" \\
    --lamp1  "/path/to/C1.tif" \\
    --marker "/path/to/C2.tif" \\
    --output ~/Desktop/sweep \\
    --projection max

  # Narrow the sweep after first run:
  python object_coloc_sweep.py \\
    --dapi ... --lamp1 ... --marker ... \\
    --residual_factors 0.5 1.0 1.5 \\
    --percentiles 80 85 90 \\
    --proximities 2 3 5
"""

import os, argparse, warnings
import numpy as np
import pandas as pd
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage import filters, morphology, measure, segmentation, feature
from skimage.filters import gaussian, rank
from skimage.morphology import disk
from scipy import ndimage as ndi
from itertools import product

warnings.filterwarnings('ignore')

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Parameter sweep for object-based colocalization')
parser.add_argument('--dapi',   type=str, default=None, help='Path to DAPI channel TIFF (not needed if --manifest provided)')
parser.add_argument('--lamp1',  type=str, default=None, help='Path to LAMP1 channel TIFF (not needed if --manifest provided)')
parser.add_argument('--marker', type=str, default=None, help='Path to Raptor/mTOR channel TIFF (not needed if --manifest provided)')
parser.add_argument('--output', type=str, default='./param_sweep')
parser.add_argument('--manifest', type=str, default=None,
                    help='Path to manifest TSV from ingest.py. '
                         'If provided, uses first collection from manifest as the sweep image.')
parser.add_argument('--projection', type=str, default='max',
                    choices=['max','best_z','mean'])
parser.add_argument('--window', type=int, default=32,
                    help='Local median window (default 32)')
parser.add_argument('--min_puncta_px', type=int, default=5,
                    help='Minimum punctum size in pixels (default 5)')

# Parameter ranges to sweep
parser.add_argument('--residual_factors', type=float, nargs='+',
                    default=[0.0, 0.5, 1.0, 1.5, 2.0],
                    help='List of residual_factor values to test')
parser.add_argument('--lamp1_percentiles', type=float, nargs='+',
                    default=[75, 80, 85],
                    help='Percentile values to test for LAMP1 detection')
parser.add_argument('--marker_percentiles', type=float, nargs='+',
                    default=[85, 90, 93, 95],
                    help='Percentile values to test for Raptor/mTOR detection')
# Keep --percentiles for backwards compat (sets both)
parser.add_argument('--percentiles', type=float, nargs='+', default=None,
                    help='Set same percentile range for both channels (overrides individual)')
parser.add_argument('--proximities', type=int, nargs='+',
                    default=[1, 3, 5, 7],
                    help='List of proximity_px values to test')
args = parser.parse_args()

os.makedirs(args.output, exist_ok=True)

FACTORS     = args.residual_factors
PROXIMITIES = args.proximities
if args.percentiles is not None:
    LAMP1_PCTS  = args.percentiles
    MARKER_PCTS = args.percentiles
else:
    LAMP1_PCTS  = args.lamp1_percentiles
    MARKER_PCTS = args.marker_percentiles

# ── Core functions (same as object_coloc.py) ──────────────────────────────────
def norm(img, plow=1, phigh=99.5):
    lo, hi = np.percentile(img, plow), np.percentile(img, phigh)
    return np.clip((img - lo) / (hi - lo + 1e-6), 0, 1)

def best_z(stack):
    return int(np.argmax([np.var(stack[z]) for z in range(stack.shape[0])]))

def load_plane(path, projection='max'):
    print(f"  Loading {os.path.basename(path)}...", end=' ', flush=True)
    stack = tifffile.imread(path).astype(np.float32)
    if stack.ndim == 2:
        print(f"done {stack.shape}", flush=True)
        return stack
    if projection == 'max':
        img = stack.max(axis=0)
    elif projection == 'best_z':
        img = stack[best_z(stack)]
    else:
        img = stack.mean(axis=0)
    print(f"done {img.shape} ({projection}, {stack.shape[0]}z)", flush=True)
    return img

def local_median_subtract(img, window=32, residual_factor=1.0):
    img_norm = img - img.min()
    scale    = 65535.0 / (img_norm.max() + 1e-6)
    img_u16  = (img_norm * scale).astype(np.uint16)
    local_bg = rank.median(img_u16, disk(window // 2)).astype(np.float32) / scale
    sub      = img - local_bg
    near_zero = sub[sub < np.percentile(sub, 50)]
    noise_std = near_zero.std() if len(near_zero) > 10 else 1.0
    sub       = sub - residual_factor * noise_std
    return np.clip(sub, 0, None)

def segment_nuclei(dapi):
    smooth      = gaussian(dapi, sigma=3)
    thresh      = filters.threshold_otsu(smooth)
    mask        = smooth > thresh
    mask        = morphology.remove_small_objects(mask, min_size=2000)
    mask        = ndi.binary_fill_holes(mask)
    mask        = morphology.binary_erosion(mask, morphology.disk(4))
    dist        = ndi.distance_transform_edt(mask)
    peaks       = feature.peak_local_max(dist, min_distance=40, labels=mask)
    pm          = np.zeros(dist.shape, dtype=bool)
    pm[tuple(peaks.T)] = True
    markers     = measure.label(pm)
    nuc_labels  = segmentation.watershed(-dist, markers, mask=mask)
    cell_mask   = morphology.dilation(mask, morphology.disk(30))
    cell_labels = segmentation.watershed(-dist, markers, mask=cell_mask)
    return nuc_labels, cell_labels

def detect_puncta(channel_sub, cyto_roi, min_px=5, percentile=85):
    ch = channel_sub * cyto_roi
    nonzero = ch[ch > 0]
    if len(nonzero) < 10:
        return np.zeros_like(ch, dtype=int), []
    thr  = np.percentile(nonzero, percentile)
    mask = ch > thr
    mask = morphology.remove_small_objects(mask, min_size=min_px)
    mask = morphology.remove_small_holes(mask, area_threshold=10)
    labeled = measure.label(mask)
    props   = measure.regionprops(labeled, intensity_image=ch)
    return labeled, props

def object_coloc(lamp1_labeled, marker_labeled, marker_props, proximity_px=3):
    lamp1_dilated = morphology.dilation(lamp1_labeled > 0, disk(proximity_px))
    n_coloc = 0
    coloc_mask = np.zeros(lamp1_labeled.shape, dtype=bool)
    for prop in marker_props:
        cy, cx = int(round(prop.centroid[0])), int(round(prop.centroid[1]))
        if 0 <= cy < lamp1_dilated.shape[0] and 0 <= cx < lamp1_dilated.shape[1]:
            if lamp1_dilated[cy, cx]:
                n_coloc += 1
                coloc_mask[marker_labeled == prop.label] = True
    return coloc_mask, n_coloc

# ── Load and prep ─────────────────────────────────────────────────────────────
# ── Load images ───────────────────────────────────────────────────────────────
print("Loading images...")
if args.manifest:
    import pandas as _pd
    mdf = _pd.read_csv(args.manifest, sep='\t')
    # Use a representative row — pick from positive control if possible
    # otherwise just use first row
    rep = mdf.iloc[0]
    dapi_path   = rep['ch_dapi']
    lamp1_path  = rep['ch_lamp1']
    marker_path = rep['ch_marker']
    print(f"Using manifest row: {rep['label']} / {rep['collection']}")
    print(f"  DAPI:   {os.path.basename(dapi_path)}")
    print(f"  LAMP1:  {os.path.basename(lamp1_path)}")
    print(f"  Marker: {os.path.basename(marker_path)}")
    c0 = load_plane(dapi_path,   args.projection)
    c1 = load_plane(lamp1_path,  args.projection)
    cm = load_plane(marker_path, args.projection)
else:
    c0 = load_plane(args.dapi,   args.projection)
    c1 = load_plane(args.lamp1,  args.projection)
    cm = load_plane(args.marker, args.projection)

# Crop center for speed
H, W = c0.shape
S = 512
sl = np.s_[H//2-S:H//2+S, W//2-S:W//2+S]
c0c, c1c, cmc = c0[sl], c1[sl], cm[sl]

print("Segmenting nuclei...", end=' ', flush=True)
nuc_lbl, cell_lbl = segment_nuclei(c0c)
n_cells = nuc_lbl.max()
print(f"done ({n_cells} nuclei in crop)")

# Build cyto ROIs once
cyto_rois = {}
for cid in range(1, cell_lbl.max() + 1):
    cyto = (cell_lbl == cid) & ~(nuc_lbl == cid)
    if cyto.sum() >= 100:
        cyto_rois[cid] = cyto

print(f"Valid cells: {len(cyto_rois)}\n")

# ── SWEEP PHASE 1: residual_factor × lamp1_pct × marker_pct ─────────────────
MID_PROXIMITY = PROXIMITIES[len(PROXIMITIES)//2]
print(f"Phase 1: sweeping rf × LAMP1_pct × Marker_pct (proximity={MID_PROXIMITY}px fixed)")
print(f"  LAMP1 percentiles:  {LAMP1_PCTS}")
print(f"  Marker percentiles: {MARKER_PCTS}")

phase1_results = []

for rf, l_pct, m_pct in product(FACTORS, LAMP1_PCTS, MARKER_PCTS):
    c1_sub = local_median_subtract(c1c, window=args.window, residual_factor=rf)
    cm_sub = local_median_subtract(cmc, window=args.window, residual_factor=rf)

    all_n_lamp1, all_n_marker, all_fracs = [], [], []

    for cid, cyto_roi in cyto_rois.items():
        l_lab, l_props = detect_puncta(c1_sub, cyto_roi, args.min_puncta_px, l_pct)
        m_lab, m_props = detect_puncta(cm_sub, cyto_roi, args.min_puncta_px, m_pct)

        if not m_props:
            continue

        _, n_coloc = object_coloc(l_lab, m_lab, m_props, MID_PROXIMITY)
        frac = n_coloc / len(m_props)

        all_n_lamp1.append(len(l_props))
        all_n_marker.append(len(m_props))
        all_fracs.append(frac)

    phase1_results.append({
        'residual_factor'  : rf,
        'lamp1_percentile' : l_pct,
        'marker_percentile': m_pct,
        'proximity_px'     : MID_PROXIMITY,
        'n_cells'          : len(all_fracs),
        'mean_n_lamp1'     : np.mean(all_n_lamp1) if all_n_lamp1 else 0,
        'mean_n_marker'    : np.mean(all_n_marker) if all_n_marker else 0,
        'mean_coloc_frac'  : np.mean(all_fracs) if all_fracs else np.nan,
        'std_coloc_frac'   : np.std(all_fracs) if all_fracs else np.nan,
    })
    print(f"  rf={rf:.1f} l_pct={l_pct:.0f} m_pct={m_pct:.0f} → "
          f"lamp1/cell={np.mean(all_n_lamp1):.0f} "
          f"marker/cell={np.mean(all_n_marker):.0f} "
          f"coloc={np.mean(all_fracs)*100:.1f}%" if all_fracs else
          f"  rf={rf:.1f} l_pct={l_pct:.0f} m_pct={m_pct:.0f} → no cells", flush=True)

df1 = pd.DataFrame(phase1_results)

# ── SWEEP PHASE 2: proximity_px (best combo from phase 1) ────────────────────
valid = df1[df1['n_cells'] > 0].copy()
valid['score'] = valid['n_cells'] - abs(valid['mean_n_marker'] - 30) * 0.1
best_row  = valid.loc[valid['score'].idxmax()]
BEST_RF   = best_row['residual_factor']
BEST_LPCT = best_row['lamp1_percentile']
BEST_MPCT = best_row['marker_percentile']

print(f"\nPhase 2: sweeping proximity_px (best rf={BEST_RF}, lamp1_pct={BEST_LPCT}, marker_pct={BEST_MPCT})")

c1_sub_best = local_median_subtract(c1c, window=args.window, residual_factor=BEST_RF)
cm_sub_best = local_median_subtract(cmc, window=args.window, residual_factor=BEST_RF)

phase2_results = []
for prox in PROXIMITIES:
    all_fracs = []
    for cid, cyto_roi in cyto_rois.items():
        l_lab, l_props = detect_puncta(c1_sub_best, cyto_roi, args.min_puncta_px, BEST_LPCT)
        m_lab, m_props = detect_puncta(cm_sub_best, cyto_roi, args.min_puncta_px, BEST_MPCT)
        if not m_props: continue
        _, n_coloc = object_coloc(l_lab, m_lab, m_props, prox)
        all_fracs.append(n_coloc / len(m_props))

    phase2_results.append({
        'proximity_px'   : prox,
        'mean_coloc_frac': np.mean(all_fracs) if all_fracs else np.nan,
        'std_coloc_frac' : np.std(all_fracs) if all_fracs else np.nan,
        'n_cells'        : len(all_fracs),
    })
    print(f"  proximity={prox}px → coloc={np.mean(all_fracs)*100:.1f}%  "
          f"(std={np.std(all_fracs)*100:.1f}%)" if all_fracs else
          f"  proximity={prox}px → no cells", flush=True)

df2 = pd.DataFrame(phase2_results)

# ── Save CSVs ─────────────────────────────────────────────────────────────────
df1.to_csv(os.path.join(args.output, 'phase1_rf_percentile.csv'), index=False)
df2.to_csv(os.path.join(args.output, 'phase2_proximity.csv'), index=False)

# ── Figure 1: Heatmaps — fix rf, show lamp1_pct vs marker_pct ────────────────
print("\nGenerating figures...")

# For heatmap: fix rf at best value, show LAMP1 pct (rows) vs Marker pct (cols)
df1_best_rf = df1[df1['residual_factor'] == BEST_RF]
l_pcts = sorted(df1['lamp1_percentile'].unique())
m_pcts = sorted(df1['marker_percentile'].unique())

def make_grid_lm(df, col):
    grid = np.full((len(l_pcts), len(m_pcts)), np.nan)
    for _, row in df.iterrows():
        if row['residual_factor'] != BEST_RF:
            continue
        li = l_pcts.index(row['lamp1_percentile'])
        mi = m_pcts.index(row['marker_percentile'])
        grid[li, mi] = row[col]
    return grid

fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor='#F8F7F4')
fig.suptitle(f'Parameter Sweep Phase 1: LAMP1_pct × Marker_pct\n'
             f'(rf={BEST_RF} fixed, proximity_px={MID_PROXIMITY} fixed, crop {S*2}×{S*2}px)',
             color='#1E2D3A', fontsize=13, fontweight='bold')

metrics = [
    ('mean_coloc_frac',  '% Raptor puncta on LAMP1\n(primary metric)', 'RdYlGn', True),
    ('mean_n_marker',    'Mean Raptor puncta per cell\n(want ~20-50)', 'Blues', False),
    ('mean_n_lamp1',     'Mean LAMP1 puncta per cell\n(more = better detection)', 'Greens', False),
    ('n_cells',          'Cells with data\n(more = better)', 'Purples', False),
]

for ax, (col, title, cmap, pct_scale) in zip(axes.flat, metrics):
    grid = make_grid_lm(df1, col)
    if pct_scale:
        grid = grid * 100
    im = ax.imshow(grid, cmap=cmap, aspect='auto',
                   vmin=np.nanmin(grid), vmax=np.nanmax(grid))
    ax.set_xticks(range(len(m_pcts)))
    ax.set_xticklabels([f'{p:.0f}' for p in m_pcts], color='#1E2D3A', fontsize=9)
    ax.set_yticks(range(len(l_pcts)))
    ax.set_yticklabels([f'{p:.0f}' for p in l_pcts], color='#1E2D3A', fontsize=9)
    ax.set_xlabel('marker_percentile (Raptor)', color='#1E2D3A', fontsize=10)
    ax.set_ylabel('lamp1_percentile (LAMP1)', color='#1E2D3A', fontsize=10)
    ax.set_title(title, color='#1E2D3A', fontsize=10, fontweight='bold')
    plt.colorbar(im, ax=ax)
    for li in range(len(l_pcts)):
        for mi in range(len(m_pcts)):
            val = grid[li, mi]
            if not np.isnan(val):
                ax.text(mi, li, f'{val:.1f}' if not pct_scale else f'{val:.0f}%',
                        ha='center', va='center', fontsize=8, color='#1E2D3A',
                        fontweight='bold')

best_li = l_pcts.index(BEST_LPCT)
best_mi = m_pcts.index(BEST_MPCT)
for ax in axes.flat:
    ax.add_patch(plt.Rectangle((best_mi-0.5, best_li-0.5), 1, 1,
                                fill=False, edgecolor='red', linewidth=2.5))

plt.tight_layout()
plt.savefig(os.path.join(args.output, 'phase1_heatmaps.png'),
            dpi=130, bbox_inches='tight', facecolor='#F8F7F4')
plt.close()

# ── Figure 2: Proximity sweep (phase 2) ──────────────────────────────────────
fig2, ax2 = plt.subplots(1, 1, figsize=(8, 5), facecolor='#F8F7F4')
ax2.set_facecolor('#F0F4F6')
proxs = df2['proximity_px'].tolist()
means = (df2['mean_coloc_frac'] * 100).tolist()
stds  = (df2['std_coloc_frac'] * 100).tolist()
ax2.errorbar(proxs, means, yerr=stds, fmt='o-', color='#4A9BAF',
             linewidth=2, markersize=8, capsize=5,
             ecolor='#2C7A8C', elinewidth=1.5)
for px, m in zip(proxs, means):
    ax2.text(px, m+1, f'{m:.1f}%', ha='center', va='bottom',
             color='#1E2D3A', fontsize=9, fontweight='bold')
ax2.set_xlabel('proximity_px (LAMP1 dilation radius)', color='#1E2D3A', fontsize=11)
ax2.set_ylabel('% Raptor puncta on LAMP1 (mean ± std)', color='#1E2D3A', fontsize=11)
ax2.set_title(f'Phase 2: Proximity threshold sweep\n'
              f'(rf={BEST_RF}, lamp1_pct={BEST_LPCT}, marker_pct={BEST_MPCT})',
              color='#1E2D3A', fontsize=12, fontweight='bold')
ax2.tick_params(colors='#1E2D3A')
for sp in ['top','right']: ax2.spines[sp].set_visible(False)
for sp in ['bottom','left']: ax2.spines[sp].set_color('#A0B8C4')
ax2.set_ylim(0, max(means)*1.3 if means else 100)
plt.tight_layout()
plt.savefig(os.path.join(args.output, 'phase2_proximity.png'),
            dpi=130, bbox_inches='tight', facecolor='#F8F7F4')
plt.close()

# ── Figure 3: Visual panels at best + bracketing combos ──────────────────────
test_combos = [
    (BEST_RF, BEST_LPCT, BEST_MPCT, MID_PROXIMITY, 'BEST'),
    (BEST_RF, max(LAMP1_PCTS[0], BEST_LPCT-5), BEST_MPCT, MID_PROXIMITY, 'lamp1_pct-5'),
    (BEST_RF, min(LAMP1_PCTS[-1], BEST_LPCT+5), BEST_MPCT, MID_PROXIMITY, 'lamp1_pct+5'),
    (BEST_RF, BEST_LPCT, max(MARKER_PCTS[0], BEST_MPCT-5), MID_PROXIMITY, 'marker_pct-5'),
    (BEST_RF, BEST_LPCT, min(MARKER_PCTS[-1], BEST_MPCT+5), MID_PROXIMITY, 'marker_pct+5'),
]

fig3, axes3 = plt.subplots(3, len(test_combos),
                            figsize=(5*len(test_combos), 12),
                            facecolor='#F8F7F4')
fig3.suptitle('Visual comparison around best parameters\n(Red box = recommended)',
              color='#1E2D3A', fontsize=12, fontweight='bold')

# Use one representative cell for visual
rep_cid = list(cyto_rois.keys())[len(cyto_rois)//2]
cyto_rep = cyto_rois[rep_cid]
ys, xs = np.where(cyto_rep)
y0,y1,x0,x1 = ys.min(), ys.max(), xs.min(), xs.max()
pad = 20
cell_sl = np.s_[max(0,y0-pad):min(H,y1+pad), max(0,x0-pad):min(W,x1+pad)]

for col, (rf, l_pct, m_pct, prox, label) in enumerate(test_combos):
    c1_s = local_median_subtract(c1c, window=args.window, residual_factor=rf)
    cm_s = local_median_subtract(cmc, window=args.window, residual_factor=rf)

    l_lab, l_props = detect_puncta(c1_s, cyto_rep, args.min_puncta_px, l_pct)
    m_lab, m_props = detect_puncta(cm_s, cyto_rep, args.min_puncta_px, m_pct)
    col_mask, n_coloc = object_coloc(l_lab, m_lab, m_props, prox)

    frac = n_coloc / len(m_props) * 100 if m_props else 0
    title = f'rf={rf} L={l_pct:.0f} M={m_pct:.0f}\n{label}'

    # LAMP1
    axes3[0, col].imshow(norm(c1_s[cell_sl]), cmap='Greens', vmin=0, vmax=1)
    ov = np.zeros((*c1_s.shape, 4))
    ov[l_lab>0, 1]=1; ov[l_lab>0, 2]=0.3; ov[l_lab>0, 3]=0.7
    axes3[0, col].imshow(ov[cell_sl])
    axes3[0, col].set_title(title, color='#1E2D3A', fontsize=9,
                             fontweight='bold' if label=='BEST' else 'normal')
    axes3[0, col].set_ylabel('LAMP1' if col==0 else '', color='#1E2D3A', fontsize=9)
    axes3[0, col].text(0.02, 0.02, f'{len(l_props)} puncta',
                       transform=axes3[0,col].transAxes, color='white', fontsize=8,
                       bbox=dict(facecolor='#2C7A8C', alpha=0.8, linewidth=0, pad=2))
    axes3[0, col].axis('off')

    # Marker
    axes3[1, col].imshow(norm(cm_s[cell_sl]), cmap='Reds', vmin=0, vmax=1)
    ov2 = np.zeros((*cm_s.shape, 4))
    ov2[m_lab>0, 0]=1; ov2[m_lab>0, 1]=0.2; ov2[m_lab>0, 3]=0.7
    axes3[1, col].imshow(ov2[cell_sl])
    axes3[1, col].set_ylabel('Marker' if col==0 else '', color='#1E2D3A', fontsize=9)
    axes3[1, col].text(0.02, 0.02, f'{len(m_props)} puncta',
                       transform=axes3[1,col].transAxes, color='white', fontsize=8,
                       bbox=dict(facecolor='#C04020', alpha=0.8, linewidth=0, pad=2))
    axes3[1, col].axis('off')

    # Colocalization
    rgb = np.zeros((*c1_s.shape, 3))
    rgb[:,:,1] = norm(c1_s)*0.6; rgb[:,:,0] = norm(cm_s)*0.6
    rgb[l_lab>0, 1]=0.8; rgb[l_lab>0, 0]=0
    rgb[m_lab>0, 0]=0.8; rgb[m_lab>0, 1]=0
    rgb[col_mask, 0]=1; rgb[col_mask, 1]=1; rgb[col_mask, 2]=0
    axes3[2, col].imshow(np.clip(rgb[cell_sl], 0, 1))
    axes3[2, col].set_ylabel('Overlay' if col==0 else '', color='#1E2D3A', fontsize=9)
    axes3[2, col].text(0.02, 0.02, f'{frac:.0f}% coloc',
                       transform=axes3[2,col].transAxes, color='white', fontsize=9,
                       fontweight='bold',
                       bbox=dict(facecolor='#1E2D3A', alpha=0.85, linewidth=0, pad=2))
    axes3[2, col].axis('off')

    # Red box on best
    if label == 'BEST':
        for row_ax in axes3[:, col]:
            for spine in row_ax.spines.values():
                spine.set_edgecolor('red')
                spine.set_linewidth(3)

plt.tight_layout()
plt.savefig(os.path.join(args.output, 'phase3_visual_comparison.png'),
            dpi=120, bbox_inches='tight', facecolor='#F8F7F4')
plt.close()

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"RECOMMENDED PARAMETERS (auto-selected):")
print(f"  --residual_factor    {BEST_RF}")
print(f"  --lamp1_percentile   {BEST_LPCT}")
print(f"  --marker_percentile  {BEST_MPCT}")
print(f"  --proximity_px       {MID_PROXIMITY}  (review phase2_proximity.png to confirm)")
print(f"{'='*60}")
print(f"\nOutputs in: {args.output}")
print(f"  phase1_heatmaps.png       — rf × percentile heatmaps")
print(f"  phase2_proximity.png      — proximity sweep")
print(f"  phase3_visual_comparison.png — puncta detection visuals")
print(f"  phase1_rf_percentile.csv  — full phase 1 numbers")
print(f"  phase2_proximity.csv      — full phase 2 numbers")