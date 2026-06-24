"""
mTOR–Lysosome Colocalization Analysis
Rogala Lab — MiaPaca2 2×2 Nutrient Perturbation Dataset

Usage:
    cd "Rogala Lab Images Olga"
    python analyze_coloc.py

Requirements:
    pip install tifffile scikit-image scipy matplotlib numpy pandas

Outputs (written to ./coloc_results/):
    coloc_summary.csv        — per-cell stats for all 4 conditions
    coloc_summary.png        — 2×2 comparison figure for the talk
    <condition>_panel.png    — individual segmentation panel per condition
"""

import os, re, warnings
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

warnings.filterwarnings('ignore')

# ── Config ────────────────────────────────────────────────────────────────────
IMAGE_DIR   = "."          # folder containing the tiffs (run script from there)
OUTPUT_DIR  = "coloc_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Segmentation parameters — tweak if needed
NUC_SIGMA        = 3       # Gaussian blur radius for DAPI
NUC_MIN_SIZE     = 2000    # min nucleus area in pixels
NUC_ERODE        = 4       # erosion to clean nucleus edges
NUC_EXPAND_PX    = 30      # cell boundary expansion beyond nucleus
NUC_PEAK_DIST    = 40      # min distance between nucleus centres (watershed)
PUNCTA_BG_SIGMA  = 20      # background blur for subtraction
PUNCTA_FG_SIGMA  = 1.5     # foreground smoothing
PUNCTA_PERCENTILE= 85      # intensity threshold percentile
PUNCTA_MIN_SIZE  = 8       # min punctum area in pixels

# ── Helpers ───────────────────────────────────────────────────────────────────
def norm(img, plow=1, phigh=99.5):
    lo, hi = np.percentile(img, plow), np.percentile(img, phigh)
    return np.clip((img - lo) / (hi - lo + 1e-6), 0, 1)

def best_z(stack):
    """Return the z-plane with highest variance (sharpest focus)."""
    return int(np.argmax([np.var(stack[z]) for z in range(stack.shape[0])]))

def segment_nuclei(dapi):
    """DAPI → nucleus labels + cell expansion labels."""
    smooth = gaussian(dapi, sigma=NUC_SIGMA)
    thresh = filters.threshold_otsu(smooth)
    mask   = smooth > thresh
    mask   = morphology.remove_small_objects(mask, min_size=NUC_MIN_SIZE)
    mask   = ndi.binary_fill_holes(mask)
    mask   = morphology.binary_erosion(mask, morphology.disk(NUC_ERODE))
    dist   = ndi.distance_transform_edt(mask)
    peaks  = feature.peak_local_max(dist, min_distance=NUC_PEAK_DIST, labels=mask)
    peak_mask = np.zeros(dist.shape, dtype=bool)
    peak_mask[tuple(peaks.T)] = True
    markers   = measure.label(peak_mask)
    nuc_labels = segmentation.watershed(-dist, markers, mask=mask)
    # expand for cell ROI
    cell_mask   = morphology.dilation(mask, morphology.disk(NUC_EXPAND_PX))
    cell_labels = segmentation.watershed(-dist, markers, mask=cell_mask)
    return nuc_labels, cell_labels, dist, markers

def detect_puncta(channel):
    """Background-subtracted puncta detection → binary mask + labels."""
    bg   = gaussian(channel, sigma=PUNCTA_BG_SIGMA)
    sub  = np.maximum(channel - bg * 0.85, 0)
    sm   = gaussian(sub, sigma=PUNCTA_FG_SIGMA)
    thr  = np.percentile(sm[sm > 0], PUNCTA_PERCENTILE)
    mask = sm > thr
    mask = morphology.remove_small_objects(mask, min_size=PUNCTA_MIN_SIZE)
    mask = morphology.remove_small_holes(mask, area_threshold=20)
    return mask, measure.label(mask)

def per_cell_stats(cell_labels, lamp1_mask, mtor_mask):
    """Return list of per-cell dicts with colocalization metrics."""
    overlap = lamp1_mask & mtor_mask
    rows = []
    for cid in range(1, cell_labels.max() + 1):
        roi        = (cell_labels == cid)
        mtor_px    = int(np.sum(mtor_mask  & roi))
        lamp1_px   = int(np.sum(lamp1_mask & roi))
        coloc_px   = int(np.sum(overlap    & roi))
        if mtor_px < 50:          # skip cells with almost no mTOR signal
            continue
        coloc_frac = coloc_px / mtor_px
        rows.append({
            'cell_id'   : cid,
            'mtor_px'   : mtor_px,
            'lamp1_px'  : lamp1_px,
            'coloc_px'  : coloc_px,
            'coloc_frac': coloc_frac,
        })
    return rows

# ── File discovery ────────────────────────────────────────────────────────────
# Pattern: MiaPaca2_{COND}_{GLUC}_..._C{ch}.tif
# Conditions: FED/ST  |  Glucose: HG/LG
COND_RE = re.compile(r'MiaPaca2_(FED|ST)_(HG|LG).*_C(\d)\.tif$')

groups = {}   # (cond, gluc) -> {0: path, 1: path, 3: path}
for fname in sorted(os.listdir(IMAGE_DIR)):
    m = COND_RE.search(fname)
    if m:
        cond, gluc, ch = m.group(1), m.group(2), int(m.group(3))
        key = (cond, gluc)
        groups.setdefault(key, {})[ch] = os.path.join(IMAGE_DIR, fname)

print(f"Found {len(groups)} conditions:")
for k, v in groups.items():
    print(f"  {k[0]}_{k[1]}: channels {sorted(v.keys())}")

# ── Main loop ─────────────────────────────────────────────────────────────────
all_rows   = []
panels     = {}   # store per-condition image arrays for composite figure

CONDITION_LABELS = {
    ('FED','HG'): 'FED / High Glucose',
    ('FED','LG'): 'FED / Low Glucose',
    ('ST', 'HG'): 'STARVED / High Glucose',
    ('ST', 'LG'): 'STARVED / Low Glucose',
}

for (cond, gluc), channels in sorted(groups.items()):
    label = CONDITION_LABELS.get((cond, gluc), f"{cond}_{gluc}")
    print(f"\n── {label} ──")

    if not all(ch in channels for ch in [0, 1, 3]):
        print(f"  WARNING: missing channels, skipping")
        continue

    # Load best focal plane
    c0_stack = tifffile.imread(channels[0]).astype(np.float32)
    c1_stack = tifffile.imread(channels[1]).astype(np.float32)
    c3_stack = tifffile.imread(channels[3]).astype(np.float32)

    c0 = c0_stack[best_z(c0_stack)]
    c1 = c1_stack[best_z(c1_stack)]
    c3 = c3_stack[best_z(c3_stack)]
    print(f"  Image size: {c0.shape}  |  Z-planes: {c0_stack.shape[0]}")

    # Segment
    nuc_labels, cell_labels, dist, markers = segment_nuclei(c0)
    print(f"  Nuclei: {nuc_labels.max()}")

    lamp1_mask, lamp1_labels = detect_puncta(c1)
    mtor_mask,  mtor_labels  = detect_puncta(c3)
    print(f"  LAMP1 puncta: {lamp1_labels.max()}  |  mTOR puncta: {mtor_labels.max()}")

    # Per-cell colocalization
    rows = per_cell_stats(cell_labels, lamp1_mask, mtor_mask)
    for r in rows:
        r['condition'] = cond
        r['glucose']   = gluc
        r['label']     = label
    all_rows.extend(rows)

    fracs = [r['coloc_frac'] for r in rows]
    print(f"  Cells with signal: {len(fracs)}")
    if fracs:
        print(f"  mTOR→LAMP1 overlap:  mean={np.mean(fracs)*100:.1f}%  "
              f"median={np.median(fracs)*100:.1f}%  std={np.std(fracs)*100:.1f}%")

    # Store normalized images for figure
    panels[(cond, gluc)] = {
        'c0': norm(c0), 'c1': norm(c1), 'c3': norm(c3),
        'nuc': nuc_labels, 'cell': cell_labels,
        'lamp1': lamp1_mask, 'mtor': mtor_mask,
        'overlap': lamp1_mask & mtor_mask,
        'fracs': fracs, 'label': label,
    }

    # ── Individual condition panel ──────────────────────────────────────────
    S = 512
    H, W = c0.shape
    sl = np.s_[H//2-S:H//2+S, W//2-S:W//2+S]

    fig, axes = plt.subplots(2, 4, figsize=(18, 9), facecolor='#0B1C2C')
    fig.suptitle(f'MiaPaca2  |  {label}  |  100×', color='white',
                 fontsize=14, fontweight='bold')

    c0c, c1c, c3c = c0[sl], c1[sl], c3[sl]
    nucc  = nuc_labels[sl];  cellc = cell_labels[sl]
    l1c   = lamp1_mask[sl];  mtc   = mtor_mask[sl];  ovc = (lamp1_mask & mtor_mask)[sl]

    # Row 0: raw channels + nucleus seg
    for ax, ch, title, cmap in zip(
        axes[0], [c0c, c1c, c3c], ['C0 DAPI','C1 LAMP1','C3 mTOR'], ['Blues','Greens','Reds']):
        ax.imshow(norm(ch), cmap=cmap); ax.set_title(title, color='white', fontsize=10); ax.axis('off')

    ax_nuc = axes[0,3]
    ax_nuc.imshow(norm(c0c), cmap='gray', vmin=0, vmax=0.4)
    nb = segmentation.find_boundaries(nucc, mode='outer')
    ov_r = np.zeros((*c0c.shape,4)); ov_r[nb,0]=1; ov_r[nb,3]=0.9
    ax_nuc.imshow(ov_r)
    for reg in measure.regionprops(nucc):
        ry, rx = reg.centroid
        if 0 < ry < 1024 and 0 < rx < 1024:
            ax_nuc.text(rx, ry, str(reg.label), color='white', fontsize=9,
                        ha='center', va='center', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.1', facecolor='#1A7C8C', alpha=0.8, linewidth=0))
    ax_nuc.set_title('Nuclei segmented', color='white', fontsize=10); ax_nuc.axis('off')

    # Row 1: puncta + coloc + bar
    axes[1,0].imshow(norm(c1c), cmap='Greens', vmin=0, vmax=0.8)
    ov1 = np.zeros((*c1c.shape,4)); ov1[l1c,1]=1; ov1[l1c,2]=0.4; ov1[l1c,3]=0.55
    axes[1,0].imshow(ov1)
    axes[1,0].set_title(f'LAMP1 puncta ({measure.label(l1c).max()})', color='white', fontsize=10); axes[1,0].axis('off')

    axes[1,1].imshow(norm(c3c), cmap='Reds', vmin=0, vmax=0.8)
    ov2 = np.zeros((*c3c.shape,4)); ov2[mtc,0]=1; ov2[mtc,1]=0.3; ov2[mtc,3]=0.55
    axes[1,1].imshow(ov2)
    axes[1,1].set_title(f'mTOR puncta ({measure.label(mtc).max()})', color='white', fontsize=10); axes[1,1].axis('off')

    rgb = np.zeros((*c1c.shape,3))
    rgb[:,:,1] = norm(c1c)*0.9; rgb[:,:,0] = norm(c3c)*0.9
    rgb[ovc,0]=1; rgb[ovc,1]=1; rgb[ovc,2]=0
    axes[1,2].imshow(np.clip(rgb,0,1))
    axes[1,2].legend(handles=[
        mpatches.Patch(color='yellow', label='Colocalized'),
        mpatches.Patch(color='#00AA44', label='LAMP1 only'),
        mpatches.Patch(color='#CC3300', label='mTOR only'),
    ], loc='lower right', fontsize=7, facecolor='#0B1C2C', labelcolor='white', framealpha=0.8)
    axes[1,2].set_title('Colocalization', color='white', fontsize=10); axes[1,2].axis('off')

    ax_bar = axes[1,3]; ax_bar.set_facecolor('#0F2030')
    if fracs:
        ax_bar.bar(range(1,len(fracs)+1), [f*100 for f in fracs], color='#E8A844', edgecolor='#C07020')
        ax_bar.axhline(np.mean(fracs)*100, color='white', linestyle='--', linewidth=1.2,
                       label=f'Mean {np.mean(fracs)*100:.1f}%')
        ax_bar.legend(fontsize=8, facecolor='#0F2030', labelcolor='white')
    ax_bar.set_ylim(0, 80); ax_bar.set_xlabel('Cell', color='white', fontsize=8)
    ax_bar.set_ylabel('% mTOR in LAMP1', color='white', fontsize=8)
    ax_bar.tick_params(colors='white', labelsize=7)
    for sp in ['top','right']: ax_bar.spines[sp].set_visible(False)
    for sp in ['bottom','left']: ax_bar.spines[sp].set_color('#4A6274')
    ax_bar.set_title('Per-cell coloc %', color='white', fontsize=10)

    for ax in axes.flat:
        ax.set_facecolor('#0B1C2C')
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, f"{cond}_{gluc}_panel.png")
    plt.savefig(out_path, dpi=130, bbox_inches='tight', facecolor='#0B1C2C')
    plt.close()
    print(f"  Saved: {out_path}")

# ── Save CSV ──────────────────────────────────────────────────────────────────
df = pd.DataFrame(all_rows)
csv_path = os.path.join(OUTPUT_DIR, 'coloc_summary.csv')
df.to_csv(csv_path, index=False)
print(f"\nCSV saved: {csv_path}")
print(df.groupby('label')['coloc_frac'].agg(['count','mean','median','std']).round(3))

# ── 2×2 Summary Figure ────────────────────────────────────────────────────────
ORDER = [('FED','HG'), ('FED','LG'), ('ST','HG'), ('ST','LG')]
ORDER = [k for k in ORDER if k in panels]

fig = plt.figure(figsize=(20, 14), facecolor='#0B1C2C')
fig.suptitle('MiaPaca2 — mTOR Lysosomal Recruitment\nNutrient Perturbation 2×2: FED vs STARVED × High vs Low Glucose',
             color='white', fontsize=16, fontweight='bold', y=0.98)

S = 400   # smaller crop for composite
positions = [(0.03,0.52), (0.27,0.52), (0.03,0.14), (0.27,0.14)]

for (cond, gluc), (left, bot) in zip(ORDER, positions):
    if (cond, gluc) not in panels:
        continue
    p = panels[(cond, gluc)]
    H, W = p['c0'].shape
    sl = np.s_[H//2-S:H//2+S, W//2-S:W//2+S]

    c1c  = p['c1'][sl]; c3c = p['c3'][sl]
    ovc  = p['overlap'][sl]

    # RGB composite
    rgb = np.zeros((*c1c.shape, 3))
    rgb[:,:,1] = norm(c1c)*0.9
    rgb[:,:,0] = norm(c3c)*0.9
    rgb[ovc,0]=1; rgb[ovc,1]=1; rgb[ovc,2]=0
    rgb = np.clip(rgb, 0, 1)

    ax_img = fig.add_axes([left, bot, 0.22, 0.36])
    ax_img.imshow(rgb)
    ax_img.axis('off')

    fracs = p['fracs']
    mean_pct = np.mean(fracs)*100 if fracs else 0
    n = len(fracs)

    # condition label
    color = '#E8A844' if cond == 'FED' else '#1A7C8C'
    ax_img.set_title(p['label'], color='white', fontsize=11, fontweight='bold', pad=4)

    # mean badge overlaid
    ax_img.text(0.97, 0.97, f'{mean_pct:.1f}%\nn={n}',
                transform=ax_img.transAxes, color='white', fontsize=13,
                fontweight='bold', ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.88, linewidth=0))

    # mini bar chart below image
    ax_bar = fig.add_axes([left, bot-0.1, 0.22, 0.09])
    ax_bar.set_facecolor('#0F2030')
    if fracs:
        ax_bar.bar(range(1,n+1), [f*100 for f in fracs], color=color, edgecolor='none', width=0.6)
        ax_bar.axhline(mean_pct, color='white', linestyle='--', linewidth=1)
    ax_bar.set_ylim(0, 80)
    ax_bar.set_xlim(0.3, max(n+0.7, 2))
    ax_bar.set_ylabel('% mTOR\nin LAMP1', color='white', fontsize=7, labelpad=2)
    ax_bar.set_xlabel('Cell', color='white', fontsize=7)
    ax_bar.tick_params(colors='white', labelsize=7)
    for sp in ['top','right']: ax_bar.spines[sp].set_visible(False)
    for sp in ['bottom','left']: ax_bar.spines[sp].set_color('#4A6274')

# Big bar chart — mean per condition
ax_main = fig.add_axes([0.56, 0.12, 0.4, 0.78])
ax_main.set_facecolor('#0F2030')

cond_means, cond_sems, cond_labels, cond_colors = [], [], [], []
for cond, gluc in ORDER:
    if (cond, gluc) not in panels:
        continue
    fracs = panels[(cond,gluc)]['fracs']
    if not fracs:
        continue
    cond_means.append(np.mean(fracs)*100)
    cond_sems.append(np.std(fracs)/np.sqrt(len(fracs))*100)
    cond_labels.append(CONDITION_LABELS[(cond,gluc)].replace(' / ','\n'))
    cond_colors.append('#E8A844' if cond=='FED' else '#1A7C8C')

x = np.arange(len(cond_means))
bars = ax_main.bar(x, cond_means, yerr=cond_sems, color=cond_colors,
                   edgecolor='none', width=0.55, capsize=6,
                   error_kw={'ecolor':'white','linewidth':1.5})

ax_main.set_xticks(x)
ax_main.set_xticklabels(cond_labels, color='white', fontsize=11)
ax_main.set_ylabel('% mTOR colocalized with LAMP1\n(mean ± SEM per cell)', color='white', fontsize=11)
ax_main.set_ylim(0, 75)
ax_main.set_title('mTOR Lysosomal Recruitment by Condition', color='white', fontsize=13, fontweight='bold', pad=10)
ax_main.tick_params(colors='white', labelsize=10)
for sp in ['top','right']: ax_main.spines[sp].set_visible(False)
for sp in ['bottom','left']: ax_main.spines[sp].set_color('#4A6274')

# Annotate bars with value
for xi, (m, s) in enumerate(zip(cond_means, cond_sems)):
    ax_main.text(xi, m + s + 1.5, f'{m:.1f}%', ha='center', va='bottom',
                 color='white', fontsize=12, fontweight='bold')

# Legend
ax_main.legend(handles=[
    mpatches.Patch(color='#E8A844', label='FED conditions'),
    mpatches.Patch(color='#1A7C8C', label='STARVED conditions'),
], facecolor='#0F2030', labelcolor='white', fontsize=10, loc='upper right')

# Interpretation text
interp = (
    "Hypothesis:\n"
    "  FED > STARVED  →  mTOR leaves lysosome when amino acids depleted\n"
    "  HG vs LG       →  glucose availability modulates via AMPK arm\n\n"
    "The yellow signal = mTOR ON the lysosome = mTOR active\n"
    "Drop in % = mTOR leaving the lysosome surface = mTOR OFF\n\n"
    "This is the imaging readout of the biology\n"
    "Rogala's structures explain at atomic resolution."
)
ax_main.text(0.02, 0.02, interp, transform=ax_main.transAxes,
             color='#A8BFCB', fontsize=9, va='bottom', fontfamily='monospace',
             bbox=dict(facecolor='#0B1C2C', alpha=0.6, linewidth=0, pad=4))

summary_path = os.path.join(OUTPUT_DIR, 'coloc_summary.png')
plt.savefig(summary_path, dpi=130, bbox_inches='tight', facecolor='#0B1C2C')
plt.close()
print(f"Summary figure saved: {summary_path}")
print("\nDone. All outputs in ./coloc_results/")
