"""
Parameter Sweep for Object-Based Colocalization — Rogala Lab v2
Imports shared functions from utils.py

Tests combinations of:
  residual_factor    — background subtraction aggressiveness
  lamp1_percentiles  — intensity threshold for reference channel (C1)
  marker_percentiles — intensity threshold for marker channel (C2)
  proximities        — colocalization distance threshold

Runs on a SINGLE representative image or first collection from manifest.

Outputs:
  phase1_heatmaps.png        — lamp1_pct × marker_pct heatmaps at best rf
  phase2_proximity.png       — proximity threshold sweep
  phase3_visual_comparison.png — puncta detection visuals
  phase1_rf_percentile.csv   — full phase 1 numbers
  phase2_proximity.csv       — full phase 2 numbers

Usage:
  # From manifest (recommended — reads channel names automatically)
  python param_sweep.py \\
    --manifest ~/Desktop/manifest.tsv \\
    --output ~/Desktop/sweep

  # From individual files
  python param_sweep.py \\
    --dapi  /path/C0.tif \\
    --lamp1 /path/C1.tif \\
    --marker /path/C2.tif \\
    --output ~/Desktop/sweep

  # Narrow the sweep after first run
  python param_sweep.py --manifest ~/Desktop/manifest.tsv \\
    --residual_factors 0.5 1.0 1.5 \\
    --lamp1_percentiles 78 82 85 \\
    --marker_percentiles 88 90 93 \\
    --proximities 2 3 5
"""

import os, re, argparse, warnings, sys
from itertools import product
import numpy as np
import pandas as pd
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (load_plane, segment_nuclei, detect_puncta_objects,
                   local_median_subtract, object_colocalization, norm)

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Parameter sweep for object-based colocalization')

# Image input — manifest or individual files
parser.add_argument('--manifest', type=str, default=None,
                    help='Manifest TSV from ingest.py. Uses first collection as sweep image.')
parser.add_argument('--dapi',    type=str, default=None,
                    help='Path to DAPI channel TIFF (not needed if --manifest provided)')
parser.add_argument('--lamp1',   type=str, default=None,
                    help='Path to reference channel TIFF (C1)')
parser.add_argument('--marker',  type=str, default=None,
                    help='Path to marker channel TIFF (C2)')

parser.add_argument('--output',     type=str, default='./param_sweep')
parser.add_argument('--projection', type=str, default='max',
                    choices=['max','best_z','mean'])
parser.add_argument('--window',     type=int, default=32,
                    help='Local median window size (default 32)')
parser.add_argument('--min_puncta_px', type=int, default=5)

# Parameter ranges
parser.add_argument('--residual_factors',    type=float, nargs='+',
                    default=[0.0, 0.5, 1.0, 1.5, 2.0])
parser.add_argument('--lamp1_percentiles',   type=float, nargs='+',
                    default=[75, 80, 85],
                    help='Percentile values for reference channel (C1)')
parser.add_argument('--marker_percentiles',  type=float, nargs='+',
                    default=[85, 90, 93, 95],
                    help='Percentile values for marker channel (C2)')
parser.add_argument('--percentiles', type=float, nargs='+', default=None,
                    help='Set same percentile range for both channels')
parser.add_argument('--proximities', type=int, nargs='+',
                    default=[1, 3, 5, 7])
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

# Channel display names — read from manifest if available
CH1_NAME    = 'Reference (C1)'
MARKER_NAME = 'Marker (C2)'
CELL_LINE   = 'HEK293'

# ── Load image ────────────────────────────────────────────────────────────────
print("Loading images...")

if args.manifest:
    mdf = pd.read_csv(args.manifest, sep='\t')
    if 'ch1_name'    in mdf.columns: CH1_NAME    = mdf['ch1_name'].iloc[0]
    if 'marker_name' in mdf.columns: MARKER_NAME = mdf['marker_name'].iloc[0]
    if 'cell_line'   in mdf.columns: CELL_LINE   = mdf['cell_line'].iloc[0]

    # Use a representative row — prefer DMSO or Basal condition
    rep = mdf.iloc[0]
    for _, row in mdf.iterrows():
        if 'DMSO' in str(row['label']) or 'Basal' in str(row['label']):
            rep = row; break

    print(f"Using: {rep['label']} / {rep['collection']}")
    print(f"  {CH1_NAME}: {os.path.basename(rep['ch_lamp1'])}")
    print(f"  {MARKER_NAME}: {os.path.basename(rep['ch_marker'])}")
    c0 = load_plane(rep['ch_dapi'],   args.projection)
    c1 = load_plane(rep['ch_lamp1'],  args.projection)
    cm = load_plane(rep['ch_marker'], args.projection)

elif args.dapi and args.lamp1 and args.marker:
    c0 = load_plane(args.dapi,   args.projection)
    c1 = load_plane(args.lamp1,  args.projection)
    cm = load_plane(args.marker, args.projection)

else:
    print("❌ Provide either --manifest or --dapi / --lamp1 / --marker")
    exit(1)

# Crop centre for speed
H, W = c0.shape
S    = 512
sl   = np.s_[H//2-S:H//2+S, W//2-S:W//2+S]
c0c, c1c, cmc = c0[sl], c1[sl], cm[sl]

print(f"\nSegmenting nuclei...", end=' ', flush=True)
nuc_lbl, cell_lbl = segment_nuclei(c0c)
n_cells = nuc_lbl.max()
print(f"done ({n_cells} nuclei in crop)")

# Build cytoplasm ROIs once
cyto_rois = {
    cid: (cell_lbl==cid) & ~(nuc_lbl==cid)
    for cid in range(1, cell_lbl.max()+1)
    if ((cell_lbl==cid) & ~(nuc_lbl==cid)).sum() >= 100
}
print(f"Valid cells: {len(cyto_rois)}\n")

# ── Phase 1: rf × lamp1_pct × marker_pct ─────────────────────────────────────
MID_PROXIMITY = PROXIMITIES[len(PROXIMITIES)//2]
print(f"Phase 1: rf × {CH1_NAME}_pct × {MARKER_NAME}_pct  (proximity={MID_PROXIMITY}px fixed)")
print(f"  rf values:           {FACTORS}")
print(f"  {CH1_NAME} percentiles:  {LAMP1_PCTS}")
print(f"  {MARKER_NAME} percentiles: {MARKER_PCTS}\n")

phase1 = []

for rf, l_pct, m_pct in product(FACTORS, LAMP1_PCTS, MARKER_PCTS):
    c1_sub = local_median_subtract(c1c, window=args.window, residual_factor=rf)
    cm_sub = local_median_subtract(cmc, window=args.window, residual_factor=rf)

    n_lamp1_list, n_marker_list, fracs = [], [], []

    for cid, cyto_roi in cyto_rois.items():
        l_lbl, l_props = detect_puncta_objects(c1_sub, cyto_roi, args.min_puncta_px, l_pct)
        m_lbl, m_props = detect_puncta_objects(cm_sub, cyto_roi, args.min_puncta_px, m_pct)
        if not m_props: continue
        _, n_coloc, _ = object_colocalization(l_lbl, m_lbl, m_props, MID_PROXIMITY)
        n_lamp1_list.append(len(l_props))
        n_marker_list.append(len(m_props))
        fracs.append(n_coloc / len(m_props))

    phase1.append({
        'residual_factor'   : rf,
        'lamp1_percentile'  : l_pct,
        'marker_percentile' : m_pct,
        'proximity_px'      : MID_PROXIMITY,
        'n_cells'           : len(fracs),
        'mean_n_lamp1'      : np.mean(n_lamp1_list) if n_lamp1_list else 0,
        'mean_n_marker'     : np.mean(n_marker_list) if n_marker_list else 0,
        'mean_coloc_frac'   : np.mean(fracs) if fracs else np.nan,
        'std_coloc_frac'    : np.std(fracs)  if fracs else np.nan,
    })
    print(f"  rf={rf:.1f} {CH1_NAME}_pct={l_pct:.0f} {MARKER_NAME}_pct={m_pct:.0f} → "
          f"{CH1_NAME}/cell={np.mean(n_lamp1_list):.0f}  "
          f"{MARKER_NAME}/cell={np.mean(n_marker_list):.0f}  "
          f"coloc={np.mean(fracs)*100:.1f}%" if fracs else
          f"  rf={rf:.1f} → no cells", flush=True)

df1 = pd.DataFrame(phase1)

# Auto-select best: maximise cells, keep marker count reasonable
valid = df1[df1['n_cells'] > 0].copy()
valid['score'] = valid['n_cells'] - abs(valid['mean_n_marker'] - 30) * 0.1
best_row  = valid.loc[valid['score'].idxmax()]
BEST_RF   = best_row['residual_factor']
BEST_LPCT = best_row['lamp1_percentile']
BEST_MPCT = best_row['marker_percentile']

print(f"\nAuto-selected best: rf={BEST_RF}, {CH1_NAME}_pct={BEST_LPCT}, "
      f"{MARKER_NAME}_pct={BEST_MPCT}")

# ── Phase 2: proximity sweep ──────────────────────────────────────────────────
print(f"\nPhase 2: proximity sweep (rf={BEST_RF}, "
      f"{CH1_NAME}_pct={BEST_LPCT}, {MARKER_NAME}_pct={BEST_MPCT})")

c1_best = local_median_subtract(c1c, window=args.window, residual_factor=BEST_RF)
cm_best = local_median_subtract(cmc, window=args.window, residual_factor=BEST_RF)

phase2 = []
for prox in PROXIMITIES:
    fracs = []
    for cid, cyto_roi in cyto_rois.items():
        l_lbl, l_props = detect_puncta_objects(c1_best, cyto_roi, args.min_puncta_px, BEST_LPCT)
        m_lbl, m_props = detect_puncta_objects(cm_best, cyto_roi, args.min_puncta_px, BEST_MPCT)
        if not m_props: continue
        _, n_coloc, _ = object_colocalization(l_lbl, m_lbl, m_props, prox)
        fracs.append(n_coloc / len(m_props))
    phase2.append({
        'proximity_px'   : prox,
        'mean_coloc_frac': np.mean(fracs) if fracs else np.nan,
        'std_coloc_frac' : np.std(fracs)  if fracs else np.nan,
        'n_cells'        : len(fracs),
    })
    print(f"  proximity={prox}px → coloc={np.mean(fracs)*100:.1f}%  "
          f"(std={np.std(fracs)*100:.1f}%)" if fracs else
          f"  proximity={prox}px → no cells", flush=True)

df2 = pd.DataFrame(phase2)

# Save CSVs
df1.to_csv(os.path.join(args.output,'phase1_rf_percentile.csv'), index=False)
df2.to_csv(os.path.join(args.output,'phase2_proximity.csv'), index=False)

# ── Figure 1: Phase 1 heatmaps ────────────────────────────────────────────────
print("\nGenerating figures...")
df1_best = df1[df1['residual_factor'] == BEST_RF]
l_pcts   = sorted(df1['lamp1_percentile'].unique())
m_pcts   = sorted(df1['marker_percentile'].unique())

def make_grid(df, col, rf):
    grid = np.full((len(l_pcts), len(m_pcts)), np.nan)
    for _, row in df[df['residual_factor']==rf].iterrows():
        li = l_pcts.index(row['lamp1_percentile'])
        mi = m_pcts.index(row['marker_percentile'])
        grid[li, mi] = row[col]
    return grid

fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor='#F8F7F4')
fig.suptitle(f'Parameter Sweep Phase 1: {CH1_NAME}_pct × {MARKER_NAME}_pct\n'
             f'(rf={BEST_RF} fixed, proximity={MID_PROXIMITY}px fixed, '
             f'crop {S*2}×{S*2}px)',
             color='#1E2D3A', fontsize=13, fontweight='bold')

metrics = [
    ('mean_coloc_frac', f'% {MARKER_NAME} on {CH1_NAME}\n(primary)', 'RdYlGn', True),
    ('mean_n_marker',   f'Mean {MARKER_NAME} puncta/cell\n(want ~20-50)', 'Blues', False),
    ('mean_n_lamp1',    f'Mean {CH1_NAME} puncta/cell\n(more = better)', 'Greens', False),
    ('n_cells',         'Cells with data\n(more = better)', 'Purples', False),
]

best_li = l_pcts.index(BEST_LPCT)
best_mi = m_pcts.index(BEST_MPCT)

for ax, (col, title, cmap, pct_scale) in zip(axes.flat, metrics):
    grid = make_grid(df1, col, BEST_RF)
    if pct_scale: grid = grid * 100
    im = ax.imshow(grid, cmap=cmap, aspect='auto',
                   vmin=np.nanmin(grid), vmax=np.nanmax(grid))
    ax.set_xticks(range(len(m_pcts)))
    ax.set_xticklabels([f'{p:.0f}' for p in m_pcts], color='#1E2D3A', fontsize=9)
    ax.set_yticks(range(len(l_pcts)))
    ax.set_yticklabels([f'{p:.0f}' for p in l_pcts], color='#1E2D3A', fontsize=9)
    ax.set_xlabel(f'{MARKER_NAME} percentile', color='#1E2D3A', fontsize=10)
    ax.set_ylabel(f'{CH1_NAME} percentile', color='#1E2D3A', fontsize=10)
    ax.set_title(title, color='#1E2D3A', fontsize=10, fontweight='bold')
    plt.colorbar(im, ax=ax)
    for li in range(len(l_pcts)):
        for mi in range(len(m_pcts)):
            v = grid[li, mi]
            if not np.isnan(v):
                ax.text(mi, li, f'{v:.0f}{"%" if pct_scale else ""}',
                        ha='center', va='center', fontsize=8,
                        color='#1E2D3A', fontweight='bold')
    ax.add_patch(plt.Rectangle((best_mi-0.5, best_li-0.5), 1, 1,
                                fill=False, edgecolor='red', linewidth=2.5))

plt.tight_layout()
plt.savefig(os.path.join(args.output,'phase1_heatmaps.png'),
            dpi=130, bbox_inches='tight', facecolor='#F8F7F4')
plt.close()

# ── Figure 2: Phase 2 proximity ───────────────────────────────────────────────
fig2, ax2 = plt.subplots(1, 1, figsize=(8, 5), facecolor='#F8F7F4')
ax2.set_facecolor('#F0F4F6')
proxs = df2['proximity_px'].tolist()
means = (df2['mean_coloc_frac']*100).tolist()
stds  = (df2['std_coloc_frac']*100).tolist()
ax2.errorbar(proxs, means, yerr=stds, fmt='o-', color='#4A9BAF',
             linewidth=2, markersize=8, capsize=5, elinewidth=1.5)
for px, m in zip(proxs, means):
    ax2.text(px, m+1.5, f'{m:.1f}%', ha='center', va='bottom',
             color='#1E2D3A', fontsize=9, fontweight='bold')
ax2.set_xlabel(f'{CH1_NAME} mask dilation radius (px)', color='#1E2D3A', fontsize=11)
ax2.set_ylabel(f'% {MARKER_NAME} on {CH1_NAME} (mean ± std)', color='#1E2D3A', fontsize=11)
ax2.set_title(f'Phase 2: Proximity threshold sweep\n'
              f'(rf={BEST_RF}, {CH1_NAME}_pct={BEST_LPCT}, '
              f'{MARKER_NAME}_pct={BEST_MPCT})',
              color='#1E2D3A', fontsize=11, fontweight='bold')
ax2.tick_params(colors='#1E2D3A')
for sp in ['top','right']: ax2.spines[sp].set_visible(False)
for sp in ['bottom','left']: ax2.spines[sp].set_color('#A0B8C4')
plt.tight_layout()
plt.savefig(os.path.join(args.output,'phase2_proximity.png'),
            dpi=130, bbox_inches='tight', facecolor='#F8F7F4')
plt.close()

# ── Figure 3: Visual comparison ───────────────────────────────────────────────
test_combos = [
    (BEST_RF, BEST_LPCT, BEST_MPCT, MID_PROXIMITY, 'BEST'),
    (BEST_RF, max(LAMP1_PCTS[0], BEST_LPCT-5), BEST_MPCT, MID_PROXIMITY,
     f'{CH1_NAME}_pct-5'),
    (BEST_RF, min(LAMP1_PCTS[-1], BEST_LPCT+5), BEST_MPCT, MID_PROXIMITY,
     f'{CH1_NAME}_pct+5'),
    (BEST_RF, BEST_LPCT, max(MARKER_PCTS[0], BEST_MPCT-5), MID_PROXIMITY,
     f'{MARKER_NAME}_pct-5'),
    (BEST_RF, BEST_LPCT, min(MARKER_PCTS[-1], BEST_MPCT+5), MID_PROXIMITY,
     f'{MARKER_NAME}_pct+5'),
]

# Pick representative cell
rep_cid  = list(cyto_rois.keys())[len(cyto_rois)//2]
cyto_rep = cyto_rois[rep_cid]
ys, xs   = np.where(cyto_rep)
pad      = 20
cell_sl  = np.s_[max(0,ys.min()-pad):min(H,ys.max()+pad),
                 max(0,xs.min()-pad):min(W,xs.max()+pad)]

fig3, axes3 = plt.subplots(3, len(test_combos),
                            figsize=(5*len(test_combos), 12),
                            facecolor='#F8F7F4')
fig3.suptitle(f'Visual comparison — {CELL_LINE}\n'
              f'Red box = auto-recommended parameters',
              color='#1E2D3A', fontsize=12, fontweight='bold')

for col, (rf, l_pct, m_pct, prox, label) in enumerate(test_combos):
    c1_s = local_median_subtract(c1c, window=args.window, residual_factor=rf)
    cm_s = local_median_subtract(cmc, window=args.window, residual_factor=rf)
    l_lbl, l_props = detect_puncta_objects(c1_s, cyto_rep, args.min_puncta_px, l_pct)
    m_lbl, m_props = detect_puncta_objects(cm_s, cyto_rep, args.min_puncta_px, m_pct)
    col_mask, n_coloc, _ = object_colocalization(l_lbl, m_lbl, m_props, prox)
    frac  = n_coloc / len(m_props) * 100 if m_props else 0
    title = f'rf={rf} L={l_pct:.0f} M={m_pct:.0f}\n{label}'

    axes3[0,col].imshow(norm(c1_s[cell_sl]), cmap='Greens', vmin=0, vmax=1)
    ov=np.zeros((*c1_s.shape,4)); ov[l_lbl>0,1]=1; ov[l_lbl>0,2]=0.3; ov[l_lbl>0,3]=0.7
    axes3[0,col].imshow(ov[cell_sl])
    axes3[0,col].set_title(title, color='#1E2D3A', fontsize=8,
                            fontweight='bold' if label=='BEST' else 'normal')
    axes3[0,col].set_ylabel(CH1_NAME if col==0 else '', color='#1E2D3A', fontsize=8)
    axes3[0,col].text(0.02, 0.02, f'{len(l_props)} puncta',
                      transform=axes3[0,col].transAxes, color='white', fontsize=7,
                      bbox=dict(facecolor='#2C7A8C', alpha=0.8, linewidth=0, pad=2))
    axes3[0,col].axis('off')

    axes3[1,col].imshow(norm(cm_s[cell_sl]), cmap='Reds', vmin=0, vmax=1)
    ov2=np.zeros((*cm_s.shape,4)); ov2[m_lbl>0,0]=1; ov2[m_lbl>0,1]=0.3; ov2[m_lbl>0,3]=0.7
    axes3[1,col].imshow(ov2[cell_sl])
    axes3[1,col].set_ylabel(MARKER_NAME if col==0 else '', color='#1E2D3A', fontsize=8)
    axes3[1,col].text(0.02, 0.02, f'{len(m_props)} puncta',
                      transform=axes3[1,col].transAxes, color='white', fontsize=7,
                      bbox=dict(facecolor='#C04020', alpha=0.8, linewidth=0, pad=2))
    axes3[1,col].axis('off')

    rgb=np.zeros((*c1_s.shape,3))
    rgb[:,:,1]=norm(c1_s)*0.6; rgb[:,:,0]=norm(cm_s)*0.6
    rgb[l_lbl>0,1]=0.8; rgb[l_lbl>0,0]=0
    rgb[m_lbl>0,0]=0.8; rgb[m_lbl>0,1]=0
    rgb[col_mask,0]=1; rgb[col_mask,1]=1; rgb[col_mask,2]=0
    axes3[2,col].imshow(np.clip(rgb[cell_sl],0,1))
    axes3[2,col].set_ylabel('Overlay' if col==0 else '', color='#1E2D3A', fontsize=8)
    axes3[2,col].text(0.02, 0.02, f'{frac:.0f}% coloc',
                      transform=axes3[2,col].transAxes, color='white', fontsize=8,
                      fontweight='bold',
                      bbox=dict(facecolor='#1E2D3A', alpha=0.85, linewidth=0, pad=2))
    axes3[2,col].axis('off')

    if label == 'BEST':
        for r_ax in axes3[:, col]:
            for spine in r_ax.spines.values():
                spine.set_edgecolor('red'); spine.set_linewidth(3)

plt.tight_layout()
plt.savefig(os.path.join(args.output,'phase3_visual_comparison.png'),
            dpi=120, bbox_inches='tight', facecolor='#F8F7F4')
plt.close()

# ── Final summary ─────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"RECOMMENDED PARAMETERS (auto-selected):")
print(f"  --residual_factor    {BEST_RF}")
print(f"  --lamp1_percentile   {BEST_LPCT}   ({CH1_NAME})")
print(f"  --marker_percentile  {BEST_MPCT}   ({MARKER_NAME})")
print(f"  --proximity_px       {MID_PROXIMITY}  (review phase2_proximity.png)")
print(f"{'='*60}")
print(f"\nOutputs in: {args.output}")
print(f"  phase1_heatmaps.png         — {CH1_NAME}_pct × {MARKER_NAME}_pct heatmaps")
print(f"  phase2_proximity.png        — proximity threshold curve")
print(f"  phase3_visual_comparison.png — single-cell visual check")