"""
Raptor / mTOR – Lysosome Colocalization Analysis
Rogala Lab — v3

Supports three dataset layouts:

  miapaca2  — flat folder, condition encoded in filename
              MiaPaca2_{FED|ST}_{HG|LG}_..._C{ch}.tif
              channels: C0=DAPI  C1=LAMP1  C3=mTOR

  hek293    — flat folder, condition encoded in filename
              Slide N_{DRUG}_{REFED}_..._C{ch}.tif
              channels: C0=DAPI  C1=LAMP1  C2=Raptor

  subdir    — one subdirectory per condition, multiple collections pooled
              <condition_dir>/SlideN-CollectionN_XY..._C{ch}.tif
              channels: C0=DAPI  C1=LAMP1  C2=Raptor
              condition label = directory name (e.g. "20µM 6698_Starve_Refed")

Usage:
  python analyze_coloc_v3.py --image_dir "Rogala Lab Images Olga"
  python analyze_coloc_v3.py --image_dir "HEK293 Images" --dataset hek293
  python analyze_coloc_v3.py --image_dir "/path/to/071526" --dataset subdir
  python analyze_coloc_v3.py --image_dir "/path/to/071526" --dataset subdir --output_dir ~/Desktop/results

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

warnings.filterwarnings('ignore')

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Lysosome colocalization analysis')
parser.add_argument('--image_dir', type=str, required=True,
                    help='Root folder containing images or condition subdirectories')
parser.add_argument('--output_dir', type=str, default=None,
                    help='Output folder (default: <image_dir>/coloc_results)')
parser.add_argument('--dataset', type=str, default='miapaca2',
                    choices=['miapaca2', 'hek293', 'subdir'],
                    help='Dataset layout type (default: miapaca2)')
parser.add_argument('--ch_dapi',   type=int, default=None, help='Override DAPI channel number')
parser.add_argument('--ch_lamp1',  type=int, default=None, help='Override LAMP1 channel number')
parser.add_argument('--ch_marker', type=int, default=None, help='Override marker channel number')
args = parser.parse_args()

IMAGE_DIR  = args.image_dir
OUTPUT_DIR = args.output_dir or os.path.join(IMAGE_DIR, 'coloc_results')
DATASET    = args.dataset
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Dataset configuration ─────────────────────────────────────────────────────
if DATASET == 'miapaca2':
    CH_DAPI, CH_LAMP1, CH_MARKER = 0, 1, 3
    MARKER_NAME = 'mTOR'
    CELL_LINE   = 'MiaPaca2'
elif DATASET == 'hek293':
    CH_DAPI, CH_LAMP1, CH_MARKER = 0, 1, 2
    MARKER_NAME = 'Raptor'
    CELL_LINE   = 'HEK293'
elif DATASET == 'subdir':
    CH_DAPI, CH_LAMP1, CH_MARKER = 0, 1, 2
    MARKER_NAME = 'Raptor'
    CELL_LINE   = 'HEK293'

# Allow CLI overrides
if args.ch_dapi   is not None: CH_DAPI   = args.ch_dapi
if args.ch_lamp1  is not None: CH_LAMP1  = args.ch_lamp1
if args.ch_marker is not None: CH_MARKER = args.ch_marker

print(f"Dataset:    {DATASET}")
print(f"Cell line:  {CELL_LINE}")
print(f"Marker:     {MARKER_NAME} (C{CH_MARKER})")
print(f"Image dir:  {IMAGE_DIR}")
print(f"Output dir: {OUTPUT_DIR}")

# ── Segmentation parameters ───────────────────────────────────────────────────
NUC_SIGMA         = 3
NUC_MIN_SIZE      = 2000
NUC_ERODE         = 4
NUC_EXPAND_PX     = 30
NUC_PEAK_DIST     = 40
PUNCTA_BG_SIGMA   = 20
PUNCTA_FG_SIGMA   = 1.5
PUNCTA_PERCENTILE = 85
PUNCTA_MIN_SIZE   = 8

# ── Core functions ────────────────────────────────────────────────────────────
def norm(img, plow=1, phigh=99.5):
    lo, hi = np.percentile(img, plow), np.percentile(img, phigh)
    return np.clip((img - lo) / (hi - lo + 1e-6), 0, 1)

def best_z(stack):
    if stack.ndim == 2:
        return None
    return int(np.argmax([np.var(stack[z]) for z in range(stack.shape[0])]))

def load_plane(path):
    stack = tifffile.imread(path).astype(np.float32)
    if stack.ndim == 3:
        return stack[best_z(stack)]
    return stack

def segment_nuclei(dapi):
    smooth = gaussian(dapi, sigma=NUC_SIGMA)
    thresh = filters.threshold_otsu(smooth)
    mask   = smooth > thresh
    mask   = morphology.remove_small_objects(mask, min_size=NUC_MIN_SIZE)
    mask   = ndi.binary_fill_holes(mask)
    mask   = morphology.binary_erosion(mask, morphology.disk(NUC_ERODE))
    dist   = ndi.distance_transform_edt(mask)
    peaks  = feature.peak_local_max(dist, min_distance=NUC_PEAK_DIST, labels=mask)
    pm     = np.zeros(dist.shape, dtype=bool)
    pm[tuple(peaks.T)] = True
    markers     = measure.label(pm)
    nuc_labels  = segmentation.watershed(-dist, markers, mask=mask)
    cell_mask   = morphology.dilation(mask, morphology.disk(NUC_EXPAND_PX))
    cell_labels = segmentation.watershed(-dist, markers, mask=cell_mask)
    return nuc_labels, cell_labels

def detect_puncta(channel):
    bg   = gaussian(channel, sigma=PUNCTA_BG_SIGMA)
    sub  = np.maximum(channel - bg * 0.85, 0)
    sm   = gaussian(sub, sigma=PUNCTA_FG_SIGMA)
    thr  = np.percentile(sm[sm > 0], PUNCTA_PERCENTILE)
    mask = sm > thr
    mask = morphology.remove_small_objects(mask, min_size=PUNCTA_MIN_SIZE)
    mask = morphology.remove_small_holes(mask, area_threshold=20)
    return mask, measure.label(mask)

def per_cell_stats(cell_labels, lamp1_mask, marker_mask, label, collection):
    overlap = lamp1_mask & marker_mask
    rows = []
    for cid in range(1, cell_labels.max() + 1):
        roi       = (cell_labels == cid)
        marker_px = int(np.sum(marker_mask & roi))
        lamp1_px  = int(np.sum(lamp1_mask  & roi))
        coloc_px  = int(np.sum(overlap     & roi))
        if marker_px < 50:
            continue
        rows.append({
            'label'     : label,
            'collection': collection,
            'cell_id'   : cid,
            'marker_px' : marker_px,
            'lamp1_px'  : lamp1_px,
            'coloc_px'  : coloc_px,
            'coloc_frac': coloc_px / marker_px,
        })
    return rows

# ── File discovery ────────────────────────────────────────────────────────────
# groups: label -> list of (c0_path, c1_path, cm_path, collection_id)

COLLECTION_FILE_RE = re.compile(r'.*_C(\d)\.tif$', re.IGNORECASE)

def find_triplet(file_list, folder):
    """Given a list of filenames in a folder, group by collection and return triplets."""
    # Group by everything before _C{n}.tif
    stems = defaultdict(dict)
    for f in file_list:
        m = COLLECTION_FILE_RE.match(f)
        if m:
            ch   = int(m.group(1))
            stem = f[:m.start(1)-1]   # everything before _C
            stems[stem][ch] = os.path.join(folder, f)
    triplets = []
    for stem, channels in stems.items():
        if all(ch in channels for ch in [CH_DAPI, CH_LAMP1, CH_MARKER]):
            triplets.append((
                channels[CH_DAPI],
                channels[CH_LAMP1],
                channels[CH_MARKER],
                os.path.basename(stem)   # collection id
            ))
    return sorted(triplets)

groups = defaultdict(list)   # label -> [(c0, c1, cm, collection_id), ...]

if DATASET == 'subdir':
    # Each subdirectory IS a condition
    for entry in sorted(os.listdir(IMAGE_DIR)):
        full = os.path.join(IMAGE_DIR, entry)
        if not os.path.isdir(full):
            continue
        if entry.startswith('.') or entry.startswith('coloc'):
            continue
        files = sorted(os.listdir(full))
        triplets = find_triplet(files, full)
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
            label = {'FED':'FED','ST':'STARVED'}[cond] + ' / ' + {'HG':'High Glucose','LG':'Low Glucose'}[gluc]
            stem  = f[:f.rfind('_C')]
            stems[(label, stem)][ch] = os.path.join(IMAGE_DIR, f)
    for (label, stem), channels in stems.items():
        if all(ch in channels for ch in [CH_DAPI, CH_LAMP1, CH_MARKER]):
            groups[label].append((channels[CH_DAPI], channels[CH_LAMP1],
                                   channels[CH_MARKER], stem))

elif DATASET == 'hek293':
    COND_RE = re.compile(
        r'(?:Slide\s*\d+_)?(DMSO|M6659|AZD8055)_(No_Refed|Refed).*_C(\d)\.tif$',
        re.IGNORECASE)
    DRUG_LABELS  = {'DMSO':'DMSO (Control)', 'M6659':'25 µM M6659', 'AZD8055':'100 nM AZD8055'}
    REFED_LABELS = {'NO_REFED':'No Refeeding', 'REFED':'Refed'}
    stems = defaultdict(dict)
    for f in sorted(os.listdir(IMAGE_DIR)):
        m = COND_RE.search(f)
        if m:
            drug  = m.group(1).upper()
            refed = m.group(2).upper().replace(' ','_')
            ch    = int(m.group(3))
            label = DRUG_LABELS.get(drug, drug) + ' / ' + REFED_LABELS.get(refed, refed)
            stem  = f[:f.rfind('_C')]
            stems[(label, stem)][ch] = os.path.join(IMAGE_DIR, f)
    for (label, stem), channels in stems.items():
        if all(ch in channels for ch in [CH_DAPI, CH_LAMP1, CH_MARKER]):
            groups[label].append((channels[CH_DAPI], channels[CH_LAMP1],
                                   channels[CH_MARKER], stem))

print(f"\nConditions found: {len(groups)}")
for label, trips in sorted(groups.items()):
    print(f"  {label}: {len(trips)} field(s) of view")

# ── Main loop ─────────────────────────────────────────────────────────────────
all_rows = []
panels   = {}   # label -> aggregated data for summary figure

# Color palette — cycles for >2 conditions
PALETTE = ['#D4845A','#4A9BAF','#7B9E6B','#A07BC4','#C4A07B','#6B9EA0','#C47B7B','#7B7BC4']

def condition_color(label, label_list):
    return PALETTE[label_list.index(label) % len(PALETTE)]

label_list = sorted(groups.keys())

for label in label_list:
    triplets = groups[label]
    print(f"\n══ {label} ({len(triplets)} field(s)) ══")

    cond_fracs = []
    panel_imgs  = {'c0':[], 'c1':[], 'cm':[], 'lamp1':[], 'marker':[], 'overlap':[]}

    for c0_path, c1_path, cm_path, collection_id in triplets:
        print(f"  Collection: {collection_id}")
        c0 = load_plane(c0_path)
        c1 = load_plane(c1_path)
        cm = load_plane(cm_path)

        nuc_labels, cell_labels = segment_nuclei(c0)
        lamp1_mask,  _          = detect_puncta(c1)
        marker_mask, _          = detect_puncta(cm)

        rows = per_cell_stats(cell_labels, lamp1_mask, marker_mask, label, collection_id)
        all_rows.extend(rows)
        fracs = [r['coloc_frac'] for r in rows]
        cond_fracs.extend(fracs)

        print(f"    Nuclei: {nuc_labels.max()}  |  "
              f"LAMP1: {lamp1_mask.sum()//1000}k px  |  "
              f"{MARKER_NAME}: {marker_mask.sum()//1000}k px  |  "
              f"Cells: {len(fracs)}  |  "
              f"Mean coloc: {np.mean(fracs)*100:.1f}%" if fracs else "    No cells with signal")

        # Store first collection's images for panel figure
        if len(panel_imgs['c0']) == 0:
            panel_imgs['c0'].append(norm(c0))
            panel_imgs['c1'].append(norm(c1))
            panel_imgs['cm'].append(norm(cm))
            panel_imgs['lamp1'].append(lamp1_mask)
            panel_imgs['marker'].append(marker_mask)
            panel_imgs['overlap'].append(lamp1_mask & marker_mask)

        # ── Per-collection panel ────────────────────────────────────────────
        S = 512
        H, W = c0.shape
        sl = np.s_[H//2-S:H//2+S, W//2-S:W//2+S]

        fig, axes = plt.subplots(2, 4, figsize=(18, 9), facecolor='#F8F7F4')
        fig.suptitle(f'{CELL_LINE}  |  {label}  |  {collection_id}',
                     color='#1E2D3A', fontsize=13, fontweight='bold')

        c0c = c0[sl]; c1c = c1[sl]; cmc = cm[sl]
        nucc = nuc_labels[sl]; l1c = lamp1_mask[sl]
        mtc  = marker_mask[sl]; ovc = (lamp1_mask & marker_mask)[sl]

        for ax, ch, title, cmap in zip(
            axes[0], [c0c, c1c, cmc],
            [f'C{CH_DAPI} DAPI', f'C{CH_LAMP1} LAMP1', f'C{CH_MARKER} {MARKER_NAME}'],
            ['Blues', 'Greens', 'Reds']):
            ax.imshow(norm(ch), cmap=cmap)
            ax.set_title(title, color='#1E2D3A', fontsize=10)
            ax.axis('off')

        ax_nuc = axes[0, 3]
        ax_nuc.imshow(norm(c0c), cmap='gray', vmin=0, vmax=0.4)
        nb = segmentation.find_boundaries(nucc, mode='outer')
        ov_r = np.zeros((*c0c.shape, 4)); ov_r[nb, 0] = 1; ov_r[nb, 3] = 0.9
        ax_nuc.imshow(ov_r)
        for reg in measure.regionprops(nucc):
            ry, rx = reg.centroid
            if 0 < ry < 1024 and 0 < rx < 1024:
                ax_nuc.text(rx, ry, str(reg.label), color='#FFFFFF', fontsize=8,
                            ha='center', va='center', fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.1', facecolor='#4A9BAF',
                                      alpha=0.85, linewidth=0))
        ax_nuc.set_title('Nuclei', color='#1E2D3A', fontsize=10); ax_nuc.axis('off')

        axes[1, 0].imshow(norm(c1c), cmap='Greens', vmin=0, vmax=0.8)
        ov1 = np.zeros((*c1c.shape, 4)); ov1[l1c,1]=1; ov1[l1c,2]=0.4; ov1[l1c,3]=0.55
        axes[1, 0].imshow(ov1)
        axes[1, 0].set_title(f'LAMP1 puncta', color='#1E2D3A', fontsize=10); axes[1, 0].axis('off')

        axes[1, 1].imshow(norm(cmc), cmap='Reds', vmin=0, vmax=0.8)
        ov2 = np.zeros((*cmc.shape, 4)); ov2[mtc,0]=1; ov2[mtc,1]=0.3; ov2[mtc,3]=0.55
        axes[1, 1].imshow(ov2)
        axes[1, 1].set_title(f'{MARKER_NAME} puncta', color='#1E2D3A', fontsize=10); axes[1, 1].axis('off')

        rgb = np.zeros((*c1c.shape, 3))
        rgb[:,:,1] = norm(c1c)*0.9; rgb[:,:,0] = norm(cmc)*0.9
        rgb[ovc,0]=1; rgb[ovc,1]=1; rgb[ovc,2]=0
        axes[1, 2].imshow(np.clip(rgb,0,1))
        axes[1, 2].legend(handles=[
            mpatches.Patch(color='yellow',  label='Colocalized'),
            mpatches.Patch(color='#00AA44', label='LAMP1 only'),
            mpatches.Patch(color='#CC3300', label=f'{MARKER_NAME} only'),
        ], loc='lower right', fontsize=7, facecolor='#FFFFFF', labelcolor='#1E2D3A')
        axes[1, 2].set_title('Colocalization', color='#1E2D3A', fontsize=10); axes[1, 2].axis('off')

        col = condition_color(label, label_list)
        ax_bar = axes[1, 3]; ax_bar.set_facecolor('#F0F4F6')
        if fracs:
            ax_bar.bar(range(1, len(fracs)+1), [f*100 for f in fracs],
                       color=col, edgecolor='none')
            ax_bar.axhline(np.mean(fracs)*100, color='#2C7A8C', linestyle='--', linewidth=1.2,
                           label=f'Mean {np.mean(fracs)*100:.1f}%')
            ax_bar.legend(fontsize=8, facecolor='#FFFFFF', labelcolor='#1E2D3A')
        ax_bar.set_ylim(0, 80)
        ax_bar.set_xlabel('Cell', color='#1E2D3A', fontsize=8)
        ax_bar.set_ylabel(f'% {MARKER_NAME}/LAMP1', color='#1E2D3A', fontsize=8)
        ax_bar.tick_params(colors='#1E2D3A', labelsize=7)
        for sp in ['top','right']: ax_bar.spines[sp].set_visible(False)
        for sp in ['bottom','left']: ax_bar.spines[sp].set_color('#A0B8C4')
        ax_bar.set_title('Per-cell coloc %', color='#1E2D3A', fontsize=10)
        for ax in axes.flat: ax.set_facecolor('#F8F7F4')

        plt.tight_layout()
        safe = re.sub(r'[^\w\-]', '_', label) + f'_{collection_id}'
        out  = os.path.join(OUTPUT_DIR, f'{safe}_panel.png')
        plt.savefig(out, dpi=120, bbox_inches='tight', facecolor='#F8F7F4')
        plt.close()

    # Condition-level summary
    panels[label] = {
        'fracs' : cond_fracs,
        'label' : label,
        'color' : condition_color(label, label_list),
        'c0'    : panel_imgs['c0'][0] if panel_imgs['c0'] else None,
        'c1'    : panel_imgs['c1'][0] if panel_imgs['c1'] else None,
        'cm'    : panel_imgs['cm'][0] if panel_imgs['cm'] else None,
        'lamp1' : panel_imgs['lamp1'][0] if panel_imgs['lamp1'] else None,
        'marker': panel_imgs['marker'][0] if panel_imgs['marker'] else None,
        'overlap':panel_imgs['overlap'][0] if panel_imgs['overlap'] else None,
    }

    if cond_fracs:
        print(f"  ► Condition total — n={len(cond_fracs)} cells  "
              f"mean={np.mean(cond_fracs)*100:.1f}%  "
              f"median={np.median(cond_fracs)*100:.1f}%  "
              f"std={np.std(cond_fracs)*100:.1f}%")

# ── CSV ───────────────────────────────────────────────────────────────────────
df = pd.DataFrame(all_rows)
csv_path = os.path.join(OUTPUT_DIR, 'coloc_summary.csv')
df.to_csv(csv_path, index=False)
print(f"\nCSV: {csv_path}")
if not df.empty:
    print(df.groupby('label')['coloc_frac'].agg(['count','mean','median','std']).round(3))

# ── Summary figure ────────────────────────────────────────────────────────────
n_cond = len(label_list)
cols   = min(n_cond, 3)
rows   = (n_cond + cols - 1) // cols

fig = plt.figure(figsize=(22, 6 + rows * 5), facecolor='#F8F7F4')
fig.suptitle(f'{CELL_LINE} — {MARKER_NAME}–Lysosome Colocalization\nAll Conditions',
             color='#1E2D3A', fontsize=14, fontweight='bold', y=0.99)

S = 320; img_w = 0.16; img_h = 0.26; bar_h = 0.07
lm = 0.02; gap = 0.015; top = 0.90

for idx, label in enumerate(label_list):
    p   = panels[label]
    col = idx % cols
    row = idx // cols
    lft = lm + col * (img_w + gap)
    bot = top - row * (img_h + bar_h + 0.07)

    if p['c1'] is not None:
        H, W = p['c1'].shape
        sl   = np.s_[H//2-S:H//2+S, W//2-S:W//2+S]
        c1c  = p['c1'][sl]; cmc = p['cm'][sl]; ovc = p['overlap'][sl]
        rgb  = np.zeros((*c1c.shape, 3))
        rgb[:,:,1] = norm(c1c)*0.9; rgb[:,:,0] = norm(cmc)*0.9
        rgb[ovc,0]=1; rgb[ovc,1]=1; rgb[ovc,2]=0
        ax_img = fig.add_axes([lft, bot, img_w, img_h])
        ax_img.imshow(np.clip(rgb,0,1)); ax_img.axis('off')
        ax_img.set_title(label, color='#1E2D3A', fontsize=8, fontweight='bold', pad=3)
        fracs    = p['fracs']
        mean_pct = np.mean(fracs)*100 if fracs else 0
        n        = len(fracs)
        ax_img.text(0.97, 0.97, f'{mean_pct:.1f}%\nn={n}',
                    transform=ax_img.transAxes, color='#FFFFFF', fontsize=10,
                    fontweight='bold', ha='right', va='top',
                    bbox=dict(boxstyle='round,pad=0.25', facecolor=p['color'], alpha=0.9, linewidth=0))

        ax_b = fig.add_axes([lft, bot - bar_h - 0.005, img_w, bar_h])
        ax_b.set_facecolor('#F0F4F6')
        if fracs:
            ax_b.bar(range(1,n+1), [f*100 for f in fracs], color=p['color'], edgecolor='none', width=0.7)
            ax_b.axhline(mean_pct, color='#2C7A8C', linestyle='--', linewidth=1)
        ax_b.set_ylim(0,80); ax_b.tick_params(colors='#1E2D3A', labelsize=5)
        ax_b.set_ylabel('%', color='#1E2D3A', fontsize=6)
        for sp in ['top','right']: ax_b.spines[sp].set_visible(False)
        for sp in ['bottom','left']: ax_b.spines[sp].set_color('#A0B8C4')

# Main bar chart right side
ax_main = fig.add_axes([0.60, 0.08, 0.37, 0.84])
ax_main.set_facecolor('#F0F4F6')
means, sems, xlabels, colors = [], [], [], []
for label in label_list:
    fracs = panels[label]['fracs']
    if not fracs: continue
    means.append(np.mean(fracs)*100)
    sems.append(np.std(fracs)/np.sqrt(len(fracs))*100)
    xlabels.append(label.replace(' / ','\n').replace(' ','\n',1))
    colors.append(panels[label]['color'])

x = np.arange(len(means))
ax_main.bar(x, means, yerr=sems, color=colors, edgecolor='none', width=0.55,
            capsize=5, error_kw={'ecolor':'#1E2D3A','linewidth':1.5})
ax_main.set_xticks(x)
ax_main.set_xticklabels(xlabels, color='#1E2D3A', fontsize=8)
ax_main.set_ylabel(f'% {MARKER_NAME} colocalized with LAMP1\n(mean ± SEM per cell)',
                   color='#1E2D3A', fontsize=10)
ax_main.set_ylim(0, 80)
ax_main.set_title(f'{MARKER_NAME}–Lysosome Recruitment', color='#1E2D3A',
                  fontsize=12, fontweight='bold', pad=8)
ax_main.tick_params(colors='#1E2D3A', labelsize=8)
for sp in ['top','right']: ax_main.spines[sp].set_visible(False)
for sp in ['bottom','left']: ax_main.spines[sp].set_color('#A0B8C4')
for xi,(m,s) in enumerate(zip(means,sems)):
    ax_main.text(xi, m+s+1, f'{m:.1f}%', ha='center', va='bottom',
                 color='#1E2D3A', fontsize=9, fontweight='bold')

summary_path = os.path.join(OUTPUT_DIR, 'coloc_summary.png')
plt.savefig(summary_path, dpi=130, bbox_inches='tight', facecolor='#F8F7F4')
plt.close()
print(f"\nSummary: {summary_path}")
print(f"Done. Outputs in: {OUTPUT_DIR}")