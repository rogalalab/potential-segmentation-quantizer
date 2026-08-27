"""
Object-Based Colocalization Analysis — Rogala Lab v4
Imports shared functions from utils.py

New in v4:
  - Imports from utils.py (no more duplicate code)
  - Puncta morphology: area, intensity, elongation per punctum
  - Nuclear/cytoplasmic ratio (--nuclear_cyto flag)
  - Spatial distribution: perinuclear vs peripheral (--spatial flag)
  - ch1_name / marker_name read from manifest

Usage:
  python object_coloc.py --manifest ~/Desktop/manifest.tsv
  python object_coloc.py --manifest ~/Desktop/manifest.tsv --nuclear_cyto --spatial
  python object_coloc.py --image_dir ~/Desktop/071526 --dataset subdir
"""

import os, re, argparse, warnings, sys
from collections import defaultdict
import numpy as np
import pandas as pd
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage import measure, segmentation
from skimage.morphology import disk

warnings.filterwarnings('ignore')

# Import shared utilities
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (load_plane, segment_nuclei, detect_puncta_objects,
                   local_median_subtract, object_colocalization,
                   nuclear_cytoplasmic_ratio, spatial_distribution,
                   spatial_summary, puncta_morphology, norm, PALETTE)

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Object-based colocalization v4')
parser.add_argument('--image_dir',       type=str, default=None)
parser.add_argument('--output_dir',      type=str, default=None)
parser.add_argument('--manifest',        type=str, default=None,
                    help='Path to manifest TSV from ingest.py')
parser.add_argument('--dataset',         type=str, default='subdir',
                    choices=['miapaca2','hek293','subdir'])
parser.add_argument('--projection',      type=str, default='max',
                    choices=['best_z','max','mean'])
parser.add_argument('--residual_factor', type=float, default=1.0)
parser.add_argument('--window',          type=int,   default=32)
parser.add_argument('--min_puncta_px',   type=int,   default=5)
parser.add_argument('--proximity_px',    type=int,   default=3)
parser.add_argument('--lamp1_percentile',  type=float, default=82)
parser.add_argument('--marker_percentile', type=float, default=93)
parser.add_argument('--puncta_percentile', type=float, default=None,
                    help='Sets both percentiles to same value')
parser.add_argument('--nuclear_cyto', action='store_true',
                    help='Compute nuclear/cytoplasmic ratio for both channels')
parser.add_argument('--spatial', action='store_true',
                    help='Compute spatial distribution (perinuclear vs peripheral)')
parser.add_argument('--ch_dapi',   type=int, default=None)
parser.add_argument('--ch_lamp1',  type=int, default=None)
parser.add_argument('--ch_marker', type=int, default=None)
args = parser.parse_args()

# Resolve percentiles
LAMP1_PCT  = args.puncta_percentile if args.puncta_percentile else args.lamp1_percentile
MARKER_PCT = args.puncta_percentile if args.puncta_percentile else args.marker_percentile

# ── Dataset config (used when no manifest) ────────────────────────────────────
if args.manifest:
    IMAGE_DIR = args.image_dir  # may be None
else:
    IMAGE_DIR = args.image_dir
    if not IMAGE_DIR:
        print("❌ Must provide --image_dir or --manifest"); exit(1)

OUTPUT_DIR = args.output_dir or (
    os.path.join(IMAGE_DIR, 'object_coloc_results') if IMAGE_DIR
    else os.path.join(os.path.dirname(args.manifest), 'object_coloc_results'))
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Defaults — overridden by manifest
CELL_LINE   = 'HEK293'
MARKER_NAME = 'Raptor'
CH1_NAME    = 'LAMP1'
CH_DAPI     = args.ch_dapi   or 0
CH_LAMP1    = args.ch_lamp1  or 1
CH_MARKER   = args.ch_marker or 2
DATASET     = args.dataset

print(f"Projection:       {args.projection}")
print(f"Residual factor:  {args.residual_factor}")
print(f"Proximity:        {args.proximity_px}px")
print(f"C1 percentile:    {LAMP1_PCT}")
print(f"Marker percentile:{MARKER_PCT}")
print(f"Nuclear/Cyto:     {args.nuclear_cyto}")
print(f"Spatial:          {args.spatial}")
print(f"Output:           {OUTPUT_DIR}\n")

# ── File discovery ────────────────────────────────────────────────────────────
COLLECTION_FILE_RE = re.compile(r'.*_C(\d)\.tif$', re.IGNORECASE)

def find_triplet(file_list, folder):
    stems = defaultdict(dict)
    for f in file_list:
        m = COLLECTION_FILE_RE.match(f)
        if m:
            ch = int(m.group(1)); stem = f[:m.start(1)-1]
            stems[stem][ch] = os.path.join(folder, f)
    return sorted(
        (channels[CH_DAPI], channels[CH_LAMP1], channels[CH_MARKER],
         os.path.basename(stem))
        for stem, channels in stems.items()
        if all(ch in channels for ch in [CH_DAPI, CH_LAMP1, CH_MARKER]))

groups = defaultdict(list)

if args.manifest:
    print(f"Loading manifest: {args.manifest}")
    mdf = pd.read_csv(args.manifest, sep='\t')
    for col in ['label','collection','ch_dapi','ch_lamp1','ch_marker']:
        if col not in mdf.columns:
            print(f"❌ Manifest missing column: {col}"); exit(1)
    for _, row in mdf.iterrows():
        groups[row['label']].append((row['ch_dapi'], row['ch_lamp1'],
                                     row['ch_marker'], row['collection']))
    if 'cell_line'   in mdf.columns: CELL_LINE   = mdf['cell_line'].iloc[0]
    if 'marker_name' in mdf.columns: MARKER_NAME = mdf['marker_name'].iloc[0]
    if 'ch1_name'    in mdf.columns: CH1_NAME    = mdf['ch1_name'].iloc[0]
    print(f"  {len(mdf)} collection(s), {len(groups)} condition(s)")
    print(f"  Cell line: {CELL_LINE}  |  C1: {CH1_NAME}  |  Marker: {MARKER_NAME}")

elif DATASET == 'subdir':
    for entry in sorted(os.listdir(IMAGE_DIR)):
        full = os.path.join(IMAGE_DIR, entry)
        if not os.path.isdir(full) or entry.startswith('.') or 'result' in entry.lower():
            continue
        trips = find_triplet(sorted(os.listdir(full)), full)
        if trips: groups[entry].extend(trips); print(f"  {entry}: {len(trips)} collection(s)")

elif DATASET == 'miapaca2':
    COND_RE = re.compile(r'MiaPaca2_(FED|ST)_(HG|LG).*_C(\d)\.tif$', re.IGNORECASE)
    stems = defaultdict(dict)
    for f in sorted(os.listdir(IMAGE_DIR)):
        m = COND_RE.search(f)
        if m:
            cond,gluc,ch = m.group(1).upper(),m.group(2).upper(),int(m.group(3))
            label = {'FED':'FED','ST':'STARVED'}[cond]+' / '+{'HG':'High Glucose','LG':'Low Glucose'}[gluc]
            stems[(label,f[:f.rfind('_C')])][ch] = os.path.join(IMAGE_DIR,f)
    for (label,stem),channels in stems.items():
        if all(ch in channels for ch in [CH_DAPI,CH_LAMP1,CH_MARKER]):
            groups[label].append((channels[CH_DAPI],channels[CH_LAMP1],channels[CH_MARKER],stem))

elif DATASET == 'hek293':
    COND_RE = re.compile(r'(?:Slide\s*\d+_)?(DMSO|M6659|AZD8055)_(No_Refed|Refed).*_C(\d)\.tif$', re.IGNORECASE)
    DL = {'DMSO':'DMSO','M6659':'25µM M6659','AZD8055':'100nM AZD8055'}
    RL = {'NO_REFED':'No Refed','REFED':'Refed'}
    stems = defaultdict(dict)
    for f in sorted(os.listdir(IMAGE_DIR)):
        m = COND_RE.search(f)
        if m:
            drug=m.group(1).upper(); refed=m.group(2).upper().replace(' ','_'); ch=int(m.group(3))
            label=DL.get(drug,drug)+' / '+RL.get(refed,refed)
            stems[(label,f[:f.rfind('_C')])][ch]=os.path.join(IMAGE_DIR,f)
    for (label,stem),channels in stems.items():
        if all(ch in channels for ch in [CH_DAPI,CH_LAMP1,CH_MARKER]):
            groups[label].append((channels[CH_DAPI],channels[CH_LAMP1],channels[CH_MARKER],stem))

label_list = sorted(groups.keys())
print(f"\nConditions: {len(groups)}")

# ── Main loop ─────────────────────────────────────────────────────────────────
all_rows    = []
cond_summary = {}

for idx, label in enumerate(label_list):
    triplets = groups[label]
    print(f"\n══ {label} ({len(triplets)} field(s)) ══", flush=True)
    cond_fracs = []

    for c0_path, c1_path, cm_path, collection_id in triplets:
        print(f"  Collection: {collection_id}", flush=True)
        print(f"    [1/4] Loading...", flush=True)
        try:
            c0 = load_plane(c0_path, args.projection)
            c1 = load_plane(c1_path, args.projection)
            cm = load_plane(cm_path, args.projection)
        except Exception as e:
            print(f"    ⚠️  SKIPPING — {e}", flush=True); continue

        print(f"    [2/4] Segmenting...", end=' ', flush=True)
        nuc_labels, cell_labels = segment_nuclei(c0)
        print(f"done ({nuc_labels.max()} nuclei)", flush=True)

        print(f"    [3/4] Background subtraction...", end=' ', flush=True)
        c1_sub = local_median_subtract(c1, window=args.window,
                                       residual_factor=args.residual_factor)
        cm_sub = local_median_subtract(cm, window=args.window,
                                       residual_factor=args.residual_factor)
        print("done", flush=True)

        print(f"    [4/4] Per-cell analysis...", end=' ', flush=True)
        rows = []

        for cid in range(1, cell_labels.max() + 1):
            cell_roi = (cell_labels == cid)
            nuc_roi  = (nuc_labels  == cid)
            cyto_roi = cell_roi & ~nuc_roi

            if cyto_roi.sum() < 100: continue

            # ── Puncta detection ──────────────────────────────────────────────
            ref_lbl,   ref_props   = detect_puncta_objects(
                c1_sub, cyto_roi, args.min_puncta_px, LAMP1_PCT)
            query_lbl, query_props = detect_puncta_objects(
                cm_sub, cyto_roi, args.min_puncta_px, MARKER_PCT)

            if not query_props: continue

            # ── Colocalization ────────────────────────────────────────────────
            coloc_mask, n_coloc, _ = object_colocalization(
                ref_lbl, query_lbl, query_props, args.proximity_px)
            frac = n_coloc / len(query_props)

            # ── Puncta morphology ─────────────────────────────────────────────
            ref_morph   = puncta_morphology(ref_props)
            query_morph = puncta_morphology(query_props)

            row = {
                'label'             : label,
                'collection'        : collection_id,
                'cell_id'           : cid,
                'n_ref_puncta'      : ref_morph['n_puncta'],
                'n_marker_puncta'   : query_morph['n_puncta'],
                'n_coloc_puncta'    : n_coloc,
                'coloc_frac'        : frac,
                # Reference channel morphology
                'ref_mean_area'     : ref_morph['mean_area'],
                'ref_mean_intensity': ref_morph['mean_intensity'],
                'ref_elongation'    : ref_morph['mean_elongation'],
                # Marker channel morphology
                'marker_mean_area'     : query_morph['mean_area'],
                'marker_mean_intensity': query_morph['mean_intensity'],
                'marker_elongation'    : query_morph['mean_elongation'],
                'cyto_px'           : int(cyto_roi.sum()),
                'residual_factor'   : args.residual_factor,
                'proximity_px'      : args.proximity_px,
                'lamp1_percentile'  : LAMP1_PCT,
                'marker_percentile' : MARKER_PCT,
                'projection'        : args.projection,
            }

            # ── Nuclear / cytoplasmic ratio ───────────────────────────────────
            if args.nuclear_cyto and nuc_roi.sum() > 50:
                nc_ref    = nuclear_cytoplasmic_ratio(c1, nuc_roi, cyto_roi)
                nc_marker = nuclear_cytoplasmic_ratio(cm, nuc_roi, cyto_roi)
                row.update({
                    'ref_nuc_frac'   : nc_ref['nuc_frac'],
                    'ref_nc_ratio'   : nc_ref['nc_ratio'],
                    'marker_nuc_frac': nc_marker['nuc_frac'],
                    'marker_nc_ratio': nc_marker['nc_ratio'],
                })

            # ── Spatial distribution ──────────────────────────────────────────
            if args.spatial:
                # Reference channel spatial distribution
                ref_dists   = spatial_distribution(ref_props,   nuc_labels, cell_labels, cid)
                query_dists = spatial_distribution(query_props, nuc_labels, cell_labels, cid)
                ref_spat    = spatial_summary(ref_dists)
                query_spat  = spatial_summary(query_dists)
                row.update({
                    'ref_mean_dist'      : ref_spat['mean_dist'],
                    'ref_frac_perinuc'   : ref_spat['frac_perinuc'],
                    'ref_frac_periph'    : ref_spat['frac_periph'],
                    'marker_mean_dist'   : query_spat['mean_dist'],
                    'marker_frac_perinuc': query_spat['frac_perinuc'],
                    'marker_frac_periph' : query_spat['frac_periph'],
                })

            rows.append(row)

        all_rows.extend(rows)
        fracs = [r['coloc_frac'] for r in rows]
        cond_fracs.extend(fracs)
        print(f"done — {len(rows)} cells  "
              f"mean {np.mean(fracs)*100:.1f}%" if fracs else "done — no cells",
              flush=True)

        # ── Per-collection panel ──────────────────────────────────────────────
        S = 512; H, W = c0.shape
        sl = np.s_[H//2-S:H//2+S, W//2-S:W//2+S]
        c1c = c1_sub[sl]; cmc = cm_sub[sl]
        nucc = nuc_labels[sl]; cellc = cell_labels[sl]

        # Build object overlays for crop
        ref_obj = np.zeros(c1c.shape, dtype=bool)
        mrkr_obj= np.zeros(cmc.shape, dtype=bool)
        col_obj = np.zeros(cmc.shape, dtype=bool)
        for cid2 in range(1, cell_labels.max()+1):
            cyto2 = (cell_labels==cid2)&~(nuc_labels==cid2)
            rl,_  = detect_puncta_objects(c1_sub, cyto2, args.min_puncta_px, LAMP1_PCT)
            ml,mp = detect_puncta_objects(cm_sub, cyto2, args.min_puncta_px, MARKER_PCT)
            ref_obj  |= (rl>0)[sl]; mrkr_obj |= (ml>0)[sl]
            if mp:
                cm2,_,_ = object_colocalization(rl, ml, mp, args.proximity_px)
                col_obj |= cm2[sl]

        fig, axes = plt.subplots(1, 4, figsize=(18, 5), facecolor='#F8F7F4')
        fig.suptitle(f'{CELL_LINE}  |  {label}  |  {collection_id}',
                     color='#1E2D3A', fontsize=12, fontweight='bold')

        axes[0].imshow(norm(c1c), cmap='Greens', vmin=0, vmax=0.8)
        ov1=np.zeros((*c1c.shape,4)); ov1[ref_obj,1]=1; ov1[ref_obj,2]=0.3; ov1[ref_obj,3]=0.7
        axes[0].imshow(ov1)
        axes[0].set_title(f'{CH1_NAME} puncta', color='#1E2D3A', fontsize=9); axes[0].axis('off')

        axes[1].imshow(norm(cmc), cmap='Reds', vmin=0, vmax=0.8)
        ov2=np.zeros((*cmc.shape,4)); ov2[mrkr_obj,0]=1; ov2[mrkr_obj,1]=0.3; ov2[mrkr_obj,3]=0.7
        axes[1].imshow(ov2)
        axes[1].set_title(f'{MARKER_NAME} puncta', color='#1E2D3A', fontsize=9); axes[1].axis('off')

        rgb=np.zeros((*c1c.shape,3))
        rgb[:,:,1]=norm(c1c)*0.6; rgb[:,:,0]=norm(cmc)*0.6
        rgb[ref_obj&~col_obj,1]=0.8; rgb[ref_obj&~col_obj,0]=0
        rgb[mrkr_obj&~col_obj,0]=0.8; rgb[mrkr_obj&~col_obj,1]=0
        rgb[col_obj,0]=1; rgb[col_obj,1]=1; rgb[col_obj,2]=0
        axes[2].imshow(np.clip(rgb,0,1))
        axes[2].set_title(f'Colocalization', color='#1E2D3A', fontsize=9)
        axes[2].legend(handles=[
            mpatches.Patch(color='yellow', label=f'{MARKER_NAME} on {CH1_NAME}'),
            mpatches.Patch(color='#00CC44', label=f'{CH1_NAME} only'),
            mpatches.Patch(color='#CC2200', label=f'{MARKER_NAME} only'),
        ], loc='lower right', fontsize=6, facecolor='white', labelcolor='#1E2D3A')
        axes[2].axis('off')

        col_color = PALETTE[idx % len(PALETTE)]
        ax_bar = axes[3]; ax_bar.set_facecolor('#F0F4F6')
        if fracs:
            ax_bar.bar(range(1,len(fracs)+1),[f*100 for f in fracs],
                       color=col_color, edgecolor='none')
            ax_bar.axhline(np.mean(fracs)*100, color='#2C7A8C',
                           linestyle='--', linewidth=1.2,
                           label=f'Mean {np.mean(fracs)*100:.1f}%')
            ax_bar.legend(fontsize=8, facecolor='white', labelcolor='#1E2D3A')
        ax_bar.set_ylim(0,100)
        ax_bar.set_xlabel('Cell', color='#1E2D3A', fontsize=8)
        ax_bar.set_ylabel(f'% {MARKER_NAME} on {CH1_NAME}', color='#1E2D3A', fontsize=8)
        ax_bar.set_title('Per-cell coloc %', color='#1E2D3A', fontsize=9)
        ax_bar.tick_params(colors='#1E2D3A', labelsize=7)
        for sp in ['top','right']: ax_bar.spines[sp].set_visible(False)
        for sp in ['bottom','left']: ax_bar.spines[sp].set_color('#A0B8C4')
        for ax in axes: ax.set_facecolor('#F8F7F4')
        plt.tight_layout()
        safe = re.sub(r'[^\w\-]','_',label)
        plt.savefig(os.path.join(OUTPUT_DIR,f'{safe}_{collection_id}_panel.png'),
                    dpi=110, bbox_inches='tight', facecolor='#F8F7F4')
        plt.close()

    cond_summary[label] = {'fracs':cond_fracs, 'color':PALETTE[idx%len(PALETTE)]}
    if cond_fracs:
        print(f"\n  ► {label}  n={len(cond_fracs)} cells  "
              f"mean={np.mean(cond_fracs)*100:.1f}%  "
              f"median={np.median(cond_fracs)*100:.1f}%")

# ── CSV ───────────────────────────────────────────────────────────────────────
df = pd.DataFrame(all_rows)
csv_path = os.path.join(OUTPUT_DIR, 'object_coloc_summary.csv')
df.to_csv(csv_path, index=False)
print(f"\nCSV: {csv_path}")
if not df.empty:
    print(df.groupby('label')['coloc_frac']
            .agg(['count','mean','median','std'])
            .round(3).sort_values('mean', ascending=False))

# ── Summary figure ────────────────────────────────────────────────────────────
n_cond = len(label_list)
cols   = min(n_cond, 3); rows_fig = (n_cond+cols-1)//cols

fig = plt.figure(figsize=(22, 6+rows_fig*5), facecolor='#F8F7F4')
fig.suptitle(f'{CELL_LINE} — {MARKER_NAME} on {CH1_NAME} (Object-Based)\n'
             f'rf={args.residual_factor}  proximity={args.proximity_px}px  '
             f'{CH1_NAME}_pct={LAMP1_PCT}  {MARKER_NAME}_pct={MARKER_PCT}',
             color='#1E2D3A', fontsize=13, fontweight='bold', y=0.99)

S=320; lm=0.02; gap=0.015; img_w=0.16; img_h=0.26; bar_h=0.07; top=0.90

for i2, label in enumerate(label_list):
    p = cond_summary[label]
    col2 = i2 % cols; row2 = i2 // cols
    lft = lm + col2*(img_w+gap); bot = top - row2*(img_h+bar_h+0.07)
    fracs = p['fracs']
    if fracs:
        ax_img = fig.add_axes([lft, bot, img_w, img_h])
        ax_img.text(0.5, 0.5, f"{np.mean(fracs)*100:.1f}%\nn={len(fracs)}",
                    transform=ax_img.transAxes, ha='center', va='center',
                    color='white', fontsize=14, fontweight='bold',
                    bbox=dict(facecolor=p['color'], alpha=0.85, pad=8, linewidth=0))
        ax_img.set_facecolor(p['color']+'33')
        ax_img.set_title(label.replace(' / ','\n'), color='#1E2D3A', fontsize=8, fontweight='bold')
        ax_img.axis('off')

        ax_b = fig.add_axes([lft, bot-bar_h-0.005, img_w, bar_h])
        ax_b.set_facecolor('#F0F4F6')
        ax_b.bar(range(1,len(fracs)+1),[f*100 for f in fracs],
                 color=p['color'], edgecolor='none', width=0.7)
        ax_b.axhline(np.mean(fracs)*100, color='#2C7A8C', linestyle='--', linewidth=1)
        ax_b.set_ylim(0,100); ax_b.tick_params(colors='#1E2D3A', labelsize=5)
        ax_b.set_ylabel('%', color='#1E2D3A', fontsize=6)
        for sp in ['top','right']: ax_b.spines[sp].set_visible(False)
        for sp in ['bottom','left']: ax_b.spines[sp].set_color('#A0B8C4')

# Main bar chart
ax_main = fig.add_axes([0.58, 0.08, 0.40, 0.84])
ax_main.set_facecolor('#F0F4F6')
means=[]; sems=[]; xlabels=[]; colors=[]
for label in label_list:
    fracs=cond_summary[label]['fracs']
    if not fracs: continue
    means.append(np.mean(fracs)*100)
    sems.append(np.std(fracs)/np.sqrt(len(fracs))*100)
    xlabels.append(label.replace(' / ','\n'))
    colors.append(cond_summary[label]['color'])
x=np.arange(len(means))
ax_main.bar(x, means, yerr=sems, color=colors, edgecolor='none', width=0.55,
            capsize=5, error_kw={'ecolor':'#1E2D3A','linewidth':1.5})
ax_main.set_xticks(x); ax_main.set_xticklabels(xlabels,color='#1E2D3A',fontsize=8,rotation=25,ha='right')
ax_main.set_ylabel(f'% {MARKER_NAME} puncta on {CH1_NAME}\n(mean ± SEM)',color='#1E2D3A',fontsize=10)
ax_main.set_ylim(0,min(100,max(means)*1.3) if means else 100)
ax_main.set_title(f'{MARKER_NAME}–{CH1_NAME} Recruitment by Condition',color='#1E2D3A',fontsize=12,fontweight='bold',pad=8)
ax_main.tick_params(colors='#1E2D3A',labelsize=8)
for sp in ['top','right']: ax_main.spines[sp].set_visible(False)
for sp in ['bottom','left']: ax_main.spines[sp].set_color('#A0B8C4')
for xi,(m,s) in enumerate(zip(means,sems)):
    ax_main.text(xi,m+s+0.8,f'{m:.1f}%',ha='center',va='bottom',
                 color='#1E2D3A',fontsize=8,fontweight='bold')

summary_path = os.path.join(OUTPUT_DIR,'object_coloc_summary.png')
plt.savefig(summary_path, dpi=130, bbox_inches='tight', facecolor='#F8F7F4')
plt.close()
print(f"\nSummary: {summary_path}")
print(f"Done. All outputs in: {OUTPUT_DIR}")