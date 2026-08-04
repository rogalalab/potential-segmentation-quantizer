"""
Object-Based Colocalization Analysis — Rogala Lab
Detects LAMP1 and Raptor as discrete puncta objects, then asks:
what fraction of Raptor puncta overlap with a LAMP1 lysosome?

This avoids the diffuse cytoplasmic Raptor problem that confounds
intensity-based MCC — only concentrated puncta are detected.

Primary metric:
  raptor_on_lamp1_frac = N Raptor puncta overlapping LAMP1 / N Raptor puncta total

Secondary metrics:
  n_raptor_puncta     — total Raptor puncta per cell
  n_lamp1_puncta      — total LAMP1 puncta per cell
  n_coloc_puncta      — Raptor puncta overlapping with LAMP1
  mean_raptor_area    — mean Raptor punctum area (proxy for size/brightness)
  mean_lamp1_area     — mean LAMP1 punctum area

Colocalization criterion:
  Raptor punctum centroid falls within a dilated LAMP1 object mask.
  Dilation radius = proximity_px (default 3px) to allow for slight
  registration offset between channels.

Usage:
  python object_coloc.py --image_dir ~/Desktop/071526 --dataset subdir
  python object_coloc.py --image_dir ~/Desktop/071526 --dataset subdir \\
      --output_dir ~/Desktop/obj_results --projection max \\
      --residual_factor 1.0 --min_puncta_px 5 --proximity_px 3

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
from skimage.filters import gaussian, rank
from skimage.morphology import disk
from scipy import ndimage as ndi

warnings.filterwarnings('ignore')

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Object-based colocalization')
parser.add_argument('--image_dir',       type=str, default=None,
                    help='Root folder containing images. Not required if --manifest is provided.')
parser.add_argument('--output_dir',      type=str, default=None)
parser.add_argument('--dataset',         type=str, default='subdir',
                    choices=['miapaca2','hek293','subdir','aa_drug'])
parser.add_argument('--projection',      type=str, default='max',
                    choices=['best_z','max','mean'])
parser.add_argument('--residual_factor', type=float, default=1.0,
                    help='Background subtraction aggressiveness (default 1.0). '
                         'Use residual_factor_sweep.py to calibrate.')
parser.add_argument('--window',          type=int, default=32,
                    help='Local median window size in pixels (default 32)')
parser.add_argument('--min_puncta_px',   type=int, default=5,
                    help='Minimum punctum area in pixels (default 5). '
                         'Smaller = detect more puncta including noise.')
parser.add_argument('--proximity_px',    type=int, default=3,
                    help='LAMP1 mask dilation radius for colocalization (default 3px). '
                         'Allows for slight channel registration offset.')
parser.add_argument('--lamp1_percentile',  type=float, default=82,
                    help='Intensity percentile for LAMP1 puncta detection (default 82). '
                         'Lower = more lysosomes detected. LAMP1 is punctate so ~80-85 works well.')
parser.add_argument('--marker_percentile', type=float, default=93,
                    help='Intensity percentile for Raptor/mTOR puncta detection (default 93). '
                         'Higher than LAMP1 to avoid detecting diffuse cytoplasmic signal. '
                         'Use param_sweep.py to calibrate.')
# Keep --puncta_percentile for backwards compatibility (sets both if individual not specified)
parser.add_argument('--puncta_percentile', type=float, default=None,
                    help='Sets BOTH lamp1_percentile and marker_percentile to same value. '
                         'Overridden by --lamp1_percentile / --marker_percentile if provided.')
parser.add_argument('--manifest',    type=str, default=None,
                    help='Path to manifest TSV from ingest.py. '
                         'If provided, --image_dir and --dataset are ignored for file discovery.')
parser.add_argument('--ch_dapi',   type=int, default=None)
parser.add_argument('--ch_lamp1',  type=int, default=None)
parser.add_argument('--ch_marker', type=int, default=None)
args = parser.parse_args()

IMAGE_DIR  = args.image_dir
OUTPUT_DIR = args.output_dir or (
    os.path.join(IMAGE_DIR, 'object_coloc_results') if IMAGE_DIR
    else os.path.join(os.path.dirname(args.manifest), 'object_coloc_results')
)

if not args.manifest and not IMAGE_DIR:
    print("❌ Must provide either --image_dir or --manifest")
    exit(1)
DATASET    = args.dataset
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Resolve percentiles — shared flag overrides individual if given
LAMP1_PCT  = args.puncta_percentile if args.puncta_percentile is not None else args.lamp1_percentile
MARKER_PCT = args.puncta_percentile if args.puncta_percentile is not None else args.marker_percentile

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

print(f"Dataset:          {DATASET}  |  Marker: {MARKER_NAME} (C{CH_MARKER})")
print(f"Projection:       {args.projection}")
print(f"Residual factor:  {args.residual_factor}")
print(f"Min punctum size: {args.min_puncta_px} px")
print(f"Proximity radius: {args.proximity_px} px")
print(f"LAMP1 percentile: {LAMP1_PCT}")
print(f"Marker percentile:{MARKER_PCT}")
print(f"Image dir:        {IMAGE_DIR}")
print(f"Output:           {OUTPUT_DIR}\n")

# ── Core helpers ──────────────────────────────────────────────────────────────
def norm(img, plow=1, phigh=99.5):
    lo, hi = np.percentile(img, plow), np.percentile(img, phigh)
    return np.clip((img - lo) / (hi - lo + 1e-6), 0, 1)

def best_z(stack):
    return int(np.argmax([np.var(stack[z]) for z in range(stack.shape[0])]))

def load_plane(path, projection='max'):
    print(f"      {os.path.basename(path)}...", end=' ', flush=True)
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
    print(f"done {img.shape} ({projection}, {n_z}z)", flush=True)
    return img

def local_median_subtract(img, window=32, residual_factor=1.0):
    """Local median background subtraction — see residual_factor_sweep.py."""
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

def detect_puncta_objects(channel_sub, cyto_roi, min_px=5, percentile=85):
    """
    Detect discrete puncta as labeled objects within cytoplasm ROI.

    Method:
      1. Restrict to cytoplasm (nucleus excluded)
      2. Threshold at given percentile of nonzero pixels
      3. Label connected components
      4. Remove objects smaller than min_px

    Returns:
      labeled image (0=background, 1..N=puncta)
      list of regionprops
    """
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

def object_colocalization(lamp1_labeled, marker_labeled, marker_props,
                          proximity_px=3):
    """
    For each Raptor/mTOR punctum, check if it overlaps with a LAMP1 object.

    Colocalization criterion:
      The Raptor punctum centroid falls within the LAMP1 mask dilated by
      proximity_px pixels. Dilation accounts for slight channel offsets
      and the fact that mTOR docks ON the lysosomal surface (adjacent, not
      necessarily pixel-identical).

    Returns:
      coloc_mask : bool array same shape as image
      n_coloc    : number of Raptor puncta overlapping LAMP1
      coloc_ids  : list of Raptor punctum label IDs that colocalize
    """
    # Dilate LAMP1 mask to allow proximity matching
    lamp1_binary  = lamp1_labeled > 0
    lamp1_dilated = morphology.dilation(lamp1_binary, disk(proximity_px))

    coloc_ids  = []
    coloc_mask = np.zeros(lamp1_labeled.shape, dtype=bool)

    for prop in marker_props:
        cy, cx = prop.centroid
        cy, cx = int(round(cy)), int(round(cx))

        # Check if centroid falls within dilated LAMP1 mask
        if 0 <= cy < lamp1_dilated.shape[0] and 0 <= cx < lamp1_dilated.shape[1]:
            if lamp1_dilated[cy, cx]:
                coloc_ids.append(prop.label)
                coloc_mask[marker_labeled == prop.label] = True

    return coloc_mask, len(coloc_ids), coloc_ids

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
    # Override cell line and marker name from manifest if present
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
    stems   = defaultdict(dict)
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

# ── Main loop ─────────────────────────────────────────────────────────────────
all_rows    = []
cond_summary = {}
PALETTE = ['#D4845A','#4A9BAF','#7B9E6B','#A07BC4','#C4A07B','#6B9EA0','#C47B7B','#7B7BC4']

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
            print(f"    ⚠️  SKIPPING — failed to load: {e}", flush=True)
            continue

        print(f"    [2/4] Segmenting...", end=' ', flush=True)
        nuc_labels, cell_labels = segment_nuclei(c0)
        print(f"done ({nuc_labels.max()} nuclei)", flush=True)

        print(f"    [3/4] Background subtraction...", end=' ', flush=True)
        c1_sub = local_median_subtract(c1, window=args.window,
                                       residual_factor=args.residual_factor)
        cm_sub = local_median_subtract(cm, window=args.window,
                                       residual_factor=args.residual_factor)
        print(f"done", flush=True)

        print(f"    [4/4] Object detection + colocalization...", end=' ', flush=True)

        rows = []
        for cid in range(1, cell_labels.max() + 1):
            cell_roi = (cell_labels == cid)
            nuc_roi  = (nuc_labels  == cid)
            cyto_roi = cell_roi & ~nuc_roi

            if cyto_roi.sum() < 100:
                continue

            # Detect puncta objects in each channel with channel-specific thresholds
            lamp1_labeled, lamp1_props = detect_puncta_objects(
                c1_sub, cyto_roi,
                min_px=args.min_puncta_px,
                percentile=LAMP1_PCT)

            marker_labeled, marker_props = detect_puncta_objects(
                cm_sub, cyto_roi,
                min_px=args.min_puncta_px,
                percentile=MARKER_PCT)

            n_lamp1  = len(lamp1_props)
            n_marker = len(marker_props)

            if n_marker == 0:
                continue   # no Raptor puncta to colocalize

            # Object-based colocalization
            coloc_mask, n_coloc, coloc_ids = object_colocalization(
                lamp1_labeled, marker_labeled, marker_props,
                proximity_px=args.proximity_px)

            frac = n_coloc / n_marker if n_marker > 0 else np.nan

            rows.append({
                'label'             : label,
                'collection'        : collection_id,
                'cell_id'           : cid,
                'n_lamp1_puncta'    : n_lamp1,
                'n_marker_puncta'   : n_marker,
                'n_coloc_puncta'    : n_coloc,
                'coloc_frac'        : frac,
                'mean_lamp1_area'   : np.mean([p.area for p in lamp1_props]) if lamp1_props else np.nan,
                'mean_marker_area'  : np.mean([p.area for p in marker_props]) if marker_props else np.nan,
                'cyto_px'           : int(cyto_roi.sum()),
                'residual_factor'   : args.residual_factor,
                'proximity_px'      : args.proximity_px,
                'lamp1_percentile'  : LAMP1_PCT,
                'marker_percentile' : MARKER_PCT,
                'projection'        : args.projection,
            })

        all_rows.extend(rows)
        fracs = [r['coloc_frac'] for r in rows if not np.isnan(r['coloc_frac'])]
        cond_fracs.extend(fracs)
        print(f"done — {len(rows)} cells  "
              f"mean coloc frac: {np.mean(fracs)*100:.1f}%" if fracs else "done — no cells",
              flush=True)

        # ── Per-collection panel ──────────────────────────────────────────────
        S  = 512
        H, W = c0.shape
        sl = np.s_[H//2-S:H//2+S, W//2-S:W//2+S]

        c1c = c1_sub[sl]; cmc = cm_sub[sl]
        nucc = nuc_labels[sl]; cellc = cell_labels[sl]

        # Build object overlays for the crop
        lamp1_obj_crop = np.zeros(c1c.shape, dtype=bool)
        marker_obj_crop = np.zeros(cmc.shape, dtype=bool)
        coloc_obj_crop  = np.zeros(cmc.shape, dtype=bool)

        for cid in range(1, cell_labels.max() + 1):
            cyto_roi = ((cell_labels == cid) & ~(nuc_labels == cid))
            cyto_crop = cyto_roi[sl]
            l_lab, l_props = detect_puncta_objects(c1_sub, cyto_roi,
                                                   args.min_puncta_px, LAMP1_PCT)
            m_lab, m_props = detect_puncta_objects(cm_sub, cyto_roi,
                                                   args.min_puncta_px, MARKER_PCT)
            lamp1_obj_crop  |= (l_lab > 0)[sl]
            marker_obj_crop |= (m_lab > 0)[sl]
            if m_props:
                cm_crop, n_c, _ = object_colocalization(l_lab, m_lab, m_props, args.proximity_px)
                coloc_obj_crop  |= cm_crop[sl]

        fig, axes = plt.subplots(1, 4, figsize=(18, 5), facecolor='#F8F7F4')
        fig.suptitle(f'{CELL_LINE}  |  {label}  |  {collection_id}',
                     color='#1E2D3A', fontsize=12, fontweight='bold')

        axes[0].imshow(norm(c1c), cmap='Greens', vmin=0, vmax=0.8)
        ov = np.zeros((*c1c.shape,4)); ov[lamp1_obj_crop,1]=1; ov[lamp1_obj_crop,2]=0.3; ov[lamp1_obj_crop,3]=0.7
        axes[0].imshow(ov)
        axes[0].set_title(f'LAMP1 puncta\n({lamp1_obj_crop.sum()//10*10}+ px detected)',
                          color='#1E2D3A', fontsize=9)
        axes[0].axis('off')

        axes[1].imshow(norm(cmc), cmap='Reds', vmin=0, vmax=0.8)
        ov2 = np.zeros((*cmc.shape,4)); ov2[marker_obj_crop,0]=1; ov2[marker_obj_crop,1]=0.2; ov2[marker_obj_crop,3]=0.7
        axes[1].imshow(ov2)
        axes[1].set_title(f'{MARKER_NAME} puncta\n({marker_obj_crop.sum()//10*10}+ px detected)',
                          color='#1E2D3A', fontsize=9)
        axes[1].axis('off')

        # Object-based colocalization overlay
        rgb = np.zeros((*c1c.shape, 3))
        rgb[:,:,1] = norm(c1c) * 0.6
        rgb[:,:,0] = norm(cmc) * 0.6
        rgb[lamp1_obj_crop & ~coloc_obj_crop, 1] = 0.8
        rgb[lamp1_obj_crop & ~coloc_obj_crop, 0] = 0
        rgb[marker_obj_crop & ~coloc_obj_crop, 0] = 0.8
        rgb[marker_obj_crop & ~coloc_obj_crop, 1] = 0
        rgb[coloc_obj_crop, 0] = 1; rgb[coloc_obj_crop, 1] = 1; rgb[coloc_obj_crop, 2] = 0
        axes[2].imshow(np.clip(rgb, 0, 1))
        axes[2].set_title('Object colocalization\nYellow = Raptor on lysosome',
                          color='#1E2D3A', fontsize=9)
        axes[2].legend(handles=[
            mpatches.Patch(color='yellow',  label='Raptor on LAMP1'),
            mpatches.Patch(color='#00CC44', label='LAMP1 only'),
            mpatches.Patch(color='#CC2200', label='Raptor only'),
        ], loc='lower right', fontsize=7, facecolor='white', labelcolor='#1E2D3A')
        axes[2].axis('off')

        # Per-cell bar
        ax_bar = axes[3]; ax_bar.set_facecolor('#F0F4F6')
        col = PALETTE[idx % len(PALETTE)]
        if fracs:
            ax_bar.bar(range(1, len(fracs)+1), [f*100 for f in fracs],
                       color=col, edgecolor='none')
            ax_bar.axhline(np.mean(fracs)*100, color='#2C7A8C',
                           linestyle='--', linewidth=1.2,
                           label=f'Mean {np.mean(fracs)*100:.1f}%')
            ax_bar.legend(fontsize=8, facecolor='white', labelcolor='#1E2D3A')
        ax_bar.set_ylim(0, 100)
        ax_bar.set_xlabel('Cell', color='#1E2D3A', fontsize=8)
        ax_bar.set_ylabel(f'% {MARKER_NAME} puncta on LAMP1', color='#1E2D3A', fontsize=8)
        ax_bar.set_title('Per-cell colocalization', color='#1E2D3A', fontsize=9)
        ax_bar.tick_params(colors='#1E2D3A', labelsize=7)
        for sp in ['top','right']: ax_bar.spines[sp].set_visible(False)
        for sp in ['bottom','left']: ax_bar.spines[sp].set_color('#A0B8C4')

        for ax in axes: ax.set_facecolor('#F8F7F4')
        plt.tight_layout()
        safe = re.sub(r'[^\w\-]', '_', label)
        plt.savefig(os.path.join(OUTPUT_DIR, f'{safe}_{collection_id}_panel.png'),
                    dpi=110, bbox_inches='tight', facecolor='#F8F7F4')
        plt.close()

    cond_summary[label] = {'fracs': cond_fracs, 'color': PALETTE[idx % len(PALETTE)]}
    if cond_fracs:
        print(f"\n  ► {label}  n={len(cond_fracs)} cells  "
              f"mean={np.mean(cond_fracs)*100:.1f}%  "
              f"median={np.median(cond_fracs)*100:.1f}%  "
              f"std={np.std(cond_fracs)*100:.1f}%")

# ── CSV ───────────────────────────────────────────────────────────────────────
df = pd.DataFrame(all_rows)
csv_path = os.path.join(OUTPUT_DIR, 'object_coloc_summary.csv')
df.to_csv(csv_path, index=False)
print(f"\nCSV: {csv_path}")

if not df.empty:
    print("\nColoc fraction by condition:")
    print(df.groupby('label')['coloc_frac']
            .agg(['count','mean','median','std'])
            .round(3).sort_values('mean', ascending=False))

# ── Summary figure ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 7), facecolor='#F8F7F4')
fig.suptitle(f'{CELL_LINE} — Object-Based Colocalization: {MARKER_NAME} puncta on LAMP1\n'
             f'rf={args.residual_factor}  proximity={args.proximity_px}px  '
             f'min_px={args.min_puncta_px}  LAMP1_pct={LAMP1_PCT}  Marker_pct={MARKER_PCT}',
             color='#1E2D3A', fontsize=12, fontweight='bold')

# Main bar: colocalization fraction
ax = axes[0]; ax.set_facecolor('#F0F4F6')
means, sems, xlabels, colors = [], [], [], []
for label in label_list:
    fracs = [f for f in cond_summary[label]['fracs'] if not np.isnan(f)]
    if not fracs: continue
    means.append(np.mean(fracs) * 100)
    sems.append(np.std(fracs) / np.sqrt(len(fracs)) * 100)
    xlabels.append(label.replace('_','\n').replace(' / ','\n'))
    colors.append(cond_summary[label]['color'])

x = np.arange(len(means))
ax.bar(x, means, yerr=sems, color=colors, edgecolor='none', width=0.55,
       capsize=5, error_kw={'ecolor':'#1E2D3A','linewidth':1.5})
ax.set_xticks(x)
ax.set_xticklabels(xlabels, color='#1E2D3A', fontsize=8, rotation=25, ha='right')
ax.set_ylabel(f'% {MARKER_NAME} puncta colocalized with LAMP1\n(mean ± SEM)',
              color='#1E2D3A', fontsize=10)
ax.set_ylim(0, min(100, max(means)*1.3) if means else 100)
ax.set_title('PRIMARY: Fraction of Raptor puncta on lysosomes',
             color='#D4845A', fontsize=11, fontweight='bold')
ax.tick_params(colors='#1E2D3A', labelsize=8)
for sp in ['top','right']: ax.spines[sp].set_visible(False)
for sp in ['bottom','left']: ax.spines[sp].set_color('#A0B8C4')
for xi,(m,s) in enumerate(zip(means, sems)):
    ax.text(xi, m+s+0.5, f'{m:.1f}%', ha='center', va='bottom',
            color='#1E2D3A', fontsize=9, fontweight='bold')

# Secondary bar: mean Raptor puncta count per cell
ax2 = axes[1]; ax2.set_facecolor('#F0F4F6')
means2, sems2, xlabels2, colors2 = [], [], [], []
for label in label_list:
    sub = df[df['label'] == label]['n_marker_puncta']
    if len(sub) == 0: continue
    means2.append(sub.mean())
    sems2.append(sub.std() / np.sqrt(len(sub)))
    xlabels2.append(label.replace('_','\n').replace(' / ','\n'))
    colors2.append(cond_summary[label]['color'])

x2 = np.arange(len(means2))
ax2.bar(x2, means2, yerr=sems2, color=colors2, edgecolor='none', width=0.55,
        capsize=5, error_kw={'ecolor':'#1E2D3A','linewidth':1.5})
ax2.set_xticks(x2)
ax2.set_xticklabels(xlabels2, color='#1E2D3A', fontsize=8, rotation=25, ha='right')
ax2.set_ylabel(f'Mean {MARKER_NAME} puncta per cell', color='#1E2D3A', fontsize=10)
ax2.set_title(f'SECONDARY: {MARKER_NAME} puncta count per cell\n'
              f'(reflects total recruitment, independent of LAMP1)',
              color='#2C7A8C', fontsize=10, fontweight='bold')
ax2.tick_params(colors='#1E2D3A', labelsize=8)
for sp in ['top','right']: ax2.spines[sp].set_visible(False)
for sp in ['bottom','left']: ax2.spines[sp].set_color('#A0B8C4')
for xi,(m,s) in enumerate(zip(means2, sems2)):
    ax2.text(xi, m+s+0.2, f'{m:.0f}', ha='center', va='bottom',
             color='#1E2D3A', fontsize=9, fontweight='bold')

plt.tight_layout()
summary_path = os.path.join(OUTPUT_DIR, 'object_coloc_summary.png')
plt.savefig(summary_path, dpi=130, bbox_inches='tight', facecolor='#F8F7F4')
plt.close()
print(f"Summary: {summary_path}")
print(f"Done. All outputs in: {OUTPUT_DIR}")