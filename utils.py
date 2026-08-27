"""
utils.py — Rogala Lab Pipeline
Shared functions used by object_coloc.py, manders_coloc.py, param_sweep.py

Import with:
    from utils import (load_plane, segment_nuclei, detect_puncta_objects,
                       local_median_subtract, object_colocalization,
                       norm, PALETTE)
"""

import numpy as np
import tifffile
import warnings
from skimage import filters, morphology, measure, segmentation, feature
from skimage.filters import gaussian, rank
from skimage.morphology import disk
from scipy import ndimage as ndi

warnings.filterwarnings('ignore')

# ── Color palette ─────────────────────────────────────────────────────────────
PALETTE = ['#D4845A','#4A9BAF','#7B9E6B','#A07BC4',
           '#C4A07B','#6B9EA0','#C47B7B','#7B7BC4',
           '#9EA07B','#A07B9E','#7B9EA0','#C4B07B']

# ── Image loading ─────────────────────────────────────────────────────────────
def best_z(stack):
    """Return index of z-plane with highest variance (sharpest focus)."""
    return int(np.argmax([np.var(stack[z]) for z in range(stack.shape[0])]))

def load_plane(path, projection='max'):
    """
    Load a TIFF and collapse z-stack.

    projection: 'max'    — maximum intensity projection (recommended, captures all puncta)
                'best_z' — single sharpest plane (fast)
                'mean'   — average projection
    """
    import os
    print(f"      {os.path.basename(path)}...", end=' ', flush=True)
    stack = tifffile.imread(path).astype(np.float32)
    if stack.ndim == 2:
        print(f"done {stack.shape} (2D)", flush=True)
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

def norm(img, plow=1, phigh=99.5):
    """Percentile normalisation to [0, 1]."""
    lo, hi = np.percentile(img, plow), np.percentile(img, phigh)
    return np.clip((img - lo) / (hi - lo + 1e-6), 0, 1)

# ── Background subtraction ────────────────────────────────────────────────────
def local_median_subtract(img, window=32, residual_factor=1.0):
    """
    Spatially adaptive background subtraction using local median.

    For each pixel, background = median of (window × window) neighborhood.
    Removes slowly-varying illumination gradients while preserving puncta.

    window:          neighborhood size in pixels (24-36 recommended for 100x)
    residual_factor: multiplier on noise std for final offset removal
                     0.0 = pure median subtraction
                     1.0 = remove 1σ residual noise (default)
                     2.0 = conservative, may remove weak signal

    Reference: Dunn KW et al., AJP Cell Physiol 2011
    """
    img_norm = img - img.min()
    scale    = 65535.0 / (img_norm.max() + 1e-6)
    img_u16  = (img_norm * scale).astype(np.uint16)
    local_bg = rank.median(img_u16, disk(window // 2)).astype(np.float32) / scale
    sub      = img - local_bg
    near_zero = sub[sub < np.percentile(sub, 50)]
    noise_std = near_zero.std() if len(near_zero) > 10 else 1.0
    sub       = sub - residual_factor * noise_std
    return np.clip(sub, 0, None)

# ── Nucleus and cell segmentation ─────────────────────────────────────────────
def segment_nuclei(dapi, nuc_sigma=3, nuc_min_size=2000,
                   nuc_erode=4, nuc_expand_px=30, nuc_peak_dist=40):
    """
    Segment nuclei from DAPI channel and expand to cell boundaries.

    Returns:
        nuc_labels  — labeled nucleus mask (0=background, 1..N=nuclei)
        cell_labels — expanded cell boundary mask (same IDs as nuc_labels)

    Method:
        Gaussian smooth → Otsu threshold → fill holes → erosion →
        distance transform → peak detection → watershed separation →
        dilation for cell boundary
    """
    smooth      = gaussian(dapi, sigma=nuc_sigma)
    thresh      = filters.threshold_otsu(smooth)
    mask        = smooth > thresh
    mask        = morphology.remove_small_objects(mask, min_size=nuc_min_size)
    mask        = ndi.binary_fill_holes(mask)
    mask        = morphology.binary_erosion(mask, morphology.disk(nuc_erode))
    dist        = ndi.distance_transform_edt(mask)
    peaks       = feature.peak_local_max(dist, min_distance=nuc_peak_dist,
                                          labels=mask)
    pm          = np.zeros(dist.shape, dtype=bool)
    pm[tuple(peaks.T)] = True
    markers     = measure.label(pm)
    nuc_labels  = segmentation.watershed(-dist, markers, mask=mask)
    cell_mask   = morphology.dilation(mask, morphology.disk(nuc_expand_px))
    cell_labels = segmentation.watershed(-dist, markers, mask=cell_mask)
    return nuc_labels, cell_labels

def get_cyto_roi(cell_labels, nuc_labels, cid):
    """
    Cytoplasmic ROI for cell cid: cell boundary minus nucleus.
    Nucleus excluded because both channels are near-zero there,
    which would artificially inflate correlation metrics.

    Reference: Dunn KW et al., AJP Cell Physiol 2011
    """
    return (cell_labels == cid) & ~(nuc_labels == cid)

# ── Puncta detection ──────────────────────────────────────────────────────────
def detect_puncta_objects(channel_sub, cyto_roi, min_px=5, percentile=85):
    """
    Detect discrete puncta as labeled objects within cytoplasm ROI.

    Uses intensity percentile threshold on background-subtracted image.
    Only signal pixels (above local background) are considered.

    Returns:
        labeled  — labeled image (0=background, 1..N=puncta)
        props    — regionprops list with area, centroid, mean_intensity etc.
    """
    ch      = channel_sub * cyto_roi
    nonzero = ch[ch > 0]
    if len(nonzero) < 10:
        return np.zeros_like(ch, dtype=int), []
    thr     = np.percentile(nonzero, percentile)
    mask    = ch > thr
    mask    = morphology.remove_small_objects(mask, min_size=min_px)
    mask    = morphology.remove_small_holes(mask, area_threshold=10)
    labeled = measure.label(mask)
    props   = measure.regionprops(labeled, intensity_image=ch)
    return labeled, props

def puncta_morphology(props):
    """
    Extract morphology metrics from a list of regionprops.

    Returns dict with:
        n_puncta         — count
        mean_area        — mean area in pixels
        median_area      — median area
        mean_intensity   — mean per-punctum intensity
        mean_elongation  — major_axis / minor_axis (1=round, >1=elongated)
    """
    if not props:
        return {
            'n_puncta': 0, 'mean_area': np.nan,
            'median_area': np.nan, 'mean_intensity': np.nan,
            'mean_elongation': np.nan,
        }
    areas       = [p.area for p in props]
    intensities = [p.mean_intensity for p in props]
    elongations = []
    for p in props:
        if p.minor_axis_length > 0:
            elongations.append(p.major_axis_length / p.minor_axis_length)
    return {
        'n_puncta'       : len(props),
        'mean_area'      : float(np.mean(areas)),
        'median_area'    : float(np.median(areas)),
        'mean_intensity' : float(np.mean(intensities)),
        'mean_elongation': float(np.mean(elongations)) if elongations else np.nan,
    }

# ── Object-based colocalization ───────────────────────────────────────────────
def object_colocalization(ref_labeled, query_labeled, query_props,
                          proximity_px=3):
    """
    For each query punctum, check if centroid falls within reference mask
    dilated by proximity_px pixels.

    ref   = reference channel (e.g. LAMP1, SAMTOR)
    query = channel being measured (e.g. Raptor, SHMT1)

    Returns:
        coloc_mask — bool array, True where query puncta colocalize
        n_coloc    — number of colocalized query puncta
        coloc_ids  — list of query punctum label IDs that colocalize
    """
    ref_dilated = morphology.dilation(ref_labeled > 0, disk(proximity_px))
    coloc_ids   = []
    coloc_mask  = np.zeros(ref_labeled.shape, dtype=bool)
    for prop in query_props:
        cy, cx = int(round(prop.centroid[0])), int(round(prop.centroid[1]))
        if (0 <= cy < ref_dilated.shape[0] and
                0 <= cx < ref_dilated.shape[1] and
                ref_dilated[cy, cx]):
            coloc_ids.append(prop.label)
            coloc_mask[query_labeled == prop.label] = True
    return coloc_mask, len(coloc_ids), coloc_ids

# ── Nuclear / cytoplasmic ratio ───────────────────────────────────────────────
def nuclear_cytoplasmic_ratio(channel, nuc_roi, cyto_roi):
    """
    Compute nuclear and cytoplasmic intensity metrics for one channel.

    Returns dict with:
        nuc_mean      — mean intensity in nucleus
        cyto_mean     — mean intensity in cytoplasm
        nuc_frac      — fraction of total cell intensity in nucleus
        nc_ratio      — nuclear mean / cytoplasmic mean
                        >1 = nuclear enriched, <1 = cytoplasmic enriched
    """
    nuc_px   = channel[nuc_roi]
    cyto_px  = channel[cyto_roi]
    cell_sum = nuc_px.sum() + cyto_px.sum()

    nuc_mean  = float(nuc_px.mean())  if nuc_px.size  > 0 else np.nan
    cyto_mean = float(cyto_px.mean()) if cyto_px.size > 0 else np.nan
    nuc_frac  = float(nuc_px.sum() / cell_sum) if cell_sum > 0 else np.nan
    nc_ratio  = float(nuc_mean / (cyto_mean + 1e-9)) if cyto_mean else np.nan

    return {
        'nuc_mean' : nuc_mean,
        'cyto_mean': cyto_mean,
        'nuc_frac' : nuc_frac,
        'nc_ratio' : nc_ratio,
    }

# ── Spatial distribution ──────────────────────────────────────────────────────
def spatial_distribution(props, nuc_labels, cell_labels, cid):
    """
    For each punctum, compute normalised distance from nucleus boundary.

    distance = 0 → punctum is at nucleus edge
    distance = 1 → punctum is at cell periphery

    This quantifies perinuclear vs peripheral distribution.

    Returns list of normalised distances, one per punctum.
    """
    nuc_roi  = (nuc_labels  == cid)
    cell_roi = (cell_labels == cid)

    if not props or nuc_roi.sum() == 0:
        return []

    # Distance from nucleus boundary (positive = outside nucleus)
    dist_from_nuc = ndi.distance_transform_edt(~nuc_roi)

    # Max distance within cell (cell radius approximation)
    cell_distances = dist_from_nuc[cell_roi]
    max_dist       = cell_distances.max() if cell_distances.size > 0 else 1.0

    norm_distances = []
    for prop in props:
        cy, cx = int(round(prop.centroid[0])), int(round(prop.centroid[1]))
        if (0 <= cy < dist_from_nuc.shape[0] and
                0 <= cx < dist_from_nuc.shape[1]):
            d = dist_from_nuc[cy, cx] / (max_dist + 1e-9)
            norm_distances.append(float(np.clip(d, 0, 1)))

    return norm_distances

def spatial_summary(norm_distances):
    """
    Summarise normalised distance distribution per cell.

    Returns dict with:
        mean_dist    — mean normalised distance (0=perinuclear, 1=peripheral)
        frac_perinuc — fraction of puncta with dist < 0.33 (perinuclear zone)
        frac_periph  — fraction of puncta with dist > 0.67 (peripheral zone)
    """
    if not norm_distances:
        return {'mean_dist': np.nan, 'frac_perinuc': np.nan,
                'frac_periph': np.nan}
    d = np.array(norm_distances)
    return {
        'mean_dist'   : float(d.mean()),
        'frac_perinuc': float((d < 0.33).mean()),
        'frac_periph' : float((d > 0.67).mean()),
    }

# ── File integrity check ──────────────────────────────────────────────────────
def check_file_integrity(path):
    """
    Try to open a TIFF and read its shape.
    Returns (True, shape_str) on success or (False, error_msg) on failure.
    """
    try:
        import tifffile as tf
        with tf.TiffFile(path) as t:
            shape = t.series[0].shape
        return True, str(shape)
    except Exception as e:
        return False, str(e)