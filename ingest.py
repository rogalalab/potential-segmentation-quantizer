"""
Image Ingestion — Rogala Lab
Discovers TIFF files, groups them by condition and collection,
validates channel assignments, and writes a manifest CSV.

The manifest is then consumed by object_coloc.py and param_sweep.py
so neither script needs to know about filename patterns.

Supported layouts:
  subdir    — one subdirectory per condition, files inside
  flat      — all files in one folder, condition encoded in filename
              with pattern: {AA+|AA-}_{drug}_{collection}_XY..._C{ch}.tif
              or:           {DRUG}_{REFED}_..._C{ch}.tif  (hek293 style)
              or:           MiaPaca2_{FED|ST}_{HG|LG}_..._C{ch}.tif

Manifest format (TSV):
  label         condition display name
  collection    field of view identifier
  ch_dapi       path to DAPI channel file
  ch_lamp1      path to LAMP1 channel file
  ch_marker     path to Raptor/mTOR channel file

Usage:
  # Subdir layout (e.g. 071526)
  python ingest.py \\
    --image_dir ~/Desktop/071526 \\
    --layout subdir \\
    --ch_dapi 0 --ch_lamp1 1 --ch_marker 2 \\
    --output ~/Desktop/071526_manifest.tsv

  # Flat layout (e.g. 080326) — verify channels first!
  python ingest.py \\
    --image_dir "/path/to/080326" \\
    --layout flat \\
    --ch_dapi 0 --ch_lamp1 1 --ch_marker 2 \\
    --output ~/Desktop/080326_manifest.tsv

  # Preview without writing (dry run)
  python ingest.py --image_dir ~/Desktop/071526 --layout subdir --dry_run

After running, OPEN THE MANIFEST and verify:
  - Label names make biological sense
  - ch_dapi / ch_lamp1 / ch_marker paths look correct
  - Channel numbers match what the microscope actually acquired
  Edit any mistakes directly in the TSV before running analysis.
"""

import os, re, argparse
from collections import defaultdict
import pandas as pd
import sys

# Add pipeline directory to path so utils is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Image ingestion — generate analysis manifest')
parser.add_argument('--image_dir', type=str, required=True,
                    help='Root folder containing images or condition subdirectories')
parser.add_argument('--layout',    type=str, required=True,
                    choices=['subdir', 'flat'],
                    help='subdir: one folder per condition | flat: all files in one folder')
parser.add_argument('--ch_dapi',   type=int, required=True,
                    help='Channel number for DAPI (e.g. 0). Verify with FIJI before running.')
parser.add_argument('--ch_lamp1',  type=int, required=True,
                    help='Channel number for reference channel C1 (e.g. 1). Verify with FIJI.')
parser.add_argument('--ch_marker', type=int, required=True,
                    help='Channel number for marker channel (e.g. 2). Verify with FIJI.')
parser.add_argument('--output',    type=str, default=None,
                    help='Output manifest path (default: <image_dir>/manifest.tsv)')
parser.add_argument('--ch1_name',    type=str, default='LAMP1',
                    help='Display name for C1 channel (e.g. LAMP1, SAMTOR). Default: LAMP1')
parser.add_argument('--marker_name', type=str, default='Raptor',
                    help='Display name for marker channel (e.g. Raptor, SHMT1). Default: Raptor')
parser.add_argument('--cell_line',   type=str, default='HEK293',
                    help='Cell line name for display (default: HEK293)')
parser.add_argument('--dry_run', action='store_true',
                    help='Print discovered files without writing manifest')
parser.add_argument('--integrity_check', action='store_true',
                    help='Test-open every TIFF before writing manifest. '
                         'Reports corrupted files. Slower but catches bad files early.')

# Flat layout options
parser.add_argument('--flat_pattern', type=str, default='auto',
                    choices=['auto', 'aa_drug', 'hek293', 'miapaca2'],
                    help='Filename pattern for flat layout (default: auto-detect)')
args = parser.parse_args()

IMAGE_DIR   = args.image_dir
OUTPUT_PATH = args.output or os.path.join(IMAGE_DIR, 'manifest.tsv')
CH_DAPI     = args.ch_dapi
CH_LAMP1    = args.ch_lamp1
CH_MARKER   = args.ch_marker

print(f"Image dir:   {IMAGE_DIR}")
print(f"Layout:      {args.layout}")
print(f"Channels:    DAPI=C{CH_DAPI}  LAMP1=C{CH_LAMP1}  {args.marker_name}=C{CH_MARKER}")
print(f"Output:      {OUTPUT_PATH}\n")
print("⚠️  Verify channel assignments with FIJI before running analysis!\n")

# ── Channel file regex — matches trailing _C{n}.tif ──────────────────────────
CH_RE = re.compile(r'_C(\d)\.tif$', re.IGNORECASE)

def extract_channel(fname):
    """Extract channel number from filename, return None if not found."""
    m = CH_RE.search(fname)
    return int(m.group(1)) if m else None

def stem_of(fname):
    """Everything before the final _C{n}.tif"""
    m = CH_RE.search(fname)
    return fname[:m.start()] if m else None

# ── Layout: subdir ────────────────────────────────────────────────────────────
def discover_subdir(image_dir):
    """
    Each subdirectory = one condition (label = directory name).
    Files inside = collections, grouped by stem before _C{n}.tif.
    """
    records = []
    for entry in sorted(os.listdir(image_dir)):
        full = os.path.join(image_dir, entry)
        if not os.path.isdir(full):
            continue
        if entry.startswith('.') or any(x in entry.lower() for x in ['result','manifest','sweep']):
            continue

        # Group files by stem
        stems = defaultdict(dict)
        for f in sorted(os.listdir(full)):
            ch = extract_channel(f)
            st = stem_of(f)
            if ch is not None and st is not None:
                stems[st][ch] = os.path.join(full, f)

        for stem, channels in sorted(stems.items()):
            missing = [ch for ch in [CH_DAPI, CH_LAMP1, CH_MARKER] if ch not in channels]
            if missing:
                print(f"  SKIP {entry}/{os.path.basename(stem)} — missing C{missing}")
                continue
            records.append({
                'label'      : entry,
                'collection' : os.path.basename(stem),
                'ch_dapi'    : channels[CH_DAPI],
                'ch_lamp1'   : channels[CH_LAMP1],
                'ch_marker'  : channels[CH_MARKER],
            })

    return records

# ── Layout: flat — pattern auto-detection ─────────────────────────────────────
# Pattern regexes — each returns (label, stem) or None
FLAT_PATTERNS = {
    'aa_drug': re.compile(
        r'^(AA[+\-]_[A-Za-z0-9µuμ]+)_(C\d+(?:\s*-\s*\d+)?)_XY\d+',
        re.IGNORECASE),
    'hek293': re.compile(
        r'^((?:Slide\s*\d+_)?(?:DMSO|M6659|AZD8055)_(?:No_Refed|Refed))',
        re.IGNORECASE),
    'miapaca2': re.compile(
        r'^(MiaPaca2_(?:FED|ST)_(?:HG|LG))',
        re.IGNORECASE),
    'exp109': re.compile(
        r'^#(\d+)[_\-](\d+)_XY\d+',
        re.IGNORECASE),
    'timeline': re.compile(
        r'^(HEK293_STARVED_HBSS_[\d_]+_MIN)',
        re.IGNORECASE),
}

def detect_flat_pattern(files):
    """Try each pattern on a sample of files, return best match."""
    for name, regex in FLAT_PATTERNS.items():
        hits = sum(1 for f in files[:20] if regex.match(f))
        if hits >= 3:
            return name
    return None

def parse_flat_label(fname, pattern_name):
    """Extract condition label from filename given pattern."""
    regex = FLAT_PATTERNS[pattern_name]
    m = regex.match(os.path.basename(fname))
    if not m:
        return None
    raw = m.group(1).strip().rstrip('_')

    if pattern_name == 'exp109':
        # Plate map from experiment description
        PLATE_MAP = {
            1:  ('Basal',                  '-Dox', '+DMSO'),
            2:  ('Basal',                  '-Dox', '+AP21967'),
            3:  ('Basal',                  '+Dox', '+DMSO'),
            4:  ('Basal',                  '+Dox', '+AP21967'),
            5:  ('Met Starvation',         '-Dox', '+DMSO'),
            6:  ('Met Starvation',         '-Dox', '+AP21967'),
            7:  ('Met Starvation',         '+Dox', '+DMSO'),
            8:  ('Met Starvation',         '+Dox', '+AP21967'),
            9:  ('Met Starv + SAM',        '-Dox', '+DMSO'),
            10: ('Met Starv + SAM',        '-Dox', '+AP21967'),
            11: ('Met Starv + SAM',        '+Dox', '+DMSO'),
            12: ('Met Starv + SAM',        '+Dox', '+AP21967'),
        }
        m2 = re.match(r'^#(\d+)[_\-](\d+)_XY', os.path.basename(fname))
        if not m2:
            return None
        well = int(m2.group(1))
        if well not in PLATE_MAP:
            return None
        metabolic, dox, drug = PLATE_MAP[well]
        return f"{metabolic} / {dox} / {drug}"

    if pattern_name == 'timeline':
        # HEK293_STARVED_HBSS_10_MIN → "10 min"
        # HEK293_STARVED_HBSS_2_5_MIN → "2.5 min"
        m2 = re.search(r'HBSS_([\d_]+)_MIN', raw, re.IGNORECASE)
        if m2:
            t = m2.group(1).rstrip('_').replace('_', '.')
            return f"{t} min"
        return raw

    if pattern_name == 'aa_drug':
        # raw = "AA+_10uM6698" or "AA-_DMSO"
        # Split on first underscore: AA+ / AA- + drug name
        parts = raw.split('_', 1)
        aa   = parts[0]   # AA+ or AA-
        drug = parts[1] if len(parts) > 1 else 'unknown'
        aa_label = 'AA+ (with amino acids)' if '+' in aa else 'AA- (no amino acids)'
        return f"{aa_label} / {drug}"

    elif pattern_name == 'hek293':
        raw = re.sub(r'^Slide\s*\d+_', '', raw, flags=re.IGNORECASE)
        parts = raw.split('_')
        drug  = parts[0]
        refed = '_'.join(parts[1:]).replace('No_Refed','No Refeeding').replace('Refed','Refed')
        return f"{drug} / {refed}"

    elif pattern_name == 'miapaca2':
        parts = raw.split('_')
        cond  = {'FED':'FED','ST':'STARVED'}.get(parts[1].upper(), parts[1])
        gluc  = {'HG':'High Glucose','LG':'Low Glucose'}.get(parts[2].upper(), parts[2])
        return f"{cond} / {gluc}"

    return raw

def discover_flat(image_dir, pattern_name='auto'):
    """
    All files in one folder.
    Groups by (label, collection_id) where collection comes from regex group 2.
    """
    files = sorted(os.listdir(image_dir))
    tif_files = [f for f in files if f.lower().endswith('.tif')]

    if pattern_name == 'auto':
        pattern_name = detect_flat_pattern(tif_files)
        if pattern_name:
            print(f"Auto-detected flat pattern: {pattern_name}")
        else:
            print("WARNING: Could not auto-detect filename pattern.")
            pattern_name = 'aa_drug'  # fallback

    # Group by (label, collection_id)
    # For aa_drug: collection = regex group 2 (e.g. "C1" or "C3 - 1")
    # For others: collection = stem before final _C{ch}.tif
    stems   = defaultdict(dict)
    skipped = []

    for f in tif_files:
        ch = extract_channel(f)
        if ch is None:
            skipped.append(f)
            continue

        label = parse_flat_label(f, pattern_name)
        if label is None:
            skipped.append(f)
            continue

        # Extract collection identifier
        if pattern_name == 'aa_drug':
            m = FLAT_PATTERNS['aa_drug'].match(os.path.basename(f))
            collection = m.group(2).strip() if m and len(m.groups()) >= 2 else stem_of(os.path.basename(f))
        elif pattern_name == 'exp109':
            # Use well-replicate as collection e.g. "well8_rep3"
            m = FLAT_PATTERNS['exp109'].match(os.path.basename(f))
            if m:
                collection = f"well{m.group(1)}_rep{m.group(2)}"
            else:
                collection = stem_of(os.path.basename(f))
        else:
            # For timeline and others: use XY coordinates as collection ID
            xy = re.search(r'(XY\d+)', os.path.basename(f))
            collection = xy.group(1) if xy else stem_of(os.path.basename(f))

        stems[(label, collection)][ch] = os.path.join(image_dir, f)

    if skipped:
        print(f"  Skipped {len(skipped)} files that didn't match pattern")

    records = []
    for (label, collection), channels in sorted(stems.items()):
        missing = [ch for ch in [CH_DAPI, CH_LAMP1, CH_MARKER] if ch not in channels]
        if missing:
            print(f"  SKIP {label} / {collection} — missing C{missing}")
            continue
        records.append({
            'label'      : label,
            'collection' : collection,
            'ch_dapi'    : channels[CH_DAPI],
            'ch_lamp1'   : channels[CH_LAMP1],
            'ch_marker'  : channels[CH_MARKER],
        })

    return records

# ── Discover ──────────────────────────────────────────────────────────────────
if args.layout == 'subdir':
    records = discover_subdir(IMAGE_DIR)
else:
    records = discover_flat(IMAGE_DIR, args.flat_pattern)

# ── Report ────────────────────────────────────────────────────────────────────
if not records:
    print("\n❌ No valid triplets found. Check:")
    print("  - --ch_dapi / --ch_lamp1 / --ch_marker match actual channel numbers")
    print("  - Files have _C{n}.tif suffix")
    print("  - --layout matches your directory structure")
    exit(1)

df = pd.DataFrame(records)
conditions = df['label'].unique()
print(f"Found {len(records)} collection(s) across {len(conditions)} condition(s):\n")

for cond in sorted(conditions):
    n = len(df[df['label'] == cond])
    print(f"  {cond}: {n} collection(s)")

# ── Integrity check ───────────────────────────────────────────────────────────
if args.integrity_check:
    print("\nRunning integrity check on all files...")
    from utils import check_file_integrity
    bad_files = []
    for _, row in df.iterrows():
        for col in ['ch_dapi','ch_lamp1','ch_marker']:
            ok, info = check_file_integrity(row[col])
            if not ok:
                bad_files.append((row['label'], row['collection'], col, info))
                print(f"  ❌ {row['label']} / {row['collection']} / {col}: {info}")
            else:
                print(f"  ✓ {os.path.basename(row[col])} {info}")
    if bad_files:
        print(f"\n⚠️  {len(bad_files)} corrupted file(s) found.")
        print("Remove or replace them before running analysis.")
    else:
        print(f"\n✓ All {len(df)*3} files opened successfully.")

# ── Dry run ───────────────────────────────────────────────────────────────────
if args.dry_run:
    print("\n[Dry run — no file written]")
    print("\nFirst few rows:")
    print(df[['label','collection']].head(10).to_string(index=False))
    exit(0)

# ── Write manifest ────────────────────────────────────────────────────────────
# Add metadata columns
df['ch1_name']    = args.ch1_name
df['marker_name'] = args.marker_name
df['cell_line']   = args.cell_line
df['ch_dapi_num']   = CH_DAPI
df['ch_lamp1_num']  = CH_LAMP1
df['ch_marker_num'] = CH_MARKER

df.to_csv(OUTPUT_PATH, sep='\t', index=False)
print(f"\n✓ Manifest written: {OUTPUT_PATH}")
print(f"  {len(df)} rows × {len(df.columns)} columns")
print(f"\n⚠️  Open the manifest and verify:")
print(f"  1. Label names make biological sense")
print(f"  2. ch_dapi / ch_lamp1 / ch_marker paths point to the right channels")
print(f"  3. If anything looks wrong, edit the TSV directly before running analysis")
print(f"\nThen run:")
print(f"  python object_coloc.py --manifest {OUTPUT_PATH} ...")
print(f"  python param_sweep.py  --manifest {OUTPUT_PATH} ...")