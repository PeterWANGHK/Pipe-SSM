"""Enrich the dataset: parse every .xpx in 'Sample Data 1/Analyzed Data' and merge with its .pn
expert annotations into merged_data_with_fault_classes_<id>.csv.

Reuses the project's proven label logic (merging_general_enhanced.merge_single_file):
FaultClass = #('Defect =' lines) when joint Flags=1, else 0  -> severity {0,1,2,3}.
"""
from __future__ import annotations
import os, sys, glob, re
import xml.etree.ElementTree as ET
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from merging_general_enhanced import merge_single_file

PYDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(PYDIR)
ANALYZED = os.path.join(REPO, 'Sample Data 1', 'Analyzed Data')

TARGET_COLS = {26: "D1_32Hz_R", 27: "D1_32Hz_Theta", 28: "D1_100Hz_R", 29: "D1_100Hz_Theta",
               33: "D2_32Hz_R", 34: "D2_32Hz_Theta", 35: "D2_100Hz_R", 36: "D2_100Hz_Theta"}


def parse_xpx(path):
    """Tolerant extraction of the <rawData> block (some .xpx have malformed XML tokens)."""
    text = None
    try:
        root = ET.parse(path).getroot()
        rd = root.find('.//rawData')
        text = rd.text if rd is not None else None
    except Exception:
        text = None
    if text is None:
        # fallback: scan file text for the rawData block, ignoring XML well-formedness
        with open(path, 'r', encoding='latin-1', errors='ignore') as f:
            raw = f.read()
        m = re.search(r'<rawData[^>]*>(.*?)</rawData>', raw, re.DOTALL)
        if not m:
            return None
        text = m.group(1)
    rows = []
    for line in text.strip().split('\n'):
        v = line.strip().split()
        if len(v) < 37:
            continue
        try:
            rows.append({c: float(v[i]) for i, c in TARGET_COLS.items()})
        except (ValueError, IndexError):
            continue
    return pd.DataFrame(rows) if rows else None


def date_to_id(basename):
    """2024-05-09-014 -> 014 ; 2025-08-15-002 -> 815002 ; 2025-08-06-002 -> 86002."""
    m = re.match(r'(\d{4})-(\d{2})-(\d{2})-(\d+)', basename)
    if not m:
        return basename
    yyyy, mm, dd, num = m.groups()
    if yyyy == '2024':
        return num                       # legacy bare ids
    return f"8{int(dd)}{num}"            # 2025 series: 8 + day + num


def build_all(only_new=True):
    existing = {re.search(r'(\d+)', os.path.basename(f)).group(1)
                for f in glob.glob(os.path.join(PYDIR, 'merged_data_with_fault_classes_*.csv'))
                if 'synthetic' not in f}
    xpx_files = sorted(glob.glob(os.path.join(ANALYZED, '*.xpx')))
    summary = []
    for xpx in xpx_files:
        base = os.path.basename(xpx).replace('.xpx', '')
        cid = date_to_id(base)
        out = os.path.join(PYDIR, f'merged_data_with_fault_classes_{cid}.csv')
        status = 'new' if cid not in existing else 'exists'
        if only_new and status == 'exists':
            summary.append((cid, base, 'skip(exists)', '-')); continue
        # find matching .pn (prefer D1.1pn)
        pn = None
        for cand in [f'{base}_D1.1pn', f'{base}_D1.5pn', f'{base}_D2.1pn']:
            p = os.path.join(ANALYZED, cand)
            if os.path.exists(p):
                pn = p; break
        if pn is None:
            summary.append((cid, base, 'NO .pn', '-')); continue
        try:
            df = parse_xpx(xpx)
            if df is None or len(df) == 0:
                summary.append((cid, base, 'parse fail', '-')); continue
            parsed_txt = os.path.join(PYDIR, f'{cid}_D1D2_parsed.txt')
            df.to_csv(parsed_txt, sep='\t', index=False)
            merged = merge_single_file(parsed_txt, pn, output_file=out, visualize=False)
            if merged is not None and os.path.exists(out):
                fc = pd.to_numeric(merged['FaultClass'], errors='coerce').fillna(0).astype(int).clip(0, 3)
                dist = {c: int((fc == c).sum()) for c in [0, 1, 2, 3]}
                faultpct = round(100 * (fc > 0).mean(), 1)
                summary.append((cid, base, f'OK n={len(merged)}', f'{dist} fault%={faultpct}'))
            else:
                summary.append((cid, base, 'merge fail', '-'))
        except Exception as e:
            summary.append((cid, base, f'ERR {type(e).__name__}: {str(e)[:80]}', '-'))
    print("\n=== BUILD SUMMARY ===")
    for cid, base, st, dist in summary:
        print(f"  {cid:10s} <- {base:18s} {st:28s} {dist}")
    return summary


if __name__ == '__main__':
    only_new = '--all' not in sys.argv
    build_all(only_new=only_new)
