"""Label-efficiency test: does the skin-depth physics constraint help when DEFECT LABELS ARE SCARCE?

Per Sel et al. 2023 (Bio-Z PINN), physics-informed gains live in the low-label regime. We vary the
number of defect-bearing training campaigns (2 / 4 / all) and compare physics vs no-physics, held-out
88005, multi-seed. If physics helps MORE at small N, 'physics-informed' is an honest headline. Resumable.
"""
from __future__ import annotations
import os, sys, json, subprocess, time, argparse
import numpy as np

PYDIR = os.path.dirname(os.path.abspath(__file__)); REPO_PY = os.path.dirname(PYDIR)
RES = os.path.join(REPO_PY, 'results'); os.makedirs(RES, exist_ok=True)
PY = sys.executable
ENV = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUTF8='1')

SIZES = {
    'n2': ['102', '87001'],
    'n4': ['102', '87001', '84001', '810001'],
    'nAll': ['102', '104', '021', '88003', '84001', '87001', '89005', '89006', '810001', '014'],
}
TEST = '88005'
SEEDS = [42, 123]
CONFIGS = {'physics': ['--physics-features', '--physics-loss'], 'nophysics': []}
BASE = ['--test-ids', TEST, '--regroup', '--window', '512', '--stride', '384',
        '--epochs', '8', '--d-model', '64', '--layers', '3', '--batch', '16']


def valid(p):
    try:
        r = json.load(open(p)); return r if 'F1_macro' in r else None
    except Exception:
        return None


def run(size, cfg, seed):
    tag = f'le_{size}_{cfg}_s{seed}'
    jp = os.path.join(RES, f'{tag}.json')
    if valid(jp) is not None:
        print(f"  [skip] {tag}"); return valid(jp)
    cmd = [PY, '-u', '-m', 'ssm_ndt.train', '--train-ids', *SIZES[size], *BASE,
           '--seed', str(seed), *CONFIGS[cfg], '--out', tag]
    print(f">>> {tag}", flush=True); t0 = time.time()
    p = subprocess.run(cmd, cwd=REPO_PY, capture_output=True, text=True, env=ENV)
    if p.returncode != 0:
        print(f"  FAILED {tag}\n{p.stderr[-400:]}", flush=True); return None
    print(f"  {time.time()-t0:.0f}s", flush=True)
    return valid(jp)


def ms(xs):
    return f"{np.mean(xs):.3f}±{np.std(xs):.3f}" if xs else "-"


def metric(size, cfg, key):
    out = []
    for s in SEEDS:
        r = valid(os.path.join(RES, f'le_{size}_{cfg}_s{s}.json'))
        if r:
            pc = r['F1_per_class']
            out.append(sum(pc[1:3]) / 2 if key == 'defect' else r['F1_macro'])
    return out


def build_table():
    md = ["# Label-efficiency: physics vs no-physics as defect-campaign count grows (held-out 88005)", "",
          "Defect-F1 (mean±std over seeds). If physics helps MORE at small N, physics-informed is honest.", "",
          "| #defect campaigns | no-physics Defect-F1 | +physics Defect-F1 | Δ (physics−base) |",
          "|---|---|---|---|"]
    for size in ['n2', 'n4', 'nAll']:
        b = metric(size, 'nophysics', 'defect'); p = metric(size, 'physics', 'defect')
        d = (np.mean(p) - np.mean(b)) if (b and p) else float('nan')
        md.append(f"| {len(SIZES[size])} ({size}) | {ms(b)} | {ms(p)} | {d:+.3f} |")
    txt = "\n".join(md)
    open(os.path.join(RES, 'LABELEFF_TABLE.md'), 'w', encoding='utf-8').write(txt)
    print("\n" + txt)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--table', action='store_true'); a = ap.parse_args()
    if not a.table:
        for size in SIZES:
            for cfg in CONFIGS:
                for seed in SEEDS:
                    run(size, cfg, seed)
                build_table()
    build_table()
    print(f"\nsaved -> {os.path.join(RES, 'LABELEFF_TABLE.md')}")


if __name__ == '__main__':
    main()
