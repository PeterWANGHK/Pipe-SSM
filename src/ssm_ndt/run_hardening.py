"""Hardening: multi-seed x multi-OOD-fold error bars for the headline comparison.

Configs: proposed (regroup + physics features + skin-depth constraint) vs no-physics vs Transformer.
Folds: hold out each defect-bearing campaign in {88005, 86005}. Seeds: {42,123,2024}.
Aggregates mean+/-std per (config,fold). Resumable.
"""
from __future__ import annotations
import os, sys, json, subprocess, time, argparse
import numpy as np

PYDIR = os.path.dirname(os.path.abspath(__file__)); REPO_PY = os.path.dirname(PYDIR)
RES = os.path.join(REPO_PY, 'results'); os.makedirs(RES, exist_ok=True)
PY = sys.executable
ENV = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUTF8='1')

MASTER = ['102', '104', '021', '88003', '84001', '87001', '89005', '89006', '810001', '014', '88005', '86005']
FOLDS = ['88005', '86005']
SEEDS = [42, 123, 2024]
CONFIGS = {
    'proposed': ['--regroup', '--physics-features', '--physics-loss'],
    'nophysics': ['--regroup'],
    'transformer': ['--regroup', '--backbone', 'transformer'],
}
BASE = ['--window', '512', '--stride', '384', '--epochs', '8', '--d-model', '64', '--layers', '3', '--batch', '16']


def valid(p):
    try:
        r = json.load(open(p)); return r if 'F1_macro' in r else None
    except Exception:
        return None


def run(cfg, fold, seed):
    tag = f'hard_{cfg}_{fold}_s{seed}'
    jp = os.path.join(RES, f'{tag}.json')
    if valid(jp) is not None:
        print(f"  [skip] {tag}"); return valid(jp)
    train = [i for i in MASTER if i != fold]
    cmd = [PY, '-u', '-m', 'ssm_ndt.train', '--train-ids', *train, '--test-ids', fold,
           *BASE, '--seed', str(seed), *CONFIGS[cfg], '--out', tag]
    print(f">>> {tag}", flush=True); t0 = time.time()
    p = subprocess.run(cmd, cwd=REPO_PY, capture_output=True, text=True, env=ENV)
    if p.returncode != 0:
        print(f"  FAILED {tag}\n{p.stderr[-400:]}", flush=True); return None
    print(f"  {time.time()-t0:.0f}s", flush=True)
    return valid(jp)


def agg(cfg, fold):
    vals = {'macro': [], 'defect': [], 'high': [], 'above': [], 'dist': []}
    for s in SEEDS:
        r = valid(os.path.join(RES, f'hard_{cfg}_{fold}_s{s}.json'))
        if not r:
            continue
        pc = r['F1_per_class']
        vals['macro'].append(r['F1_macro']); vals['defect'].append(sum(pc[1:3]) / 2)
        vals['high'].append(pc[2]); vals['above'].append(r['above_thresh']); vals['dist'].append(r['dev_distance_m'])
    return vals


def ms(a):
    return f"{np.mean(a):.3f}±{np.std(a):.3f}" if a else "-"


def build_table():
    md = ["# Hardening: multi-seed (42/123/2024) x multi-fold error bars", "",
          "Mean±std over 3 seeds. Folds = held-out campaign. per-metric on regroup {0,low,high}.", ""]
    for fold in FOLDS:
        md += [f"## Held-out test = {fold}", "",
               "| Config | Macro-F1 | Defect-F1 | High-F1 | Above-thr | Dist(m) |",
               "|---|---|---|---|---|---|"]
        for cfg in ['proposed', 'nophysics', 'transformer']:
            v = agg(cfg, fold)
            name = {'proposed': '**SST-SSM + physics (proposed)**', 'nophysics': 'SST-SSM (no physics)',
                    'transformer': 'Transformer'}[cfg]
            md.append(f"| {name} | {ms(v['macro'])} | {ms(v['defect'])} | {ms(v['high'])} | {ms(v['above'])} | {ms(v['dist'])} |")
        md.append("")
    txt = "\n".join(md)
    open(os.path.join(RES, 'HARDENING_TABLE.md'), 'w', encoding='utf-8').write(txt)
    print("\n" + txt)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--table', action='store_true'); a = ap.parse_args()
    if not a.table:
        for fold in FOLDS:
            for cfg in CONFIGS:
                for seed in SEEDS:
                    run(cfg, fold, seed)
                build_table()
    build_table()
    print(f"\nsaved -> {os.path.join(RES, 'HARDENING_TABLE.md')}")


if __name__ == '__main__':
    main()
