"""E2/E3/E4 runner for the Dodd-Deeds physics-inversion module. Resumable, detached-friendly.

Stage A (prewarm): run latent inversion once per campaign -> physinv_cache/*.npz (+ E2 table).
Stage B (E3): OOD benefit — physics-inversion features vs statistical normalization.
              Baseline rows reuse existing hard_nophysics_* JSONs (same pools/hyperparams).
Stage C (E4): label-efficiency re-run with the inversion module vs existing le_*_nophysics rows.
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
BASE = ['--regroup', '--window', '512', '--stride', '384', '--epochs', '8',
        '--d-model', '64', '--layers', '3', '--batch', '16']
LE_SIZES = {'n2': ['102', '87001'], 'n4': ['102', '87001', '84001', '810001'],
            'nAll': ['102', '104', '021', '88003', '84001', '87001', '89005', '89006', '810001', '014']}


def valid(p):
    try:
        r = json.load(open(p)); return r if 'F1_macro' in r else None
    except Exception:
        return None


def prewarm():
    print("== Stage A: inversion prewarm ==", flush=True)
    sys.path.insert(0, REPO_PY)
    from ssm_ndt.data import load_raw
    from ssm_ndt.latent_inversion import get_or_invert, e2_drift_attribution
    for fid in MASTER:
        p = os.path.join(REPO_PY, f'merged_data_with_fault_classes_{fid}.csv')
        t0 = time.time()
        X, *_ = load_raw(p)
        theta, A = get_or_invert(fid, X)
        print(f"  [{fid}] N={len(X)} inverted/cached in {time.time()-t0:.0f}s "
              f"(sm={theta[:,0].mean():.2f}, lift={theta[:,1].mean()*1000:.0f}mm)", flush=True)
    print("\n== E2 drift attribution ==", flush=True)
    rows = e2_drift_attribution(tuple(MASTER))
    md = ["# E2: Physical drift attribution (Dodd-Deeds inversion)", "",
          "| Campaign | log10(sigma*mu)_eff | within-std | liftoff_eff (mm) | mean |anomaly| |", "|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r[0]} | {r[1]:.3f} | {r[2]:.3f} | {r[3]:.1f} | {r[4]:.4f} |")
    sms = [r[1] for r in rows]; wst = float(np.mean([r[2] for r in rows]))
    md += ["", f"Cross-campaign spread of log10(sigma_mu): **{np.std(sms):.3f} dex** vs within-campaign "
               f"{wst:.3f} -> campaign drift {'IS' if np.std(sms) > wst else 'is NOT clearly'} attributable "
               f"to effective material state."]
    open(os.path.join(RES, 'E2_DRIFT_ATTRIBUTION.md'), 'w', encoding='utf-8').write("\n".join(md))
    print("saved -> results/E2_DRIFT_ATTRIBUTION.md", flush=True)


def run(tag, train_ids, test_id, extra, seed):
    jp = os.path.join(RES, f'{tag}.json')
    if valid(jp) is not None:
        print(f"  [skip] {tag}"); return
    cmd = [PY, '-u', '-m', 'ssm_ndt.train', '--train-ids', *train_ids, '--test-ids', test_id,
           *BASE, '--seed', str(seed), *extra, '--out', tag]
    print(f">>> {tag}", flush=True); t0 = time.time()
    p = subprocess.run(cmd, cwd=REPO_PY, capture_output=True, text=True, env=ENV)
    if p.returncode != 0:
        print(f"  FAILED {tag}\n{p.stderr[-400:]}", flush=True); return
    print(f"  {time.time()-t0:.0f}s", flush=True)


def agg(pattern, seeds):
    out = {'macro': [], 'defect': [], 'high': [], 'above': [], 'dist': []}
    for s in seeds:
        r = valid(os.path.join(RES, pattern.format(seed=s)))
        if not r:
            continue
        pc = r['F1_per_class']
        out['macro'].append(r['F1_macro']); out['defect'].append(sum(pc[1:3]) / 2)
        out['high'].append(pc[2]); out['above'].append(r['above_thresh']); out['dist'].append(r['dev_distance_m'])
    return out


def ms(a):
    return f"{np.mean(a):.3f}±{np.std(a):.3f}" if a else "-"


def build_tables():
    md = ["# E3: OOD benefit of physics-inversion features (3 seeds)", ""]
    for fold in FOLDS:
        md += [f"## Held-out {fold}", "",
               "| Config | Macro-F1 | Defect-F1 | High-F1 | Above-thr | Dist(m) |", "|---|---|---|---|---|---|"]
        rows = [('statistical norm (baseline)', f'hard_nophysics_{fold}_s{{seed}}.json', SEEDS),
                ('+ physics-inversion (plus)', f'p2_pi_{fold}_s{{seed}}.json', SEEDS),
                ('physics-inversion (replaces common-mode)', f'p2_pirep_{fold}_s{{seed}}.json', SEEDS)]
        for name, pat, seeds in rows:
            v = agg(pat, seeds)
            md.append(f"| {name} | {ms(v['macro'])} | {ms(v['defect'])} | {ms(v['high'])} | {ms(v['above'])} | {ms(v['dist'])} |")
        md.append("")
    md += ["# E4: label-efficiency with physics-inversion (2 seeds)", "",
           "| #defect campaigns | no-physics | + physics-inversion | Δ |", "|---|---|---|---|"]
    for size in LE_SIZES:
        b = agg(f'le_{size}_nophysics_s{{seed}}.json', [42, 123])['defect']
        p = agg(f'p2_le_{size}_s{{seed}}.json', [42, 123])['defect']
        d = (np.mean(p) - np.mean(b)) if (b and p) else float('nan')
        md.append(f"| {len(LE_SIZES[size])} | {ms(b)} | {ms(p)} | {d:+.3f} |")
    txt = "\n".join(md)
    open(os.path.join(RES, 'PHYSICS2_TABLE.md'), 'w', encoding='utf-8').write(txt)
    print("\n" + txt)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--table', action='store_true'); a = ap.parse_args()
    if not a.table:
        prewarm()
        print("\n== Stage B: E3 ==", flush=True)
        for fold in FOLDS:
            train = [i for i in MASTER if i != fold]
            for seed in SEEDS:
                run(f'p2_pi_{fold}_s{seed}', train, fold, ['--physics-inversion'], seed)
                run(f'p2_pirep_{fold}_s{seed}', train, fold, ['--physics-inversion', '--no-common-mode'], seed)
            build_tables()
        print("\n== Stage C: E4 ==", flush=True)
        for size, ids in LE_SIZES.items():
            for seed in [42, 123]:
                run(f'p2_le_{size}_s{seed}', ids, '88005', ['--physics-inversion'], seed)
            build_tables()
    build_tables()


if __name__ == '__main__':
    main()
