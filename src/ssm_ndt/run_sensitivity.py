"""Parameter-sensitivity suite (reviewer backup). Resumable; builds results/SENSITIVITY_TABLE.md.

Stage A — physics-parameter sensitivity (inversion only, one-factor-at-a-time around the center):
    shell thickness {0.7, 1.4, 2.8 mm}, mu_r {25, 50, 100}, calibration prior {0.1, 0.3, 1.0},
    sigma-mu smoothness {2, 5, 15}. Metric: cross-campaign attribution ratio
    (std of campaign-mean sigma_mu / mean within-campaign std) over 4 campaigns.
Stage B — SSM hyperparameter sensitivity (fold 88005, seed 42, physics-inversion features):
    window {256,512,1024}, d_model {32,64,128}, boundary pos-weight {5,10,20}.
Stage C — localization tolerance sensitivity: Above-thr at tol {0.25, 0.5, 1.0} m for the
    fold-88005 checkpoint.
"""
from __future__ import annotations
import os, sys, json, subprocess, time, argparse
import numpy as np

PYDIR = os.path.dirname(os.path.abspath(__file__)); REPO_PY = os.path.dirname(PYDIR)
RES = os.path.join(REPO_PY, 'results'); os.makedirs(RES, exist_ok=True)
PY = sys.executable
ENV = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUTF8='1')
SFILE = os.path.join(RES, 'sensitivity.json')

A_CAMPAIGNS = ['102', '88005', '86005', '87001']
A_CENTER = dict(thickness=0.0014, mu_r=50.0, cal_prior_w=0.3, smooth_sm_w=5.0)
A_SWEEPS = {'thickness': [0.0007, 0.0028], 'mu_r': [25.0, 100.0],
            'cal_prior_w': [0.1, 1.0], 'smooth_sm_w': [2.0, 15.0]}

B_TRAIN = ['102', '104', '021', '88003', '84001', '87001', '89005', '89006', '810001', '014', '86005']
B_CENTER = dict(window=512, d_model=64, bpw=10.0)
# window=1024, d_model=128 AND d_model=96 all exceed the bidirectional parallel-scan activation
# memory (~O(B*L*logL*d_inner*N) kept for autograd) on the 16 GB GPU (observed thrash at 15.9/16.3
# GB). Upward sweeps capped at feasibility: window {256,384,512}, d_model {32,64}. Documented as a
# practical scaling limit of the pure-PyTorch scan (a fused kernel, e.g. official mamba-ssm,
# removes it but is uninstallable on this Windows box).
B_SWEEPS = {'window': [256, 384], 'd_model': [32], 'bpw': [5.0, 20.0]}


def load_state():
    try:
        return json.load(open(SFILE))
    except Exception:
        return {}


def save_state(st):
    json.dump(st, open(SFILE, 'w'), indent=1)


def stageA():
    sys.path.insert(0, REPO_PY)
    from ssm_ndt.data import load_raw
    from ssm_ndt.latent_inversion import invert_sequence
    st = load_state(); st.setdefault('A', {})
    configs = [('center', A_CENTER)]
    for k, vals in A_SWEEPS.items():
        for v in vals:
            c = dict(A_CENTER); c[k] = v
            configs.append((f'{k}={v}', c))
    Xs = {}
    for cid in A_CAMPAIGNS:
        X, *_ = load_raw(os.path.join(REPO_PY, f'merged_data_with_fault_classes_{cid}.csv'))
        Xs[cid] = X
    for name, c in configs:
        if name in st['A']:
            print(f"  [skip] A:{name}"); continue
        t0 = time.time(); means, withins = [], []
        for cid in A_CAMPAIGNS:
            th, _, _ = invert_sequence(Xs[cid], iters=300, **c)
            means.append(float(np.mean(th[:, 0]))); withins.append(float(np.std(th[:, 0])))
        ratio = float(np.std(means) / (np.mean(withins) + 1e-9))
        st['A'][name] = dict(spread=float(np.std(means)), within=float(np.mean(withins)), ratio=ratio)
        save_state(st)
        print(f"  A:{name}: spread={np.std(means):.3f} within={np.mean(withins):.3f} "
              f"ratio={ratio:.1f} ({time.time()-t0:.0f}s)", flush=True)


def stageB():
    st = load_state(); st.setdefault('B', {})
    configs = [('center', B_CENTER)]
    for k, vals in B_SWEEPS.items():
        for v in vals:
            c = dict(B_CENTER); c[k] = v
            configs.append((f'{k}={v}', c))
    for name, c in configs:
        if name in st['B']:
            print(f"  [skip] B:{name}"); continue
        tag = f"sens_{name.replace('=', '_').replace('.', 'p')}"
        ckpt = ['--save-ckpt', os.path.join(REPO_PY, 'models', 'sens_88005.pt')] if name == 'center' else []
        cmd = [PY, '-u', '-m', 'ssm_ndt.train', '--train-ids', *B_TRAIN, '--test-ids', '88005',
               '--regroup', '--physics-inversion', '--window', str(c['window']), '--stride', '384',
               '--epochs', '8', '--d-model', str(c['d_model']), '--layers', '3', '--batch', '16',
               '--boundary-pos-weight', str(c['bpw']), '--seed', '42', '--out', tag, *ckpt]
        t0 = time.time()
        p = subprocess.run(cmd, cwd=REPO_PY, capture_output=True, text=True, env=ENV)
        if p.returncode != 0:
            print(f"  B:{name} FAILED\n{p.stderr[-300:]}", flush=True); continue
        r = json.load(open(os.path.join(RES, f'{tag}.json')))
        pc = r['F1_per_class']
        st['B'][name] = dict(macro=r['F1_macro'], defect=sum(pc[1:3]) / 2,
                             above=r['above_thresh'], dist=r['dev_distance_m'])
        save_state(st)
        print(f"  B:{name}: macro={r['F1_macro']:.3f} defect={sum(pc[1:3])/2:.3f} "
              f"above={r['above_thresh']:.3f} ({time.time()-t0:.0f}s)", flush=True)


def stageC():
    """Tolerance sensitivity on the fold-88005 center checkpoint."""
    st = load_state()
    if 'C' in st:
        print("  [skip] C"); return
    sys.path.insert(0, REPO_PY)
    import torch
    from ssm_ndt.data import FeatureConfig, ECTWindows
    from ssm_ndt.model import SSTSSM
    from ssm_ndt.train import evaluate, id_to_path
    ck = torch.load(os.path.join(REPO_PY, 'models', 'sens_88005.pt'), map_location='cuda', weights_only=False)
    cfg = FeatureConfig(**ck['cfg'])
    ds = ECTWindows([id_to_path('88005')], cfg, train=False)
    for it in ds.items:
        it['fault'][it['fault'] == 3] = 2
    model = SSTSSM(ck['feat_dim'], d_model=ck['d_model'], n_layers=ck['n_layers'],
                   backbone=ck['backbone'], coupled=ck['coupled'], n_classes=3).to('cuda').eval()
    model.load_state_dict(ck['state_dict'])
    out = {}
    for tol in (0.25, 0.5, 1.0):
        r = evaluate(model, ds, cfg, 'cuda', tol_m=tol, n_classes=3)
        out[str(tol)] = dict(above=r['above_thresh'], dist=r['dev_distance_m'])
        print(f"  C: tol={tol}m -> Above-thr={r['above_thresh']:.3f} Dist={r['dev_distance_m']:.3f}", flush=True)
    st['C'] = out; save_state(st)


def build_table():
    st = load_state()
    md = ["# Parameter-Sensitivity Suite (reviewer backup)", ""]
    if 'A' in st:
        md += ["## A. Physics parameters — drift-attribution ratio (higher = clearer attribution)",
               "", "| Config | cross-campaign spread (dex) | within (dex) | ratio |", "|---|---|---|---|"]
        for k, v in st['A'].items():
            md.append(f"| {k} | {v['spread']:.3f} | {v['within']:.3f} | {v['ratio']:.1f} |")
        md.append("")
    if 'B' in st:
        md += ["## B. SSM hyperparameters (fold 88005, seed 42)", "",
               "| Config | Macro-F1 | Defect-F1 | Above-thr | Dist(m) |", "|---|---|---|---|---|"]
        for k, v in st['B'].items():
            md.append(f"| {k} | {v['macro']:.3f} | {v['defect']:.3f} | {v['above']:.3f} | {v['dist']:.3f} |")
        md.append("")
    if 'C' in st:
        md += ["## C. Localization tolerance sensitivity", "",
               "| tol (m) | Above-thr | Dist(m) |", "|---|---|---|"]
        for k, v in st['C'].items():
            md.append(f"| {k} | {v['above']:.3f} | {v['dist']:.3f} |")
    txt = "\n".join(md)
    open(os.path.join(RES, 'SENSITIVITY_TABLE.md'), 'w', encoding='utf-8').write(txt)
    print("\n" + txt)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--table', action='store_true'); a = ap.parse_args()
    if not a.table:
        print("== Stage A: physics parameters ==", flush=True); stageA()
        print("== Stage B: SSM hyperparameters ==", flush=True); stageB()
        print("== Stage C: tolerance ==", flush=True); stageC()
    build_table()


if __name__ == '__main__':
    main()
