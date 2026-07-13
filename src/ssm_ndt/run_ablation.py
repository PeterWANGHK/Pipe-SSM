"""Run the ablation matrix (OOD-evaluated) + final full-dataset model. RESUMABLE.

Each ablation row = train on TRAIN_IDS, evaluate on held-out OOD_TEST_IDS (leakage-free).
Configs whose results/<tag>.json already exists (and is valid) are SKIPPED, so the sweep can
be driven in short foreground chunks that survive session teardown. The summary table is
(re)built from whatever JSONs exist on each invocation.

  python -m ssm_ndt.run_ablation --stage ablation   # sweep remaining configs (resumable)
  python -m ssm_ndt.run_ablation --stage final      # train full-dataset model + checkpoint
  python -m ssm_ndt.run_ablation --stage table      # rebuild table from disk only
"""
from __future__ import annotations
import os, sys, json, subprocess, time, argparse

PYDIR = os.path.dirname(os.path.abspath(__file__))
REPO_PY = os.path.dirname(PYDIR)
RESULTS = os.path.join(REPO_PY, 'results')
MODELS = os.path.join(REPO_PY, 'models')
os.makedirs(RESULTS, exist_ok=True); os.makedirs(MODELS, exist_ok=True)
PY = sys.executable

TRAIN_IDS = ['102', '104', '014', '021', '022', '023', '004']
OOD_TEST_IDS = ['88003', '86002']
ALL_IDS = ['001', '004', '014', '015', '021', '022', '023', '102', '104',
           '105', '815001', '815002', '86002', '86005', '88003']
COMMON = ['--window', '512', '--stride', '384', '--epochs', '6',
          '--d-model', '64', '--layers', '3', '--batch', '16', '--seed', '42']

ABLATIONS = [
    ('full',            []),
    ('wo_velocity',     ['--no-velocity-norm']),
    ('wo_instancenorm', ['--no-instance-norm']),
    ('wo_commonmode',   ['--no-common-mode']),
    ('singlefreq32',    ['--freqs', '32']),
    ('wo_contrast',     ['--no-contrast']),
    ('backbone_gru',    ['--backbone', 'gru']),
    ('backbone_tfm',    ['--backbone', 'transformer']),
    ('decoupled',       ['--decoupled']),
]


def valid(jpath):
    try:
        r = json.load(open(jpath))
        return r if 'F1_weighted' in r else None
    except Exception:
        return None


def run(tag, extra, train_ids, test_ids, save_ckpt=None, force=False):
    jpath = os.path.join(RESULTS, f'{tag}.json')
    if not force:
        v = valid(jpath)
        if v is not None:
            print(f"  [skip] {tag} (already done)")
            return v
    cmd = [PY, '-u', '-m', 'ssm_ndt.train', '--train-ids', *train_ids, '--test-ids', *test_ids,
           *COMMON, *extra, '--out', tag]
    if save_ckpt:
        cmd += ['--save-ckpt', save_ckpt]
    print(f"\n>>> [{tag}] {' '.join(extra) or '(full)'}", flush=True)
    t0 = time.time()
    p = subprocess.run(cmd, cwd=REPO_PY, capture_output=True, text=True)
    if p.returncode != 0:
        print(f"  FAILED ({tag}):\n{p.stdout[-1200:]}\n{p.stderr[-800:]}", flush=True)
        return None
    for line in p.stdout.splitlines():
        if line.strip().startswith(('Localization', 'Classification')):
            print('  ' + line.strip(), flush=True)
    print(f"  done in {time.time()-t0:.0f}s", flush=True)
    return valid(jpath)


def fmt(res):
    if not res:
        return dict(drift='-', dist='-', pct='-', minmax='-', above='-', pp='-', f1w='-')
    return dict(drift=f"{res['dev_drift_samples']:+.1f}", dist=f"{res['dev_distance_m']:.3f}",
                pct=f"{res['dev_percentage']:.2f}", minmax=f"{res['dev_min_m']:.2f}/{res['dev_max_m']:.2f}",
                above=f"{res['above_thresh']:.3f}", pp=f"{res['prop_precision']:.3f}",
                f1w=f"{res['F1_weighted']:.3f}")


def build_table():
    rows = []
    for tag, _ in ABLATIONS:
        rows.append((tag, valid(os.path.join(RESULTS, f'{tag}.json'))))
    fd = valid(os.path.join(RESULTS, 'full_dataset.json'))
    if fd:
        rows.append(('full_dataset(all)', fd))
    md = [f"# Ablation Matrix (OOD: train→{TRAIN_IDS}, test→{OOD_TEST_IDS})", "",
          "Leakage-free cross-campaign. Drift in samples; Distance/Min/Max in meters "
          "(nominal 18 mm/sample where odometry absent).", "",
          "| Config | Drift(smp) | Distance(m) | Dev% | Min/Max(m) | Above-thr | PropPrec | F1w |",
          "|---|---|---|---|---|---|---|---|"]
    for tag, res in rows:
        v = fmt(res)
        md.append(f"| {tag} | {v['drift']} | {v['dist']} | {v['pct']} | {v['minmax']} | {v['above']} | {v['pp']} | {v['f1w']} |")
    txt = "\n".join(md)
    open(os.path.join(RESULTS, 'ABLATION_TABLE.md'), 'w', encoding='utf-8').write(txt)
    print("\n" + txt)
    return txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['ablation', 'final', 'table', 'all'], default='all')
    ap.add_argument('--max', type=int, default=99, help='run at most this many not-yet-done configs')
    args = ap.parse_args()
    if args.stage in ('ablation', 'all'):
        ran = 0
        for tag, extra in ABLATIONS:
            if valid(os.path.join(RESULTS, f'{tag}.json')) is not None:
                continue
            if ran >= args.max:
                print(f"  [--max {args.max} reached; rerun to continue]"); break
            run(tag, extra, TRAIN_IDS, OOD_TEST_IDS); ran += 1
            build_table()
    if args.stage in ('final', 'all'):
        ckpt = os.path.join(MODELS, 'sstssm_full.pt')
        if not os.path.exists(ckpt):
            run('full_dataset', [], ALL_IDS, OOD_TEST_IDS, save_ckpt=ckpt)
        else:
            print(f"  [skip] checkpoint exists: {ckpt}")
    build_table()
    print(f"\nsaved -> {os.path.join(RESULTS, 'ABLATION_TABLE.md')}")


if __name__ == '__main__':
    main()
