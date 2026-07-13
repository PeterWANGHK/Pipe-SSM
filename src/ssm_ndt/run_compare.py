"""Head-to-head JOINT-TASK comparison on the ENRICHED dataset with a DEFECT-RICH held-out campaign.

Proposed SST-SSM vs baselines (CNN-LSTM, TCN, LSTM, GRU, Transformer) — all trained on the same
data, same coupled seg+cls heads, same metrics — plus the key imbalance/OOD ablations. Resumable.
Reports Macro-F1 and defect-class F1 (the honest metrics) rather than imbalance-gamed F1w.

  python -m ssm_ndt.run_compare            # run all (resumable)
  python -m ssm_ndt.run_compare --table    # rebuild table from disk
"""
from __future__ import annotations
import os, sys, json, subprocess, time, argparse

PYDIR = os.path.dirname(os.path.abspath(__file__)); REPO_PY = os.path.dirname(PYDIR)
RES = os.path.join(REPO_PY, 'results'); MODELS = os.path.join(REPO_PY, 'models')
os.makedirs(RES, exist_ok=True); os.makedirs(MODELS, exist_ok=True)
PY = sys.executable
ENV = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUTF8='1')

# enriched, defect-bearing train pool; defect-rich held-out OOD campaign (classes 1&2)
TRAIN_IDS = ['102', '104', '86005', '88003', '84001', '87001', '89005', '89006', '810001', '014']
OOD_TEST_IDS = ['88005']
COMMON = ['--train-ids', *TRAIN_IDS, '--test-ids', *OOD_TEST_IDS,
          '--window', '512', '--stride', '384', '--epochs', '8',
          '--d-model', '64', '--layers', '3', '--batch', '16', '--seed', '42']

RUNS = [
    # tag,            extra flags                         (proposed + baselines on the joint task)
    ('cmp_sstssm',     ['--backbone', 'ssm']),
    ('cmp_cnnlstm',    ['--backbone', 'cnnlstm']),
    ('cmp_tcn',        ['--backbone', 'tcn']),
    ('cmp_lstm',       ['--backbone', 'lstm']),
    ('cmp_gru',        ['--backbone', 'gru']),
    ('cmp_transformer',['--backbone', 'transformer']),
    # key ablations of the proposed model
    ('cmp_nofocal',    ['--backbone', 'ssm', '--no-focal']),
    ('cmp_nobalance',  ['--backbone', 'ssm', '--no-balanced-sampler']),
    ('cmp_noinorm',    ['--backbone', 'ssm', '--no-instance-norm']),
    ('cmp_nocommonmode',['--backbone', 'ssm', '--no-common-mode']),
]


def valid(p):
    try:
        r = json.load(open(p)); return r if 'F1_macro' in r else None
    except Exception:
        return None


def run(tag, extra, ckpt=None):
    jp = os.path.join(RES, f'{tag}.json')
    v = valid(jp)
    if v is not None:
        print(f"  [skip] {tag}"); return v
    cmd = [PY, '-u', '-m', 'ssm_ndt.train', *COMMON, *extra, '--out', tag]
    if ckpt:
        cmd += ['--save-ckpt', ckpt]
    print(f"\n>>> {tag}: {' '.join(extra)}", flush=True); t0 = time.time()
    p = subprocess.run(cmd, cwd=REPO_PY, capture_output=True, text=True, env=ENV)
    if p.returncode != 0:
        print(f"  FAILED {tag}\n{p.stdout[-900:]}\n{p.stderr[-600:]}", flush=True); return None
    for ln in p.stdout.splitlines():
        if ln.strip().startswith(('Localization', 'Classification')):
            print('  ' + ln.strip(), flush=True)
    print(f"  {time.time()-t0:.0f}s", flush=True)
    return valid(jp)


def defectF1(r):
    pc = r['F1_per_class']; return sum(pc[1:]) / 3.0


def build_table():
    md = [f"# Joint-Task Comparison — enriched data, defect-rich OOD (train={TRAIN_IDS}, test={OOD_TEST_IDS})", "",
          "Honest metrics: **Macro-F1** and **Defect-F1** (mean F1 of classes 1-3); F1w is shown but is "
          "gamed by the normal majority. Localization: Above-thr (recall@0.5m), Distance (m).", "",
          "| Model / Config | Macro-F1 | Defect-F1 | per-class F1 [0,1,2,3] | Above-thr | Dist(m) | PropPrec | F1w |",
          "|---|---|---|---|---|---|---|---|"]
    order = ['cmp_sstssm', 'cmp_cnnlstm', 'cmp_tcn', 'cmp_lstm', 'cmp_gru', 'cmp_transformer',
             'cmp_nofocal', 'cmp_nobalance', 'cmp_noinorm', 'cmp_nocommonmode']
    names = {'cmp_sstssm': '**SST-SSM (proposed)**', 'cmp_cnnlstm': 'CNN-LSTM', 'cmp_tcn': 'TCN',
             'cmp_lstm': 'BiLSTM', 'cmp_gru': 'BiGRU', 'cmp_transformer': 'Transformer',
             'cmp_nofocal': 'SST-SSM w/o focal', 'cmp_nobalance': 'SST-SSM w/o balanced-sampler',
             'cmp_noinorm': 'SST-SSM w/o instance-norm', 'cmp_nocommonmode': 'SST-SSM w/o common-mode'}
    for tag in order:
        r = valid(os.path.join(RES, f'{tag}.json'))
        if not r:
            md.append(f"| {names[tag]} | - | - | - | - | - | - | - |"); continue
        pc = [round(x, 3) for x in r['F1_per_class']]
        md.append(f"| {names[tag]} | {r['F1_macro']:.3f} | {defectF1(r):.3f} | {pc} | "
                  f"{r['above_thresh']:.3f} | {r['dev_distance_m']:.3f} | {r['prop_precision']:.3f} | {r['F1_weighted']:.3f} |")
    txt = "\n".join(md)
    open(os.path.join(RES, 'COMPARISON_TABLE.md'), 'w', encoding='utf-8').write(txt)
    print("\n" + txt)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--table', action='store_true'); a = ap.parse_args()
    if not a.table:
        for i, (tag, extra) in enumerate(RUNS):
            ck = os.path.join(MODELS, 'sstssm_enriched.pt') if tag == 'cmp_sstssm' else None
            run(tag, extra, ckpt=ck); build_table()
    build_table()
    print(f"\nsaved -> {os.path.join(RES, 'COMPARISON_TABLE.md')}")


if __name__ == '__main__':
    main()
