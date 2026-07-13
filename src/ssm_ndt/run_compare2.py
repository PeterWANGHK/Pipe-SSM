"""Cycle-2 comparison: GAN-class-3 ablation + PatchTST + Mamba(ours) + baselines.

Held-out OOD test = 86005 (the held-out campaign that CONTAINS class-3) so the GAN's effect on the
ultra-rare class is measurable. The GAN is trained EXCLUDING 86005 (no leakage). Reports class-3 F1
explicitly. Resumable.
"""
from __future__ import annotations
import os, sys, json, subprocess, time, argparse

PYDIR = os.path.dirname(os.path.abspath(__file__)); REPO_PY = os.path.dirname(PYDIR)
RES = os.path.join(REPO_PY, 'results'); os.makedirs(RES, exist_ok=True)
PY = sys.executable
ENV = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUTF8='1')
SYN = os.path.join(REPO_PY, 'synthetic_class3_no86005.npy')

# train pool keeps the class-3 sources 021/102/104 (+ defect-bearing variety); test 86005 held out
TRAIN_IDS = ['102', '104', '021', '88003', '84001', '87001', '89005', '89006', '810001', '014', '88005']
OOD_TEST_IDS = ['86005']
COMMON = ['--train-ids', *TRAIN_IDS, '--test-ids', *OOD_TEST_IDS,
          '--window', '512', '--stride', '384', '--epochs', '8',
          '--d-model', '64', '--layers', '3', '--batch', '16', '--seed', '42']

RUNS = [
    ('c2_sstssm',       ['--backbone', 'ssm']),                       # = real Mamba (pure-torch parallel scan)
    ('c2_sstssm_gan',   ['--backbone', 'ssm', '--gan-synthetic', SYN]),
    ('c2_patchtst',     ['--backbone', 'patchtst']),
    ('c2_patchtst_gan', ['--backbone', 'patchtst', '--gan-synthetic', SYN]),
    ('c2_cnnlstm',      ['--backbone', 'cnnlstm']),
    ('c2_transformer',  ['--backbone', 'transformer']),
    ('c2_tcn',          ['--backbone', 'tcn']),
]


def valid(p):
    try:
        r = json.load(open(p)); return r if 'F1_macro' in r else None
    except Exception:
        return None


def run(tag, extra):
    jp = os.path.join(RES, f'{tag}.json')
    if valid(jp) is not None:
        print(f"  [skip] {tag}"); return valid(jp)
    cmd = [PY, '-u', '-m', 'ssm_ndt.train', *COMMON, *extra, '--out', tag]
    print(f"\n>>> {tag}: {' '.join(extra)}", flush=True); t0 = time.time()
    p = subprocess.run(cmd, cwd=REPO_PY, capture_output=True, text=True, env=ENV)
    if p.returncode != 0:
        print(f"  FAILED {tag}\n{p.stdout[-900:]}\n{p.stderr[-500:]}", flush=True); return None
    for ln in p.stdout.splitlines():
        if ln.strip().startswith(('Localization', 'Classification', '[data] injected')):
            print('  ' + ln.strip(), flush=True)
    print(f"  {time.time()-t0:.0f}s", flush=True)
    return valid(jp)


def build_table():
    names = {'c2_sstssm': '**SST-SSM (Mamba, ours)**', 'c2_sstssm_gan': '**SST-SSM + GAN-class3**',
             'c2_patchtst': 'PatchTST', 'c2_patchtst_gan': 'PatchTST + GAN-class3',
             'c2_cnnlstm': 'CNN-LSTM', 'c2_transformer': 'Transformer', 'c2_tcn': 'TCN'}
    md = [f"# Cycle-2: GAN-class3 + PatchTST + baselines (OOD test=86005, contains class-3)", "",
          "Held-out campaign 86005 contains class-3; GAN trained EXCLUDING 86005 (no leakage). "
          "**Class3-F1** is the key column for the rare-class question.", "",
          "| Model / Config | Macro-F1 | Defect-F1 | **Class3-F1** | per-class [0,1,2,3] | Above-thr | Dist(m) |",
          "|---|---|---|---|---|---|---|"]
    for tag in ['c2_sstssm', 'c2_sstssm_gan', 'c2_patchtst', 'c2_patchtst_gan', 'c2_cnnlstm', 'c2_transformer', 'c2_tcn']:
        r = valid(os.path.join(RES, f'{tag}.json'))
        if not r:
            md.append(f"| {names[tag]} | - | - | - | - | - | - |"); continue
        pc = r['F1_per_class']
        md.append(f"| {names[tag]} | {r['F1_macro']:.3f} | {sum(pc[1:])/3:.3f} | **{pc[3]:.3f}** | "
                  f"{[round(x,3) for x in pc]} | {r['above_thresh']:.3f} | {r['dev_distance_m']:.3f} |")
    txt = "\n".join(md)
    open(os.path.join(RES, 'COMPARISON2_TABLE.md'), 'w', encoding='utf-8').write(txt)
    print("\n" + txt)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--table', action='store_true'); a = ap.parse_args()
    if not a.table:
        for tag, extra in RUNS:
            run(tag, extra); build_table()
    build_table()
    print(f"\nsaved -> {os.path.join(RES, 'COMPARISON2_TABLE.md')}")


if __name__ == '__main__':
    main()
