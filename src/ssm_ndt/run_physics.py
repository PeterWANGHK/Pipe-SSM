"""Physics-informed experiment: does impedance-plane physics (features + skin-depth constraint)
improve defect classification? All variants use regroup {0,low,high}; OOD test=88005 (has high
severity = orig class-2). Resumable. Honest test of whether 'physics-informed' is useful, not decor.
"""
from __future__ import annotations
import os, sys, json, subprocess, time, argparse

PYDIR = os.path.dirname(os.path.abspath(__file__)); REPO_PY = os.path.dirname(PYDIR)
RES = os.path.join(REPO_PY, 'results'); os.makedirs(RES, exist_ok=True)
PY = sys.executable
ENV = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUTF8='1')

TRAIN_IDS = ['102', '104', '021', '88003', '84001', '87001', '89005', '89006', '810001', '014']
OOD_TEST_IDS = ['88005']
COMMON = ['--train-ids', *TRAIN_IDS, '--test-ids', *OOD_TEST_IDS, '--regroup',
          '--window', '512', '--stride', '384', '--epochs', '8',
          '--d-model', '64', '--layers', '3', '--batch', '16', '--seed', '42']

RUNS = [
    ('phys_base',     []),                                           # regroup, no physics
    ('phys_feat',     ['--physics-features']),                       # + impedance-plane features
    ('phys_featloss', ['--physics-features', '--physics-loss']),     # + skin-depth constraint
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
    print(f"\n>>> {tag}: {' '.join(extra) or '(no physics)'}", flush=True); t0 = time.time()
    p = subprocess.run(cmd, cwd=REPO_PY, capture_output=True, text=True, env=ENV)
    if p.returncode != 0:
        print(f"  FAILED {tag}\n{p.stdout[-900:]}\n{p.stderr[-500:]}", flush=True); return None
    for ln in p.stdout.splitlines():
        if ln.strip().startswith(('Localization', 'Classification')):
            print('  ' + ln.strip(), flush=True)
    print(f"  {time.time()-t0:.0f}s", flush=True)
    return valid(jp)


def build_table():
    names = {'phys_base': 'SST-SSM (regroup, no physics)', 'phys_feat': '+ physics features',
             'phys_featloss': '+ physics features + skin-depth constraint'}
    md = ["# Physics-informed experiment (regroup {0,low,high}; OOD test=88005)", "",
          "Honest test: do impedance-plane physics features + a skin-depth severity constraint "
          "improve defect classification? per-class = [normal, low, high].", "",
          "| Config | Macro-F1 | Defect-F1 | per-class [0,low,high] | High-F1 | Above-thr | Dist(m) |",
          "|---|---|---|---|---|---|---|"]
    for tag in ['phys_base', 'phys_feat', 'phys_featloss']:
        r = valid(os.path.join(RES, f'{tag}.json'))
        if not r:
            md.append(f"| {names[tag]} | - | - | - | - | - | - |"); continue
        pc = r['F1_per_class']
        md.append(f"| {names[tag]} | {r['F1_macro']:.3f} | {sum(pc[1:3])/2:.3f} | "
                  f"{[round(x,3) for x in pc[:3]]} | {pc[2]:.3f} | {r['above_thresh']:.3f} | {r['dev_distance_m']:.3f} |")
    txt = "\n".join(md)
    open(os.path.join(RES, 'PHYSICS_TABLE.md'), 'w', encoding='utf-8').write(txt)
    print("\n" + txt)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--table', action='store_true'); a = ap.parse_args()
    if not a.table:
        for tag, extra in RUNS:
            run(tag, extra); build_table()
    build_table()
    print(f"\nsaved -> {os.path.join(RES, 'PHYSICS_TABLE.md')}")


if __name__ == '__main__':
    main()
