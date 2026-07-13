"""Paper summary plots: (1) label-efficiency curve (physics vs no-physics), (2) baseline bar chart.
SciencePlots / Okabe-Ito, vector PDF output.

  python -m ssm_ndt.paper_plots
"""
from __future__ import annotations
import os, sys, json
import numpy as np
import matplotlib.pyplot as plt

PYDIR = os.path.dirname(os.path.abspath(__file__)); REPO_PY = os.path.dirname(PYDIR)
REPO = os.path.dirname(REPO_PY); FIG = os.path.join(REPO, 'figures'); RES = os.path.join(REPO_PY, 'results')
os.makedirs(FIG, exist_ok=True)


def style():
    try:
        import scienceplots  # noqa
        plt.style.use(['science', 'no-latex'])
    except Exception:
        pass
    plt.rcParams.update({'figure.dpi': 120, 'font.size': 8, 'legend.fontsize': 7})


def _val(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def labeleff_plot():
    sizes = {'n2': 2, 'n4': 4, 'nAll': 10}; seeds = [42, 123]
    def coll(size, cfg):
        xs = []
        for s in seeds:
            r = _val(os.path.join(RES, f'le_{size}_{cfg}_s{s}.json'))
            if r:
                pc = r['F1_per_class']; xs.append(sum(pc[1:3]) / 2)
        return (np.mean(xs), np.std(xs)) if xs else (np.nan, 0)
    N = [sizes[s] for s in sizes]
    base = [coll(s, 'nophysics') for s in sizes]; phys = [coll(s, 'physics') for s in sizes]
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    ax.errorbar(N, [m for m, _ in base], yerr=[e for _, e in base], marker='s', ms=4, capsize=2,
                color='#0072B2', label='No physics')
    ax.errorbar(N, [m for m, _ in phys], yerr=[e for _, e in phys], marker='o', ms=4, capsize=2,
                color='#D55E00', label='+ skin-depth physics')
    ax.set_xlabel('# defect-bearing training campaigns'); ax.set_ylabel('Defect-F1 (held-out 88005)')
    ax.set_xticks(N); ax.set_title('Physics aids defect detection under label scarcity')
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(FIG, 'paper_label_efficiency')
    fig.savefig(out + '.png', dpi=400, bbox_inches='tight'); fig.savefig(out + '.pdf', bbox_inches='tight')
    print(f"saved -> {out}.png (+ .pdf)")


def baseline_bar():
    """Hardened joint-task comparison on held-out 88005 (3 seeds): MAD (localization) + Defect-F1."""
    cfgs = [('proposed', 'SST-SSM'), ('nophysics', 'SST-SSM\n(no phys)'), ('transformer', 'Transformer')]
    fold = '88005'
    def agg(cfg, key):
        xs = []
        for s in [42, 123, 2024]:
            r = _val(os.path.join(RES, f'hard_{cfg}_{fold}_s{s}.json'))
            if r:
                xs.append(r['dev_distance_m'] if key == 'mad' else sum(r['F1_per_class'][1:3]) / 2)
        return (np.mean(xs), np.std(xs)) if xs else (np.nan, 0)
    labels = [n for _, n in cfgs]
    mad = [agg(c, 'mad') for c, _ in cfgs]; dfc = [agg(c, 'defect') for c, _ in cfgs]
    x = np.arange(len(cfgs))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 2.7))
    a1.bar(x, [m for m, _ in mad], yerr=[e for _, e in mad], capsize=3, color='#009E73')
    a1.set_xticks(x); a1.set_xticklabels(labels); a1.set_ylabel('Localization MAD (m) ↓')
    a1.set_title('(a) Joint localization'); a1.grid(True, axis='y', alpha=0.3)
    a2.bar(x, [m for m, _ in dfc], yerr=[e for _, e in dfc], capsize=3, color='#E69F00')
    a2.set_xticks(x); a2.set_xticklabels(labels); a2.set_ylabel('Defect-F1 ↑')
    a2.set_title('(b) Defect classification'); a2.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    out = os.path.join(FIG, 'paper_baseline_bars')
    fig.savefig(out + '.png', dpi=400, bbox_inches='tight'); fig.savefig(out + '.pdf', bbox_inches='tight')
    print(f"saved -> {out}.png (+ .pdf)")


if __name__ == '__main__':
    style()
    labeleff_plot()
    baseline_bar()
