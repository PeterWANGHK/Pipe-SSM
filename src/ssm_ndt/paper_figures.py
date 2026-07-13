"""Paper-grade qualitative figure: joint segmentation + severity classification on a HELD-OUT
campaign, with ground-truth overlay and a selective-prediction (abstention) panel.

IEEE/SciencePlots styling, Okabe-Ito colorblind-safe palette, physical chainage axis, panel labels.

  python -m ssm_ndt.paper_figures --ckpt models/sstssm_proposed.pt --file 88005
"""
from __future__ import annotations
import os, sys, argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ssm_ndt.data import FeatureConfig, load_raw, resample_uniform, build_features, physics_signals
from ssm_ndt.model import SSTSSM
from ssm_ndt.metrics import peaks_from_field

PYDIR = os.path.dirname(os.path.abspath(__file__)); REPO_PY = os.path.dirname(PYDIR)
REPO = os.path.dirname(REPO_PY); FIG = os.path.join(REPO, 'figures'); os.makedirs(FIG, exist_ok=True)

# Okabe-Ito colorblind-safe; classes {0 normal, 1 low, 2 high}
CMAP = {0: '#009E73', 1: '#E69F00', 2: '#D55E00', 3: '#9D0208'}
CNAME = {0: 'Normal', 1: 'Low severity', 2: 'High severity', 3: 'Sev-3'}


def style():
    try:
        import scienceplots  # noqa
        plt.style.use(['science', 'no-latex'])
    except Exception:
        pass
    plt.rcParams.update({'figure.dpi': 120, 'font.size': 8, 'axes.titlesize': 8,
                         'axes.labelsize': 8, 'legend.fontsize': 6.5, 'lines.linewidth': 0.8})


@torch.no_grad()
def run(ckpt, fid, device):
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = FeatureConfig(**ck['cfg'])
    n_classes = ck['state_dict']['class_head.weight'].shape[0]
    p = os.path.join(REPO_PY, f'merged_data_with_fault_classes_{fid}.csv')
    X, fault, labelnum, dist, _ = load_raw(p)
    X, fault, labelnum, spacing = resample_uniform(X, fault, labelnum, dist if cfg.velocity_norm else None,
                                                   cfg.grid_spacing_m)
    if n_classes == 3:
        fault = np.where(fault == 3, 2, fault)
    F, _ = build_features(X, fault, cfg)
    if cfg.physics_features:
        pi, _ = physics_signals(X); pi = (pi - pi.mean(0)) / (pi.std(0) + 1e-6)
        F = np.concatenate([F, pi.astype(np.float32)], 1)
    model = SSTSSM(ck['feat_dim'], d_model=ck['d_model'], n_layers=ck['n_layers'],
                   backbone=ck['backbone'], coupled=ck['coupled'], n_classes=n_classes).to(device).eval()
    model.load_state_dict(ck['state_dict'])
    Ft = torch.from_numpy(F).float().unsqueeze(0).to(device)
    N = Ft.shape[1]; W = cfg.window
    bprob = np.zeros(N); cprob = np.zeros((N, n_classes))
    for s in range(0, N, W):
        bl, cl = model(Ft[:, s:s + W])
        bprob[s:s + bl.shape[1]] = torch.sigmoid(bl)[0, :bl.shape[1]].cpu().numpy()
        cprob[s:s + cl.shape[1]] = torch.softmax(cl, -1)[0, :cl.shape[1]].cpu().numpy()
    return X, fault, labelnum, spacing, bprob, cprob, n_classes


def seg_classes(labelnum, fault):
    """ground-truth per-segment class + boundary indices from LabelNumber."""
    bnds = np.where(np.diff(labelnum) != 0)[0] + 1
    edges = np.concatenate(([0], bnds, [len(labelnum)]))
    segs = [(a, b, int(np.bincount(fault[a:b]).argmax())) for a, b in zip(edges[:-1], edges[1:]) if b > a]
    return bnds, segs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=os.path.join(REPO_PY, 'models', 'sstssm_proposed.pt'))
    ap.add_argument('--file', default='88005')
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--len', type=int, default=6000)
    ap.add_argument('--abstain', type=float, default=0.6, help='confidence threshold for flag-for-review')
    a = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    style()
    X, fault, labelnum, spacing, bprob, cprob, ncl = run(a.ckpt, a.file, device)

    s0, s1 = a.start, min(a.start + a.len, len(X))
    sl = slice(s0, s1)
    disp = spacing if spacing != 1.0 else 0.018      # nominal 18 mm/sample when no odometry
    xl = 'Chainage (m)' if spacing != 1.0 else 'Approx. chainage (m, nominal 18 mm/sample)'
    x = np.arange(s0, s1) * disp

    gt_bnds, gt_segs = seg_classes(labelnum, fault)
    ypred = cprob.argmax(1); conf = cprob.max(1)
    pred_bnds = peaks_from_field(bprob, prominence=0.2, distance=40)
    pr_edges = np.concatenate(([0], np.sort(pred_bnds), [len(X)])).astype(int)
    pred_seg_cls = np.zeros(len(X), int)
    for aa, bb in zip(pr_edges[:-1], pr_edges[1:]):
        if bb > aa:
            pred_seg_cls[aa:bb] = np.bincount(ypred[aa:bb], minlength=ncl).argmax()

    fig, ax = plt.subplots(4, 1, figsize=(7.2, 6.6), sharex=True,
                           gridspec_kw={'height_ratios': [2, 2, 1.1, 1.1]})

    # (a) D1 signals + GT class shading + GT/pred joints
    ax[0].plot(x, X[sl, 0], color='#0072B2', lw=0.7, label='D1 32 Hz |Z|')
    ax[0].plot(x, X[sl, 2], color='#56B4E9', lw=0.6, alpha=0.8, label='D1 100 Hz |Z|')
    for (aa, bb, c) in gt_segs:
        if c > 0 and bb > s0 and aa < s1:
            ax[0].axvspan(max(x[0], aa * disp), min(x[-1], bb * disp), color=CMAP[c], alpha=0.16, lw=0)
    ax[0].set_ylabel('Detector D1'); ax[0].set_title('(a) ECT signal with ground-truth severity shading', loc='left')

    # (b) D2 + predicted joints (dashed) vs GT joints (solid)
    ax[1].plot(x, X[sl, 4], color='#009E73', lw=0.7, label='D2 32 Hz |Z|')
    ax[1].plot(x, X[sl, 6], color='#66C2A5', lw=0.6, alpha=0.8, label='D2 100 Hz |Z|')
    for b in gt_bnds:
        if s0 <= b < s1:
            ax[1].axvline(b * disp, color='0.4', lw=0.7, alpha=0.8)
    for b in pred_bnds:
        if s0 <= b < s1:
            ax[1].axvline(b * disp, color='k', ls='--', lw=0.7, alpha=0.9)
    ax[1].set_ylabel('Detector D2'); ax[1].set_title('(b) Joint localization: GT (grey) vs predicted (dashed)', loc='left')

    # (c) per-sample class: GT vs predicted
    ax[2].step(x, fault[sl], color='0.3', lw=1.0, where='mid', label='Ground truth')
    ax[2].step(x, pred_seg_cls[sl], color='#D55E00', lw=1.0, ls='--', where='mid', label='Predicted')
    ax[2].set_ylabel('Severity'); ax[2].set_yticks(range(ncl)); ax[2].set_ylim(-0.3, ncl - 0.7)
    ax[2].set_title('(c) Severity classification (per segment)', loc='left'); ax[2].legend(loc='upper right', ncol=2)

    # (d) selective prediction confidence + flagged-for-review band
    ax[3].plot(x, conf[sl], color='#6a0dad', lw=0.8, label='Confidence')
    ax[3].axhline(a.abstain, color='red', ls=':', lw=0.8, label=f'abstain < {a.abstain}')
    flag = conf[sl] < a.abstain
    ax[3].fill_between(x, 0, 1, where=flag, color='red', alpha=0.12, step='mid', label='flag for review')
    ax[3].set_ylim(0, 1.02); ax[3].set_ylabel('Confidence'); ax[3].set_xlabel(xl)
    ax[3].set_title('(d) Selective prediction: low-confidence regions flagged for expert review', loc='left')
    ax[3].legend(loc='lower right', ncol=3)

    for a_ in ax:
        a_.grid(True, alpha=0.25, lw=0.4)
    handles = [Patch(facecolor=CMAP[c], alpha=0.4, label=CNAME[c]) for c in range(1, ncl)]
    handles += [Line2D([0], [0], color='0.4', lw=1, label='GT joint'),
                Line2D([0], [0], color='k', ls='--', lw=1, label='Predicted joint')]
    fig.legend(handles=handles, loc='upper center', ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, 1.005))
    fig.suptitle(f'SST-SSM on a held-out inspection campaign (segmentation + classification + triage)',
                 y=1.03, fontsize=8.5)
    fig.tight_layout()
    out = os.path.join(FIG, f'paper_segcls_{a.file}.png')
    fig.savefig(out, dpi=400, bbox_inches='tight')
    fig.savefig(out.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"saved -> {out}  (+ .pdf)")


if __name__ == '__main__':
    main()
