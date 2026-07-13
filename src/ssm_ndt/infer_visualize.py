"""Run a trained SST-SSM on an UNLABELED D1/D2 parsed file and visualize segmentation + classification.

- Loads a checkpoint (models/sstssm_full.pt by default).
- Picks a random *_D1D2_parsed.txt (unlabeled) unless one is given.
- Reconstructs features exactly as in training (instance-norm => self-normalising, no labels needed).
- Plots D1/D2 time-series with predicted joint boundaries and per-segment defect-class shading.
- SciencePlots style; figure saved under pipeline_diagnostics/figures/.

Usage:
  python -m ssm_ndt.infer_visualize                       # random unlabeled file
  python -m ssm_ndt.infer_visualize --txt 002_D1D2_parsed.txt
"""
from __future__ import annotations
import os, sys, glob, random, argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ssm_ndt.data import FeatureConfig, build_features, resample_uniform, CH8
from ssm_ndt.model import SSTSSM
from ssm_ndt.metrics import peaks_from_field

PYDIR = os.path.dirname(os.path.abspath(__file__)); REPO_PY = os.path.dirname(PYDIR)
REPO = os.path.dirname(REPO_PY)
FIGDIR = os.path.join(REPO, 'figures'); os.makedirs(FIGDIR, exist_ok=True)

CLASS_COLORS = {0: '#2E8B57', 1: '#F4A261', 2: '#E76F51', 3: '#9D0208'}
CLASS_NAMES = {0: 'Normal', 1: 'Defect 1', 2: 'Defect 2', 3: 'Defect 3'}


def use_scienceplots():
    try:
        import scienceplots  # noqa
        plt.style.use(['science', 'no-latex', 'grid'])
        return True
    except Exception as e:
        print(f"[warn] scienceplots unavailable ({e}); default style.")
        return False


def load_unlabeled(txt_path):
    df = pd.read_csv(txt_path, sep='\t')
    for c in CH8:
        if c not in df.columns:
            raise ValueError(f"{txt_path} missing column {c}")
    return df[CH8].astype(float).ffill().fillna(0.0).values


@torch.no_grad()
def infer(ckpt_path, X, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = FeatureConfig(**ck['cfg'])
    # unlabeled: no faults -> common-mode fit is unsupervised; instance-norm self-normalises
    dummy_fault = np.zeros(len(X), int)
    Xr, fault, _, spacing = resample_uniform(X, dummy_fault, dummy_fault,
                                              None, cfg.grid_spacing_m)  # no odometry in parsed txt
    if cfg.instance_norm:
        F, _ = build_features(Xr, fault, cfg, mean=None, std=None, V=None)
    else:
        st = ck['stats']
        F, _ = build_features(Xr, fault, cfg, mean=np.array(st['mean']), std=np.array(st['std']),
                              V=(np.array(st['V']) if st['V'] is not None else None))
    model = SSTSSM(ck['feat_dim'], d_model=ck['d_model'], n_layers=ck['n_layers'],
                   backbone=ck['backbone'], coupled=ck['coupled']).to(device).eval()
    model.load_state_dict(ck['state_dict'])

    Ft = torch.from_numpy(F).float().unsqueeze(0).to(device)
    N = Ft.shape[1]; W = cfg.window
    bprob = np.zeros(N); cprob = np.zeros((N, 4))
    for s in range(0, N, W):
        ch = Ft[:, s:s + W]
        bl, cl = model(ch)
        bprob[s:s + ch.shape[1]] = torch.sigmoid(bl)[0, :ch.shape[1]].cpu().numpy()
        cprob[s:s + ch.shape[1]] = torch.softmax(cl, -1)[0, :ch.shape[1]].cpu().numpy()
    return Xr, bprob, cprob, spacing


def plot(Xr, bprob, cprob, spacing, title, out_png):
    N = len(Xr)
    bnds = peaks_from_field(bprob, prominence=0.2, distance=40)
    edges = np.concatenate(([0], bnds, [N])).astype(int)
    # per-segment class by pooling
    ypred = cprob.argmax(1)
    seg_class = np.zeros(N, int)
    seg_info = []
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            continue
        c = int(np.bincount(ypred[a:b], minlength=4).argmax())
        seg_class[a:b] = c
        seg_info.append((a, b, c))
    x = np.arange(N) * (spacing if spacing != 1.0 else 1.0)
    xl = 'Chainage (m)' if spacing != 1.0 else 'Sample index'

    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    # 1) D1
    axes[0].plot(x, Xr[:, 0], lw=0.7, color='#1f77b4', label='D1 32Hz R')
    axes[0].plot(x, Xr[:, 2], lw=0.7, color='#4c9be8', alpha=0.7, label='D1 100Hz R')
    axes[0].set_ylabel('D1 |Z|'); axes[0].legend(loc='upper right', fontsize=7)
    # 2) D2
    axes[1].plot(x, Xr[:, 4], lw=0.7, color='#2ca02c', label='D2 32Hz R')
    axes[1].plot(x, Xr[:, 6], lw=0.7, color='#74c476', alpha=0.7, label='D2 100Hz R')
    axes[1].set_ylabel('D2 |Z|'); axes[1].legend(loc='upper right', fontsize=7)
    # class shading + boundaries on both signal axes
    for ax in axes[:2]:
        for (a, b, c) in seg_info:
            if c > 0:
                ax.axvspan(x[a], x[min(b, N - 1)], color=CLASS_COLORS[c], alpha=0.18, lw=0)
        for bnd in bnds:
            ax.axvline(x[bnd], color='k', ls='--', lw=0.6, alpha=0.5)
    # 3) discontinuity field + class track
    axes[2].plot(x, bprob, color='#6a0dad', lw=0.8, label='Joint-activity field g(s)')
    axes[2].plot(x, seg_class / 3.0, color='#d62728', lw=0.9, alpha=0.8, label='Defect class (0-3)')
    for bnd in bnds:
        axes[2].axvline(x[bnd], color='k', ls='--', lw=0.6, alpha=0.5)
    axes[2].set_ylabel('g(s) / class'); axes[2].set_xlabel(xl)
    axes[2].legend(loc='upper right', fontsize=7)

    from matplotlib.patches import Patch
    handles = [Patch(facecolor=CLASS_COLORS[c], alpha=0.3, label=CLASS_NAMES[c]) for c in [1, 2, 3]]
    handles.append(plt.Line2D([0], [0], color='k', ls='--', lw=0.8, label='Predicted joint'))
    fig.legend(handles=handles, loc='upper center', ncol=4, fontsize=7, frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(title, y=1.04, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    print(f"  n_joints={len(bnds)}  defect_segments={sum(1 for _,_,c in seg_info if c>0)}/{len(seg_info)}")
    print(f"  saved -> {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=os.path.join(REPO_PY, 'models', 'sstssm_full.pt'))
    ap.add_argument('--txt', default=None, help='unlabeled *_D1D2_parsed.txt (random if omitted)')
    ap.add_argument('--seed', type=int, default=None)
    args = ap.parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    if not os.path.exists(args.ckpt):
        sys.exit(f"checkpoint not found: {args.ckpt} (run ssm_ndt.run_ablation first)")
    if args.txt:
        txt = args.txt if os.path.isabs(args.txt) else os.path.join(REPO_PY, args.txt)
    else:
        cands = sorted(glob.glob(os.path.join(REPO_PY, '*_D1D2_parsed.txt')))
        txt = random.choice(cands)
    name = os.path.basename(txt)
    print(f"[infer] {name}")

    use_scienceplots()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    X = load_unlabeled(txt)
    Xr, bprob, cprob, spacing = infer(args.ckpt, X, device)
    out = os.path.join(FIGDIR, f'sstssm_segcls_{name.replace(".txt","")}.png')
    plot(Xr, bprob, cprob, spacing, f'SST-SSM segmentation + classification — {name} (unlabeled)', out)


if __name__ == '__main__':
    main()
