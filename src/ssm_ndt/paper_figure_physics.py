"""Flagship qualitative figure: full pipeline on a TRULY UNLABELED campaign —
signal + model-based inversion (sigma-mu, anomaly) + segmentation + classification + triage.

  python -m ssm_ndt.paper_figure_physics --seed 7           # random unlabeled file
  python -m ssm_ndt.paper_figure_physics --txt 86001_D1D2_parsed.txt
"""
from __future__ import annotations
import os, sys, glob, random, argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ssm_ndt.data import FeatureConfig, resample_uniform, build_features, CH8
from ssm_ndt.latent_inversion import get_or_invert
from ssm_ndt.model import SSTSSM
from ssm_ndt.metrics import peaks_from_field

PYDIR = os.path.dirname(os.path.abspath(__file__)); REPO_PY = os.path.dirname(PYDIR)
REPO = os.path.dirname(REPO_PY); FIG = os.path.join(REPO, 'figures'); os.makedirs(FIG, exist_ok=True)
CMAP = {0: '#009E73', 1: '#E69F00', 2: '#D55E00'}
CNAME = {0: 'Normal', 1: 'Low severity', 2: 'High severity'}


def style():
    try:
        import scienceplots  # noqa
        plt.style.use(['science', 'no-latex'])
    except Exception:
        pass
    plt.rcParams.update({'figure.dpi': 120, 'font.size': 8, 'legend.fontsize': 6.5})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=os.path.join(REPO_PY, 'models', 'sstssm_final.pt'))
    ap.add_argument('--txt', default=None)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--abstain', type=float, default=0.6)
    a = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    style()

    # pick a truly unlabeled parsed file (no merged csv) unless given
    if a.txt is None:
        random.seed(a.seed)
        cands = [f for f in glob.glob(os.path.join(REPO_PY, '*_D1D2_parsed.txt'))
                 if not os.path.exists(os.path.join(
                     REPO_PY, f"merged_data_with_fault_classes_"
                              f"{os.path.basename(f).split('_')[0]}.csv"))]
        a.txt = random.choice(sorted(cands))
    name = os.path.basename(a.txt); fid = name.split('_')[0]
    print(f"[eval] truly unlabeled campaign: {name}")

    import pandas as pd
    df = pd.read_csv(a.txt if os.path.isabs(a.txt) else os.path.join(REPO_PY, a.txt), sep='\t')
    X = df[CH8].astype(float).ffill().fillna(0.0).values

    ck = torch.load(a.ckpt, map_location=device, weights_only=False)
    cfg = FeatureConfig(**ck['cfg'])
    n_classes = ck['state_dict']['class_head.weight'].shape[0]

    # replicate the training feature pipeline for one unlabeled sequence
    theta, anom = get_or_invert(f'unl{fid}', X)
    Xe = np.hstack([X, theta, anom])
    Xe, fault, labelnum, spacing = resample_uniform(Xe, np.zeros(len(X), int), np.zeros(len(X), int),
                                                    None, cfg.grid_spacing_m)
    Xr, pi_ext = Xe[:, :8], Xe[:, 8:]
    F, _ = build_features(Xr, fault, cfg)
    pm, ps = pi_ext.mean(0), pi_ext.std(0) + 1e-6
    F = np.concatenate([F, ((pi_ext - pm) / ps).astype(np.float32)], axis=1)

    model = SSTSSM(ck['feat_dim'], d_model=ck['d_model'], n_layers=ck['n_layers'],
                   backbone=ck['backbone'], coupled=ck['coupled'], n_classes=n_classes).to(device).eval()
    model.load_state_dict(ck['state_dict'])
    Ft = torch.from_numpy(F).float().unsqueeze(0).to(device)
    N = Ft.shape[1]; W = cfg.window
    bprob = np.zeros(N); cprob = np.zeros((N, n_classes))
    with torch.no_grad():
        for s in range(0, N, W):
            bl, cl = model(Ft[:, s:s + W])
            bprob[s:s + bl.shape[1]] = torch.sigmoid(bl)[0, :bl.shape[1]].cpu().numpy()
            cprob[s:s + cl.shape[1]] = torch.softmax(cl, -1)[0, :cl.shape[1]].cpu().numpy()

    bnds = peaks_from_field(bprob, distance=40)
    edges = np.concatenate(([0], np.sort(bnds), [N])).astype(int)
    ypred = cprob.argmax(1); conf = cprob.max(1)
    segcls = np.zeros(N, int); seginfo = []
    for aa, bb in zip(edges[:-1], edges[1:]):
        if bb > aa:
            c = int(np.bincount(ypred[aa:bb], minlength=n_classes).argmax())
            segcls[aa:bb] = c; seginfo.append((aa, bb, c))

    x = np.arange(N) * 0.018   # nominal chainage
    logsm = pi_ext[:, 0]; an = np.abs(pi_ext[:, 2:4]).sum(1)

    fig, ax = plt.subplots(5, 1, figsize=(7.2, 8.2), sharex=True,
                           gridspec_kw={'height_ratios': [2, 1.3, 1.3, 1.1, 1.1]})
    # (a) signal + predicted joints + severity shading
    ax[0].plot(x, Xr[:, 0], lw=0.6, color='#0072B2', label='D1 32 Hz |Z|')
    ax[0].plot(x, Xr[:, 4], lw=0.6, color='#009E73', alpha=0.8, label='D2 32 Hz |Z|')
    for (aa, bb, c) in seginfo:
        if c > 0:
            ax[0].axvspan(x[aa], x[min(bb, N - 1)], color=CMAP[c], alpha=0.15, lw=0)
    for b in bnds:
        ax[0].axvline(x[b], color='k', ls='--', lw=0.5, alpha=0.6)
    ax[0].set_ylabel('|Z|'); ax[0].legend(loc='upper right', ncol=2)
    ax[0].set_title('(a) Unlabeled inspection sequence: signal, predicted joints (dashed), predicted severity shading', loc='left')
    # (b) inverted effective material state
    ax[1].plot(x, logsm, lw=0.8, color='#6a0dad')
    ax[1].set_ylabel(r'$\log_{10}(\sigma\mu)_{\rm eff}$')
    ax[1].set_title('(b) Model-based inversion: effective material state (drift attribution)', loc='left')
    # (c) physics-normalised anomaly
    ax[2].plot(x, an, lw=0.6, color='#D55E00')
    for b in bnds:
        ax[2].axvline(x[b], color='k', ls='--', lw=0.4, alpha=0.5)
    ax[2].set_ylabel('|anomaly|')
    ax[2].set_title('(c) Physics-normalised anomaly stream (what the layered model cannot explain)', loc='left')
    # (d) severity track
    ax[3].step(x, segcls, where='mid', color='#D55E00', lw=1.0)
    ax[3].set_yticks(range(n_classes)); ax[3].set_ylim(-0.3, n_classes - 0.7)
    ax[3].set_ylabel('Severity'); ax[3].set_title('(d) Predicted severity (per segment)', loc='left')
    # (e) confidence + triage
    ax[4].plot(x, conf, lw=0.7, color='#6a0dad', label='confidence')
    ax[4].axhline(a.abstain, color='red', ls=':', lw=0.8)
    ax[4].fill_between(x, 0, 1, where=conf < a.abstain, color='red', alpha=0.12, step='mid',
                       label='flag for review')
    ax[4].set_ylim(0, 1.02); ax[4].set_ylabel('Confidence')
    ax[4].set_xlabel('Approx. chainage (m, nominal 18 mm/sample)')
    ax[4].set_title('(e) Selective prediction: low-confidence regions flagged for expert review', loc='left')
    ax[4].legend(loc='lower right', ncol=2)
    for a_ in ax:
        a_.grid(True, alpha=0.25, lw=0.4)
    fig.suptitle('SST-SSM + model-based latent inversion — end-to-end on a previously unseen, unlabeled campaign',
                 y=1.005, fontsize=9)
    fig.tight_layout()
    out = os.path.join(FIG, f'paper_full_pipeline_{fid}')
    fig.savefig(out + '.png', dpi=400, bbox_inches='tight'); fig.savefig(out + '.pdf', bbox_inches='tight')
    nseg = len(seginfo); ndef = sum(1 for _, _, c in seginfo if c > 0)
    print(f"  joints={len(bnds)} segments={nseg} defect-flagged={ndef} "
          f"low-conf={100 * float((conf < a.abstain).mean()):.0f}%")
    print(f"saved -> {out}.png (+.pdf)")


if __name__ == '__main__':
    main()
