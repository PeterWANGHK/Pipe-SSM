"""Fig.1 framework diagram + E1b gauge-degeneracy scatter for the TIM manuscript.

  python -m ssm_ndt.make_framework_e1b
"""
from __future__ import annotations
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ssm_ndt.forward_dodd_deeds import dz_numpy

PYDIR = os.path.dirname(os.path.abspath(__file__)); REPO_PY = os.path.dirname(PYDIR)
REPO = os.path.dirname(REPO_PY); FIG = os.path.join(REPO, 'figures'); os.makedirs(FIG, exist_ok=True)


def style():
    try:
        import scienceplots  # noqa
        plt.style.use(['science', 'no-latex'])
    except Exception:
        pass
    plt.rcParams.update({'figure.dpi': 120, 'font.size': 8})


def save(fig, name):
    out = os.path.join(FIG, name)
    fig.savefig(out + '.png', dpi=400, bbox_inches='tight')
    fig.savefig(out + '.pdf', bbox_inches='tight')
    print(f"saved -> {out}.png (+.pdf)")


def _box(ax, x, y, w, h, text, fc, fontsize=7.2, ec='0.25'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.012',
                                fc=fc, ec=ec, lw=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fontsize)


def _arrow(ax, x0, y0, x1, y1, text=None, fs=6.2, color='0.2'):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>',
                                 mutation_scale=9, lw=0.9, color=color))
    if text:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.018, text, ha='center', fontsize=fs, color=color)


def framework():
    fig, ax = plt.subplots(figsize=(7.16, 3.1))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

    # input column
    _box(ax, 0.01, 0.62, 0.155, 0.30,
         'Dual-frequency ECT\n2 detectors $\\times$ {32,100} Hz\nmagnitude + phase', '#eaf2fb')
    _box(ax, 0.01, 0.22, 0.155, 0.26, 'Odometer + IMU\n(tool motion)', '#eaf2fb')
    _box(ax, 0.20, 0.42, 0.14, 0.30,
         'Chainage resampling\n(uniform 20 mm grid)\n+ instance norm.', '#f2f2f2')
    _arrow(ax, 0.165, 0.74, 0.20, 0.62)
    _arrow(ax, 0.165, 0.35, 0.20, 0.50)

    # Stage I
    ax.text(0.475, 0.97, 'Stage I — explanation (model-based latent inversion, self-supervised)',
            fontsize=7.6, ha='center', style='italic')
    _box(ax, 0.37, 0.60, 0.205, 0.30,
         'Analytical forward operator\n$\\Delta\\bar{Z}(\\omega;\\vartheta,\\ell)$, shell $\\Gamma$\n(BCs in closed form)', '#fdeee2')
    _box(ax, 0.37, 0.16, 0.205, 0.30,
         'Variational inversion\n$\\min_{\\theta,G,O} \\sum_{d,\\omega}\\|\\tilde{Z}-(G\\Delta\\bar{Z}+O)\\|^2$\n+ smoothness + bounded calib.', '#fdeee2')
    _arrow(ax, 0.34, 0.57, 0.37, 0.57)
    _arrow(ax, 0.4725, 0.60, 0.4725, 0.46)

    # Stage I outputs
    _box(ax, 0.615, 0.70, 0.145, 0.22, 'Effective state $\\vartheta(s)$\n(drift attribution)', '#fff6d9')
    _box(ax, 0.615, 0.42, 0.145, 0.22, 'Anomaly stream $a(s)$\n(physics-normalized)', '#fff6d9')
    _box(ax, 0.615, 0.14, 0.145, 0.22, 'Per-sequence calibration\n$(G,O)$ per channel', '#fff6d9')
    for yy in (0.80, 0.53, 0.26):
        _arrow(ax, 0.575, 0.31 if yy == 0.26 else 0.31, 0.615, yy)

    # Stage II
    ax.text(0.875, 0.97, 'Stage II — decision (supervised)', fontsize=7.6, ha='center', style='italic')
    _box(ax, 0.80, 0.56, 0.185, 0.34,
         'Bidirectional selective SSM\n(3 blocks, assoc.\\ scan)\ncoupled heads', '#e8f4ea')
    _box(ax, 0.80, 0.10, 0.185, 0.38,
         'Outputs\njoint boundaries (MAD 0.11 m)\nseverity {normal, low, high}\nconfidence $\\to$ flag-for-review', '#e8f4ea')
    _arrow(ax, 0.76, 0.81, 0.80, 0.78)
    _arrow(ax, 0.76, 0.53, 0.80, 0.68)
    _arrow(ax, 0.8925, 0.56, 0.8925, 0.48)
    ax.text(0.5, 0.02, 'Stage I runs per sequence with no labels (identical on unseen campaigns); '
                       'Stage II is trained on campaign-level splits.', fontsize=6.4, ha='center', color='0.35')
    save(fig, 'paper_framework')


def e1b_gauge(n=3000, seed=0):
    """Gauge degeneracy: recover theta from dual-freq dZ under per-sample complex gain,
    unconstrained vs bounded (the calibration prior). MLP inverse, same protocol as E1."""
    import torch
    import torch.nn as nn
    rng = np.random.default_rng(seed)
    logsm = rng.uniform(7.0, 9.0, n)
    lift = rng.uniform(0.06, 0.16, n)
    Z32 = dz_numpy(10 ** logsm, lift, 32.0)
    Z100 = dz_numpy(10 ** logsm, lift, 100.0)
    Z32 /= np.abs(Z32).mean(); Z100 /= np.abs(Z100).mean()

    def corrupt(bounded):
        """Per-(frequency) INDEPENDENT complex gains — the actual field calibration gauge.
        Unconstrained independent gains destroy the cross-frequency dispersion lever;
        the bounded prior (gains near unity) preserves it."""
        def gain(width_mag, width_ph):
            return (1 + width_mag * rng.standard_normal(n)) * \
                   np.exp(1j * width_ph * rng.standard_normal(n))
        if bounded:
            g32, g100 = gain(0.1, 0.1), gain(0.1, 0.1)
        else:
            g32 = (0.3 + 1.4 * rng.random(n)) * np.exp(1j * rng.uniform(-np.pi, np.pi, n))
            g100 = (0.3 + 1.4 * rng.random(n)) * np.exp(1j * rng.uniform(-np.pi, np.pi, n))
        A, B = g32 * Z32, g100 * Z100
        X = np.column_stack([A.real, A.imag, B.real, B.imag])
        return (X - X.mean(0)) / (X.std(0) + 1e-12)

    def recover(X):
        import torch.nn.functional as F
        k = int(0.8 * n)
        Y = (logsm - logsm.mean()) / logsm.std()
        net = nn.Sequential(nn.Linear(4, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 1))
        opt = torch.optim.Adam(net.parameters(), 1e-3)
        xt = torch.tensor(X[:k], dtype=torch.float32)
        yt = torch.tensor(Y[:k, None], dtype=torch.float32)
        for _ in range(1500):
            opt.zero_grad(); F.mse_loss(net(xt), yt).backward(); opt.step()
        with torch.no_grad():
            yp = net(torch.tensor(X[k:], dtype=torch.float32)).numpy().ravel() * logsm.std() + logsm.mean()
        return logsm[k:], yp

    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.7), sharey=True)
    for i, (bounded, title) in enumerate([(False, '(a) Unconstrained calibration gauge'),
                                          (True, '(b) Bounded calibration prior')]):
        yt, yp = recover(corrupt(bounded))
        r = np.corrcoef(yt, yp)[0, 1]
        ax[i].plot(yt, yp, '.', ms=2, alpha=0.4, color='#0072B2' if bounded else '#D55E00')
        ax[i].plot([7, 9], [7, 9], 'k--', lw=0.8)
        ax[i].set_xlabel(r'true $\log_{10}(\sigma\mu_r)$')
        ax[i].set_title(f'{title}\n$r={r:.2f}$', fontsize=8)
        ax[i].grid(True, alpha=0.3)
        print(f"  e1b {'bounded' if bounded else 'free'}: corr={r:.3f}")
    ax[0].set_ylabel(r'recovered $\log_{10}(\sigma\mu_r)$')
    fig.suptitle('E1b: the calibration gauge hides the state unless bounded (Prop. 5)', y=1.04, fontsize=8.5)
    fig.tight_layout()
    save(fig, 'paper_e1b_gauge')


if __name__ == '__main__':
    style()
    framework()
    e1b_gauge()
