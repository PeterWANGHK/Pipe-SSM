"""Theory-accompanying + physics-module result figures for the TIM manuscript.

T1 paper_theory_forward   : forward-operator theory (impedance-plane loci, shell effect, skin depth)
T2 paper_inversion_fit_102: measured vs calibrated canonical fit on RAW signal + anomaly (campaign 102)
T3 paper_drift_box        : per-campaign distribution of inverted effective state (drift attribution)

  python -m ssm_ndt.make_theory_figures
"""
from __future__ import annotations
import os, sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ssm_ndt.forward_dodd_deeds import dz_numpy, SHELL_D, MU0

PYDIR = os.path.dirname(os.path.abspath(__file__)); REPO_PY = os.path.dirname(PYDIR)
REPO = os.path.dirname(REPO_PY); FIG = os.path.join(REPO, 'figures')
CACHE = os.path.join(REPO_PY, 'physinv_cache')
os.makedirs(FIG, exist_ok=True)


def style():
    try:
        import scienceplots  # noqa
        plt.style.use(['science', 'no-latex'])
    except Exception:
        pass
    plt.rcParams.update({'figure.dpi': 120, 'font.size': 8, 'legend.fontsize': 6.5})


def save(fig, name):
    out = os.path.join(FIG, name)
    fig.savefig(out + '.png', dpi=400, bbox_inches='tight')
    fig.savefig(out + '.pdf', bbox_inches='tight')
    print(f"saved -> {out}.png (+.pdf)")


def t1_theory():
    sm = np.logspace(7, 9, 60)
    lift = np.full_like(sm, 0.102)
    fig, ax = plt.subplots(1, 3, figsize=(7.16, 2.4))

    # (a) impedance-plane loci at both frequencies (normalized): the dispersion lever
    for f, c in [(32.0, '#0072B2'), (100.0, '#D55E00')]:
        z = dz_numpy(sm, lift, f)
        zn = z / np.abs(z).mean()
        ax[0].plot(zn.real, zn.imag, '-', color=c, lw=1.1, label=f'{f:.0f} Hz')
        for dec in [1e7, 1e8, 1e9]:
            k = np.argmin(np.abs(sm - dec))
            ax[0].plot(zn.real[k], zn.imag[k], 'o', color=c, ms=3.5)
            if f == 32.0:
                ax[0].annotate(f'$10^{{{int(np.log10(dec))}}}$', (zn.real[k], zn.imag[k]),
                               fontsize=6, xytext=(3, 3), textcoords='offset points')
    ax[0].set_xlabel(r'Re $\Delta\bar{Z}$'); ax[0].set_ylabel(r'Im $\Delta\bar{Z}$')
    ax[0].set_title(r'(a) Impedance-plane loci vs $\sigma\mu_r$', fontsize=8)
    ax[0].legend(); ax[0].grid(True, alpha=0.3)

    # (b) finite-thickness shell effect at 32 Hz: |dZ| vs sigma-mu for d_s sweep + half-space
    for d, c, lb in [(0.0007, '#009E73', '0.7 mm'), (SHELL_D, '#0072B2', '1.4 mm (17 ga)'),
                     (0.0028, '#E69F00', '2.8 mm'), (None, '0.3', r'half-space')]:
        z = dz_numpy(sm, lift, 32.0, thickness=d)
        ax[1].semilogx(sm, np.abs(z) / np.abs(dz_numpy(np.array([1e8]), np.array([0.102]),
                                                       32.0, thickness=SHELL_D))[0],
                       '-' if d else '--', color=c, lw=1.1, label=lb)
    ax[1].set_xlabel(r'$\sigma\mu_r$ (S/m)'); ax[1].set_ylabel(r'$|\Delta Z|$ (norm.)')
    ax[1].set_title(r'(b) Shell-thickness effect, $\Gamma$ of Eq.(7)', fontsize=8)
    ax[1].legend(); ax[1].grid(True, alpha=0.3, which='both')

    # (c) skin depth vs sigma-mu at both freqs, with the shell thickness line
    mu_r = 50.0
    for f, c in [(32.0, '#0072B2'), (100.0, '#D55E00')]:
        delta = np.sqrt(2.0 / (2 * np.pi * f * MU0 * sm)) * 1000  # sigma*mu_r product form, mm
        ax[2].loglog(sm, delta, color=c, lw=1.1, label=f'{f:.0f} Hz')
    ax[2].axhline(SHELL_D * 1000, color='k', ls=':', lw=1)
    ax[2].text(1.3e7, SHELL_D * 1000 * 1.15, 'shell $d_s$ = 1.4 mm', fontsize=6.5)
    ax[2].set_xlabel(r'$\sigma\mu_r$ (S/m)'); ax[2].set_ylabel(r'skin depth $\delta$ (mm)')
    ax[2].set_title(r'(c) $\delta(\omega)$ straddles the wall', fontsize=8)
    ax[2].legend(); ax[2].grid(True, alpha=0.3, which='both')
    fig.tight_layout()
    save(fig, 'paper_theory_forward')


def t2_inversion_fit(fid='102', s0=2000, s1=9000):
    import pandas as pd
    d = pd.read_csv(os.path.join(REPO_PY, f'merged_data_with_fault_classes_{fid}.csv'))
    cols = ['D1_32Hz_R', 'D1_32Hz_Theta']
    z = d[cols[0]].values * np.exp(1j * np.deg2rad(d[cols[1]].values))
    zt = z / (np.mean(np.abs(z)) + 1e-9)                       # same normalization as inversion
    cach = np.load(os.path.join(CACHE, f'{fid}.npz'))
    a = cach['anom'][:, 0] + 1j * cach['anom'][:, 1]           # 32 Hz, D1
    fit = zt - a                                                # calibrated canonical fit
    ln = pd.to_numeric(d['LabelNumber'], errors='coerce').fillna(-1).values
    fc = pd.to_numeric(d['FaultClass'], errors='coerce').fillna(0).astype(int).clip(0, 3).values
    bnds = np.where(np.diff(ln) != 0)[0] + 1
    x = np.arange(s0, s1) * 0.018

    fig, ax = plt.subplots(2, 1, figsize=(7.16, 3.2), sharex=True,
                           gridspec_kw={'height_ratios': [2, 1.2]})
    ax[0].plot(x, np.abs(zt[s0:s1]), color='0.45', lw=0.7, label=r'measured $|\tilde{Z}|$ (D1, 32 Hz)')
    ax[0].plot(x, np.abs(fit[s0:s1]), color='#D55E00', lw=1.1,
               label=r'calibrated canonical fit $|G\,\Delta\bar{Z}(\theta)+O|$')
    ax[0].set_ylabel('normalized $|Z|$')
    ax[0].set_title('(a) Model-based fit on the raw signal (held-out inspection sequence)', loc='left', fontsize=8)
    ax[0].legend(ncol=2); ax[0].grid(True, alpha=0.3)
    ax[1].plot(x, np.abs(a[s0:s1]), color='#6a0dad', lw=0.7, label=r'anomaly $|a(s)|$ (residual)')
    for b in bnds:
        if s0 <= b < s1:
            ax[1].axvline(b * 0.018, color='k', ls='--', lw=0.5, alpha=0.5)
    m = fc[s0:s1] > 0
    if m.any():
        ax[1].fill_between(x, 0, np.nanmax(np.abs(a[s0:s1])), where=m,
                           color='orange', alpha=0.15, step='mid', label='expert defect region')
    ax[1].set_ylabel('$|a|$'); ax[1].set_xlabel('Approx. chainage (m, nominal 18 mm/sample)')
    ax[1].set_title('(b) Physics-normalized anomaly: joints (dashed) and defects emerge as residual',
                    loc='left', fontsize=8)
    ax[1].legend(ncol=2); ax[1].grid(True, alpha=0.3)
    fig.tight_layout()
    save(fig, 'paper_inversion_fit')


def t3_drift_box():
    ids = ['102', '104', '021', '88003', '84001', '87001', '89005', '89006',
           '810001', '014', '88005', '86005']
    data, labels = [], []
    for cid in ids:
        p = os.path.join(CACHE, f'{cid}.npz')
        if not os.path.exists(p):
            continue
        th = np.load(p)['theta'][:, 0]
        data.append(th[np.isfinite(th)]); labels.append(cid)
    order = np.argsort([np.mean(d) for d in data])
    data = [data[i] for i in order]
    # anonymized display labels (campaign ids are engineering-log metadata, not paper content)
    disp = [f'C{i+1}' for i in range(len(data))]
    means = [float(np.mean(d)) for d in data]
    between = np.std(means); within = np.mean([np.std(d) for d in data])

    fig, ax = plt.subplots(figsize=(7.16, 2.5))
    bp = ax.boxplot(data, tick_labels=disp, showfliers=False, patch_artist=True, widths=0.6)
    for i, b in enumerate(bp['boxes']):
        b.set_facecolor('#86b6e2' if i < len(data) - 1 else '#e2a186'); b.set_alpha(0.8)
    for med in bp['medians']:
        med.set_color('k')
    ax.set_ylabel(r'$\log_{10}(\sigma\mu_r)_{\mathrm{eff}}$')
    ax.set_xlabel('Inspection campaign (sorted by inferred effective state)')
    ax.set_title(f'Physical drift attribution: between-campaign spread {between:.2f} dex '
                 f'vs within-campaign {within:.2f} dex '
                 f'(~{between/within:.0f}$\\times$); extreme campaign highlighted', fontsize=8)
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    save(fig, 'paper_drift_attribution')


if __name__ == '__main__':
    style()
    t1_theory()
    t2_inversion_fit()
    t3_drift_box()
