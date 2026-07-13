"""L2 — Self-supervised latent physical-state inversion via the analytical forward operator.

Classical model-based inversion posture (per-sequence direct optimisation, no training data,
no leakage): for one inspection sequence, jointly estimate
  - theta(s) = (log10 sigma_mu_eff(s), liftoff_eff(s)) on a coarse node grid (interpolated),
  - per-(detector,frequency) complex calibration (gain G, offset O)  [answers the calibration
    critique: this IS the calibration procedure, estimated from the data],
  - anomaly residual a(s) = Z_meas - G*F(theta) - O  (what the layered model cannot explain:
    joints, defects, geometry mismatch).
Loss = sum |a|^2 + smoothness(theta) + weak priors (liftoff near nominal 0.10 m).
Both co-located detectors and both frequencies share the SAME theta(s) — the identifiability lever.

Outputs cached to python/physinv_cache/<id>.npz so multi-seed experiment runs reuse them.

  python -m ssm_ndt.latent_inversion --id 102 --plot     # sanity on one campaign
  python -m ssm_ndt.latent_inversion --e2                # drift attribution across campaigns
"""
from __future__ import annotations
import os, sys, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ssm_ndt.forward_dodd_deeds import dz_torch, LIFT0, SHELL_D

PYDIR = os.path.dirname(os.path.abspath(__file__)); REPO_PY = os.path.dirname(PYDIR)
REPO = os.path.dirname(REPO_PY)
CACHE = os.path.join(REPO_PY, 'physinv_cache'); os.makedirs(CACHE, exist_ok=True)
FREQS = (32.0, 100.0)


def _complex_channels(X8):
    """(N,8) R/Theta-deg -> dict[(det,freq)] complex (N,)."""
    out = {}
    idx = {('D1', 32): (0, 1), ('D1', 100): (2, 3), ('D2', 32): (4, 5), ('D2', 100): (6, 7)}
    for k, (ir, it) in idx.items():
        out[k] = X8[:, ir] * np.exp(1j * np.deg2rad(X8[:, it]))
    return out


def invert_sequence(X8, node_every=64, iters=400, lr=0.05, device=None, verbose=False,
                    thickness=None, mu_r=50.0, cal_prior_w=0.3, smooth_sm_w=5.0):
    """Return theta (N,2)=[log10(sigma_mu), liftoff], anomaly a (N,4 complex as (N,8) float), fit info."""
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    N = len(X8)
    Z = _complex_channels(X8)
    n_nodes = max(8, N // node_every)
    # normalise measured Z per channel (scale-free fit; calibration soaks absolute units)
    Zt = {}
    for k, z in Z.items():
        s = np.mean(np.abs(z)) + 1e-9
        Zt[k] = torch.tensor(z / s, dtype=torch.complex128, device=device)

    # parameters: theta nodes + complex calib per (det,freq)
    logsm_n = torch.full((n_nodes,), 8.3, dtype=torch.float64, device=device, requires_grad=True)
    lift_n = torch.full((n_nodes,), float(LIFT0), dtype=torch.float64, device=device, requires_grad=True)
    Gr = {k: torch.tensor([1.0, 0.0], dtype=torch.float64, device=device, requires_grad=True) for k in Z}
    Or = {k: torch.tensor([0.0, 0.0], dtype=torch.float64, device=device, requires_grad=True) for k in Z}

    # interpolation matrix nodes -> samples (linear)
    pos = torch.linspace(0, n_nodes - 1, N, dtype=torch.float64, device=device)
    i0 = pos.floor().long().clamp(0, n_nodes - 2); w1 = (pos - i0.to(pos.dtype)).clamp(0, 1)

    def interp(v):
        return v[i0] * (1 - w1) + v[i0 + 1] * w1

    params = [logsm_n, lift_n] + list(Gr.values()) + list(Or.values())
    opt = torch.optim.Adam(params, lr=lr)
    n_col = min(N, 16384)                                           # stochastic collocation subset
    for it in range(iters):
        opt.zero_grad()
        sub = torch.randint(0, N, (n_col,), device=device) if N > n_col else slice(None)
        logsm = interp(logsm_n)[sub]; lift = interp(lift_n).clamp(0.03, 0.25)[sub]
        loss = 0.0
        th = SHELL_D if thickness is None else thickness
        for f in FREQS:
            dz = dz_torch(10 ** logsm, lift, f, thickness=th, mu_r=mu_r)
            dz = dz / (dz.abs().mean() + 1e-12)                    # scale-free model shape
            for det in ['D1', 'D2']:
                k = (det, int(f))
                G = torch.complex(Gr[k][0], Gr[k][1]); O = torch.complex(Or[k][0], Or[k][1])
                res = Zt[k][sub] - (G * dz + O)
                loss = loss + (res.abs() ** 2).mean()
        loss = loss + smooth_sm_w * ((logsm_n[1:] - logsm_n[:-1]) ** 2).mean() \
                    + 50.0 * ((lift_n[1:] - lift_n[:-1]) ** 2).mean() \
                    + 0.5 * ((lift_n - LIFT0) ** 2).mean()
        # calibration prior: instrument normalisation is bounded (.pn CustomNormAmp/Phase),
        # so G ~ 1, O ~ 0. Restores identifiability lost to full calibration freedom (E1b).
        for k in Zt:
            loss = loss + cal_prior_w * (((Gr[k][0] - 1.0) ** 2 + Gr[k][1] ** 2) + (Or[k] ** 2).sum())
        loss.backward(); opt.step()
        if verbose and (it + 1) % 100 == 0:
            print(f"    inv iter {it+1}/{iters} loss={float(loss):.5f}")

    with torch.no_grad():
        logsm_f = interp(logsm_n).cpu().numpy(); lift_f = interp(lift_n).clamp(0.03, 0.25).cpu().numpy()
        A = np.zeros((N, 8), np.float32)                            # anomaly per (det,freq) re/im
        CH = 32768                                                  # chunked final residual pass
        for c0 in range(0, N, CH):
            c1 = min(c0 + CH, N)
            lo = torch.tensor(10 ** logsm_f[c0:c1], dtype=torch.float64, device=device)
            lf = torch.tensor(lift_f[c0:c1], dtype=torch.float64, device=device)
            col = 0
            th = SHELL_D if thickness is None else thickness
            for f in FREQS:
                dz = dz_torch(lo, lf, f, thickness=th, mu_r=mu_r)
                dz = dz / (dz.abs().mean() + 1e-12)
                for det in ['D1', 'D2']:
                    k = (det, int(f))
                    G = torch.complex(Gr[k][0], Gr[k][1]); O = torch.complex(Or[k][0], Or[k][1])
                    res = (Zt[k][c0:c1] - (G * dz + O)).cpu().numpy()
                    A[c0:c1, col] = res.real; A[c0:c1, col + 1] = res.imag; col += 2
        theta = np.column_stack([logsm_f, lift_f]).astype(np.float32)
    return theta, A, float(loss)


def get_or_invert(fid, X8, **kw):
    p = os.path.join(CACHE, f'{fid}.npz')
    if os.path.exists(p):
        d = np.load(p)
        if len(d['theta']) == len(X8):
            return d['theta'], d['anom']
    theta, A, loss = invert_sequence(X8, **kw)
    np.savez_compressed(p, theta=theta, anom=A, loss=loss)
    return theta, A


def e2_drift_attribution(ids=('102', '104', '88003', '86005', '88005', '87001')):
    """E2: do inferred effective parameters differ by campaign (physical drift attribution)?"""
    import pandas as pd
    rows = []
    for fid in ids:
        p = os.path.join(REPO_PY, f'merged_data_with_fault_classes_{fid}.csv')
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p)
        cols = ['D1_32Hz_R', 'D1_32Hz_Theta', 'D1_100Hz_R', 'D1_100Hz_Theta',
                'D2_32Hz_R', 'D2_32Hz_Theta', 'D2_100Hz_R', 'D2_100Hz_Theta']
        X8 = df[cols].astype(float).ffill().fillna(0.0).values
        theta, A = get_or_invert(fid, X8, verbose=False)
        rows.append((fid, float(np.mean(theta[:, 0])), float(np.std(theta[:, 0])),
                     float(np.mean(theta[:, 1]) * 1000), float(np.mean(np.abs(A)))))
        print(f"  {fid:8s}: log10(sigma_mu)={rows[-1][1]:.3f}±{rows[-1][2]:.3f}  "
              f"liftoff={rows[-1][3]:.1f} mm  |anomaly|={rows[-1][4]:.4f}")
    sms = [r[1] for r in rows]
    print(f"\n  campaign-to-campaign spread of log10(sigma_mu): {np.std(sms):.3f} dex "
          f"(within-campaign {np.mean([r[2] for r in rows]):.3f}) -> drift IS attributable to "
          f"effective material state" if np.std(sms) > np.mean([r[2] for r in rows]) else "  (spread small)")
    # significance: one-way ANOVA on subsampled node values (subsampling reduces autocorrelation;
    # p-value still optimistic under residual correlation — reported with that caveat)
    try:
        from scipy.stats import f_oneway
        groups = []
        for fid, *_ in rows:
            d = np.load(os.path.join(CACHE, f'{fid}.npz'))
            groups.append(d['theta'][::512, 0])
        F, p = f_oneway(*groups)
        print(f"  one-way ANOVA across campaigns (subsampled): F={F:.1f}, p={p:.2e} "
              f"(autocorrelation caveat applies)")
    except Exception as e:
        print(f"  (ANOVA skipped: {e})")
    return rows


def e1b_calibrated_identifiability(n_seq=6, L=1024, noise=0.02, seed=0, bounded=True):
    """E1b (methodology-gate item): identifiability of theta(s) WITH per-(det,freq) calibration freedom.

    Synthesize sequences with smooth theta(s), apply random unknown complex gain/offset per channel
    (the field setting), then run the ACTUAL invert_sequence machinery and measure recovery.
    Expectation: sigma_mu (cross-frequency, along-sequence shape) survives; liftoff largely does not
    (absorbed by the gain — the G-liftoff degeneracy that explains liftoff bound-saturation in field data).
    """
    from ssm_ndt.forward_dodd_deeds import dz_numpy
    rng = np.random.default_rng(seed)
    r_sm, r_lift, corr_sm = [], [], []
    for k in range(n_seq):
        # smooth ground-truth latents
        t = np.linspace(0, 4 * np.pi, L)
        logsm_gt = 8.4 + 0.35 * np.sin(t + rng.uniform(0, 6)) + 0.1 * rng.standard_normal()
        lift_gt = 0.10 + 0.02 * np.sin(0.7 * t + rng.uniform(0, 6))
        X8 = np.zeros((L, 8))
        for j, f in enumerate([32.0, 100.0]):
            z = dz_numpy(10 ** logsm_gt, lift_gt, f)
            z = z / np.mean(np.abs(z))
            for d in range(2):                                   # two "detectors"
                if bounded:   # realistic post-CustomNorm residual calibration error
                    G = (1.0 + rng.uniform(-0.2, 0.2)) * np.exp(1j * rng.uniform(-0.35, 0.35))
                    O = 0.05 * (rng.standard_normal() + 1j * rng.standard_normal())
                else:         # adversarial: full calibration freedom (degeneracy demo)
                    G = (0.5 + rng.uniform(0, 1.5)) * np.exp(1j * rng.uniform(-np.pi, np.pi))
                    O = 0.2 * (rng.standard_normal() + 1j * rng.standard_normal())
                zm = G * z + O + noise * (rng.standard_normal(L) + 1j * rng.standard_normal(L))
                c0 = (d * 4) + (j * 2)
                X8[:, c0] = np.abs(zm); X8[:, c0 + 1] = np.rad2deg(np.angle(zm))
        theta, A, loss = invert_sequence(X8, node_every=32, iters=500, verbose=False)
        # sigma_mu: report correlation (shape) and de-meaned dex error (offset ambiguity allowed)
        c = np.corrcoef(theta[:, 0], logsm_gt)[0, 1]
        e_sm = np.mean(np.abs((theta[:, 0] - theta[:, 0].mean()) - (logsm_gt - logsm_gt.mean())))
        e_l = np.mean(np.abs(theta[:, 1] - lift_gt)) * 1000
        corr_sm.append(c); r_sm.append(e_sm); r_lift.append(e_l)
        print(f"  seq{k}: corr(sigma_mu)={c:+.3f}  demeaned dex err={e_sm:.3f}  liftoff err={e_l:.0f} mm")
    print(f"\nE1b summary (n={n_seq}, noise={noise}):")
    print(f"  sigma_mu: corr={np.mean(corr_sm):+.3f}±{np.std(corr_sm):.3f}, "
          f"demeaned err={np.mean(r_sm):.3f} dex  -> {'IDENTIFIABLE (shape)' if np.mean(corr_sm) > 0.7 else 'weak'}")
    print(f"  liftoff : err={np.mean(r_lift):.0f} mm (truth range ±20 mm) "
          f"-> {'NOT identifiable under calibration freedom (G-liftoff degeneracy, as expected)' if np.mean(r_lift) > 20 else 'partially identifiable'}")
    return dict(corr_sm=float(np.mean(corr_sm)), err_sm=float(np.mean(r_sm)), err_lift=float(np.mean(r_lift)))


def sanity_plot(fid='102'):
    import pandas as pd, matplotlib.pyplot as plt
    try:
        import scienceplots  # noqa
        plt.style.use(['science', 'no-latex'])
    except Exception:
        pass
    p = os.path.join(REPO_PY, f'merged_data_with_fault_classes_{fid}.csv')
    df = pd.read_csv(p)
    cols = ['D1_32Hz_R', 'D1_32Hz_Theta', 'D1_100Hz_R', 'D1_100Hz_Theta',
            'D2_32Hz_R', 'D2_32Hz_Theta', 'D2_100Hz_R', 'D2_100Hz_Theta']
    X8 = df[cols].astype(float).ffill().fillna(0.0).values
    fault = pd.to_numeric(df['FaultClass'], errors='coerce').fillna(0).astype(int).clip(0, 3).values
    ln = pd.to_numeric(df['LabelNumber'], errors='coerce').fillna(-1).values
    theta, A = get_or_invert(fid, X8, verbose=True)
    bnds = np.where(np.diff(ln) != 0)[0] + 1
    anom = np.abs(A[:, :2]).sum(1)
    fig, ax = plt.subplots(3, 1, figsize=(7, 5), sharex=True)
    ax[0].plot(theta[:, 0], lw=0.8, color='#0072B2'); ax[0].set_ylabel(r'$\log_{10}(\sigma\mu)_{\rm eff}$')
    ax[0].set_title(f'(a) Inferred effective material state — campaign {fid}', loc='left')
    ax[1].plot(theta[:, 1] * 1000, lw=0.8, color='#009E73'); ax[1].set_ylabel('liftoff (mm)')
    ax[1].set_title('(b) Inferred effective liftoff', loc='left')
    ax[2].plot(anom, lw=0.6, color='#D55E00'); ax[2].set_ylabel('|anomaly|')
    ax[2].set_title('(c) Physics-normalised anomaly stream (joints/defects)', loc='left')
    for b in bnds:
        ax[2].axvline(b, color='k', ls='--', lw=0.4, alpha=0.4)
    for a_ in ax:
        a_.grid(True, alpha=0.3)
    if (fault > 0).any():
        for aa in np.split(np.where(fault > 0)[0], np.where(np.diff(np.where(fault > 0)[0]) > 1)[0] + 1):
            ax[2].axvspan(aa[0], aa[-1], color='orange', alpha=0.15, lw=0)
    ax[2].set_xlabel('Sample index')
    fig.tight_layout()
    out = os.path.join(REPO, 'figures', f'paper_E2_inversion_{fid}')
    fig.savefig(out + '.png', dpi=400, bbox_inches='tight'); fig.savefig(out + '.pdf', bbox_inches='tight')
    print(f"figure -> {out}.png (+.pdf)")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--id', default=None); ap.add_argument('--plot', action='store_true')
    ap.add_argument('--e2', action='store_true'); ap.add_argument('--e1b', action='store_true')
    a = ap.parse_args()
    if a.id:
        sanity_plot(a.id)
    if a.e1b:
        print("E1b identifiability under calibration freedom:")
        e1b_calibrated_identifiability()
    if a.e2:
        print("E2 drift attribution:")
        e2_drift_attribution()
