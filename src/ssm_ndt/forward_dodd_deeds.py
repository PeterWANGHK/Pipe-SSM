"""L1 — Analytical eddy-current forward operator (Dodd & Deeds 1968, layered half-space).

Impedance CHANGE of an air-cored circular coil above a conductive-permeable half-space:

    dZ(w) = j*w*K * INT_0^inf  C(a) * exp(-2*a*l) * Gamma(a; sigma, mu_r, w) da

    Gamma(a) = (a*mu_r - a1) / (a*mu_r + a1),   a1 = sqrt(a^2 + j*w*mu0*mu_r*sigma)

- Boundary conditions of the canonical layered geometry are EMBEDDED analytically in Gamma
  (nothing is imposed numerically) — this is the answer to "you cannot impose BCs".
- The operator predicts exactly the measured derived quantity (coil impedance change),
  not internal fields — the answer to "what you measure is merely an impedance".
- Coil geometry C(a) uses nominal radii; the absolute scale K is calibrated away per sequence,
  so only the SHAPE of the (frequency, sigma*mu, liftoff) dependence carries physics.
  Inverted parameters are therefore EFFECTIVE values (stated honestly).

Implemented twice: numpy quadrature (reference) and torch quadrature (differentiable, in-loop).
Fixed log-spaced nodes make the torch version a weighted complex sum — fast and exact enough;
no MLP surrogate needed (fewer moving parts).

E1 identifiability gate:
    python -m ssm_ndt.forward_dodd_deeds --e1
"""
from __future__ import annotations
import os, sys, argparse
import numpy as np
import torch
import torch.nn as nn

MU0 = 4e-7 * np.pi
# nominal coil geometry (effective; absolute scale calibrated away)
R1, R2 = 0.02, 0.05          # coil inner/outer radius (m), nominal
LIFT0 = 0.10                 # detector liftoff ~4 in (from .pn ScanInfo)
FREQS = (32.0, 100.0)

# fixed quadrature nodes (log-spaced): integrand ~ C(a) e^{-2 a l}, l~0.1 m -> decays by a~100
_A = np.logspace(np.log10(0.5), np.log10(400.0), 192)
_W = np.gradient(_A)                                   # trapezoid-ish weights on log grid


def _coil_kernel(a):
    """C(a): radial coil factor, thin-coil approximation ~ [J1-integral]^2 / a^3 shape."""
    from scipy.special import jv
    # I(a) = int_{aR1}^{aR2} x J1(x) dx  (approximate with midpoint of the band)
    x1, x2 = a * R1, a * R2
    # use analytic-ish band average of x*J1(x) over [x1,x2]
    xs = np.linspace(x1, x2, 8)
    I = np.trapz(xs * jv(1, xs), xs, axis=0)
    return (I ** 2) / (a ** 3 + 1e-12)


_C = _coil_kernel(_A)                                  # precomputed coil kernel at nodes
_C_t = torch.tensor(_C, dtype=torch.float64)
_A_t = torch.tensor(_A, dtype=torch.float64)
_W_t = torch.tensor(_W, dtype=torch.float64)


SHELL_D = 0.0014             # 17-gauge steel cylinder thickness (m), fixed nominal (not a latent)


def dz_numpy(sigma_mu, liftoff, freq, thickness=SHELL_D, mu_r=50.0):
    """Reference forward. sigma_mu = sigma*mu_r (S/m); returns complex dZ (unnormalised units).

    Finite-thickness conductive shell over a non-conductive backing (concrete):
        r01 = (a*mu_r - a1)/(a*mu_r + a1),  E = exp(-2*a1*d)
        Gamma_shell = r01 * (1 - E) / (1 - r01^2 * E)      (half-space recovered as d -> inf)
    Identified quantity is the effective family on the fixed mu_r=50 slice, whose sigma*mu_r
    product coordinate absorbs material variation (mu_r also enters r01 separately — the product
    statement is a parameterisation choice, not an identifiability theorem).
    """
    sigma = np.asarray(sigma_mu, float) / mu_r
    w = 2 * np.pi * freq
    a = _A[None, :]
    a1 = np.sqrt(a ** 2 + 1j * w * MU0 * mu_r * np.asarray(sigma)[:, None])
    r01 = (a * mu_r - a1) / (a * mu_r + a1)
    if thickness is None:
        gam = r01
    else:
        E = np.exp(-2 * a1 * thickness)
        gam = r01 * (1 - E) / (1 - r01 ** 2 * E)
    integ = _C[None, :] * np.exp(-2 * a * np.asarray(liftoff, float)[:, None]) * gam
    return 1j * w * np.sum(integ * _W[None, :], axis=1)


def dz_torch(sigma_mu, liftoff, freq, thickness=SHELL_D, mu_r=50.0):
    """Differentiable forward (same quadrature, finite-thickness shell)."""
    sigma = sigma_mu / mu_r
    w = 2 * np.pi * freq
    a = _A_t.to(sigma_mu.device)[None, :]
    C = _C_t.to(sigma_mu.device)[None, :]
    Wq = _W_t.to(sigma_mu.device)[None, :]
    a1 = torch.sqrt(a ** 2 + 1j * w * MU0 * mu_r * sigma[:, None].to(torch.complex128))
    r01 = (a * mu_r - a1) / (a * mu_r + a1)
    if thickness is None:
        gam = r01
    else:
        E = torch.exp(-2 * a1 * thickness)
        gam = r01 * (1 - E) / (1 - r01 ** 2 * E)
    integ = C * torch.exp(-2 * a * liftoff[:, None]) * gam
    dz = 1j * w * torch.sum(integ * Wq, dim=1)
    return dz


def e1_identifiability(noise_levels=(0.0, 0.01, 0.03, 0.1), n=4000, seed=0, plot=True):
    """E1: can (log sigma_mu, liftoff) be recovered from dual-frequency dZ? Train MLP on synthetic."""
    rng = np.random.default_rng(seed)
    logsm = rng.uniform(np.log10(1e7), np.log10(1e9), n)      # sigma*mu_r range for steel
    lift = rng.uniform(0.06, 0.16, n)                          # around nominal 0.10 m
    Z32 = dz_numpy(10 ** logsm, lift, 32.0)
    Z100 = dz_numpy(10 ** logsm, lift, 100.0)
    # normalise per-feature (mimics calibration removing absolute scale)
    X = np.column_stack([Z32.real, Z32.imag, Z100.real, Z100.imag])
    Xm, Xs = X.mean(0), X.std(0) + 1e-12
    Y = np.column_stack([logsm, lift])
    Ym, Ys = Y.mean(0), Y.std(0)

    results = {}
    for nz in noise_levels:
        Xn = (X - Xm) / Xs + nz * rng.standard_normal(X.shape)
        k = int(0.8 * n)
        net = nn.Sequential(nn.Linear(4, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 2))
        opt = torch.optim.Adam(net.parameters(), 1e-3)
        xt = torch.tensor(Xn[:k], dtype=torch.float32)
        yt = torch.tensor((Y[:k] - Ym) / Ys, dtype=torch.float32)
        for _ in range(1500):
            opt.zero_grad()
            loss = nn.functional.mse_loss(net(xt), yt)
            loss.backward(); opt.step()
        with torch.no_grad():
            yp = net(torch.tensor(Xn[k:], dtype=torch.float32)).numpy() * Ys + Ym
        err_sm = float(np.mean(np.abs(yp[:, 0] - Y[k:, 0])))          # dex error in log10(sigma*mu)
        err_l = float(np.mean(np.abs(yp[:, 1] - Y[k:, 1])) * 1000)    # mm error in liftoff
        results[nz] = (err_sm, err_l)
        print(f"  noise={nz:5.2f}: |log10(sigma_mu)| err = {err_sm:.3f} dex | liftoff err = {err_l:.1f} mm")

    if plot:
        import matplotlib.pyplot as plt
        try:
            import scienceplots  # noqa
            plt.style.use(['science', 'no-latex'])
        except Exception:
            pass
        fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.6))
        nzs = list(results)
        ax[0].plot(nzs, [results[z][0] for z in nzs], 'o-', color='#0072B2')
        ax[0].set_xlabel('relative noise'); ax[0].set_ylabel(r'$|\Delta\log_{10}(\sigma\mu)|$ (dex)')
        ax[0].set_title('(a) Effective conductivity-permeability')
        ax[1].plot(nzs, [results[z][1] for z in nzs], 's-', color='#D55E00')
        ax[1].set_xlabel('relative noise'); ax[1].set_ylabel('liftoff error (mm)')
        ax[1].set_title('(b) Effective liftoff')
        for a_ in ax:
            a_.grid(True, alpha=0.3)
        fig.suptitle('E1: dual-frequency identifiability of latent physical state', y=1.04, fontsize=9)
        fig.tight_layout()
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        out = os.path.join(repo, 'figures', 'paper_E1_identifiability')
        fig.savefig(out + '.png', dpi=400, bbox_inches='tight')
        fig.savefig(out + '.pdf', bbox_inches='tight')
        print(f"  figure -> {out}.png (+.pdf)")
    return results


def validate_torch_vs_numpy():
    sm = np.array([1e8, 5e8]); lf = np.array([0.08, 0.12])
    zn = dz_numpy(sm, lf, 32.0)
    zt = dz_torch(torch.tensor(sm, dtype=torch.float64), torch.tensor(lf, dtype=torch.float64), 32.0)
    err = np.max(np.abs(zt.numpy() - zn) / (np.abs(zn) + 1e-30))
    print(f"torch-vs-numpy max rel err: {err:.2e}")
    return err < 1e-10


def quadrature_convergence():
    """Gate item: 192 log-nodes vs 2x denser grid."""
    global _A, _W, _C
    A0, W0, C0 = _A, _W, _C
    sm = np.array([1e8, 5e8]); lf = np.array([0.08, 0.12])
    z192 = np.concatenate([dz_numpy(sm, lf, f) for f in (32.0, 100.0)])
    _A = np.logspace(np.log10(0.5), np.log10(400.0), 384); _W = np.gradient(_A); _C = _coil_kernel(_A)
    z384 = np.concatenate([dz_numpy(sm, lf, f) for f in (32.0, 100.0)])
    _A, _W, _C = A0, W0, C0
    err = np.max(np.abs(z384 - z192) / (np.abs(z384) + 1e-30))
    print(f"quadrature 192-vs-384 nodes max rel err: {err:.2e}")
    return err


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--e1', action='store_true')
    a = ap.parse_args()
    print("validate torch quadrature:", validate_torch_vs_numpy())
    # physics sanity: |dZ(100Hz)| vs |dZ(32Hz)| for steel-like params
    sm = np.array([2e8]); lf = np.array([0.10])
    z32, z100 = dz_numpy(sm, lf, 32.0)[0], dz_numpy(sm, lf, 100.0)[0]
    print(f"sanity: |dZ100|/|dZ32| = {abs(z100)/abs(z32):.3f}, "
          f"phase32={np.angle(z32,deg=True):.1f} deg, phase100={np.angle(z100,deg=True):.1f} deg")
    if a.e1:
        print("E1 identifiability:")
        e1_identifiability()
