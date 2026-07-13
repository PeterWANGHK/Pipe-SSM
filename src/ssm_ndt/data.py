"""Data pipeline: load -> odometry resample -> low-rank common-mode removal -> features -> windows.

Grounded in the measured data-support findings (see EXPERIMENT_AUDIT.md / memory):
  - sampling is non-uniform in space  -> optional velocity (odometry) resampling
  - drift is low-rank common-mode      -> optional common-mode subspace removal
  - dual-frequency is complementary    -> keep both 32/100 Hz
  - |D1-D2| contrast is discriminative -> optional contrast features
"""
from __future__ import annotations
import os, glob
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from scipy.signal import savgol_filter

CH8 = ['D1_32Hz_R', 'D1_32Hz_Theta', 'D1_100Hz_R', 'D1_100Hz_Theta',
       'D2_32Hz_R', 'D2_32Hz_Theta', 'D2_100Hz_R', 'D2_100Hz_Theta']
ODO_DIST = ['Tool Absolute Distance', 'x_meters', 'Tool Distance']
ODO_SPEED = 'Tool Speed'


@dataclass
class FeatureConfig:
    velocity_norm: bool = True
    instance_norm: bool = True          # per-sequence standardisation (drift handling; offline ok)
    common_mode_removal: bool = True
    common_mode_rank: int = 2
    physics_features: bool = False       # append impedance-plane physics features (anomaly phase, skin-depth)
    physics_inversion: bool = False      # append Dodd-Deeds latent inversion features theta(s), a(s)
    use_freqs: tuple = (32, 100)        # set to (32,) for single-freq ablation
    contrast: bool = True               # |D1-D2| features
    grid_spacing_m: float = 0.02        # target uniform spatial spacing
    window: int = 1024
    stride: int = 512


def _find_full_csv(merged_path: str) -> str:
    """Prefer the *_with_xpx_* sibling (has odometry) if it exists."""
    base = os.path.basename(merged_path)
    fid = ''.join(ch for ch in base if ch.isdigit())
    cand = glob.glob(os.path.join(os.path.dirname(merged_path), f'merged_full_with_xpx_{fid}.csv'))
    return cand[0] if cand else merged_path


def load_raw(path: str):
    """Return (df8 [N,8] float, fault [N] int, labelnum [N] int, dist [N] float or None, speed or None)."""
    full = _find_full_csv(path)
    df = pd.read_csv(full)
    for c in CH8:
        if c not in df.columns:
            df[c] = 0.0
    X = df[CH8].astype(float).ffill().fillna(0.0)
    fault = pd.to_numeric(df.get('FaultClass', 0), errors='coerce').fillna(0).astype(int).values
    # severity set is {0,1,2,3}; a few files carry a stray label 4 (e.g. 021) -> clamp to ceiling
    fault = np.clip(fault, 0, 3)
    labelnum = pd.to_numeric(df.get('LabelNumber', 0), errors='coerce').fillna(0).astype(int).values
    dist = None
    for c in ODO_DIST:
        if c in df.columns:
            d = pd.to_numeric(df[c], errors='coerce').values.astype(float)
            if np.isfinite(d).sum() > 0.5 * len(d):
                dist = d; break
    speed = pd.to_numeric(df[ODO_SPEED], errors='coerce').values.astype(float) if ODO_SPEED in df.columns else None
    return X.values, fault, labelnum, dist, speed


def resample_uniform(X, fault, labelnum, dist, grid_spacing_m):
    """Resample onto a uniform spatial grid. Returns resampled arrays + spacing (m)."""
    if dist is None:
        return X, fault, labelnum, 1.0  # spacing unknown -> 1 "sample" unit
    d = np.array(dist, float)
    d = np.maximum.accumulate(np.nan_to_num(d, nan=np.nanmin(d[np.isfinite(d)]) if np.isfinite(d).any() else 0.0))
    lo, hi = d[0], d[-1]
    if hi - lo < 1e-6:
        return X, fault, labelnum, 1.0
    grid = np.arange(lo, hi, grid_spacing_m)
    idx = np.searchsorted(d, grid).clip(0, len(d) - 1)        # nearest-left original sample
    Xr = np.empty((len(grid), X.shape[1]), float)
    for j in range(X.shape[1]):
        Xr[:, j] = np.interp(grid, d, X[:, j])
    return Xr, fault[idx], labelnum[idx], float(grid_spacing_m)


def fit_common_mode(X_std_normal, rank):
    """Return projection matrix V (D,r) spanning the common-mode (drift) subspace."""
    if X_std_normal.shape[0] < rank + 1:
        return np.zeros((X_std_normal.shape[1], rank))
    _, _, vt = np.linalg.svd(X_std_normal - X_std_normal.mean(0, keepdims=True), full_matrices=False)
    return vt[:rank].T                                         # (D, r)


def build_features(X, fault, cfg: FeatureConfig, mean=None, std=None, V=None):
    """Standardise, optionally remove common-mode, select freqs, add contrast. Returns (F[N,Fdim], stats)."""
    if mean is None:
        mean, std = X.mean(0), X.std(0) + 1e-6
    Xs = (X - mean) / std

    feats = [Xs]
    if cfg.common_mode_removal:
        if V is None:
            V = fit_common_mode(Xs[fault == 0] if (fault == 0).any() else Xs, cfg.common_mode_rank)
        resid = Xs - (Xs @ V) @ V.T
        feats = [resid]                                       # working signal = drift-removed
        feats.append(Xs @ V)                                  # learned baseline projection (context)
    # frequency selection (drop 100Hz columns for single-freq ablation)
    if set(cfg.use_freqs) != {32, 100}:
        keep = [i for i, c in enumerate(CH8) if any(f'{f}Hz' in c for f in cfg.use_freqs)]
        feats = [f[:, keep] if f.shape[1] == len(CH8) else f for f in feats]
    if cfg.contrast:
        # |D1 - D2| per (freq, component) on standardised signal
        d1 = Xs[:, 0:4]; d2 = Xs[:, 4:8]
        feats.append(np.abs(d1 - d2))
    F = np.concatenate(feats, axis=1).astype(np.float32)
    return F, dict(mean=mean, std=std, V=V)


def physics_signals(X, win=301):
    """Impedance-plane physics features + a phase-derived severity proxy (skin-depth phenomenology).

    From raw 8 channels [D1_32R,D1_32T,D1_100R,D1_100T,D2_32R,D2_32T,D2_100R,D2_100T] build the
    complex anomaly (Z - smoothed baseline) and extract physically-meaningful descriptors:
    anomaly phase angle (deg/90), depth ratio |Z100|/|Z32|, cross-frequency phase diff.
    Returns (pi_feats [N,6], phys_target [N] in [0,1]) where higher target = more severe
    (anomaly phase rotated toward 0, per the measured 38.8deg->1.0deg trend).
    """
    N = len(X)
    win = max(5, min(win, (N // 2 * 2 - 1)))

    def anom(rcol, tcol):
        z = X[:, rcol] * np.exp(1j * np.deg2rad(X[:, tcol]))
        base = savgol_filter(z.real, win, 3, mode='interp') + 1j * savgol_filter(z.imag, win, 3, mode='interp')
        return z - base

    a132, a1100 = anom(0, 1), anom(2, 3)
    a232, a2100 = anom(4, 5), anom(6, 7)
    eps = 1e-6
    ph1 = np.angle(a132, deg=True); ph2 = np.angle(a232, deg=True)
    pi = np.column_stack([
        ph1 / 90.0, ph2 / 90.0,
        np.abs(a1100) / (np.abs(a132) + eps), np.abs(a2100) / (np.abs(a232) + eps),
        (np.angle(a1100, deg=True) - ph1) / 180.0, (np.angle(a2100, deg=True) - ph2) / 180.0,
    ]).astype(np.float32)
    pi = np.nan_to_num(pi, nan=0.0, posinf=0.0, neginf=0.0)
    # severity proxy: phase rotates from ~38deg (normal) toward 0 with severity -> t in [0,1]
    PHASE_REF, SCALE = 38.0, 12.0
    t = 1.0 / (1.0 + np.exp((ph1 - PHASE_REF) / SCALE))      # low phase -> t->1 (severe)
    return pi, np.nan_to_num(t, nan=0.0).astype(np.float32)


def boundary_field_from_labels(labelnum, sigma=2.0):
    """1 at label transitions, Gaussian-smoothed; also return boundary indices."""
    b = np.zeros(len(labelnum), np.float32)
    tr = np.where(np.diff(labelnum) != 0)[0] + 1
    for t in tr:
        for o in range(-int(3 * sigma), int(3 * sigma) + 1):
            i = t + o
            if 0 <= i < len(b):
                b[i] = max(b[i], np.exp(-o * o / (2 * sigma * sigma)))
    return b, tr


class ECTWindows(Dataset):
    """Windowed dataset over one or more campaign files, sharing fit stats from training files."""

    def __init__(self, files, cfg: FeatureConfig, stats=None, train=True, synthetic_class3=None):
        self.cfg = cfg
        self.items = []           # list of dicts per file (full sequences kept for eval)
        self.windows = []         # (file_i, start)
        self.synth = None         # synthetic class-3 feature windows (N, W, F)
        # fit standardisation/common-mode on the FIRST training file pool if no stats given
        fit_stats = stats
        for fi, path in enumerate(files):
            X, fault, labelnum, dist, speed = load_raw(path)
            pi_ext = None
            if cfg.physics_inversion:
                # Dodd-Deeds latent inversion on the RAW sequence (cached per campaign),
                # then resampled jointly with X so indices stay aligned.
                from ssm_ndt.latent_inversion import get_or_invert
                import re as _re
                fid = ''.join(_re.findall(r'\d+', os.path.basename(path))) or os.path.basename(path)
                theta, anom = get_or_invert(fid, X)
                X = np.hstack([X, theta, anom])                    # (N, 8+2+8)
            X, fault, labelnum, spacing = resample_uniform(
                X, fault, labelnum, dist if cfg.velocity_norm else None, cfg.grid_spacing_m)
            if cfg.physics_inversion:
                pi_ext = X[:, 8:]                                   # theta(2) + anomaly(8)
                X = X[:, :8]
            if cfg.instance_norm:
                # each sequence self-normalises (and refits its own common-mode subspace, unsupervised)
                F, used = build_features(X, fault, cfg, mean=None, std=None, V=None)
            else:
                F, used = build_features(X, fault, cfg,
                                         mean=fit_stats['mean'] if fit_stats else None,
                                         std=fit_stats['std'] if fit_stats else None,
                                         V=fit_stats['V'] if (fit_stats and cfg.common_mode_removal) else None)
            if fit_stats is None:
                fit_stats = used      # freeze stats from first file for the rest (train-time)
            bfield, btrue = boundary_field_from_labels(labelnum)
            if cfg.physics_features:
                pi, phys = physics_signals(X)
                pimu, pisd = pi.mean(0), pi.std(0) + 1e-6
                F = np.concatenate([F, ((pi - pimu) / pisd).astype(np.float32)], axis=1)
            else:
                phys = np.zeros(len(F), np.float32)
            if pi_ext is not None:
                # standardise inverted latents/anomaly per sequence and append
                pm, ps = pi_ext.mean(0), pi_ext.std(0) + 1e-6
                F = np.concatenate([F, ((pi_ext - pm) / ps).astype(np.float32)], axis=1)
            self.items.append(dict(F=F, fault=fault.astype(np.int64), bfield=bfield, phys=phys,
                                   btrue=btrue, labelnum=labelnum, spacing=spacing, path=path))
            N = len(F)
            step = cfg.stride if train else cfg.window
            for s in range(0, max(1, N - cfg.window + 1), step):
                self.windows.append((fi, s))
        self.stats = fit_stats
        self.feat_dim = self.items[0]['F'].shape[1]
        self.train = train
        self.augment = False               # set True to jitter defect windows during training
        # GAN class-3 synthetic windows -> appended as extra training items (classification-only)
        if train and synthetic_class3 and os.path.exists(synthetic_class3):
            syn = np.load(synthetic_class3).astype(np.float32)
            if syn.shape[2] == self.feat_dim:
                self.synth = syn
                base = len(self.windows)
                for j in range(len(syn)):
                    self.windows.append(('synth', j))
                print(f"[data] injected {len(syn)} synthetic class-3 windows ({base}->{len(self.windows)})")

    def __len__(self):
        return len(self.windows)

    def window_weights(self, defect_boost=4.0):
        """Sampling weight per window: defect-heavy windows oversampled (fights imbalance)."""
        w = np.ones(len(self.windows), dtype=np.float32)
        for k, (fi, s) in enumerate(self.windows):
            if fi == 'synth':                          # synthetic class-3 -> max boost
                w[k] = 1.0 + defect_boost; continue
            fc = self.items[fi]['fault'][s:s + self.cfg.window]
            if len(fc):
                w[k] = 1.0 + defect_boost * float((fc > 0).mean())
        return torch.from_numpy(w)

    def __getitem__(self, k):
        fi, s = self.windows[k]
        W = self.cfg.window
        if fi == 'synth':                  # synthetic class-3 window (all class-3, no joints)
            w = self.synth[s]
            w = np.pad(w, ((0, W - len(w)), (0, 0))) if len(w) < W else w[:W]
            fc = np.full(W, 3, np.int64); bf = np.zeros(W, np.float32); ph = np.ones(W, np.float32)
            if self.augment:
                w = w + np.random.normal(0, 0.03, w.shape).astype(np.float32)
            return (torch.from_numpy(w.astype(np.float32)), torch.from_numpy(bf),
                    torch.from_numpy(fc), torch.from_numpy(ph))
        it = self.items[fi]
        e = s + W
        F = it['F'][s:e].copy(); bf = it['bfield'][s:e]; fc = it['fault'][s:e]; ph = it['phys'][s:e]
        if len(F) < W:
            pad = W - len(F)
            F = np.pad(F, ((0, pad), (0, 0))); bf = np.pad(bf, (0, pad))
            fc = np.pad(fc, (0, pad)); ph = np.pad(ph, (0, pad))
        if self.train and self.augment and (fc > 0).any():
            F = F + np.random.normal(0, 0.03, F.shape).astype(np.float32)
            F = F * np.float32(np.random.uniform(0.95, 1.05))
        return (torch.from_numpy(F), torch.from_numpy(bf.astype(np.float32)),
                torch.from_numpy(fc.astype(np.int64)), torch.from_numpy(ph.astype(np.float32)))


def class_weights(files, cfg, n_classes=4, regroup=False):
    counts = np.ones(n_classes)
    for p in files:
        _, fault, _, _, _ = load_raw(p)
        if regroup:
            fault = np.where(fault == 3, 2, fault)
        for c in range(n_classes):
            counts[c] += (fault == c).sum()
    w = counts.sum() / (n_classes * counts)
    return torch.tensor(w / w.mean(), dtype=torch.float32)
