"""Conditional rare-class augmentation: a 1-D DCGAN that synthesizes class-3 (highest severity)
feature-windows directly in the model's 14-dim input space, so they inject without re-deriving
features. Reuses the Generator/Discriminator design of synthetic_GAN.py (num_channels generalized).

Honest caveat: class-3 = ~13 real segments -> low diversity; GAN may memorize. We report this.

  python -m ssm_ndt.gan_class3 --epochs 600 --n-synth 400
"""
from __future__ import annotations
import os, sys, math, argparse
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ssm_ndt.data import FeatureConfig

PYDIR = os.path.dirname(os.path.abspath(__file__)); REPO_PY = os.path.dirname(PYDIR)
CLASS3_FILES = ['021', '102', '104', '86005']      # files containing class-3 segments


def _id_to_path(i):
    return os.path.join(REPO_PY, f'merged_data_with_fault_classes_{i}.csv')


def extract_class3_windows(seq_len=128, target_class=3, exclude=()):
    """Return real feature-windows (N, seq_len, F) dominated by the target class.

    `exclude`: campaign ids to omit (e.g. the held-out OOD test file) to prevent leakage.
    """
    from ssm_ndt.data import load_raw, resample_uniform, build_features
    cfg = FeatureConfig()
    wins = []
    for cid in [c for c in CLASS3_FILES if c not in exclude]:
        p = _id_to_path(cid)
        if not os.path.exists(p):
            continue
        X, fault, labelnum, dist, _ = load_raw(p)
        X, fault, labelnum, _ = resample_uniform(X, fault, labelnum, None, cfg.grid_spacing_m)
        F, _ = build_features(X, fault, cfg)
        idx = np.where(fault == target_class)[0]
        if len(idx) == 0:
            continue
        # contiguous runs of the target class
        splits = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
        for run in splits:
            a, b = run[0], run[-1]
            # slide windows across the run (+context), keep those >50% target class
            for s in range(max(0, a - seq_len // 4), min(len(F) - seq_len, b), seq_len // 2):
                w = F[s:s + seq_len]
                if w.shape[0] == seq_len and (fault[s:s + seq_len] == target_class).mean() > 0.4:
                    wins.append(w)
    if not wins:
        return np.empty((0, seq_len, build_features(np.zeros((1, 8)), np.zeros(1), cfg)[0].shape[1]), np.float32)
    return np.stack(wins).astype(np.float32)


class Gen(nn.Module):
    def __init__(self, latent, C, L):
        super().__init__()
        self.L0 = math.ceil(L / 8); self.C = C; self.L = L
        self.fc = nn.Linear(latent, 128 * self.L0)
        self.net = nn.Sequential(
            nn.BatchNorm1d(128), nn.Upsample(scale_factor=2), nn.Conv1d(128, 64, 3, 1, 1), nn.BatchNorm1d(64), nn.ReLU(True),
            nn.Upsample(scale_factor=2), nn.Conv1d(64, 32, 3, 1, 1), nn.BatchNorm1d(32), nn.ReLU(True),
            nn.Upsample(scale_factor=2), nn.Conv1d(32, C, 3, 1, 1))

    def forward(self, z):
        x = self.fc(z).view(z.size(0), 128, self.L0)
        x = self.net(x)
        return x[..., :self.L].transpose(1, 2)             # (B, L, C)


class Disc(nn.Module):
    def __init__(self, C, L):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(C, 32, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv1d(32, 64, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x.transpose(1, 2))


def train_and_generate(seq_len=128, latent=64, epochs=600, n_synth=400, out=None, exclude=()):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    real = extract_class3_windows(seq_len, exclude=exclude)
    print(f"[gan] real class-3 windows: {real.shape}")
    if len(real) < 8:
        print("[gan] too few class-3 windows; aborting"); return None
    F = real.shape[2]
    mu, sd = real.mean((0, 1)), real.std((0, 1)) + 1e-6
    realn = (real - mu) / sd
    Xr = torch.tensor(realn, device=dev)
    G, D = Gen(latent, F, seq_len).to(dev), Disc(F, seq_len).to(dev)
    og = torch.optim.Adam(G.parameters(), 2e-4, betas=(0.5, 0.9))
    od = torch.optim.Adam(D.parameters(), 2e-4, betas=(0.5, 0.9))
    bce = nn.BCEWithLogitsLoss()
    bs = min(64, len(Xr))
    for ep in range(epochs):
        idx = torch.randint(0, len(Xr), (bs,), device=dev)
        rb = Xr[idx]
        z = torch.randn(bs, latent, device=dev); fake = G(z).detach()
        od.zero_grad()
        ld = bce(D(rb), torch.ones(bs, 1, device=dev)) + bce(D(fake), torch.zeros(bs, 1, device=dev))
        ld.backward(); od.step()
        z = torch.randn(bs, latent, device=dev)
        og.zero_grad()
        lg = bce(D(G(z)), torch.ones(bs, 1, device=dev))
        lg.backward(); og.step()
        if (ep + 1) % 150 == 0:
            print(f"[gan] ep{ep+1} D={ld.item():.3f} G={lg.item():.3f}")
    G.eval()
    with torch.no_grad():
        synth = []
        for _ in range(math.ceil(n_synth / 256)):
            z = torch.randn(256, latent, device=dev)
            synth.append((G(z).cpu().numpy() * sd + mu))
        synth = np.concatenate(synth)[:n_synth].astype(np.float32)
    out = out or os.path.join(REPO_PY, 'synthetic_class3.npy')
    np.save(out, synth)
    print(f"[gan] saved {synth.shape} -> {out}")
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq-len', type=int, default=128)
    ap.add_argument('--epochs', type=int, default=600)
    ap.add_argument('--n-synth', type=int, default=400)
    ap.add_argument('--exclude', nargs='*', default=[], help='campaign ids to omit (held-out test)')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    train_and_generate(seq_len=a.seq_len, epochs=a.epochs, n_synth=a.n_synth,
                       out=a.out, exclude=tuple(a.exclude))
