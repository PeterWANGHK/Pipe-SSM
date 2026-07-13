"""
Pure-PyTorch selective state-space (Mamba/S6-style) block.

No dependency on the `mamba_ssm` CUDA package (which is painful on Windows / AMD).
Runs on CPU or CUDA. Uses a sequential selective scan — O(L) in length, vectorised
over batch and channels. For our sequence lengths (windowed to ~1k) this is fine; the
linear-in-L scaling is the point (see EXPERIMENT_PLAN C4).
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveSSM(nn.Module):
    """Single-direction selective SSM (S6 core) with diagonal state.

    Input/Output: (B, L, D). State size N per channel. Input-dependent Delta, B, C.
    """

    def __init__(self, d_model: int, d_state: int = 16, dt_rank: int | None = None):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dt_rank = dt_rank or max(1, d_model // 16)

        # x -> (delta_low_rank, B, C)
        self.x_proj = nn.Linear(d_model, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_model, bias=True)

        # A is diagonal, parameterised as -exp(A_log) for stability (negative real part).
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_model, 1)
        self.A_log = nn.Parameter(torch.log(A))           # (D, N)
        self.D = nn.Parameter(torch.ones(d_model))        # skip connection

        # init dt_proj bias so softplus(dt) starts in a sensible range
        with torch.no_grad():
            dt = torch.exp(torch.rand(d_model) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))  # inverse softplus

    @staticmethod
    def _parallel_scan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Vectorised inclusive scan of h_t = a_t * h_{t-1} + b_t  (h_{-1}=0).

        Hillis-Steele over the time axis (dim=1): log2(L) steps, no recurrence in Python.
        Mathematically identical to the sequential loop; a,b: (B,L,D,N).
        """
        L = a.shape[1]
        A = a; H = b
        d = 1
        while d < L:
            # shift right by d along time: identity is a=1, h=0 for the missing prefix
            A_sh = torch.nn.functional.pad(A, (0, 0, 0, 0, d, 0))[:, :L]
            H_sh = torch.nn.functional.pad(H, (0, 0, 0, 0, d, 0))[:, :L]
            H = A * H_sh + H
            A = A * A_sh
            d *= 2
        return H

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, L, D)
        N = self.d_state
        A = -torch.exp(self.A_log)                        # (D, N)

        x_dbl = self.x_proj(x)                            # (B,L,dt_rank+2N)
        dt, Bm, Cm = torch.split(x_dbl, [self.dt_rank, N, N], dim=-1)
        dt = F.softplus(self.dt_proj(dt))                 # (B,L,D) > 0

        # discretise: dA = exp(dt * A), dB = dt * B  (ZOH, simplified)
        dA = torch.exp(dt.unsqueeze(-1) * A)              # (B,L,D,N)
        dBx = dt.unsqueeze(-1) * Bm.unsqueeze(2) * x.unsqueeze(-1)  # (B,L,D,N)

        h = self._parallel_scan(dA, dBx)                  # (B,L,D,N)
        y = torch.einsum('bldn,bln->bld', h, Cm)          # (B,L,D)
        y = y + x * self.D
        return y


class BiMambaBlock(nn.Module):
    """Bidirectional Mamba-style block: norm -> in_proj -> conv -> SiLU -> (fwd+bwd SSM) -> gate -> out."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2,
                 bidirectional: bool = True, dropout: float = 0.0):
        super().__init__()
        self.bidirectional = bidirectional
        d_inner = expand * d_model
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)
        self.conv1d = nn.Conv1d(d_inner, d_inner, kernel_size=d_conv,
                                groups=d_inner, padding=d_conv - 1, bias=True)
        self.ssm_fwd = SelectiveSSM(d_inner, d_state)
        self.ssm_bwd = SelectiveSSM(d_inner, d_state) if bidirectional else None
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,L,D)
        res = x
        x = self.norm(x)
        xz = self.in_proj(x)
        xi, z = xz.chunk(2, dim=-1)                       # (B,L,d_inner) each

        L = xi.shape[1]
        xc = self.conv1d(xi.transpose(1, 2))[..., :L].transpose(1, 2)
        xc = F.silu(xc)

        y = self.ssm_fwd(xc)
        if self.ssm_bwd is not None:
            y_b = self.ssm_bwd(torch.flip(xc, dims=[1]))
            y = y + torch.flip(y_b, dims=[1])

        y = y * F.silu(z)                                 # gate
        y = self.out_proj(y)
        return res + self.drop(y)
