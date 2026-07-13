"""SST-SSM model: multi-channel embed -> bidirectional selective-SSM stack -> coupled heads.

Backbone is swappable (ssm | gru | transformer) for the EXPERIMENT_PLAN ablation C4 / SSM->X rows.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .ssm import BiMambaBlock


class _GRUBlock(nn.Module):
    def __init__(self, d_model, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.gru = nn.GRU(d_model, d_model, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(2 * d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        y, _ = self.gru(self.norm(x))
        return x + self.drop(self.proj(y))


class _TCNBlock(nn.Module):
    """Dilated causal Conv1d residual block (TCN baseline). Stack N -> exponential receptive field."""
    def __init__(self, d_model, dilation=1, k=3, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        pad = (k - 1) * dilation
        self.conv1 = nn.Conv1d(d_model, d_model, k, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(d_model, d_model, k, padding=pad, dilation=dilation)
        self.pad = pad; self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.norm(x).transpose(1, 2)
        L = h.shape[-1]
        h = F.relu(self.conv1(h)[..., :L])
        h = F.relu(self.conv2(h)[..., :L]).transpose(1, 2)
        return x + self.drop(h)


class _LSTMBlock(nn.Module):
    def __init__(self, d_model, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.lstm = nn.LSTM(d_model, d_model, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(2 * d_model, d_model); self.drop = nn.Dropout(dropout)

    def forward(self, x):
        y, _ = self.lstm(self.norm(x))
        return x + self.drop(self.proj(y))


class _AttnBlock(nn.Module):
    def __init__(self, d_model, nhead=4, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.ff = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 2 * d_model),
                                nn.GELU(), nn.Linear(2 * d_model, d_model))

    def forward(self, x):
        h = self.norm(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        return x + self.ff(x)


class _PatchTSTEncoder(nn.Module):
    """PatchTST-style encoder adapted for DENSE (per-sample) labeling.

    Patchify (patch_len, stride) -> linear patch embed -> Transformer over patches ->
    project each patch token back to its samples (overlap-add) -> per-sample d_model.
    Faithful to Nie et al. 2023 (patching + transformer); head re-purposed for seg+cls.
    """
    def __init__(self, d_model, n_layers, patch_len=16, stride=8, nhead=4, dropout=0.1):
        super().__init__()
        self.patch_len, self.stride = patch_len, stride
        self.embed = nn.Linear(d_model * patch_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, 2 * d_model, dropout,
                                           batch_first=True, activation='gelu')
        self.tr = nn.TransformerEncoder(layer, n_layers)
        self.unembed = nn.Linear(d_model, d_model * patch_len)

    def forward(self, x):                                  # (B,L,D)
        B, L, D = x.shape
        pl, st = self.patch_len, self.stride
        pad = (math.ceil((L - pl) / st) * st + pl) - L if L > pl else pl - L
        xp = F.pad(x.transpose(1, 2), (0, max(0, pad))).transpose(1, 2)
        Lp = xp.shape[1]
        starts = list(range(0, Lp - pl + 1, st))
        patches = torch.stack([xp[:, s:s + pl] for s in starts], 1)   # (B,P,pl,D)
        tok = self.embed(patches.reshape(B, len(starts), pl * D))     # (B,P,D)
        tok = self.tr(tok)
        out = self.unembed(tok).reshape(B, len(starts), pl, D)        # (B,P,pl,D)
        # overlap-add back to length Lp
        acc = x.new_zeros(B, Lp, D); cnt = x.new_zeros(B, Lp, 1)
        for i, s in enumerate(starts):
            acc[:, s:s + pl] += out[:, i]; cnt[:, s:s + pl] += 1
        return (acc / cnt.clamp_min(1))[:, :L]


class SSTSSM(nn.Module):
    def __init__(self, feat_dim: int, d_model: int = 96, n_layers: int = 4, n_classes: int = 4,
                 backbone: str = 'ssm', d_state: int = 16, coupled: bool = True, dropout: float = 0.1):
        super().__init__()
        self.coupled = coupled
        self.embed = nn.Sequential(nn.Linear(feat_dim, d_model), nn.GELU(), nn.LayerNorm(d_model))
        self.patchtst = _PatchTSTEncoder(d_model, n_layers) if backbone == 'patchtst' else None
        blocks = []
        for i in range(0 if backbone == 'patchtst' else n_layers):
            if backbone == 'ssm':
                blocks.append(BiMambaBlock(d_model, d_state=d_state, dropout=dropout))
            elif backbone == 'gru':
                blocks.append(_GRUBlock(d_model, dropout))
            elif backbone == 'lstm':
                blocks.append(_LSTMBlock(d_model, dropout))
            elif backbone == 'transformer':
                blocks.append(_AttnBlock(d_model, dropout=dropout))
            elif backbone == 'tcn':
                blocks.append(_TCNBlock(d_model, dilation=2 ** i, dropout=dropout))
            elif backbone == 'cnnlstm':                    # CNN front-end then BiLSTM
                blocks.append(_TCNBlock(d_model, dilation=1, dropout=dropout) if i < max(1, n_layers // 2)
                              else _LSTMBlock(d_model, dropout))
            else:
                raise ValueError(backbone)
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(d_model)
        self.boundary_head = nn.Linear(d_model, 1)
        self.depth_head = nn.Linear(d_model, 1)            # physics: skin-depth severity proxy
        self._h = None
        # coupled: classifier sees encoder feature + boundary logit (segment context)
        self.class_head = nn.Linear(d_model + (1 if coupled else 0), n_classes)

    def forward(self, x):                                  # x: (B,L,feat_dim)
        h = self.embed(x)
        if self.patchtst is not None:
            h = self.patchtst(h)
        for blk in self.blocks:
            h = blk(h)
        h = self.norm(h)
        self._h = h                                        # stash for physics_pred() (no signature change)
        b_logit = self.boundary_head(h)                   # (B,L,1)
        cin = torch.cat([h, b_logit], dim=-1) if self.coupled else h
        c_logit = self.class_head(cin)                    # (B,L,n_classes)
        return b_logit.squeeze(-1), c_logit

    def physics_pred(self):
        """Predicted skin-depth severity proxy in [0,1] from the last forward's encoder output."""
        return torch.sigmoid(self.depth_head(self._h).squeeze(-1))


def focal_ce(logits, target, weight=None, gamma=2.0):
    """Multi-class focal loss: down-weights easy (majority) samples to fight imbalance."""
    logp = F.log_softmax(logits, dim=-1)
    p = logp.exp()
    logp_t = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    p_t = p.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    loss = -((1 - p_t) ** gamma) * logp_t
    if weight is not None:
        loss = loss * weight.to(logits.device)[target]
    return loss.mean()


def total_loss(b_logit, c_logit, b_target, c_target, class_w=None, lam_seg=1.0, lam_cls=1.0,
               focal=True, focal_gamma=2.0, boundary_pos_weight=10.0):
    # boundary head: positives (joints) are rare -> upweight them
    pw = torch.tensor(boundary_pos_weight, device=b_logit.device)
    seg = F.binary_cross_entropy_with_logits(b_logit, b_target, pos_weight=pw)
    cw = class_w.to(c_logit.device) if class_w is not None else None
    flat_logit = c_logit.reshape(-1, c_logit.shape[-1]); flat_tgt = c_target.reshape(-1)
    if focal:
        cls = focal_ce(flat_logit, flat_tgt, weight=cw, gamma=focal_gamma)
    else:
        cls = F.cross_entropy(flat_logit, flat_tgt, weight=cw)
    return lam_seg * seg + lam_cls * cls, {"seg": float(seg.detach()), "cls": float(cls.detach())}
