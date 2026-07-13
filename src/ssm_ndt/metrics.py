"""Honest, internally-consistent evaluation metrics.

Metric philosophy (per project definition):
  Localization = deviation of predicted joint start/end positions w.r.t. ground-truth, all
  derived from ONE signed-deviation array so the numbers are mutually consistent:
    - Deviation Drift      : mean SIGNED deviation (systematic early/late bias)   [samples]
    - Deviation Distance   : mean ABSOLUTE deviation (localization error)         [samples & m]
    - Deviation Percentage : mean(|dev| / local segment length) * 100            [%]
    - Min / Max Deviation  : extremes of |dev|                                    [samples & m]
    - Above-Threshold      : fraction of true boundaries localized within tol     [reliability]
  Propagated Precision = classification quality GIVEN the predicted segmentation: predicted
  boundaries define the segment spans, each span is classified by pooling, and minor
  segmentation error therefore propagates into the class decision.

Deviations are computed in the resampled grid's native unit (samples). When odometry is
available the grid spacing is in meters; otherwise a documented nominal mm/sample is used
only to express the *_m fields (clearly nominal), while *_samples fields are exact.
"""
from __future__ import annotations
import numpy as np
from scipy.signal import find_peaks
from sklearn.metrics import f1_score, precision_score

NOMINAL_MM_PER_SAMPLE = 18.0   # fleet-average from odometry files (102: 379 m / 21034 samples)


def peaks_from_field(field, prominence=None, distance=60, k=2.0):
    """Detect joint peaks. Adaptive mode (prominence=None): per-sequence robust height threshold
    (median + k·1.4826·MAD), which transfers across campaigns whose g(s) has different scale/noise —
    fixes the fixed-threshold collapse seen on held-out 86005. Pass a float prominence for legacy mode.
    """
    field = np.asarray(field, float)
    if field.max() > field.min():
        field = (field - field.min()) / (field.max() - field.min())
    if prominence is None:
        med = np.median(field); mad = np.median(np.abs(field - med))
        thr = min(0.5, max(0.10, med + k * 1.4826 * mad))    # adaptive, bounded
        pk, _ = find_peaks(field, height=thr, distance=distance, prominence=0.05)
    else:
        pk, _ = find_peaks(field, prominence=prominence, distance=distance)
    return pk


def _match_true_to_pred(pred_idx, true_idx, max_match_samples):
    """For each TRUE boundary, nearest PRED within max_match. Returns signed dev per matched true."""
    pred = np.sort(np.asarray(pred_idx, float))
    devs, matched_mask = [], np.zeros(len(true_idx), bool)
    for i, t in enumerate(true_idx):
        if len(pred) == 0:
            break
        j = int(np.argmin(np.abs(pred - t)))
        if abs(pred[j] - t) <= max_match_samples:
            devs.append(float(pred[j] - t))      # signed: + = predicted late (further along)
            matched_mask[i] = True
    return np.array(devs), matched_mask


def localization_metrics(pred_field, true_boundaries, seq_len,
                         spacing_m=None, tol_m=0.5, max_match_m=2.0,
                         prominence=None, distance=60):
    """All localization metrics from a single signed-deviation array (consistent by construction)."""
    true_idx = np.asarray(true_boundaries, int)
    mps = (spacing_m if spacing_m else NOMINAL_MM_PER_SAMPLE / 1000.0)  # meters per sample
    spacing_known = spacing_m is not None
    tol_s = tol_m / mps
    max_match_s = max_match_m / mps

    pred_idx = peaks_from_field(pred_field, prominence, distance)
    devs, matched = _match_true_to_pred(pred_idx, true_idx, max_match_s)

    # local segment length (samples) per true boundary = distance to next true boundary
    edges = np.concatenate(([0], true_idx, [seq_len]))
    seg_len_after = np.diff(edges)[1:]                       # length following each true boundary
    seg_len_for_true = seg_len_after[:len(true_idx)] if len(true_idx) else np.array([])

    out = dict(n_true=int(len(true_idx)), n_pred=int(len(pred_idx)),
               n_matched=int(len(devs)), spacing_known=bool(spacing_known),
               meters_per_sample=float(mps))
    if len(devs) == 0:
        out.update(dev_drift_samples=float('nan'), dev_distance_samples=float('nan'),
                   dev_distance_m=float('nan'), dev_percentage=float('nan'),
                   dev_min_m=float('nan'), dev_max_m=float('nan'),
                   dev_min_samples=float('nan'), dev_max_samples=float('nan'),
                   above_thresh=0.0, match_rate=0.0)
        return out

    absd = np.abs(devs)
    # percentage relative to the local segment lengths of matched true boundaries
    seg_for_matched = seg_len_for_true[matched] if len(seg_len_for_true) else np.array([np.nan])
    pct = float(np.mean(absd / np.maximum(seg_for_matched, 1e-9)) * 100.0)
    within_tol = absd <= tol_s

    out.update(
        dev_drift_samples=float(np.mean(devs)),             # SIGNED systematic bias
        dev_distance_samples=float(np.mean(absd)),
        dev_distance_m=float(np.mean(absd) * mps),
        dev_percentage=pct,
        dev_min_samples=float(np.min(absd)), dev_max_samples=float(np.max(absd)),
        dev_min_m=float(np.min(absd) * mps), dev_max_m=float(np.max(absd) * mps),
        above_thresh=float(within_tol.sum() / max(len(true_idx), 1)),   # reliability (recall@tol)
        match_rate=float(matched.mean()),
    )
    return out


def propagated_precision(pred_field, per_sample_true_class, per_sample_pred_class,
                         prominence=None, distance=60):
    """Classification precision GIVEN the predicted segmentation (length-weighted over predicted spans).

    Predicted boundaries -> spans; each span pooled to one predicted & one true class; a span is
    correct iff pooled-pred == pooled-true. Misaligned spans (segmentation error) pool the wrong
    samples and are penalised, so segmentation error propagates into classification.
    """
    yt = np.asarray(per_sample_true_class, int); yp = np.asarray(per_sample_pred_class, int)
    N = len(yt)
    pred_idx = peaks_from_field(pred_field, prominence, distance)
    edges = np.concatenate(([0], np.sort(pred_idx), [N])).astype(int)
    corr_len, tot_len = 0, 0
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            continue
        pc = np.bincount(yp[a:b], minlength=4).argmax()
        tc = np.bincount(yt[a:b], minlength=4).argmax()
        tot_len += (b - a)
        if pc == tc:
            corr_len += (b - a)
    return float(corr_len / max(tot_len, 1))


def classification_metrics(y_true, y_pred, n_classes=4):
    y_true = np.asarray(y_true, int); y_pred = np.asarray(y_pred, int)
    labels = list(range(n_classes))
    per = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {"F1_per_class": [float(x) for x in per],
            "F1_macro": float(f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)),
            "F1_weighted": float(f1_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0)),
            "precision_weighted": float(precision_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0))}
