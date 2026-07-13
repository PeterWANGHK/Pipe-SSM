"""C4 — Selective prediction for safety-critical triage.

The rare high-severity defects are not reliably learnable (see cycle-2 negative result), so instead
of silently mis-predicting them we ABSTAIN and flag them for expert review. Confidence combines
(i) model max-softmax and (ii) cross-detector agreement (D1 vs D2 consistency). We report a
risk-coverage curve + selective accuracy: at coverage c, the model auto-decides the most-confident
c% of segments; the rest go to a human.

  python -m ssm_ndt.selective_prediction --ckpt models/sstssm_deploy.pt --test 88005
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ssm_ndt.data import FeatureConfig, load_raw, resample_uniform, build_features, CH8
from ssm_ndt.model import SSTSSM

PYDIR = os.path.dirname(os.path.abspath(__file__)); REPO_PY = os.path.dirname(PYDIR)
REPO = os.path.dirname(REPO_PY); FIG = os.path.join(REPO, 'figures'); os.makedirs(FIG, exist_ok=True)


def _scienceplots():
    try:
        import scienceplots  # noqa
        plt.style.use(['science', 'no-latex', 'grid']); return True
    except Exception:
        return False


@torch.no_grad()
def infer_probs(ckpt_path, cid, device, n_classes):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = FeatureConfig(**ck['cfg'])
    p = os.path.join(REPO_PY, f'merged_data_with_fault_classes_{cid}.csv')
    X, fault, labelnum, dist, _ = load_raw(p)
    X, fault, labelnum, spacing = resample_uniform(X, fault, labelnum, None, cfg.grid_spacing_m)
    if n_classes == 3:
        fault = np.where(fault == 3, 2, fault)
    F, _ = build_features(X, fault, cfg)
    model = SSTSSM(ck['feat_dim'], d_model=ck['d_model'], n_layers=ck['n_layers'],
                   backbone=ck['backbone'], coupled=ck['coupled'], n_classes=n_classes).to(device).eval()
    model.load_state_dict(ck['state_dict'])
    Ft = torch.from_numpy(F).float().unsqueeze(0).to(device)
    N = Ft.shape[1]; W = cfg.window; cprob = np.zeros((N, n_classes))
    for s in range(0, N, W):
        _, cl = model(Ft[:, s:s + W])
        cprob[s:s + cl.shape[1]] = torch.softmax(cl, -1)[0, :cl.shape[1]].cpu().numpy()
    # cross-detector agreement: corr of |D1| and |D2| anomaly in a local window (proxy confidence)
    d1 = np.abs(X[:, 0]); d2 = np.abs(X[:, 4])
    return cprob, fault, labelnum, d1, d2


def segmentwise(cprob, fault, labelnum, d1, d2):
    segs = []
    for seg in np.unique(labelnum):
        m = labelnum == seg
        if m.sum() < 10:
            continue
        prob = cprob[m].mean(0)
        pred = int(prob.argmax()); true = int(np.bincount(fault[m]).argmax())
        conf_model = float(prob.max())                       # max-softmax
        # detector agreement: 1 - normalized |corr deficit| of D1,D2 in segment
        a, b = d1[m], d2[m]
        agree = float(np.corrcoef(a, b)[0, 1]) if len(a) > 3 else 0.0
        agree = 0.0 if np.isnan(agree) else agree
        conf = 0.7 * conf_model + 0.3 * max(0.0, agree)      # combined confidence
        segs.append((pred, true, conf, conf_model, max(0.0, agree)))
    return np.array(segs)


def risk_coverage(segs):
    pred, true, conf = segs[:, 0], segs[:, 1], segs[:, 2]
    order = np.argsort(-conf)                                 # most confident first
    pred, true = pred[order].astype(int), true[order].astype(int)
    cov, risk = [], []
    n = len(pred)
    for k in range(max(1, n // 20), n + 1, max(1, n // 20)):
        cov.append(k / n)
        risk.append(float((pred[:k] != true[:k]).mean()))    # error rate on auto-decided
    aurc = float(np.trapz(risk, cov))
    return np.array(cov), np.array(risk), aurc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=os.path.join(REPO_PY, 'models', 'sstssm_deploy.pt'))
    ap.add_argument('--test', default='88005')
    ap.add_argument('--n-classes', type=int, default=4)
    a = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cprob, fault, labelnum, d1, d2 = infer_probs(a.ckpt, a.test, device, a.n_classes)
    segs = segmentwise(cprob, fault, labelnum, d1, d2)
    cov, risk, aurc = risk_coverage(segs)
    full_err = float((segs[:, 0] != segs[:, 1]).mean())
    # selective accuracy at 80% coverage
    i80 = int(np.argmin(np.abs(cov - 0.8)))
    print(f"[C4] test={a.test} segments={len(segs)} | full-coverage error={full_err:.3f} | "
          f"AURC={aurc:.3f} | error@80%coverage={risk[i80]:.3f} (flag 20% for review)")

    _scienceplots()
    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.plot(cov * 100, risk * 100, 'o-', lw=1.5, ms=3, color='#6a0dad', label='SST-SSM selective')
    ax.axhline(full_err * 100, ls='--', color='grey', lw=1, label=f'no abstention ({full_err*100:.0f}%)')
    ax.axvline(80, ls=':', color='red', lw=1, alpha=0.6)
    ax.set_xlabel('Coverage (% auto-decided)'); ax.set_ylabel('Risk (error %)')
    ax.set_title(f'Selective prediction risk–coverage (test {a.test})'); ax.legend(fontsize=7)
    fig.tight_layout()
    out = os.path.join(FIG, f'sstssm_risk_coverage_{a.test}.png')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print(f"  saved -> {out}")
    json.dump({'cov': cov.tolist(), 'risk': risk.tolist(), 'aurc': aurc, 'full_err': full_err},
              open(os.path.join(REPO_PY, 'results', f'selective_{a.test}.json'), 'w'), indent=2)


if __name__ == '__main__':
    main()
