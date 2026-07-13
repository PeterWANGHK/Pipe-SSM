"""Train + evaluate SST-SSM. Produces results JSON from committed code (audit rule).

Examples
--------
# smoke (tiny) on one file, auto train/test split by windows:
python -m ssm_ndt.train --ids 102 --window 256 --epochs 1 --d-model 48 --layers 2 --smoke

# cross-campaign OOD:
python -m ssm_ndt.train --train-ids 102 88003 --test-ids 86002 --epochs 8

# ablation rows:
python -m ssm_ndt.train --ids 102 --no-common-mode
python -m ssm_ndt.train --ids 102 --backbone gru
"""
from __future__ import annotations
import os, sys, json, time, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

# allow `python -m ssm_ndt.train` and `python ssm_ndt/train.py`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ssm_ndt.data import FeatureConfig, ECTWindows, class_weights, CH8
from ssm_ndt.model import SSTSSM, total_loss
from ssm_ndt import metrics as M

PYDIR = os.path.dirname(os.path.abspath(__file__))
REPO_PY = os.path.dirname(PYDIR)


def id_to_path(i):
    return os.path.join(REPO_PY, f'merged_data_with_fault_classes_{i}.csv')


@torch.no_grad()
def evaluate(model, ds, cfg, device, tol_m=0.5, n_classes=4):
    model.eval()
    per_file, all_yt, all_yp = [], [], []
    for it in ds.items:
        F = torch.from_numpy(it['F']).unsqueeze(0).to(device)
        N = F.shape[1]; W = cfg.window
        bprob = np.zeros(N); cprob = np.zeros((N, n_classes))
        for s in range(0, N, W):
            chunk = F[:, s:s + W]
            bl, cl = model(chunk)
            bprob[s:s + chunk.shape[1]] = torch.sigmoid(bl)[0, :chunk.shape[1]].cpu().numpy()
            cprob[s:s + chunk.shape[1]] = torch.softmax(cl, -1)[0, :chunk.shape[1]].cpu().numpy()
        ypred = cprob.argmax(1); ytrue = it['fault']
        all_yt.append(ytrue); all_yp.append(ypred)
        spacing = it['spacing'] if it['spacing'] != 1.0 else None   # None => nominal mm/sample
        loc = M.localization_metrics(bprob, it['btrue'], seq_len=N,
                                     spacing_m=spacing, tol_m=tol_m)
        pp = M.propagated_precision(bprob, ytrue, ypred)
        per_file.append(dict(path=os.path.basename(it['path']), prop_precision=pp, **loc))
    cm = M.classification_metrics(np.concatenate(all_yt), np.concatenate(all_yp), n_classes=n_classes)

    def agg(key):
        return float(np.nanmean([d[key] for d in per_file]))
    return dict(
        # localization (consistent set)
        dev_drift_samples=agg('dev_drift_samples'),
        dev_distance_m=agg('dev_distance_m'),
        dev_distance_samples=agg('dev_distance_samples'),
        dev_percentage=agg('dev_percentage'),
        dev_min_m=agg('dev_min_m'), dev_max_m=agg('dev_max_m'),
        above_thresh=agg('above_thresh'),
        # classification given segmentation
        prop_precision=agg('prop_precision'),
        **cm, per_file=per_file)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ids', nargs='*', default=None, help='single pool, auto window-split')
    ap.add_argument('--train-ids', nargs='*', default=None)
    ap.add_argument('--test-ids', nargs='*', default=None)
    ap.add_argument('--window', type=int, default=512)
    ap.add_argument('--stride', type=int, default=256)
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--d-model', type=int, default=96)
    ap.add_argument('--layers', type=int, default=4)
    ap.add_argument('--backbone', default='ssm',
                    choices=['ssm', 'gru', 'lstm', 'transformer', 'tcn', 'cnnlstm', 'patchtst'])
    ap.add_argument('--gan-synthetic', default=None, help='path to synthetic_class3.npy to inject')
    ap.add_argument('--no-velocity-norm', action='store_true')
    ap.add_argument('--no-instance-norm', action='store_true')
    ap.add_argument('--no-common-mode', action='store_true')
    ap.add_argument('--freqs', nargs='*', type=int, default=[32, 100])
    ap.add_argument('--no-contrast', action='store_true')
    ap.add_argument('--decoupled', action='store_true')
    ap.add_argument('--no-focal', action='store_true', help='use plain CE instead of focal loss')
    ap.add_argument('--no-balanced-sampler', action='store_true', help='disable defect-window oversampling')
    ap.add_argument('--no-augment', action='store_true', help='disable defect-window jitter augmentation')
    ap.add_argument('--boundary-pos-weight', type=float, default=10.0)
    ap.add_argument('--regroup', action='store_true', help='merge severity 3->2 ({0,low,high}, 3 classes)')
    ap.add_argument('--physics-features', action='store_true', help='append impedance-plane physics features')
    ap.add_argument('--physics-inversion', action='store_true', help='append Dodd-Deeds latent inversion theta/anomaly features')
    ap.add_argument('--physics-loss', action='store_true', help='add skin-depth severity-proxy consistency loss')
    ap.add_argument('--lam-phys', type=float, default=0.3)
    ap.add_argument('--backbone-layers', type=int, default=None)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--out', default=None)
    ap.add_argument('--save-ckpt', default=None, help='path to save model checkpoint for inference')
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    cfg = FeatureConfig(velocity_norm=not args.no_velocity_norm,
                        instance_norm=not args.no_instance_norm,
                        common_mode_removal=not args.no_common_mode,
                        use_freqs=tuple(args.freqs), contrast=not args.no_contrast,
                        physics_features=args.physics_features,
                        physics_inversion=args.physics_inversion,
                        window=args.window, stride=args.stride)

    if args.train_ids and args.test_ids:
        train_files = [id_to_path(i) for i in args.train_ids]
        test_files = [id_to_path(i) for i in args.test_ids]
        train_ds = ECTWindows(train_files, cfg, train=True, synthetic_class3=args.gan_synthetic)
        test_ds = ECTWindows(test_files, cfg, stats=train_ds.stats, train=False)
        split = f"OOD train={args.train_ids} test={args.test_ids}"
    else:
        ids = args.ids or ['102']
        files = [id_to_path(i) for i in ids]
        full = ECTWindows(files, cfg, train=True)
        n = len(full.windows); k = int(0.8 * n)
        train_ds = full
        train_ds.windows = full.windows[:k]
        test_ds = ECTWindows(files, cfg, stats=full.stats, train=False)
        split = f"ID pool={ids} (80/20 window split)"

    n_classes = 4
    if args.regroup:                                   # {0 normal, 1 low, 2 high(=orig 2&3)}
        n_classes = 3
        for ds in (train_ds, test_ds):
            for it in ds.items:
                it['fault'][it['fault'] == 3] = 2
        split += " | regroup{0,low,high}"

    cw = class_weights(args.train_ids and [id_to_path(i) for i in args.train_ids]
                       or [id_to_path(i) for i in (args.ids or ['102'])], cfg, n_classes=n_classes,
                       regroup=args.regroup)

    model = SSTSSM(train_ds.feat_dim, d_model=args.d_model, n_layers=args.layers,
                   backbone=args.backbone, coupled=not args.decoupled, n_classes=n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    train_ds.augment = not args.no_augment
    if args.no_balanced_sampler:
        dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=False)
    else:
        from torch.utils.data import WeightedRandomSampler
        wts = train_ds.window_weights()
        sampler = WeightedRandomSampler(wts, num_samples=len(wts), replacement=True)
        dl = DataLoader(train_ds, batch_size=args.batch, sampler=sampler, drop_last=False)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[setup] {split} | feat_dim={train_ds.feat_dim} | backbone={args.backbone} | "
          f"params={n_params/1e3:.0f}K | focal={not args.no_focal} balanced={not args.no_balanced_sampler} "
          f"aug={not args.no_augment} | device={device}")

    t0 = time.time()
    for ep in range(args.epochs):
        model.train(); losses = []
        for F_, bf, fc, ph in dl:
            F_, bf, fc, ph = F_.to(device), bf.to(device), fc.to(device), ph.to(device)
            bl, cl = model(F_)
            loss, parts = total_loss(bl, cl, bf, fc, class_w=cw, focal=not args.no_focal,
                                     boundary_pos_weight=args.boundary_pos_weight)
            if args.physics_loss:
                import torch.nn.functional as _F
                loss = loss + args.lam_phys * _F.mse_loss(model.physics_pred(), ph)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); losses.append(float(loss))
            if args.smoke:
                break
        print(f"  epoch {ep+1}/{args.epochs} loss={np.mean(losses):.4f}")
        if args.smoke:
            break
    train_time = time.time() - t0

    res = evaluate(model, test_ds, cfg, device, n_classes=n_classes)
    res.update(dict(split=split, backbone=args.backbone, feat_dim=train_ds.feat_dim,
                    n_params=n_params, train_time_s=train_time, config=vars(args)))
    print("\n=== RESULTS ===")
    print(f"  Localization: Drift={res['dev_drift_samples']:+.1f} smp | "
          f"Distance={res['dev_distance_m']:.3f} m ({res['dev_distance_samples']:.1f} smp) | "
          f"Dev%={res['dev_percentage']:.2f}% | Min/Max={res['dev_min_m']:.3f}/{res['dev_max_m']:.3f} m | "
          f"Above-thresh={res['above_thresh']:.3f}")
    print(f"  Classification: MacroF1={res['F1_macro']:.3f} | F1w={res['F1_weighted']:.3f} | "
          f"PropPrec={res['prop_precision']:.3f} | F1_per_class={[round(x,3) for x in res['F1_per_class']]}")

    os.makedirs(os.path.join(REPO_PY, 'results'), exist_ok=True)
    tag = args.out or f"{args.backbone}_{'-'.join(args.ids or args.train_ids or ['x'])}_seed{args.seed}"
    out = os.path.join(REPO_PY, 'results', f'{tag}.json')
    with open(out, 'w') as f:
        json.dump(res, f, indent=2)
    print(f"  saved -> {out}")

    if args.save_ckpt:
        st = train_ds.stats
        ckpt = dict(state_dict=model.state_dict(), feat_dim=train_ds.feat_dim,
                    d_model=args.d_model, n_layers=args.layers, backbone=args.backbone,
                    coupled=not args.decoupled, cfg=vars(cfg),
                    stats=dict(mean=st['mean'].tolist(), std=st['std'].tolist(),
                               V=(st['V'].tolist() if st.get('V') is not None else None)))
        torch.save(ckpt, args.save_ckpt)
        print(f"  checkpoint -> {args.save_ckpt}")


if __name__ == '__main__':
    main()
