"""Train PHANTOM or a baseline on the processed Kepler DR24 TCE set.

Every run writes a JSON record of its configuration, learning curve and test
metrics plus the raw test-set scores, so that ablations, ensembles and the
conformal calibration in ``scripts/evaluate.py`` all read from the same
artefacts rather than being recomputed ad hoc.

Examples::

    python scripts/train.py --model phantom  --seed 0 --tag full
    python scripts/train.py --model astronet --seed 0 --tag astronet
    python scripts/train.py --model phantom  --seed 0 --no-harmonic --tag no_harmonic
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, average_precision_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score,
)
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exonet.data import ScalarConditioner, TCEDataset, load_h5, make_splits
from exonet.model import PHANTOM, VIEW_TABLE, AstroNet, count_parameters


def build_scalars(blob, groups: list[str]):
    """Assemble the scalar feature matrix from the requested feature groups."""
    names, cols = [], []
    cat_cols, der_cols = blob["cat_cols"], blob["der_cols"]
    g = blob["groups"]
    if "transit" in groups:
        for c in g["transit_cols"]:
            names.append(c); cols.append(blob["catalog"][:, cat_cols.index(c)])
    if "stellar" in groups:
        for c in g["stellar_cols"]:
            names.append(c); cols.append(blob["catalog"][:, cat_cols.index(c)])
    if "dv" in groups:
        for c in g["dv_diag_cols"]:
            names.append(c); cols.append(blob["catalog"][:, cat_cols.index(c)])
    if "derived" in groups:
        for i, c in enumerate(der_cols):
            names.append(c); cols.append(blob["derived"][:, i])
    if not cols:  # a model with no scalar inputs still needs a well-formed tensor
        return np.zeros((len(blob["kepid"]), 1), dtype=np.float32), ["_none"]
    return np.stack(cols, axis=1).astype(np.float32), names


@torch.no_grad()
def evaluate(model, loader, device, use_decoder: bool):
    model.eval()
    scores, labels, params = [], [], []
    for views, scal, y, _ in loader:
        views = {k: v.to(device, non_blocking=True) for k, v in views.items()}
        scal = scal.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model(views, scal)
        scores.append(torch.sigmoid(out["logit"].float()).cpu().numpy())
        labels.append(y.numpy())
        if use_decoder and "params" in out:
            params.append(out["params"].float().cpu().numpy())
    return (
        np.concatenate(scores),
        np.concatenate(labels),
        np.concatenate(params) if params else None,
    )


def metrics(y: np.ndarray, s: np.ndarray, threshold: float = 0.5) -> dict:
    pred = (s >= threshold).astype(int)
    out = {
        "auc": float(roc_auc_score(y, s)),
        "ap": float(average_precision_score(y, s)),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
    }
    # Precision at fixed recall is the operationally meaningful number: a vetting
    # pipeline is run at high recall and judged on how much human follow-up the
    # resulting false positives cost.
    prec, rec, _ = precision_recall_curve(y, s)
    for target in (0.90, 0.95, 0.99):
        ok = rec >= target
        out[f"precision_at_recall_{int(target*100)}"] = (
            float(prec[ok].max()) if ok.any() else 0.0
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/processed/dr24_views.h5")
    ap.add_argument("--model", choices=["phantom", "astronet"], default="phantom")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--split-seed", type=int, default=-1,
        help="seed for the train/val/test partition; defaults to --seed. Fixing it\n             while varying --seed gives members that share a test set and can\n             therefore be ensembled",
    )
    ap.add_argument("--split-mode", choices=["group", "tce"], default="group")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--recon-weight", type=float, default=5.0)
    ap.add_argument("--no-harmonic", action="store_true")
    ap.add_argument("--no-decoder", action="store_true")
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument(
        "--views", default="all",
        help="comma-separated view subset, or 'all', or 'sv' for global+local only",
    )
    ap.add_argument(
        "--scalar-groups", default="transit,stellar,dv,derived",
        help="comma-separated subset of transit,stellar,dv,derived,none",
    )
    ap.add_argument("--out-dir", default="results/runs")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.views == "all":
        view_names = list(VIEW_TABLE.keys())
    elif args.views == "sv":
        view_names = ["global", "local"]
    else:
        view_names = args.views.split(",")

    blob = load_h5(args.data, view_names)
    groups = [g for g in args.scalar_groups.split(",") if g and g != "none"]
    scalars_raw, scalar_names = build_scalars(blob, groups)

    label = blob["label"]
    y = (label == "PC").astype(np.float32)
    kepid = blob["kepid"]

    split_seed = args.seed if args.split_seed < 0 else args.split_seed
    tr, va, te = make_splits(kepid, len(y), mode=args.split_mode, seed=split_seed)
    print(
        f"split[{args.split_mode}] train={len(tr)} val={len(va)} test={len(te)} "
        f"| positives {y[tr].mean():.3f}/{y[va].mean():.3f}/{y[te].mean():.3f}",
        flush=True,
    )

    cond = ScalarConditioner(scalar_names).fit(scalars_raw[tr])
    scalars = cond.transform(scalars_raw)

    # Regression targets are carried through for downstream parameter analysis
    # but are not used in the loss (see exonet/model.py).
    cat_cols = blob["cat_cols"]
    reg = np.stack(
        [
            blob["catalog"][:, cat_cols.index("tce_depth")],
            blob["catalog"][:, cat_cols.index("tce_duration")],
        ],
        axis=1,
    ).astype(np.float32)

    def subset(ix, augment):
        return TCEDataset(
            {n: blob["views"][n][ix] for n in view_names},
            scalars[ix], y[ix], reg[ix], augment=augment,
        )

    dl_kw = dict(num_workers=4, pin_memory=True, persistent_workers=True)
    train_dl = DataLoader(
        subset(tr, not args.no_augment), batch_size=args.batch_size,
        shuffle=True, drop_last=True, **dl_kw,
    )
    val_dl = DataLoader(subset(va, False), batch_size=256, **dl_kw)
    test_dl = DataLoader(subset(te, False), batch_size=256, **dl_kw)

    use_decoder = (args.model == "phantom") and not args.no_decoder
    if args.model == "phantom":
        model = PHANTOM(
            n_scalars=scalars.shape[1], view_names=view_names,
            use_decoder=use_decoder, use_harmonic=not args.no_harmonic,
        )
    else:
        model = AstroNet(n_scalars=scalars.shape[1])
    model = model.to(device)
    print(f"{args.model}: {count_parameters(model)/1e6:.2f}M parameters", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * len(train_dl), pct_start=0.15,
    )

    best_ap, best_state, best_epoch, history = -1.0, None, -1, []
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        tot_loss = tot_cls = tot_rec = 0.0
        n_batches = 0
        for views, scal, yy, _ in train_dl:
            views = {k: v.to(device, non_blocking=True) for k, v in views.items()}
            scal = scal.to(device, non_blocking=True)
            yy = yy.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                out = model(views, scal)
                cls_loss = F.binary_cross_entropy_with_logits(out["logit"].float(), yy)
                if use_decoder:
                    rec_loss = F.mse_loss(out["model"].float(), views["local"].float())
                else:
                    rec_loss = torch.zeros((), device=device)
                loss = cls_loss + args.recon_weight * rec_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            tot_loss += loss.item(); tot_cls += cls_loss.item()
            tot_rec += float(rec_loss); n_batches += 1

        s_va, y_va, _ = evaluate(model, val_dl, device, use_decoder)
        m_va = metrics(y_va, s_va)
        history.append(
            {
                "epoch": epoch, "loss": tot_loss / n_batches,
                "cls_loss": tot_cls / n_batches, "recon_loss": tot_rec / n_batches,
                "val_auc": m_va["auc"], "val_ap": m_va["ap"],
            }
        )
        star = ""
        if m_va["ap"] > best_ap:
            best_ap, best_epoch = m_va["ap"], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            star = " *"
        print(
            f"  epoch {epoch:3d} loss {tot_loss/n_batches:.4f} "
            f"(cls {tot_cls/n_batches:.4f} rec {tot_rec/n_batches:.4f}) "
            f"val AUC {m_va['auc']:.4f} AP {m_va['ap']:.4f}{star}",
            flush=True,
        )
        if epoch - best_epoch >= args.patience:
            print(f"  early stop at epoch {epoch} (best {best_epoch})", flush=True)
            break

    model.load_state_dict(best_state)
    s_te, y_te, p_te = evaluate(model, test_dl, device, use_decoder)
    s_va, y_va, _ = evaluate(model, val_dl, device, use_decoder)
    m_te = metrics(y_te, s_te)
    print(json.dumps(m_te, indent=2), flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    stem = f"{args.tag}_seed{args.seed}_{args.split_mode}"
    record = {
        "tag": args.tag, "model": args.model, "seed": args.seed,
        "split_seed": split_seed,
        "split_mode": args.split_mode, "views": view_names,
        "scalar_groups": groups, "n_scalars": int(scalars.shape[1]),
        "use_harmonic": (args.model == "phantom") and not args.no_harmonic,
        "use_decoder": use_decoder, "augment": not args.no_augment,
        "n_params": count_parameters(model), "epochs_run": len(history),
        "best_epoch": best_epoch, "train_minutes": (time.time() - t0) / 60,
        "test_metrics": m_te, "val_metrics": metrics(y_va, s_va), "history": history,
        "n_train": len(tr), "n_val": len(va), "n_test": len(te),
    }
    with open(os.path.join(args.out_dir, f"{stem}.json"), "w") as fh:
        json.dump(record, fh, indent=2)

    np.savez_compressed(
        os.path.join(args.out_dir, f"{stem}_preds.npz"),
        test_idx=te, test_score=s_te, test_label=y_te,
        val_idx=va, val_score=s_va, val_label=y_va,
        test_params=p_te if p_te is not None else np.zeros(0),
        kepid=kepid[te], plnt=blob["plnt"][te],
    )
    torch.save(best_state, os.path.join(args.out_dir, f"{stem}.pt"))
    print(f"wrote {args.out_dir}/{stem}.*", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
