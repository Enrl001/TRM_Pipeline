"""STEP 3 (model 2): graph attention network (GAT) on the per-window connectivity graphs.

Reads the same .npz files as the baseline, applies the same graph filtering
(--density), and uses the SAME subject-wise outer folds and metrics
(02_baseline_modares.py: outer_splits, scores, subject_level), so the two models
are compared on identical splits.

Per window: nodes = channels, node features = log band power (z-scored on the
training subjects), edges = filtered connectivity with one weight per band.
Inside each outer fold, a share of the training subjects (--val-frac) is held out
for early stopping; test subjects are only used to score the final model.

Usage
-----
python 03_gat.py --features features/mydata --conn wpli
python 03_gat.py --features features/mydata --conn pdc \
       --manifest data/mydata/manifest.csv --label-col label_3class

Main options
  --density 0.9          keep the strongest 90 % of edges (same as the baseline)
  --node-features bp     'bp' = band power; 'bp+conn' adds each node's edge weights
  --hidden 32 --heads 4 --layers 2 --dropout 0.3
  --epochs 100 --patience 15 --batch 256 --lr 1e-3 --weight-decay 1e-4
  --ensemble 1           models per fold with different seeds, logits averaged
  --n-perm 0             permutation test (subject labels shuffled, whole CV rerun)
  --loso / --paper-split as in the baseline
"""
import argparse
import importlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit

from paineeg.data import load_subjects
from paineeg import graph_metrics as gm
from paineeg import gat as G

base = importlib.import_module("02_baseline_modares")      # shared splits and metrics


# ------------------------------------------------------------------ data
def load_dataset(args):
    """Dense graphs, node features, labels and subject ids for every window."""
    subjects = load_subjects(args.features, conn=args.conn)
    if args.manifest:
        man = pd.read_csv(args.manifest).astype({"subject": str}).set_index("subject")
        for s in subjects:
            s["label"] = int(man.loc[s["subject"], args.label_col])
    elif args.label_col != "label":
        raise SystemExit(f"--label-col {args.label_col} needs --manifest")
    _, lo, hi = gm.version_for_density(args.density)
    A, X, groups = [], [], []
    for s in subjects:
        g = G.filtered_graphs(s["conn"], s["directed"], lo, hi)
        A.append(g)
        X.append(G.node_features(s["bandpower"], g, args.node_features))
        groups += [s["subject"]] * len(g)
    subj = pd.DataFrame(dict(subject=[s["subject"] for s in subjects],
                             label=[s["label"] for s in subjects]))
    return (np.concatenate(X), np.concatenate(A).astype(np.float32), np.array(groups), subj,
            subjects[0]["ch_names"], subjects[0]["bands"])


# ------------------------------------------------------------------ training
def batches(idx, size, rng=None):
    if rng is not None:
        idx = rng.permutation(idx)
    for k in range(0, len(idx), size):
        yield idx[k:k + size]


def run_model(model, X, A, idx, args, return_attention=False):
    """Logits (and last-layer attention) for windows idx, in eval mode."""
    model.eval()
    out, att = [], []
    with torch.no_grad():
        for b in batches(idx, args.batch * 4):
            x, ei, ea, bt = G.to_batch(X[b], A[b])
            r = model(x, ei, ea, bt, return_attention=return_attention)
            if return_attention:
                r, a = r
                att.append((b, a[0].cpu().numpy(), a[1].cpu().numpy()))
            out.append(r.float().cpu())
    logits = torch.cat(out).numpy()
    return (logits, att) if return_attention else logits


def train_one(X, A, yi, tr, va, n_classes, args, seed, device):
    """Train on windows tr, early-stop on validation loss over windows va."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = G.GAT(X.shape[2], A.shape[1], n_classes, args.hidden, args.heads,
                  args.layers, args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    counts = np.bincount(yi[tr], minlength=n_classes)
    w = torch.tensor(len(tr) / (n_classes * np.maximum(counts, 1)), dtype=torch.float32, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=w)           # balanced, as class_weight in the SVM
    y_dev = torch.as_tensor(yi, device=device)
    best, best_loss, wait, log = None, np.inf, 0, []
    for ep in range(args.epochs):
        model.train()
        tot = 0.0
        for b in batches(tr, args.batch, rng):
            x, ei, ea, bt = G.to_batch(X[b], A[b])
            loss = loss_fn(model(x, ei, ea, bt), y_dev[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        lv = run_model(model, X, A, va, args)
        v_loss = float(loss_fn(torch.as_tensor(lv, device=device), y_dev[va]))
        v_bacc = balanced_accuracy_score(yi[va], lv.argmax(1))
        log.append(dict(epoch=ep, train_loss=tot / len(tr), val_loss=v_loss, val_window_bacc=v_bacc))
        if v_loss < best_loss - 1e-4:
            best_loss, wait = v_loss, 0
            best = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= args.patience:
                break
    model.load_state_dict(best)
    return model, pd.DataFrame(log)


def evaluate(X_np, A_np, groups, subj, args, device, seed, keep_details=True, verbose=True):
    """Whole subject-wise CV (same outer folds as the baseline)."""
    classes = [int(c) for c in sorted(subj["label"].unique())]
    lab = dict(zip(subj["subject"], subj["label"]))
    y = np.array([lab[g] for g in groups])
    yi = np.searchsorted(classes, y)
    binary = len(classes) == 2
    C = X_np.shape[1]
    A = torch.as_tensor(A_np, device=device)
    folds, oof, subj_rows, logs = [], [], [], []
    att_sum = np.zeros((len(classes), C, C))
    att_n = np.zeros((len(classes), C, C))
    for k, (tr_s, te_s) in enumerate(base.outer_splits(subj, args, seed)):
        t0 = time.time()
        # validation subjects for early stopping, taken from the training subjects
        tr_lab = subj.set_index("subject").loc[tr_s, "label"].to_numpy()
        n_val = max(len(classes), int(round(args.val_frac * len(tr_s))))
        ss = StratifiedShuffleSplit(n_splits=1, test_size=n_val, random_state=seed + k)
        fit_i, val_i = next(ss.split(tr_s, tr_lab))
        tr = np.where(np.isin(groups, tr_s[fit_i]))[0]
        va = np.where(np.isin(groups, tr_s[val_i]))[0]
        te = np.where(np.isin(groups, te_s))[0]
        # z-score node features with the fitting windows only
        mu, sd = X_np[tr].mean(axis=0), X_np[tr].std(axis=0) + 1e-6
        X = torch.as_tensor((X_np - mu) / sd, device=device)
        logits, epochs = 0, []
        for e in range(args.ensemble):
            model, log = train_one(X, A, yi, tr, va, len(classes), args, seed + 100 * e + k, device)
            lt, att = run_model(model, X, A, te, args, return_attention=True)
            logits = logits + lt / args.ensemble
            epochs.append(len(log))
            logs.append(log.assign(fold=k, member=e))
            for b, ei, alpha in att:                          # last-layer attention per class
                win = b[ei[0] // C]
                ci = yi[win]
                np.add.at(att_sum, (ci, ei[0] % C, ei[1] % C), alpha)
                np.add.at(att_n, (ci, ei[0] % C, ei[1] % C), 1)
        pred = np.array(classes)[logits.argmax(1)]
        score = logits[:, -1] - logits[:, 0] if binary else None
        sdf = base.subject_level(groups[te], y[te], pred, score, classes, args.subject_rule)
        sdf["fold"] = k
        w = base.scores(y[te], pred, score, classes)
        s = base.scores(sdf["true"].to_numpy(), sdf["pred"].to_numpy(),
                        sdf["mean_score"].to_numpy() if binary else None, classes)
        folds.append(dict(fold=k, n_test_subjects=len(te_s),
                          **{f"window_{m}": v for m, v in w.items()},
                          **{f"subject_{m}": v for m, v in s.items()},
                          epochs=";".join(map(str, epochs))))
        oof.append(pd.DataFrame(dict(group=groups[te], y=y[te], pred=pred,
                                     score=score if binary else np.nan)))
        subj_rows.append(sdf)
        if verbose:
            print(f"  fold {k}: subject bAcc={s['balanced_accuracy']:.3f} window bAcc="
                  f"{w['balanced_accuracy']:.3f} epochs={epochs} ({time.time() - t0:.0f} s)", flush=True)
    oof = pd.concat(oof)
    sdf = pd.concat(subj_rows, ignore_index=True)
    res = dict(classes=classes, order=classes,
               pooled_window=base.scores(oof["y"].to_numpy(), oof["pred"].to_numpy(),
                                         oof["score"].to_numpy() if binary else None, classes),
               pooled_subject=base.scores(sdf["true"].to_numpy(), sdf["pred"].to_numpy(),
                                          sdf["mean_score"].to_numpy() if binary else None, classes),
               per_fold=pd.DataFrame(folds))
    if keep_details:
        with np.errstate(invalid="ignore"):
            res.update(oof=oof, subjects=sdf, train_log=pd.concat(logs, ignore_index=True),
                       attention=att_sum / att_n)
    return res


# ------------------------------------------------------------------ outputs
def attention_table(att, classes, ch_names):
    """Mean last-layer attention per edge (source -> target) and class, strongest first."""
    rows = []
    for c, M in zip(classes, att):
        for i, src in enumerate(ch_names):
            for j, dst in enumerate(ch_names):
                if np.isfinite(M[i, j]):
                    rows.append(dict(true_class=c, source=src, target=dst, attention=M[i, j]))
    return pd.DataFrame(rows).sort_values(["true_class", "attention"], ascending=[True, False])


def write_outputs(res, args, out, ch_names, perm, extra):
    out.mkdir(parents=True, exist_ok=True)
    res["per_fold"].to_csv(out / "per_fold.csv", index=False)
    res["subjects"].to_csv(out / "subject_predictions.csv", index=False)
    res["train_log"].to_csv(out / "training_log.csv", index=False)
    attention_table(res["attention"], res["classes"], ch_names).to_csv(out / "attention_edges.csv", index=False)
    base.plot_confusion(res, out / "confusion_matrix.png")
    num = res["per_fold"].drop(columns=["fold", "n_test_subjects", "epochs"])
    metrics = dict(model="GAT", conn=args.conn, label_col=args.label_col, classes=res["classes"],
                   graph_version=gm.version_for_density(args.density)[0],
                   validation="loso" if args.loso else "paper-split" if args.paper_split
                   else f"stratified group {len(num)}-fold",
                   pooled_window=res["pooled_window"], pooled_subject=res["pooled_subject"],
                   per_fold_mean_sd={c: dict(mean=float(num[c].mean()),
                                             sd=float(num[c].std(ddof=1)) if len(num) > 1 else 0.0)
                                     for c in num.columns})
    if perm is not None:
        metrics["permutation"] = perm
    metrics.update(extra)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "run_config.json").write_text(json.dumps(vars(args), indent=2, default=str))


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", required=True, help="folder of .npz files from 01_build_dataset.py")
    ap.add_argument("--conn", choices=["pdc", "wpli"], default="wpli")
    ap.add_argument("--manifest", default=None, help="CSV with a 'subject' column and the label column")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--density", type=float, default=0.9)
    ap.add_argument("--node-features", choices=["bp", "bp+conn"], default="bp")
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--val-frac", type=float, default=0.2, help="share of training subjects for early stopping")
    ap.add_argument("--ensemble", type=int, default=1)
    ap.add_argument("--loso", action="store_true")
    ap.add_argument("--paper-split", action="store_true")
    ap.add_argument("--test-size", type=float, default=0.25)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--n-perm", type=int, default=0)
    ap.add_argument("--subject-rule", choices=["vote", "mean"], default="vote")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None, help="cuda / cpu / mps (default: cuda if available)")
    ap.add_argument("--out", default=None, help="default results/gat_<conn>/")
    args = ap.parse_args()
    if args.loso and args.paper_split:
        raise SystemExit("choose --loso or --paper-split, not both")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out = Path(args.out or f"results/gat_{args.conn}")
    t0 = time.time()

    X, A, groups, subj, ch_names, bands = load_dataset(args)
    print(f"{len(subj)} subjects, {len(X)} windows, {X.shape[1]} nodes x {X.shape[2]} features, "
          f"edge features {bands}, classes {sorted(subj['label'].unique().tolist())}, device {device}")
    res = evaluate(X, A, groups, subj, args, device, args.seed)
    print(f"[window ] {base.fmt(res['pooled_window'])}")
    print(f"[subject] {base.fmt(res['pooled_subject'])}")

    perm = None
    if args.n_perm > 0:
        null = []
        for i in range(args.n_perm):
            p = subj.copy()
            p["label"] = np.random.default_rng(10_000 + i).permutation(p["label"].to_numpy())
            r = evaluate(X, A, groups, p, args, device, args.seed, keep_details=False, verbose=False)
            null.append([r["pooled_subject"]["balanced_accuracy"], r["pooled_window"]["balanced_accuracy"],
                         r["pooled_subject"].get("auc", np.nan)])
            print(f"  perm {i + 1}/{args.n_perm}: subject bAcc={null[-1][0]:.3f}", flush=True)
        null = np.array(null)
        perm = dict(n_perm=args.n_perm, labels_shuffled_at="subject level")
        stats = [("subject_balanced_accuracy", res["pooled_subject"]["balanced_accuracy"]),
                 ("window_balanced_accuracy", res["pooled_window"]["balanced_accuracy"])]
        if "auc" in res["pooled_subject"]:
            stats.append(("subject_auc", res["pooled_subject"]["auc"]))
        for i, (name, obs) in enumerate(stats):
            perm[name] = dict(observed=obs, null_mean=float(np.nanmean(null[:, i])),
                              p=float((1 + (null[:, i] >= obs).sum()) / (args.n_perm + 1)))
            print(f"[perm   ] {name}: observed {obs:.3f}, null mean {perm[name]['null_mean']:.3f}, "
                  f"p={perm[name]['p']:.3f}")

    extra = dict(runtime_s=round(time.time() - t0, 1), device=str(device))
    write_outputs(res, args, out, ch_names, perm, extra)
    print(f"Saved results to {out}/ ({extra['runtime_s']} s)")


if __name__ == "__main__":
    main()
