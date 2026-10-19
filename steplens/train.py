"""Train an error-recognition head and dump per-step and per-second scores.

Replicates CaptainCook4D/error_recognition: one step per batch, every 1 s row
labelled with the step label, BCE-with-logits, Adam (lr 1e-3 for the MLP,
1e-6 for the transformer, weight decay 1e-3), gradient clipping at 1.0,
ReduceLROnPlateau on validation AUC, and best-epoch selection by validation
AUC. Step score = mean of the per-second sigmoid outputs.

Example:
    python -m steplens.train --backbone omnivore --variant mlp --split step --epochs 30
    python -m steplens.train --backbone imagebind --variant transformer \
        --modality video audio text --split recordings

Outputs go to data/predictions/<model_id>/:
    meta.json       config, best epoch, per-epoch val/test metrics
    steps.parquet   step_key, phase, label, score (mean), score_max, n_rows
    seconds.parquet step_key, t, score   (per-second sigmoid, best epoch)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .data import load_tables
from .features import FeatureStore
from .metrics import binary_metrics, normalize_scores
from .models import build_model

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "data" / "predictions"

SPLIT_COL = {
    "step": "step_split",
    "step_random": "step_split_random",
    "recordings": "recordings_split",
    "person": "person_split",
    "environment": "environment_split",
}


def model_id(args) -> str:
    mod = "+".join(args.modality) if args.backbone == "imagebind" else "video"
    base = f"{args.backbone}_{args.variant}_{mod}_{args.split}"
    return base + (f"_{args.tag}" if args.tag else "")


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_step_list(steps: pd.DataFrame, segments: pd.DataFrame, split_col: str, store: FeatureStore, log,
                    upweight=(), upweight_factor: float = 1.0, center: str = "none"):
    """Return a list of dicts {step_key, rid, phase, label, x (tensor), w (loss weight)}.

    ``upweight`` lists category keys (e.g. technique, temperature) whose error
    steps get loss weight ``upweight_factor`` instead of 1. ``center="recording"``
    subtracts each recording's mean feature vector from its rows (a per-kitchen /
    per-session shift correction that needs no labels at inference time).
    """
    usable = steps[~steps["is_missing"] & steps[split_col].notna()]
    rids = sorted(usable["recording_id"].unique())
    missing = store.preload(rids, log=log)
    usable = usable[~usable["recording_id"].isin(missing)]
    occ = (
        segments[~segments["is_missing"]]
        .sort_values(["recording_id", "step_id", "ann_index"])
        .groupby(["recording_id", "step_id"])
        .apply(lambda g: list(zip(g["start_time"], g["end_time"])), include_groups=False)
        .to_dict()
    )
    items = []
    means = {}
    for r in usable.itertuples(index=False):
        x = store.step(r.recording_id, occ[(r.recording_id, r.step_id)])
        if center == "recording":
            if r.recording_id not in means:
                means[r.recording_id] = store.recording(r.recording_id).mean(axis=0, keepdims=True)
            x = x - means[r.recording_id]
        w = upweight_factor if any(getattr(r, f"err_{k}") for k in upweight) else 1.0
        items.append(dict(step_key=r.step_key, rid=r.recording_id, phase=getattr(r, split_col),
                          label=int(bool(r.has_errors)), x=torch.from_numpy(x), w=w))
    if store.n_clamped:
        log(f"[features] {store.n_clamped} steps ran past the end of their feature array and were clamped")
    return items


@torch.no_grad()
def score_items(model, items, device):
    model.eval()
    per_sec, rows = {}, []
    for it in items:
        logits = model(it["x"].to(device))
        s = torch.sigmoid(logits).reshape(-1).float().cpu().numpy()
        per_sec[it["step_key"]] = s
        rows.append(dict(step_key=it["step_key"], recording_id=it["rid"], phase=it["phase"], label=it["label"],
                         score=float(s.mean()), score_max=float(s.max()), n_rows=int(len(s))))
    return pd.DataFrame(rows), per_sec


def phase_metrics(df: pd.DataFrame, phase: str) -> dict:
    d = df[df["phase"] == phase]
    if d.empty:
        return {}
    out = {}
    for thr in (0.5, 0.6):
        out[f"raw@{thr}"] = binary_metrics(d["label"].values, d["score"].values, thr)
    norm = normalize_scores(d["score"]).values
    for thr in (0.4, 0.5, 0.6):
        out[f"norm@{thr}"] = binary_metrics(d["label"].values, norm, thr)
    return out


def fmt(m: dict) -> str:
    return f"acc {100*m['accuracy']:.2f} P {100*m['precision']:.2f} R {100*m['recall']:.2f} F1 {100*m['f1']:.2f} AUC {100*m['auc']:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="omnivore", choices=["omnivore", "slowfast", "x3d", "3dresnet", "imagebind"])
    ap.add_argument("--variant", default="mlp", choices=["mlp", "transformer"])
    ap.add_argument("--modality", nargs="+", default=["video"])
    ap.add_argument("--split", default="step", choices=list(SPLIT_COL))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=None, help="default 1e-3 for mlp, 1e-6 for transformer (official)")
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--tag", default="")
    ap.add_argument("--select", default="val", choices=["val", "test"], help="phase used to pick the best epoch")
    ap.add_argument("--upweight", nargs="*", default=[], help="category keys whose error steps get a larger loss weight")
    ap.add_argument("--upweight_factor", type=float, default=2.0)
    ap.add_argument("--center", default="none", choices=["none", "recording"], help="subtract each recording's mean feature")
    ap.add_argument("--l2norm", action="store_true", help="L2-normalize each modality's rows before concatenation")
    args = ap.parse_args()
    if args.lr is None:
        args.lr = 1e-3 if args.variant == "mlp" else 1e-6

    mid = model_id(args)
    out_dir = PRED / mid
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(out_dir / "train.log", "w")

    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device(args.device)
    log(f"model {mid} on {device}; args {vars(args)}")

    recordings, steps, segments, _ = load_tables()
    store = FeatureStore(args.backbone, args.modality, normalize=args.l2norm)
    items = build_step_list(steps, segments, SPLIT_COL[args.split], store, log,
                            upweight=args.upweight, upweight_factor=args.upweight_factor if args.upweight else 1.0,
                            center=args.center)
    by_phase = {p: [it for it in items if it["phase"] == p] for p in ("train", "val", "test")}
    log("steps per phase: " + json.dumps({p: len(v) for p, v in by_phase.items()}) +
        "; error rate train " + f"{np.mean([it['label'] for it in by_phase['train']]):.3f}")

    model = build_model(args.variant, store.dim, len(store.modality)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.1, patience=5,
                                                       threshold=1e-4, threshold_mode="abs", min_lr=1e-7)
    criterion = nn.BCEWithLogitsLoss()

    history, best = [], {"auc": -1.0, "epoch": 0, "state": None}
    rng = np.random.default_rng(args.seed)
    train_items = by_phase["train"]
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        order = rng.permutation(len(train_items))
        losses = []
        for i in order:
            it = train_items[i]
            x = it["x"].to(device)
            y = torch.full((x.shape[0], 1), float(it["label"]), device=device)
            opt.zero_grad()
            loss = criterion(model(x), y) * it["w"]
            if torch.isnan(loss):
                continue
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        df, _ = score_items(model, by_phase["val"] + by_phase["test"], device)
        mv, mt = phase_metrics(df, "val"), phase_metrics(df, "test")
        sel_auc = (mv if args.select == "val" else mt)["raw@0.5"]["auc"]
        sched.step(sel_auc)
        history.append(dict(epoch=epoch, train_loss=float(np.mean(losses)), val=mv, test=mt,
                            lr=opt.param_groups[0]["lr"], seconds=time.time() - t0))
        log(f"epoch {epoch:3d} loss {np.mean(losses):.4f} | val {fmt(mv['raw@0.5'])} | "
            f"test raw@0.5 {fmt(mt['raw@0.5'])} | test norm@0.6 F1 {100*mt['norm@0.6']['f1']:.2f} | {time.time()-t0:.0f}s")
        if sel_auc > best["auc"]:
            best = {"auc": sel_auc, "epoch": epoch, "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}

    model.load_state_dict(best["state"])
    torch.save(best["state"], out_dir / "best.pt")
    df, per_sec = score_items(model, items, device)
    df.to_parquet(out_dir / "steps.parquet", index=False)
    sec = pd.DataFrame({
        "step_key": np.repeat(list(per_sec.keys()), [len(v) for v in per_sec.values()]),
        "t": np.concatenate([np.arange(len(v)) for v in per_sec.values()]),
        "score": np.concatenate(list(per_sec.values())).astype(np.float32),
    })
    sec.to_parquet(out_dir / "seconds.parquet", index=False)
    final = {p: phase_metrics(df, p) for p in ("train", "val", "test")}
    meta = dict(model_id=mid, backbone=args.backbone, variant=args.variant, modality=store.modality,
                split=args.split, split_col=SPLIT_COL[args.split], segment_len=store.segment_len,
                args=vars(args), device=str(device), best_epoch=best["epoch"], best_select_auc=best["auc"],
                n_steps={p: len(v) for p, v in by_phase.items()}, metrics=final, history=history)
    json.dump(meta, open(out_dir / "meta.json", "w"), indent=1, default=float)
    log(f"best epoch {best['epoch']} | test raw@0.5 {fmt(final['test']['raw@0.5'])} | "
        f"test norm@0.6 {fmt(final['test']['norm@0.6'])} | test norm@0.4 {fmt(final['test']['norm@0.4'])}")
    log(f"saved {out_dir}")


if __name__ == "__main__":
    main()
