"""Slice-aware threshold calibration: a model fix guided by StepLens findings.

The slice table shows that some metadata groups (e.g. one kitchen, one recipe)
are missed far more often than average at a single global threshold. This
script tunes one threshold per group on the model's validation steps and
evaluates the result on its test steps, against the single global threshold.

It writes a derived prediction set ``<model_id>_cal-<by>`` whose scores are
shifted per group (s' = s - t_group + t_global) so that the usual single
threshold t_global on s' reproduces the per-group decision; the tool then shows
the calibrated model like any other.

    python -m steplens.calibrate --model omnivore_mlp_video_step_lr1e-4 --by environment_id
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .data import CAT_KEYS, load_tables
from .metrics import binary_metrics, normalize_scores, per_category

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "data" / "predictions"
GRID = np.round(np.arange(0.01, 1.0, 0.01), 2)


def best_threshold(y: np.ndarray, s: np.ndarray, criterion: str) -> float:
    y = y.astype(bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    best_t, best_v = 0.5, -np.inf
    for t in GRID:
        pos = s > t
        tp = int((pos & y).sum())
        fp = int((pos & ~y).sum())
        fn = n_pos - tp
        if criterion == "f1":
            v = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        else:  # balanced: TPR - FPR
            v = (tp / n_pos if n_pos else 0.0) - (fp / n_neg if n_neg else 0.0)
        if v > best_v + 1e-12:
            best_t, best_v = float(t), v
    return best_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--by", default="environment_id", choices=["environment_id", "activity_id", "person_id", "duration_bucket", "position_bucket"])
    ap.add_argument("--criterion", default="balanced", choices=["balanced", "f1"])
    ap.add_argument("--norm", action="store_true", help="calibrate on min-max normalized step scores")
    ap.add_argument("--min_errors", type=int, default=10, help="groups with fewer val error steps keep the global threshold")
    ap.add_argument("--tune_on", default="val")
    ap.add_argument("--eval_on", default="test")
    args = ap.parse_args()

    src = PRED / args.model
    meta = json.load(open(src / "meta.json"))
    pred = pd.read_parquet(src / "steps.parquet")
    _, steps, _, _ = load_tables()
    df = steps.set_index("step_key").loc[pred["step_key"].values].reset_index()
    df["score"] = pred["score"].values
    df["phase"] = pred["phase"].values
    df["score_norm"] = df.groupby("phase")["score"].transform(normalize_scores)
    col = "score_norm" if args.norm else "score"

    tune = df[df["phase"] == args.tune_on]
    y_t, s_t = tune["has_errors"].values.astype(bool), tune[col].values
    t_global = best_threshold(y_t, s_t, args.criterion)
    groups = {}
    for g, sub in tune.groupby(args.by):
        n_err = int(sub["has_errors"].sum())
        t_g = best_threshold(sub["has_errors"].values.astype(bool), sub[col].values, args.criterion) if n_err >= args.min_errors else t_global
        groups[str(g)] = dict(threshold=t_g, n_val=int(len(sub)), n_val_errors=n_err)

    def thr_of(v):
        return groups.get(str(v), {}).get("threshold", t_global)

    df["t_group"] = df[args.by].map(thr_of).astype(float)
    df["score_cal"] = np.clip(df[col] - df["t_group"] + t_global, 0.0, 1.0)

    def report(phase):
        d = df[df["phase"] == phase]
        y, s, sc = d["has_errors"].values.astype(bool), d[col].values, d["score_cal"].values
        before = binary_metrics(y, s, t_global)
        after = binary_metrics(y, sc, t_global)
        per_group = []
        for g, sub in d.groupby(args.by):
            yy = sub["has_errors"].values.astype(bool)
            if yy.sum() == 0:
                continue
            per_group.append(dict(group=str(g), n_errors=int(yy.sum()), threshold=thr_of(g),
                                  miss_before=float(1 - (sub[col].values > t_global)[yy].mean()),
                                  miss_after=float(1 - (sub["score_cal"].values > t_global)[yy].mean()),
                                  fa_before=float((sub[col].values > t_global)[~yy].mean()) if (~yy).sum() else np.nan,
                                  fa_after=float((sub["score_cal"].values > t_global)[~yy].mean()) if (~yy).sum() else np.nan))
        return dict(before=before, after=after, per_category_before=per_category(d, pd.Series(s > t_global, index=d.index)),
                    per_category_after=per_category(d, pd.Series(sc > t_global, index=d.index)), per_group=per_group)

    rep = dict(model=args.model, by=args.by, criterion=args.criterion, norm=args.norm, t_global=t_global, groups=groups,
               tune_on=args.tune_on, eval_on=args.eval_on, results={ph: report(ph) for ph in (args.tune_on, args.eval_on)})

    # derived prediction set
    out_id = f"{args.model}_cal-{args.by.replace('_id', '')}"
    out = PRED / out_id
    out.mkdir(parents=True, exist_ok=True)
    new_pred = pred.copy()
    new_pred["score"] = df.set_index("step_key").loc[pred["step_key"].values, "score_cal"].values
    new_pred["score_max"] = np.clip(new_pred["score_max"] - df.set_index("step_key").loc[pred["step_key"].values, "t_group"].values + t_global, 0, 1)
    new_pred.to_parquet(out / "steps.parquet", index=False)
    sec = pd.read_parquet(src / "seconds.parquet")
    shift = (t_global - df.set_index("step_key")["t_group"]).to_dict()
    sec["score"] = np.clip(sec["score"].values + sec["step_key"].map(shift).fillna(0).values, 0, 1).astype(np.float32)
    sec.to_parquet(out / "seconds.parquet", index=False)
    shutil.copy(src / "best.pt", out / "best.pt") if (src / "best.pt").exists() else None

    from .train import phase_metrics  # noqa: WPS433  (reuse the trainer's summary format)
    scored = new_pred.rename(columns={})
    new_meta = dict(meta)
    new_meta.update(model_id=out_id, derived_from=args.model, calibration=dict(by=args.by, criterion=args.criterion, norm=args.norm, t_global=t_global, groups=groups),
                    history=[], metrics={ph: phase_metrics(scored, ph) for ph in ("train", "val", "test")})
    new_meta["args"] = dict(meta.get("args", {}), tag=((meta.get("args", {}).get("tag") or "") + f"+cal-{args.by.replace('_id', '')}").lstrip("+"))
    json.dump(new_meta, open(out / "meta.json", "w"), indent=1, default=float)
    json.dump(rep, open(out / "calibration_report.json", "w"), indent=1, default=float)

    ev = rep["results"][args.eval_on]
    print(f"model {args.model} · calibrate by {args.by} ({args.criterion}, {'normalized' if args.norm else 'raw'}) · global threshold {t_global:.2f}")
    print(f"group thresholds: " + ", ".join(f"{g}={v['threshold']:.2f} (n_err={v['n_val_errors']})" for g, v in groups.items()))
    b, a = ev["before"], ev["after"]
    print(f"{args.eval_on}: recall {100*b['recall']:.1f} -> {100*a['recall']:.1f} | false alarm {100*b['false_alarm']:.1f} -> {100*a['false_alarm']:.1f} | "
          f"F1 {100*b['f1']:.1f} -> {100*a['f1']:.1f} | acc {100*b['accuracy']:.1f} -> {100*a['accuracy']:.1f} | AUC {100*b['auc']:.1f} -> {100*a['auc']:.1f}")
    for g in ev["per_group"]:
        print(f"  {args.by}={g['group']:>4} n_err {g['n_errors']:3d} thr {g['threshold']:.2f} miss {100*g['miss_before']:5.1f} -> {100*g['miss_after']:5.1f}  FA {100*g['fa_before']:5.1f} -> {100*g['fa_after']:5.1f}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
