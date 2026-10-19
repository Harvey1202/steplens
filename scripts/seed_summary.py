"""Aggregate models trained with several seeds into docs/seeds.md.

Groups prediction sets by (backbone, variant, modality, split, lr) ignoring the seed
tag (`-s1`, `-s2`, …), and reports mean ± std of test AUC and of the balanced-point
recall / false alarm / F1, plus two slice statistics per seed (person 2 miss rate,
kitchen 2 miss rate and AUC) to show whether the slice findings survive re-training.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from steplens.calibrate import best_threshold  # noqa: E402

PRED = ROOT / "data" / "predictions"
steps = pd.read_parquet(ROOT / "data/processed/steps.parquet").set_index("step_key")


def evaluate(model_dir: Path):
    meta = json.load(open(model_dir / "meta.json"))
    if "derived_from" in meta or not (model_dir / "steps.parquet").exists():
        return None
    p = pd.read_parquet(model_dir / "steps.parquet").set_index("step_key")
    d = steps.loc[p.index].copy()
    d["score"], d["phase"] = p["score"], p["phase"]
    val, te = d[d.phase == "val"], d[d.phase == "test"]
    t = best_threshold(val.has_errors.values.astype(bool), val.score.values, "balanced")
    pos = te.score > t
    err, norm = te.has_errors.astype(bool), ~te.has_errors.astype(bool)
    rec = pos[err].mean()
    fa = pos[norm].mean()
    prec = pos[err].sum() / max(1, pos.sum())
    p2 = te[te.person_id == 2]
    k2 = te[te.environment_id == 2]
    pos_p2, pos_k2 = pos.loc[p2.index], pos.loc[k2.index]
    tag = meta["args"].get("tag", "")
    base = re.sub(r"-s\d+$", "", tag)            # drop the seed suffix
    base = re.sub(r"^lr[0-9e.-]+", "", base).strip("-")  # the learning rate is already in the key
    key = f"{meta['backbone']} {meta['variant']} {'+'.join(meta['modality'])} · {meta['split']} · lr {meta['args']['lr']:g}" + (f" · {base}" if base else "")
    return dict(
        key=key, model=meta["model_id"], seed=meta["args"].get("seed", 42), auc=roc_auc_score(te.has_errors, te.score),
        recall=rec, fa=fa, f1=2 * prec * rec / (prec + rec) if prec + rec else 0.0, thr=t,
        p2_miss=1 - pos_p2[p2.has_errors.astype(bool)].mean() if p2.has_errors.sum() else np.nan,
        k2_miss=1 - pos_k2[k2.has_errors.astype(bool)].mean() if k2.has_errors.sum() else np.nan,
        k2_auc=roc_auc_score(k2.has_errors, k2.score) if 5 <= k2.has_errors.sum() < len(k2) else np.nan,
        overall_miss=1 - rec,
    )


def main():
    rows = [r for d in sorted(PRED.iterdir()) if (d / "meta.json").exists() for r in [evaluate(d)] if r]
    df = pd.DataFrame(rows)
    multi = df.groupby("key").filter(lambda g: len(g) >= 2)
    lines = ["# Seed robustness", "",
             "Models trained with more than one seed (same data, split, recipe). Test phase, balanced operating point tuned on each run's validation steps.", "",
             "| configuration | seeds | test AUC | recall | false alarm | F1 | overall miss | person 2 miss | kitchen 2 miss | kitchen 2 AUC |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    pm = lambda s, pct=True: (f"{100*s.mean():.1f} ± {100*s.std(ddof=0):.1f}" if pct else f"{s.mean():.3f} ± {s.std(ddof=0):.3f}")
    for key, g in multi.groupby("key"):
        lines.append(f"| {key} | {len(g)} | {pm(g.auc)} | {pm(g.recall)} | {pm(g.fa)} | {pm(g.f1)} | {pm(g.overall_miss)} | {pm(g.p2_miss)} | {pm(g.k2_miss)} | {pm(g.k2_auc)} |")
    lines += ["", "Per run:", "", "| run | seed | AUC | F1 | overall miss | person 2 miss | kitchen 2 miss | kitchen 2 AUC |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in multi.sort_values(["key", "seed"]).itertuples():
        lines.append(f"| {r.model} | {r.seed} | {100*r.auc:.1f} | {100*r.f1:.1f} | {100*r.overall_miss:.0f}% | {100*r.p2_miss:.0f}% | {100*r.k2_miss:.0f}% | {r.k2_auc:.2f} |")
    out = ROOT / "docs" / "seeds.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
