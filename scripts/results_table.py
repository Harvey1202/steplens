"""Summarize every trained model into docs/results.md.

For each prediction set: best epoch, test AUC, the paper-protocol numbers
(min-max normalized step scores at 0.6 for step splits / 0.4 otherwise), and the
numbers at the balanced operating point tuned on the model's validation steps.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from steplens.calibrate import best_threshold  # noqa: E402
from steplens.metrics import binary_metrics, normalize_scores  # noqa: E402

PRED = ROOT / "data" / "predictions"
PAPER = {  # Table 10 of the paper, step split (S) and recording split (R)
    ("omnivore", "mlp", "step"): dict(acc=71.09, f1=24.26, auc=75.7),
    ("omnivore", "transformer", "step"): dict(acc=69.96, f1=55.39, auc=75.7),
    ("imagebind", "transformer", "recordings"): dict(acc=63.87, f1=56.02, auc=65.25),
}


def main():
    rows = []
    for d in sorted(PRED.iterdir()):
        if not (d / "meta.json").exists():
            continue
        meta = json.load(open(d / "meta.json"))
        pred = pd.read_parquet(d / "steps.parquet")
        pred["score_norm"] = pred.groupby("phase")["score"].transform(normalize_scores)
        test, val = pred[pred.phase == "test"], pred[pred.phase == "val"]
        paper_t = 0.6 if meta["split"] in ("step", "step_random") else 0.4
        paper = binary_metrics(test.label.values, test.score_norm.values, paper_t)
        t_bal = best_threshold(val.label.values.astype(bool), val.score.values, "balanced")
        bal = binary_metrics(test.label.values, test.score.values, t_bal)
        ref = PAPER.get((meta["backbone"], meta["variant"], meta["split"]))
        rows.append(dict(
            model=meta["model_id"], split=meta["split"], lr=meta["args"].get("lr"), tag=meta["args"].get("tag", ""),
            epoch=meta["best_epoch"], n_test=len(test),
            auc=100 * paper["auc"], paper_auc=ref["auc"] if ref else np.nan,
            acc_paper=100 * paper["accuracy"], f1_paper=100 * paper["f1"], recall_paper=100 * paper["recall"],
            paper_f1=ref["f1"] if ref else np.nan,
            t_bal=t_bal, recall_bal=100 * bal["recall"], fa_bal=100 * bal["false_alarm"], f1_bal=100 * bal["f1"],
        ))
    df = pd.DataFrame(rows)
    lines = ["| model | split | lr | best ep | test AUC | paper AUC | paper protocol: acc / F1 / recall | paper F1 | balanced thr (val) | balanced: recall / false alarm / F1 |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in df.itertuples():
        lines.append(f"| {r.model} | {r.split} | {r.lr:g} | {r.epoch} | {r.auc:.1f} | {'' if np.isnan(r.paper_auc) else f'{r.paper_auc:.1f}'} | "
                     f"{r.acc_paper:.1f} / {r.f1_paper:.1f} / {r.recall_paper:.1f} | {'' if np.isnan(r.paper_f1) else f'{r.paper_f1:.1f}'} | "
                     f"{r.t_bal:.2f} | {r.recall_bal:.1f} / {r.fa_bal:.1f} / {r.f1_bal:.1f} |")
    out = ROOT / "docs" / "results.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("# Model results (test phase of each model's own split)\n\n"
                   "Paper protocol = min-max normalized step scores, threshold 0.6 (step splits) or 0.4 (other splits), as the error_recognition README states. "
                   "Balanced = threshold maximizing TPR − FPR on the model's validation steps, applied to raw mean per-second scores. "
                   "Paper columns are Table 10 of Peddi et al. (2024) where a matching row exists.\n\n" + "\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
