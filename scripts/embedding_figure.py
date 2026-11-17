"""UMAP of step features colored by kitchen and by one model's outcome (docs/figures/umap_kitchen_outcome.png).

Reads data/processed/embedding_<name>.parquet (python -m steplens.embedding) and one model's
predictions; the model is at its balanced threshold chosen on its own validation steps, as in
the tool, and its outcome is shown for its test steps (all other steps in light grey).

    python scripts/embedding_figure.py                       # omnivore features, tuned Omnivore MLP
    python scripts/embedding_figure.py --name omnivore-center --model omnivore_transformer_video_step_lr1e-5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from steplens.calibrate import best_threshold  # noqa: E402

KITCHEN_COLORS = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948", "#b07aa1", "#ff9da7", "#9c755f",
                  "#bab0ac"]  # Tableau 10, the tool's kitchen palette
OUTCOMES = [("TN", "normal, not flagged", "#c9ced6", 4), ("TP", "error caught", "#0ca30c", 7),
            ("FP", "false alarm", "#ec835a", 7), ("FN", "error missed", "#d03b3b", 11)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="omnivore")
    ap.add_argument("--model", default="omnivore_mlp_video_step_lr1e-4")
    ap.add_argument("--label", default="the tuned Omnivore MLP", help="model name for the panel title")
    ap.add_argument("--phase", default="test")
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "figures" / "umap_kitchen_outcome.png")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    emb = pd.read_parquet(ROOT / "data" / "processed" / f"embedding_{a.name}.parquet").set_index("step_key")
    steps = pd.read_parquet(ROOT / "data" / "processed" / "steps.parquet").set_index("step_key")
    d = emb.join(steps[["environment_id", "has_errors"]])
    pred = pd.read_parquet(ROOT / "data" / "predictions" / a.model / "steps.parquet").set_index("step_key")
    val = pred[pred["phase"] == "val"]
    t = best_threshold(steps.loc[val.index, "has_errors"].values.astype(bool), val["score"].values, "balanced")
    d = d.join(pred[["phase", "score"]])
    in_phase = d["phase"] == a.phase
    pos = d["score"] > t
    d["outcome"] = None
    d.loc[in_phase & d["has_errors"] & pos, "outcome"] = "TP"
    d.loc[in_phase & d["has_errors"] & ~pos, "outcome"] = "FN"
    d.loc[in_phase & ~d["has_errors"] & pos, "outcome"] = "FP"
    d.loc[in_phase & ~d["has_errors"] & ~pos, "outcome"] = "TN"

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.4), dpi=200)
    ax = axes[0]
    kitchens = sorted(d["environment_id"].unique())
    for i, k in enumerate(kitchens):
        m = d["environment_id"] == k
        ax.scatter(d.loc[m, "x"], d.loc[m, "y"], s=3, color=KITCHEN_COLORS[i % 10], label=f"kitchen {k}", linewidths=0)
    ax.set_title(f"All {len(d)} scored steps, colored by kitchen", fontsize=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.0), fontsize=7, markerscale=3, frameon=False, ncol=5,
              handletextpad=0.2, columnspacing=0.9)
    ax = axes[1]
    other = d["outcome"].isna()
    ax.scatter(d.loc[other, "x"], d.loc[other, "y"], s=2, color="#e6e8ec", linewidths=0, label="other phases")
    for key, label, color, size in OUTCOMES:
        m = d["outcome"] == key
        ax.scatter(d.loc[m, "x"], d.loc[m, "y"], s=size, color=color, linewidths=0, label=f"{label} ({int(m.sum())})")
    ax.set_title(f"{a.phase.capitalize()} steps of {a.label}, colored by outcome", fontsize=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.0), fontsize=7, markerscale=1.5, frameon=False, ncol=3,
              handletextpad=0.2, columnspacing=0.9)
    for ax in axes:
        ax.set_xticks([]), ax.set_yticks([])
        for side in ax.spines.values():
            side.set_visible(False)
    fig.tight_layout()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, bbox_inches="tight")
    print(f"threshold {t:.2f}; outcomes: {d['outcome'].value_counts().to_dict()}; wrote {a.out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
