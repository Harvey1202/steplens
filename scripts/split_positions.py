"""Where in the recipe do the train / "test" / "val" steps come from? (docs/figures/split_positions.png)

For every scored step (Missing Step rows have no video and are left out), its relative position
within its recording (0 = first performed step, 1 = last), by phase, under

* the official step split of the released error_recognition code, which takes each recording's
  steps in annotation order (train = first 75%, "test" = next 16%, "val" = last 9%), and
* our seeded random step split with the same proportions (steplens/data.py, add_step_splits).

    python scripts/split_positions.py   # writes docs/figures/split_positions.png and prints the means
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures" / "split_positions.png"
PHASES = [("train", "#2a78d6"), ("test", "#eb6834"), ("val", "#1baf7a")]
PANELS = [("step_split", "Official step split (released code)"),
          ("step_split_random", "Seeded random step split (ours)")]


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = pd.read_parquet(ROOT / "data" / "processed" / "steps.parquet")
    steps = steps[~steps["is_missing"]]
    bins = np.linspace(0, 1, 21)
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.15), dpi=200, sharey=True)
    for ax, (col, title) in zip(axes, PANELS):
        for phase, color in PHASES:
            pos = steps.loc[steps[col] == phase, "rel_position"]
            ax.hist(pos, bins=bins, density=True, histtype="step", linewidth=2.2, color=color,
                    label=f"{phase} (n={len(pos)}, mean {pos.mean():.2f})")
            print(f"{col:18s} {phase:5s} n={len(pos):5d} mean={pos.mean():.3f} min={pos.min():.3f}")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("relative position of the step in its recording (0 = first, 1 = last)", fontsize=9)
        ax.legend(loc="upper center", frameon=False, fontsize=8.5)
        ax.tick_params(labelsize=8.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("density", fontsize=9)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
