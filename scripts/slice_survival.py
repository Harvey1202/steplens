"""How often does a slice flagged on one phase survive on another? (proposal feedback, point 3)

For each model: pick its balanced threshold on the validation steps, enumerate slices
on the validation steps (single attributes and pairs, optionally crossed with an error
category, at least 10 error steps), flag those whose miss rate is significantly above
the validation miss rate (z >= 2), then re-evaluate each flagged slice on the test
steps. A flagged slice

* *holds* if its test miss rate is above the test miss rate overall,
* *holds strongly* if it is also significant on test (z >= 2 with >= 5 test errors),
* *reverses* if its test miss rate is at or below the overall test miss rate,
* is *untestable* if it has fewer than 5 error steps on test.

We also check stability across seeds: of the slices flagged for seed A, how many are
flagged (z >= 2) for seed B of the same configuration on the same phase.

    python scripts/slice_survival.py   # writes docs/slice_survival.md and docs/figures/slice_survival.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from steplens.calibrate import best_threshold  # noqa: E402
from steplens.metrics import confirm_slices, slice_table  # noqa: E402

PRED = ROOT / "data" / "predictions"
steps = pd.read_parquet(ROOT / "data/processed/steps.parquet").set_index("step_key")

MODELS = {
    "Omnivore MLP, random step split (seed 42)": "omnivore_mlp_video_step_random_lr1e-4",
    "Omnivore MLP, random step split (seed 1)": "omnivore_mlp_video_step_random_lr1e-4-s1",
    "Omnivore MLP, random step split (seed 2)": "omnivore_mlp_video_step_random_lr1e-4-s2",
    "Omnivore MLP, official step split (seed 42)": "omnivore_mlp_video_step_lr1e-4",
    "Omnivore transformer, official step split": "omnivore_transformer_video_step_lr1e-5",
    "Omnivore transformer, recording split": "omnivore_transformer_video_recordings",
    "Authors' Omnivore transformer, step split": "paper_omnivore_transformer_video_step",
    "Authors' Omnivore MLP, recording split": "paper_omnivore_mlp_video_recordings",
}
Z = 2.0


def scored(model_id):
    p = pd.read_parquet(PRED / model_id / "steps.parquet").set_index("step_key")
    d = steps.loc[p.index].copy()
    d["score"], d["phase"] = p["score"], p["phase"]
    val = d[d.phase == "val"]
    t = best_threshold(val.has_errors.values.astype(bool), val.score.values, "balanced")
    d["pos"] = d.score > t
    return d


def flagged_on(d, phase):
    sub = d[d.phase == phase]
    tab = slice_table(sub, sub["pos"], min_size=10, min_errors=10, max_order=2, include_categories=True)
    return sub, tab[tab["z_miss"] >= Z].reset_index(drop=True)


def survival(model_id):
    d = scored(model_id)
    _, flagged = flagged_on(d, "val")
    te = d[d.phase == "test"]
    conf, overall_test = confirm_slices(te, te["pos"], flagged)
    n_err_t = conf["confirm_n_error"]
    k = (conf["confirm_miss_rate"] * n_err_t).round()
    p0 = overall_test
    z_test = (conf["confirm_miss_rate"] - p0) / np.sqrt(p0 * (1 - p0) / n_err_t.replace(0, np.nan))
    testable = n_err_t >= 5
    holds = testable & (conf["confirm_miss_lift"] > 0)
    strong = holds & (z_test >= Z)
    reverses = testable & ~holds
    conf["z_test"], conf["status"] = z_test, np.select([strong, holds, reverses], ["holds (significant)", "holds", "reverses"], "untestable")
    return dict(flagged=len(conf), testable=int(testable.sum()), holds=int(holds.sum()), strong=int(strong.sum()),
                reverses=int(reverses.sum()), untestable=int((~testable).sum())), conf


def seed_overlap(a, b, phase="val"):
    _, fa = flagged_on(scored(a), phase)
    _, fb = flagged_on(scored(b), phase)
    sa, sb = set(fa["slice"]), set(fb["slice"])
    return len(sa), len(sb), len(sa & sb)


def main():
    rows, confs = [], {}
    for label, mid in MODELS.items():
        if not (PRED / mid / "steps.parquet").exists():
            continue
        r, conf = survival(mid)
        rows.append(dict(model=label, **r))
        confs[label] = conf
    df = pd.DataFrame(rows)
    pct = lambda a, b: f"{100*a/b:.0f}%" if b else "–"
    lines = ["# Slice survival: flagged on validation, re-checked on test", "",
             "Each model at its balanced threshold (chosen on validation). Slices: single attributes and pairs over recipe, person, kitchen, step-duration bucket and recipe-position bucket, optionally crossed with one error category; at least 10 validation error steps; flagged when the validation miss rate is above the overall validation miss rate with z ≥ 2. Re-checked on the test steps: *holds* = test miss rate above the overall test miss rate; *significant* = also z ≥ 2 on test; *reverses* = at or below the overall test rate; *untestable* = fewer than 5 test error steps.", "",
             "| Model | Flagged on val | Testable | Holds on test | of which significant | Reverses | Untestable |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in df.itertuples():
        lines.append(f"| {r.model} | {r.flagged} | {r.testable} | {r.holds} ({pct(r.holds, r.testable)}) | {r.strong} ({pct(r.strong, r.testable)}) | {r.reverses} ({pct(r.reverses, r.testable)}) | {r.untestable} |")
    tot = df[["flagged", "testable", "holds", "strong", "reverses", "untestable"]].sum()
    lines.append(f"| **All models** | {tot.flagged} | {tot.testable} | {tot.holds} ({pct(tot.holds, tot.testable)}) | {tot.strong} ({pct(tot.strong, tot.testable)}) | {tot.reverses} ({pct(tot.reverses, tot.testable)}) | {tot.untestable} |")

    lines += ["", "## Across seeds (same configuration, same phase)", "",
              "| Pair (random step split, validation phase) | Flagged in A | Flagged in B | Flagged in both |", "| --- | --- | --- | --- |"]
    seeds = ["omnivore_mlp_video_step_random_lr1e-4", "omnivore_mlp_video_step_random_lr1e-4-s1", "omnivore_mlp_video_step_random_lr1e-4-s2"]
    for i in range(3):
        for j in range(i + 1, 3):
            na, nb, nab = seed_overlap(seeds[i], seeds[j])
            lines.append(f"| seed {['42','1','2'][i]} vs seed {['42','1','2'][j]} | {na} | {nb} | {nab} |")

    lines += ["", "## Examples (random step split, seed 42)", "",
              "| Slice flagged on validation | val miss (n) | val z | test miss (n) | test z | status |", "| --- | --- | --- | --- | --- | --- |"]
    ex = confs.get("Omnivore MLP, random step split (seed 42)")
    if ex is not None:
        for r in ex.sort_values("z_miss", ascending=False).itertuples():
            tm = f"{100*r.confirm_miss_rate:.0f}% ({r.confirm_n_error})" if r.confirm_n_error else "–"
            zt = f"{r.z_test:.1f}" if np.isfinite(r.z_test) else "–"
            lines.append(f"| {r.slice} | {100*r.miss_rate:.0f}% ({r.n_error}) | {r.z_miss:.1f} | {tm} | {zt} | {r.status} |")
    out = ROOT / "docs" / "slice_survival.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:20]))

    # figure: validation lift vs test lift for every flagged slice, all models
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    allc = pd.concat([c.assign(model=m) for m, c in confs.items()])
    allc = allc[allc["confirm_n_error"] >= 5]
    fig, ax = plt.subplots(figsize=(5.2, 4.0), dpi=200)
    colors = {"holds (significant)": "#2a78d6", "holds": "#86b6ef", "reverses": "#eb6834"}
    for st, g in allc.groupby("status"):
        ax.scatter(100 * g["miss_lift"], 100 * g["confirm_miss_lift"], s=np.clip(g["confirm_n_error"], 8, 60), c=colors.get(st, "#898781"),
                   edgecolor="white", lw=0.6, label=f"{st} ({len(g)})", alpha=0.9)
    ax.axhline(0, color="#898781", lw=0.8)
    ax.set_xlabel("miss-rate lift on validation (points above overall)", fontsize=8)
    ax.set_ylabel("miss-rate lift on test (points above overall)", fontsize=8)
    ax.set_title("Slices flagged on validation, re-checked on test (8 models)", fontsize=9)
    ax.tick_params(labelsize=7.5)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(ROOT / "docs" / "figures" / "slice_survival.png", bbox_inches="tight")
    print(f"\nwrote {out} and docs/figures/slice_survival.png")


if __name__ == "__main__":
    main()
