"""Build the user-study baseline notebook (pandas + metric tables, no plots) and the
CSV files it reads, so it runs anywhere with pandas only.

    python scripts/build_baseline_notebook.py
    jupyter nbconvert --to notebook --execute --inplace docs/user_study/baseline_notebook.ipynb
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "docs" / "user_study"
DATA = OUT / "data"
MODELS = ["omnivore_mlp_video_step_lr1e-4", "slowfast_mlp_video_step_lr1e-4", "omnivore_transformer_video_step_lr1e-5"]
KEEP = ["step_key", "recording_id", "activity_id", "activity_name", "step_id", "step_description", "person_id", "environment_id",
        "start_time", "end_time", "duration", "position_idx", "n_performed", "rel_position", "duration_bucket", "position_bucket",
        "has_errors", "is_missing", "error_tags", "error_list", "err_preparation", "err_measurement", "err_order", "err_timing",
        "err_technique", "err_temperature", "err_missing", "err_other"]


def export_csv():
    DATA.mkdir(parents=True, exist_ok=True)
    steps = pd.read_parquet(ROOT / "data/processed/steps.parquet")[KEEP].copy()
    for c in ("error_tags", "error_list"):
        steps[c] = steps[c].apply(lambda v: " | ".join(list(v)))
    steps.to_csv(DATA / "steps.csv", index=False)
    for m in MODELS:
        p = pd.read_parquet(ROOT / "data/predictions" / m / "steps.parquet")[["step_key", "phase", "score"]]
        p.to_csv(DATA / f"pred_{m}.csv", index=False)


def build_notebook():
    nb = nbf.v4.new_notebook()
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    cells = [
        md("""# Baseline: diagnosing an error-recognition model with metric tables

This notebook gives you the same facts as StepLens, as pandas tables. Each step of every
recording is one row; the model gives each step a score; a step is flagged as an error
when its score is above the model's threshold (chosen on the validation steps).

**Your task (15 minutes):** find the model's main failure modes — groups of steps
(error category, recipe, person, kitchen, step duration, position in the recipe, or a
combination) where it misses more errors than overall, or raises more false alarms.
Write each finding on the task sheet with its numbers. Run the cells top to bottom,
then edit the slicing cells near the end as you like."""),
        code("""import pandas as pd
pd.set_option("display.max_rows", 200, "display.width", 160)

MODEL = "omnivore_mlp_video_step_lr1e-4"          # the model you are diagnosing
OTHER = "slowfast_mlp_video_step_lr1e-4"          # a second model for comparison
PHASE = "test"                                     # evaluate on these steps
HOLDOUT = "val"                                    # confirm findings on these steps

CATS = ["preparation", "measurement", "order", "timing", "technique", "temperature"]

steps = pd.read_csv("data/steps.csv")
def load(model):
    pred = pd.read_csv(f"data/pred_{model}.csv")
    d = steps.merge(pred, on="step_key")
    val = d[d.phase == "val"]
    # balanced threshold: maximizes TPR - FPR on the validation steps
    best, best_t = -1, 0.5
    for t in [i / 100 for i in range(1, 100)]:
        pos = val.score > t
        tpr = pos[val.has_errors].mean(); fpr = pos[~val.has_errors].mean()
        if tpr - fpr > best: best, best_t = tpr - fpr, t
    d["flagged"] = d.score > best_t
    d["missed"] = d.has_errors & ~d.flagged
    d["false_alarm"] = ~d.has_errors & d.flagged
    return d, best_t

df, THRESHOLD = load(MODEL)
other, OTHER_THRESHOLD = load(OTHER)
test = df[df.phase == PHASE]
hold = df[df.phase == HOLDOUT]
print(f"{MODEL}: threshold {THRESHOLD:.2f}; {len(test)} {PHASE} steps, {int(test.has_errors.sum())} with errors")"""),
        md("## 1. Overall performance on the test steps"),
        code("""def summary(d):
    err, norm = d[d.has_errors], d[~d.has_errors]
    return pd.Series({"steps": len(d), "error steps": len(err), "recall": err.flagged.mean(),
                      "miss rate": 1 - err.flagged.mean(), "false alarm rate": norm.flagged.mean(),
                      "precision": err.flagged.sum() / max(1, d.flagged.sum()), "accuracy": (d.flagged == d.has_errors).mean()})
pd.DataFrame({PHASE: summary(test), HOLDOUT: summary(hold)}).round(3)"""),
        md("## 2. By error category (a step can carry several categories; missing steps are never scored)"),
        code("""rows = []
for c in CATS:
    m = test[test[f"err_{c}"]]
    h = hold[hold[f"err_{c}"]]
    rows.append({"category": c, "error steps": len(m), "missed": int(m.missed.sum()), "miss rate": m.missed.mean(),
                 f"miss rate ({HOLDOUT})": h.missed.mean(), f"n ({HOLDOUT})": len(h)})
pd.DataFrame(rows).round(3)"""),
        md("## 3. By recipe, person, kitchen, step duration and position\n\nCompare each row with the overall miss rate from section 1. `holdout` columns show the same group on the hold-out steps."),
        code("""OVERALL_MISS = test[test.has_errors].missed.mean()
OVERALL_FA = test[~test.has_errors].false_alarm.mean()

def by(cols, min_errors=10, d=test, h=hold):
    err = d[d.has_errors].groupby(cols).agg(error_steps=("missed", "size"), missed=("missed", "sum"), miss_rate=("missed", "mean"))
    fa = d[~d.has_errors].groupby(cols).agg(normal_steps=("false_alarm", "size"), false_alarm_rate=("false_alarm", "mean"))
    herr = h[h.has_errors].groupby(cols).agg(holdout_error_steps=("missed", "size"), holdout_miss_rate=("missed", "mean"))
    out = err.join(fa, how="left").join(herr, how="left")
    out["miss_vs_overall"] = out.miss_rate - OVERALL_MISS
    out["fa_vs_overall"] = out.false_alarm_rate - OVERALL_FA
    return out[out.error_steps >= min_errors].sort_values("miss_rate", ascending=False).round(3)

print(f"overall miss rate {OVERALL_MISS:.3f}, overall false-alarm rate {OVERALL_FA:.3f}")
by("activity_name")"""),
        code("""by("person_id")"""),
        code("""by("environment_id")"""),
        code("""by("duration_bucket")"""),
        code("""by("position_bucket")"""),
        md("## 4. Two-way slices\n\nEdit the column list. Combinations with fewer than `min_errors` error steps are hidden."),
        code("""by(["environment_id", "position_bucket"], min_errors=10).head(15)"""),
        code("""by(["person_id", "environment_id"], min_errors=10).head(15)"""),
        md("### A category inside a group\n\nRestrict to one category first, then group."),
        code("""CATEGORY = "order"
sub = test[~test.has_errors | test[f"err_{CATEGORY}"]]
subh = hold[~hold.has_errors | hold[f"err_{CATEGORY}"]]
by("environment_id", min_errors=8, d=sub, h=subh)"""),
        md("## 5. Does the other model miss the same errors?"),
        code("""o = other[other.phase == PHASE].set_index("step_key")
t = test.set_index("step_key")
common = t.index.intersection(o.index)
a, b, y = t.loc[common, "flagged"], o.loc[common, "flagged"], t.loc[common, "has_errors"]
rows = []
for c in CATS + ["normal"]:
    m = (~y) if c == "normal" else t.loc[common, f"err_{c}"].astype(bool)
    rows.append({"category": c, "n": int(m.sum()), "both flag": int((a & b & m).sum()), f"{MODEL} only": int((a & ~b & m).sum()),
                 f"{OTHER} only": int((~a & b & m).sum()), "neither": int((~a & ~b & m).sum())})
pd.DataFrame(rows)"""),
        md("## 6. Look at individual steps\n\nFor example, the missed error steps of one recording, with the error descriptions."),
        code("""rid = test[test.missed].recording_id.value_counts().index[0]
cols = ["recording_id", "activity_name", "person_id", "environment_id", "start_time", "end_time", "duration", "position_idx", "error_list", "score", "flagged"]
df[(df.recording_id == rid)].sort_values("start_time")[cols]"""),
        md("## Scratch"),
        code("""# your own queries here\n"""),
    ]
    nb["cells"] = cells
    nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbf.write(nb, OUT / "baseline_notebook.ipynb")


if __name__ == "__main__":
    export_csv()
    build_notebook()
    print("wrote", OUT / "baseline_notebook.ipynb", "and", DATA)
