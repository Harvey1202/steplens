"""Numbers for docs/case_studies.md: per-recording walkthroughs, per-category recall of
the best models, position-bucket miss rates on the two step splits, and per-person /
per-kitchen rates. Thresholds are each model's balanced operating point on its own
validation steps (the tool's default)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from steplens.calibrate import best_threshold  # noqa: E402
from steplens.data import CAT_KEYS, KEY_TO_CAT  # noqa: E402

PRED = ROOT / "data" / "predictions"
steps = pd.read_parquet(ROOT / "data/processed/steps.parquet").set_index("step_key")
recs = pd.read_parquet(ROOT / "data/processed/recordings.parquet").set_index("recording_id")


def scored(model):
    p = pd.read_parquet(PRED / model / "steps.parquet").set_index("step_key")
    d = steps.loc[p.index].copy()
    d["score"] = p["score"]
    d["phase"] = p["phase"]
    val = d[d.phase == "val"]
    t = best_threshold(val.has_errors.values.astype(bool), val.score.values, "balanced")
    d["pos"] = d.score > t
    return d, t


MODELS = ["omnivore_transformer_video_step_lr1e-5", "omnivore_mlp_video_step_lr1e-4", "slowfast_mlp_video_step_lr1e-4",
          "omnivore_mlp_video_step_random_lr1e-4"]
data = {m: scored(m) for m in MODELS}

print("=== thresholds")
for m, (d, t) in data.items():
    print(f"  {m}: {t:.2f}")

print("\n=== per-category recall on test (balanced thresholds); paper zero-shot F1 for reference: Technique 0.6-0.7, Temperature 0.9-1.2, Timing 1.1-5.1")
for m in MODELS[:3]:
    d, _ = data[m]
    te = d[d.phase == "test"]
    row = []
    for k in CAT_KEYS[:6]:
        mm = te[f"err_{k}"].astype(bool)
        row.append(f"{KEY_TO_CAT[k].split()[0]} {100*te.pos[mm].mean():.0f}% (n={int(mm.sum())})")
    fa = te.pos[~te.has_errors].mean()
    print(f"  {m}: " + ", ".join(row) + f"; false alarm {100*fa:.0f}%")

print("\n=== miss rate by position bucket on test: official step split vs random step split (Omnivore MLP lr1e-4)")
for m in ["omnivore_mlp_video_step_lr1e-4", "omnivore_mlp_video_step_random_lr1e-4"]:
    d, _ = data[m]
    te = d[(d.phase == "test") & d.has_errors]
    g = te.groupby("position_bucket")["pos"].agg(["count", "mean"])
    print(f"  {m}: " + ", ".join(f"{b} miss {100*(1-r['mean']):.0f}% (n={int(r['count'])})" for b, r in g.iterrows()))
    # position distribution of test steps
    allte = d[d.phase == "test"]
    print("     test steps by bucket: " + ", ".join(f"{b}={int(c)}" for b, c in allte.position_bucket.value_counts().sort_index().items()))

print("\n=== recordings with video: per-recording test outcomes for the transformer")
d, t = data["omnivore_transformer_video_step_lr1e-5"]
for rid in ["25_48", "16_27", "20_48", "15_37", "21_50", "13_38", "10_6"]:
    r = recs.loc[rid]
    te = d[(d.recording_id == rid) & (d.phase == "test")]
    allr = d[d.recording_id == rid]
    print(f"  {rid} {r.activity_name} person {r.person_id} kitchen {r.environment_id} dur {r.duration_sec/60:.1f} min | all steps {len(allr)} err {int(allr.has_errors.sum())} | "
          f"test steps {len(te)} err {int(te.has_errors.sum())} missed {int((te.has_errors & ~te.pos).sum())} FA {int((~te.has_errors & te.pos).sum())}")

print("\n=== walkthrough 25_48 (all steps, transformer lr1e-5 + MLP lr1e-4)")
dm, tm = data["omnivore_mlp_video_step_lr1e-4"]
w = d[d.recording_id == "25_48"].sort_values("start_time")
for s in w.itertuples():
    mlp = dm.loc[s.Index]
    tags = ", ".join(s.error_tags) if len(s.error_tags) else "normal"
    print(f"  {int(s.start_time)//60:2d}:{int(s.start_time)%60:02d}-{int(s.end_time)//60:2d}:{int(s.end_time)%60:02d} {s.phase:5s} {s.step_description.split(' -')[0]:<12s} {tags:<40s} T {s.score:.2f} {'POS' if s.pos else 'neg'} | M {mlp.score:.2f} {'POS' if mlp.pos else 'neg'}" + ("  <-- skipped" if s.is_missing else ""))
    if s.has_errors:
        for e in s.error_list:
            print(f"        · {e}")

print("\n=== per-person test miss rate and AUC (transformer lr1e-5)")
te = d[d.phase == "test"]
for p, g in te.groupby("person_id"):
    e = g[g.has_errors]
    auc = roc_auc_score(g.has_errors, g.score) if g.has_errors.sum() >= 10 else np.nan
    print(f"  person {p}: errors {len(e)} missed {100*(1-e.pos.mean()):.0f}% AUC {auc:.2f}")
print("\n=== per-kitchen test miss rate and AUC (transformer lr1e-5)")
for k, g in te.groupby("environment_id"):
    e = g[g.has_errors]
    if len(e) < 10:
        continue
    print(f"  kitchen {k}: errors {len(e)} missed {100*(1-e.pos.mean()):.0f}% AUC {roc_auc_score(g.has_errors, g.score):.2f} FA {100*g.pos[~g.has_errors].mean():.0f}%")
