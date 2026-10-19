"""Data layer for StepLens.

Turns the CaptainCook4D annotation release (vendor/annotations) and the
error-recognition split files (vendor/error_recognition/er_annotations) into
three tidy tables plus the recipe task graphs:

* recordings  one row per recording (384)
* steps       one row per (recording_id, step_id), the unit the official
              error-recognition models score (5,700 rows incl. missing steps)
* segments    one row per step occurrence with its own start/end time
              (a few steps are performed more than once in a recording)

Run as a script to (re)build data/processed/*.parquet and task_graphs.json.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ANN = ROOT / "vendor" / "annotations"
ER_SPLITS = ROOT / "vendor" / "error_recognition" / "er_annotations"
PROCESSED = ROOT / "data" / "processed"

# Category names exactly as they appear in the annotation files.
CATEGORIES = [
    "Preparation Error",
    "Measurement Error",
    "Order Error",
    "Timing Error",
    "Technique Error",
    "Temperature Error",
    "Missing Step",
    "Other",
]
# Short column-safe keys, in the same order.
CAT_KEYS = ["preparation", "measurement", "order", "timing", "technique", "temperature", "missing", "other"]
CAT_TO_KEY = dict(zip(CATEGORIES, CAT_KEYS))
KEY_TO_CAT = dict(zip(CAT_KEYS, CATEGORIES))

# The official step split (CaptainCookStepDataset._init_step_split) takes the
# first 75% of each recording's normal steps and error steps for training, the
# next 16% for the phase it calls "test", and the remaining 9% for "val".
STEP_SPLIT_FRACTIONS = (0.75, 0.16)


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_raw():
    complete = json.load(open(ANN / "annotation_json" / "complete_step_annotations.json"))
    errors = json.load(open(ANN / "annotation_json" / "error_annotations.json"))
    step_ann = json.load(open(ANN / "annotation_json" / "step_annotations.json"))
    video_info = pd.read_csv(ANN / "metadata" / "video_information.csv")
    return complete, errors, step_ann, video_info


def load_er_split(name: str) -> dict[str, str]:
    """Map recording_id -> train/val/test for an error_recognition split file."""
    d = json.load(open(ER_SPLITS / f"{name}_combined_splits.json"))
    out = {}
    for phase, ids in d.items():
        for rid in ids:
            out[rid] = phase
    return out


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def build_tables():
    complete, errors, step_ann, video_info = load_raw()
    err_by_rec = {r["recording_id"]: r for r in errors}
    vinfo = video_info.set_index("recording_id")

    rec_split = load_er_split("recordings")
    person_split = load_er_split("person")
    env_split = load_er_split("environment")

    seg_rows, step_rows, rec_rows = [], [], []

    for rid, rec in complete.items():
        meta = {
            "recording_id": rid,
            "activity_id": int(rec["activity_id"]),
            "activity_name": rec["activity_name"],
            "person_id": int(rec["person_id"]),
            "environment_id": int(rec["environment"]),
        }
        err_rec = err_by_rec[rid]
        # error_annotations.json carries the error tags; step_annotations.json
        # carries has_errors. Both list steps in the same (annotation) order.
        ann_steps = step_ann[rid]["steps"]
        err_steps = err_rec["step_annotations"]
        assert len(ann_steps) == len(err_steps) == len(rec["steps"]), rid

        # --- segments: one row per occurrence --------------------------------
        occ = []
        for i, (s, e) in enumerate(zip(ann_steps, err_steps)):
            assert s["step_id"] == e["step_id"], (rid, i)
            tags = [x["tag"] for x in e.get("errors", [])]
            descs = [x.get("description", "") for x in e.get("errors", [])]
            missing = s["start_time"] < 0 or s["end_time"] < 0
            occ.append(
                dict(
                    **meta,
                    step_id=int(s["step_id"]),
                    step_description=s["description"],
                    ann_index=i,
                    start_time=float(s["start_time"]),
                    end_time=float(s["end_time"]),
                    duration=0.0 if missing else float(s["end_time"]) - float(s["start_time"]),
                    is_missing=missing,
                    has_errors=bool(s["has_errors"]),
                    error_tags=tags,
                    error_descriptions=descs,
                    modified_description=e.get("modified_description"),
                )
            )
        # position among performed (non-missing) occurrences, by start time
        performed = sorted((o for o in occ if not o["is_missing"]), key=lambda o: o["start_time"])
        n_perf = len(performed)
        for rank, o in enumerate(performed):
            o["position_idx"] = rank
            o["n_performed"] = n_perf
            o["rel_position"] = rank / (n_perf - 1) if n_perf > 1 else 0.0
        for o in occ:
            o.setdefault("position_idx", -1)
            o.setdefault("n_performed", n_perf)
            o.setdefault("rel_position", np.nan)
        seg_rows.extend(occ)

        # --- steps: group occurrences by step_id (official unit) -------------
        by_step: dict[int, list] = {}
        for o in occ:
            by_step.setdefault(o["step_id"], []).append(o)
        n_error_steps = 0
        for step_id, occs in by_step.items():
            first = occs[0]
            perf = [o for o in occs if not o["is_missing"]]
            tags = sorted({t for o in occs for t in o["error_tags"]})
            has_err = any(o["has_errors"] for o in occs)
            n_error_steps += int(has_err)
            row = dict(
                **meta,
                step_id=step_id,
                step_description=first["step_description"],
                n_occurrences=len(occs),
                is_missing=len(perf) == 0,
                start_time=min((o["start_time"] for o in perf), default=-1.0),
                end_time=max((o["end_time"] for o in perf), default=-1.0),
                duration=sum(o["duration"] for o in perf),
                # the official loader floors/ceils each occurrence, so the
                # number of 1 s feature rows can exceed the duration slightly
                n_seconds=sum(int(np.ceil(o["end_time"]) - np.floor(o["start_time"])) for o in perf),
                position_idx=min((o["position_idx"] for o in perf), default=-1),
                n_performed=n_perf,
                rel_position=min((o["rel_position"] for o in perf), default=np.nan),
                has_errors=has_err,
                n_errors=sum(len(o["error_tags"]) for o in occs),
                error_tags=tags,
                error_descriptions=[d for o in occs for d in o["error_descriptions"]],
                error_list=[f"{t}: {d}" for o in occs for t, d in zip(o["error_tags"], o["error_descriptions"])],
            )
            for cat, key in CAT_TO_KEY.items():
                row[f"err_{key}"] = cat in tags
            row["recordings_split"] = rec_split.get(rid)
            row["person_split"] = person_split.get(rid)
            row["environment_split"] = env_split.get(rid)
            step_rows.append(row)

        rec_rows.append(
            dict(
                **meta,
                is_error_recording=bool(err_rec["is_error"]),
                duration_sec=float(vinfo.loc[rid, "duration(sec)"]) if rid in vinfo.index else np.nan,
                n_steps=len(by_step),
                n_occurrences=len(occ),
                n_missing_steps=sum(1 for v in by_step.values() if all(o["is_missing"] for o in v)),
                n_error_steps=n_error_steps,
                recordings_split=rec_split.get(rid),
                person_split=person_split.get(rid),
                environment_split=env_split.get(rid),
            )
        )

    recordings = pd.DataFrame(rec_rows).sort_values("recording_id").reset_index(drop=True)
    steps = pd.DataFrame(step_rows)
    segments = pd.DataFrame(seg_rows)

    steps = add_step_splits(steps)
    steps["step_key"] = steps["recording_id"] + ":" + steps["step_id"].astype(str)
    segments["step_key"] = segments["recording_id"] + ":" + segments["step_id"].astype(str)
    # duration bucket for slicing
    steps["duration_bucket"] = pd.cut(
        steps["duration"].where(~steps["is_missing"]),
        bins=[-0.01, 15, 30, 60, 120, 1e9],
        labels=["<15s", "15-30s", "30-60s", "60-120s", ">120s"],
    ).astype(object)
    steps["position_bucket"] = pd.cut(
        steps["rel_position"], bins=[-0.01, 0.25, 0.5, 0.75, 1.0], labels=["Q1", "Q2", "Q3", "Q4"]
    ).astype(object)
    return recordings, steps, segments


def add_step_splits(steps: pd.DataFrame) -> pd.DataFrame:
    """Add the official (unshuffled) step split and a seeded random one.

    Both keep the official proportions per recording and per class
    (normal/error): 75% train, 16% "test", 9% "val". Missing steps get NaN
    because the official loader drops them.
    """
    official = pd.Series(index=steps.index, dtype=object)
    random = pd.Series(index=steps.index, dtype=object)
    rng = np.random.default_rng(1000)
    f_train, f_test = STEP_SPLIT_FRACTIONS
    for _, grp in steps[~steps["is_missing"]].groupby("recording_id", sort=False):
        for is_err, sub in grp.groupby("has_errors", sort=False):
            idx = list(sub.index)  # annotation order, as the official loader keeps it
            n = len(idx)
            a, b = int(n * f_train), int(n * (f_train + f_test))
            official.loc[idx[:a]] = "train"
            official.loc[idx[a:b]] = "test"
            official.loc[idx[b:]] = "val"
            perm = list(rng.permutation(idx))
            random.loc[perm[:a]] = "train"
            random.loc[perm[a:b]] = "test"
            random.loc[perm[b:]] = "val"
    steps = steps.copy()
    steps["step_split"] = official
    steps["step_split_random"] = random
    return steps


# --------------------------------------------------------------------------- #
# Task graphs
# --------------------------------------------------------------------------- #
def build_task_graphs(steps: pd.DataFrame) -> dict:
    """Load the per-recipe DAGs and map their local node ids to global step_ids.

    Task graph files use local ids with 0 = START and the last id = END and
    edges as [from, to]. Nodes are matched to global step ids by description.
    """
    desc_to_step = (
        steps.groupby(["activity_id", "step_description"])["step_id"].agg(lambda s: s.mode().iloc[0]).to_dict()
    )
    activities = steps[["activity_id", "activity_name"]].drop_duplicates().sort_values("activity_id")
    graphs = {}
    for _, a in activities.iterrows():
        path = ANN / "task_graphs" / f"{_slug(a.activity_name)}.json"
        g = json.load(open(path))
        local = {int(k): v for k, v in g["steps"].items()}
        nodes = []
        unmatched = []
        for lid, desc in sorted(local.items()):
            if desc in ("START", "END"):
                nodes.append({"id": lid, "label": desc, "step_id": None, "kind": desc.lower()})
                continue
            sid = desc_to_step.get((int(a.activity_id), desc))
            if sid is None:
                # fall back to a whitespace/case-insensitive match
                cands = {k[1].strip().lower(): v for k, v in desc_to_step.items() if k[0] == int(a.activity_id)}
                sid = cands.get(desc.strip().lower())
            if sid is None:
                unmatched.append(desc)
            nodes.append({"id": lid, "label": desc, "step_id": sid, "kind": "step"})
        graphs[int(a.activity_id)] = {
            "activity_id": int(a.activity_id),
            "activity_name": a.activity_name,
            "nodes": nodes,
            "edges": [{"source": int(s), "target": int(t)} for s, t in g["edges"]],
            "unmatched": unmatched,
        }
    return graphs


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def save_tables(recordings, steps, segments, graphs):
    PROCESSED.mkdir(parents=True, exist_ok=True)
    recordings.to_parquet(PROCESSED / "recordings.parquet", index=False)
    steps.to_parquet(PROCESSED / "steps.parquet", index=False)
    segments.to_parquet(PROCESSED / "segments.parquet", index=False)
    json.dump(graphs, open(PROCESSED / "task_graphs.json", "w"), indent=1)


def load_tables():
    recordings = pd.read_parquet(PROCESSED / "recordings.parquet")
    steps = pd.read_parquet(PROCESSED / "steps.parquet")
    segments = pd.read_parquet(PROCESSED / "segments.parquet")
    graphs = json.load(open(PROCESSED / "task_graphs.json"))
    return recordings, steps, segments, graphs


def summary(recordings, steps, segments, graphs):
    lines = []
    lines.append(f"recordings: {len(recordings)}  steps: {len(steps)}  segments: {len(segments)}")
    lines.append(f"error steps: {int(steps.has_errors.sum())}  missing steps: {int(steps.is_missing.sum())}")
    counts = {cat: int(steps[f'err_{key}'].sum()) for cat, key in CAT_TO_KEY.items()}
    lines.append("category counts (steps): " + json.dumps(counts))
    for col in ["recordings_split", "person_split", "environment_split", "step_split", "step_split_random"]:
        lines.append(f"{col}: {steps[col].value_counts(dropna=False).to_dict()}")
    um = {g['activity_name']: g['unmatched'] for g in graphs.values() if g['unmatched']}
    lines.append(f"task graphs: {len(graphs)} recipes, unmatched nodes: {um if um else 'none'}")
    # position bias of the official step split
    pos = steps[~steps.is_missing].groupby("step_split")["rel_position"].mean().round(3).to_dict()
    posr = steps[~steps.is_missing].groupby("step_split_random")["rel_position"].mean().round(3).to_dict()
    lines.append(f"mean rel_position by official step split: {pos}")
    lines.append(f"mean rel_position by random step split:   {posr}")
    return "\n".join(lines)


if __name__ == "__main__":
    recordings, steps, segments = build_tables()
    graphs = build_task_graphs(steps)
    save_tables(recordings, steps, segments, graphs)
    print(summary(recordings, steps, segments, graphs))
