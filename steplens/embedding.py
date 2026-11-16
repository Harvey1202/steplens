"""Step-feature embedding: do the frozen features say more about who recorded a step than about its errors?

For every scored step, average the segment features the error-recognition heads see for it, then

* project the means to 2-D with UMAP for the tool's embedding view, and
* measure, in the full feature space (after PCA), how often a step's k nearest neighbours share its
  recording, person, kitchen, recipe, recipe step and error label, and, among the test error steps of
  a few reference models, its outcome (caught or missed). Each rate is compared with chance: the
  probability that a random other step shares the label.

If neighbours share the person or recording far above chance while they share the outcome only at
chance, the features encode who and where a step was recorded more than whether it went wrong.

    python -m steplens.embedding --backbone omnivore                     # needs data/features
    python -m steplens.embedding --backbone omnivore --center recording  # subtract each recording's mean first

Writes data/processed/embedding_<name>.parquet (step_key, x, y), data/processed/embedding_<name>.json
(settings and neighbour statistics) and docs/embedding_<name>.md. The tool reads the first two, so the
view works without the features once these small files exist.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_tables
from .features import FEATURES, FeatureStore

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
PRED = ROOT / "data" / "predictions"
DOCS = ROOT / "docs"

NEIGHBOUR_LABELS = [  # (column or derived key, label in reports)
    ("recording_id", "recording"),
    ("person_id", "person"),
    ("environment_id", "kitchen"),
    ("activity_id", "recipe"),
    ("recipe_step", "recipe step"),
    ("has_errors", "has an error"),
]
REFERENCE_MODELS = ["omnivore_mlp_video_step_lr1e-4", "omnivore_transformer_video_step_lr1e-5",
                    "paper_omnivore_transformer_video_step", "omnivore_transformer_video_recordings"]


def embedding_name(backbone: str, modality=("video",), center: str = "none") -> str:
    name = backbone + (f"-{'+'.join(modality)}" if backbone == "imagebind" else "")
    return name + ("-center" if center == "recording" else "")


def step_occurrences(segments: pd.DataFrame) -> dict[tuple[str, int], list[tuple[float, float]]]:
    """(recording_id, step_id) -> [(start, end), ...] in annotation order, as the training loader uses them."""
    seg = segments[~segments["is_missing"]].sort_values(["recording_id", "step_id", "ann_index"])
    return {k: list(zip(g["start_time"], g["end_time"])) for k, g in seg.groupby(["recording_id", "step_id"])}


def step_means(steps: pd.DataFrame, segments: pd.DataFrame, store: FeatureStore, center: str = "none", log=print):
    """(step keys, (n, D) matrix of mean segment features) for every scored step with features."""
    usable = steps[~steps["is_missing"]]
    occ = step_occurrences(segments)
    keys, rows, missing = [], [], []
    for rid, g in usable.groupby("recording_id", sort=True):
        if not store.has(rid):
            missing.append(rid)
            continue
        rec = store.recording(rid)
        shift = rec.mean(axis=0) if center == "recording" else 0.0
        for r in g.itertuples(index=False):
            x = store.step(rid, occ[(rid, r.step_id)])
            keys.append(r.step_key)
            rows.append(x.mean(axis=0) - shift)
        store._cache.pop(rid, None)  # one recording in memory at a time
    if missing and log:
        log(f"[features] {len(missing)} recordings without features were skipped: {missing[:10]}"
            f"{'...' if len(missing) > 10 else ''}")
    return keys, np.vstack(rows).astype(np.float32) if rows else np.zeros((0, store.dim), np.float32)


def knn_agreement(Z: np.ndarray, labels, k: int = 10) -> dict:
    """How often a point's k nearest neighbours (self excluded) share its label, against chance.

    Chance is the probability that a random other point shares the label: sum_c n_c (n_c - 1) / (N (N - 1)).
    """
    from sklearn.neighbors import NearestNeighbors

    y = np.asarray(labels)
    n = len(y)
    if n < 3:
        return dict(n=n, k=0, agreement=float("nan"), chance=float("nan"), lift=float("nan"))
    k = min(k, n - 1)
    idx = NearestNeighbors(n_neighbors=k + 1).fit(Z).kneighbors(Z, return_distance=False)
    nb = np.array([row[row != i][:k] for i, row in enumerate(idx)])  # drop self even when duplicates tie
    agreement = float((y[nb] == y[:, None]).mean())
    _, counts = np.unique(y, return_counts=True)
    chance = float((counts * (counts - 1)).sum() / (n * (n - 1)))
    return dict(n=int(n), k=int(k), agreement=agreement, chance=chance,
                lift=agreement / chance if chance > 0 else float("nan"))


def reference_outcomes(keys, Z, steps: pd.DataFrame, models=REFERENCE_MODELS, phase: str = "test", k: int = 10):
    """Neighbour agreement of missed-vs-caught among each reference model's error steps in ``phase``.

    Each model at its balanced threshold on its own validation steps, as in the tool.
    """
    from .calibrate import best_threshold

    pos = {key: i for i, key in enumerate(keys)}
    out = []
    for mid in models:
        p = PRED / mid / "steps.parquet"
        if not p.exists():
            continue
        pred = pd.read_parquet(p).set_index("step_key")
        lab = steps.set_index("step_key").loc[pred.index, "has_errors"].astype(bool)
        val = pred["phase"] == "val"
        t = best_threshold(lab[val].values, pred.loc[val, "score"].values, "balanced")
        sel = (pred["phase"] == phase) & lab & pred.index.isin(list(pos))
        idx = [pos[s] for s in pred.index[sel]]
        missed = (pred.loc[sel, "score"] <= t).values
        if len(idx) < 3:
            continue
        r = knn_agreement(Z[idx], missed, k)
        out.append(dict(model_id=mid, phase=phase, threshold=t, missed_rate=float(missed.mean()), **r))
    return out


def run(backbone: str, modality=("video",), center: str = "none", features: Path = FEATURES, k: int = 10,
        pca_dims: int = 50, n_neighbors: int = 15, min_dist: float = 0.1, seed: int = 0,
        out_dir: Path = PROCESSED, docs_dir: Path | None = DOCS, log=print) -> dict:
    import umap
    from sklearn.decomposition import PCA

    _, steps, segments, _ = load_tables()
    store = FeatureStore(backbone, modality, root=features)
    t0 = time.time()
    keys, X = step_means(steps, segments, store, center=center, log=log)
    if len(keys) < 20:
        raise SystemExit(f"only {len(keys)} steps have {backbone} features under {features}; "
                         "run scripts/download_features.sh and unzip the archives first")
    log(f"[embedding] {len(keys)} steps × {X.shape[1]} dims in {time.time() - t0:.0f}s")
    Z = PCA(n_components=min(pca_dims, X.shape[1], len(keys) - 1), random_state=seed).fit_transform(X)
    xy = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=seed).fit_transform(Z)

    meta = steps.set_index("step_key").loc[keys]
    meta = meta.assign(recipe_step=meta["activity_id"].astype(str) + ":" + meta["step_id"].astype(str))
    neighbours = [dict(key=col, label=label, **knn_agreement(Z, meta[col].values, k)) for col, label in NEIGHBOUR_LABELS]
    outcomes = reference_outcomes(keys, Z, steps, k=k)

    name = embedding_name(backbone, modality, center)
    info = dict(name=name, backbone=backbone, modality=list(store.modality), center=center, n_steps=len(keys),
                feature_dims=int(X.shape[1]), pca_dims=int(Z.shape[1]), k=k,
                umap=dict(n_neighbors=n_neighbors, min_dist=min_dist, seed=seed),
                created=time.strftime("%Y-%m-%d"), neighbours=neighbours, outcomes=outcomes)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(dict(step_key=keys, x=xy[:, 0].astype(np.float32), y=xy[:, 1].astype(np.float32))).to_parquet(
        out_dir / f"embedding_{name}.parquet", index=False)
    json.dump(info, open(out_dir / f"embedding_{name}.json", "w"), indent=1)
    report = markdown(info)
    if docs_dir is not None:
        Path(docs_dir, f"embedding_{name}.md").write_text(report)
    log(report)
    return info


def markdown(info: dict) -> str:
    pct = lambda v: f"{100 * v:.1f}%" if np.isfinite(v) else "–"
    x = lambda v: f"{v:.1f}×" if np.isfinite(v) else "–"
    lines = [f"# Step-feature neighbours: {info['name']}", "",
             f"{info['n_steps']} scored steps; each step is the mean of its {info['backbone']} segment features"
             f"{' minus its recording mean' if info['center'] == 'recording' else ''} ({info['feature_dims']} dims, "
             f"PCA to {info['pca_dims']}). For each step, its {info['k']} nearest neighbours in that space; "
             "*agree* = share of neighbours with the same label, *chance* = probability that a random other step "
             "shares it. Regenerate with `python -m steplens.embedding`.", "",
             "| Neighbours share the step's … | agree | chance | lift |", "| --- | --- | --- | --- |"]
    for r in info["neighbours"]:
        lines.append(f"| {r['label']} | {pct(r['agreement'])} | {pct(r['chance'])} | {x(r['lift'])} |")
    if info["outcomes"]:
        lines += ["", "Among each model's error steps on its test phase (balanced threshold on its validation steps), "
                  "do neighbours share the outcome (missed vs caught)?", "",
                  "| Model | error steps | missed | agree | chance | lift |", "| --- | --- | --- | --- | --- | --- |"]
        for r in info["outcomes"]:
            lines.append(f"| {r['model_id']} | {r['n']} | {pct(r['missed_rate'])} | {pct(r['agreement'])} | "
                         f"{pct(r['chance'])} | {x(r['lift'])} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbone", default="omnivore", choices=["omnivore", "slowfast", "imagebind", "x3d", "3dresnet"])
    ap.add_argument("--modality", nargs="+", default=["video"], help="ImageBind only: video audio text")
    ap.add_argument("--center", default="none", choices=["none", "recording"])
    ap.add_argument("--features", type=Path, default=FEATURES)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--pca", type=int, default=50)
    ap.add_argument("--n_neighbors", type=int, default=15)
    ap.add_argument("--min_dist", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", type=Path, default=PROCESSED)
    ap.add_argument("--docs_dir", type=Path, default=DOCS)
    a = ap.parse_args()
    run(a.backbone, a.modality, a.center, a.features, a.k, a.pca, a.n_neighbors, a.min_dist, a.seed, a.out_dir,
        a.docs_dir)


if __name__ == "__main__":
    main()
