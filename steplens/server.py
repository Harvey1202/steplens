"""FastAPI backend for StepLens.

Serves the processed tables, every trained model's predictions, slice
discovery, and the static front end.

    .venv/bin/uvicorn steplens.server:app --port 8781 --reload
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

from .data import CATEGORIES, CAT_KEYS, load_tables
from .metrics import SLICE_ATTRS, binary_metrics, confirm_slices, normalize_scores, per_category, slice_table

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "data" / "predictions"
VIDEOS = ROOT / "data" / "videos"
FRONTEND = ROOT / "frontend"

SPLITS = ["step", "step_random", "recordings", "person", "environment"]


def _json(df: pd.DataFrame) -> Response:
    """DataFrame -> JSON records with NaN as null and arrays as lists."""
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].apply(lambda v: list(v) if isinstance(v, (np.ndarray, list, tuple)) else v)
    return Response(content=df.to_json(orient="records"), media_type="application/json")


class Store:
    def __init__(self):
        self.recordings, self.steps, self.segments, self.graphs = load_tables()
        self.steps = self.steps.set_index("step_key", drop=False)
        self.models: dict[str, dict] = {}
        self.refresh_models()

    def refresh_models(self):
        seen = set()
        if PRED.exists():
            for d in sorted(PRED.iterdir()):
                meta_p, steps_p = d / "meta.json", d / "steps.parquet"
                if not (meta_p.exists() and steps_p.exists()):
                    continue
                seen.add(d.name)
                mtime = meta_p.stat().st_mtime
                if d.name in self.models and self.models[d.name]["mtime"] == mtime:
                    continue
                meta = json.load(open(meta_p))
                df = pd.read_parquet(steps_p)
                # official "step normalization": min-max within each phase
                df["score_norm"] = df.groupby("phase")["score"].transform(normalize_scores)
                self.models[d.name] = dict(meta=meta, steps=df, mtime=mtime, seconds=None)
        for k in list(self.models):
            if k not in seen:
                del self.models[k]

    def model(self, model_id: str) -> dict:
        self.refresh_models()
        if model_id not in self.models:
            raise HTTPException(404, f"unknown model {model_id}")
        return self.models[model_id]

    def seconds(self, model_id: str) -> pd.DataFrame:
        m = self.model(model_id)
        if m["seconds"] is None:
            m["seconds"] = pd.read_parquet(PRED / model_id / "seconds.parquet")
        return m["seconds"]

    def scored_steps(self, model_id: str, phase: str | None) -> pd.DataFrame:
        """Step metadata joined with one model's scores (optionally one phase)."""
        m = self.model(model_id)
        pred = m["steps"]
        if phase and phase != "all":
            pred = pred[pred["phase"] == phase]
        df = self.steps.loc[pred["step_key"].values].copy()
        df["score"] = pred["score"].values
        df["score_norm"] = pred["score_norm"].values
        df["phase"] = pred["phase"].values
        return df


store = Store()
app = FastAPI(title="StepLens")


@app.middleware("http")
async def no_cache(request, call_next):
    """The front end is edited in place and predictions appear as runs finish, so never let the browser cache."""
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-cache"
    return response


@app.get("/api/meta")
def meta():
    s = store.steps
    acts = s[["activity_id", "activity_name"]].drop_duplicates().sort_values("activity_id")
    return dict(
        categories=CATEGORIES,
        category_keys=CAT_KEYS,
        slice_attrs=SLICE_ATTRS,
        splits=SPLITS,
        activities=[dict(activity_id=int(a), activity_name=n) for a, n in acts.itertuples(index=False)],
        persons=sorted(int(p) for p in s["person_id"].unique()),
        environments=sorted(int(e) for e in s["environment_id"].unique()),
        n_steps=int(len(s)),
        n_recordings=int(len(store.recordings)),
        videos=sorted(p.name.replace("_360p.mp4", "") for p in VIDEOS.glob("*_360p.mp4")) if VIDEOS.exists() else [],
    )


@app.get("/api/steps")
def steps():
    return _json(store.steps.reset_index(drop=True))


@app.get("/api/segments")
def segments(recording_id: str | None = None):
    df = store.segments
    if recording_id:
        df = df[df["recording_id"] == recording_id]
    return _json(df)


@app.get("/api/recordings")
def recordings():
    return _json(store.recordings)


@app.get("/api/task_graphs")
def task_graphs():
    return store.graphs


@app.get("/api/models")
def models():
    store.refresh_models()
    out = []
    for mid, m in store.models.items():
        meta = m["meta"]
        hist = [
            dict(epoch=h["epoch"], train_loss=h["train_loss"],
                 val_auc=h["val"].get("raw@0.5", {}).get("auc"), val_f1=h["val"].get("raw@0.5", {}).get("f1"),
                 test_auc=h["test"].get("raw@0.5", {}).get("auc"), test_f1=h["test"].get("raw@0.5", {}).get("f1"))
            for h in meta.get("history", [])
        ]
        out.append(dict(
            model_id=mid, backbone=meta["backbone"], variant=meta["variant"], modality=meta["modality"],
            split=meta["split"], split_col=meta["split_col"], segment_len=meta.get("segment_len", 1),
            best_epoch=meta["best_epoch"], n_steps=meta["n_steps"], metrics=meta["metrics"], history=hist,
            tag=meta.get("args", {}).get("tag", ""),
            label=f"{meta['backbone']} {meta['variant']}" + (
                f" ({'+'.join(meta['modality'])})" if meta["backbone"] == "imagebind" else "") + f" · {meta['split']}" + (
                f" · {meta['args']['tag']}" if meta.get("args", {}).get("tag") else ""),
        ))
    return out


@app.get("/api/predictions/{model_id}")
def predictions(model_id: str):
    m = store.model(model_id)
    return _json(m["steps"][["step_key", "phase", "label", "score", "score_norm", "score_max", "n_rows"]])


@app.get("/api/seconds/{model_id}")
def seconds(model_id: str, recording_id: str = Query(...)):
    sec = store.seconds(model_id)
    keys = store.steps.index[store.steps["recording_id"] == recording_id]
    sub = sec[sec["step_key"].isin(keys)]
    out = {k: g.sort_values("t")["score"].round(4).tolist() for k, g in sub.groupby("step_key")}
    return dict(model_id=model_id, recording_id=recording_id, segment_len=store.model(model_id)["meta"].get("segment_len", 1), seconds=out)


def _pred(df: pd.DataFrame, threshold: float, norm: bool) -> pd.Series:
    col = "score_norm" if norm else "score"
    return df[col] > threshold


@app.get("/api/metrics")
def metrics(model_id: str, phase: str = "test", threshold: float = 0.5, norm: bool = False):
    df = store.scored_steps(model_id, phase)
    pred = _pred(df, threshold, norm)
    return dict(
        overall=binary_metrics(df["has_errors"].values, df["score_norm" if norm else "score"].values, threshold),
        per_category=per_category(df, pred),
    )


@app.get("/api/slices")
def slices(model_id: str, phase: str = "test", threshold: float = 0.5, norm: bool = False,
           min_size: int = 10, min_errors: int = 10, max_order: int = 2, categories: bool = True, limit: int = 400,
           sort: str = "z_miss", activity_id: list[int] | None = Query(None),
           person_id: list[int] | None = Query(None), environment_id: list[int] | None = Query(None),
           category: str | None = None, confirm: str | None = None):
    """Slices found on `phase`; with `confirm=<phase>` each slice is re-evaluated on that hold-out phase."""
    if category and category not in CAT_KEYS:
        raise HTTPException(400, f"unknown category {category}")

    def subset(ph: str) -> pd.DataFrame:
        df = store.scored_steps(model_id, ph)
        if category:
            df = df[~df["has_errors"] | df[f"err_{category}"].astype(bool)]
        if activity_id:
            df = df[df["activity_id"].isin(activity_id)]
        if person_id:
            df = df[df["person_id"].isin(person_id)]
        if environment_id:
            df = df[df["environment_id"].isin(environment_id)]
        return df

    df = subset(phase)
    if df.empty:
        return dict(overall={}, slices=[], n_slices=0)
    pred = _pred(df, threshold, norm)
    tab = slice_table(df, pred, min_size=min_size, max_order=max_order, include_categories=categories,
                      min_errors=min_errors, scores=df["score_norm" if norm else "score"])
    confirm_overall = None
    if len(tab):
        if sort not in tab.columns:
            raise HTTPException(400, f"unknown sort column {sort}")
        asc = sort in ("recall_lift", "recall")
        tab = tab.sort_values(sort, ascending=asc, na_position="last").head(limit)
        if confirm and confirm != phase:
            dfc = subset(confirm)
            if not dfc.empty:
                tab, confirm_overall = confirm_slices(dfc, _pred(dfc, threshold, norm), tab)
    overall = binary_metrics(df["has_errors"].values, df["score_norm" if norm else "score"].values, threshold)
    return dict(overall=overall, n_slices=int(len(tab)), confirm=confirm, confirm_overall_miss=confirm_overall,
                slices=json.loads(tab.to_json(orient="records")))


@app.get("/api/compare")
def compare(a: str, b: str, phase: str = "test", threshold: float = 0.5, norm: bool = False):
    """Per-step agreement of two models on the steps both scored in this phase."""
    da = store.scored_steps(a, phase)
    db = store.scored_steps(b, phase)
    common = da.index.intersection(db.index)
    da, db = da.loc[common], db.loc[common]
    pa, pb = _pred(da, threshold, norm), _pred(db, threshold, norm)
    y = da["has_errors"].astype(bool)
    out = []
    for key in CAT_KEYS + ["normal"]:
        m = (~y) if key == "normal" else da[f"err_{key}"].astype(bool)
        out.append(dict(key=key, n=int(m.sum()), both=int((pa & pb & m).sum()), a_only=int((pa & ~pb & m).sum()),
                        b_only=int((~pa & pb & m).sum()), neither=int((~pa & ~pb & m).sum())))
    return dict(a=a, b=b, n_common=int(len(common)), agreement=out,
                step_flags={k: int(v) for k, v in zip(common, (pa.astype(int) * 2 + pb.astype(int)))})


if VIDEOS.exists():
    app.mount("/videos", StaticFiles(directory=str(VIDEOS)), name="videos")
app.mount("/", StaticFiles(directory=str(FRONTEND), html=True), name="frontend")
