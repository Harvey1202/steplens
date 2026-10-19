"""Step-level metrics, per-category breakdowns and slice discovery.

All functions take a DataFrame with one row per step that has at least:
``has_errors`` (bool label), ``score`` (mean per-second sigmoid) and the
``err_<category>`` boolean columns from :mod:`steplens.data`.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .data import CAT_KEYS, KEY_TO_CAT

SLICE_ATTRS = ["activity_name", "person_id", "environment_id", "duration_bucket", "position_bucket"]


def normalize_scores(scores: pd.Series) -> pd.Series:
    """Official 'step normalization': min-max over the evaluated set."""
    lo, hi = scores.min(), scores.max()
    if hi - lo <= 0:
        return scores * 0.0
    return (scores - lo) / (hi - lo)


def binary_metrics(y: np.ndarray, s: np.ndarray, threshold: float) -> dict:
    y = np.asarray(y).astype(int)
    pred = (np.asarray(s) > threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    out = dict(
        n=int(len(y)), n_error=int(y.sum()), tp=tp, fp=fp, fn=fn, tn=tn,
        accuracy=(tp + tn) / len(y) if len(y) else 0.0,
        precision=precision, recall=recall, f1=f1,
        false_alarm=fp / (fp + tn) if fp + tn else 0.0,
    )
    try:
        out["auc"] = float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")
    except ValueError:
        out["auc"] = float("nan")
    return out


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion k/n."""
    if not n:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (float(max(0.0, centre - half)), float(min(1.0, centre + half)))


def confirm_slices(df: pd.DataFrame, pred: pd.Series, tab: pd.DataFrame):
    """Re-evaluate slices found on one phase on another (hold-out) phase.

    Adds confirm_n_error / confirm_miss_rate / confirm_miss_lo / confirm_miss_hi /
    confirm_miss_lift columns to ``tab`` and returns (tab, overall_miss_on_df).
    """
    y = df["has_errors"].astype(bool).values
    p = pred.astype(bool).values
    overall_miss = 1 - p[y].mean() if y.sum() else np.nan
    attr_cols = [c for c in tab.columns if c.startswith("attr_")]
    cols = {k: [] for k in ("confirm_n_error", "confirm_miss_rate", "confirm_miss_lo", "confirm_miss_hi", "confirm_miss_lift")}
    for _, r in tab.iterrows():
        mask = np.ones(len(df), bool)
        for c in attr_cols:
            v = r[c]
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            if isinstance(v, float) and v.is_integer():
                v = int(v)
            mask &= df[c[5:]].astype(str).values == str(v)
        cat = r.get("category")
        if isinstance(cat, str) and cat:
            key = {v: k for k, v in KEY_TO_CAT.items()}.get(cat)
            if key:
                mask &= df[f"err_{key}"].astype(bool).values
        em = mask & y
        n = int(em.sum())
        k = int((~p[em]).sum()) if n else 0
        rate = k / n if n else np.nan
        lo, hi = wilson(k, n)
        cols["confirm_n_error"].append(n)
        cols["confirm_miss_rate"].append(rate)
        cols["confirm_miss_lo"].append(lo)
        cols["confirm_miss_hi"].append(hi)
        cols["confirm_miss_lift"].append(rate - overall_miss if n else np.nan)
    tab = tab.copy()
    for k, v in cols.items():
        tab[k] = v
    return tab, overall_miss


def per_category(df: pd.DataFrame, pred: pd.Series) -> list[dict]:
    """Recall of each error category plus the normal-step false-alarm rate."""
    rows = []
    normal = ~df["has_errors"]
    for key in CAT_KEYS:
        m = df[f"err_{key}"].astype(bool)
        n = int(m.sum())
        rows.append(dict(category=KEY_TO_CAT[key], key=key, n=n,
                         recall=float(pred[m].mean()) if n else float("nan"),
                         caught=int(pred[m].sum())))
    rows.append(dict(category="Normal (false alarm)", key="normal", n=int(normal.sum()),
                     recall=float(pred[normal].mean()) if normal.sum() else float("nan"),
                     caught=int(pred[normal].sum())))
    return rows


def slice_table(df: pd.DataFrame, pred: pd.Series, min_size: int = 10, max_order: int = 2,
                attrs=SLICE_ATTRS, include_categories: bool = True, min_errors: int = 10,
                scores: pd.Series | None = None) -> pd.DataFrame:
    """Enumerate metadata slices and rank them by miss rate and false-alarm lift.

    A slice is a conjunction of up to ``max_order`` attribute=value conditions,
    optionally crossed with one error category. For each slice we report the
    recall on its error steps and the false-alarm rate on its normal steps,
    plus lift against the overall rates and a z-statistic for each lift (a
    binomial test of the slice rate against the overall rate). Ranking by z
    discounts tiny slices; ``min_errors`` drops slices with too few error steps.
    """
    y = df["has_errors"].astype(bool).values
    p = pred.astype(bool).values
    sc = None if scores is None else np.asarray(scores, dtype=float)
    overall_recall = p[y].mean() if y.sum() else np.nan
    overall_fa = p[~y].mean() if (~y).sum() else np.nan
    overall_miss = 1 - overall_recall

    def slice_auc(mask):
        """AUC of error-vs-normal within a slice: low = the model cannot separate them there."""
        if sc is None or y[mask].sum() < 5 or (~y[mask]).sum() < 5:
            return np.nan
        try:
            return float(roc_auc_score(y[mask], sc[mask]))
        except ValueError:
            return np.nan

    def z(rate, base, n):
        var = base * (1 - base) / n if n else np.nan
        return float((rate - base) / np.sqrt(var)) if var and var > 0 else np.nan

    conds = []  # list of (name, value, mask)
    for a in attrs:
        if a not in df.columns:
            continue
        col = df[a]
        for v in col.dropna().unique():
            conds.append((a, v, (col == v).values))
    cat_conds = [(None, None, np.ones(len(df), bool))]
    if include_categories:
        for key in CAT_KEYS:
            cat_conds.append(("category", KEY_TO_CAT[key], df[f"err_{key}"].astype(bool).values))

    rows = []
    for order in range(1, max_order + 1):
        for combo in itertools.combinations(range(len(conds)), order):
            names = [conds[i][0] for i in combo]
            if len(set(names)) < len(names):
                continue
            mask = np.logical_and.reduce([conds[i][2] for i in combo])
            if mask.sum() < min_size:
                continue
            desc = {conds[i][0]: conds[i][1] for i in combo}
            for cname, cval, cmask in cat_conds:
                if cname is None:
                    err_mask = mask & y
                else:
                    err_mask = mask & cmask & y  # category errors inside the slice
                n_err = int(err_mask.sum())
                norm_mask = mask & ~y
                n_norm = int(norm_mask.sum())
                if n_err < min_errors:
                    continue
                recall = p[err_mask].mean() if n_err else np.nan
                fa = p[norm_mask].mean() if n_norm else np.nan
                n_missed = int((~p[err_mask]).sum()) if n_err else 0
                miss_lo, miss_hi = wilson(n_missed, n_err)
                rows.append(dict(
                    slice=" & ".join(f"{k}={v}" for k, v in desc.items()) + (f" & category={cval}" if cname else ""),
                    order=order, category=cval, **{f"attr_{k}": v for k, v in desc.items()},
                    n=int(mask.sum()), n_error=n_err, n_normal=n_norm,
                    recall=recall, miss_rate=1 - recall if n_err else np.nan, n_missed=n_missed,
                    miss_lo=miss_lo, miss_hi=miss_hi,
                    recall_lift=recall - overall_recall if n_err else np.nan,
                    miss_lift=(1 - recall) - overall_miss if n_err else np.nan,
                    z_miss=z(1 - recall, overall_miss, n_err) if n_err else np.nan,
                    false_alarm=fa, false_alarm_lift=fa - overall_fa if n_norm else np.nan,
                    z_fa=z(fa, overall_fa, n_norm) if n_norm else np.nan,
                    auc=slice_auc(mask if cname is None else (mask & (cmask | ~y))),
                ))
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values("z_miss", ascending=False, na_position="last").reset_index(drop=True)
    return out


def bootstrap_ci(y: np.ndarray, p: np.ndarray, stat: str = "recall", n_boot: int = 500, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(bool)
    p = np.asarray(p).astype(bool)
    vals = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yy, pp = y[idx], p[idx]
        if stat == "recall":
            vals.append(pp[yy].mean() if yy.sum() else np.nan)
        else:
            vals.append(pp[~yy].mean() if (~yy).sum() else np.nan)
    vals = np.array(vals)
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))
