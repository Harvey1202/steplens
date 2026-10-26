"""Score every step with one of the authors' released checkpoints.

The CaptainCook4D authors released the checkpoints used for the paper's
supervised error-recognition results (error_recognition issue #1, Box folder
`checkpoints/error_recognition_best`). Their parameter names match the heads in
:mod:`steplens.models`, so we load them unchanged, score every step on the
released features, and write a prediction set the tool can show next to our own
models.

    python -m steplens.evaluate_checkpoint data/paper_checkpoints/MLP_omnivore_step.pt \
        --backbone omnivore --variant mlp --split step --epoch 43
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .data import load_tables
from .features import FeatureStore
from .metrics import binary_metrics, normalize_scores
from .models import build_model
from .train import PRED, SPLIT_COL, build_step_list, phase_metrics, score_items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--backbone", required=True, choices=["omnivore", "slowfast", "x3d", "3dresnet", "imagebind"])
    ap.add_argument("--variant", required=True, choices=["mlp", "transformer"])
    ap.add_argument("--modality", nargs="+", default=["video"])
    ap.add_argument("--split", required=True, choices=["step", "recordings", "person", "environment"])
    ap.add_argument("--epoch", type=int, default=None, help="epoch in the checkpoint file name (for the record)")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    mod = "+".join(args.modality) if args.backbone == "imagebind" else "video"
    model_id = f"paper_{args.backbone}_{args.variant}_{mod}_{args.split}"
    out = PRED / model_id
    out.mkdir(parents=True, exist_ok=True)

    _, steps, segments, _ = load_tables()
    store = FeatureStore(args.backbone, args.modality)
    items = build_step_list(steps, segments, SPLIT_COL[args.split], store, print)

    model = build_model(args.variant, store.dim, len(store.modality))
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(args.device)
    df, per_sec = score_items(model, items, torch.device(args.device))

    df.to_parquet(out / "steps.parquet", index=False)
    import pandas as pd
    pd.DataFrame({
        "step_key": np.repeat(list(per_sec.keys()), [len(v) for v in per_sec.values()]),
        "t": np.concatenate([np.arange(len(v)) for v in per_sec.values()]),
        "score": np.concatenate(list(per_sec.values())).astype(np.float32),
    }).to_parquet(out / "seconds.parquet", index=False)

    metrics = {p: phase_metrics(df, p) for p in ("train", "val", "test")}
    # the paper's protocol: min-max normalize the test step scores, threshold 0.6 (step) or 0.4 (other splits)
    te = df[df.phase == "test"]
    t_paper = 0.6 if args.split == "step" else 0.4
    paper = binary_metrics(te.label.values, normalize_scores(te.score).values, t_paper)
    meta = dict(model_id=model_id, backbone=args.backbone, variant=args.variant, modality=store.modality,
                split=args.split, split_col=SPLIT_COL[args.split], segment_len=store.segment_len,
                args=dict(lr=float("nan"), tag="paper-ckpt", seed=None, checkpoint=Path(args.checkpoint).name),
                source="authors' released checkpoint", best_epoch=args.epoch, best_select_auc=None,
                n_steps={p: int((df.phase == p).sum()) for p in ("train", "val", "test")},
                metrics=metrics, paper_protocol=dict(threshold=t_paper, **paper), history=[])
    json.dump(meta, open(out / "meta.json", "w"), indent=1, default=float)
    print(f"{model_id}: test AUC {100*paper['auc']:.2f} | paper protocol (norm@{t_paper}) acc {100*paper['accuracy']:.2f} "
          f"P {100*paper['precision']:.2f} R {100*paper['recall']:.2f} F1 {100*paper['f1']:.2f} | n_test {len(te)}")


if __name__ == "__main__":
    main()
