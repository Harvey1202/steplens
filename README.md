# StepLens

Visual analytics for diagnosing step-level error-detection models on
[CaptainCook4D](https://github.com/CaptainCook4D) (DS-GA 3001, NYU, Fall 2026).

StepLens reproduces the paper's supervised error-recognition baselines over the
released frozen features, stores every model's per-step and per-second scores,
and links four views: an error-category × model miss-rate matrix, recipe task
graphs colored by miss rate, recording timelines with video, and a slice table
that ranks metadata subgroups by how much worse the model does there.

![overview](docs/screenshots/overview-light.png)

Screenshots: [overview](docs/screenshots/overview-light.png) (matrix, compare strip, slices,
recipes), [kitchen 2 drill-down](docs/screenshots/kitchen2-drilldown.png),
[task graph in dark mode](docs/screenshots/taskgraph-dark.png).

## Layout

```
steplens/          python package
  data.py          annotations -> data/processed/{recordings,steps,segments}.parquet + task_graphs.json
  features.py      per-recording feature loader (official file layout)
  models.py        MLP and transformer heads ported from error_recognition
  train.py         training/eval, writes data/predictions/<model_id>/
  metrics.py       step metrics, per-category recall, slice discovery and slice stability
  embedding.py     per-step mean features -> UMAP + nearest-neighbour statistics for the embedding view
  server.py        FastAPI backend + static front end
frontend/          index.html, styles.css, app.js, views.js (D3 v7 from cdnjs, no build step)
scripts/           download_features.sh, download_videos.sh
vendor/            official repos (annotations, error_recognition, feature_extractors, downloader)
data/              features/, processed/, predictions/, videos/ (all generated or downloaded, not committed)
```

## Quick start (no training needed)

The repository contains the processed tables (`data/processed`); the models' predictions
are published as a release asset. You only need Python 3.10+:

```bash
git clone https://github.com/Harvey1202/steplens && cd steplens
# download StepLens_predictions_v0.2.zip from the Releases page and unzip it in this folder,
# which creates data/predictions/<model_id>/{meta.json,steps.parquet,seconds.parquet}
unzip ~/Downloads/StepLens_predictions_v0.2.zip
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn steplens.server:app --port 8781
open http://127.0.0.1:8781/
```

If you received the project as a full zip instead, `data/predictions` is already there;
skip the unzip line.

Features (3.7 GB) and videos are not in the bundle; they are only needed to retrain or
to play clips, and `scripts/download_features.sh` / `scripts/download_videos.sh` fetch
them from the public Box archive.

## Setup

```bash
# python 3.10 with torch, pandas, scikit-learn, then:
python -m venv --system-site-packages .venv   # or a fresh venv + pip install -r requirements.txt
.venv/bin/pip install fastapi "uvicorn[standard]" h5py pyarrow tqdm

git clone --depth 1 https://github.com/CaptainCook4D/annotations vendor/annotations
git clone --depth 1 https://github.com/CaptainCook4D/error_recognition vendor/error_recognition
git clone --depth 1 https://github.com/CaptainCook4D/downloader vendor/downloader

scripts/download_features.sh          # ~3.7 GB: omnivore, slowfast (1 s) + imagebind video/audio/text (2 s)
cd data/features && for z in gopro_1s/*.zip; do unzip -q -o "$z" -d gopro_1s; done \
  && for z in gopro_2s/*.zip; do unzip -q -o "$z" -d gopro_2s; done && cd ../..

.venv/bin/python steplens/data.py     # build the processed tables
```

## Train a model

```bash
.venv/bin/python -m steplens.train --backbone omnivore --variant mlp --split step --epochs 30 --device cpu
.venv/bin/python -m steplens.train --backbone omnivore --variant transformer --split step --epochs 40 --device mps
.venv/bin/python -m steplens.train --backbone imagebind --variant transformer --modality video audio text --split step
```

Splits: `step` (the official per-recording step split, which is *not* shuffled:
train = first 75% of each recording's steps, "test" = next 16%, "val" = last 9%),
`step_random` (same proportions, seeded shuffle), `recordings`, `person`,
`environment` (the official recording-level splits). Learning rates default to
the official values (1e-3 MLP, 1e-6 transformer).

Each run writes `data/predictions/<model_id>/{meta.json,steps.parquet,seconds.parquet,best.pt,train.log}`.
Extra options: `--lr`, `--tag` (suffix for the model id), `--center recording`
(subtract each recording's mean feature), `--upweight technique temperature --upweight_factor 2`
(loss weight for error steps of those categories).

## The authors' checkpoints

The authors released the checkpoints used in the paper (error_recognition issue #1). To
score every step with one of them and show it in the tool next to our models:

```bash
# download e.g. Transformer/omnivore/error_recognition_Transformer_omnivore_step_epoch_9.pt
# from https://utdallas.app.box.com/s/uz3s1alrzucz03sleify8kazhuc1ksl3 into data/paper_checkpoints/
.venv/bin/python -m steplens.evaluate_checkpoint data/paper_checkpoints/Transformer_omnivore_step.pt \
    --backbone omnivore --variant transformer --split step --epoch 9
```

## Findings-guided fixes and reports

```bash
.venv/bin/python -m steplens.calibrate --model omnivore_mlp_video_step_lr1e-4 --by environment_id
    # per-kitchen thresholds tuned on val, evaluated on test; writes <model>_cal-environment/ as a derived model
.venv/bin/python scripts/results_table.py      # docs/results.md: every model vs the paper's numbers
.venv/bin/python scripts/slice_survival.py     # docs/slice_survival.md + docs/figures/slice_survival.png: how often flagged slices survive on test
.venv/bin/python scripts/split_positions.py    # docs/figures/split_positions.png: where each phase's steps sit in the recipe
.venv/bin/python scripts/embedding_figure.py   # docs/figures/umap_kitchen_outcome.png: UMAP by kitchen and by outcome
.venv/bin/python -m unittest discover -s tests # metric and slice-discovery tests
```

Documents: `proposal/Team17.pdf` (the submitted proposal), `docs/feedback_response.md` (response to the proposal feedback, with evidence), `docs/findings.md` (findings log), `docs/case_studies.md` (walkthroughs with
video), `docs/results.md` (all models), `docs/seeds.md` (seed robustness), `docs/report.md`
(report draft), `docs/slides_outline.md`, `docs/user_study/` (protocol, task sheet, SUS,
consent, reference findings, baseline notebook + CSVs; designed, not run).

License note: the dataset is released under Apache 2.0 (website, downloader and
error_recognition repos; the annotations README says the same, its LICENSE file is MIT).

## Reproducing the figures

Figures made by scripts (run from the repository root; they write to `docs/figures/`):

| Figure | Command |
| --- | --- |
| `split_positions.png` (position of each phase's steps in the recipe) | `python scripts/split_positions.py` |
| `slice_survival.png` (flagged slices re-checked on test) | `python scripts/slice_survival.py` |
| `umap_kitchen_outcome.png` (UMAP by kitchen and by outcome) | `python -m steplens.embedding --backbone omnivore` (once, needs the features), then `python scripts/embedding_figure.py` |

Screenshots of the tool: start it (`uvicorn steplens.server:app --port 8781`), open the URL
(after `http://127.0.0.1:8781/`) in a 1500 × 1000 window at 2× pixel density, and capture the element:

| Screenshot | URL | Element |
| --- | --- | --- |
| Overview | `?modelA=omnivore_mlp_video_step_lr1e-4&rule=balanced&theme=light` | top 960 px, matrix scrolled so model A's column is third |
| Task graph | `?modelA=omnivore_mlp_video_step_random_lr1e-4&rule=balanced&theme=light&activity_id=25` | the task graph |
| Slice stability | `?modelA=omnivore_mlp_video_step_lr1e-4&rule=balanced&theme=light` | slice table header + first 7 rows |
| Compare strip | `?modelA=omnivore_mlp_video_step_lr1e-4&modelB=slowfast_mlp_video_step_lr1e-4&rule=balanced&theme=light` | the compare strip |
| Timeline | `?modelA=omnivore_transformer_video_recordings&rule=balanced&theme=light&recording=25_40&person_id=2` | the timeline card down to its legend |

## Run the tool

```bash
.venv/bin/uvicorn steplens.server:app --port 8781     # or scripts/serve.sh
open http://127.0.0.1:8781/
```

The page keeps its state in the URL (model A/B, phase, operating point, filters,
recording, theme), so a view can be shared as a link. Operating points: each
model's threshold is chosen on its own validation steps (balanced = max TPR −
FPR, or best F1), or the paper protocol (min-max normalized scores at 0.6 / 0.4),
or one custom threshold for all models.

Optional videos for the timeline view: `scripts/download_videos.sh 16_27 25_48`
(360p GoPro, ~100 MB per 10-minute recording).

**Slice stability.** Next to the hold-out column, the slice table shows for every slice in
how many runs of model A's configuration (*seeds*, e.g. 3/3) and in how many other models
on the same split (*models*, e.g. 7/19, the authors' checkpoints included) the same slice is
flagged: miss rate above that model's own overall rate with z ≥ 2, each model at its own
threshold under the current operating-point rule. Hover a cell to see which models; sort
by *flagged by the most other models* to put the slices that do not depend on one run first.

**Step-feature embedding (UMAP).** The embedding card plots every step from the mean of its
frozen segment features, colored by model A's outcome, person, kitchen or recipe, and lists
how often a step's nearest neighbours in feature space share its recording, person,
kitchen, recipe, error label and outcome, against chance. It needs a small file computed
once from the features (data/features, see Setup; `pip install umap-learn`):

```bash
.venv/bin/python -m steplens.embedding --backbone omnivore                      # ~1 min after loading features
.venv/bin/python -m steplens.embedding --backbone omnivore --center recording    # optional: recording mean removed
```

This writes `data/processed/embedding_omnivore.{parquet,json}` (about 100 KB, read by the
tool, so it can be committed and viewed without the features) and `docs/embedding_omnivore.md`.
The repository already includes both results (`omnivore` and `omnivore-center`), so the view
works out of the box.

## Notes on the data

* Missing Step errors sit on steps with start/end = -1; they have no features and
  no model ever scores them. The matrix shows them as "–".
* The annotation files include an eighth category, Other (8 errors).
* ImageBind features exist for 383 of 384 recordings (12_6 is absent); no depth
  features are in the public archive.
