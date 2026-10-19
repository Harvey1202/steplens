# StepLens: Visual Analytics for Diagnosing Error-Detection Models in Procedural Cooking Videos

*DS-GA 3001, Fall 2026 · [team names] · draft of Oct 18, 2026*

## Abstract

CaptainCook4D (Peddi et al., NeurIPS 2024) labels every recipe step in 384 egocentric
cooking recordings as normal or as one of seven kinds of mistake, and benchmarks
supervised error-recognition models with a single accuracy, F1 and AUC per model. We
built StepLens, an interactive system that links an error-category × model miss-rate
matrix, recipe task graphs colored by miss rate, recording timelines aligned with the
video, and a slice table that ranks metadata subgroups by how much worse a model does
there, with hold-out confirmation. We reproduced the paper's baselines over its released
features (Omnivore MLP 74.1 vs 75.7 AUC; transformer 73.3 vs 75.7) and used the tool to
answer where they fail. Failures are not category-specific: per-category recall is flat
within a few points for every Omnivore model. They are cook- and recording-specific: one
participant is missed 61–69% of the time by every model on every split, and on held-out
recordings steps with a single subtle error are missed far more often than steps with
compound errors. Along the way the tool exposed two properties of the benchmark that the
paper's tables hide: the Missing Step category is never scored, and the released step
split is unshuffled, so its validation and test phases are different populations. Two
findings-guided fixes did not improve F1, and we explain why. We evaluate the tool
through four case studies, each of which produced a finding that the paper's tables
cannot show and that survives the tool's hold-out, model-switch and seed checks.

## 1. Introduction

Models that watch a person follow a procedure and flag mistakes are evaluated, almost
without exception, with aggregate metrics. CaptainCook4D is the first large egocentric
dataset with deliberately induced, step-level, categorized errors, and its benchmark
reports accuracy, F1 and AUC per backbone and head. Those numbers do not say which of the
seven error categories a model catches, whether it fails on particular recipes or steps,
or whether two backbones fail on the same steps. We asked three questions:

* **RQ1.** Where do the supervised error-recognition models fail: which categories,
  recipes, steps, cooks, kitchens, step lengths and positions account for the misses and
  the false alarms, and are the failures systematic?
* **RQ2.** Do Omnivore, SlowFast and multimodal ImageBind models miss the same errors?

Our contributions are the open-source tool, a failure analysis of the paper's baselines
broken down by category and slice, an evaluation of two fixes suggested by that analysis,
and a case-study evaluation of the tool. Sections 3 and 5 also document several
properties of the released benchmark that affect how its numbers should be read.

## 2. Related work

Assembly101, HoloAssist and PREGO study mistakes in procedural tasks; HD-EPIC annotates
real kitchens in detail; Ego4D and Ego-Exo4D provide keystep annotations at scale. All of
them evaluate models with aggregate metrics and none ships a tool for inspecting
failures. On the visual-analytics side, ModelTracker and Squares show per-instance and
per-class performance, Manifold compares models where they disagree, the What-If Tool
probes slices, Calibrate analyzes probabilistic outputs and Domino discovers
underperforming slices automatically. StepLens borrows per-class matrices, model
comparison and slice discovery from this line of work and adds what procedural video
needs: steps aligned in time, task graphs, linked playback, and a hold-out check on every
slice. (Full references in the proposal.)

## 3. Data, and what re-implementing the benchmark revealed

We use the public annotations (step and error labels, task graphs, splits), the released
frozen features (Omnivore, SlowFast, X3D, 3D-ResNet at 1 s; ImageBind video, audio and
text at 2 s) and the 360p GoPro videos of ten recordings. The dataset is released under
Apache 2.0 (project website, downloader and error-recognition repositories; the
annotations repository's README says the same while its LICENSE file is MIT). The feature archive is public
and 114 GB; the 9 GB of segment features are all that the benchmark needs. Four facts
matter for everything that follows:

1. **Missing Step errors are never scored.** Skipped steps carry start and end time −1
   and the official dataloader drops them, so 285 of the 2,566 annotated errors, one of
   the seven categories, are outside the supervised benchmark. StepLens shows them as
   "–" in every model column.
2. **The step split is not random.** The official code shuffles a temporary list and
   then slices the steps in annotation order: train = the first 75% of each recording's
   steps, the phase the code calls "test" = the next 16%, "val" = the last 9%. Their mean
   relative positions in the recipe are 0.35, 0.78 and 0.91. The test set has 557 of its
   799 steps in the last quarter of the recipe and one in the first. We add a seeded
   random split with the same proportions (`step_random`) as a control.
3. **A step can carry several errors** (518 of 1,933 error steps do), and the annotation
   files include an eighth category, Other, with 8 errors.
4. **The public archive has no depth features**, so the paper's four-modality ImageBind
   model cannot be reproduced; ImageBind features cover 383 of 384 recordings.

## 4. StepLens

**Prediction store.** Every trained model writes a per-step score (mean of the per-second
sigmoid outputs, as in the official evaluation) and the per-second scores themselves,
with the phase each step belongs to. The tool reads these files and the processed
annotation tables; adding a model is dropping a folder.

**Views.** (1) The *matrix* has one row per error category plus a row for normal steps,
one column per model; a cell is the share of that category's error steps the model
misses (or, in the last row, the false-alarm rate) with its counts, on a single-hue
sequential scale. (2) The *task-graph view* lists recipes by miss rate and, for one
recipe, draws its DAG of step-order constraints with each step colored by miss rate.
(3) The *timeline* shows one recording's ground-truth steps colored by category, the
outcome of each test step (caught, missed, false alarm), the per-second score curves of
up to two models with their thresholds, and the video, which seeks to a step on click.
(4) The *slice table* enumerates single and pairwise conjunctions of recipe, person,
kitchen, step-duration bucket and recipe-position bucket, optionally crossed with a
category, and reports for each the miss rate with a 95% Wilson interval, the gap to the
overall rate, a binomial z-statistic (the default ranking, which discounts tiny slices),
the AUC inside the slice, and the same slice re-evaluated on the hold-out phase. A
*compare strip* shows, per category, how many errors both models catch, only one
catches, or neither.

**Operating points.** Raw score scales differ by model (the balanced threshold is 0.01
for one transformer and 0.07 for an MLP), so one global threshold would be unfair. Each
model's threshold is chosen on its own validation steps, by a rule the user picks:
balanced (maximize TPR − FPR, the default), best F1, the paper's protocol (min-max
normalize the step scores, threshold 0.6 on step splits and 0.4 otherwise), or one
custom threshold for all.

**Linking.** Every click is a filter (a matrix cell sets category and model, a slice row
sets its attributes, a task-graph node sets the step, a recipe bar opens its graph), and
all views re-render against the same selection. The complete state lives in the URL, so a
finding is a link. Light and dark themes use the same validated palette.

**Implementation.** Python (pandas, PyTorch, FastAPI) and D3 without a build step; 24
prediction sets at the time of writing; headless-browser click tests cover every linked
interaction.

## 5. Reproducing the baselines

We ported the official MLP and transformer heads and training loop (one step per batch,
BCE on every 1-second row, Adam, gradient clipping, model selection by validation AUC).

| Model, official step split, test phase | AUC (ours / paper) | Paper protocol F1 (ours / paper) | Balanced point: recall / false alarm / F1 |
| --- | --- | --- | --- |
| Omnivore MLP, official lr 1e-3 | 68.0 / 75.7 | 3.9 / 24.3 | 75.0 / 50.8 / 52.6 |
| Omnivore MLP, lr 1e-4 | 74.1 / 75.7 | 3.9 / 24.3 | 73.8 / 40.6 / 56.4 |
| Omnivore transformer, official lr 1e-6 | 71.8 / 75.7 | 11.2 / 55.4 | 76.6 / 44.2 / 56.2 |
| Omnivore transformer, lr 1e-5 | 73.3 / 75.7 | 30.7 / 55.4 | 87.7 / 55.9 / 56.7 |
| SlowFast MLP, lr 1e-4 | 68.3 / – | 19.2 / – | 61.5 / 35.1 / 51.8 |
| ImageBind V+A+T transformer, official | 60.5 / – | 14.1 / – | 53.6 / 40.8 / 44.3 |
| Omnivore MLP, lr 1e-4, random step split | 80.1 / – | 3.1 / – | 65.5 / 20.1 / 62.6 |

Three observations. First, with a ten-times smaller learning rate the Omnivore models
land within 2.4 AUC points of the paper; at the official rate the MLP's training loss
never drops below the constant predictor's. Second, the paper's threshold protocol is
brittle: min-max normalization depends on the single highest score, and for all our
models the 0.6 threshold flags almost nothing. We never reproduced the reported F1 of
55.4, while at a balanced operating point every Omnivore model, MLP or transformer,
lands at F1 56–57. Third, the split matters more than the head: the same MLP recipe
scores 74.1 on the official step split and 80.1 on the random one, partly because
selection on the official "val" phase (last steps of each recipe) is unreliable for its
"test" phase (middle steps).

The three-modality ImageBind models fail at every learning rate we tried. The released
embeddings are on different scales (video rows are unit vectors, audio rows have norm
about 20, text rows about 80) and the official code concatenates them as is; but
normalizing them does not rescue the model (61.5 AUC), and single-modality heads show
why: audio alone reaches 59.2 and text 56.2. Without the depth stream, the multimodal
features are weak for this task. Full table: `docs/results.md`.

## 6. Findings

**RQ1a: not by category.** At its balanced point the tuned Omnivore MLP misses 22–28% of
every category (Preparation 27, Measurement 27, Order 28, Timing 25, Technique 27,
Temperature 22); the transformer misses 0–16%. Supervised models catch the Timing and
Temperature errors that the paper's zero-shot models almost never detect, at the price of
35–56% false alarms. SlowFast is the only backbone with a category profile: it misses
Preparation and Technique errors most (45%, 44%) and Temperature least (11%).

**RQ1b: by cook.** Person 2 is the hardest participant for every model on every split
and every seed: missed 65 ± 6% by the Omnivore MLP (three seeds) and 38 ± 5% by the
transformer (three seeds, against 17 ± 4% overall) on the step split, and 61% by the
recording-split transformer on five held-out recordings, with per-person AUC 0.56–0.61
against 0.68–0.74 overall. The walkthroughs (case study 1) show misses that are quantity
and duration deviations, caught errors that involve a visible event.

**RQ1c: by kitchen, with a caveat.** For the tuned MLP on the official step split, kitchen
2 is missed 60% of the time, confirmed on the hold-out phase, with slice AUC 0.65. The same
recipe on the random split and the recording-split transformer do not show it, and
re-training the same recipe with two more seeds moves kitchen 2's miss rate between 23% and
60% (41 ± 15%) while person 2's stays at 65 ± 6%. A slice is a failure mode only when it
survives the hold-out check, a change of model and a change of seed; the tool's hold-out
column catches sampling noise but not run-to-run variance, so we report seeds separately
(`docs/seeds.md`).

**RQ1d: position and compound errors.** On the random split the first and last quarters
of a recipe are missed most (40%, 42% vs 25–31%); the official split cannot show this
because it has 27 early test steps. On held-out recordings, steps with a single error are
missed 30% of the time and steps with two or more errors 18% (Measurement-only 27% vs 5%,
Timing-only 50% vs 18%). Step duration and quantity wording have no consistent effect.

**RQ2: do backbones miss the same errors?** Omnivore MLP vs SlowFast MLP on 252 test
error steps: both catch 124, Omnivore alone 62, SlowFast alone 31, neither 35. A union
would reach 86% recall but also 54% false alarms (both flag 116 normal steps, Omnivore
alone 106, SlowFast alone 76). The complementarity is at the operating point, not in
category-specific skills, and Omnivore features dominate on every head and split.

## 7. Fixes guided by the findings

| Fix | Motivation | Result on test | Why |
| --- | --- | --- | --- |
| Per-kitchen thresholds tuned on validation | kitchen 2 slice | kitchen 2 misses 60% → 4%, its false alarms 23% → 87%; overall F1 56.4 → 55.8 | a threshold moves the operating point inside a slice, it cannot add separation (slice AUC 0.65) |
| Per-recipe thresholds | recipe slices | F1 56.4 → 54.2 | overfits small per-recipe validation sets |
| Per-recording feature centering | kitchen/session shift | AUC 74.1 → 70.4; kitchen 2 AUC 0.65 → 0.63 | the recording mean carries signal the head uses |
| Category upweighting | — | not run | the findings do not point at categories |

The negative results are themselves informative: the models' remaining errors are
single, subtle deviations of quantity, duration and order on recordings of particular
cooks, which per-step appearance features at 360p cannot represent. The fixes that the
findings point to are temporal context across steps (order errors are relative to other
steps) and the depth stream for the multimodal model.

## 8. Evaluating the tool: case studies

We evaluate StepLens the way visual-analytics systems are usually evaluated when a
controlled study is out of scope: by case studies that ask whether the tool surfaces
findings the existing workflow (the paper's aggregate tables) cannot, and whether its
own safeguards tell a real finding from a spurious one. Full walkthroughs with video
links are in `docs/case_studies.md`.

| Case study | View that surfaced it | What the aggregate tables could not show | Safeguard outcome |
| --- | --- | --- | --- |
| 1. A held-out cook (person 2, recordings 25_40 and 27_37) | slice table → timeline + video | one participant is missed 61–69% by every model; the misses are quantity and duration deviations, the catches visible events | holds on hold-out, across splits, across seeds |
| 2. Temperature and Timing errors | matrix | supervised models catch them as well as any category; Missing Step is never scored by anyone | holds for every model |
| 3. Kitchen 2 | slice table (z, hold-out, slice AUC) → compare mode | a 60%-missed slice with a confirmed hold-out that still turned out to be a property of one run | fails the model-switch and seed checks; reported as a non-finding |
| 4. Position in the recipe | slice table on two splits | the official split has one early test step; the random split shows first/last quarters missed most | holds on the random split only |

Two conclusions about the tool itself. First, the slice table alone would have reported
kitchen 2 as a failure mode; only the combination of the hold-out column, the AUC inside
the slice and switching models exposed it, which argues for keeping all three in the
interface. Second, every finding was reached by a chain of at most three clicks and is
reproducible from a link, which is what made the seed study and the cross-split checks
cheap to run. A controlled user study against a notebook baseline was designed
(`docs/user_study/`: protocol, task sheet, SUS, consent, reference answers and an executed
notebook) but not run; it is the natural next step.

## 9. Limitations

Three seeds for the three key MLP recipes and the tuned transformer, one seed for the
rest; across seeds the official-split Omnivore MLP varies by ±2.2 AUC, as large as its gap
to the paper (`docs/seeds.md`); 360p
features released by the authors rather than our own; no depth features; Temperature has
nine test errors; the official step split's biases limit what can be said about
position; the tool is evaluated by case studies rather than a controlled user study; all
numbers come from one dataset and one domain.

## 10. Conclusion

The paper's aggregate numbers hid an excluded category, an unshuffled split, a brittle
threshold protocol and a cook that no model can read. A tool that keeps steps, slices and
video linked, insists on hold-out confirmation, and lets the analyst switch models found
all of them in a day, and also showed which of its own findings not to trust. The next
things to do are a controlled user study, multi-seed confidence intervals for every
model, and models with cross-step context.

## Appendix

* Reproduction commands: `README.md`. All results: `docs/results.md`. Findings log:
  `docs/findings.md`. Case studies with video: `docs/case_studies.md`.
* AI-use disclosure: Claude (Anthropic) was used to draft the proposal, implement the
  system and analysis code, run the experiments, and draft this report under our
  direction; all numbers were produced by the code in the repository.
