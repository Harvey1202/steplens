# StepLens findings — first pass (Oct 17, 2026)

What the pipeline and the tool have shown so far. Numbers are on the **test phase of
each model's own split**; the full table is in [results.md](results.md) (regenerate with
`python scripts/results_table.py`). Links open the tool with the matching state
(server on port 8781).

## 1. Reproduction of the paper's baselines

| Model (official step split) | Our AUC | Paper AUC | Paper-protocol F1 (ours / paper) | Balanced operating point (ours): recall / false alarm / F1 |
| --- | --- | --- | --- | --- |
| Omnivore + MLP, official recipe (lr 1e-3, 30 epochs) | 68.0 | 75.7 | 3.9 / 24.3 | 75.0 / 50.8 / 52.6 |
| Omnivore + MLP, lr 1e-4 (40 epochs) | 74.1 | 75.7 | 3.9 / 24.3 | 73.8 / 40.6 / 56.4 |
| Omnivore + MLP, lr 1e-4, random step split (control) | 80.1 | – | 3.1 / – | 65.5 / 20.1 / 62.6 |
| SlowFast + MLP, official recipe | 65.7 | – | 9.4 / – | 94.4 / 85.7 / 49.6 |
| SlowFast + MLP, lr 1e-4 | 68.3 | – | 19.2 / – | 61.5 / 35.1 / 51.8 |
| Omnivore + transformer, official recipe (lr 1e-6, 40 epochs) | 71.8 | 75.7 | 11.2 / 55.4 | 76.6 / 44.2 / 56.2 |
| Omnivore + transformer, lr 1e-5 (30 epochs) | 73.3 | 75.7 | 30.7 / 55.4 | 87.7 / 55.9 / 56.7 |
| ImageBind video+audio+text + transformer, official recipe (lr 1e-6, 40 epochs), step split | 60.5 | – | 14.1 / – | 53.6 / 40.8 / 44.3 |
| ImageBind video+audio+text + transformer, official recipe, **recording split** | 57.0 | 65.3 (with depth) | 10.9 / 56.0 | 61.6 / 53.4 / 48.1 |
| ImageBind video+audio+text + transformer, lr 1e-5 (30 epochs), step split | 58.5 | – | 6.7 / – | 56.3 / 43.8 / 44.9 |
| Omnivore + transformer, official recipe, **recording split** | 67.5 | – | 9.8 / – | 74.4 / 53.1 / 55.4 |
| SlowFast + transformer, official recipe | 64.1 | – | 9.6 / – | 59.9 / 37.7 / 49.6 |
| ImageBind video only + MLP, lr 1e-4 (diagnostic) | 63.0 | – | 41.3 / – | 50.0 / 35.0 / 44.3 |
| ImageBind video+audio+text + MLP, lr 1e-4 (diagnostic) | 61.1 | – | 20.8 / – | 91.3 / 82.4 / 49.4 |
| ImageBind video+audio+text + MLP, per-modality L2 norm | 61.5 | – | 39.5 / – | 58.7 / 42.5 / 46.8 |
| ImageBind audio only + MLP, L2 norm | 59.2 | – | 17.3 / – | 87.7 / 77.1 / 49.4 |
| ImageBind text only + MLP, L2 norm | 56.2 | – | 20.6 / – | 79.8 / 72.9 / 47.2 |

* With the official recipe the Omnivore MLP lands 7.7 AUC points below the paper; with
  a 10× smaller learning rate it lands within 2 points (74.1 vs 75.7). The official
  learning rate (1e-3, Adam, one step per batch) makes the training loss sit *above*
  the trivial constant predictor's loss (0.80 vs 0.62) for the whole run.
* The transformer head at the official learning rate (1e-6) is still improving at epoch
  40 (test AUC 0.61 → 0.72, training loss still falling), so the paper's 75.7 AUC and
  55.4 F1 probably come from a much longer run; the balanced operating point of our run
  gives the same F1 as the tuned MLP (56.2 vs 56.4). At lr 1e-5 the transformer
  reaches 73.3 AUC in 14 epochs (within 2.4 points of the paper) and its raw scores
  are far better calibrated: at the plain 0.5 threshold it gets precision 62% / recall
  25% / F1 35, and the paper protocol gives F1 30.7 instead of 11.2. Even so, the
  paper's F1 of 55.4 at that protocol is not reproduced by any run; at a balanced
  operating point every Omnivore model lands at F1 56–57 regardless of head.
* The three-modality ImageBind transformer does not learn at lr 1e-6: its training loss
  rises from 1.04 to 1.11 over 40 epochs and validation AUC plateaus at 0.59 (best epoch
  9, test AUC 60.5). Raising the learning rate to 1e-5 does not help either (AUC 58.5,
  best epoch 4, loss flat at 1.03–1.05), so the three-modality transformer head with
  the official fixed fusion weights is the weakest model in the tool.
* **Diagnosis.** MLP heads on the same ImageBind features load fine but are weak:
  video only reaches AUC 63.0 and video+audio+text 61.1, so adding audio and text
  *hurts*. The released embeddings are on very different scales: ImageBind video rows
  are unit vectors (norm ≈ 0.95), audio rows have norm ≈ 20 and text rows ≈ 82. The
  official code concatenates them as is, so audio and text dominate the shared encoder
  and the 0.4/0.3/0.3 fusion weights are applied to an already skewed representation.
  Per-modality L2 normalization (`--l2norm`) removes the scale problem but does not
  rescue the model (video+audio+text MLP: 61.1 → 61.5 AUC, still below video only at
  63.0), and the single-modality heads show why: audio alone gives 59.2 and text alone
  56.2, barely above chance. So the released 2-second ImageBind features without depth
  carry little signal for this task, audio adds almost nothing and the LaViLa-narration
  text embeddings nothing; the paper's multimodal gains rest on the depth stream that
  is not in the public archive.
* The paper's evaluation protocol (min-max normalize the step scores over the test set,
  then threshold at 0.6) is brittle: for all of our models it flags almost nothing
  (recall 2–12%), because the normalization depends on the single highest score. The
  tool therefore lets each model choose its own operating point on its validation
  steps (balanced = max TPR − FPR, or best F1) and reports that alongside the protocol
  numbers.

## 2. Problems in the benchmark found while re-implementing it

1. **Missing Step errors are never scored.** Skipped steps carry start/end time −1 and
   the official dataloader drops them, so 285 of 2,566 annotated errors (one of the
   seven categories) are outside the supervised benchmark. The matrix shows them as "–".
2. **The step split is not shuffled.** The code shuffles a temporary list and then slices
   the steps in annotation order: train = first 75% of each recording's steps, the
   phase it calls "test" = next 16%, "val" = last 9%. Mean relative position in the
   recipe is 0.35 / 0.78 / 0.91 for train / test / val, so the phases are different
   populations. Consequences we measured: model selection on "val" picks epoch 13 for
   the tuned MLP although test AUC keeps rising to 0.78 by epoch 29; the random step
   split with the same proportions gives AUC 80.1 for the same recipe. Any "position in
   the recipe" slice on the official split is confounded by this.
3. **Thresholds.** The README's 0.6 / 0.4 appear nowhere in the code; `evaluate.py`
   sweeps 0.5 and 0.6 with min-max normalization on. The sub-step normalization branch
   can never run (`if start - end > 1` is always false).
4. **Public features.** The Box archive (114 GB) has no depth features, so the four-
   modality ImageBind model cannot be reproduced; ImageBind files cover 383 of 384
   recordings (12_6 is absent). The segment features we use are 9 GB.

## 3. Where the tuned Omnivore MLP fails (RQ1)

At its balanced operating point (threshold 0.07 on the mean per-second score):

* **Not by category.** Miss rates are flat across categories: Preparation 27%,
  Measurement 27%, Order 28%, Timing 25%, Technique 27%, Temperature 22% (n = 9).
  The failures of this model are not category-specific.
* **By kitchen.** Kitchen 2 is missed 60% of the time (32 of 53 error steps) against
  26% overall; the same slice is missed 62% on the hold-out ("val") steps; z = 5.7.
  The within-slice AUC is 0.65 against 0.74 overall, so the model separates errors from
  normal steps *worse* there, it is not just a score shift (per-kitchen AUC: k1 0.72,
  k2 0.65, k5 0.78, k6 0.76, k7 0.76). Kitchen 2 × last quarter of the recipe is the
  single worst slice (71% missed, hold-out 60%).
  [open](http://127.0.0.1:8781/?modelA=omnivore_mlp_video_step_lr1e-4&environment_id=2&recording=25_48)
* **By person and recipe.** Person 2 (69% missed, hold-out 59%, per-person AUC 0.60
  against 0.74 overall; this person recorded in kitchens 1, 2, 5 and 6, so it is not
  only the kitchen) and the Spicy Tuna Avocado Wraps recipe (54% missed, hold-out 42%).
* The step split's position bias means the Q3/Q4 slices should be read with care.
* **Does the kitchen-2 finding replicate across splits?** Only partly. It is stable
  within the official step split (test and hold-out phases agree), but the same recipe
  trained on the *random* step split gives kitchen 2 an AUC of 0.77 against 0.80
  overall (kitchen 5 is then the worst at 0.72), and on the recording split kitchen 2
  sits at the overall level (0.65 vs 0.66; kitchen 3 is the worst at 0.58). So the
  slice is a property of this model and split, not of the kitchen itself. That is
  exactly why the tool pairs every slice with a hold-out check *and* lets you switch
  models: a failure mode is only worth acting on when it survives both.

### 3b. Findings that do replicate across splits (Oct 18)

* **Person 2 is the hardest cook for every model on every split.** Omnivore MLP
  (step split): 69% missed, AUC 0.60; Omnivore transformer lr 1e-5 (step split): 38%
  missed, AUC 0.56; Omnivore transformer on the *recording* split, where this person's
  five test recordings are fully held out: 61% missed (11/18), AUC 0.61 against 0.68
  overall. Walkthroughs with video in `case_studies.md` (recordings 25_40 and 27_37).
* **Compound errors are caught, single subtle ones are missed, on held-out recordings.**
  Recording-split transformer, test: steps with one error category 30% missed, steps
  with two or more 18%; Measurement-only 27% vs 5%, Timing-only 50% vs 18%, Order-only
  28% vs 17%. On the step split (same recordings seen in training) the gap vanishes.
* **Position in the recipe, seen only on the random split.** The official test set has
  557 of 799 steps in the last quarter and one in the first. On the random split the
  tuned MLP misses the first and last quarters most (40%, 42%) and the middle least
  (31%, 25%).
* **Kitchen 2 on held-out recordings is average** (28% missed, AUC 0.68), confirming
  that the kitchen-2 slice belongs to one model and split.
* **No consistent effect** of step duration (transformer flat at 9–19%; MLP 12–37% with
  no monotone trend) or of quantity-worded errors (tsp/tbsp/cup: 25% vs 26% missed).

### 3b′. Slice discovery on the random step split (found on val, confirmed on test)

Because the official "val" and "test" phases are different populations, the proposal
says slice discovery and confirmation should use the seeded random step split. For the
tuned Omnivore MLP on that split (balanced threshold 0.06; overall miss 38% on val, 35%
on test), the slices found on the validation steps and confirmed on the test steps are:

| Slice (found on val) | val miss (n) | z | test miss (n) | confirmed |
| --- | --- | --- | --- | --- |
| kitchen 5 | 53% (47) | 2.1 | 53% (43) | ✓ |
| kitchen 5 & step < 15 s | 82% (11) | 3.0 | 67% (6) | ✓ |
| kitchen 5 & Order errors | 64% (14) | 2.0 | 50% (18) | ✓ |
| last quarter of the recipe (Q4) | 52% (58) | 2.1 | 42% (57) | ✓ |
| Q2 & Technique errors | 67% (15) | 2.3 | 39% (23) | ✓ |
| step 60–120 s & Preparation errors | 75% (12) | 2.6 | 25% (4) | ✗ |
| step < 15 s & Technique errors | 69% (16) | 2.5 | 29% (17) | ✗ |

So on the random split the confirmed kitchen is kitchen 5, not kitchen 2 (kitchen 5 also
has the lowest per-kitchen AUC for this model, 0.72), and the recipe-position effect
(Q4) is confirmed. Half of the val-only slices evaporate on test, which is exactly what
the confirmation column is for. Person 2 is not in this model's top-10 by z because it
has only 16 test errors, but its miss rate is 62% [38–81] against 35% overall.

**Bootstrap 95% intervals (500 resamples over error steps) for the headline numbers:**

| Model (test) | Overall missed | Person 2 missed | Kitchen 2 missed |
| --- | --- | --- | --- |
| Omnivore MLP lr 1e-4, official step split | 26% [21–32] | 69% [44–88] | 60% [47–73] |
| Omnivore transformer lr 1e-5, official step split | 12% [9–16] | 38% [12–62] | 23% [11–32] |
| Omnivore transformer, recording split (held out) | 26% [20–31] | 61% [39–83] | 28% [16–41] |
| Omnivore MLP lr 1e-4, random step split | 35% [29–40] | 62% [38–81] | 30% [19–42] |

Person 2's interval excludes the overall rate for every model; kitchen 2's does so only
for the first model.

### 3d. Revisions after the proposal feedback (Oct 26)

* The authors' released checkpoints reproduce Table 10 on our test-phase reconstruction
  (transformer accuracy 69.96 = paper), see `feedback_response.md`.
* Person 2 is missed more than average by all four of the authors' Omnivore checkpoints
  as well (62/100/89/22% vs 40/85/63/14% under the paper's protocol).
* Kitchen 2: the 60% run was an outlier, but kitchen 2 is modestly harder on most
  step-split models (+7 to +15 points), so it is "somewhat harder, unstable size", not a
  pure coincidence.
* Slice survival across eight models: 73% of validation-flagged slices stay above average
  on test, 32% stay significant, 27% reverse (`slice_survival.md`).

### 3c. Seed robustness (Oct 18, `seeds.md`)

Three seeds of each key MLP recipe, test phase, balanced points:

| Configuration | Test AUC | F1 | Person 2 missed | Kitchen 2 missed |
| --- | --- | --- | --- | --- |
| Omnivore MLP lr 1e-4, official step split | 74.3 ± 2.2 | 57.0 ± 2.0 | 65 ± 6% | 41 ± 15% |
| Omnivore MLP lr 1e-4, random step split | 79.9 ± 0.7 | 61.0 ± 1.2 | 67 ± 3% | 44 ± 13% |
| SlowFast MLP lr 1e-4, official step split | 67.7 ± 0.5 | 52.1 ± 0.3 | 25 ± 0% | 42 ± 2% |
| Omnivore transformer lr 1e-5, official step split | 73.3 ± 0.0 | 57.3 ± 0.6 | 38 ± 5% (overall 17 ± 4%) | 25 ± 2% |

Person 2's miss rate is stable across seeds (and across splits); kitchen 2's swings from
23% to 60% between seeds of the same MLP recipe. For the transformer, kitchen 2 is a
consistent but modest effect (25 ± 2% vs 17% overall, slice AUC 0.63 vs 0.73), and the
transformer itself is remarkably seed-stable (AUC 73.3 on all three seeds). The seed study makes the two cases in
section 3 explicit: one is a failure mode of the data, the other of a particular run.
Omnivore's official-split AUC varies by ±2.2 across seeds, which is the same size as its
gap to the paper, so the reproduction should be read as "within seed noise".

## 4. Do the backbones miss the same errors? (RQ2)

Omnivore MLP (lr 1e-4) vs SlowFast MLP (lr 1e-4), both at their balanced points, on
the 252 test error steps: both catch 124, Omnivore only 62, SlowFast only 31, neither
35. A union would catch 86% of the errors, but it would also raise the false-alarm rate
from 41% / 35% to 54% (both flag 116 normal steps, Omnivore only 106, SlowFast only 76).
Per category the pattern is the same (Order: both 67, A-only 36, B-only 18, neither 22),
so the two backbones are complementary mostly through their operating points, not
through category-specific skills.

## 5. Findings-guided model fixes tried

| Fix | Result on test | Verdict |
| --- | --- | --- |
| Per-kitchen thresholds tuned on val (`calibrate.py --by environment_id`) | kitchen 2 misses 60% → 4%, but its false alarms 23% → 87%; overall recall 73.8 → 73.0, false alarm 40.6 → 40.8, F1 56.4 → 55.8 | equalizes *where* the misses happen, no net gain, consistent with the lower AUC inside kitchen 2 |
| Per-recipe thresholds tuned on val | F1 56.4 → 54.2 | overfits the small per-recipe validation sets |
| Per-recording feature centering (`--center recording`), Omnivore MLP lr 1e-4 | AUC 74.1 → 70.4 on the official step split, 80.1 → 78.2 on the random split; kitchen 2 AUC 0.65 → 0.63 while kitchens 1 and 6 drop too | hurts: the per-recording mean carries signal the head uses, and it does not repair kitchen 2 |
| Category upweighting (`--upweight`) | not run | the findings do not point at categories for this model |

## 6. Still running / to do

* All official-recipe runs are done (see the table above and results.md). Tuned
  transformer runs (Omnivore lr 1e-5, ImageBind lr 1e-5) are queued in the background;
  rerun with the commands in the README if the queue was interrupted.
* Across splits, Omnivore features beat SlowFast and three-modality ImageBind on every
  head and split we trained, and the recording split is harder than the step split for
  every backbone (Omnivore transformer 67.5 vs 71.8 AUC), as the paper also reports.
* Kitchen 2 checked on the recording split and across seeds (sections 3b, 3c): not a
  property of the data. Case studies written (`case_studies.md`). The controlled user
  study was dropped in favor of the case-study evaluation; its materials stay in
  `user_study/` as an appendix.
