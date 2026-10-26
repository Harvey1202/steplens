# Response to the proposal feedback (Oct 26, 2026)

Proposal grade: 10/10. The feedback has five suggestions for the final paper. For each
one: what we did, the evidence, and the wording we will use. Code references are to the
authors' `error_recognition` repository at commit `35ef284` (Nov 2, 2024) and
`annotations` at `a8a920a`, cloned under `vendor/`.

## 1. Evidence for the claims about the key paper, worded neutrally

### What we added

* **The authors' own checkpoints.** In `error_recognition` issue #1 the first author
  released the checkpoints used for the paper. We loaded eight of them unchanged into our
  heads (`steplens/evaluate_checkpoint.py`, predictions `data/predictions/paper_*`) and
  scored every step on the released features. On our reconstruction of the step split's
  test phase they reproduce the paper's Table 10 under the README protocol:

  | Authors' checkpoint (step split, test) | Accuracy | Precision | Recall | F1 | AUC |
  | --- | --- | --- | --- | --- | --- |
  | Omnivore + MLP, ours / paper | 70.84 / 71.09 | 67.27 / 66.07 | 14.68 / 14.86 | 24.10 / 24.26 | 76.2 / 75.7 |
  | Omnivore + transformer, ours / paper | **69.96 / 69.96** | 52.08 / 51.56 | 59.52 / 59.84 | 55.56 / 55.39 | 75.8 / 75.7 |

  This confirms three things at once: our reconstruction of the test phase is the one the
  paper evaluated on, our feature loading is correct, and the reported numbers come from
  min-max normalized step scores thresholded at 0.6 (the README protocol).

* **Code references for each claim** (permalinks at commit `35ef284`):

  | Claim | Evidence |
  | --- | --- |
  | Skipped steps are not scored | `dataloader/CaptainCookStepDataset.py` L89–90: `if step_start_time < 0 or step_end_time < 0: # Ignore missing steps` → `continue`. 285 Missing Step errors sit on such steps (`annotations`, `step_annotations.json`, start/end = −1). |
  | The step split is taken in annotation order | Same file L146–147: `np.random.shuffle(list(self._normal_step_dict.keys()))` shuffles a new list that is then discarded; L152 sets the proportions `[0.75, 0.16, 0.9]`; L165–189 slice the unshuffled key order, with the middle 16% assigned to `phase == 'test'` and the remainder to `phase == 'val'`. |
  | Thresholds are documented in the README only | `README.md` L31–32 (0.6 step split, 0.4 recordings split); `core/evaluate.py` L99, L134, L160 loop over `[0.5, 0.6]`. The checkpoints above reproduce Table 10 with the README values. |
  | Training settings in the code differ from the paper | `base.py` L180 (BCE without a positive-class weight), L295 (batch size 1), `core/config.py` L70–71 (10 epochs, lr 1e-3). The paper states batch 512, positive weight 1.5, 50 epochs. Issue #3 (Mar 2025) raised this; the first author replied that the paper's configuration is the one used and the code is a reference. |

* **Plot of the relative-position distribution** (`docs/figures/split_positions.png`): under
  the released split the train, "test" and "val" phases have mean relative positions
  0.35, 0.78 and 0.91 (test minimum 0.22); under our seeded random split all three are
  0.48–0.53. A direct consequence, measured on the authors' own Omnivore MLP checkpoint:
  validation AUC 65.6 against test AUC 76.2.

* **Recomputed F1 for Table 8.** The TimeChat Missing Steps row reports P = 2.2 and
  R = 0.4. F1 = 2PR/(P+R) = 2 × 2.2 × 0.4 / 2.6 = 0.68, not the reported 12.4 (which equals
  the same model's Preparation F1 in the same table).

* **The two 75.7 AUCs.** Re-evaluating the released MLP and transformer checkpoints gives
  76.2 and 75.8. The two models do have similar AUCs; we no longer call the identical
  value an inconsistency, only note that the table shows the same number for both.

* **Authors' repository issues.** We read every issue on `error_recognition` (3) and
  `annotations` (5). Only #3 touches our claims (training settings, above). None
  discusses the step-split order, the skipped steps or the thresholds. We did not contact
  the authors.

### A new observation, stated with care

In `core/evaluate.py` two lists of epochs are defined, `val_best_epochs` (L87) and
`test_best_epochs` (L90), and a third, `best_epochs` (L93), is the one used. For the step
split `best_epochs` equals `val_best_epochs`; for the recording, person and environment
splits it equals `test_best_epochs` in 36 of 42 configurations (and both lists agree in
the remaining ones except one). The released checkpoints carry exactly these epochs. If
the list names mean what they say, the non-step results were selected on the test set.
Re-evaluating the alternative epochs changes test F1 in both directions (Omnivore MLP,
recording split: 48.7 at the validation-selected epoch vs 55.6 at the epoch used;
SlowFast MLP: 54.0 vs 52.7), so we cannot say whether the reported numbers are inflated.
We report this as an observation, not as a finding.

### Neutral wording for the final paper

| Proposal wording | Final-paper wording |
| --- | --- |
| "Flaws we found" | "Properties of the released benchmark code that affect how its numbers are read" |
| "The released step split is not random" | "In the released code, the step split assigns each recording's steps to train, test and validation in annotation order (Fig. X); the three phases therefore differ in position within the recipe." |
| "never scores Missing Step errors" | "Steps annotated as skipped have no video segment and are excluded by the released dataloader, so the supervised benchmark does not evaluate the Missing Step category." |
| "Decision thresholds are not fully documented" | "The thresholds that reproduce Table 10 (min-max normalization, 0.6 / 0.4) are given in the README; the evaluation script sweeps 0.5 and 0.6." |
| "small inconsistencies … suggesting a copy error" | "In Table 8, one F1 value (TimeChat, Missing Steps) differs from the value implied by its precision and recall (12.4 vs 0.68)." |

## 2. Avoiding circular case studies: a strong notebook baseline

We strengthened the notebook baseline (`docs/user_study/baseline_notebook.ipynb`): slicing
tables with hold-out columns for every attribute and pair, plus plots of the miss rate by
category, person, kitchen, position, duration and recipe with Wilson intervals, a
person × kitchen heat map and score histograms. Then we asked, finding by finding, which
ones it surfaces.

| Finding | Notebook (slicing + plots) | StepLens |
| --- | --- | --- |
| Categories are missed at similar rates | yes, bar chart | yes, matrix |
| Person 2 is missed far more than average | yes, bar chart with CI | yes, top of the slice table |
| Kitchen 2 at 60% for one run | yes, and it looks just as convincing (CI 47–73%) | yes, with hold-out ✓ and slice AUC |
| Kitchen 2's size is unstable across seeds and models | only by re-running the notebook per model and seed | one-click model switch; seeds compared offline (`docs/seeds.md`) |
| Kitchen 1 raises the most false alarms | yes, in the tables (not in the plots) | yes, sort by false alarms |
| Pairwise slices and their validation-to-test survival | tables yes; survival needs extra code | built into every row |
| Why misses happen (quantity and duration deviations invisible, visible events caught) | no; error descriptions can be read but not seen | timeline + video |
| Single-category errors are missed more than compound ones on held-out recordings | no (needs a derived feature) | not directly; the hypothesis came from watching clips, the number from code |
| Missing Step is never scored | only if the analyst notices the absent rows | "–" row in the matrix |
| The step split's position bias | no | no; found by reading the code |

Our honest reading: a careful notebook finds most single-attribute rate findings. What
the tool adds is (a) validation attached to every slice and fast switching between
models, which is what exposed the instability of kitchen 2, and (b) the link from a
number to the video, which is where the explanations came from. Neither view found the
benchmark properties; reading the code did. The final paper will present the case studies
with this table next to them.

On outside feedback: we will not run a controlled study (IRB). If we collect anything,
it will be a formative, qualitative expert walk-through.

## 3. Small categories, noisy models: how often flagged slices survive

`scripts/slice_survival.py` → `docs/slice_survival.md`, `docs/figures/slice_survival.png`.
For eight models, slices flagged on validation (z ≥ 2, ≥ 10 error steps) were re-checked
on test:

| | Flagged | Testable (≥ 5 test errors) | Holds on test | Also significant on test | Reverses |
| --- | --- | --- | --- | --- | --- |
| All eight models | 71 | 59 | 43 (73%) | 19 (32%) | 16 (27%) |
| Random step split, three seeds | 28 | 27 | 18 (67%) | 4 (15%) | 9 (33%) |
| Official step split (two models) | 17 | 17 | 17 (100%) | 13 (76%) | 0 |
| Recording split (two models) | 22 | 12 | 5 (42%) | 1 (8%) | 7 (58%) |

Across seeds of the same configuration only 4–7 of 8–11 flagged slices are flagged
again. Two readings for the paper: one slice in four reverses on new data, and one in
three survives significantly, so a single flagged slice is a hypothesis, not a finding;
and the official step split's 100% survival is not reassuring, because its validation
and test steps come from the same recordings, so a person or kitchen effect carries
over by construction. We will show the figure and this table, including the reversals.

Applied to our own headline findings:

* **Person 2** holds on every check, including on the authors' four checkpoints (missed
  62% vs 40%, 100% vs 85%, 89% vs 63%, 22% vs 14% under the paper's protocol).
* **Kitchen 2** is more subtle than we first wrote. It is modestly above average on most
  step-split models (our transformer +7 to +11 points on each of its three seeds, the
  authors' transformer +15) but the 60% we first saw was one run (+34 points; 23% on another
  seed). The finding is "kitchen 2 is somewhat harder; the size of the effect is not
  stable", not "kitchen 2 is a coincidence".

## 4. Wording of the novelty claim

New wording: "To our knowledge, none of these datasets ships a tool for inspecting the
failures of the models benchmarked on it. The closest system we know of is ARGUS
(Castelo et al., IEEE VIS 2023), which visualizes the outputs of perception and
step-detection models next to the sensor streams of AR task-guidance sessions, live and
after the fact. StepLens differs in aggregating across hundreds of recordings by error
category and metadata slice and in checking whether a slice is stable." ARGUS will be
added to the references.

## 5. Multi-label steps

What the numbers mean, now stated in the tool and the paper:

* The model makes one error/normal decision per step. A step with several categories
  counts in every category row it carries and is caught or missed in all of them at once,
  so rows overlap: the 252 test error steps contribute 342 category entries.
* Co-occurrence is common: 518 of 1,659 scored error steps carry two or more of the six
  scored categories, and 75% of Temperature steps also carry another category (only two
  test steps are Temperature-only).
* Missing Step is never scored (section 1) and Other (8 errors) is shown separately and
  left out of category claims.
* The matrix now has a switch, *Steps with several categories*: count in every category
  (default) or single-category steps only. With the transformer, per-category recall
  changes by at most 6 points between the two (e.g. Technique 84% → 78%, Timing 89% → 94%),
  so the "categories are similar" finding does not depend on the convention.
