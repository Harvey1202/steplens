# Reference findings for scoring (experimenter only — do not show participants)

All numbers: test steps of the official step split, each model at its balanced
operating point (threshold chosen on the validation steps), hold-out = the same slice
on the validation steps. Produced with `scripts/case_study_data.py` and the
`/api/slices` endpoint on Oct 18, 2026; regenerate if the models are retrained.

Scoring: **correct** = the participant names the group and the direction (missed more /
false alarms more) with numbers within ±10 points of the reference, or a sub-slice of a
reference slice; **partial** = right group, no numbers or wrong magnitude; **incorrect**
= a group not in the list with a claimed gap > 10 points that the data do not support,
or wrong direction. "Unsupported" claims (no numbers, no group) are not scored. A
participant does not need the composite slices; the single-attribute version counts.

## Model 1: `omnivore_mlp_video_step_lr1e-4` (threshold 0.07)

Overall: misses 26% of error steps (66 of 252), false-alarm rate 41%, hold-out miss 34%.

| # | Finding | Evidence |
| --- | --- | --- |
| R1 | **Kitchen 2** is the main failure slice: 60% missed (32/53) vs 26% overall; hold-out 62%; AUC inside the slice 0.65 vs 0.74 overall | slice table rows 1–2, matrix with kitchen filter |
| R2 | Within kitchen 2 the last quarter of the recipe is worst: 71% missed (22/31), hold-out 60% | slice table row 1 |
| R3 | **Person 2** is missed 69% (11/16), hold-out 59% (person 2 also records in kitchens 1, 5, 6, so it is not only R1) | slice table, person filter |
| R4 | **Categories are flat**: every category is missed 22–28% (Preparation 27, Measurement 27, Order 28, Timing 25, Technique 27, Temperature 22); no category stands out | matrix |
| R5 | Recipe **Spicy Tuna Avocado Wraps** 54% missed (7/13), hold-out 42% | recipes list / slice table |
| R6 | **False alarms concentrate in kitchen 1**: 72% of its normal steps flagged vs 41% overall (and person 1 in kitchen 6: 94%) | slice table sorted by false alarms, matrix last row |
| R7 | Compared with SlowFast (model 2), this model catches 62 errors SlowFast misses and misses 31 SlowFast catches; the union would catch 86%, but both flag the same ~116 normal steps | compare strip |
| R8 | Missing-step errors are never scored by any model (no features) | matrix row "–" |

Non-findings that should be scored **incorrect** if claimed with a big gap: step duration
(<15 s 28%, 15–30 s 37%, >120 s 12%: at most a weak trend), position bucket alone (Q3 24%,
Q4 29%), quantity-worded errors (25% vs 26%).

## Model 2: `slowfast_mlp_video_step_lr1e-4` (threshold 0.02)

Overall: misses 38% of error steps (97 of 252), false-alarm rate 35%, hold-out miss 43%.

| # | Finding | Evidence |
| --- | --- | --- |
| S1 | **Preparation and Technique errors are missed most** (45% and 44%), Order 41%; **Timing (28%) and Temperature (11%, n = 9) are caught best** | matrix |
| S2 | **Person 6** is missed 62% (18/29) vs 38%; hold-out 62% | slice table |
| S3 | **Kitchen 5, last quarter of the recipe**: 75% missed (12/16), hold-out 50%; slice AUC 0.53 | slice table row 1 |
| S4 | **Person 4**: Order errors 59% missed (19/32), Preparation errors 69% (9/13); hold-out 40% / 44% | slice table |
| S5 | Short Preparation errors (15–30 s) 69% missed (11/16), hold-out 45% | slice table |
| S6 | False alarms concentrate in **kitchen 1** (52% vs 35%) and person 7's 30–60 s steps (71%) | slice table sorted by false alarms |
| S7 | Compared with Omnivore (model 1): SlowFast misses 62 errors Omnivore catches and catches 31 Omnivore misses; it has fewer false alarms (35% vs 41%) at a lower recall (62% vs 74%) | compare strip / summary line |
| S8 | Missing-step errors are never scored | matrix |

Non-findings for model 2: kitchen 2 is *not* a failure slice for SlowFast (claiming it is
= incorrect); duration alone and position alone are weak.

## Shared facts both models expose

* The official step split's test steps are almost all from the second half of each
  recipe (Q3 215, Q4 557, Q1–Q2 27 of 799), so "position" findings are confounded.
* Both models' false alarms are high at the balanced point (35–41%); a participant who
  reports the trade-off (recall vs false alarms) gets a partial credit note but it is not
  a failure mode.
