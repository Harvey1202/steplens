# Case studies

Four walkthroughs with the tool. Links open StepLens in the matching state (server on
port 8781); recordings marked *video* have a local 360p GoPro file so the timeline plays
the clip. Thresholds are each model's balanced operating point on its validation steps.

## 1. A held-out cook: person 2 making Pan Fried Tofu (recording 25_40, *video*)

Model: `omnivore_transformer_video_recordings` (Omnivore features, transformer head,
official recipe, recording split). Recording 25_40 is one of the 48 held-out recordings,
so the model never saw this cook, kitchen or recording during training.
[open in StepLens](http://127.0.0.1:8781/?modelA=omnivore_transformer_video_recordings&recording=25_40&person_id=2)

The recording has 10 error steps. The model catches 4 and misses 6, and raises 5 false
alarms on the 9 normal steps:

| Time | Step | Error | Score | Outcome |
| --- | --- | --- | --- | --- |
| 2:38–3:30 | Cut the tofu into large cubes | Measurement: cut small cubes instead of big ones | 0.00 | missed |
| 4:19–4:24 | Add 1/4 tsp salt | Measurement: a tbsp instead of a tsp | 0.01 | missed |
| 5:16–5:24 | Add the tofu cubes to the pan | Technique: dropped a few cubes while adding | 0.09 | caught |
| 5:36–7:20 | Cook 5–6 minutes until browned | Timing: cooked 2–3 minutes | 0.02 | caught |
| 9:57–10:05 | Briefly remove from the heat | Preparation: placed on a non-heat-resistant surface | 0.01 | caught |
| 11:00–11:20 | Drizzle 1 tbsp sesame oil | Measurement: two tbsp instead of one | 0.02 | caught |
| 11:40–11:42 | Return to low heat | Temperature: medium instead of low | 0.02 | caught |
| 11:43–13:00 | Cook 2 minutes | Timing: cooked 1.5 minutes | 0.00 | missed |
| 13:05–13:15 | Flip the tofu | Technique: did not flip all pieces, spilled oil | 0.01 | missed |
| 13:16–14:23 | Cook 2 minutes until darkened | Timing: overcooked | 0.00 | missed |

What the video shows is consistent with the scores: the misses are quantity and duration
deviations that leave the frames looking like a normal step (a tablespoon of salt looks
like a teaspoon at 360p; 1.5 minutes of cooking looks like 2), while the catches involve
a visible event (dropping cubes, a burner knob) or a long step. Note that even the
"catches" are weak: the model's balanced threshold is 0.01 and its scores on this
recording never exceed 0.09, so the per-second track in the timeline is nearly flat and
the decisions rest on tiny differences.

Person 2 is not an accident of this recording. Across every split and head we trained,
this cook is the hardest person in the data:

| Model (test steps) | Person 2 missed | All persons missed | Person 2 AUC | Overall AUC |
| --- | --- | --- | --- | --- |
| Omnivore transformer, recording split (person 2 held out, 5 recordings) | 61% (11/18) | 26% | 0.61 | 0.68 |
| Omnivore MLP lr 1e-4, step split | 69% (11/16) | 26% | 0.60 | 0.74 |
| Omnivore transformer lr 1e-5, step split | 38% (6/16) | 12% | 0.56 | 0.73 |

The second held-out recording of this cook, Cheese Pimiento (27_37, *video*), tells the
same story: 6 of 8 errors missed, including a Technique error (bell pepper chopped into
large pieces with the seeds left in) and a Preparation error (cheese put in the bowl
instead of the cup) that a human spots at once. This is the kind of finding the paper's
aggregate tables cannot show, and it is the first thing to look at when choosing which
recordings to add or re-annotate.

## 2. Do supervised models miss the Temperature and Timing errors that zero-shot models cannot see?

The paper reports that zero-shot video-language models essentially never detect
Technique, Temperature and Timing errors (per-category F1 between 0.6 and 5.1, its
Table 8). Its supervised results have no per-category breakdown. StepLens' matrix gives
one. On the test steps of the official step split, at each model's balanced point:

| Model | Preparation | Measurement | Order | Timing | Technique | Temperature (n = 9) | False alarms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Omnivore transformer lr 1e-5 | 88% | 88% | 87% | 89% | 84% | 100% | 56% |
| Omnivore MLP lr 1e-4 | 73% | 73% | 72% | 75% | 73% | 78% | 41% |
| SlowFast MLP lr 1e-4 | 55% | 63% | 59% | 72% | 56% | 89% | 35% |

(recall per category; a step with two categories counts in both)

So no: the supervised models catch Timing and Temperature errors at least as often as
any other category. For the Omnivore models the per-category recalls are flat within a
few points, which means that *category is not where these models fail*; they fail by
cook and by recording, and they pay for their recall with false-alarm rates of 35–56%.
The matrix also shows the category the benchmark never evaluates at all: Missing Step
(282 steps with start and end time −1, no features, every cell "–").
[open the matrix](http://127.0.0.1:8781/?modelA=omnivore_transformer_video_step_lr1e-5&modelB=slowfast_mlp_video_step_lr1e-4)

## 3. Kitchen 2: a failure mode that is real for one model and gone for the next

For the tuned Omnivore MLP on the official step split, the slice table's first rows are
all kitchen 2: 60% of its error steps are missed (32 of 53) against 26% overall, z = 5.7,
and the hold-out check agrees (62% on the validation steps). The AUC inside the slice is
0.65 against 0.74 overall, so the model separates errors from normal steps worse there;
it is not only a shifted score. [open](http://127.0.0.1:8781/?modelA=omnivore_mlp_video_step_lr1e-4&environment_id=2&recording=20_48)

Two attempts to fix it failed in instructive ways. Per-kitchen thresholds tuned on the
validation steps bring kitchen 2's misses down to 4% but push its false alarms from 23% to
87%, with no change in overall F1 (56.4 → 55.8): a threshold moves the operating point,
it cannot create separation. Subtracting each recording's mean feature (a shift
correction) lowers AUC everywhere (74.1 → 70.4) and does not help kitchen 2 (0.65 → 0.63).

Then the model and seed switches: the same recipe trained on the *random* step split gives
kitchen 2 an AUC of 0.77 against 0.80 overall (kitchen 5 is then the worst); on held-out
recordings the recording-split transformer puts kitchen 2 at the overall level (28%
missed vs 26%); and two more seeds of the same MLP recipe give 23% and about 40%. What
remains is smaller: kitchen 2 is a few points harder on most step-split models (our
transformer +7 to +11 points on each of three seeds, the authors' released transformer
+15). So the 34-point gap was a property of one run, while a modest kitchen effect may be
real. The tool's hold-out column would have passed the large gap; only switching models
and seeds sized it correctly. That is the practical rule we take from this: a slice is a
failure mode when it survives the hold-out check *and* a change of model and seed, and its
size should be quoted from the stable version.

## 4. What the split hides: position in the recipe

The released step split takes the first 75% of each recording's steps for training, the
next 16% as "test" and the last 9% as "val", because its shuffle call acts on a temporary
list. The tool makes the consequence visible in the slice table's position buckets: of
the 799 test steps, 557 are in the last quarter of their recipe, 215 in the third, 26 in
the second and one in the first. Any claim about "early" versus "late" steps on this
split is a claim about 27 steps.

On the random step split (same proportions, seeded shuffle, `step_split_random`) the
test steps are spread evenly (214 / 235 / 174 / 176) and a real pattern appears for the
tuned Omnivore MLP: the first and last quarters are missed most (40% and 42%) and the
middle least (31% and 25%). First steps are preparation and measuring, last steps are
transfers and clean-up; both are short and look alike whether or not they went wrong.
The official split cannot show this because it has almost no early test steps.
[open](http://127.0.0.1:8781/?modelA=omnivore_mlp_video_step_random_lr1e-4&position_bucket=Q1)

A related observation on held-out recordings: steps that carry two or more errors are
caught more often than steps with a single error (18% vs 30% missed for the
recording-split transformer; Measurement-only 27% vs 5%, Timing-only 50% vs 18%). Compound
mistakes look more abnormal; a single subtle one does not. On the step split, where the
model has seen the same recordings in training, this difference disappears.
