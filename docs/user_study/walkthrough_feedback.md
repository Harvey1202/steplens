# Classmate walkthrough feedback (should-have item, ~20 minutes per person)

Goal: two or three classmates who work on video models walk through one case study in
StepLens and tell us what the views did and did not help them see. Not a controlled
study; the notes feed the report's evaluation section as qualitative feedback.

## Setup

1. Server running (`scripts/serve.sh`), browser at the case-study link below.
2. Give the participant the one-paragraph background from `task_sheet.md` (what the
   model does, what the labels are).
3. Open **case study 1** (held-out cook, person 2):
   `http://127.0.0.1:8781/?modelA=omnivore_transformer_video_recordings&recording=25_40&person_id=2`

## Walkthrough script (experimenter reads, participant drives)

1. *Matrix:* "Which error category does this model miss most? Is any category special?"
2. *Slice table:* "Find the group of steps this model misses far more than average. Which
   columns tell you whether to trust it?" (expect: hold-out column, AUC inside the slice)
3. *Timeline + video:* "Open recording 25_40. Pick one missed error and one caught error
   and watch the clips. What differs?"
4. *Compare mode:* switch Model B to `omnivore_mlp_video_step_lr1e-4`. "Does the second
   model miss the same steps?"
5. *Model switch:* set Model A to `omnivore_mlp_video_step_random_lr1e-4` and filter
   kitchen 2. "Is kitchen 2 still a problem? What would you conclude?"

## Record (one sheet per participant)

Participant: ______  Background (video models? yes/no, years): ______  Date: ______

| Step | What they found (their words) | Time (s) | What confused them |
| --- | --- | --- | --- |
| 1 Matrix | | | |
| 2 Slice table | | | |
| 3 Timeline + video | | | |
| 4 Compare | | | |
| 5 Model switch | | | |

Three closing questions:

* Which view would you use first on your own model, and why?
* Which number did you not trust, and what would convince you?
* What is missing for a model you work on (different labels, no task graph, longer videos)?

Summarize the sheets in the report as 3–5 bullet points with quotes, naming what changed
in the tool as a result (if anything).
