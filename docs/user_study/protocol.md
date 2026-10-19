# StepLens user study protocol

> Status: designed and fully prepared (protocol, task sheet, SUS, consent, reference
> findings, executed baseline notebook) but **not run**. The project evaluates the tool
> through case studies instead (`docs/case_studies.md`, report section 8); this kit is
> kept as an appendix for future work.

**Question.** With StepLens, can people identify a model's main failure modes faster and
more accurately than with a notebook of metric tables?

**Design.** Within-subjects, two conditions, counterbalanced order.

| | Condition A: StepLens | Condition B: notebook |
| --- | --- | --- |
| Interface | the tool at `http://127.0.0.1:8781/` | `docs/user_study/baseline_notebook.ipynb` (pandas + metric tables, same data, no plots) |
| Model under diagnosis | one of two matched models (see below) | the other model |
| Time | 15 minutes | 15 minutes |
| Output | a written list of the model's failure modes with evidence | same |

Each participant diagnoses a *different* model in each condition so that nothing learned
in the first condition can be copied into the second. The two models are matched in
difficulty and each has a known set of failure modes (`reference_findings.md`):

* Model 1: `omnivore_mlp_video_step_lr1e-4` (Omnivore MLP, official step split)
* Model 2: `slowfast_mlp_video_step_lr1e-4` (SlowFast MLP, official step split)

Both conditions use the same operating point (balanced on validation) and the same
phase (test), and both expose the same facts: per-category recall and false alarms, per
recipe / person / kitchen / duration / position rates, hold-out rates, and a second
model to compare with.

**Counterbalancing** (8 participants; with 6, drop P7 and P8):

| Participant | First condition | Model in first | Second condition | Model in second |
| --- | --- | --- | --- | --- |
| P1 | StepLens | 1 | notebook | 2 |
| P2 | notebook | 1 | StepLens | 2 |
| P3 | StepLens | 2 | notebook | 1 |
| P4 | notebook | 2 | StepLens | 1 |
| P5 | StepLens | 1 | notebook | 2 |
| P6 | notebook | 1 | StepLens | 2 |
| P7 | StepLens | 2 | notebook | 1 |
| P8 | notebook | 2 | StepLens | 1 |

**Participants.** 6–8 data science students who have trained at least one classifier.
No prior exposure to CaptainCook4D or StepLens. Recruit from the course; no compensation
beyond snacks; a pilot with one extra participant first to fix wording and timing.

**Session (about 50 minutes).**

1. Consent and background questions (3 min): years of ML experience, familiarity with
   pandas and with visualization tools. `consent.md`.
2. Training (7 min): a 3-minute walkthrough of the first interface on a *third* model
   (`omnivore_transformer_video_step_lr1e-5`), then the participant tries one query.
   The same walkthrough is given for the second interface before its block.
3. Block 1 (15 min): task sheet (`task_sheet.md`). The experimenter records the clock
   time of each finding the participant writes down.
4. SUS questionnaire for the first interface (2 min). `sus.md`.
5. Block 2 (15 min) with the other interface and the other model, then SUS.
6. Debrief (5 min): which interface they preferred and why, what was missing, what they
   did not trust. Audio notes or written notes.

**Measures.**

* *Correct findings*: each line the participant wrote is scored against
  `reference_findings.md` as correct (matches a reference finding, with evidence),
  partial (right direction, wrong magnitude or no evidence), or incorrect/unsupported.
  Primary outcome: number of correct findings; secondary: precision of findings
  (correct / total written).
* *Time to first correct finding* and *time to the third correct finding* (seconds from
  the start of the block).
* *SUS score* per interface (0–100).
* Free-text comments, coded afterwards into themes.

**Analysis.** Paired comparison per participant (StepLens − notebook) on each measure.
With n ≤ 8 report the paired differences, their mean with a bootstrap 95% CI, and a
Wilcoxon signed-rank test; do not rely on p-values alone. Also report per-model results
to check the two models were comparably hard. Two scorers score the findings
independently and reconcile; report agreement (Cohen's kappa) on the correct/incorrect
labels.

**Threats and how we handle them.**

* Order and learning effects: counterbalanced order and different models per block.
* Unequal difficulty of the two models: both have the same number of reference
  findings; results are also reported per model.
* The notebook could be made artificially bad: it exposes exactly the same tables the
  tool computes (`baseline_notebook.ipynb` calls the same backend), so the comparison
  is visualization and linking vs. tables, not information access.
* Experimenter bias in scoring: blind scoring (the scorer does not see which condition
  produced a sheet) and two scorers.

**Materials checklist.** consent.md · task_sheet.md · sus.md · reference_findings.md ·
baseline_notebook.ipynb (executed once, outputs cleared) · a timer · the server running
with the 24 prediction sets · a scoring spreadsheet with one row per written finding.
