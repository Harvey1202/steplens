---
geometry: margin=0.55in
fontsize: 10pt
mainfont: "Helvetica Neue"
colorlinks: true
urlcolor: blue
header-includes:
  - \usepackage{titlesec}
  - \titlespacing*{\section}{0pt}{5pt}{2pt}
  - \titleformat{\section}{\normalsize\bfseries}{\thesection}{0.5em}{}
  - \setlength{\parskip}{3pt}
  - \pagestyle{empty}
  - \usepackage{enumitem}
  - \setlist{itemsep=1pt,topsep=2pt,parsep=0pt,leftmargin=14pt}
---

**StepLens: Visual Analytics for Diagnosing Error Detection Models in Procedural Cooking Videos — Project Update**\
Team 17 · Harvey Yang (cy2941@nyu.edu) · Jeremy Hsu (wh2757@nyu.edu) · DS-GA 3001, Fall 2026 · November 3, 2026

# Progress since the proposal

Most of the system was already working at the proposal, because we built it while checking feasibility; we list it first, then what is new.

- **Already working at the proposal.** Annotations and the 9 GB of segment features are processed, and 32 error-recognition models are trained (Omnivore, SlowFast and ImageBind features; MLP and transformer heads; official step, random step and recording splits; three seeds for four configurations). Our Omnivore models reach 74.3 ± 2.2 AUC (MLP) and 73.3 (transformer) with a tuned learning rate, against 75.7 in the paper; with the official recipe the MLP reaches 68.0. All four linked views and compare mode work: category × model matrix, task graph, timeline with 360p video, and a slice table with Wilson intervals, z-ranking, the AUC inside each slice and a hold-out column. Each model's threshold is chosen on its own validation steps, and every view state is a shareable URL.
- **New: the authors' checkpoints.** The authors released the paper's checkpoints in their issue tracker. Scored on our reconstruction of the test phase, they reproduce Table 10 (transformer accuracy 69.96, identical to the paper; F1 55.6 vs 55.4), so our data loading, test split and scoring match the paper's.
- **New: checks on the findings.** Slice survival, the notebook baseline comparison and a multi-label switch in the matrix (below).
- **First findings.** Miss rates vary more across participants than across error categories. For our MLP, category miss rates range from 22% to 28%; with 9 to 143 test errors per category, moderate differences cannot be ruled out. One participant's errors are missed more often than average by every model we checked, including the authors' four (69% vs 26% overall for our MLP). On held-out recordings, a step with one error category is missed more often than a step with several (30% vs 18%), and steps whose only error is timing are missed 50% (our transformer) and 60% (the authors') of the time, on 10 steps. Kitchen 2, flagged at 60% missed for one model, is 7 to 15 points above average across seeds and models.

# Response to feedback

1. **Evidence and neutral wording.** Each claim now cites code lines, a plot of relative position by phase, and the recomputed F1 (0.68, not 12.4), and is reworded, e.g. "the released code assigns steps to phases in annotation order." We checked the claims against the authors' materials (their issues, one of which states that the code's training settings differ from the paper's, and their checkpoints) instead of contacting them, because the checkpoints reproduce Table 10 and answer the questions we would have asked.
2. **Circular case studies.** We built the suggested notebook baseline (slicing tables plus plots) and compared finding by finding. It surfaces most single-attribute findings. The tool adds per-slice validation, fast model switching (which showed that the kitchen effect varies across models) and the link to video, where our explanations came from. We will report this as is.
3. **Slice stability.** On our random step split, the control the proposal named, 28 slices were flagged on validation across three seeds; of the 27 testable, 18 stay above average on test, 4 stay significant and 9 reverse. On the official split every flagged slice holds, but there validation and test share recordings. We will show all of them, including the reversals. Per-category rates get intervals too; with 9 test errors for Temperature, we do not rank categories.
4. **Novelty claim.** Now "to our knowledge", naming ARGUS (Castelo et al., VIS 2023) as the closest system.
5. **Multi-label steps.** A step counts in every category it carries. The matrix states this and can switch to single-category steps; per-category recall moves by at most 6 points.

# Scope and plan

The proposal's must-have and should-have items are done, so the remaining time goes to deeper analysis and richer visualizations.

| Dates | Work |
| ----------------- | ------------------------------------------------------------------------- |
| Nov 3 – 16 | Bring stability into the slice table: for each slice, in how many models and seeds it is flagged |
| Nov 17 – 29 | Embedding view: UMAP of step features colored by person, kitchen and outcome, to check whether step features group by person or recording more than by outcome |
| Nov 30 – Dec 14 | Case studies with video, presentation (Dec 1 or 8), final report (Dec 14) |

# GitHub repository

[github.com/Harvey1202/steplens](https://github.com/Harvey1202/steplens): code, documentation, and a release with all model predictions.
