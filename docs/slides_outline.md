# Final presentation outline (12 slides, ~12 minutes)

1. **Title.** StepLens: where and why error-detection models fail on CaptainCook4D.
   Team, course, date. One screenshot of the tool.
2. **The gap.** CaptainCook4D reports one number per model (best: 75.7 AUC, 55.4 F1).
   Which errors, recipes, cooks, kitchens are missed? The paper cannot say. Two RQs.
3. **The data in one slide.** 384 recordings, 5,613 scored steps, 1,933 error steps,
   7 categories + Other. Two things we found by re-implementing: Missing Step is never
   scored; the step split is not shuffled (train/test/val mean position 0.35/0.78/0.91).
4. **StepLens.** The four linked views (matrix, task graph, timeline + video, slice
   table) and compare mode; per-model operating points; hold-out confirmation; URL state.
   Live demo or 60-second screen recording.
5. **Reproduction.** Table: official recipe vs tuned learning rate vs paper (AUC 68.0 →
   74.1 vs 75.7 for the MLP; 71.8 → 73.3 for the transformer). The paper's
   normalize-then-0.6 protocol is brittle (F1 4–31 for us vs 24–55 reported).
6. **RQ1 — where they fail.** Categories are flat (matrix screenshot); failures are by
   cook and recording: person 2 missed 61–69% across splits; compound errors caught,
   single subtle ones missed on held-out recordings.
7. **RQ1 — the kitchen 2 lesson.** Real for one model (60% missed, confirmed on hold-out,
   slice AUC 0.65), gone for the next (random split 0.77, recording split average).
   Rule: a failure mode must survive hold-out *and* a model switch.
8. **RQ2 — do backbones miss the same errors?** Compare strip: Omnivore vs SlowFast,
   both 124 / A-only 62 / B-only 31 / neither 35; union would reach 86% recall at 54%
   false alarms. ImageBind without depth: weak everywhere (AUC ≤ 63), text adds nothing.
9. **Model fixes guided by the findings.** Per-kitchen calibration (moves misses into
   false alarms), feature centering (hurts), category upweighting (not motivated). Honest
   negative results; what would be needed instead (temporal context for order errors,
   depth for multimodal).
10. **Evaluating the tool — case studies.** One row per case study: the view that
    surfaced it, what the tables could not show, which safeguard (hold-out, model switch,
    seed) it passed or failed. Lesson: the slice table alone would have reported kitchen
    2; the safeguards are part of the tool.
11. **Limitations.** Three seeds only for the key models; 360p features only; no depth;
    small test sets for Temperature (n = 9); the official split's biases limit position
    claims; no controlled user study (designed, not run).
12. **Take-aways.** Aggregate numbers hid an excluded category, a biased split and a
    cook the models cannot read; a linked, slice-aware tool found them in hours. Repo
    link, what to do next.
