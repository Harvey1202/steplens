# CaptainCook4D Project Proposal (Revised draft)

Oct 17, 2026 · @Harvey

## Overview

**Working title:** StepLens: Visual Analytics for Diagnosing Error Detection Models in Procedural Cooking Videos

**Team:** \[names, NetIDs\] · **Course:** DS-GA 3001, Fall 2026

We will build StepLens, an interactive visual analytics system that shows *where and why* video models fail to detect mistakes in cooking videos. CaptainCook4D gives us 384 egocentric recordings (94.5 hours) with step-level error labels in 7 categories, so we know the ground truth for every miss. The paper reports only aggregate numbers: the best supervised model reaches 75.7 AUC and 55.4 F1 on the step split. Those numbers hide which error types, recipes, steps, people, and kitchens the models miss.

StepLens links four views: an error-category × model performance matrix, recipe task graphs colored by miss rate, recording timelines that align ground-truth errors with per-second model scores and the video, and a slice table that ranks metadata subgroups by error rate. We will use it to compare backbones (Omnivore, SlowFast, ImageBind multimodal). We will check that the failure modes it surfaces hold on held-out data, across models and across training seeds, and we will evaluate the tool through case-study walkthroughs with video evidence.

## Key paper: CaptainCook4D

Peddi et al., [*CaptainCook4D: A Dataset for Understanding Errors in Procedural Activities*](https://proceedings.neurips.cc/paper_files/paper/2024/hash/f4a04396c2ed1342a5d8d05e94cb6101-Abstract.html), NeurIPS 2024 Datasets and Benchmarks Track (UT Dallas, University of Florida).

**What it does.** The paper releases an egocentric 4D dataset of people following recipes, where some recordings contain intentional mistakes. Eight participants recorded 384 videos (94.5 hours) of 24 WikiHow recipes in 10 kitchens. Each person wore a HoloLens 2 (RGB, depth, IMU, head and hand pose, audio) and a head-mounted GoPro (4K, 30 fps).

**Annotations.** There are 5.3K step annotations with start and end times and 10K fine-grained action annotations for 20% of the data. Each recipe also has a task graph, a DAG of step-order constraints. Every step is labeled with errors from 7 categories, and one step can carry several errors at once. The released annotation files also have an eighth catch-all column, Other, which the paper does not report; we will show it separately and exclude it from category-level claims. The counts below are read from the bar chart in the paper's Figure 3, so they are approximate:

| Error category | Annotated errors |
| --- | --- |
| Order | 795 |
| Technique | 502 |
| Preparation | 410 |
| Measurement | 331 |
| Missing step | 285 |
| Timing | 177 |
| Temperature | 66 |

**Benchmarks and results.** The paper defines supervised error recognition, zero-shot error recognition with video-language models, multi-step localization, and procedure learning. For supervised error recognition, it splits each video into 1-second segments, extracts features with frozen backbones (3D-ResNet, SlowFast, X3D, Omnivore, ImageBind), and trains an MLP or transformer head that labels each step as error or normal. The paper defines six data splits (environment, person, recipe, recording, step, and recording type). The best step-split F1 comes from Omnivore with a transformer head: 55.39 F1, 75.7 AUC, 69.96 accuracy. On the harder recording split, multimodal ImageBind (video, audio, depth, text) reaches 56.02 F1 and 65.25 AUC. Zero-shot video-language models do much worse: per-category F1 for Technique, Temperature, and Timing errors is between 0.6 and 5.1. Localization models trained on normal videos also drop sharply on error videos. For example, in the robust multi-step localization benchmark with Omnivore features, mAP at tIoU 0.1 on the environment split falls from 34.65 on normal videos to 12.51 on error videos.

**What we take from it.** We reuse the dataset, its annotations, its step and recording splits, its released features, and its supervised error-recognition pipeline as our ML baseline. We also adopt its 7-category error taxonomy as the main axis of our analysis.

## Critical reading

**Main contributions.** The paper's main contribution is data: errors that are deliberately induced and labeled at the step level, with a category and a free-text description. Most egocentric datasets (Ego4D, EPIC-KITCHENS) record only correct behavior. Synchronized 4D sensing and per-recipe task graphs make the dataset useful for more than error detection.

**General or special case?** The collection protocol (scripted recipes, error scripts, a head-mounted rig) transfers to other procedural domains, such as assembly and lab or medical procedures. The benchmark results are narrower: they cover one domain, 24 recipes, 8 participants, and one geographic area. The Persons split holds out only 2 people.

**Flaws and neat details.**

- Results are reported only as aggregate accuracy, F1, and AUC. Supervised models get no per-category, per-recipe, or per-step breakdown, so we cannot tell which errors are actually caught.
- The supervised benchmark never scores Missing Step errors. Skipped steps are annotated with start and end time −1, and the released dataloader drops them, so 285 of the 2,566 errors (one of the seven categories) are outside the benchmark entirely.
- The released step split is not random. The code shuffles a temporary list and then slices the steps in annotation order, so training takes the first 75% of each recording's steps, the phase it calls "test" the next 16%, and "val" the last 9%. The three phases are therefore different populations (mean relative position in the recipe 0.35, 0.78 and 0.91), which biases any analysis of position in the recipe and makes model selection on "val" unreliable for "test". We add a seeded random step split with the same proportions as a control.
- Some errors follow a script and are partly designed by the authors, so the category frequencies reflect the protocol, not natural behavior. The categories are also imbalanced: Order has 795 errors and Temperature 66.
- Accuracy and F1 disagree sharply on imbalanced labels. Omnivore with an MLP head scores 71.09 accuracy but only 24.26 F1. The repository README states that the reported results use fixed decision thresholds (0.6 for the step split, 0.4 for the recording split), but the released evaluation code only sweeps 0.5 and 0.6 for every split and never uses 0.4. Reported F1 therefore depends on a threshold choice that is not fully documented.
- The tables have small inconsistencies. On the step split, the Omnivore MLP and transformer variants report exactly the same 75.7 AUC. In Table 8, the TimeChat row for Missing Steps reports precision 2.2 and recall 0.4 but F1 12.4, where the formula gives about 0.7. The 12.4 is the same value as that model's Preparation F1, which suggests a copy error.
- A neat detail: the paper evaluates localization models trained only on normal videos against error videos. This shows models break exactly where people deviate from the procedure.

**Fit with related work.** CaptainCook4D sits beside Assembly101 and HoloAssist, which also study mistakes in procedural tasks, and beside HD-EPIC, which adds detailed annotations of real home kitchens. None of these papers provides an interactive tool to diagnose model failures.

**Questions it provokes.** Are failures concentrated in specific error categories or recipe steps? Do different backbones miss the same errors or different ones? Do failures correlate with person, kitchen, or step duration? We cannot answer these questions from the paper's tables, and they motivate our project.

## Research questions and contribution

We ask two questions, each answerable with evidence from the labeled errors:

1. **RQ1: Where do error-recognition models fail?** Which error categories, recipes, steps, and conditions (person, kitchen, step length, position in the recipe) account for most missed errors and false alarms? Are these failures systematic or random?
2. **RQ2: How do models differ?** Do the Omnivore, SlowFast, and multimodal ImageBind models miss the same errors, or do they complement each other?

We plan four contributions that go beyond the paper:

- **StepLens**, an open-source visual analytics system for step-level error-detection models. It combines a category × model matrix, task-graph views of where errors are missed, prediction timelines aligned with video, and metadata slice discovery.
- **A failure analysis of the paper's baselines** that breaks results down by category and by slice. This is the breakdown the paper does not report.
- **A model improvement guided by the findings**, for example category-aware thresholds or reweighting of under-detected categories. We will measure it against the baseline F1 and AUC.
- **A case-study evaluation** of StepLens: walkthroughs that show which failure modes the linked views surface that the paper's tables cannot, and whether those findings survive hold-out, model-switch and seed checks.

## Data

All the data we need is public. The project website, the downloader, and the error-recognition code are released under Apache 2.0. The annotations repository's README also says Apache 2.0, while its LICENSE file is MIT; both are permissive, and we cite the dataset as Apache 2.0 with a note on the discrepancy. The feature archive on Box downloads without a login. It holds 114 GB in total, but the segment features we need are 9 GB (four backbones at 1-second resolution plus ImageBind video, audio and text at 2 seconds); the 105 GB of frame features are not needed. The archive has no depth features, so we reproduce the paper's video, audio and text ImageBind model rather than the four-modality one. We have checked each source below. We do not need raw 4K or 3D sensor data, which runs to several terabytes.

| What we use | Source | Format | Use in project |
| --- | --- | --- | --- |
| Step and error annotations, task graphs, data splits | [annotations repo](https://github.com/CaptainCook4D/annotations) | CSV and JSON | Ground truth, category labels, slices (person, environment) |
| Precomputed features: Omnivore, SlowFast, 3D-ResNet, X3D at 1 s; ImageBind video, audio and text at 2 s (the public archive has no depth features) | [Box archive](https://utdallas.box.com/s/zzuglo0j0loo8ymdsxfr2zzgf9q2jajc) linked from the [feature\_extractors repo](https://github.com/CaptainCook4D/feature_extractors) | 1-second segment features | Train and evaluate models without raw video |
| Baseline training code | [error\_recognition repo](https://github.com/CaptainCook4D/error_recognition) | One shell script per backbone that calls per-task Python training scripts (MLP and transformer heads) | Reproduce the paper's baselines |
| GoPro video at 360p | [downloader](https://github.com/CaptainCook4D/downloader) via Box | MP4 | Video playback in the timeline view |

The 384 recordings include roughly 60 that have only GoPro video and no HoloLens data, and about 40 more that lost the spatial and IMU streams to device instability. ImageBind features exist for 383 of the 384 recordings (12_6 is absent), so we report which recordings each model covers.

## Method

StepLens has two parts: an ML layer that reproduces the paper's error-recognition models, and a visualization layer that turns their predictions into linked views.

&#91;embedded content: StepLens pipeline · data, models, predictions, four linked views\]

**ML layer.** We will train the paper's supervised error-recognition models with the released features and code. These are Omnivore with an MLP head and with a transformer head, SlowFast, and multimodal ImageBind (video, audio, text), each on the step and recording splits. We train each with the official recipe (learning rate 1e-3 for the MLP head and 1e-6 for the transformer head, model selection by validation AUC) and with a tuned learning rate, and we also train on the random step split. For every step we store the predicted error probability and the per-second scores. We also store metadata: recipe, step, error categories, person, kitchen, step length, and position in the recipe. Training uses small heads over frozen features, so it fits on one GPU (NYU HPC or Colab).

**Visualization layer.** The four views are linked, so a selection in one filters the others:

1. **Performance matrix.** Rows are error categories and columns are models. Cells show recall and false-alarm rate. A threshold slider replaces the paper's fixed 0.6 and 0.4 thresholds.
2. **Task-graph view.** Each recipe's DAG is drawn with steps colored by miss rate. This shows whether failures cluster at particular stages, such as measuring or heating.
3. **Timeline and video.** For one recording, ground-truth step segments and error labels sit above each model's per-second scores. Clicking a step plays that clip of the 360p GoPro video.
4. **Slice table.** We enumerate metadata slices (for example, one error category for one person in one kitchen) and rank them by error rate and by lift over the average, with a minimum slice size.

A **compare mode** shows errors caught by one model but missed by another (RQ2). As a stretch goal, we will add a UMAP projection of step features colored by correct versus missed. The stack is a Python backend (PyTorch, pandas, FastAPI) and a D3.js front end.

## Evaluation plan

We will evaluate at three levels: the models, the findings, and the tool.

| Level | What we measure | Baseline | Success looks like |
| --- | --- | --- | --- |
| Models (sanity check) | Accuracy, F1, and AUC on the step and recording splits | Paper's Table 2, e.g. Omnivore transformer head at 75.7 AUC on the step split | Our reproduction lands within about 2 points |
| Findings (RQ1, RQ2) | Per-category recall, false-alarm rate, slice error rate, and model agreement (overlap of caught errors) | Aggregate metrics only, as in the paper | Failure slices found on the validation split also hold on the test split |
| Model fix | F1 and AUC after a change guided by the findings (slice-aware threshold calibration, per-recording feature centering, or reweighting of under-detected categories) | Unchanged baseline model | Higher F1 on the test split, with gains in the slices or categories we targeted |
| Tool | Case studies: whether each linked view surfaces a failure mode the paper's aggregate tables cannot, and whether the finding survives the hold-out, model-switch and seed checks | The paper's aggregate tables | Each case study yields a confirmed, reproducible finding with a shareable view link and video evidence |

**Findings must generalize.** To guard against over-reading noise, we will find slices on the validation split and confirm them on the test split. We will report bootstrap confidence intervals and require a minimum slice size.

**Case studies.** We will write three or four walkthroughs, for example whether supervised models also miss the Temperature and Timing errors that zero-shot models almost never detect, and whether a slice flagged for one model is a property of the data or of that run, with video clips as evidence. We do not run a controlled user study; its materials (protocol, task sheet, questionnaire, reference answers and a notebook baseline) are prepared and kept as an appendix for future work.

## Timeline and risks

The course runs through early December, so we have about eight weeks after this proposal. We split the work into must-have, should-have, and stretch items so the project degrades gracefully.

| Weeks | Dates | Work | Owner |
| --- | --- | --- | --- |
| 1 to 2 | Oct 19 to Nov 1 | Download features and annotations, confirm archive size and license, reproduce the Omnivore baselines on the step split | \[name\] |
| 3 to 4 | Nov 2 to Nov 15 | Reproduce ImageBind multimodal and SlowFast, store per-step predictions and metadata, build the performance matrix and slice table | \[name\] |
| 5 to 6 | Nov 16 to Nov 29 | Timeline and video view, compare mode, task-graph view, failure analysis with validation-to-test confirmation | \[name\] |
| 7 | Nov 30 to Dec 6 | Model fix guided by the findings, seed robustness runs, case-study walkthroughs | \[name\] |
| 8 | Dec 7 to Dec 13 | Case studies, final report, and presentation | all |

**Must-have.** Omnivore (MLP and transformer heads) and ImageBind multimodal baselines, the performance matrix, the slice table, the timeline and video view, compare mode, and the per-category and per-slice failure analysis (RQ1, RQ2).

**Should-have.** SlowFast baseline, the task-graph view, the threshold-based model fix, and multi-seed confidence intervals for the key models.

**Stretch.** The UMAP projection of step features and reweighting-based model fixes.

**Risks and fallbacks.**

- The feature archive is too large or does not download. Fallback: use Omnivore features only, or extract Omnivore features ourselves from the 360p GoPro video.
- Our reproduction lands more than 2 points from the paper. Fallback: report our own numbers, document the differences (thresholds, seeds, head configuration), and use our models for the analysis anyway.
- Video playback is slow or hosting is a problem. Fallback: pre-cut clips for each step and serve them as static files.

## Related work

**Mistakes in procedural video.** Assembly101 \[2\] and HoloAssist \[3\] record mistakes during assembly and assisted tasks. PREGO \[4\] detects procedural mistakes online, as they happen. HD-EPIC \[5\] annotates real home kitchens in fine detail, including recipes, ingredients, and gaze, and benchmarks video-language models on question answering. Ego4D and Ego-Exo4D \[6, 7\] are large-scale egocentric datasets with keystep annotations. All of these works evaluate models with aggregate metrics. None of them provides a tool for inspecting failures.

**Visual analytics for model diagnosis.** Hohman et al. \[8\] survey visual analytics for deep learning. ModelTracker \[9\] and Squares \[10\] show per-instance and per-class performance for classifiers. Manifold \[11\] compares models to find where they disagree, and the What-If Tool \[12\] probes model behavior on data slices. Calibrate \[13\], which became NYU VIDA's PyCalibrate library, analyzes probabilistic model outputs. Domino \[14\] automatically discovers data slices where models underperform. StepLens borrows per-class matrices, model comparison, and slice discovery from this line of work. It adds what video procedures need: steps aligned in time, recipe task graphs, and linked video playback.

**Our gap.** Earlier tools target static classification on images or tabular data. StepLens targets temporal, procedure-structured errors, where the order of steps and their timing are part of the label.

## AI use disclosure

As the course AI policy requires, we disclose that we used Claude (Anthropic) to help prepare this proposal. It helped us brainstorm the topic, summarize the CaptainCook4D paper, check that the dataset and code are available, and draft the text, tables, and the system diagram. We also used it to fact-check the proposal against the paper and the released repositories.

## References

1. R. Peddi et al. [CaptainCook4D: A Dataset for Understanding Errors in Procedural Activities](https://proceedings.neurips.cc/paper_files/paper/2024/hash/f4a04396c2ed1342a5d8d05e94cb6101-Abstract.html). NeurIPS 2024, Datasets and Benchmarks Track. Code and data: [github.com/CaptainCook4D](https://github.com/CaptainCook4D).
2. F. Sener et al. [Assembly101: A Large-Scale Multi-View Video Dataset for Understanding Procedural Activities](https://openaccess.thecvf.com/content/CVPR2022/html/Sener_Assembly101_A_Large-Scale_Multi-View_Video_Dataset_for_Understanding_Procedural_Activities_CVPR_2022_paper). CVPR 2022.
3. X. Wang et al. [HoloAssist: An Egocentric Human Interaction Dataset for Interactive AI Assistants in the Real World](https://arxiv.org/abs/2309.17024). ICCV 2023.
4. A. Flaborea et al. [PREGO: Online Mistake Detection in PRocedural EGOcentric Videos](https://openaccess.thecvf.com/content/CVPR2024/html/Flaborea_PREGO_Online_Mistake_Detection_in_PRocedural_EGOcentric_Videos_CVPR_2024_paper.html). CVPR 2024.
5. T. Perrett et al. [HD-EPIC: A Highly-Detailed Egocentric Video Dataset](https://arxiv.org/abs/2502.04144). CVPR 2025.
6. K. Grauman et al. [Ego4D: Around the World in 3,000 Hours of Egocentric Video](https://openaccess.thecvf.com/content/CVPR2022/html/Grauman_Ego4D_Around_the_World_in_3000_Hours_of_Egocentric_Video_CVPR_2022_paper.html). CVPR 2022.
7. K. Grauman et al. [Ego-Exo4D: Understanding Skilled Human Activity from First- and Third-Person Perspectives](https://openaccess.thecvf.com/content/CVPR2024/html/Grauman_Ego-Exo4D_Understanding_Skilled_Human_Activity_from_First-_and_Third-Person_Perspectives_CVPR_2024_paper.html). CVPR 2024.
8. F. Hohman, M. Kahng, R. Pienta, D. H. Chau. [Visual Analytics in Deep Learning: An Interrogative Survey for the Next Frontiers](https://arxiv.org/abs/1801.06889). IEEE TVCG 2019.
9. S. Amershi et al. [ModelTracker: Redesigning Performance Analysis Tools for Machine Learning](https://www.microsoft.com/en-us/research/publication/modeltracker-redesigning-performance-analysis-tools-for-machine-learning/). CHI 2015.
10. D. Ren et al. [Squares: Supporting Interactive Performance Analysis for Multiclass Classifiers](https://www.microsoft.com/en-us/research/publication/squares-supporting-interactive-performance-analysis-multiclass-classifiers/). IEEE TVCG 2017.
11. J. Zhang et al. [Manifold: A Model-Agnostic Framework for Interpretation and Diagnosis of Machine Learning Models](https://arxiv.org/abs/1808.00196). IEEE TVCG 2019.
12. J. Wexler et al. [The What-If Tool: Interactive Probing of Machine Learning Models](https://research.google/pubs/the-what-if-tool-interactive-probing-of-machine-learning-models/). IEEE TVCG 2020.
13. P. Xenopoulos et al. [Calibrate: Interactive Analysis of Probabilistic Model Output](https://arxiv.org/abs/2207.13770). IEEE TVCG 2023 (IEEE VIS 2022).
14. S. Eyuboglu et al. [Domino: Discovering Systematic Errors with Cross-Modal Embeddings](https://arxiv.org/abs/2203.14960). ICLR 2022.
