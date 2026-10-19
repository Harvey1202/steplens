# Model results (test phase of each model's own split)

Paper protocol = min-max normalized step scores, threshold 0.6 (step splits) or 0.4 (other splits), as the error_recognition README states. Balanced = threshold maximizing TPR − FPR on the model's validation steps, applied to raw mean per-second scores. Paper columns are Table 10 of Peddi et al. (2024) where a matching row exists.

| model | split | lr | best ep | test AUC | paper AUC | paper protocol: acc / F1 / recall | paper F1 | balanced thr (val) | balanced: recall / false alarm / F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| imagebind_mlp_audio_step_lr1e-4-l2norm | step | 0.0001 | 8 | 59.2 |  | 65.2 / 17.3 / 11.5 |  | 0.19 | 87.7 / 77.1 / 49.4 |
| imagebind_mlp_text_step_lr1e-4-l2norm | step | 0.0001 | 2 | 56.2 |  | 65.2 / 20.6 / 14.3 |  | 0.18 | 79.8 / 72.9 / 47.2 |
| imagebind_mlp_video+audio+text_step_lr1e-4 | step | 0.0001 | 2 | 61.1 |  | 65.7 / 20.8 / 14.3 |  | 0.01 | 91.3 / 82.4 / 49.4 |
| imagebind_mlp_video+audio+text_step_lr1e-4-l2norm | step | 0.0001 | 2 | 61.5 |  | 63.9 / 39.5 / 37.3 |  | 0.13 | 58.7 / 42.5 / 46.8 |
| imagebind_mlp_video_step_lr1e-4 | step | 0.0001 | 2 | 63.0 |  | 65.0 / 41.3 / 38.9 |  | 0.22 | 50.0 / 35.0 / 44.3 |
| imagebind_transformer_video+audio+text_recordings | recordings | 1e-06 | 18 | 57.0 | 65.2 | 63.5 / 10.9 / 6.2 | 56.0 | 0.02 | 61.6 / 53.4 / 48.1 |
| imagebind_transformer_video+audio+text_step | step | 1e-06 | 9 | 60.5 |  | 66.3 / 14.1 / 8.7 |  | 0.03 | 53.6 / 40.8 / 44.3 |
| imagebind_transformer_video+audio+text_step_lr1e-5 | step | 1e-05 | 4 | 58.5 |  | 68.4 / 6.7 / 3.6 |  | 0.04 | 56.3 / 43.8 / 44.9 |
| omnivore_mlp_video_recordings | recordings | 0.001 | 9 | 62.9 |  | 64.4 / 13.7 / 7.9 |  | 0.06 | 31.4 / 17.5 / 38.7 |
| omnivore_mlp_video_recordings_lr1e-4 | recordings | 0.0001 | 24 | 65.6 |  | 64.2 / 6.2 / 3.3 |  | 0.05 | 62.8 / 43.8 / 52.2 |
| omnivore_mlp_video_step | step | 0.001 | 13 | 68.0 | 75.7 | 69.1 / 3.9 / 2.0 | 24.3 | 0.03 | 75.0 / 50.8 / 52.6 |
| omnivore_mlp_video_step_lr1e-4 | step | 0.0001 | 13 | 74.1 | 75.7 | 69.1 / 3.9 / 2.0 | 24.3 | 0.07 | 73.8 / 40.6 / 56.4 |
| omnivore_mlp_video_step_lr1e-4-center | step | 0.0001 | 13 | 70.4 | 75.7 | 70.6 / 17.0 / 9.5 | 24.3 | 0.08 | 67.9 / 37.5 / 54.5 |
| omnivore_mlp_video_step_lr1e-4-s1 | step | 0.0001 | 13 | 77.0 | 75.7 | 69.1 / 3.9 / 2.0 | 24.3 | 0.05 | 83.3 / 44.1 / 59.7 |
| omnivore_mlp_video_step_lr1e-4-s2 | step | 0.0001 | 4 | 71.7 | 75.7 | 68.7 / 1.6 / 0.8 | 24.3 | 0.05 | 68.7 / 37.3 / 55.0 |
| omnivore_mlp_video_step_lr1e-4_cal-activity | step | 0.0001 | 13 | 70.8 | 75.7 | 69.0 / 3.1 / 1.6 | 24.3 | 0.07 | 64.7 / 34.2 / 54.2 |
| omnivore_mlp_video_step_lr1e-4_cal-environment | step | 0.0001 | 13 | 72.7 | 75.7 | 69.2 / 4.7 / 2.4 | 24.3 | 0.07 | 73.0 / 40.8 / 55.8 |
| omnivore_mlp_video_step_random | step_random | 0.001 | 30 | 71.1 |  | 69.8 / 12.4 / 6.7 |  | 0.07 | 67.1 / 34.9 / 55.2 |
| omnivore_mlp_video_step_random_lr1e-4 | step_random | 0.0001 | 40 | 80.1 |  | 69.0 / 3.1 / 1.6 |  | 0.06 | 65.5 / 20.1 / 62.6 |
| omnivore_mlp_video_step_random_lr1e-4-center | step_random | 0.0001 | 37 | 78.2 |  | 69.7 / 10.4 / 5.6 |  | 0.06 | 76.6 / 32.7 / 61.9 |
| omnivore_mlp_video_step_random_lr1e-4-s1 | step_random | 0.0001 | 28 | 78.9 |  | 69.0 / 3.1 / 1.6 |  | 0.06 | 66.7 / 24.1 / 60.9 |
| omnivore_mlp_video_step_random_lr1e-4-s2 | step_random | 0.0001 | 36 | 80.6 |  | 68.7 / 1.6 / 0.8 |  | 0.06 | 56.7 / 15.5 / 59.6 |
| omnivore_transformer_video_recordings | recordings | 1e-06 | 38 | 67.5 |  | 64.4 / 9.8 / 5.4 |  | 0.01 | 74.4 / 53.1 / 55.4 |
| omnivore_transformer_video_step | step | 1e-06 | 39 | 71.8 | 75.7 | 70.2 / 11.2 / 6.0 | 55.4 | 0.01 | 76.6 / 44.2 / 56.2 |
| omnivore_transformer_video_step_lr1e-5 | step | 1e-05 | 14 | 73.3 | 75.7 | 71.2 / 30.7 / 20.2 | 55.4 | 0.01 | 87.7 / 55.9 / 56.7 |
| omnivore_transformer_video_step_lr1e-5-s1 | step | 1e-05 | 13 | 73.3 | 75.7 | 71.3 / 30.0 / 19.4 | 55.4 | 0.02 | 78.6 / 42.4 / 58.1 |
| omnivore_transformer_video_step_lr1e-5-s2 | step | 1e-05 | 23 | 73.3 | 75.7 | 71.7 / 37.6 / 27.0 | 55.4 | 0.01 | 83.7 / 50.5 / 57.1 |
| slowfast_mlp_video_step | step | 0.001 | 25 | 65.7 |  | 68.7 / 9.4 / 5.2 |  | 0.02 | 94.4 / 85.7 / 49.6 |
| slowfast_mlp_video_step_lr1e-4 | step | 0.0001 | 24 | 68.3 |  | 69.5 / 19.2 / 11.5 |  | 0.02 | 61.5 / 35.1 / 51.8 |
| slowfast_mlp_video_step_lr1e-4-s1 | step | 0.0001 | 19 | 67.8 |  | 69.3 / 20.7 / 12.7 |  | 0.03 | 61.9 / 34.7 / 52.2 |
| slowfast_mlp_video_step_lr1e-4-s2 | step | 0.0001 | 17 | 67.1 |  | 68.8 / 14.4 / 8.3 |  | 0.02 | 67.5 / 41.3 / 52.5 |
| slowfast_transformer_video_step | step | 1e-06 | 26 | 64.1 |  | 69.2 / 9.6 / 5.2 |  | 0.03 | 59.9 / 37.7 / 49.6 |
