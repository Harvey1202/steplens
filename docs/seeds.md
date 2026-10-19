# Seed robustness

Models trained with more than one seed (same data, split, recipe). Test phase, balanced operating point tuned on each run's validation steps.

| configuration | seeds | test AUC | recall | false alarm | F1 | overall miss | person 2 miss | kitchen 2 miss | kitchen 2 AUC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| omnivore mlp video · step · lr 0.0001 | 3 | 74.3 ± 2.2 | 75.3 ± 6.1 | 40.6 ± 2.8 | 57.0 ± 2.0 | 24.7 ± 6.1 | 64.6 ± 5.9 | 40.9 ± 15.4 | 66.7 ± 4.0 |
| omnivore mlp video · step_random · lr 0.0001 | 3 | 79.9 ± 0.7 | 63.0 ± 4.4 | 19.9 ± 3.5 | 61.0 ± 1.2 | 37.0 ± 4.4 | 66.7 ± 2.9 | 44.0 ± 12.5 | 76.6 ± 0.7 |
| omnivore transformer video · step · lr 1e-05 | 3 | 73.3 ± 0.0 | 83.3 ± 3.7 | 49.6 ± 5.6 | 57.3 ± 0.6 | 16.7 ± 3.7 | 37.5 ± 5.1 | 25.2 ± 2.4 | 62.8 ± 1.3 |
| slowfast mlp video · step · lr 0.0001 | 3 | 67.7 ± 0.5 | 63.6 ± 2.7 | 37.1 ± 3.0 | 52.1 ± 0.3 | 36.4 ± 2.7 | 25.0 ± 0.0 | 42.1 ± 1.8 | 64.3 ± 0.3 |

Per run:

| run | seed | AUC | F1 | overall miss | person 2 miss | kitchen 2 miss | kitchen 2 AUC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| omnivore_mlp_video_step_lr1e-4-s1 | 1 | 77.0 | 59.7 | 17% | 56% | 23% | 0.72 |
| omnivore_mlp_video_step_lr1e-4-s2 | 2 | 71.7 | 55.0 | 31% | 69% | 40% | 0.63 |
| omnivore_mlp_video_step_lr1e-4 | 42 | 74.1 | 56.4 | 26% | 69% | 60% | 0.65 |
| omnivore_mlp_video_step_random_lr1e-4-s1 | 1 | 78.9 | 60.9 | 33% | 69% | 42% | 0.76 |
| omnivore_mlp_video_step_random_lr1e-4-s2 | 2 | 80.6 | 59.6 | 43% | 69% | 60% | 0.77 |
| omnivore_mlp_video_step_random_lr1e-4 | 42 | 80.1 | 62.6 | 35% | 62% | 30% | 0.77 |
| omnivore_transformer_video_step_lr1e-5-s1 | 1 | 73.3 | 58.1 | 21% | 44% | 28% | 0.61 |
| omnivore_transformer_video_step_lr1e-5-s2 | 2 | 73.3 | 57.1 | 16% | 31% | 25% | 0.62 |
| omnivore_transformer_video_step_lr1e-5 | 42 | 73.3 | 56.7 | 12% | 38% | 23% | 0.65 |
| slowfast_mlp_video_step_lr1e-4-s1 | 1 | 67.8 | 52.2 | 38% | 25% | 43% | 0.65 |
| slowfast_mlp_video_step_lr1e-4-s2 | 2 | 67.1 | 52.5 | 33% | 25% | 40% | 0.64 |
| slowfast_mlp_video_step_lr1e-4 | 42 | 68.3 | 51.8 | 38% | 25% | 43% | 0.64 |
