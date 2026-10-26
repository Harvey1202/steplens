# Slice survival: flagged on validation, re-checked on test

Each model at its balanced threshold (chosen on validation). Slices: single attributes and pairs over recipe, person, kitchen, step-duration bucket and recipe-position bucket, optionally crossed with one error category; at least 10 validation error steps; flagged when the validation miss rate is above the overall validation miss rate with z ≥ 2. Re-checked on the test steps: *holds* = test miss rate above the overall test miss rate; *significant* = also z ≥ 2 on test; *reverses* = at or below the overall test rate; *untestable* = fewer than 5 test error steps.

| Model | Flagged on val | Testable | Holds on test | of which significant | Reverses | Untestable |
| --- | --- | --- | --- | --- | --- | --- |
| Omnivore MLP, random step split (seed 42) | 9 | 8 | 4 (50%) | 1 (12%) | 4 (50%) | 1 |
| Omnivore MLP, random step split (seed 1) | 8 | 8 | 6 (75%) | 2 (25%) | 2 (25%) | 0 |
| Omnivore MLP, random step split (seed 2) | 11 | 11 | 8 (73%) | 1 (9%) | 3 (27%) | 0 |
| Omnivore MLP, official step split (seed 42) | 12 | 12 | 12 (100%) | 9 (75%) | 0 (0%) | 0 |
| Omnivore transformer, official step split | 5 | 5 | 5 (100%) | 4 (80%) | 0 (0%) | 0 |
| Omnivore transformer, recording split | 9 | 4 | 0 (0%) | 0 (0%) | 4 (100%) | 5 |
| Authors' Omnivore transformer, step split | 4 | 3 | 3 (100%) | 1 (33%) | 0 (0%) | 1 |
| Authors' Omnivore MLP, recording split | 13 | 8 | 5 (62%) | 1 (12%) | 3 (38%) | 5 |
| **All models** | 71 | 59 | 43 (73%) | 19 (32%) | 16 (27%) | 12 |

## Across seeds (same configuration, same phase)

| Pair (random step split, validation phase) | Flagged in A | Flagged in B | Flagged in both |
| --- | --- | --- | --- |
| seed 42 vs seed 1 | 9 | 8 | 4 |
| seed 42 vs seed 2 | 9 | 11 | 4 |
| seed 1 vs seed 2 | 8 | 11 | 7 |

## Examples (random step split, seed 42)

| Slice flagged on validation | val miss (n) | val z | test miss (n) | test z | status |
| --- | --- | --- | --- | --- | --- |
| environment_id=5 & duration_bucket=<15s | 82% (11) | 3.0 | 67% (6) | 1.7 | holds |
| duration_bucket=60-120s & category=Preparation Error | 75% (12) | 2.6 | 25% (4) | -0.4 | untestable |
| duration_bucket=<15s & category=Technique Error | 69% (16) | 2.5 | 29% (17) | -0.4 | reverses |
| position_bucket=Q2 & category=Technique Error | 67% (15) | 2.3 | 39% (23) | 0.5 | holds |
| person_id=1 & position_bucket=Q4 | 65% (17) | 2.2 | 33% (12) | -0.1 | reverses |
| position_bucket=Q4 | 52% (58) | 2.1 | 42% (57) | 1.2 | holds |
| environment_id=5 | 53% (47) | 2.1 | 53% (43) | 2.6 | holds (significant) |
| person_id=1 & environment_id=6 | 70% (10) | 2.1 | 20% (10) | -1.0 | reverses |
| duration_bucket=15-30s & category=Preparation Error | 67% (12) | 2.0 | 27% (15) | -0.6 | reverses |
