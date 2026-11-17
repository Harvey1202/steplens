# Step-feature neighbours: omnivore

5333 scored steps; each step is the mean of its omnivore segment features (1024 dims, PCA to 50). For each step, its 10 nearest neighbours in that space; *agree* = share of neighbours with the same label, *chance* = probability that a random other step shares it. Regenerate with `python -m steplens.embedding`.

| Neighbours share the step's … | agree | chance | lift |
| --- | --- | --- | --- |
| recording | 17.0% | 0.3% | 63.7× |
| person | 41.2% | 13.3% | 3.1× |
| kitchen | 77.9% | 15.3% | 5.1× |
| recipe | 32.6% | 4.5% | 7.2× |
| recipe step | 9.2% | 0.3% | 34.0× |
| has an error | 62.3% | 57.1% | 1.1× |

Among each model's error steps on its test phase (balanced threshold on its validation steps), do neighbours share the outcome (missed vs caught)?

| Model | error steps | missed | agree | chance | lift |
| --- | --- | --- | --- | --- | --- |
| omnivore_mlp_video_step_lr1e-4 | 252 | 26.2% | 74.0% | 61.2% | 1.2× |
| omnivore_transformer_video_step_lr1e-5 | 252 | 12.3% | 85.7% | 78.3% | 1.1× |
| paper_omnivore_transformer_video_step | 252 | 17.5% | 74.9% | 71.1% | 1.1× |
| omnivore_transformer_video_recordings | 242 | 25.6% | 67.4% | 61.7% | 1.1× |
