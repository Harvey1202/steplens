# Step-feature neighbours: omnivore-center

5333 scored steps; each step is the mean of its omnivore segment features minus its recording mean (1024 dims, PCA to 50). For each step, its 10 nearest neighbours in that space; *agree* = share of neighbours with the same label, *chance* = probability that a random other step shares it. Regenerate with `python -m steplens.embedding`.

| Neighbours share the step's … | agree | chance | lift |
| --- | --- | --- | --- |
| recording | 5.3% | 0.3% | 19.9× |
| person | 24.6% | 13.3% | 1.9× |
| kitchen | 33.7% | 15.3% | 2.2× |
| recipe | 30.0% | 4.5% | 6.6× |
| recipe step | 17.0% | 0.3% | 63.0× |
| has an error | 58.6% | 57.1% | 1.0× |

Among each model's error steps on its test phase (balanced threshold on its validation steps), do neighbours share the outcome (missed vs caught)?

| Model | error steps | missed | agree | chance | lift |
| --- | --- | --- | --- | --- | --- |
| omnivore_mlp_video_step_lr1e-4 | 252 | 26.2% | 67.3% | 61.2% | 1.1× |
| omnivore_transformer_video_step_lr1e-5 | 252 | 12.3% | 83.5% | 78.3% | 1.1× |
| paper_omnivore_transformer_video_step | 252 | 17.5% | 73.8% | 71.1% | 1.0× |
| omnivore_transformer_video_recordings | 242 | 25.6% | 63.3% | 61.7% | 1.0× |
