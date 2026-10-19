# Task sheet (give one copy per block)

Participant ID: ______   Block: 1 / 2   Interface: StepLens / notebook   Model: ______

## Background (read to the participant)

A model watches cooking videos and scores every recipe step: a high score means "the
cook made a mistake in this step". The dataset labels each step as normal or as an error
of one of seven kinds (preparation, measurement, order, timing, technique, temperature,
missing step), and records which recipe, which person and which kitchen the step comes
from, how long it took and where in the recipe it sits.

The model you are given is not good enough. Your job is to find out **where** it fails.

## Your task (15 minutes)

Write down the model's main failure modes. A failure mode is a group of steps on which
the model does clearly worse than it does overall, for example:

* an error category it misses more than the others,
* a recipe, a person, a kitchen, a step length or a position in the recipe where it
  misses more errors (or raises more false alarms) than average,
* whether the errors it misses are also missed by the other model that is available for
  comparison, or are specific to this model.

Write one finding per line, in this form:

> `<group>` is missed `<X>%` of the time vs `<Y>%` overall (`<n>` error steps); evidence: `<where you saw it>`

Also note anything that makes you distrust a finding (too few steps, does not hold on the
hold-out data, and so on).

You may use any part of the interface. You do not need to find everything; prefer
findings you can support with numbers over guesses. Say "found one" out loud each time
you write a line so the experimenter can note the time.

## Answers

1. ____________________________________________________________________
2. ____________________________________________________________________
3. ____________________________________________________________________
4. ____________________________________________________________________
5. ____________________________________________________________________
6. ____________________________________________________________________

Doubts / comments: ____________________________________________________________
