# Patchworks

*Patches of a manifold, each cell in its own chart. And a quilt of research pieces, assembled by
taste.*

---

Both readings of the name are meant. The first is the architecture: a world too big for any one
predictor, cut into pieces small enough that each can be modelled flat, and stitched back together
by making the seams disagree as little as possible. The second is the method. None of the ideas
here is new on its own. The bet is on the conjunction, and the conjunction is a matter of taste
until it runs.

## The conjunction

Five commitments, none of which is remarkable alone:

- **Small, rigid predictors.** Every cell computes in twelve dimensions and talks in thirty-two —
  narrower inside than out, which is the whole reason it has to model anything.
- **A cellular sheaf, not an atlas.** Each cell has its own basis and its own scale. An atlas would
  demand they all share a dimension; a sheaf is that idea with the requirement removed, and it is
  the sharpest answer to *why a sheaf*.
- **Local rules only.** No gradient crosses a cell boundary. Two rules — one trains a cell's
  inference on its own prediction error, the other trains its transport on how much it disagrees
  with its neighbours — sharing no objective.
- **Two phases a tick.** Predict, then reconcile: exactly one descent step, never a solve.
- **No reward.** Behaviour comes from prediction error, from a single scalar written in from
  outside asserting that something is satisfied when it isn't, and from a human interfering.

Take any one away and the others don't obviously stand up. That is the thesis, and it is why
there is no ablation study in the plan.

## How it was built

**Design first, cite afterwards** — a deliberate inversion, and the one process decision worth
knowing about. Every architectural choice was made from implicit knowledge and written down before
any literature was read. Only then did a citation pass go looking for what the field already knew.

It has been a productive way to be wrong. The literature has contradicted the reason for a
decision far more often than the decision: the argument for uniform cell dimension turned out to
rest on a neuroscience claim nobody makes, the case against relay cells was right for the wrong
reason, and a remembered result about Rao's cross-map transfer turned out not to be in the paper
at all. Each of those is a closed ticket with the correction in it.

It has also turned up two things nobody appears to have done. Getting timescale separation out of
*persistence alone* — no schedule, no gate, no rate parameter per unit — is, as far as fourteen
citation passes can tell, without precedent. And engineering a network's Jacobian spectra to be
**wide** cuts against a literature that spends its time trying to make them narrow.

## What it will be judged by

One live interaction, fixed in advance. A human does three things in sequence — knocks the arm,
teleports a puck, changes the goal — and the measurement is **how long until the first corrective
torque**, per event. Not whether it recovers. How *far in* the recovery had to come from.

The design also says what would falsify it. Structured disagreement that never drains on an edge
means the world is curved where the model assumed it was flat. Behaviour identical across
different tasks means one scalar was never enough to steer 150 cells. An arm that stalls
mid-swing means two routes blended into standing still.

## Where things are

**Nothing is trained.** The design is finished; the build has not started.

- [`docs/spec/`](docs/spec/) — ten files, in reading order. The system, completely specified.
- [`docs/adr/`](docs/adr/) — eleven decisions that needed a reason on the record.
- [`docs/research/`](docs/research/) — the citation passes, including the ones that found defects.
- [`CONTEXT.md`](CONTEXT.md) — the vocabulary. Narrow senses, deliberately.
- [`prototypes/sandbox/`](prototypes/sandbox/) — the world, which runs today.
- [Issue #1](https://github.com/NGL321/patchworks/issues/1) — the map every one of those decisions
  was made on.

```
cd prototypes/sandbox && python watch.py    # live viewer; ctrl-drag a puck to interfere
```
