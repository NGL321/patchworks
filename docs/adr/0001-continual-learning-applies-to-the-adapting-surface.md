---
status: accepted
---

# Continual learning constrains the adapting surface, not every parameter

Patchworks holds a standing constraint that **weights never freeze**: there is no train-then-deploy
split, and reproducibility comes from full state snapshots rather than a freeze toggle. Taken at its
word, that constraint rules out the design chosen for the cell body — a single **cell body** shared by
every cell and frozen, with adaptation confined to per-cell biases and restriction maps.

The constraint was always aimed at **preserving continual-learning capacity**, not at forbidding any
fixed parameter anywhere; the original wording simply came out stronger than the commitment behind it.
We therefore narrow it rather than abandon either. Continual learning governs the surface that
*adapts*. A shared frozen body is **infrastructure**, in the same category as a fixed nonlinearity or a
fixed basis, and does not count as frozen weights. The **adapting surface** — biases and restriction
maps — never freezes, so continual learning holds where it does work.

## A restore is not a reset

Snapshot/restore was introduced here as the *reproducibility* mechanism that replaces a freeze
toggle. [patchworks#23](https://github.com/NGL321/patchworks/issues/23) gave it a second job — a
defined start for the acceptance demo's repeated trials — and that job only holds if restore and
`reset()` are kept apart as different kinds of operation.

`reset()` is **in-band**. It rearranges the world, the agent lives through it, physics time runs on,
and nothing announces it; the agent finds out the way it finds out anything, by being wrong. A
**restore** rewinds the entire state including the adapting surface, so there is no tick at which any
cell could observe one. It never appears in the env's contract and is not an operation the agent is
subject to — it is an experimenter's tool.

That is what lets evaluation have a defined start without weakening the reset-free contract. Every
reset-free system in the cited literature reintroduces a start distribution for evaluation
specifically; the distinction above is how Patchworks does the same thing without an in-band reset
the agent could learn to anticipate. It is also why the protocol in
[`08-the-acceptance-demo.md`](../spec/08-the-acceptance-demo.md) can restore mid-run without that
being a train-then-deploy split arriving through the back door: nothing is frozen, and nothing inside
the graph is aware.

## Consequences

- A future reader encountering a frozen body alongside "weights never freeze" needs this ADR; the
  contradiction is real at the surface and resolved only by this narrowing.
- The narrowing was originally conditional on the frozen-body design surviving. It has
  ([patchworks#14](https://github.com/NGL321/patchworks/issues/14)), so the conditional is discharged.
  The design remains explicitly non-load-bearing, though: the graph, the sheaf, and a predictive
  feed-forward component in each cell are what the architecture rests on, and the freeze is the top rung
  of the flex-priority ladder in [`01-cell-and-sheaf.md`](../spec/01-cell-and-sheaf.md). If a later rung
  is taken and the body unfreezes per-cell, revisit this ADR rather than leaving it standing as a
  general licence to freeze things.

### Amended by [#138](https://github.com/NGL321/patchworks/issues/138): that trigger has fired

The clause above named its own trigger, and the Koopman conversion pulls it. A rung of the ladder is
taken and the body unfreezes per-cell — **for one map of the three.** The amendment discharges the
trigger rather than leaving it hanging, and restates what "the shared frozen body" now means.

**The frozen body is `encode`, nonlinear, and `decode`, linear and gauge-fixed
([ADR-0014](./0014-the-linear-readout-is-gauge-fixed.md)).** What advanced the chart — a frozen
nonlinear `step` — is gone, replaced by `K`, a **per-cell learned linear operator**. So the adapting
surface is now the per-cell biases, the per-cell operators `K`, and the restriction maps.

This is a genuine narrowing of the freeze and it is taken deliberately, not conceded: `K`'s spectrum
is a settable design variable where a frozen random map's Jacobian was not, and that settability is
the whole reason for the conversion. The freeze remains non-load-bearing, and one map of three coming
unfrozen is exactly the ladder behaving as designed.

**The invariant that keeps the split honest, written down rather than left emergent:** *buffers are
the frozen body; parameters are the adapting surface.* The body's weights are registered as buffers,
so no optimiser can reach them and the freeze is enforced by construction; the per-cell surface is
registered as parameters, and the prediction rule's target is *defined* as what was registered rather
than filtered against a list. It is enforced rather than merely asserted — the standing perturbation
test already checks that no cell's update moves when another's parameters do
([ADR-0011](./0011-the-locality-guarantee-is-enforced-not-inherited.md)).
- The motivating thesis is worth recording, because it is what the frozen body is *for*: **highly
  constrained small networks, each solving a specific sub-problem and coupled by this graph and sheaf,
  beat a wider unconstrained network at the same job.** Constraint is the design rather than a
  concession to compute. Cells need not learn different activities — identical machinery suffices,
  provided each cell's metric space is tuned to a separate linear decomposition of the non-linear global
  problem.

## Considered and rejected

- **Take the constraint literally** and rule the frozen body out. Rejected: it would discard the design
  before it has been tested, over a constraint aimed at a different concern (avoiding a
  train-then-deploy split).
- **A slowly-refreshed body** rather than a frozen one. Rejected for now: it honours the constraint
  literally but requires a second, global-ish training process, which sits badly with local rules.
