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
