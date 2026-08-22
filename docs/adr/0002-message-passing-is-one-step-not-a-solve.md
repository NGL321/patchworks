---
status: accepted
---

# Message passing is a single per-tick step, not an iterative solve

The obvious design for reconciliation, borrowed from most message-passing GNNs, is to run
several local update rounds per tick until disagreement stabilises, gated by a round-count
hyperparameter or a convergence check. We reject that: the message-passing phase runs **exactly
one** simultaneous local descent step per cell per tick, full stop.

The reason is locality, not economy. Any legitimate stopping rule for an iterative solve —
"stop when disagreement is small enough" — needs some read of disagreement across the graph,
which is exactly the all-to-all aggregation [ADR-0001](./0001-continual-learning-applies-to-the-adapting-surface.md)'s
sibling decisions and the cell-and-sheaf contract already rule out. A fixed round count `R > 1`
sidesteps that specific problem but replaces it with an unmotivated constant nothing else in the
architecture needs. One step is the only choice that adds neither.

## Consequences

- Per-tick compute is bounded by construction — one forward pass per cell, one
  restriction-and-reconcile per edge — with no tunable solver depth to budget for.
- Any settling toward equilibrium, oscillation, or chaos in the graph's more distant regions is
  now necessarily a **cross-tick** phenomenon, arising from the persisted chart and the one-tick
  edge delay together, not a within-tick one. See "Known exposure" in
  [`docs/spec/02-tick-semantics.md`](../spec/02-tick-semantics.md).
- The only remaining lever on how fast disagreement propagates is graph topology (relay cells,
  effective distance), not a solver parameter. A future engineer should reach for topology
  changes, not for reopening this decision to add rounds back.

## Considered and rejected

- **Fixed `R > 1` local descent steps per tick.** Rejected: no principled way to pick `R` that
  isn't itself an arbitrary constant, and it does nothing an extra tick of the recurrence
  doesn't already do more cheaply.
- **Iterate to convergence, capped by a max-rounds guard.** Rejected: detecting convergence
  requires reading total disagreement across the graph, which is a global aggregate — the thing
  graph-locality exists to rule out.
