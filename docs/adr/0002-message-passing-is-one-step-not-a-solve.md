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

## The three grounds

Reaffirmed by [#230](https://github.com/NGL321/patchworks/issues/230) against a shortfall this
decision had never been weighed against: [#214](https://github.com/NGL321/patchworks/issues/214)
measured rim-to-core transmission failing by **1.15e9x**, and `docs/research/150` had already found
**52% of the per-hop loss is the reconciliation gain** — the step size this ADR sets. One step per
tick was the candidate that deficit was going to be spent on, for the second time in this record.

It stands, and it stands on three independent grounds, written here deepest first. The first comes
from what the architecture is for rather than from what a solve would cost.

### 1. A cognitive system is not a function solver

The architecture's motivating objection to free-energy formulations is that a brain is not an
optimiser evaluating an objective to convergence, and that no plausible evolutionary route places one
there. **A tick that ran reconciliation to its fixed point would be exactly that.** One local
relaxation step per tick is a geometry that relaxes, not an optimiser that solves — and the
difference is the architecture's own claim about what a cognitive system is, not a property of this
implementation.

### 2. Uniformity of the clock

Every cell infers one tick ahead on the same clock, so no cell is frozen while its neighbours run.
Heterogeneous internal rollout would break that, and with it the wave picture the architecture is
built on: ripples entering at the rim and crossing a graph whose parts all advance together.

**What this ground forbids is precise, and it is narrower than it looks.** It rules out cells having
**different** rollout lengths. A *uniform* extra sweep count — every cell running `R` steps, the same
`R` everywhere — would not violate it. That variant is refused on ground 1, which it violates
squarely, and on ground 3.

### 3. Cost

A solve trades a local rule for a global loop, and the record has priced that repeatedly.

## Consequences

- Per-tick compute is bounded by construction — one forward pass per cell, one
  restriction-and-reconcile per edge — with no tunable solver depth to budget for.
- Any settling toward equilibrium, oscillation, or chaos in the graph's more distant regions is
  now necessarily a **cross-tick** phenomenon, arising from the persisted chart and the one-tick
  edge delay together, not a within-tick one. See "Known exposure" in
  [`docs/spec/02-tick-semantics.md`](../spec/02-tick-semantics.md).
- **The lever on how fast disagreement propagates is retention, not topology — and not a solver
  parameter under either reading.** What stood here, quoted rather than deleted because the
  reasoning that produced it is worth keeping legible:

  > The only remaining lever on how fast disagreement propagates is graph topology (relay cells,
  > effective distance), not a solver parameter.

  *Superseded by [#230](https://github.com/NGL321/patchworks/issues/230)*, which closed the
  topological family outright on a measurement the record already owned: rim to apex is seven hops
  but only **1.82 unit-resistance edges**
  ([`docs/research/150`](../research/150-effective-resistance-and-the-gauge.md) §1), reducible to at
  best ~1.0 because the patch's own leaf edge is irreducible — **under 2x**, against ~866x per hop
  in the model term. [#214](https://github.com/NGL321/patchworks/issues/214) corroborates from the
  far end: the binding edges are `L6/core—L7/core` and `L1/vision—L2/vision`, not the cut the spec
  predicted. **The deficit is temporal, not spatial.** Every edge costs exactly one tick and a tick
  is one relaxation step — this decision — while every cell's effective timescale is about one tick,
  flat graph-wide at 0.91 at the apex against 0.99 at the rim
  ([`05-timescales.md`](../spec/05-timescales.md)). The perturbation needs seven ticks to cross a
  graph that holds nothing for more than about one, so what a future engineer reaches for is a
  **stability gradient**: retention, the resource
  [ADR-0005](./0005-timescale-is-persistence-not-a-schedule.md) already spends, priced against it in
  [#235](https://github.com/NGL321/patchworks/issues/235).
- **Neither lever is a reason to add rounds back**, which is the half of the superseded sentence that
  survives it unchanged. Slow propagation is not remedied by reopening this decision under the
  topological reading or the temporal one, and the three grounds above are what say so.
- **The topological reading reopens on exactly one condition, and it has a ticket:**
  [#237](https://github.com/NGL321/patchworks/issues/237).
  [`docs/research/150`](../research/150-effective-resistance-and-the-gauge.md) computed the **graph**
  Laplacian's effective resistance and explicitly not the **sheaf** Laplacian's, which can be
  arbitrarily worse in specific directions. So the closure above is measured on the graph and not
  along the learned channel [ADR-0022](./0022-a-hop-is-an-operator-norm-along-a-learned-channel.md)
  names, and this ADR carries its own reopening condition rather than reading as closed forever.

## Considered and rejected

- **Fixed `R > 1` local descent steps per tick.** Rejected: no principled way to pick `R` that
  isn't itself an arbitrary constant, and it does nothing an extra tick of the recurrence
  doesn't already do more cheaply.
- **Iterate to convergence, capped by a max-rounds guard.** Rejected: detecting convergence
  requires reading total disagreement across the graph, which is a global aggregate — the thing
  graph-locality exists to rule out.
