# The local learning rule

What updates a cell's adapting surface, given the cell and sheaf contract fixed in
[`01-cell-and-sheaf.md`](./01-cell-and-sheaf.md) and the tick semantics fixed in
[`02-tick-semantics.md`](./02-tick-semantics.md). Settled in
[patchworks#5](https://github.com/NGL321/patchworks/issues/5); see
[ADR-0008](../adr/0008-the-local-rule-splits-by-parameter-not-by-cell.md).

Terms used here are defined in [`CONTEXT.md`](../../CONTEXT.md).

## Two rules, not one

The adapting surface has two parameter groups doing different jobs — biases run inference, restriction
maps run transport (see *The division of labour between the two adapting surfaces*,
[`01-cell-and-sheaf.md`](./01-cell-and-sheaf.md)) — and the local learning rule respects that split
rather than collapsing it. It is **two rules**, each training one parameter group off one of the two
signals already sitting at a cell without any new channel.

### The bias rule

Trains biases on **prediction error**: the difference between what `decode` predicted last tick and the
node stalk the cell reads in as evidence this tick. Because reconciliation edits the node stalk between
the two ticks, this signal already carries whatever the neighbours' disagreement did to the cell's
belief — without the bias rule ever reading a neighbour directly.

The update is a local gradient step through the cell's own frozen forward path — `encode`, `step`,
`decode` — stopped at the per-cell biases the shared body doesn't own. This is a closed backprop
entirely inside one cell's small MLP: no different in kind from ordinary backprop-to-input, and no
different in cost, since the body is small and the pass never leaves the cell.

This is the predictive-coding element of the architecture. It trains inference — the body's operating
point — and never touches a restriction map.

### The transport rule

Trains restriction maps on **disagreement**: already computed during the message-passing phase, per
[ADR-0007](../adr/0007-the-disagreement-floor-is-tolerated-not-represented.md) constrained to learn on
*change* in disagreement or disagreement relative to that edge's own recent scale, never toward a zero
target.

The update is a local gradient step on disagreement, composed **in the same step** with the sparsity
pressure already named in [`01-cell-and-sheaf.md`](./01-cell-and-sheaf.md) — one additive penalty term
inside one descent step, not a second update loop running alongside it.

This trains transport: the basis in which a cell's features become comparable to a neighbour's. It
never reads a neighbour's raw node stalk — only the disagreement already derived from it during
reconciliation — because the whole point of the map is to make two cells' features comparable when they
don't yet share a basis; a raw neighbour stalk is in the wrong space until the map has done that work.

## Locality boundary

Strictly the cell. Both rules read only the chart, node stalk, and per-edge disagreement the cell
already has once reconciliation finishes in the same tick. Nothing is read that the architecture wasn't
already routing to the cell through the ordinary message-passing channel.

## Permitted global signals

Exactly two, both schedule-shaped rather than information-shaped, so neither carries any particular
cell's error across the graph:

- a single global **learning-rate scalar**, mirroring reconciliation's `γ`
- the **sparsity-pressure anneal** already named in [`06-graph-topology.md`](./06-graph-topology.md)

Nothing else broadcasts. A global loss, confidence readout, or any other signal carrying content about
a specific cell's error is explicitly rejected — that would be backprop across the architecture wearing
a different name.

## What information cohomology does not supply

Considered and disqualified as the shape of either rule; recorded in
[`01-cell-and-sheaf.md`](./01-cell-and-sheaf.md#two-cohomologies-which-are-not-the-same-cohomology) and
`docs/research/015-information-cohomology.md`. No variational principle, no graph in the theory, and an
`O(2ⁿ)` cost ceiling that would not batch. Retained only as an interpretive lens.

## Stability under simultaneous, cell-local learning

Settled in [#33](https://github.com/NGL321/patchworks/issues/33); ADR-0007 amended. Within a tick,
both rules already take the same shape ADR-0002 requires of reconciliation — a single local step,
applied after that tick's signals are read — so simultaneity across cells adds no new locality
problem. Across ticks the risk splits by parameter group and neither half needed new mechanism:

- The **bias rule** can leave a mid-depth cell oscillating between activation regions under
  ambiguous evidence rather than settling — the **settling floor**, a third kind alongside static and
  lag in ADR-0007's taxonomy. Bounded by the same `γ` that bounds reconciliation; tolerated, not
  represented, like the other two.
- The **transport rule** doesn't need a bound of its own — it makes ADR-0007's existing
  `γ × floor < fold margin` check go stale, since it trains the magnitudes the `Σ_e m_e` proxy stands
  in for. The check is re-derived locally, per cell, from that cell's own current restriction maps, on
  the same schedule the global learning-rate and sparsity anneals already use, rather than once at
  construction.

Neither [#20](https://github.com/NGL321/patchworks/issues/20)'s change gate nor the probabilistic
sheaf plays a role: confirmed orthogonal, and declined a third time respectively. See ADR-0007,
*Simultaneous learning does not need its own bound*.
