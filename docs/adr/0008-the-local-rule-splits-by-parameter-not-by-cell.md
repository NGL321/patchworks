# ADR-0008: The local learning rule splits by parameter, not by cell

**Status:** accepted

## Context

The adapting surface has two parameter groups that do different jobs: biases run the predictive-coding
element and set a cell's inference (its operating point in the shared frozen body); restriction maps do
transport — fixing the basis in which a cell's features become comparable to a neighbour's. A single
learning rule serving both would have to pick one target and force the other parameter group to serve
it, or read something not already local to the cell.

Two signals are already sitting at a cell without any new channel: its own **prediction error**
(`decode`'s prediction from last tick against the node stalk it now reads in as evidence — a quantity
already shaped by whatever reconciliation did between the two ticks), and per-edge **disagreement**
(already computed during the message-passing phase, per
[ADR-0007](./0007-the-disagreement-floor-is-tolerated-not-represented.md)).

## Decision

**The local learning rule is two rules, not one, split by parameter group and by which of the two
already-local signals each one trains on:**

- **The bias rule** trains biases on **prediction error**, by a local gradient step through the cell's
  own frozen forward path (`encode`/`step`/`decode`). This is a closed backprop entirely inside one
  cell's small MLP — no different in kind from ordinary backprop-to-input, just stopped at the
  parameters the shared body doesn't own. It is the predictive-coding element, and it trains inference.
- **The transport rule** trains restriction maps on **disagreement**, composed in the same gradient step
  with the sparsity pressure already named in `01-cell-and-sheaf.md`. It trains transport, never
  inference, and a neighbour's raw node stalk is explicitly not an input to it — that would defeat the
  map's purpose, which is to make two cells' features comparable when they don't yet share a basis.

**Locality boundary**: strictly the cell. Both rules read only the chart, node stalk, and per-edge
disagreement the cell already has by the time reconciliation finishes in the same tick — nothing crosses
the graph that the architecture wasn't already routing.

**Permitted global signals**: exactly two, both schedule-shaped rather than information-shaped, so
neither smuggles in a global error channel — a single learning-rate scalar (mirroring reconciliation's
`γ`) and the sparsity-pressure anneal already named in `06-graph-topology.md`. Nothing else broadcasts.

## Consequences

- `CONTEXT.md` gains **Prediction error**, **Bias rule**, and **Transport rule** as vocabulary,
  distinguishing prediction error (cell-owned, temporal) from disagreement (edge-owned, spatial) —
  a distinction the existing glossary's `_Avoid_` lists already implied but never named.
- Sparsity pressure is not a separate mechanism or a second update loop on the restriction maps; it is
  one additive term inside the transport rule's single gradient step.
- **Stability under simultaneous, cell-local learning is explicitly not resolved by this ADR.** It
  entangles with change gating ([#20](https://github.com/NGL321/patchworks/issues/20)) and the
  probabilistic sheaf (held in the map's fog), and is carried forward as its own ticket rather than
  forced here.

## Alternatives considered

- **One rule, one signal**, training both parameter groups off a single combined error. Rejected: biases
  and restriction maps are training different tasks (inference vs. transport), and collapsing them to
  one signal would make one parameter group serve an objective that isn't its own — the same shape of
  mistake `01-cell-and-sheaf.md` already rejected for `step` learning to weight a second correction
  stream.
- **Reading a neighbour's raw node stalk** as an input to the transport rule. Rejected: it defeats the
  restriction map's purpose, which is to make a basis change meaningful; a raw neighbour stalk is in the
  wrong metric space until the map itself has done that work.
- **Hebbian or forward-forward-style updates**, needing no backprop through the cell's own forward path.
  Rejected in favour of gradient descent: the predictive-coding framing already implies a comparison
  between a prediction and an observation, which is exactly a gradient target, and the frozen body's
  forward pass is small enough that a local backprop through it costs nothing a Hebbian rule would have
  saved.
