# Tick semantics

How a tick actually executes, given the cell and sheaf contract fixed in
[`01-cell-and-sheaf.md`](./01-cell-and-sheaf.md).

Terms used here are defined in [`CONTEXT.md`](../../CONTEXT.md).

## Two phases, run every tick

Every tick is exactly two phases, run in this order:

1. **Inference phase.** Every cell, simultaneously and independently, runs its forward path —
   `encode` fuses its persisted chart with the node stalk the previous message-passing phase
   left behind, `step` advances that fused chart one tick, `decode` reads off the predicted
   node stalk. No cell reads another cell's state here.
2. **Message-passing phase.** Every cell, simultaneously, restricts its own predicted node
   stalk onto each incident edge and runs one local reconciliation step against the belief its
   neighbour restricted onto that same edge — see *Delay*, below, for which belief that is. The
   result edits the node stalk only, never the chart (per `01-cell-and-sheaf.md`).

The node stalk an inference phase reads is always exactly what the prior message-passing phase
deposited — sensory and motor edges are not a distinct input channel, and neither would a future
top-down module be: any external edge, concrete or abstract, enters and leaves through the same
message-passing phase as any other, so nothing about this contract needs to change to
accommodate one.

## Delay

An edge costs exactly one tick (`01-cell-and-sheaf.md`). Concretely: at tick `t`, a cell's
message-passing step combines its own restriction, computed this same tick, with its
neighbour's restriction as that neighbour broadcast it during *its* message-passing phase at
`t − 1`. What a cell reconciles against is always one tick stale.

## One step, not a solve

The message-passing phase runs **exactly one** simultaneous local descent step per cell per
tick — not a tunable round count, and not an iterate-to-convergence loop. All cells read the
same prior round's incoming values and update at once (synchronous / Jacobi-style); there is no
visiting order to define. See [ADR-0002](../adr/0002-message-passing-is-one-step-not-a-solve.md)
for why this is fixed rather than a hyperparameter.

Per-tick cost is therefore bounded by construction: one `encode`/`step`/`decode` per cell, plus
one restriction-and-reconcile per edge. No convergence check, no early-stopping logic.

## Known exposure

- **Cross-tick settling, not within-tick settling.** Because message passing is a single step
  and every edge carries delay, any settling toward an equilibrium, an oscillation, or something
  chaotic in the more graph-distant regions is a multi-tick property of the recurrent chart
  interacting with delayed message passing — not something engineered by a round count. Expected
  to be observed during the build, not designed for here. Whether relay-cell topology can be used
  deliberately to shape this (shorten effective distance, change the settling regime) is a
  separate, later question.
