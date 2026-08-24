# Tick semantics

How a tick actually executes, given the cell and sheaf contract fixed in
[`01-cell-and-sheaf.md`](./01-cell-and-sheaf.md).

Terms used here are defined in [`CONTEXT.md`](../../CONTEXT.md).

## Two phases, run every tick

Every tick is exactly two phases, run in this order. Whatever is outside the sheaf writes its boundary
cells after the second of them (*External writes*, below); that is an ordering, not a phase, and
nothing in the graph computes during it.

1. **Inference phase.** Every cell, simultaneously and independently, runs its forward path —
   `encode` fuses its persisted chart with the node stalk the previous message-passing phase
   left behind, `step` advances that fused chart one tick, `decode` reads off the predicted
   node stalk. No cell reads another cell's state here.
2. **Message-passing phase.** Every cell, simultaneously, restricts its own predicted node
   stalk onto each incident edge and runs one local reconciliation step against the belief its
   neighbour restricted onto that same edge — see *Delay*, below, for which belief that is. The
   result edits the node stalk only, never the chart (per `01-cell-and-sheaf.md`).

The node stalk an inference phase reads is always exactly what the prior message-passing phase
deposited, or what an external write left on top of it (*External writes*, below) — sensory and
motor edges are not a distinct input channel, and neither would a future top-down module be: any
external edge, concrete or abstract, enters and leaves through the same message-passing phase as
any other, so nothing about this contract needs to change to accommodate one.

## External writes

**Whatever is outside the sheaf writes its boundary cells after the message-passing phase, as the last
word in a tick.** The world writes the sensory patch cells and the actuator's efference components; the
drive is written by the human, and later by an internal faculty
([`04-action-and-the-boundary.md`](./04-action-and-the-boundary.md)).

This is not a third phase. Nothing in the graph computes during it and no cell reads another; it is a
statement about **ordering**, and the ordering is what matters. Reconciliation is free to edit any node
stalk, including a boundary cell's, and an external write simply lands afterward and wins.

Two things fall out, neither of which needed machinery:

- **A standing assertion actually stands.** A drive boundary cell is the first cell where reconciliation
  and an outside write want the same components. Without the ordering, eight apex cells disagreeing with
  the drive would erode it every tick and ADR-0009's *the assertion stands forever* would be false.
  With it, disagreement on a drive edge can only ever be reduced by the cell moving — which is exactly
  what makes a drive edge a motor edge.
- **The motor pathway is untouched.** The actuator boundary cell's three *commanded* components are
  written by nobody outside, so reconciliation fills them and the world reads them — which is how a
  command reaches the arm at all. A blanket "boundary stalks are not reconciled" rule would have severed
  this; the ordering rule needs no such exemption, and no per-component bookkeeping.

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

## Reconciliation gain

One descent step needs a step size. It is **per cell**, normalised by that cell's total incident mask
width:

```
gain_v  =  γ / Σ_{e∋v} m_e          with a single global γ ≤ 1
```

`Σ_e m_e` tracks the largest eigenvalue of the cell's local Laplacian block, so this normalisation
**equalises** the effective step across the graph: every cell takes roughly the same descent on its
own local energy regardless of how many edges it sits on. It removes a degree artifact; it is not a
timescale knob and must not become one. A gain deliberately graded by depth would be the explicit
per-cell clock divisor [ADR-0005](../adr/0005-timescale-is-persistence-not-a-schedule.md) rejected,
wearing a different name.

A single global scalar is not sufficient: degrees run from low at the rim to ~6 in the core
([`06-graph-topology.md`](./06-graph-topology.md)), and one constant is either too slow at one end or
unstable at the other. A per-edge gain derived from that edge's recent scale is rejected for the same
reason a tracking baseline is — it needs its own time constant
([ADR-0007](../adr/0007-the-disagreement-floor-is-tolerated-not-represented.md)).

**The bound on `γ`.** A disagreement floor produces a bounded standing offset on the reconciled
component of a node stalk. That offset is a shift in the cell's operating point, and the operating
point selects the cell's activation region and therefore its effective timescale (ADR-0005). So:

> `γ × floor` must stay below the cell's **fold margin** — its distance to the nearest activation
> boundary.

Because `Σ_e m_e` **falls with depth** (`06-graph-topology.md`, *Private dimension is a gradient*),
`gain_v` is largest at the apex, and this bound binds hardest exactly where timescale matters most.
It is a construction-time check, run per cell across the taper and folded into
[#27](https://github.com/NGL321/patchworks/issues/27)'s bias-sampling rig — same sweep, same
afternoon. The drive edges into the apex (`06-graph-topology.md`, *Where the drive attaches*) make
this **slacker** rather than tighter — an extra incident edge lowers `gain_v` — but by about 6%, which
is not enough to lean on. If the apex fails it, `γ` is capped globally by the tightest cell; paying that price
everywhere costs only some reconciliation speed at the rim, which is the cheapest thing in the system
to give up.

**The check does not stay construction-time only.** Per
[#33](https://github.com/NGL321/patchworks/issues/33), the transport rule
([`07-local-learning-rule.md`](./07-local-learning-rule.md)) trains the restriction-map magnitudes
`Σ_e m_e` stands in for, so the proxy can drift away from the block's true spectral radius as training
proceeds. Each cell re-derives its own estimate from its own current incident maps — already-owned
data, no new channel — on the same schedule the global learning-rate and sparsity anneals use, not
every tick. See ADR-0007, *Simultaneous learning does not need its own bound*.

## Known exposure

- **Cross-tick settling, not within-tick settling.** Because message passing is a single step
  and every edge carries delay, any settling toward an equilibrium, an oscillation, or something
  chaotic in the more graph-distant regions is a multi-tick property of the recurrent chart
  interacting with delayed message passing — not something engineered by a round count. Expected
  to be observed during the build, not designed for here. Whether relay-cell topology can be used
  deliberately to shape this (shorten effective distance, change the settling regime) is a
  separate, later question.
