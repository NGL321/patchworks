# The cell and its sheaf

The contract every node of the Patchworks graph satisfies, and the sheaf structure that couples
them. Everything downstream of this section inherits it.

Terms used here are defined in [`CONTEXT.md`](../../CONTEXT.md).

## Three tiers

A cell is not identified with its stalk. There are three distinct spaces, and keeping them distinct
is the central commitment of this section:

| Tier | Space | Dim | Owner | Touched by reconciliation |
|---|---|---|---|---|
| 1 | **Chart** — the cell's working coordinates | `k` | the cell, privately | never |
| 2 | **Node stalk** — the cell's public face | `n` | the graph | yes |
| 3 | **Edge stalk** — shared with one neighbour | `m` | the edge | yes |

The reason for three rather than two: Patchworks decomposes one hard problem into many small ones
and then recomposes them. Solving and recomposing are different operations, so they get different
spaces. A cell's inference happens in its chart and is never edited from outside; agreement between
cells happens in stalks and is never confused with inference.

This also makes two things expressible that a two-tier scheme cannot represent:

- **Features private to a cell's sub-problem** — node stalk components that participate in no edge.
- **Relay cells** — cells with stalks and restriction maps but no inference, providing a shared
  metric space for distant cells. Incoherent if the stalk *were* the internal representation.

## The cell

### Forward path

The cell's forward path factors in three parts:

```
encode:  chart (k) × node stalk (n) → chart (k)      fuses persisted state with new evidence
step:    chart (k)                  → chart (k)      the prediction
decode:  chart (k)                  → node stalk (n)
```

`encode` takes the persisted chart as well as the incoming node stalk: it fuses the cell's
prior belief with new evidence into a single chart, which `step` alone advances. See
[`02-tick-semantics.md`](./02-tick-semantics.md) for why (patchworks#4) — in short, `step`
already committed to a single argument, and reconciliation's corrections are meant to re-enter
as evidence rather than as a second stream `step` has to learn to weight.

**`k < n`, fixed by construction.** This is the low-dimensional requirement, and it is a shape
invariant no training story may violate. *Which* features occupy the `k` chart dimensions is
entirely learned, and they need not correlate with any exposed feature — the chart is a compressed
set derived from the node stalk, not a subset of it.

`n` and `k` are **global constants**, identical for every cell. `n` is fixed on a canonical-microcircuit
rationale: a cortical column has an efficient size, and cells are the analogue. The *degree* of
compression (`n/k`) is a hyperparameter; the spec commits to `k < n` and nothing more. That a useful
`k` turns out to be much smaller than `n` is a finding the proof-of-concept reports, not a number
fixed here.

### What a cell predicts

**One prediction: the temporal one.** The cell advances its chart one tick, `z(t) → ẑ(t+1)`, and
that forward state is decoded to the node stalk and restricted onto every incident edge. Edge
predictions are the shadow of the temporal prediction, never independent heads — the guard that
stops a cell from learning to model its neighbours instead of its own sub-problem.

Because edges carry unit delay (below), what arrives on an edge is a *past* state of a neighbour.
Transport is therefore a real channel with its own structure, and predicting what will arrive is
genuine modelling rather than a redundant recomputation.

### State and persistence

**The chart persists across ticks.** It is the cell's state; `step` moves it. The cell is therefore
a recurrent unit, and inherits the classical failure modes of one — see *Known exposure* below.

The **adapting surface** — biases and restriction maps — persists and never freezes (see
[ADR-0001](../adr/0001-continual-learning-applies-to-the-adapting-surface.md)). The rest of the
cell body does not adapt at all; see *The cell body* below.

### Uniformity

Cells are **uniform in contract**: same interface, same algorithm, same `n`, same `k`. A cell's
individuality is not in its machinery but in **what its features mean**, which is fixed entirely by
its restriction maps and biases. This is the precise sense in which each cell works in its own
metric space, and it is compatible with every cell running identical machinery.

**Schedule** is the only per-cell freedom this section grants. Whether cells need differing clock
rates at all is deferred to the multi-timescale section.

A relay cell is the degenerate instance of the contract: `step` is the identity.

### The cell body

Uniformity above is taken in its strongest available form. The **cell body** — the `encode` / `step` /
`decode` machinery of the forward path — is **one set of weights, shared by every cell and frozen**.
It never adapts. All adaptation lives in the **adapting surface**: per-cell biases, and the
restriction maps.

**The restriction maps carry the specialisation.** This is the load-bearing half of the claim and the
half that is easiest to under-read. Each cell learns its own linear map into each incident edge stalk,
independently at both ends of every edge, under a sparsity pressure and a structural mask. That is a
substantial, genuinely per-cell surface — not a thin residue left over after freezing the body. Cells
do not need to learn *different activities*: identical machinery is sufficient, and arguably desirable,
provided each cell's metric space is tuned to a separate linear decomposition of the highly non-linear
global problem. The decomposition specialises the cell; the machinery does not have to.

**`step` is a feed-forward map**, a single forward pass — not a descent flow run across a fixed surface.
Whatever geometry the body's solution space turns out to have is a property of the trained map, never an
inner loop at inference time. A per-tick inner solve would contradict
[ADR-0002](../adr/0002-message-passing-is-one-step-not-a-solve.md), cost unbounded compute, and blur
*one prediction, the temporal one*.

**Initialisation is a parameter of the body, not a commitment of this spec.** The proof-of-concept runs
a **random, non-degenerate** initialisation: the reservoir-computing precedent — fixed internal
dynamics, thin adapting readout — applies directly and requires no corpus to be invented. Pretraining
the body on synthetic prediction tasks is a documented swap-in, not the baseline. If a randomly
initialised frozen body works, the pretraining claim was never needed; if it does not, there is then a
specific reason to build the corpus. See [patchworks#13](https://github.com/NGL321/patchworks/issues/13)
for what the literature does and does not support here.

**Execution is batched.** One shared frozen body means every cell's forward pass is the same operator,
so the whole graph's inference phase is a single batched evaluation rather than one pass per cell. This
is the design's strongest concrete argument for the single-consumer-GPU constraint, and it is a
commitment rather than an implementation detail: the runtime-heterogeneity option the map holds in
reserve would cost it.

**Why constrain this hard.** The thesis is that highly constrained small networks, each solving a very
specific sub-problem and coupled by this graph and sheaf, beat a wider unconstrained network at the same
job. Constraint is the design, not a concession to compute — and constraint has *depth*. A shared frozen
body is its most rigid setting. The *Flex priority* ladder below loosens rigidity one rung at a time
without ever leaving "constrained": every rung keeps the size constraints and the connectivity
constraints identical.

**What is load-bearing, and what is not.** The graph, the sheaf, and a predictive feed-forward component
in each cell are load-bearing. The freeze and the sharing are not. They are the top rung of the ladder,
and dropping them costs no other part of the architecture.

## The sheaf

### Edge stalks carry belief

An edge stalk carries a belief about a latent variable both endpoint cells are modelling in common.
It has **no committed semantics** — it is not a message, not a prediction of the neighbour's state,
and it carries **no error channel**. Error is derived, never transported.

### Restriction maps

Each cell holds one restriction map per incident edge, from its node stalk into that edge stalk.
They are:

- **Linear.** All nonlinearity lives inside the cell. This keeps the cellular-sheaf formalism real —
  a genuine sheaf Laplacian, disagreement as Dirichlet energy, cheap reconciliation — and avoids
  turning reconciliation into a nested optimisation.
- **Learned**, under a sparsity pressure. This is the local-neuroplasticity analogue: pruning within
  what structure permits.
- **Masked** by a hand-specified structural mask, set at graph construction, naming which node stalk
  features may participate on that edge. The mask is graph structure, not a parameter. **It closes and
  never re-opens** — re-opening a masked feature is structural growth, which is out of scope.
- **Independent at the two ends of an edge.** Each cell learns its own map into the shared space. If
  they were tied, agreement would be definitional and disagreement could carry no information.

Edge stalk dimension `m` is **determined by the mask**: the shared space is exactly large enough to
hold the features that edge permits. `m` is therefore not an independent parameter and varies across
edges — the sheaf Laplacian has no uniform block structure, which is accepted.

### Disagreement, and what is done about it

Disagreement is the difference, in an edge stalk, between the two endpoint cells' restrictions of
their node stalks. It is Patchworks' only error signal, and it is one edge's term of the sheaf's
Dirichlet energy — the global energy is the sum of these terms over every edge, but no cell ever
reads that sum; it only ever sees its own edges' terms. Predictive coding's error and the sheaf's
inconsistency are **the same quantity**, not two objects that need relating.

Agreement is **penalised, not enforced.** Reconciliation runs exactly one local descent step on
disagreement per tick (see [`02-tick-semantics.md`](./02-tick-semantics.md)) and never clears it.
Residual disagreement *is* the signal the local learning rule consumes; a hard projection onto the
consistent subspace would zero out the quantity the architecture runs on, and would drag a global
solve into a system whose thesis is locality.

### How reconciliation re-enters inference

Reconciliation edits the **node stalk only**. It never reaches into the chart. On the following tick
the cell reads its own stalk as input, so the correction arrives **as evidence**, like any other
observation, and the cell learns what to do about disagreement rather than having a correction
imposed on its internal state.

### Unit delay

**Every edge costs exactly one tick.** Graph distance is literally temporal distance, and a relay
chain buys reach at the price of latency.

Two consequences:

- **Graph-locality is structural, not a preference.** With per-edge delay there is no "now" spanning
  the graph, so a global aggregation step is not expressible.
- **Depth buys horizon, not rate.** Every cell still updates every tick; a distant cell sees staler
  information and sits in a longer loop, but its update rate is unchanged. Depth alone does not
  produce slowly-integrating cells. Whether an explicit schedule is needed for that belongs to the
  multi-timescale section.

## Known exposure

Recorded, not pre-emptively solved.

- **Recurrent failure modes.** Persistent chart plus stalk feedback makes each cell an RNN, with the
  attendant risks. The known escape hatch is a designated pass-through subset of the edge stalk
  carrying state across the recurrence — an LSTM-shaped fix. Not built until the problem is observed.
- **The shared frozen body is a bet, and it has a first experiment.** Nothing in the literature
  demonstrates its sufficiency, because no prior system trains a frozen-body-plus-thin-surface
  architecture by cell-local rules alone — that conjunction *is* the thesis, so the demonstration is the
  proof-of-concept itself. The build therefore owes an early falsification test, before anything depends
  on the body holding: **train the sandbox's sensory cells and its abstract/planning cells with one
  shared frozen body, and compare each against a per-cell-body control.** Cells at opposite ends of the
  graph — closest to raw pixels, and furthest from them — are the premise's hardest case; if a shared
  body holds across both, it has survived cheaply. Requires the local learning rule
  ([patchworks#5](https://github.com/NGL321/patchworks/issues/5)) first.

- **Flex priority.** Fixed parameters, ordered by willingness to see them become hyperparameters, so
  later pressure hits the most flexible first. Read it as the constraint ladder: each rung loosens how
  rigid the constraint on a small network is, and none of them abandons constraint.
  1. **Per-cell low-rank adapters** over the frozen body — cheapest, and keeps the body shared.
  2. **Heterogeneous bodies**: same size, different shape. Size and connectivity constraints unchanged;
     only internal machinery varies. Permitted by the cell contract already, which fixes interface,
     algorithm, `n` and `k` while letting capacity vary.
  3. **Unfreeze the body per-cell** — expensive, because it re-opens the local-learning problem for the
     whole body rather than for a thin surface.
  4. Degree of compression `n/k`.
  5. `k` — may become a range or a gradient across the graph if uniformity fails.

  **This ordering is a deliberate reversal.** `k` was formerly the *first* thing this spec was willing to
  flex and is now the last: widening `k` weakens the low-dimensional claim, which is more load-bearing
  than uniform machinery is.

  **`n` is deliberately absent from this list.** It is fixed and intended to stay fixed.
