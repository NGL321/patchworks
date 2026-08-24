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

`n` and `k` are **global constants**, identical for every **predicting** cell — boundary cells are
exempt and carry world-shaped node stalks
([ADR-0006](../adr/0006-boundary-cell-stalks-are-world-shaped.md)). The proof of concept's values,
and the arithmetic that sets them, are in
[`06-graph-topology.md`](./06-graph-topology.md): `n = 32`, `k = 12`.

**Why `n` is uniform.** Underneath the choice is a bet: that different sub-problems are solvable in a
shared solution space and a shared inferential geometry, provided each cell's metric space is
configured so that distances between its features reflect a locally linear problem. The bet bears
hardest on the **private** features, which are constructed rather than passed — nothing outside the
cell ever sees them, so nothing outside the cell can compensate for a badly-shaped one. It is the same
bet *The cell body* makes about the machinery, seen from the side of the dimension: shared machinery
and a shared width stand or fall together, since a body driven by cells of differing width is not the
same object twice.

A uniform `n` is what makes that bet **checkable**. Every construction diagnostic — `χ`, and the
private-dimension bound `dim H⁰ ≥ Σ_v max(0, n − Σ_{e∋v} m_e)` — is then comparable across cells, so
a difference between two cells is attributable to topology alone. That is load-bearing downstream:
[`06-graph-topology.md`](./06-graph-topology.md) reads the private-dimension gradient off the taper,
and [`05-timescales.md`](./05-timescales.md) builds the timescale gradient on that. Let `n` vary
per cell and the gradient is confounded with a per-cell width choice, and no cross-cell diagnostic
means much on its own. The plainer version of the same point: with one width there are no intra- or
inter-cell dimension effects to disentangle, and only network dynamics are left to explain a result.
Batched execution wants few operator shapes anyway, which supports uniformity without being the reason
for it — batching would tolerate a handful of groups; the diagnostics would not.

No biological claim is doing any work here. An earlier draft fixed `n` on a canonical-microcircuit
rationale — a cortical column has an efficient size, and cells are the analogue — and the literature
does not support it: the column is widely held not to be a discrete, uniformly sized unit, no source
claims it has an efficient size, and the canonical microcircuit offered in its place is deliberately
defined without a spatial boundary, so it cannot license a dimension at all. What survives of that
biology supports *uniform contract*, not uniform width, and is cited where it belongs under
*Uniformity* below. See `docs/research/016-cell-contract-citations.md`.

The *degree* of compression (`n/k`) is a hyperparameter; the spec commits to `k < n` and nothing
more. That a useful `k` turns out to be much smaller than `n` is a finding the proof-of-concept
reports, not a number fixed here.

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
its restriction maps and biases. It is compatible with every cell running identical machinery.

This is the one place in the cell contract where the neuroscience genuinely warrants something.
Harris & Shepherd's "themes and variations" — a basic circuit pattern repeated across neocortical
areas, with area- and species-specific modifications — is evidence for exactly this shape: uniform
machinery carrying per-cell specialisation. It is evidence for a uniform **contract**, and not for a
uniform **dimension**; what fixes `n` is argued under *The cell*, on other grounds entirely.

**A cell's own metric space is its node stalk**, and the basis of that space is fixed by its
restriction maps. Not the chart: the chart is shared machinery (see *The cell body*), and every cell
drives the same one. Locating "its own metric space" in the stalk rather than the chart is what makes
that shared body consistent with the claim, and *The geometry* below states the division of labour
exactly.

**Schedule** is the only per-cell freedom this section grants, and
[`05-timescales.md`](./05-timescales.md) declines to spend it: every cell runs every tick, and a
slow cell is slow because its content *persists*, not because it updates rarely. A hand-forced clock
divisor survives there as an instrument, never as the mechanism.

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

### The body's construction

The three maps' *interface* dimensions are `n` and `k`, fixed above. Their *internal* width, their
depth, and their activation were left open, and all three are spent by the same quantity — the **fold
margin** [ADR-0007](../adr/0007-the-disagreement-floor-is-tolerated-not-represented.md) bounds and
[`05-timescales.md`](./05-timescales.md) makes a precondition of persistence — so they are settled
together here.

**The activation is piecewise-linear; ReLU is the instance.** This is a premise several decisions were
already resting on rather than a new choice, and it had never been written down. A piecewise-linear
activation — ReLU, leaky ReLU, hard-tanh, maxout — is one built from straight segments, and every
network built from them partitions its input space into finitely many **convex polytopes on each of
which the whole map is exactly affine**. That partition *is* the object the timescale mechanism is
made of: an *activation region* is one polytope, a *fold* a face between two, the *regional spectrum*
the affine part inside one, *region dwell* how long a chart stays inside one, and the *fold margin*
the distance to the nearest face. Under a smooth activation the Jacobian simply varies continuously
everywhere: there are no regions to dwell in and no folds to cross, and
[ADR-0005](../adr/0005-timescale-is-persistence-not-a-schedule.md)'s mechanism loses its referent
rather than merely weakening. The class is what the mechanism requires; **ReLU is what the numbers
below were computed for**, and a later swap within the class keeps every term intact but must
re-derive them.

**Each map's hidden width is its own minimum, `max{d_x + 1, d_y}`** — Park et al.'s exact floor for
universal approximation by ReLU networks, below which Lu et al. find a phase transition rather than a
degradation. Evaluated at `n = 32`, `k = 12`, that is **45** for `encode` (`ℝ³² × ℝ¹² → ℝ¹²`), **13**
for `step` (`ℝ¹² → ℝ¹²`), and **32** for `decode` (`ℝ¹² → ℝ³²`). Written as the rule rather than as
three constants because `n/k` and `k` are both rungs on the *Flex priority* ladder below: pull one and
the widths re-derive themselves.

**Sized *at* the floor, not above it.** The floor and the optimum turn out to be the same point, which
is why this costs nothing to respect. Measured on
[#42](https://github.com/NGL321/patchworks/issues/42)'s rig at `σ_w² = 1.2`, the median fold margin is
**0.019** for `encode`/`step` at `[45]`/`[13]` against **0.0067** for a `[128]`/`[32]` body — while the
spread the widths are sometimes imagined to buy is flat across the range (τ ratio 2.7 against 2.4).
Wider bodies pay margin for nothing, exactly as Hanin & Rolnick's `1/#neurons` scaling predicts.
Dropping `encode` *below* its floor buys nothing back either (44 measures 0.018, inside the noise), so
the floor is free in both directions.

**The margin is read from `encode` and `step` only.** The quantity ADR-0007 bounds is a property of the
chart's own round trip, `d chart_{t+1} / d chart_t`, and `decode` is not on it — so a `decode` fold
cannot move a cell's region, its dwell, or its timescale. It is still a fold: a cell crossing one has
its *prediction* jump discontinuously. That arrives as prediction error and is the bias rule's food
([`07-local-learning-rule.md`](./07-local-learning-rule.md)) — reducible model error, not a floor. The
body has three maps and the margin has two, and that asymmetry is derived, not an oversight.

**One hidden layer per map, and the reason is not that depth is useless.** With one hidden layer each
per-cell bias translates its own fold and every fold stays a hyperplane — *The geometry* below reads
this as one arrangement of folds, translated per cell. A second layer defines its folds in the first
layer's activation space, so pulled back into the chart they are **bent** surfaces whose bends move
when the first layer's biases move: the same bias budget would buy a strictly richer per-cell geometry,
without unfreezing a single weight. Two things say not yet. It is measured expensive — `[45,45]`/`[13,13]`
halves the median fold margin, to 0.0093 — and it lands a **fourth** geometric job on a bias vector
*Known exposure* below already calls over-subscribed with three. So depth is what to reach for once the
adapting surface is wider than biases (rung 1 of the ladder), not while it is only biases.

**The floor is necessary, not sufficient, and the spec claims only the necessary half.** Park et al.'s
bound is proved for width-bounded, *depth-unbounded* networks: it says no depth rescues a width below
45, not that width 45 suffices at any given depth. A single hidden layer at the floor therefore sits on
the bound without inheriting its guarantee. The PoC does not need the guarantee — nothing trains these
weights, and what a frozen random body owes the design is a non-degenerate map with well-separated
regions, not approximation of a specified target. What the floor is doing here is protecting the
**pretraining swap-in** (*The cell body*, above), which is where an approximation claim would first be
made, and buying at zero cost the one failure mode the literature says is abrupt.

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
  turning reconciliation into a nested optimisation. Linearity is also a **geometric commitment**,
  not only an efficiency one: it assumes each overlap is locally flat. And it is load-bearing a third
  time, which is easy to miss: the identification of predictive-coding error with the sheaf coboundary
  is *derived for linear networks*, so under nonlinear restriction maps "disagreement and prediction
  error are the same quantity" would stop being **true**, not merely become expensive to compute. See
  [ADR-0004](../adr/0004-linear-restriction-maps-assume-local-flatness.md) and *The geometry* below.
- **Learned**, under a sparsity pressure. This is the local-neuroplasticity analogue: pruning within
  what structure permits. The pressure is an **L1 on the normalised map**, so it redistributes weight
  across the map's directions rather than removing it — see *Scale is gauge-fixed* below and
  [`06-graph-topology.md`](./06-graph-topology.md).
- **Masked** by a hand-specified structural mask, set at graph construction, naming which node stalk
  features may participate on that edge. The mask is graph structure, not a parameter. **It closes and
  never re-opens** — re-opening a masked feature is structural growth, which is out of scope.
- **Independent at the two ends of an edge.** Each cell learns its own map into the shared space. If
  they were tied, agreement would be definitional and disagreement could carry no information.

Edge stalk dimension `m` is **fixed at construction by the edge's role**, and the mask constrains
which node stalk directions may participate on that edge. The two are separate acts and must stay
separate: **the mask selects** — structurally, once, closing and never re-opening — and **the map
compresses**, densely and by learning, mixing every permitted direction into the `m`-dimensional
shared space. `m` varies across edges, so the sheaf Laplacian has no uniform block structure, which
is accepted.

This sentence used to read that `m` was "determined by the mask", the shared space "exactly large
enough to hold the features that edge permits", and `m` "therefore not an independent parameter".
That is wrong and [#83](https://github.com/NGL321/patchworks/issues/83) corrected it:
[`06-graph-topology.md`](./06-graph-topology.md) governs. Read literally, `m` would be the *count* of
permitted features, so a patch cell's `m = 8` edge would permit 8 of its 48 stalk directions and the
map would **select 8 and discard 40** — five-sixths of the patch thrown away down the only route that
patch's information ever takes, which is not a compression and contradicts `06`'s "a patch cell's
48 → 8 restriction *is* the compression of that patch". The dependency also runs backwards against
`06`'s own treatment: boundary edges are *given* `m = 8` against the interior's 4 for a stated
reason, and `m` is the first rung on [#14](https://github.com/NGL321/patchworks/issues/14)'s
constraint ladder, priced to be widened. Both only make sense if `m` is chosen at construction.

What a cell may transmit is its budget `Σ_e m_e`, so it must compress what matters into a reduced
latent and **what will not fit is what stays private** — which is why the same budget appears in the
`H⁰` bound below.

**`m` is fixed at construction and never changes.** The sparsity pressure prunes *within* the mask —
it drives weights to zero; it does not shrink the stalk. A stalk dimension that moved during a run
would make the sheaf a moving target and would edge toward structural growth, which is out of scope.
Whether pruning and re-opening could later alternate in developmental phases is held in the map's
fog, not exercised here.

### Scale is gauge-fixed

A restriction map's overall magnitude is **not identified by the transport rule's objective**. The rule
learns on disagreement relative to the restricted beliefs' own current magnitudes
([`07-local-learning-rule.md`](./07-local-learning-rule.md)), which is invariant under scaling both of
an edge's maps together, and the sparsity pressure is an L1 on the normalised map, which is blind to
magnitude too. Nothing in the rule has an opinion about it.

Left free, an edge's joint scale **grows**. For a scale-invariant parameter the gradient is always
perpendicular to it, so `‖F_{u◁e}‖_F² + ‖F_{v◁e}‖_F²` is non-decreasing at every step and strictly
increasing whenever anything is learned (Arora, Li & Lyu, Lemma 2.4; Salimans & Kingma report the same).
The failure that produces is not the collapsed sheaf but **maps that stop moving** — a vanishing
effective step size, with every norm-based reading healthy. In the other direction — one end of an edge
up and the other down — the objective points *away* from collapse: the relative disagreement lies in
`[0, 1]` and sending either map to zero sends it to `1`, its worst value.

So the magnitude is fixed rather than learned. Fixing it removes a free parameter; it does not cap a
learned one.

- **Interior maps** carry a band: `‖F‖_F ∈ [1/ρ, ρ]`, with `ρ = 2` fixed at construction. The two ends
  do different jobs. The **upper** face is the working constraint — the joint scale grows into it, so
  the larger end of every interior edge rides `‖F‖_F = ρ` — and what the band actually leaves free is an
  **edge's scale ratio**, which is how a cell's node stalk stays its own metric space in **basis and
  scale**. The **lower** bound is a guardrail against arithmetic and the residual asymmetry; nothing in
  the objective drives toward it.
- **Boundary-cell maps** carry the exact gauge, `‖F‖_F = 1`. A boundary cell runs no body and its stalk
  is world-shaped ([ADR-0006](../adr/0006-boundary-cell-stalks-are-world-shaped.md)), so it has no
  metric individuality to protect; unit conversion is the environment contract's job, not the sheaf's.

Enforced by **projection** after each transport step. The norm is Frobenius, not spectral, and the
difference matters: a spectral gauge pins only the largest singular value, leaving every other direction
free to shrink at no cost, so it would exclude `F = 0` and leave `F → rank 1` open. A Frobenius gauge
pins the budget across all directions, so rank concentration buys per-direction gain and pays for it
elsewhere — though **whether that is a price or a reward depends on the objective**, and here it is a
weak price at best: matching a neighbour on any single tick needs only that tick's direction, and the
L1 on the normalised map is minimised, at fixed Frobenius norm, by the sparsest map. What resists
concentration is that a map serves the whole distribution of stalk states, not one draw.

**No rank floor is imposed** — learned rank-deficiency is wanted, and its degenerate limit is
instrumented rather than forbidden (*Known exposure*). With the price weak, that instrument is what
says which regime the maps are in, and it carries the argument alone.

A map's **norm is not a diagnostic**: the upper face binds continuously, so an interior edge's larger
end reads `ρ` whether learning is healthy or frozen.

The bound, which is the band's real content rather than an incidental cost: an edge's representable
scale ratio is `ρ²` times the range rank concentration affords. Genuine mismatch beyond that is
irreducible, and joins the static floor in
[ADR-0007](../adr/0007-the-disagreement-floor-is-tolerated-not-represented.md).

See [ADR-0010](../adr/0010-restriction-map-scale-is-gauge-fixed.md); settled in
[#37](https://github.com/NGL321/patchworks/issues/37).

### Disagreement, and what is done about it

Disagreement is the difference, in an edge stalk, between the two endpoint cells' restrictions of
their node stalks. It is Patchworks' only error signal, and it is one edge's term of the sheaf's
Dirichlet energy — the global energy is the sum of these terms over every edge, but no cell ever
reads that sum; it only ever sees its own edges' terms. Predictive coding's error and the sheaf's
inconsistency are **the same quantity**, not two objects that need relating.

Agreement is **penalised, not enforced.** Reconciliation runs exactly one local descent step on
disagreement per tick (see [`02-tick-semantics.md`](./02-tick-semantics.md)) and never clears it.

**Locality is not what rules enforcement out.** It would be convenient to say that driving
disagreement to zero requires a global solve, and it is not true: Hansen & Ghrist enforce `L_F x = 0`
exactly by Lagrangian saddle-point dynamics in which every node uses only quantities computed from its
own incident edges. Exact enforcement is reachable by graph-local means — asymptotically, over many
rounds, rather than as a one-shot projection, but reachable. It is declined for two other reasons.
First, it needs a **dual variable per edge**: per-edge state with its own dynamics, which is the same
object and the same objection as the per-edge baseline
[ADR-0007](../adr/0007-the-disagreement-floor-is-tolerated-not-represented.md) rejects. Second, and
sufficient on its own, it would drive to zero the quantity the whole architecture runs on.

Worth noting what that leaves. In Hansen & Ghrist's formulation the Dirichlet-energy penalty is not
the enforcer at all — it is present only to improve stability and convergence, "leaving the minimum
and minimizer undisturbed" — and the dual variables do the enforcing. **Patchworks keeps the
stabiliser and discards the enforcer.** Its one descent step per tick *is* a graph-local dynamic that
would approach agreement asymptotically; it simply never arrives, because the world keeps moving and
the floor below is always there.

This is a separate rejection from
[ADR-0002](../adr/0002-message-passing-is-one-step-not-a-solve.md)'s, and does not disturb it. That
decision rules out iterating *within a tick*, on the ground that any legitimate stopping rule needs a
read of disagreement across the graph. Nothing above touches that argument: local dynamics that flow
without stopping have no stopping rule to globalise, and asking them whether they have converged yet
would reintroduce exactly the global aggregate ADR-0002 excludes.

Residual disagreement divides in two, and only one half is a signal:

- the **reducible** part — model error, the cell simply being wrong. This is what the local learning
  rule consumes.
- the **disagreement floor** — the part learning cannot remove, in two kinds. A **static floor** is a
  function of configuration: curvature the linear map cannot follow
  ([ADR-0004](../adr/0004-linear-restriction-maps-assume-local-flatness.md)), mask or learned rank
  deficiency, aleatoric noise. A **lag floor** is a function of motion: two endpoints whose contents
  live at different timescales ([ADR-0005](../adr/0005-timescale-is-persistence-not-a-schedule.md)),
  so the slow end is behind.

Nothing in the architecture represents the floor, and the learning rule is constrained never to
target zero residual. See
[ADR-0007](../adr/0007-the-disagreement-floor-is-tolerated-not-represented.md), which also carries the
offline probe that separates the two kinds.

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
  information and sits in a longer loop, but its update rate is unchanged. The reason, argued in
  [`05-timescales.md`](./05-timescales.md), is that **delay is a phase shift, not a decimation**: it
  removes no frequency content, so a deep cell has the same input bandwidth as one at the rim and is
  merely looking at older data. Depth alone does not produce slowly-integrating cells. Where they do
  come from is `05-timescales.md` — persistence in the private features, not a schedule.

Delay has a **second axis**, and the spec claims the favourable side of it deliberately rather than by
accident. The frequency argument above is about time; the other axis is what a delayed value has *not
yet been mixed with*. DRew, the one piece of work that tunes this parameter directly, describes the
trade both ways: a larger delay means features arrive before repeated message passing has smoothed
them, and a smaller delay means a node also leverages the structure around its neighbour. Every
Patchworks edge carries delay one, which is strictly more delayed than standard message passing, so
what arrives is less pre-smoothed — wanted, since disagreement is the only error signal and smoothing
is what erodes it. (How far that erosion can go on its own is a separate question, open at
[patchworks#37](https://github.com/NGL321/patchworks/issues/37), not settled here.)

The cost, stated so that a later failure can be traced back here: a cell sees less of its neighbour's
neighbourhood already mixed in. That too is wanted, and for a reason the spec has already committed
to — edge predictions are the shadow of the temporal prediction, never independent heads, precisely so
that a cell does not learn to model its neighbours instead of its own piece. Transport is a real
channel with its own structure, and predicting what will arrive is the modelling work, not a
redundancy to be smoothed away. If the recomposition the taper depends on ever proves to be
under-mixed, this is the trade to revisit.

## The geometry

Where the geometric structure of Patchworks actually lives, and — as much of the point — where it
does not. Settled in [patchworks#15](https://github.com/NGL321/patchworks/issues/15).

### The piece, and why there is a chart at all

**Each cell owns a `k`-dimensional, locally Euclidean piece of the problem.** Its chart is a chart of
*that piece*, in the strict mathematical sense. This is the geometric statement of the thesis: the
world is highly nonlinear and entangled and is not claimed to be a manifold; the pieces are, each is
flat enough at its own scale for linear transport, and the sheaf recomposes them.

It is also why `k < n` rather than merely that: **`k` is the dimension of the piece; `n` is the room
needed to talk about it with neighbours.**

**Patchworks is a sheaf, not an atlas.** An atlas requires every chart to have the dimension of the
manifold it covers; ours do not, and the world does not have one dimension to have. A cellular sheaf
is the atlas idea with the constant-dimension requirement removed — locally-defined data, overlaps,
gluing conditions, no demand that the union be a manifold at all. That is the reason the formalism
fits, and "patches of a manifold" in the project's name should be read as *patches with gluing*.

**"Manifold" is not general vocabulary here.** Node stalks, edge stalks and their direct sums are
**vector spaces** — flat, uncurved, uninteresting as geometry — and are called that. The word is
reserved for the local flatness above, which is the one place something depends on it
([ADR-0004](../adr/0004-linear-restriction-maps-assume-local-flatness.md)).

### The division of labour between the two adapting surfaces

The cell body is shared and frozen, so `step` is **one** map on **one** `k`-dimensional space: there
are not `N` little geometries. But the biases are per-cell, and in a piecewise-linear network the
weights fix the *directions* of every folding hyperplane while the biases fix **where each fold
sits**. Cells therefore share one arrangement of folds **up to translation** — same fold directions,
different offsets, and so genuinely different activation patterns and local behaviour.

**One arrangement of folds, translated per cell.** This is why bias-only adaptation is not the thin
residue it sounds like.

| Surface | Where it acts | What it does geometrically |
|---|---|---|
| **Biases** | inside the shared body | translate the folds of the shared nonlinear map |
| **Restriction maps** | outside, on stalks | fix the basis of the node stalk and the transport into each edge stalk |

The restriction maps never touch the body's geometry. They fix what features mean and how they
relate; the biases move the shape that consumes them.

### Three lossy maps

Information is discarded in exactly three places, and they are not the same kind of act:

| Map | Loses | Kind |
|---|---|---|
| world → sensory tiling (*the sensory slice*) | everything outside the render; all object pose | fixed by the environment |
| node stalk → chart (*compression*, in `encode`) | whatever the cell's piece does not need | **nonlinear, private** |
| node stalk → edge stalk (*restriction*) | everything the mask forbids | **linear, shared** |

That asymmetry — compression nonlinear and private, restriction linear and shared — is the geometric
statement of why there are three tiers rather than two.

**"Projection" is deliberately not vocabulary.** It would invite the reading that inference happens
inside a restriction map, which is exactly what the three-tier split exists to prevent.

### `H⁰` is the private features

The coboundary `δ` maps a configuration to its per-edge disagreements; the sheaf Laplacian is
`L = δᵀδ`, and `xᵀLx` is the Dirichlet energy — the sum over edges of squared disagreement.
`H⁰ = ker δ = ker L` is the set of configurations on which every edge disagrees by zero.

Under Patchworks' masks that space has a concrete identity: **masked-out node stalk directions are
global sections.** A direction that participates in no edge cannot disagree on any edge, so it lies
in `H⁰` by construction. The *features private to a cell's sub-problem* — named at the top of this
document as one of the two things a two-tier scheme cannot express — **are `H⁰`**. Not analogous to
it; identical.

This gives a bound holding for **any** restriction maps, learned or not:

```
dim H⁰  ≥  Σ_v max(0, n − Σ_{e∋v} m_e)
```

Two consequences. `H⁰` is **large by construction and enlarged by sparsity**, so consistency is not
scarce in the way a naive dimension count suggests. And low-degree cells manufacture private
structure permanently.

**Corrected by [`06-graph-topology.md`](./06-graph-topology.md):** this bound and `χ` below are
computed over **predicting cells only**. Boundary cells were originally counted here as the
lowest-degree cells in the graph, which is arithmetically true and diagnostically worthless — the
world overwrites their stalks every tick and no boundary cell holds anything. With their stalks now
world-shaped rather than `n`-shaped ([ADR-0006](../adr/0006-boundary-cell-stalks-are-world-shaped.md)),
including them swamps the measurement outright.

A third, from [`05-timescales.md`](./05-timescales.md): because reconciliation descends along
`im δᵀ` and `ker δ = (im δᵀ)^⊥`, **the private component of a node stalk is exactly invariant under
reconciliation**. That insulation is what makes slow state possible at all, so the bound above also
sizes a cell's capacity to hold a slowly-varying variable — and turns that capacity into a
construction quantity set by the masks.

**What is recorded, and what it is worth:**

- **`χ = Σ_v n − Σ_e m_e = dim H⁰ − dim H¹`.** Fixed at construction by the masks, invariant under
  learning — no learned parameter appears in it. Computed and recorded at graph construction. It is
  a diagnostic, not a budget to hit, and nothing branches on it.
- **`dim H⁰`** and the **minimum achievable Dirichlet energy under the world's boundary conditions** are
  run-time measurements against the learned maps, not construction constants.

**`H⁰` and `H¹` are not topological invariants here.** They are weight-dependent linear subspaces
that move as the restriction maps learn, and `H¹` has two sources — graph cycles, and map
rank-deficiency, which is invisible to graph topology and which Patchworks' masked maps guarantee.
Claims of the form "`H¹` vanishes on a tree" hold only for surjective restriction maps and are false
here. Topological invariance is **not** load-bearing on the sheaf side; where it does bite is the
world's workspace ([patchworks#25](https://github.com/NGL321/patchworks/issues/25)).

### Two cohomologies, which are not the same cohomology

Sheaf cohomology above is taken over the **graph**, with stalks as coefficients and restriction maps
as the differential. **Information cohomology** (Baudot & Bennequin, *The Homological Nature of
Entropy*, Entropy 17(5), 2015) is taken over a **poset of partitions of one sample space**, with
functionals of a probability law as coefficients. There is no graph in it. The two share the letter
`H` and nothing else, and their `H¹`s are different groups of different complexes over different
sites.

In that theory `H⁰` is the constants, `H¹` is generated by Shannon entropy — the chain rule *is* the
1-cocycle condition — and the higher mutual informations are **coboundaries**, hence zero in
cohomology. It is retained as an **interpretive lens** and is the right citation for why entropy and
the chain rule are canonical. It is **not** available as the shape of the local learning rule; see
[patchworks#5](https://github.com/NGL321/patchworks/issues/5) and
`docs/research/015-information-cohomology.md` for why, including a cost ceiling that would not batch.

## Known exposure

Recorded, not pre-emptively solved.

- **Recurrent failure modes.** Persistent chart plus stalk feedback makes each cell an RNN, with the
  attendant risks — and [`05-timescales.md`](./05-timescales.md) raises the stakes, since slow state
  now *depends* on the recurrence holding content for hundreds of ticks. The fix, if it is ever
  needed, is a two-rung ladder, and neither rung is built.

  **Rung one: a protected linear channel through `step`** — a designated subspace of the chart that
  `step` passes with unit gain, ungated. It is an LSTM constant-error carousel and not a gate. It
  costs no parameters and breaks no freeze, because it is a **construction** choice about the shared
  body, of the same kind as its initialisation, rather than per-cell adaptation. Per-cell variation
  keeps arriving where it already does, through the biases. Note that this rung does the *whole* of
  the job Patchworks actually has: the carousel was invented to preserve gradients through
  backpropagation-through-time, and there is no BPTT here — the bias rule is a single local gradient
  step through one tick of the cell's own frozen forward path
  ([`07-local-learning-rule.md`](./07-local-learning-rule.md)). What transfers is the forward job,
  holding activation content across many ticks, and that is exactly what an ungated channel does.

  **Rung two: a learned gate on `encode`'s fusion** — the GGNN-shaped fix, a small gate deciding per
  tick how much incoming evidence to admit against how much of the persisted chart to keep. **This is
  the first thing to reach for if rung one proves insufficient**, and the trigger is the one thing
  rung one structurally cannot do: **clear a channel deliberately.** An ungated channel holds
  unconditionally; it cannot flush stale content on decisive evidence. Its price is steep and should
  be paid knowingly — a gate is per-cell parameters that are neither biases nor restriction maps, so a
  **third parameter group**, and [ADR-0008](../adr/0008-the-local-rule-splits-by-parameter-not-by-cell.md)
  splits the local rule *by* parameter group, meaning a third learning rule with its own signal; or
  else part of the body comes unfrozen.

  One correction, and one object the wrong version was hiding. This entry previously named the escape
  hatch on the **edge stalk**, which is the wrong tier: the recurrence is `chart(t) → chart(t+1)` through `step`, and the edge
  stalk is not on that loop — its contents reach the chart a tick later, via reconciliation and
  `encode`. A pass-through subset of the edge stalk is a skip on the *spatial* path. And such a subset
  would be a different object worth naming separately: an edge-stalk direction that always passes
  through is a direction on which disagreement is meant to be small — a hand-placed `H⁰`-like channel
  the architecture deliberately never learns from, not a fix for the recurrence.

  Rung one's real limitation is not new. It cannot clear a channel, which means it protects a bad
  commitment exactly as well as a good one — the same price
  [`04-action-and-the-boundary.md`](./04-action-and-the-boundary.md) already accepts for `H⁰`
  insulation, appearing a second time one tier down.
- **Over-smoothing here is the error signal vanishing, not a quality loss.** Cai & Wang
  ([arXiv:2006.13318](https://arxiv.org/abs/2006.13318)) state over-smoothing precisely as the Dirichlet
  energy of the embeddings converging to zero. Since disagreement *is* the Dirichlet energy and is the
  only error signal this architecture has, the classical failure mode arrives here as the disappearance
  of the quantity both halves of the local learning rule are computed from — **and of the only
  instrument that would show it happening**. This is state collapse, the sibling of the parameter
  collapse *Scale is gauge-fixed* excludes; the gauge does not exclude it, and nothing else does either.

  It is not, on its own, a pathology. Boundary cells are written by the world every tick, prediction
  pushes states off the consistent set continuously, and `H⁰` is large by construction, so a consistent
  section here is content-rich rather than degenerate. Energy draining **at rest** is the quiescent hold
  ([ADR-0007](../adr/0007-the-disagreement-floor-is-tolerated-not-represented.md)) working as designed.
  The alarming case is energy falling **while the world drives**, and telling the two apart is what the
  paired instrument is for: per-edge energy read alongside per-edge **effective rank**, the
  participation ratio `(Σσᵢ²)² / Σσᵢ⁴` of a restriction map's singular values, on the diagnostic cadence.
  Energy down under drive with effective rank sliding toward 1 across the fleet is collapse; energy down
  at rest with rank steady is the lag floor draining.

  The counterweight the literature offers is **orientation, not authority**: Bodnar et al.'s
  sheaf-diffusion result that a rich harmonic space resists collapse is proved for *orthogonal* sheaves,
  and per `docs/research/015-sheaf-geometry.md` those theorems do not reach a sheaf whose maps are
  masked, learned, and merely norm-bounded. The proof-of-concept instruments for this; it does not
  assume it away.

- **The shared frozen body is a bet, and it has a first experiment.** Nothing in the literature
  demonstrates its sufficiency, because no prior system trains a frozen-body-plus-thin-surface
  architecture by cell-local rules alone — that conjunction *is* the thesis, so the demonstration is the
  proof-of-concept itself. The build therefore owes an early falsification test, before anything depends
  on the body holding: **train the sandbox's sensory cells and its abstract/planning cells with one
  shared frozen body, and compare each against a per-cell-body control.** Cells at opposite ends of the
  graph — closest to raw pixels, and furthest from them — are the premise's hardest case; if a shared
  body holds across both, it has survived cheaply. Requires the local learning rule
  ([`07-local-learning-rule.md`](./07-local-learning-rule.md)), settled but for the stability question
  it carries forward as its own open ticket.

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

  **`n` is deliberately absent from this list.** It is fixed and intended to stay fixed. The reason is
  the one given under *The cell contract* — uniform width is what makes every construction diagnostic
  comparable across cells — and unlike the biological rationale it replaces, it names its own cost: a
  per-cell `n` confounds the private-dimension gradient with a per-cell width choice, so the timescale
  gradient built on that gradient stops being attributable to topology, and cross-cell diagnostics
  stop meaning much individually. Anyone reaching for `n` anyway is trading away the project's
  measurement apparatus, not just its uniformity.
