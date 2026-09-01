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
   left behind, the cell's own operator `K` advances that fused chart one tick, `decode` reads
   off the predicted node stalk. No cell reads another cell's state here.
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

**Every sensory boundary cell is written every tick, and silence is a value rather than an absence.**
[#128](https://github.com/NGL321/patchworks/issues/128) found that this ordering quietly assumed a
world that speaks on every tick — the sandbox advances whether or not the agent acts, so something
always lands. A world that can be *quiet* breaks that assumption, and the contract makes the
requirement explicit instead: **a rim whose world can fall silent owes an idle symbol in its
encoding**, and that symbol is written like any other value.

The two alternatives were rejected because both cut into the ordering above, which is load-bearing well
beyond the rim. Skipping the external write on a quiet tick leaves a stalk holding whatever
reconciliation last did to it, so silence would read as the graph's own most recent guess. An
event-driven tick is worse: it lets the agent's own emissions set the clock, which is the same failure
arriving from the other direction.

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

Per-tick cost is therefore bounded by construction: one `encode`/`K`/`decode` per cell, plus one
restriction-and-reconcile per edge. No convergence check, no early-stopping logic. The Koopman
conversion made this **cheaper** rather than dearer: two of the body's three maps lost their hidden
layer, and `K` is one batched matrix multiply over the population.

## Reconciliation gain

One descent step needs a step size. It is **per cell**, and it is the largest step that is stable
against that cell's own local Laplacian block:

```
gain_v  =  γ / ( g_v² · c_v )                        with a single global γ ≤ 1

  g_v  =  ρ                                          predicting cells (band gauge, ‖F‖_F ≤ ρ)
       =  1                                          boundary cells   (exact gauge, ‖F‖_F = 1)

  c_v  =  min( deg(v), max( c, ⌈deg(v) / n_v⌉ ) )    c = 2, declared globally alongside ρ
```

Ruled by [#190](https://github.com/NGL321/patchworks/issues/190) against the ledger
[#189](https://github.com/NGL321/patchworks/issues/189) assembled. Three terms left the denominator at
once — `Σ_e m_e`, `ρ² · deg(v)`, and the `max` that chose between them — and what replaced them bounds
the same quantity, `λ_max(Σ_e F_evᵀF_ev)`, by construction rather than by inheritance. Every surviving
term is read off the built graph or declared once, so the shape of the formula is what it always
claimed to be: one global `γ` over a denominator with **nothing per-cell to set**.

**Why the two old terms went.** `Σ_e m_e` was doing two jobs under one symbol and neither survived the
ledger. As a **bound** it was never derived — nothing shows `Σ_e m_e` bounds `λ_max(Σ_e F_evᵀF_ev)` at
all. As a **normalisation** it was defended in its own right, for *equalising* the effective step
across the graph, and #189 measured that property for the first time: in the only units that make the
claim mean anything — `gain_v · λ_max`, the step as a fraction of the largest stable one — the spread
across the taper is **3.57x at initialisation and ~2.6x taught, and graded by depth, largest at the
apex**. It does not remove the depth gradient. It inverts and shrinks it. **The property was never
held, so striking it spends nothing**, and this is a correction rather than a purchase. `ρ² · deg(v)`
is a true bound and goes for a different reason: it is tight only where a cell's incident maps load the
same input direction, which they do not, and `c_v` is the term that says so.

**Nothing here is a runtime read.** A construction-time read of `λ_max` would not merely go loose as
the maps learn — it would go **unsafe**, because `λ_max` grows *toward* any bound fixed at
construction: `bound / true λ_max` falls **41.29 untrained → 4.671 taught**
([#182](https://github.com/NGL321/patchworks/issues/182)). A per-tick read is a per-cell runtime
quantity computed from live parameters, which is the shape this architecture avoids everywhere else.
So the bound is **made true instead**, in
[ADR-0011](../adr/0011-the-locality-guarantee-is-enforced-not-inherited.md)'s idiom — enforced, not
inherited. [ADR-0010](../adr/0010-restriction-map-scale-is-gauge-fixed.md)'s projection already runs
after every transport step to restore the mask and the norm band; it also holds a cell's incident maps'
top singular directions apart, and `c` is the count it holds them to. **Without that term the
denominator is not loose but false**, which is why `c` is gauge-fixed alongside `ρ` rather than
tracked, and why it is ADR-0010 that owns it.

**`c_v`'s two clamps are facts, not hedges.** The **pigeonhole floor** `⌈deg(v) / n_v⌉` is a statement
about stalk dimensions: the drive cell carries `deg = 8` incident maps on a stalk of dimension 1, eight
directions cannot be mutually orthogonal in one dimension, they coincide, and `λ_max` there is exactly
`deg`. A bare global `c = 2` at that cell would be an **unsafe** bound rather than a loose one. The
**ceiling** at `deg(v)` keeps the denominator from ever exceeding the bound `ρ² · deg(v)` already gave.
Every cell on this dome lands identically with or without the ceiling; it is carried because the
un-ceilinged form breaks on a graph that is not this one.

**One formula, evaluated against each cell's actual gauge.** ADR-0010 pins a boundary cell's maps to
the exact gauge `‖F‖_F = 1`, so `Σ_e ‖F‖_F² = deg(v)` is an *equality* there and not a band the
transport rule grows into. The record applied `Σ_e m_e = 8 · deg(v)` to all 264 of them — loose by a
**permanent 8x**, with no gauge headroom to spend, ever. `g_v` is what separates the two cases, and it
is read off the graph like every other term.

**`gain_v` is uniform across the interior, for the first time.** At `ρ = 2, c = 2` every predicting
cell takes `γ/8`. What the correction is worth runs the other way from the old shape — 2.50x at the
apex, 3.00–3.75x through the core, **6.10x at the rim**, 8.00x at the sensory boundary cells and
**12.0x at the actuator**, whose commanded components are the return path's last step into the arm.
The old denominator was loosest where the graph is widest, so the raise is largest at the rim and
smallest at the apex, which is the inverse of what the superseded reading predicted.

**The equalisation claim named [ADR-0005](../adr/0005-timescale-is-persistence-not-a-schedule.md), and
misnamed it.** This section used to argue that a denominator graded by depth would be the explicit
per-cell clock divisor that ADR rejected, wearing a different name. ADR-0005 rejects a **schedule** — a
hand-set per-cell rate — and its prohibition is explicitly about *runtime*, the #41 amendment calling
the restriction structural rather than disciplinary because a per-tick draw is not a value anything
could branch on. A construction-time denominator read off the built graph is neither hand-set nor
readable by anything at runtime. The depth-invariance gloss was **this file's attribution, not that
ADR's claim**: ADR-0005 is untouched and this file is corrected.

A single global scalar in place of `gain_v` is still not sufficient: degrees run from low at the rim to
~6 in the core ([`06-graph-topology.md`](./06-graph-topology.md)) and stalk dimensions differ across
the boundary, so one constant is either too slow at one end or unstable at the other.

**What is rejected is a gain that tracks a signal — not a gain that is a function of the maps.** This
section used to reject "a per-edge gain derived from that edge's recent scale", and the sentence read
wider than it was meant to, as though any denominator finer than a global constant were suspect. The
objection is narrower and it is
[ADR-0007](../adr/0007-the-disagreement-floor-is-tolerated-not-represented.md)'s: a gain derived from
an edge's **recent disagreement scale** needs a tracking window and its own time constant, which is the
per-edge auxiliary variable with a hand-set rate that ADR rejects. `g_v` and `c_v` are neither. They
are functions of the graph and of the gauge — **state, not signal** — fixed when the dome is built and
constant for as long as the run is. `gain_v` remains **one scalar per cell, formed once at
construction**, and that is the property worth protecting.

**What constrains `γ`, stated without embellishment.** Exactly two things:

1. it is capped at **1.0** globally, and
2. the denominator above — a bound on `λ_max(Σ_e F_evᵀF_ev)` that ADR-0010's projection makes good for
   as long as the run does.

**Not, in practice, the fold margin** — and that is a change, made by
[#140](https://github.com/NGL321/patchworks/issues/140). This section used to carry a third:

> `γ × floor` must stay below the cell's **fold margin** — its distance to the nearest activation
> boundary.

That bound is **demoted, not deleted**, and it failed on three counts at once. It was **never
binding**: `γ` is already at 1.0, the global ceiling this section permits, so no fold margin was ever
what held it down. *And its divisor was misnamed:* what the margin is weighed against is the
**standing offset** — the displacement reconciliation leaves, whatever caused it — of which the
disagreement floor is one contributor and, at construction, not the dominant one
([#160](https://github.com/NGL321/patchworks/issues/160),
[ADR-0019](../adr/0019-construction-nominates-the-run-decides.md)). The bound is written
`gain_v × offset < margin_v` from here on. Its stated *reason* — that a shifted operating point changes the cell's effective
timescale — is the premise the Koopman conversion retired, since timescale now lives in `K`'s spectrum
rather than in which activation region a cell occupies
([`05-timescales.md`](./05-timescales.md)). And the margin itself is now read from `encode` alone,
because `encode` is the body's only nonlinearity and so the only map with folds at all.

**A third quantity is compared against ADR-0007's floor, and it is not this one.**
[ADR-0021](../adr/0021-rim-to-core-detectability-is-a-bottleneck-ratio.md) states the transmission
predicate — whether a perturbation at the rim stays distinguishable from the floor on every edge of a
path to the apex, and back. It has nothing to do with this bound: that is a **bottleneck ratio**
against the quiescent-hold floor, this is a standing offset against a fold margin, and the only thing
they share is a word the two have now been separated on. The predicate lives there rather than here
because this file owns the reconciliation step, not transmission.

What survives is the **check** — and #160 moved it off construction as well. It is run per cell
across the taper on the same sweep, and what it produces is a **nomination**: the cap this body's draw
permits, before anything runs. See
[ADR-0007](../adr/0007-the-disagreement-floor-is-tolerated-not-represented.md), amended, and
[ADR-0019](../adr/0019-construction-nominates-the-run-decides.md).

**No claim is made about where the bound binds hardest.** This section used to argue that because
`Σ_e m_e` falls with depth, `gain_v` is largest at the apex and the check binds hardest exactly where
timescale matters most. [#190](https://github.com/NGL321/patchworks/issues/190) made `gain_v` uniform
across the interior, so it binds on each cell's own margin draw and nowhere in particular, and #160
**struck the claim without replacing it**: #158's offset profile down levels 1–7 is not monotone, and
[#178](https://github.com/NGL321/patchworks/issues/178) found the 30k reading was a local high of a
quantity that wanders 3.8x with no trend. Boundary cells were never in it — they run no body, so they
have no fold margin at all.

**The conversion loosened it and the two corrections since have tightened it further, on net 2.56x.**
Linearising `step` took its folds off the round trip, so the margin is `encode`'s alone rather than the
minimum over two maps, and the nominated cap on `gain_v × offset` rose from **0.2600 to 0.3502**.
[#195](https://github.com/NGL321/patchworks/issues/195) then re-measured it in the right space and
against #190's denominator, and it lands at **0.1369** — which reproduces `02`'s own published number
exactly. Measured on the construction sweep, 8192 draws, seed 42, at the `a = 1.0` the selection rig
returns.

The two corrections run in opposite directions and the tightening wins:

| reading | nominated cap |
|---|---|
| full `encode` input space, old denominator | 0.3502 |
| the node-stalk subspace alone | 0.4107 |
| #190's denominator alone | 0.1167 |
| **both, as the check now stands** | **0.1369** |

**The margin is read in the subspace reconciliation actually moves.** `encode` takes `R^k × R^n`, and
the check took its row norms over the whole of that input — but reconciliation writes the **node stalk
alone** and never the chart, so the displacement lives in the `R^n` block. Restricting the gradient
there is the margin in the space the offset moves in, and it is worth **1.183x** measured on the built
body across four budgets, against the isotropic expectation `√(44/32) = 1.173`. It was never unsafe:
the whole-space reading is uniformly *tighter*, so the standing check was too strict rather than too
loose, which is why it was left standing rather than treated as a live bug.

**`gain_v` is the subject of the bound, and `γ` was shorthand.** The implementation has always used the
per-cell gain; one sentence of this file never matched its own code. Under the old denominator the two
forms differed per cell by up to 2.6x, and the shorthand is what hid that the *"binds hardest at the
apex"* claim was a claim about the **denominator's shape** rather than about the check. #190 flattens
the denominator to a constant at every predicting cell, so the two forms now differ by one global
factor and which cell binds no longer depends on which is written. This is a correction to the prose,
not a third defect in the mechanism.

One rider, because the number is easy to misread. The 0.3502 is **larger than the 0.3278 first
reported**, which was the figure for linearising `step` alone; the full conversion linearises `decode`
too, and losing that map's hidden layer moves the operating point the margin is read at.

**And the apex does not bind.** The tightest cell moved to level 3 after the conversion, and at 100,000
ticks the apex is the **loosest** level on the built surface, 0.0540 against a graph median of 0.0192.
That is what a struck depth claim looks like when it is read: #190 predicted it in the abstract and
#195 measured it. A cell's fold margin is uncorrelated with everything else about it, so which cell
binds is largely a draw — and #195 found it does not even reproduce, four runs at one seed and one
budget giving four different binding cells and a 5.8x spread on the cap. **A per-cell extremum on this
surface is not a reproducible quantity**, which bears on anything in the record quoting one off a long
run.

**The denominator is settled at construction; the rest of the bound is not.**
[#33](https://github.com/NGL321/patchworks/issues/33) found the denominator could not stay fixed: the
transport rule ([`07-local-learning-rule.md`](./07-local-learning-rule.md)) trains the restriction-map
magnitudes `Σ_e m_e` stood in for, so the proxy drifted away from the block's true spectral radius as
training proceeded, and each cell had to re-derive its own estimate on the anneal schedule.
[ADR-0010](../adr/0010-restriction-map-scale-is-gauge-fixed.md) **removed that drift at its source** by
fixing the magnitudes the rule no longer identifies, so the denominator above is a bound that holds for
as long as the run does. The periodic re-derivation is struck rather than kept just-in-case — a
maintenance step retained "in case the bound slips" invites the build to trust a check that is no
longer being made. See ADR-0007, *Simultaneous learning does not need its own bound*.

**Both remaining sides of the bound move, so the check is read live** (#160,
[ADR-0019](../adr/0019-construction-nominates-the-run-decides.md)). The standing offset is dominated at
construction by model error and falls 144x through a run (#158); and the **folds themselves move**,
because their positions are the per-cell biases the prediction rule trains — one frozen set of
orientations, rigidly translated per cell, sliding under the operating point for the length of the run.
Neither is what #37 struck: that exchange is about the denominator, and this is about the arrangement.
ADR-0019 exists so the two do not read as the same decision reversed.

The live read is **free of new state and of any new time constant**, which is what sank every earlier
proposal: the margin's numerator is the pre-activation `encode`'s forward pass already computes, its
denominator is the shared frozen weight rows — one graph-wide constant — and the offset is the norm of
the displacement the message-passing phase already forms.
`patchworks.tick.FoldRead` is the instrument.

**Construction nominates, the run decides.** What the run decides on is **region dwell**
([`05-timescales.md`](./05-timescales.md)); the live margin-against-offset comparison is the
**attribution**, because dwell alone cannot say whether it was *reconciliation* that moved the cell,
which is the thing ADR-0007 forbids.

**`γ` stays at 1.0, the breach is standing, and the comparison carries no threshold.** #178 measured
only 0.90x permitted at 2,000 ticks, so the build starts outside its own bound — and
[#202](https://github.com/NGL321/patchworks/issues/202) measured that it never gets back inside, on a
read [#206](https://github.com/NGL321/patchworks/issues/206) then re-ran against the corrected
fold-margin denominator. Over 100,000 ticks **not one tick was free of a breaching cell**; all 150
cells breached at least once, 103 were still breaching after tick 90,000, and the density falls
28 → 15 cells and then **plateaus** (p05 9, median 15, p95 22 over the second half).
This section briefly stated the bound as holding *after a burn-in*; there is no such count, and
[#206](https://github.com/NGL321/patchworks/issues/206) struck the clause without replacing it.

**Nothing replaces it, and that is the point.** Under *Construction nominates, the run decides* above,
the verdict is measured region dwell and this comparison is the **attribution** — it says *why* a cell
lost its region, not whether the build is healthy, and an attribution has nothing to pass. A tolerance
(*no more than `k` cells breach*) was declined for wanting a constant read off one run's plateau.
`reconciliation_reaches` is **reported, never asserted**: expect it true, at any tick, on cells that
are perfectly healthy.

**The two readings are not in conflict.** The run found no clean tick *and* 131 of 150 cells clearing
[`05-timescales.md`](./05-timescales.md)'s dwell precondition at the horizon. Both hold because they
are readings of different quantities. The precondition is where the early breaches are accounted for:
dwell/`τ` is 0.96 at tick 100 and 86.2 at 100,000, so it **fails early and is earned over the run** —
which is what a cell whose region flips at tick 2,000 having no slow content to protect actually
buys.

**Both of those figures are the windowed estimator, and the pass condition is not.** Since
[#208](https://github.com/NGL321/patchworks/issues/208) dwell is published as the **cumulative** mean
residency to the horizon, on which the same run gives a median `dwell/τ` of **9.49** and **125 of
150** cells clearing `2.6 τ` — against 86.2 and 131 windowed over the last 25,000 ticks. The reading
above is unchanged in shape, and the pass condition it clears is `dwell > τ` on the **median** cell,
not the `2.6 τ` count. The estimator is named because leaving it unstated let one measurement
circulate as three different numbers.

**On `γ` itself:** ramping it is declined, because the density plateaus and a ramp targets a transient
the run does not have. A permanently lower `γ` is **no longer declined** — its ground was safety in a
window where nothing was at stake, and there is no such window — but nothing here adopts it either;
see [ADR-0019](../adr/0019-construction-nominates-the-run-decides.md), decision 5.

## Known exposure

- **Cross-tick settling, not within-tick settling.** Because message passing is a single step
  and every edge carries delay, any settling toward an equilibrium, an oscillation, or something
  chaotic in the more graph-distant regions is a multi-tick property of the recurrent chart
  interacting with delayed message passing — not something engineered by a round count. Expected
  to be observed during the build, not designed for here. Whether relay-cell topology can be used
  deliberately to shape this (shorten effective distance, change the settling regime) is a
  separate, later question.
