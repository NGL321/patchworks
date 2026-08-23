# Graph topology

The shape of the graph, its size, and where the world attaches — given the cell contract in
[`01-cell-and-sheaf.md`](./01-cell-and-sheaf.md), the tick in
[`02-tick-semantics.md`](./02-tick-semantics.md), the sandbox in
[`03-the-sandbox.md`](./03-the-sandbox.md), the boundary in
[`04-action-and-the-boundary.md`](./04-action-and-the-boundary.md), and timescales in
[`05-timescales.md`](./05-timescales.md).

Terms used here are defined in [`CONTEXT.md`](../../CONTEXT.md).

## The shape

### Why the disk failed

The working intuition inherited from `04-action-and-the-boundary.md` was a **disk**: external edges
around the rim, abstraction toward the centre, sensorimotor edges sharing one hemisphere. It does not
survive contact with the sensory tiling.

**A disk's rim is one-dimensional. The sensory tiling is two-dimensional.** The tiling exists so that
"adjacent patches are adjacent cells" and retinotopy falls out rather than being designed. A 16×16
grid of patches has two-dimensional adjacency, and two-dimensional adjacency does not embed in a
circle. Laying the patches around a rim in any order destroys the one property the tiling was for.

The disk was a two-dimensional cartoon of an object whose boundary is itself two-dimensional, which
puts the object one dimension higher.

### The dome

The shape is a **taper from a two-dimensional boundary sheet to a deep core**:

- **The base** is the sensorimotor boundary — the sensory patch lattice, with the proprioceptive,
  touch, and actuator boundary cells attached as a cluster at one region of it.
- **Levels inward** are successively coarser lattices, each connected locally to the level below and
  laterally within itself.
- **The apex** is a small deep core. This is `04-action-and-the-boundary.md`'s **internal rim**:
  internal faculties attach here — the drive boundary cell already does, and a limbic-analogue
  appetite or hippocampal-analogue memory would.

Everything handed down by `04-action-and-the-boundary.md` survives: one boundary set with depth away
from it, sensorimotor edges close together, internal faculties abstract by attachment point,
abstraction measured as hop distance from the sensorimotor base. What dies is only the claim that the
boundary is a circle.

**The dome is explicitly abandonable.** It is an intuition-derived shape, adopted because it is
cheap, legible, and satisfies every inherited constraint — not because anything proves it necessary.
If it fails, the fallback is a shape-free construction rule: a boundary set, a target depth, and
per-level degree constraints, with connectivity sampled to satisfy them. Everything else in this
document — the dimensions, the boundary exemption, the sparsity treatment, the absence of relays — is
independent of the dome and survives its abandonment.

### The construction layout is an index, not an embedding

Cells are indexed by **level** and **lattice position**. There is no metric embedding: no cell has a
coordinate, and no distance kernel generates the mask.

A Euclidean embedding was considered and rejected. It buys nothing an explicit rule does not — graded
`m_e`, hop depth, and a plot are all available without it — and it costs one real thing: it invites
confusion with the geometry of `01-cell-and-sheaf.md`, where "chart", "locally Euclidean", and
"piece" are load-bearing and have nothing to do with where a cell sits in a picture. The one property
an embedding uniquely supplies is a taper dictated by annulus area, and the taper is something to
choose rather than inherit from π.

The construction layout has **no runtime role**. It generates the structural mask at construction and
is thereafter used only for plotting.

## The levels

| level | contents | predicting cells |
|---|---|---|
| **L0** — boundary | 16×16 sensory patch cells; 3 proprioceptive; 3 touch; 1 actuator | — (263 boundary cells, no body) |
| **L1** | 8×8 vision lattice, each over a 2×2 block of patches; ~6 somatomotor | 70 |
| **L2** | 4×4 vision; ~4 somatomotor | 20 |
| **L3–L7** | core, five levels of ~16 / 14 / 12 / 10 / 8 | 60 |
| **internal rim** | 1 drive boundary cell, attached at L7 | — (1 boundary cell, no body) |
| | **total** | **~150 predicting, ~264 boundary** |

Every count is a **construction parameter**, not a constant of the architecture.

**Depth does not come from the vision cone.** The convolutional instinct is a stride-2 taper —
16→8→4→2→1 — which is four levels and leaves nowhere near enough hop depth for "recovered at the
appropriate level" to be a measurable claim. Tapering slowly enough to get eight lattice levels gives
over five hundred predicting cells with no argument for any of them. So vision tapers fast, in three
stages, and the depth lives in a core that is not a lattice at all. The rough biological parallel is
a few visual stages feeding association cortex.

### Why 150 and not 500

The sandbox's *state* is roughly twenty-dimensional — three joints × 2, three pucks × 4. Its
difficulty is concentrated in inverting the render and in contact dynamics. Enumerating
micro-problems by intuition reaches about a hundred, dominated entirely by the vision tiling, and no
amount of enumeration reaches a large interior.

The framing answer is that pieces are not enumerable by intuition: `01-cell-and-sheaf.md` has a
cell's piece as a **linear decomposition of a nonlinear global problem**, not a named subproblem, so
asking which five hundred things there are is like asking a decomposition to name its components in
advance. That answer is true and is also exactly the kind of argument that lets a number pass
unexamined, so it is not relied on alone.

The substantive answer is a **falsification sweep, written into the build**: halve the core and
re-run. If the acceptance demo is unaffected, the sandbox is not exercising the division thesis. That
is a finding worth having either way, and it costs one run. Whether the sandbox is the right size of
world at all is [#30](https://github.com/NGL321/patchworks/issues/30).

## Connectivity

- **Vertical.** Each cell connects to the cells it covers on the level below and to the cell covering
  it above. On the vision lattices this is a 2×2 block.
- **Lateral.** Four-neighbour within each vision lattice; sparse within the core.
- **Core.** Uniform degree ~6, no lattice.
- **Drive.** One edge from the drive boundary cell to each of the eight L7 cells — see *Where the
  drive attaches*, below.
- **Mean degree ~7** across predicting cells; roughly 698 edges in total.

### The core is uniform, and the integration/holding pair is held in reserve

`05-timescales.md` handed down that wide integration and slow holding may need to be separate cells:
an integrator of high degree feeding a holder of low degree. The core here is **uniform at degree
~6** instead.

The reason is ordering, not disagreement. `05-timescales.md` built the case for the pair *before*
observing the failure it repairs, and it presumes we know which cells need to hold before watching
any cell hold anything. The uniform core is built first; the demo already requires a live
private-component readout against hop distance, which is exactly the measurement that would show slow
state failing to appear in deep cells. If it does, the pair is a construction-parameter change.

Worth noting because it is the obvious objection: the pair would introduce no cell *types*. Degree is
topology, not contract. An integrator and a holder run the identical frozen body under the identical
contract, and `01-cell-and-sheaf.md`'s uniformity is untouched either way.

### Where the drive attaches

[ADR-0009](../adr/0009-a-drive-is-a-motor-edge-attached-deep.md) puts a **drive boundary cell** at the
core and left its attachment set open — "a subset of core cells", deliberately unchosen. It is chosen
here, because it is a structural-mask question and this is where the mask is fixed.

**The drive attaches at the apex level, entire: one edge to each of the eight L7 cells.** A rule, not a
hand-pick — the same standard the rest of this document holds itself to, and the reason no cell is
singled out.

Three things recommend the apex over any other core level:

- **It is the most abstract place there is.** Abstraction is hop distance from the sensorimotor rim
  (`04-action-and-the-boundary.md`), and ADR-0009 makes abstract action *literally a motor edge
  attached deep*. The apex is as deep as the graph goes.
- **It is where the slack is.** Guaranteed private dimension runs ~8 through L3–L6 and ~16 at L7. A
  drive edge spends private dimension at the cell it lands on, and the apex has roughly twice as much
  to spend as anywhere else in the core.
- **It leaves the gradient intact.** Attaching to every core cell would flatten the private-dimension
  gradient across the whole core, which is the thing the taper supplies for free (*Private dimension is
  a gradient*, below) and which `05-timescales.md` depends on.

**Drive edges are `m_e = 1`, and the drive cell's node stalk is 1-dimensional.** These are one
decision, not two. A restriction map out of a `d`-dimensional stalk has rank at most `d`, so an
`m_e` wider than the stalk cannot widen the channel — it only lets the drive assert **zero** in the
surplus directions, which is an arbitrary constraint on a core cell expressed in content the drive does
not have. Matching them makes the drive edge a clean scalar channel and makes *valence, not
specification* exact rather than approximate.

**Strength is fan-out, not width.** ADR-0009 recorded mask width as the strength knob; that is
corrected here. A drive edge's share of a cell's reconciliation pull is `m_e / Σ_e m_e`, so at the apex
one edge is worth about **6%**. Widening the edge raises that share but also **dilutes every other edge
the cell has**, by the same `1/Σ_e m_e` normalisation — at `m_e = 4` the drive would be turning down
the apex's ability to hear the rest of the graph by 20% in order to be heard itself, which is not
behaviour a strength knob should have. Fan-out is not free either — each new attachment point costs
*that* cell one dimension of privacy and the same ~6% dilution — but it **spreads** the cost across
cells instead of concentrating it, and it never makes any single cell pay more than the minimum. If the
drive proves too weak, the response is **more attachment points**, and only after that the learned
drive vector ADR-0009 holds as its escape hatch.

The cost is one dimension of privacy at each apex cell: `Σ_e m_e` goes 16 → 17 and guaranteed private
dimension 16 → 15. This is not free — [#25](https://github.com/NGL321/patchworks/issues/25) cashes the
apex's private dimension as the substrate of **commitment** — but it is the cheapest place in the core
to take it from, and the gradient still peaks at the apex.

[ADR-0007](../adr/0007-the-disagreement-floor-is-tolerated-not-represented.md)'s
`γ × floor <` fold margin bound gets **slacker**, not tighter: the per-cell gain is
`γ / Σ_e m_e`, so an extra edge lowers it. The apex check is about 6% easier to satisfy than before.

### The somatomotor column, and where the modalities meet

Proprioception, touch, and the actuator form a **parallel column** — their own cells at L1 and L2 —
and vision and proprioception do not share a cell until **L3, the first core level**.

Two consequences, both wanted:

- **The reflex loop is three ticks and purely somatomotor.** Proprioceptive boundary cell → an L1
  somatomotor cell → actuator boundary cell. `04-action-and-the-boundary.md` made the shortest
  sensorimotor loop a design budget; here it is satisfied by construction rather than by hand-wiring,
  and a corrective twitch never waits on vision.
- **Cross-modal binding is a function of depth.** "My tip is near the red thing" cannot be
  represented below L3, so a correction requiring visual context is at least four hops out and back.

The second gives the acceptance demo its sharpest falsifiable form yet. It predicts **two different
depths for two different perturbations**: a perturbation to the arm alone should be corrected at
roughly one hop, a perturbation that moves a puck at roughly four. "Recovered at the appropriate
level of the hierarchy" becomes a comparison rather than a single number, which hands
[#10](https://github.com/NGL321/patchworks/issues/10) a real criterion for choosing the demo
perturbation.

Merging the modalities at L1 was rejected: a two-dimensional joint signal against a
forty-eight-dimensional patch would be swamped, and the likely outcome is that only the bare minimum
of reflexive behaviour concentrates at low depth. Lateral cross-modal edges at L2 were rejected as
hand-specified wiring for something that should follow from position.

## Dimensions, and the boundary's exemption

| quantity | value |
|---|---|
| node stalk of a predicting cell, `n` | 32 |
| chart, `k` | 12 |
| typical interior edge stalk, `m` | 4 |
| boundary edge stalk, `m` | 8 |
| drive edge stalk, `m` | 1 |
| sensory patch | 4×4 px RGB → node stalk **48** |
| proprioceptive cell | node stalk **2** (angle, velocity) |
| actuator cell | node stalk **6** (3 commanded, 3 efference) |
| drive cell | node stalk **1** |

**Boundary cells are exempt from `n`. Their node stalk is the world's shape.**

The world writes or reads a boundary cell's **node stalk** directly, and there is no compressor
between the render and that stalk — putting one there is the act
`04-action-and-the-boundary.md` bans. So the patch must fit in the stalk, raw. With `n` a global
constant, an 8×8 patch would force a 192-dimensional stalk on *every cell in the graph*, handing each
interior cell some 170 private dimensions to fill with junk in a world whose state is about twenty
numbers.

The exemption is cheap rather than a concession: `n`'s globality exists so the shared frozen body can
run **batched** ([ADR-0001](../adr/0001-continual-learning-applies-to-the-adapting-surface.md),
`01-cell-and-sheaf.md`), and boundary cells run no body — they perform no inference and were never in
the batch. See [ADR-0006](../adr/0006-boundary-cell-stalks-are-world-shaped.md).

**The world only ever touches node stalks.** There is no edge between the world and the graph; the
world is not a cell and holds no restriction map. Every edge stalk in the graph, including those
incident on boundary cells, is ordinary and `m`-sized, reached by an ordinary linear masked
restriction map. A patch cell's 48 → 8 restriction *is* the compression of that patch, performed by a
cell, inside the graph, costing a tick.

Boundary edges are given `m = 8` against the interior's 4 for a specific reason: a patch cell's edges
are the only route that patch's information ever takes, unlike an interior cell, which is reachable
many ways.

**The drive edge is the exception, and in the other direction.** It is a boundary edge at `m = 1`,
because the argument above is about *bandwidth the world needs* and the drive is not the world: it
asserts one number, and an edge stalk wider than its stalk carries nothing extra (*Where the drive
attaches*). The drive cell is likewise exempt from `n` without being world-shaped — nothing outside
gives it a dimension, so its stalk is sized by what it asserts. See
[ADR-0006](../adr/0006-boundary-cell-stalks-are-world-shaped.md).

### Tiling granularity

4×4 pixel patches, 16×16 = 256 sensory boundary cells. The arena is 1.04 m across 64 px, so pucks are
**4.3 / 5.5 / 6.8 px** wide.

**No cell ever sees a whole puck.** Recomposing one requires reconciliation across two to four
neighbouring patch cells, on every puck, every tick. This is the patchwork thesis exercised at the
seam rather than decoratively, and it is the direct answer to the worry that the sandbox may be too
simple to need division.

8×8 patches were rejected twice over: a puck would fit inside one patch, making vision easier in
exactly the way that lets the architecture pass without being tested, and the boundary restriction
map would be squeezing 192 → 8 in a single linear step.

### Private dimension is a gradient, and it falls out

`05-timescales.md`'s bound `dim H⁰ ≥ Σ_v max(0, n − Σ_{e∋v} m_e)` applied to the levels above:

| cell | `Σ_e m_e` | guaranteed private dimension |
|---|---|---|
| L1 vision | 4×8 + 4×4 + 4 = 52 | 0 |
| L2 vision | 4×4 + 4×4 + 4 = 36 | 0 |
| L3–L6 core | ~24 | ~8 |
| L7 apex | ~16 + 1 drive edge = 17 | **15** |

**Guaranteed private dimension is zero at the rim and rises to about fifteen at the apex.** Slow state
lives deep by construction, which is precisely the gradient `05-timescales.md` wanted and did not have
a mechanism for. It is not designed here; it falls out of the taper, because rim-adjacent cells are
necessarily high-degree — an L1 vision cell must read four patches — and depth reduces degree.

Zero *guaranteed* private dimension is not zero private dimension: the bound is a lower bound, and
learned rank-deficiency enlarges `H⁰` past it. What the gradient says is that near the rim a cell's
privacy is contingent on learning, and deep it is structural.

`χ = Σ_v n − Σ_e m_e` over predicting cells is approximately **+980**. The eight drive edges move it
by 8, which is inside the rounding on every other number here; the drive's cost is local to the apex
cells, not to the diagnostic.

**`χ` must be computed over predicting cells only.** Including boundary cells swamps it — 256 cells ×
48 dimensions of nominally private state that the world overwrites every tick and no cell holds. This
is a correction to the diagnostic as recorded in `01-cell-and-sheaf.md`.

## Sparsity is a property of the maps, not of the graph

**No edge is ever removed.** `01-cell-and-sheaf.md` fixes `m` at construction and closes the mask
permanently; the sparsity pressure prunes *within* the mask, driving restriction-map weights toward
zero without shrinking a stalk.

So "sparsity annealing" is a **schedule on the sparsity pressure**, not a structural process. A fully
zeroed edge is functionally dead but structurally present: it still costs a tick, still contributes
`m_e` to `χ`, and — through learned rank-deficiency — still enlarges `H⁰`, which is the effect
`05-timescales.md` actually wanted from sparsity. Fragmentation is therefore never structural and
needs no guard.

Structural edge deletion was considered. It runs the same direction as pruning and is not growth, and
it would buy compute back. It was rejected because it makes `χ` move mid-run, and `χ`'s invariance
under learning is a recorded diagnostic — at 150 predicting cells, compute is not the binding
constraint and the invariant is worth more.

## Broadcast subspaces, and why there are no relay cells

`CONTEXT.md` defines a **relay cell**: inference is the identity, it holds stalks and restriction maps
only, it exists to give distant cells a shared metric space. The idea comes from two places. The
biological one is interregional communication in neocortex being largely one-to-one through
dimensionally-reduced *messaging subspaces*, with the rarer many-to-many **broadcast subspaces**
almost all corticothalamocortical. The functional one is the documented difficulty of long-range
transport in graph neural networks.

**The proof of concept builds none.** Three findings, in order of weight:

- **The core already is the broadcast subspace.** Every region is within about four hops of it, it is
  low-dimensional, and it is the shared space through which distant regions communicate. That is
  structurally the corticothalamocortical picture, and a better match to it than chords across the
  surface would be — the thalamus is a deep low-dimensional shared space, not a shortcut wire.
- **Relays solve reach, and reach is not what is squeezed.** The graph's diameter is about **9 hops**
  between the most distant predicting cells, about 11 world-to-world. The real bottleneck is the
  taper: 256 patches × 48 dimensions funnelling into ~60 core cells at `n = 32` through edge stalks of
  `m = 4`. That is severe oversquashing, and a relay makes distant things closer without widening the
  funnel.
- **Scale.** At 150 cells this is nearer a fly's brain than a mammal's. Among 150 columnar units of
  neocortex one would not expect significant thalamic communication either.

**The pre-specified intervention**, if measurement shows two core regions that must agree failing to:
a small set of relay cells attached across L4–L6 at degree ~10, spanning sectors. Several narrow
relays rather than one wide one, since a single relay carries everything through one `n`-dimensional
stalk.

Two things rejected outright:

- **Relays as chords across the vision levels.** That is reach, which is not the problem, and a relay
  near the rim collapses hop distance — which is the abstraction measure and the acceptance demo's
  yardstick.
- **Making the apex relay cells.** The apex is the deepest, lowest-degree region and therefore where
  `05-timescales.md`'s persistence most wants to live. Identity-`step` cells there would evict slow
  state from the one place the timescale mechanism counts on.

**No new object is needed.** A hyperedge — one stalk many cells restrict into simultaneously — would
break the cellular sheaf on a graph and rewrite `01-cell-and-sheaf.md`. It is not on the table.

## Hard partitions are held in reserve

Explicit divisions communicating only through designated channels — neocortical-lobe style, enforced
through the structural mask rather than as new mechanism — are **not built**.

The taper already localises: cells at distant lattice positions simply do not connect, and distance
induces latency. What a hard partition adds beyond that is a *guarantee* that two regions communicate
only through one channel, which is hand-designed structure in a graph whose thesis is that role
emerges from proximity to boundaries. Excluding the lobes, undifferentiated-with-distance may also be
the more faithful reading of mammalian neocortex.

**The trigger, recorded so the reserve is usable:** if cells at every depth respond to everything, the
graph needs constraining and partitions are the constraint.

## What the cycles do

A lattice has an enormous cycle count — `β₁` for a bare 16×16 four-neighbour grid is 225, and the
vertical edges add more. This is **not** a cost, and an earlier reading that treated it as one was
wrong.

**Nothing in Patchworks ever experiences `H¹`.** No cell reads it; no rule branches on it.
`01-cell-and-sheaf.md` already commits to disagreement being **penalised, never cleared**, and to
residual disagreement *being* the signal the local learning rule consumes. More disagreement that no
node-stalk assignment can zero is not a cost in a system that never zeroes disagreement. The global
space is meant to be nonlinear and complex; the linearity claim is only ever local.

The positive form: **a cycle is what makes disagreement informative.** On a tree, every disagreement
can be explained away by moving node stalks — the sheaf tells you nothing the nodes could not have
arranged. A cycle is where two independent routes make claims that must be reconciled. High `β₁` is
where the sheaf earns its place, and it is what the lateral edges are for.

Two residual costs, both smaller than performance costs:

- **The `H¹` diagnostic dulls.** [ADR-0004](../adr/0004-linear-restriction-maps-assume-local-flatness.md)
  makes persistent structured irreducible disagreement a falsification signature — curvature the
  linear map cannot follow. If topology guarantees plenty of irreducible disagreement regardless, that
  attribution needs a **topology-only baseline** to compare against rather than a comparison with
  zero. Recorded as a measurement requirement.
- **A raised energy floor** slightly compresses signal-to-noise for the local learning rule
  ([#5](https://github.com/NGL321/patchworks/issues/5)), largely absorbed if thresholds are derived
  relative to an edge's own scale rather than set absolutely — which the change gate also requires,
  though it takes that scale from the restricted belief's *current* magnitude rather than from a
  running average ([`05-timescales.md`](./05-timescales.md), *The change gate, pre-specified*).

This softens half of the wheel critique recorded in `01-cell-and-sheaf.md`. Its load-bearing
objections survive intact — a rim-adjacent hub collapses the diameter and destroys the abstraction
measure, and `n` being a global constant means no predicting cell can be dimensioned as a hub. Its
`Δ(dim H¹) ≥ N·m − n_h` half belongs in the same diagnostic register as everything else about `H¹`.

## Known exposure

- **This is a convolutional pyramid with lateral and top-down edges**, and the shared frozen body
  means there is weight sharing across position too. Structurally it is close to a recurrent CNN, and
  saying otherwise would be the kind of claim a citation pass demolishes. The connectivity is not
  where Patchworks differs; the argument was never that the wiring is novel. What differs is
  mechanism on the same connectivity: no pooling (tapering is fewer cells, not an aggregation
  operator), bidirectional edges with unit delay rather than a forward pass, a per-cell learned linear
  change of basis into each shared edge stalk rather than a shared kernel, reconciliation rather than
  activation, and local rules rather than backprop.
- **Translation equivariance is absent.** A CNN applies one kernel at every position; here every cell
  holds its **own** restriction maps, so the change of basis is per-position. Any result from the
  convolutional literature that rests on weight-shared kernels is not ours to borrow. What does
  transfer is receptive-field arithmetic and taper-schedule practice.
- **The lateral edges are the load-bearing difference from a CNN.** A convolutional network can only
  resolve cross-patch inconsistency by going deeper — there is no mechanism at a level for two
  neighbouring units to agree. Lateral edges plus reconciliation supply exactly that, at that level,
  in one tick. If the dome is worth trying, that is why, and if lateral reconciliation turns out to
  contribute nothing measurable, the dome's main justification is gone.
- **The taper is the real bottleneck**, not distance: 12,288 numbers at the base reaching ~60 core
  cells of dimension 32. Nothing in this document fixes that, and no relay would.
- **Vision dominates the boundary.** 256 of 264 boundary cells are visual. If the acceptance demo
  turns out to rest on proprioception, the graph is built around the wrong modality.
