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
document — the dimensions, the boundary exemption, the refusal of edge removal, the absence of
relays — is independent of the dome and survives its abandonment.

**That rule has since been exercised, and the sentence above is under-specified.**
[#130](https://github.com/NGL321/patchworks/issues/130) produced a second topology — the **wedge**,
for a rim that is a character stream — and
[`11-the-language-graph.md`](./11-the-language-graph.md) expresses both it and the dome in one
construction rule rather than writing a second graph by hand. Doing that found the fallback missing
two ingredients: a **covering rule** and a **lateral rule**. *Sampled* connectivity cannot express
this document's own load-bearing properties — a degree-constrained sample reproduces the dome's degree
table exactly and its retinotopy with probability zero, and no degree constraint can say "no cell
sees a whole puck", which is a statement about the covering. The claim in the paragraph above holds
otherwise, and holds in the stronger form that a second domain has now instantiated it: what is
listed here as surviving the dome's abandonment did survive it.

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

The sandbox's *state* is roughly twenty-dimensional — three joints × 2, three pucks × 4 — and
enumerating micro-problems by intuition reaches about a hundred, dominated entirely by the vision
tiling. **That framing is retired** by [#30](https://github.com/NGL321/patchworks/issues/30): the
state is the answer key, and the agent is never given it. What the graph faces is 12,288 render
numbers plus six proprioceptive and touch channels in, three torques out, with the twenty dimensions
as something it must invent. Difficulty is concentrated in inverting the render and in contact
dynamics.

The framing answer is that pieces are not enumerable by intuition: `01-cell-and-sheaf.md` has a
cell's piece as a **linear decomposition of a nonlinear global problem**, not a named subproblem, so
asking which five hundred things there are is like asking a decomposition to name its components in
advance. That answer is true and is also exactly the kind of argument that lets a number pass
unexamined, so it is not relied on alone.

The substantive answer is a **falsification sweep, written into the build**, in **two
conditions**. #30 found the single-condition version confounded: it varied the core, which is not
where the forced division lives.

- **Halve the core.** Rebuild with L3–L7 at half their cell counts and re-run. If the acceptance demo
  is unaffected, the core is oversized — a finding about *this topology*.
- **Coarsen the tiling.** Rebuild the sensory base at 8×8 px patches, so a patch cell does see a
  whole puck, and re-run. If the acceptance demo is unaffected, the division the 4×4 tiling was
  chosen to force was decorative — a finding about *the division thesis*.

Only the second tests the thesis this section is defending, because the 4×4 tiling is the thing
deliberately built to make division unavoidable and "halve the core" leaves L0–L2 untouched. Both are
findings worth having either way, and each costs one run.

Whether the sandbox is the right *size* of world was #30's question, and it is answered there: by the
measure that matters — **precedence depth**, the longest chain of sub-goals that must be reached in
order — the sandbox is thin, deliberately so, and enriching it belongs to a second proof of concept
rather than to this one.

## Connectivity

- **Vertical.** Each cell connects to the cells it covers on the level below and to the cell covering
  it above. On the vision lattices this is a 2×2 block.
- **Lateral.** Four-neighbour within each vision lattice; sparse within the core.
- **Core.** Uniform degree ~6, no lattice — **except the apex, at ~4**. L7 has no *predicting* level
  above it to connect up to, so its cells lose their up-edges by construction rather than by
  stipulation. This is what the private-dimension table below depends on, and the two statements
  contradicted each other until now: at a uniform degree ~6 the apex would carry `Σ_e m_e = 24` and a
  guaranteed private dimension of 7, flat with the rest of the core, and the slack the drive is
  attached *for* would not exist.
- **Drive.** One edge from the drive boundary cell to each of the eight L7 cells — see *Where the
  drive attaches*, below.
- **Actuator.** One motor edge from the actuator boundary cell to the L1 somatomotor cell covering
  **each** joint's proprioception — three, at `m = 8`. Every other sensorimotor boundary cell has
  exactly one edge; the actuator is the single exception, and it is what makes the reflex loop three
  ticks at every joint rather than at whichever one happened to share the actuator's L1 cell
  ([#83](https://github.com/NGL321/patchworks/issues/83)).
- **Mean degree 7.27** across predicting cells; **682 edges** in total. Both are measured from the
  built graph ([#83](https://github.com/NGL321/patchworks/issues/83)) rather than estimated: the
  earlier "~698" was an estimate, and the connectivity rules above cannot place that many without
  raising core degrees, which would flatten the private-dimension table below.

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

**What the apex has above it is the internal rim, and it is a budget.** "No level above" is true only
of *predicting* levels: the apex is precisely where internal faculties attach, and one already does.
The drive boundary cell is there now; `04-action-and-the-boundary.md`'s hippocampal-analogue is the
other candidate, and it would attach the same way, as an ordinary boundary cell at the same rim. Each
such attachment is another incident edge on every apex cell, so the apex's privacy is a **budget spent
by the internal rim**, not a fixed figure: `Σ_e m_e` 16 → 17 → 18 and guaranteed private dimension
16 → 15 → 14 as faculties are added. At scalar stalks and `m_e = 1` this is cheap, which is a second
reason the width decision above matters; it is also the quantity to check before a *third* faculty is
attached, since [#25](https://github.com/NGL321/patchworks/issues/25) cashes exactly this dimension as
**commitment**.

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

- **The reflex loop is three ticks and purely somatomotor, at every joint.** Proprioceptive boundary
  cell → an L1 somatomotor cell → actuator boundary cell. `04-action-and-the-boundary.md` made the
  shortest sensorimotor loop a design budget; here it is satisfied by construction rather than by
  hand-wiring, and a corrective twitch never waits on vision. **At every joint** is what costs the
  actuator its three edges (*Connectivity*, above): with one, only the joint sharing its L1 cell
  reflexes in three ticks and the other two take four, which is a design budget met for one third of
  the arm.
- **Cross-modal binding is a function of depth.** "My tip is near the red thing" cannot be
  represented below L3, so a correction requiring visual context is at least four hops out and back.

The second gives the acceptance demo its sharpest falsifiable form yet. It predicts **two different
depths for two different perturbations**: a perturbation to the arm alone should be corrected at
roughly one hop, a perturbation that moves a puck at roughly four. "Recovered at the appropriate
level of the hierarchy" becomes a comparison rather than a single number, which gave
[#10](https://github.com/NGL321/patchworks/issues/10) its criterion for choosing the demo
perturbation. It chose a **three-event staircase** — an arm impulse, a puck teleport, a retarget —
and cashed the reflex-loop figure as the shallow rung's expected **onset latency**:
[`08-the-acceptance-demo.md`](./08-the-acceptance-demo.md).

**Read those depths as an upper bound on locality, not as a prediction of where influence actually
lands.** The one-hop / four-hop figures are *nominal* receptive-field arithmetic, and Luo et al. (2016)
find the effective receptive field is Gaussian-distributed and occupies only a fraction of the
theoretical one — so real influence is more concentrated than the count suggests. Their derivation
also assumes a feedforward stack, which the dome's bidirectional unit-delay edges are not, so it
transfers as a caution rather than a correction. The comparison survives either way, because what the
demo claims is that the two depths **differ**, not that either equals its nominal value.

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

**`n` and `k` have a definition site, and it is not `DomeSpec`.** They live as module-level constants
`NODE_STALK_DIM` and `CHART_DIM` in `src/patchworks/body.py`, the module that consumes them
([#186](https://github.com/NGL321/patchworks/issues/186)). Every other count in this table is a
construction parameter on `DomeSpec`; these two are not, because they size the *shared frozen body*
that both domains run rather than this graph's topology, and
[#128](https://github.com/NGL321/patchworks/issues/128) fixed one of each across both domains. The
three constant registers under `docs/registers/` carry their provenance.

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

**Of the four committed dimensions, `m = 4` has the least theoretical headroom.**
[#32](https://github.com/NGL321/patchworks/issues/32) found `n = 32`, `k = 12` and `m = 8` all
comfortable and `m = 4` thin: the one dimensionally commensurate body of theory (delay embedding) puts
the governing quantity at **twice the box-counting dimension** of the piece being carried, so an
interior edge stalk of 4 supports a shared piece of box dimension under 2, and **no source was found
either way** on whether that is enough. Recorded rather than acted on, for two reasons: `m` is the
*first* rung on [#14](https://github.com/NGL321/patchworks/issues/14)'s constraint ladder, so it is the
cheapest exposure in the design to carry; and widening it trades against delay and size directly —
every interior stalk widened raises `Σ_e m_e` at every cell, which lowers the per-cell reconciliation
gain and eats the private dimension the taper exists to produce. If a piece turns out not to fit
through 4, that is the rung to pull first.

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

| cell | degree | `Σ_e m_e` | guaranteed private dimension |
|---|---|---|---|
| L1 vision (interior) | 4 down + 4 lateral + 1 up = 9 | 4×8 + 4×4 + 4 = 52 | 0 |
| L1 vision (lattice corner) | 4 + 2 + 1 = 7 | 32 + 8 + 4 = 44 | 0 |
| L2 vision (interior) | 4 + 4 + 1 = 9 | 4×4 + 4×4 + 4 = 36 | 0 |
| L2 vision (lattice corner) | 4 + 2 + 1 = 7 | 16 + 8 + 4 = 28 | **4** |
| L3–L6 core | ~6 | ~24 | ~8 |
| L7 apex | ~4 + 1 drive edge | ~16 + 1 = 17 | **15** |

**Guaranteed private dimension is zero at the rim and rises to about fifteen at the apex.** Slow state
lives deep by construction, which is precisely the gradient `05-timescales.md` wanted and did not have
a mechanism for.

Two corrections to how that sentence used to read, neither of which moves the headline:

- **"Zero at the rim" has exceptions at the corners.** The four corner cells of the 4×4 L2 lattice
  have only two lateral neighbours, which leaves them `Σ_e m_e = 28` and a guaranteed private
  dimension of **4**. Small, and it argues nothing — but the unqualified "zero at the rim" was false,
  and the four cells with structural privacy at L2 are worth knowing about if the private-component
  readout is ever run per-cell rather than per-level.
- **It is a step, not a ramp.** 0 at the vision levels, ~8 flat across L3–L6, 15 at the apex. Degree
  falls at the apex and nowhere else in the core, so nothing about this gradient is smooth.
  [#41](https://github.com/NGL321/patchworks/issues/41) already half-said this from the other
  direction — the gradient is one in *means*, with adjacent depths overlapping per tick — and the
  structural picture agrees: what the taper buys is deep-versus-shallow, not a graded ordering
  through the core. It is not designed here; it falls out of the taper, because rim-adjacent cells are
necessarily high-degree — an L1 vision cell must read four patches — and depth reduces degree.

Zero *guaranteed* private dimension is not zero private dimension: the bound is a lower bound, and
learned rank-deficiency enlarges `H⁰` past it. What the gradient says is that near the rim a cell's
privacy is contingent on learning, and deep it is structural.

`χ = Σ_v n − Σ_e m_e` over predicting cells is **+1036**, measured from the built graph. The eight
drive edges move it by 8; the drive's cost is local to the apex cells, not to the diagnostic.

**`χ` restricts the node sum only. Every edge in the graph still counts.** The node term runs over
predicting cells alone — including boundary cells swamps it, 256 cells × 48 dimensions of nominally
private state that the world overwrites every tick and no cell holds — but the edge term runs over
**all** edges, boundary-incident ones included, because those edge stalks are ordinary and are the
route the boundary's information actually takes (*The world only ever touches node stalks*, above).
The rule was previously stated as "computed over predicting cells only", which licensed the wrong
computation: dropping boundary edges as well as boundary nodes gives χ ≈ +3200, three times the figure
this section carries. This is a correction to the diagnostic as recorded in `01-cell-and-sheaf.md`.

The value is edge-count sensitive and is not a target. This section used to carry ~980 against a
~698-edge estimate and note that a ~663-edge count would give ~+1096 — a spread it owned in advance
and declared not to matter. The built graph's 682 edges land inside that band at **+1036**, so the
estimates are simply retired in favour of the measurement
([#83](https://github.com/NGL321/patchworks/issues/83)). Nothing is contradicted, because what is
load-bearing about `χ` is its **invariance under learning**, not its value.

## No edge is ever removed

**No edge is ever removed.** `01-cell-and-sheaf.md` fixes `m` at construction and closes the mask
permanently. Nothing in the run deletes an edge, narrows a stalk, or re-opens a masked feature, and
nothing later may either.

**This section used to carry a mechanism, and the mechanism is deleted.** It read *sparsity is a
property of the maps, not of the graph*: a sparsity pressure — an L1 on the normalised map
([ADR-0010](../adr/0010-restriction-map-scale-is-gauge-fixed.md)) — pruned *within* the mask, so
"sparsity annealing" was a **schedule on the sparsity pressure** rather than a structural process, and
a fully zeroed edge was functionally dead but structurally present. [#406](https://github.com/NGL321/patchworks/issues/406)
deleted the term from the transport rule outright; [ADR-0031](../adr/0031-the-sparsity-pressure-is-deleted.md) carries the
grounds. Two claims went with it and are struck rather than restated: that the pressure *enlarges
`H⁰` through learned rank-deficiency*, measured at **+1.3%** and exactly zero below `λ = 0.3`
([#393](https://github.com/NGL321/patchworks/issues/393)); and that this was *"the effect
`05-timescales.md` actually wanted from sparsity"*, which it was not — `H⁰` is a per-cell floor set
by the masks, and the holding job moved to `K` at
[#143](https://github.com/NGL321/patchworks/issues/143)/ADR-0028.

What survives is this section's actual subject, and it never depended on the term: **fragmentation is
never structural and needs no guard.** An edge whose map has learned its way to nothing is still an
edge — it costs a tick and still contributes `m_e` to `χ` — because the graph has no removal path at
all, not because a pressure was careful about how it pruned.

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

**The two halves of that biological premise have very different standing, and the spec should not
write them with equal confidence.** The messaging half is published under the name *communication
subspace*: Semedo et al. (2019) identify a low-dimensional subspace of source-population fluctuations
most predictive of target-population fluctuations and propose it as a general mechanism for
selectively routing activity between areas, and Kang et al. (2024) extend it to whole-cortex models,
where long-range projections "selectively route a small number of dimensions of neural dynamics while
maintaining others private". The **broadcast** half — the many-to-many corticothalamocortical
counterpart — has **no source in print**:
[#31](https://github.com/NGL321/patchworks/issues/31) searched for it and found nothing, and Kang et
al. explicitly scope it out, being corticocortical and without cell types. It is **unpublished
conference-talk provenance**, and the first finding below rests on it. That is not a reason to drop
the finding — it is a reason not to let it borrow the authority of the cited material next to it.

**The proof of concept builds none.** Three findings, in order of weight:

- **The core already is the broadcast subspace.** Every region is within about four hops of it, it is
  low-dimensional, and it is the shared space through which distant regions communicate. That is
  structurally the corticothalamocortical picture, and a better match to it than chords across the
  surface would be — the thalamus is a deep low-dimensional shared space, not a shortcut wire.
  **Weight this by its provenance** (above): the analogy is to an object that is unpublished, so this
  is the weakest-sourced of the three findings despite being the first, and it is doing structural
  rather than evidential work.
- **Reach and squeeze are the same quantity, and the relays that would relieve it are the ones
  rejected below.** An earlier version of this finding separated them — "relays solve reach, and reach
  is not what is squeezed" — and indexed reach by the graph's ~9-hop diameter. That separation does not
  survive the literature. Over-squashing is indexed by **commute time / effective resistance**, not hop
  distance (Di Giovanni et al. 2023, which runs exactly the width/depth/topology decomposition this
  section needs and finds topology dominant); by that measure the taper *is* the reach problem, stated
  in the one unit the field uses.

  The quantitative form is the **per-tick capacity of each cut**, which is this section's own best
  evidence: `12,288 → 2,120 → 280 → 80`. The entire sensory boundary reaches the core through
  **eighty numbers per tick**, a 154:1 squeeze at a single cut, while the two farthest predicting
  cells are only ~9 hops apart. Both readings live in that number and the tension is owned rather
  than resolved by choosing a favourable metric.

  What still holds is the conclusion, on arithmetic rather than on the discarded distinction.
  Effective resistance from rim to rim is a **series** quantity, and it is dominated by the single
  L2→L3 cut — twenty edges at `m = 4`. The pre-specified relays sit across **L4–L6, entirely inside
  the core**, above that cut: they add parallel paths to a term that is not the binding one, and
  leave the cut untouched. The relays that *would* parallel it are **chords across the vision
  levels**, and those are rejected below at a price this design is not willing to pay — they collapse
  hop distance, which is the abstraction measure and the acceptance demo's yardstick.

  Read the other way, this **vindicates the pre-specified intervention** rather than weakening it:
  relays across L4–L6 are correctly aimed at the core-internal resistance term, which is the term
  relays can actually lower, and they are already gated on the measurement that would show it binding
  — two core regions that must agree failing to.

  **One mitigation is available in the literature and is deliberately not claimed.** Arroyo et al.
  (2025) restate over-squashing unchanged for the recurrent setting and factor sensitivity into a
  topology term and a **model-dynamics** term, mitigated by direct control of the Jacobian spectrum —
  and Patchworks is recurrent with unit-delay bidirectional edges, so the term exists here. It is not
  claimed because the spectrum is **already spoken for three times over**: the body is shared and
  frozen, the regional spectrum is deliberately *spread* to supply timescales
  ([`05-timescales.md`](./05-timescales.md), [#27](https://github.com/NGL321/patchworks/issues/27)),
  spread and stability are the same global knob
  ([#42](https://github.com/NGL321/patchworks/issues/42)), and `γ` is fixed for stability and recorded
  as explicitly not a timescale knob
  ([ADR-0007](../adr/0007-the-disagreement-floor-is-tolerated-not-represented.md), promoted to a
  precondition by [#41](https://github.com/NGL321/patchworks/issues/41)). Buying long-range
  sensitivity there would spend what the timescale mechanism runs on. Recorded as a route, with the
  reason it is shut.
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
  `05-timescales.md`'s persistence most wants to live. Relay cells there — cells whose `K` is left at
  the identity — would evict slow state from the one place the timescale mechanism counts on.

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

  This is the best-cited claim in the document and the caveat is in the same sources. Linsley et al.
  (2018) show a single recurrent lateral layer matching or beating feedforward hierarchies with orders
  of magnitude more parameters, and Spoerer et al. (2017) show recurrence beating parameter-matched
  feedforward controls at recognising occluded objects. But Spoerer's **clutter** condition — several
  overlapping objects, the closest published analogue to two patch cells disagreeing about one puck —
  is exactly where their recurrent advantage is **weakest**: recurrent networks stay better in
  absolute terms but take similar hits to the error rate. The nearest prior art supports the dome's
  main justification least strongly on the condition the dome cares most about, and the 4×4 tiling is
  chosen specifically to manufacture that condition.

- **The dome is a locally connected network, which is the configuration the literature warns about —
  and the shared frozen body is what keeps it off the bad extreme.** Elsayed et al. (2020) define
  locally connected layers as differing from convolutional ones *only* in lacking spatial invariance,
  and report that they usually perform poorly; the mechanism is statistical, not representational —
  they are strictly more powerful and could in principle converge to the convolution solution, but in
  practice they **overfit**, because nearby regions share statistical structure that a shared kernel
  exploits and per-position parameters throw away. Per-cell restriction maps put the dome squarely in
  that class.

  Two things answer it, and one does not. **The body is shared**: the nonlinearity is weight-shared
  across every cell in the graph (ADR-0001, for the unrelated reason that it must batch), and what
  varies per position is a thin linear change of basis — a masked 32→4 restriction map is 128
  parameters. That is the same *strategy* as the low-rank locally connected layer Elsayed et al.
  propose, whose whole finding is that partial relaxation of spatial invariance beats **both**
  convolution and full local connectivity, though it is not their construction (they vary combining
  weights over a basis of filter banks; here there is one shared body and the variation is in the maps
  into it). **And the sample-scarcity premise does not bind**: "typical datasets are too small to
  constrain the parameters of a locally connected layer" is a claim about datasets, and there is no
  dataset here — experience is generated by interaction with the sandbox, so the sample budget is
  unbounded by construction (`03-the-sandbox.md`; distinct from, and not implied by, that contract's
  absence of episodes).

  What is *not* answered: the **biases are per-cell too**, and `05-timescales.md` already records them
  as over-subscribed — three geometrically distinct jobs on one vector. Overfitting is precisely the
  pressure that makes an over-subscribed per-cell parameter worse, so this is a second and independent
  argument for pulling per-cell adapters off
  [#14](https://github.com/NGL321/patchworks/issues/14)'s constraint ladder early.
- **The taper is the real bottleneck**, and *distance is not a separate thing from it*: the cut
  capacities run `12,288 → 2,120 → 280 → 80`, so the whole sensory boundary reaches the core through
  eighty numbers per tick. In the literature's own units that single narrow cut **is** the high
  effective resistance between distant cells, which is why this document no longer argues that reach
  and squeeze are different problems (*Broadcast subspaces*, above). Nothing here fixes it; the relays
  that could are the ones rejected for collapsing the abstraction measure, and the deep relays that
  are pre-specified address a different, smaller term.
- **Vision dominates the boundary.** 256 of 264 boundary cells are visual. If the acceptance demo
  turns out to rest on proprioception, the graph is built around the wrong modality.
