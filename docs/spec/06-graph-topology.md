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
  **each** joint's proprioception — three, at `m = 4`. Every other sensorimotor boundary cell has
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
| typical interior lane, `m` | 3 |
| boundary lane, `m` | 4 |
| drive lane, `m` | 1 |
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
world is not a cell and holds no restriction map. Every lane in the graph, including those
incident on boundary cells, is ordinary and `m`-sized, reached by an ordinary linear masked
restriction map. A patch cell's 48 → 4 restriction *is* the compression of that patch, performed by a
cell, inside the graph, costing a tick.

Boundary edges are given `m = 4` against the interior's 3 for a specific reason: a patch cell's edges
are the only route that patch's information ever takes, unlike an interior cell, which is reachable
many ways.

**Neither number is free-standing, and both are derived from one invariant.**
[#474](https://github.com/NGL321/patchworks/issues/474) set them jointly, from the construction
invariant this file now owns — **`Σ_e m_e ≤ n − 1` at every predicting cell**, which is
`05-timescales.md`'s `dim H⁰` bound read as a floor of 1 rather than as a total. The rule that picks
a point on it is *spend the least possible on the thinnest dimension*: **take the largest feasible
`interior_m`, then the largest `boundary_m` that clears.** The binding cell is L1 vision at degree 9 —
4 rim + 4 lateral + 1 up — where the invariant reads `4·boundary_m + 5·interior_m ≤ 31`, and the
feasible frontier is exactly three points: **(3, 4)**, (2, 5) and (1, 6). A session re-deriving the
pair should land on (3, 4) and not on the other two. `interior_m = 4` appears nowhere on that
frontier — not because `boundary_m` cannot be made small enough, but because the 12 L2 vision cells
sit at `8 × interior_m` with **no boundary edge at all**, so `boundary_m` cannot reach them at any
value.

**The 2x was demoted, not dropped by accident.** These lanes were `8` against `4`, *twice the
interior's deliberately*, and the **reason above is unchanged and still stated in the same words**:
a boundary cell's edges are the only route its information ever takes. What #474 changed is what that
reason buys — an **ordering** rather than a multiple. 4 against 3 is still wider. (2, 5) would have
preserved a ratio above 2 by spending a second unit on the thinnest dimension in the design, and the
ratio is an implementation of the reason rather than the reason itself.

**Of the four committed dimensions, `m` has the least theoretical headroom, and this ruling spent
some of it.** [#32](https://github.com/NGL321/patchworks/issues/32) found `n = 32`, `k = 12` and
`m = 8` all comfortable and `m = 4` thin: the one dimensionally commensurate body of theory (delay
embedding) puts the governing quantity at **twice the box-counting dimension** of the piece being
carried, so an interior lane of 4 supported a shared piece of box dimension under 2 — and at 3 that
reading is **1.5**, with **no source found either way** on whether either is enough. `m` is the
*first* rung on [#14](https://github.com/NGL321/patchworks/issues/14)'s constraint ladder and the rung
to pull first if a piece turns out not to fit through it. #474 pulled it, **downward**: the ladder was
priced for widening, and the private floor was worth one unit in the other direction. One unit, not
two — [#440](https://github.com/NGL321/patchworks/issues/440) has re-opened whether the piece has a
box dimension at all, having split `piece` from `situation set`, so the quantity this reading governs
is itself live. The trade runs both ways and is the same trade: every interior lane widened raises
`Σ_e m_e` — the cell's communication bus — at every cell, which lowers the per-cell reconciliation
gain and eats the private dimension the taper exists to produce; every lane narrowed buys that
dimension back.

**The exposure this takes, stated rather than buried: the patch's compression goes 6:1 to 12:1.**
This file accepted `48 → 8` and **rejected 24:1 in this same section**, when it rejected 8×8 patches
on the ground that *the boundary restriction map would be squeezing 192 → 8 in a single linear step*
(*Tiling granularity*, below). **12:1 sits between the two and has never been ruled on**, and that is
the honest statement of what #474 knowingly took. The counterweight is aggregate rather than
per-patch: rim bandwidth halves, 2048 lanes to 1024, against this file's own yardstick of *a world
whose state is about twenty numbers* — so the aggregate is still ~50x the world's state, and what is
exposed is the single patch cell's linear step, not the fleet's bandwidth. If the 12:1 later proves to
be what breaks, the thing to reach for is **one patch per L1 cell**, which #474 declined on the route
rather than ruling out of scope: it attacks the actual cause, the fan-in of four rim edges into each
L1 vision cell, and it was declined because that fan-in is the patchwork thesis at the seam
(*Tiling granularity*) rather than because it does not work.

**Narrowing lanes is transmission-neutral, and buys no gain.** Worth stating because the arithmetic
invites the opposite conclusion: an older form of the per-cell gain divided by `Σ_e m_e`, under which
this edit would read as a free 1.44x. [#189](https://github.com/NGL321/patchworks/issues/189)/[#190](https://github.com/NGL321/patchworks/issues/190)
removed `Σ_e m_e` from the gain outright, and `02-tick-semantics.md` reads
`gain_v = γ / (g_v² · c_v)`. No transmission bonus is claimed here.

**The drive edge is the exception, and in the other direction.** It is a boundary edge at `m = 1`,
because the argument above is about *bandwidth the world needs* and the drive is not the world: it
asserts one number, and a lane wider than its stalk carries nothing extra (*Where the drive
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

**The second half of that rejection was read at `boundary_m = 8` and has since got stronger, not
weaker.** At today's `boundary_m = 4` an 8×8 patch would be 192 → 4, **48:1**, against the 24:1 this
section refused. The rejection stands on its own first ground regardless; what moved is that
[#474](https://github.com/NGL321/patchworks/issues/474) narrowed the lane the argument measures, and
in doing so took the 4×4 patch itself from 6:1 to **12:1** — halfway to the compression this
paragraph calls too much (*Dimensions*, above). That is the one place in this file where #474's
exposure and this section's own bar are the same quantity, and nothing reconciles them yet.

### Private dimension is a gradient, and it falls out

`05-timescales.md`'s bound `dim H⁰ ≥ Σ_v max(0, n − Σ_{e∋v} m_e)` applied to the levels above:

| cell | degree | `Σ_e m_e` | guaranteed private dimension |
|---|---|---|---|
| L1 vision (interior) | 4 down + 4 lateral + 1 up = 9 | 4×4 + 4×3 + 3 = 31 | **1** |
| L1 vision (lattice edge) | 4 + 3 + 1 = 8 | 16 + 9 + 3 = 28 | **4** |
| L1 vision (lattice corner) | 4 + 2 + 1 = 7 | 16 + 6 + 3 = 25 | **7** |
| L1 somatomotor | 7–8 | 22–26 | **6 and 10** |
| L2 vision (interior) | 4 + 4 + 1 = 9 | 4×3 + 4×3 + 3 = 27 | **5** |
| L2 vision (lattice edge) | 4 + 3 + 1 = 8 | 12 + 9 + 3 = 24 | **8** |
| L2 vision (lattice corner) | 4 + 2 + 1 = 7 | 12 + 6 + 3 = 21 | **11** |
| L3–L6 core | ~6 | ~18 | **14** |
| L7 apex | ~4 + 1 drive edge | ~12 + 1 = 13 | **19** |

Measured from the built graph on `DEFAULT_SPEC`, `n = 32`, at `interior_m = 3`, `boundary_m = 4`.

**Guaranteed private dimension is one at the thinnest cell in the graph and rises to nineteen at the
apex.** Slow state lives deep by construction, which is precisely the gradient `05-timescales.md`
wanted and did not have a mechanism for.

**It is nowhere zero, and that is [#474](https://github.com/NGL321/patchworks/issues/474)'s doing.**
This table used to read `0` for every vision row, and 82 of the 150 predicting cells had no private
width at all. The lanes above were `8` and `4`; they are now `4` and `3`, derived from
`Σ_e m_e ≤ n − 1` — the invariant whose whole content is that this column is never zero. The floor is
**`p_v ≥ 1`** and nothing above 1 is claimed: the 36 binding cells sit exactly at 1, which is a
derived non-zero rather than a chosen margin.

Two corrections to how the headline used to read, neither of which moves it:

- **The rim's exceptions are no longer exceptional.** The four corner cells of the 4×4 L2 lattice
  have only two lateral neighbours, which used to leave them the graph's only structural privacy
  outside the core, at **4** against a rim of zeroes. They now sit at **11**, and every cell around
  them is non-zero too, so the corner is a high point on a gradient rather than an exception to a
  flat zero. Still worth knowing about if the private-component readout is run per-cell rather than
  per-level, and for the opposite reason: it is where the rim has the most, not the only place it has
  any.
- **It is a step, not a ramp.** 1–11 at the vision levels, 14 flat across L3–L6, 19 at the apex. Degree
  falls at the apex and nowhere else in the core, so nothing about this gradient is smooth.
  [#41](https://github.com/NGL321/patchworks/issues/41) already half-said this from the other
  direction — the gradient is one in *means*, with adjacent depths overlapping per tick — and the
  structural picture agrees: what the taper buys is deep-versus-shallow, not a graded ordering
  through the core. It is not designed here; it falls out of the taper, because rim-adjacent cells are
necessarily high-degree — an L1 vision cell must read four patches — and depth reduces degree.

A *guaranteed* private dimension is a lower bound, and learned rank-deficiency used to be read as
enlarging `H⁰` past it. What the gradient says is that near the rim a cell's privacy is thin, and deep
it is broad — but at every cell it is now **structural**, which is the change #474 made.

*Amended by [ADR-0032](../adr/0032-the-maps-learn-isometric-transport-and-a-spectral-floor-expresses-it.md):
the slack this paragraph leans on largely closes.* A map held to the spectral floor has rank exactly
`m`, so no single map contributes a dead direction to `H⁰` any more. What excess can survive comes from
misalignment across a cell's **incident** maps rather than from deficiency within one of them — and
`GAUGE_C` pushes that the other way, because incoherence raises the stacked operator's rank. So near
the rim privacy stops being contingent on learning and becomes what construction says it is.
**#474 is why that is now a floor rather than a cliff**: with the maps' slack closed, `p_v` is all a
rim cell has, and construction sets it to at least 1 everywhere rather than to 0 at 82 cells.
[#385](https://github.com/NGL321/patchworks/issues/385) ruled that this was the mask's to supply and
[#474](https://github.com/NGL321/patchworks/issues/474) supplied it.

#### The zero row was two populations, and what released each of them

*Written by [#475](https://github.com/NGL321/patchworks/issues/475) on
[#385](https://github.com/NGL321/patchworks/issues/385)'s ruling, while the zero stood; amended by
[#474](https://github.com/NGL321/patchworks/issues/474), which released it.* **There is no zero row
now** — the table above reads 1 at its thinnest. This subsection is kept as the record of **why the
pin existed and what released it**, because the analysis is what selected the pair of knobs, and a
session re-deriving `interior_m` and `boundary_m` needs it.

**The pin, as it stood.** 82 of the 150 predicting cells had `p_v = 0`, and they were two
populations, not one: **`interior_m` reached 18 of them and could not reach the other 64 at any
value.**

| population | cells | what filled the bus | of which non-interior | did `interior_m` reach it? |
|---|---|---|---|---|
| **L1 vision** | **64** | 4 rim edges × `boundary_m` = 8, plus lateral and up | 44–52, of which **32 is rim-side alone** | **No**, at 4, 1 or 0 alike |
| L1 somatomotor | 6 | 1–2 rim edges plus interior | 0–16 | Yes — `interior_m` 3 cleared 3 of them, 2 cleared all 6 |
| L2 vision (lattice edge and interior) | 12 | interior only, degree 8–9 | 0 | Yes — `interior_m` 3 cleared all 12 |

**For the 64 the communication bus was full before a single interior edge was counted.**
`4 × boundary_m = 4 × 8 = 32 = n`, so `p_v = max(0, n − Σ_e m_e)` was `0` whatever the interior did:
dropping `interior_m` to 1, or to 0, left all 64 exactly where they were. **Lateral degree was
powerless there too** — the lattice corner, edge and interior cells differ by two lateral neighbours
and all three read 0, because the rim edges alone had already exhausted the stalk.

**So this section's own framing was right and incomplete.** *Rim-adjacent cells are necessarily
high-degree — an L1 vision cell must read four patches* is true, and it points at the **count** of
neighbours when the binding quantity is the rim edges' **width**. A cell reading four patches through
lanes of 2 would have `p_v = 24`; it was `boundary_m = 8`, deliberately *twice the interior's*, that
made four of them fill `n` exactly.

**What released it was neither knob alone.** The three candidates this subsection first listed —
`boundary_m ≤ 2`, `n ≥ 53`, one patch per L1 cell — read as exhaustive and were not. They are the
**`boundary_m`-only slice of a two-variable constraint**, and the constraint is:

> **`4·boundary_m + 5·interior_m ≤ n − 1 = 31`** at the binding cell, L1 vision at degree 9.

Its feasible frontier is exactly **`(interior_m, boundary_m) ∈ {(3,4), (2,5), (1,6)}`**, and #474 took
`(3, 4)` by *spend the least possible on the thinnest dimension*. Two things follow that the
`boundary_m`-only reading could not see:

- **`interior_m = 4` is infeasible at *every* `boundary_m`**, not merely awkward. The 12 L2 vision
  cells in the table above sit at `8 × interior_m` with **no boundary edge at all** — the row is
  already here, and its consequence was not drawn. `4·boundary_m + 20 ≤ 31` needs `boundary_m ≤ 2`,
  and `boundary_m = 2` still strands those 12. So `interior_m` had to move whatever else happened.
- **`boundary_m ≤ 2` was an artifact of holding `interior_m` at 4.** With the pair swept jointly the
  boundary lane clears at 4, and the 24:1 patch compression that number implied is not the price
  anyone actually pays; 12:1 is (*Dimensions*, above).

`n` is **refused**, not un-chosen: `body.py` declares it *fixed and intended to stay fixed*, absent
from `01-cell-and-sheaf.md`'s Flex priority ladder, and this file's own argument against 8×8 patches
applies verbatim — at `n = 53` the apex would carry `p_v = 36` in a world whose state is about twenty
numbers. One patch per L1 cell is **declined on the route** and preserved as the thing to reach for if
the 12:1 breaks (*Dimensions*, above).

**What this section records now is the invariant, not the constraint.** `p_v` at the rim was *a
construction quantity nobody had set* — a residual of `interior_m` and the lateral fill's degree
targets. It is now **set**, by `Σ_e m_e ≤ n − 1` at every predicting cell, and the first rung on
[#14](https://github.com/NGL321/patchworks/issues/14)'s constraint ladder reaches it after all — by
being pulled downward, in company, rather than alone.

`χ = Σ_v n − Σ_e m_e` over predicting cells is **+2505**, measured from the built graph at
`interior_m = 3`, `boundary_m = 4`. It was **+1036** at `(4, 8)`; `χ` is not a target and the move is
the mask's arithmetic, not a finding. The eight
drive edges move it by 8; the drive's cost is local to the apex cells, not to the diagnostic.

**`χ` restricts the node sum only. Every edge in the graph still counts.** The node term runs over
predicting cells alone — including boundary cells swamps it, 256 cells × 48 dimensions of nominally
private state that the world overwrites every tick and no cell holds — but the edge term runs over
**all** edges, boundary-incident ones included, because those lanes are ordinary and are the
route the boundary's information actually takes (*The world only ever touches node stalks*, above).
The rule was previously stated as "computed over predicting cells only", which licensed the wrong
computation: dropping boundary edges as well as boundary nodes gives **+3573** against this section's
**+2505**. This is a correction to the diagnostic as recorded in `01-cell-and-sheaf.md`. The two
figures were **+3164** against **+1036** on the `boundary_m = 8` surface, where the wrong rule read
three times the right one; at `boundary_m = 4` it reads 1.4x, because narrowing the boundary lanes is
most of what the wrong rule was dropping. **The gap narrowed and the error did not** — a reader who
finds the two figures close should not conclude the rule matters less.

The value is edge-count sensitive and is not a target. This section used to carry ~980 against a
~698-edge estimate and note that a ~663-edge count would give ~+1096 — a spread it owned in advance
and declared not to matter. The built graph's 682 edges landed inside that band at **+1036**, so the
estimates were retired in favour of the measurement
([#83](https://github.com/NGL321/patchworks/issues/83)). **Every figure in that history was read at
`interior_m = 4`, `boundary_m = 8`**, and [#474](https://github.com/NGL321/patchworks/issues/474)
moved the surface: the edge count is unchanged at 682 and the value is **+2505**. Nothing is
contradicted, because what is load-bearing about `χ` is its **invariance under learning**, not its
value — and that is exactly why a construction change moves it freely.

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
  evidence: `12,288 → 1,060 → 210 → 60`. The entire sensory boundary reaches the core through
  **sixty numbers per tick**, a 205:1 squeeze at a single cut, while the two farthest predicting
  cells are only ~9 hops apart. Both readings live in that number and the tension is owned rather
  than resolved by choosing a favourable metric.

  **These were `12,288 → 2,120 → 280 → 80` and a 154:1 squeeze**, read at `interior_m = 4`,
  `boundary_m = 8`. [#474](https://github.com/NGL321/patchworks/issues/474) narrowed both lanes to buy
  the private-dimension floor, and the cuts moved with them — the taper's capacities are set by `m`
  ([ADR-0030](../adr/0030-the-conversion-buys-a-design-variable-and-the-price-is-booked.md)), so this
  is arithmetic rather than a new finding. **It is the honest cost of that ruling recorded where the
  bottleneck is argued**: #474 priced the trade at the cell, per-patch, and it also tightens the
  graph's worst cut by a third. Nothing here was re-measured against a transmission reading, because
  narrowing lanes is transmission-neutral under `gain_v = γ / (g_v² · c_v)` (*Dimensions*, above) —
  what moves is capacity, not gain.

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

  **One mitigation is available in the literature, and it is already claimed — the conversion is
  it.** Arroyo et al. (2025) restate over-squashing unchanged for the recurrent setting and factor
  sensitivity into a topology term and a **model-dynamics** term, mitigated by direct control of the
  Jacobian spectrum — and Patchworks is recurrent with unit-delay bidirectional edges, so the term
  exists here. The remedy that control names is a **state-space formulation with a controlled
  spectrum**, and that is what the Koopman conversion
  ([#138](https://github.com/NGL321/patchworks/issues/138)) made this body: a learned linear `K` per
  cell, banded, with `ρ(K)` reported. The model-dynamics term has therefore been under direct control
  since stage 1. [`docs/research/148`](../research/148-local-linear-operator-citations.md) §10.2 reads
  it the same way and says so outright — *"Patchworks **is** a recurrent state-space GNN. A learned
  linear `K` with controlled `rho(K)` is precisely 'a state-space formulation'."*

  **The budget this paragraph once said was spent has split, not vacated**, and the half the
  mitigation reads is the free half. `σ_max(K)` — the body — **is** spent, by
  [ADR-0015](../adr/0015-the-cell-operator-band-is-on-the-spectral-norm.md), at the
  maximal-transmission face of **exactly 1**, so there is no headroom below the ceiling to reclaim.
  `ρ(K)` is **free**: [#143](https://github.com/NGL321/patchworks/issues/143) and
  [ADR-0028](../adr/0028-a-cell-holds-a-spectrum-of-retention-constants.md) moved retention onto a
  learned `λ(K)`, so the regional spectrum this paragraph once spent on timescales
  ([#27](https://github.com/NGL321/patchworks/issues/27),
  [#42](https://github.com/NGL321/patchworks/issues/42)) is no longer the mechanism. `γ` stands, but
  was demoted twice ([#140](https://github.com/NGL321/patchworks/issues/140),
  [#160](https://github.com/NGL321/patchworks/issues/160)) and sits at its global ceiling of 1.0; it
  was never the operative claimant. **Transmission spends the norm; retention reads the radius** —
  [`05-timescales.md`](./05-timescales.md), which has carried the correction since #143 while this
  paragraph did not.

  **Two limits travel with the claim and are part of it.** Arroyo's finding (iii) is a
  **conjunction** — *"a combination of graph rewiring and vanishing gradient mitigation"* — and this
  design refuses rewiring, which [#237](https://github.com/NGL321/patchworks/issues/237) then
  measured worthless sheaf-side (parallel routes worth **1.39×–1.57×**, against **3.80×**
  graph-side). And the vanishing-gradient half is *"about backpropagated gradients and does not
  transfer to a local learning rule; no source was found analysing over-squashing under local
  learning rules"*
  ([`docs/research/031`](../research/031-graph-topology-citations.md)). And weight the source as this
  section weights its others: both Arroyo readings in the record are **from the abstract only**, with
  Theorem 5.1 quoted secondhand. **The remedy is taken; it is not thereby sufficient.**

  **And per-cell control is not composed control — confirmed, and no longer open.** Sensitivity from
  rim to apex reads a **chain** of per-cell operators through learned restriction maps, seven hops,
  and a band on each `σ_max(K)` bounds the links, not the chain. ADR-0015 named that quantity exactly
  and deferred the argument, wanting the per-hop budget the effective-resistance work would produce.
  That work reported ([#237](https://github.com/NGL321/patchworks/issues/237)),
  [#423](https://github.com/NGL321/patchworks/issues/423) made the argument, and
  [ADR-0015](../adr/0015-the-cell-operator-band-is-on-the-spectral-norm.md) now carries it. The
  composed **upper** bound is derivable from gauges already declared — `σ_max(composed) ≤ ρ^{2h} =
  4⁷ = 16,384` over seven hops, no constant invented — and **inert**, some twenty-one orders above
  #237's measured `σ₁ ≈ 4.5e-17`. **`ρ(K)` buys none of it**: submultiplicativity consumes `σ_max`,
  which ADR-0015 spent at exactly 1, so the free half of the split above never enters the product at
  all. **That sharpens this paragraph rather than undercutting it** — the claim on Arroyo's
  model-dynamics term is on the **per-cell** operator and stays exactly that size, but the chain runs
  21 orders **under**, not over, so its narrowness is not a present danger. What no spectral bound
  supplies is the **lower** side, and under [#144](https://github.com/NGL321/patchworks/issues/144)
  it is not owed one: persistence is sustained by the driven field, not stored in an operator.
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
  change of basis into each shared lane rather than a shared kernel, reconciliation rather than
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
  varies per position is a thin linear change of basis — a masked 32→3 restriction map is 96
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
  capacities run `12,288 → 1,060 → 210 → 60`, so the whole sensory boundary reaches the core through
  sixty numbers per tick — down from eighty, because
  [#474](https://github.com/NGL321/patchworks/issues/474) narrowed both lane widths (*Broadcast
  subspaces*, above). In the literature's own units that single narrow cut **is** the high
  effective resistance between distant cells, which is why this document no longer argues that reach
  and squeeze are different problems (*Broadcast subspaces*, above). Nothing here fixes it; the relays
  that could are the ones rejected for collapsing the abstraction measure, and the deep relays that
  are pre-specified address a different, smaller term.
- **Vision dominates the boundary.** 256 of 264 boundary cells are visual. If the acceptance demo
  turns out to rest on proprioception, the graph is built around the wrong modality.
