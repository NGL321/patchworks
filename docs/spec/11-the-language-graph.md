# The language graph

The shape of the graph when the rim is a character stream rather than a render — given the cell
contract in [`01-cell-and-sheaf.md`](./01-cell-and-sheaf.md), the tick in
[`02-tick-semantics.md`](./02-tick-semantics.md), the boundary in
[`04-action-and-the-boundary.md`](./04-action-and-the-boundary.md), timescales in
[`05-timescales.md`](./05-timescales.md), and the dome in
[`06-graph-topology.md`](./06-graph-topology.md).

Terms used here are defined in [`CONTEXT.md`](../../CONTEXT.md). The decision this document records is
[#130](https://github.com/NGL321/patchworks/issues/130); the rim's contents are
[`12-the-interlocutor.md`](./12-the-interlocutor.md)'s, from
[#129](https://github.com/NGL321/patchworks/issues/129), and the rim's ban is from
[#128](https://github.com/NGL321/patchworks/issues/128). **This document owns the shape; `12` owns the
world** — where the two touch, the contents are `12`'s and the counts are this document's. The depth semantics are
[ADR-0014](../adr/0014-depth-decimates-in-time-and-not-in-space.md).

**This document is not a second topology.** It is the second *instantiation* of a construction rule
that `06-graph-topology.md` already named as its own fallback, and the first section here is that
rule with the dome expressed in it. Writing the wedge any other way would have left the project with
two hand-written graphs and a note claiming the first was abandonable.

## The construction rule

### What `06`'s fallback got wrong

`06-graph-topology.md` says the dome is explicitly abandonable, and names the replacement:

> a boundary set, a target depth, and per-level degree constraints, with connectivity sampled to
> satisfy them

**As written, that rule cannot express the dome**, and the finding is worth more than the
inconvenience. Three of its four clauses survive. The fourth is the mechanism, and it is wrong in a
specific way:

- **Sampling destroys retinotopy.** The dome exists so that "adjacent patches are adjacent cells" and
  retinotopy *falls out of the index rather than being designed*. Adjacency is not a degree
  statistic. A connectivity sampled to a per-level degree target reproduces the dome's degree table
  exactly and its adjacency with probability zero. `graph.py`'s builder is, in its own words,
  "deterministic and free of any draw".
- **Degree is an output, not an input, everywhere except the core.** In the vision lattices a cell's
  degree is whatever the covering and the four-neighbour rule leave it — 7, 8 or 9 depending on
  whether it sits at a lattice corner, edge, or interior. Only the core takes a degree *target*, and
  even there the fill is Havel–Hakimi with a locality tie-break, not a draw.
- **A degree constraint cannot say "no cell sees a whole puck".** That property is a statement about
  the *covering* — which cells below a cell is responsible for — and the covering is the thing the
  4×4 tiling was chosen to force. It is the dome's load-bearing claim and the fallback has no
  vocabulary for it.

So the rule below has **two ingredients the fallback sentence omits**: a covering rule and a lateral
rule. This is a repair rather than a widening, and the distinction matters because widening a general
rule until it accommodates a specific graph is exactly the failure this exercise was run to avoid.
The defence is that **neither addition is new**: both are already in the dome's builder, and both are
used by the wedge in the same form with different parameters. Nothing here is an ingredient only one
of the two topologies needs. The one genuine escape hatch is named as an escape hatch below rather
than dressed up as derivation.

### The rule

A graph is built from:

1. **Boundary groups.** Each has a name, a count, a stalk dimension, a kind — sensory, motor, or
   internal rim — and an attachment. Boundary cells run no body
   ([ADR-0006](../adr/0006-boundary-cell-stalks-are-world-shaped.md)).
2. **Columns.** A column is a stack of levels; a level is an **index box**, a tuple of axis extents.
   `(16, 16)` is a two-axis box of 256 cells; `(128,)` is a one-axis box of 128.
3. **A covering rule** between consecutive levels of a column: each axis of the box above covers its
   corresponding axis below **proportionally**, with a per-axis **fan**. Fan 1 is a partition — every
   cell below has exactly one up-edge. Fan 2 gives each cell two, which is what makes the topmost
   level's missing up-edges worth two of its degree.
4. **A lateral rule** per level, one of three: **axis-neighbour** (neighbours along each axis of the
   box), **complete**, or **degree-target fill** (Havel–Hakimi, ties broken toward locality in the
   level's cyclic index).
5. **A merge level**, where the columns' topmost levels cover into one shared column — the **core**.
   Where several columns merge into one level, their coverings **interleave** by column index, so
   cells fed by different columns are adjacent in the core's index rather than bunched at a seam.
6. **Degree targets** for the core's levels. The topmost level — the **apex** — takes a lower target,
   because it has no predicting level above it to connect up to and so loses its up-edges by
   construction rather than by stipulation.
7. **Dimensions**: `n`, `k`, and `m` per edge kind.
8. **Attachment exceptions** — the escape hatch, below.

Everything else in `06-graph-topology.md` — the dimensions and the boundary's exemption, the sparsity
treatment, the absence of relay cells, the broadcast-subspace argument, hard partitions held in
reserve — is independent of which instantiation is running and is **not restated here**. That was
`06`'s own claim about what survives the dome's abandonment, and it holds.

The **axis-neighbour** rule is where the two topologies come closest to looking like one thing. On a
two-axis box it is four-neighbour, which is what the dome's vision lattices have. On a one-axis box
it is two-neighbour, which is what the wedge's buffer levels have. One rule, and the difference is
the number of axes rather than a second lateral scheme.

### The dome expressed in it

| ingredient | dome |
|---|---|
| boundary groups | vision, box `(16, 16)`, stalk 48, sensory; proprioceptive 3, stalk 2; touch 3, stalk 1; actuator 1, stalk 6, motor; drive 1, stalk 1, internal rim |
| columns | **vision**: `(16,16) → (8,8) → (4,4)`, fan 1 per axis, laterals axis-neighbour. **somatomotor**: `(7,) → (6,) → (4,)`, fan 1, laterals complete |
| merge | at L3, two columns interleaved |
| core | `16, 14, 12, 10, 8`, fan 2, laterals degree-target, targets 6 and apex 4 |
| exceptions | actuator; drive |

The proportional covering is what lets the somatomotor column taper 7 → 6 → 4 without the ratios
dividing, and the per-axis fan-1 covering on a two-axis box is exactly the 2×2 block: `(r, c)` covered
by `(r // 2, c // 2)`. A single-axis covering over a row-major flattening is **not** the same map and
would not give the dome its blocks — which is why the rule carries axes rather than sizes.

### The one escape hatch

An **attachment exception** is a boundary group that attaches by a rule other than "to the cell
covering it". Both topologies use it, and it is called an escape hatch because it is not derived from
anything above it — it is where a fact about the world is written into the graph by hand.

- **The dome's actuator** takes one edge to the L1 cell covering *each* joint's proprioception, three
  in all, which is what makes the reflex loop three ticks at every joint rather than at whichever one
  happened to share the actuator's L1 cell.
- **The drive**, in both, attaches entire: one `m = 1` edge to every apex cell, fan-out as the
  strength knob ([ADR-0009](../adr/0009-a-drive-is-a-motor-edge-attached-deep.md)).

The dome uses the hatch twice, the wedge once. **The wedge's non-use of it is the decision**, not an
absence: the wedge has no actuator-style exception because it has no reflex arc, and that is
deliberate. See *Where the columns merge*, below.

## The wedge

**Keep the taper, drop the sheet.** The taper is general — fewer cells with depth, so information
must compress — and `06`'s cut capacities are the modality-independent part of its argument. The
two-dimensional sheet is the part that is vision, and none of its justification survives a token
stream: there is no answer to "which cell is adjacent to this one" that comes from the sensor.

In its place, a rim whose one axis is **time**.

### The rim is a shifting buffer

The rim is a buffer of the last `B = 128` ticks, and a cell's position in it means *how far back*
rather than *where*.

- **Tick-indexed, not event-indexed.** Half-duplex with an idle floor means most slots hold idle.
  That is signal, not padding: pauses, latency, and who-waits-for-whom are conversational dynamics,
  and event-indexing deletes exactly them. It costs buffer efficiency, and that cost is what sets `B`.
- **One stream, two rims.** Heard and spoken boundary cells are separate — #128 bans a sensory and a
  motor rim sharing a boundary cell — but they are **aligned on the same tick axis**, so slot `i` of
  each refers to the same tick. A graph that never sees its own words beside the interlocutor's is
  modelling two monologues.
- **The buffer shifts.** At tick `t`, slot `i` holds what happened at tick `t − i`; at `t + 1` it
  holds what slot `i − 1` held. The world writes the head; the rest is history moving along. A slot's
  index is an **age**, not the identity of a moment. This has a consequence the taper depends on and
  an exposure it does not escape — both under *Known exposure*, below.

**`B = 128` is not derived from turn length, because there is none.** #129 makes turn length
deliberately not a tick count: the agent emits every tick and must fall silent to be heard. `B` is
set by the **boundary-cell budget** instead. At `B = 128` the rim is 256 boundary cells against the
dome's 264, and the L0→L1 cut is 2,048 numbers per tick against the dome's 2,120. It lands on the
dome's figure because that is the figure the rest of the architecture was sized against, and 128
ticks is a sentence or two of history.

### The levels

**Disjoint 4:1 pooling, not overlapping dilation.** The dilated-convolution precedent uses
overlapping windows; the dome pools disjointly. Disjoint wins here because it makes the **straddle**
load-bearing — no cell covers a span crossing a pooling boundary, so laterals repair it, and that is
the mechanism this architecture is about. Overlap covers straddles directly at higher degree, and
degree is what `Σ_e m_e` charges private dimension for. The precedent is worth less than the
consistency.

| level | contents | predicting cells | temporal extent |
|---|---|---|---|
| **L0** — rim | 128 heard + 128 spoken, tick-indexed | — (256 boundary cells, no body) | 1 tick |
| **L1** | 32 heard + 32 spoken, each over 4 slots | 64 | 4 ticks |
| **L2** | 8 heard + 8 spoken, each over 4 L1 cells | 16 | 16 ticks |
| **L3–L7** | core, not a lattice, 16 / 14 / 12 / 10 / 8 | 60 | — (abstraction) |
| **internal rim** | 1 drive boundary cell, attached at L7 | — (1 boundary cell, no body) | — |
| | **total** | **140 predicting, 257 boundary** | |

Against the dome's 150 and 264. Every count is a **construction parameter**, not a constant of the
architecture.

In the rule: two columns, **heard** and **spoken**, each `(128,) → (32,) → (8,)` at fan 1 with
axis-neighbour laterals; merging at L3 into the dome's own core, unchanged, interleaved so that heard
index `i` covers core cell `2i` and spoken index `i` covers `2i + 1`.

### Why 140 and not 255

The obvious wedge is a **dyadic stack** over the buffer — the rim's 256 slots halving to 128, 64, 32,
16, 8, 4, 2, 1 — and it is wrong for the reason `06` already rejected its own equivalent. That gives
**255 predicting cells** with no argument for any of them, and walks straight into the number `06`
turned down when it refused to taper vision slowly enough to get eight lattice levels.

So the buffer tapers **fast, in two stages**, and the depth lives in a core that is not a lattice at
all. That is `06`'s construction, not a compromise struck to reach a cell count: it is the same
answer to the same pressure, which is that hop depth has to come from somewhere that is not the
sensor's own resolution.

### Connectivity

- **Vertical.** Each cell connects to the four cells it covers on the level below and to the cell
  covering it above. Every rim slot, L1 and L2 cell has exactly one up-edge.
- **Lateral.** Two-neighbour within each buffer column at L1 and L2 — the axis-neighbour rule on a
  one-axis box. Sparse within the core, to a degree target, as in the dome.
- **Core.** Uniform degree ~6, no lattice — except the apex at ~4. Unchanged from `06`.
- **Drive.** One `m = 1` edge from the drive boundary cell to each of the eight L7 cells.
- **No actuator exception.** There is no motor cell reaching down into the taper.
- **524 edges**, mean degree **5.60** over predicting cells, against the dome's 682 and 7.27.

**The wedge is a materially sparser graph than the dome, and the cause is the axis count.** A
two-neighbour lateral rule costs half what four-neighbour does, so a wedge cell of the same rank in
the taper carries fewer edges and therefore a smaller `Σ_e m_e`. The effect shows up in the private
dimension table below as guaranteed private dimension appearing **one level earlier** than in the
dome. It is recorded rather than corrected: it is a consequence of the domain's dimensionality, and
it moves the quantity slow state needs in the direction slow state wants.

### Lateral edges transfer, with one named exposure

The dome's laterals resolve disagreement about an object **straddling** two patches — `06` calls them
"the load-bearing difference from a CNN". The analogue is a linguistic unit straddling buffer
positions, and **"no cell ever sees a whole puck" becomes "no cell ever sees a whole word"**, true by
construction at four characters the way it is at 4×4 pixels: an English word with its following space
runs about 5.5 characters, so a four-tick window cannot hold one. The dome's main justification does
transfer, and the wedge is not the weaker structure it might look.

**The exposure, recorded rather than resolved:** two adjacent patches are simultaneous and symmetric;
two adjacent buffer slots are **ordered**, and one causes the other. A lateral edge in time may be
carrying order, which the sheaf has no vocabulary for. Nothing in `01-cell-and-sheaf.md` distinguishes
the two ends of an edge by anything but which cell holds which map, and that is by design — see *A
one-way edge is not a sheaf edge* there, which forecloses the obvious repair.

### Where the columns merge, and why there is no reflex arc

`04-action-and-the-boundary.md` made the shortest sensorimotor loop a design budget, and `06` spent
three actuator edges to hit three ticks at every joint. **That budget does not transfer, deliberately.**

The reason is **echo**. A three-tick heard→spoken path lets the agent repeat what it hears at a small
lag, which is a cheap way to score well on the interlocutor's next-character surprisal — a degenerate
attractor sitting directly on the drive.

The repair is the dome's existing device: **parallel columns**. Vision and proprioception do not
share a cell until L3; heard and spoken do not share one until the first core level. The shortest
heard→spoken path is then **six hops, about seven ticks**, through the core. Echo is designed out by
construction, and **the merge level is the construction parameter that sets it** — the knob to pull if
echo turns out wanted rather than feared.

Two repairs were available and both were declined:

- **A directed heard→spoken edge.** Rejected as a category error, not as a cost: see *A one-way edge
  is not a sheaf edge* in `01-cell-and-sheaf.md`. It is a wire.
- **Asymmetric masking** — masking heard-side directions out of a shared edge while leaving the
  spoken side free. Legal under the contract, since the masks at the two ends are independent. Declined
  as hand-specified content restriction, which is the thing `06` rejected when it turned down lateral
  cross-modal edges at L2 as "hand-specified wiring for something that should follow from position".

This carries a falsification, and it is already in the register
([ADR-0014](../adr/0014-depth-decimates-in-time-and-not-in-space.md) records it):
**if the agent converges on echo, the merge is too shallow.**

### Dimensions

`n = 32`, `k = 12`, interior `m = 4`, boundary `m = 8`, drive `m = 1` — one `n` and one frozen
dictionary across both domains, which is #128's decision and not this document's to revisit.

Boundary stalks:

| group | stalk | contents |
|---|---|---|
| **heard** | 97 | one character, one-hot: printable ASCII plus idle and turn-boundary |
| **spoken** | 196 | commanded character (97), character as taken up (97), uptake flag (1), coherence (1) |
| **drive** | 1 | valence, not specification |

**The spoken stalk's width is chosen here, not recorded.** #129 fixes the spoken rim's *contents* —
commanded character, and a readback of character-as-taken-up, uptake, and coherence — and it had no
buffer to size them over. Carrying that per-slot gives 196, on the same principle as the dome's
actuator stalk of 6: three commanded and three efference, the command and what the world made of it,
in one cell. The alternative — a readback only at the head — would put a boundary cell's stalk
dimension in the hands of its index, which nothing else in the architecture does.
[`12-the-interlocutor.md`](./12-the-interlocutor.md) **ratifies the choice rather than superseding
it**, having written the contents down: four contents, per slot, 196.

### Cut capacities and private dimension

| cut | numbers per tick | dome |
|---|---|---|
| L0 → L1 | 2,048 | 2,120 |
| L1 → L2 | 256 | 280 |
| L2 → L3 | 64 | 80 |
| L3 → L4 | 128 | 128 |
| L4 → L5 | 112 | 112 |
| L5 → L6 | 96 | 96 |
| L6 → L7 | 80 | 80 |

The core is the dome's, so the last four rows are identical by construction. The whole rim reaches
the core through **64 numbers per tick**.

| cell | degree | `Σ_e m_e` | `dim H⁰ ≥` |
|---|---|---|---|
| L1 (interior) | 7 | 44 | 0 |
| L1 (buffer end) | 6 | 40 | 0 |
| L2 (interior) | 7 | 28 | 4 |
| L2 (buffer end) | 6 | 24 | 8 |
| L3–L6 core | 6 | 24 | 8 |
| L7 apex, with the drive edge | 5 | 17 | 15 |

The gradient is the dome's — zero at the rim, ~8 through the core, 15 at the apex — with guaranteed
private dimension arriving at L2 rather than L3, for the reason given under *Connectivity*. The apex
figure is identical, so **the drive attaches to exactly the slack it attaches to in the sandbox** and
ADR-0009 transfers with zero change.

## Depth means two things, and they are kept apart

**A fast rim taper where depth is temporal extent, then a core where depth is abstraction.** Two
regions, two meanings, neither borrowed from the other.

This is the wedge's central claim and it is load-bearing enough to have its own record:
[ADR-0014](../adr/0014-depth-decimates-in-time-and-not-in-space.md). In outline — the ADR carries the
argument and the costs:

- `05-timescales.md`'s *Depth does not supply it* is about **spatial** pooling and is sound there. A
  wedge cell aggregating buffer slots is a **temporal** pooling, a genuine decimation. The prohibition
  is **domain-scoped**, and nothing in the record noticed because until now there was one domain.
- The wedge's taper is therefore `05`'s **instrument, not its mechanism**, and it **costs a piece of
  evidence**: the depth/timescale correspondence becomes built rather than found, so it can no longer
  be cited as persistence working. Persistence stays the mechanism under test.
- The **taper/core split** is what stops horizon and abstraction collapsing into each other, and it
  is the same construction rule as the column merge.

## The falsification sweep

`06` writes a sweep into the build in two conditions, and only one of them tests the division thesis.
The wedge owes the same, and the analogue of *coarsen the tiling* is exact:

- **Coarsen the pooling.** Rebuild with L1 covering **8 slots** rather than 4, so an L1 cell does see
  a whole word, and re-run. If the acceptance demo is unaffected, the division the 4:1 pooling was
  chosen to force was decorative — a finding about **the division thesis** in a second domain, which
  is worth strictly more than the sandbox's version of it, because it is the same thesis tested where
  the modality shares nothing with the first test.
- **Halve the core.** Rebuild L3–L7 at half their cell counts. As in `06`, a finding about *this
  topology* and not about the thesis.

Each costs one run, and both are findings either way.

## Where the static wedge hurts

The wedge routes statically because a buffer slot's index is fixed at construction. Pixels do not move
relative to each other; which words modify which is content-dependent and different every sentence.
The two-region shape gives that failure a number rather than a hedge.

**A cell's window is fixed by construction, so a dependency spanning more slots than its level covers
cannot be represented at that level and must go deeper — and deeper means the core, where abstraction
lives, not more history.** The longest span any single cell covers is **16 ticks**, at L2. Long-range
agreement — a pronoun and its referent, a question and its answer — is therefore forced up into the
core, where it competes for the same private dimension slow state needs.

That is measurable on an instrument that already exists: `05-timescales.md`'s per-cell
private-component readout against hop distance.

**This is recorded as the wedge's known failure mode and is not fixed here.** Content-dependent
routing is a separate question, and [#142](https://github.com/NGL321/patchworks/issues/142)'s
retirement of attention does not reach it: that was a **transmission** argument, and the `deg` it
refused is bought by reading `λ_max`. Nothing in it bears on routing by content.

## Known exposure

- **Most of the rim is trivially predictable, and the taper's apparent width overstates the
  prediction pressure on it.** The buffer shifts, so for every slot but the head the next value is the
  current value of its neighbour — a deterministic shift, known to the graph. Two of 256 boundary
  cells carry a genuinely new value per tick; the rim holds 37,504 numbers of which 293 are new. The
  taper's own arithmetic is unaffected — the cut capacities above are edge budgets and stand — but a
  reader who reads 2,048 numbers per tick into L1 as 2,048 numbers' worth of *problem* will be wrong
  by two orders of magnitude. This is not a defect of the buffer: holding a value still is precisely
  what lets a decimating taper see a span, and it is why the buffer exists. It is recorded because it
  changes what the rim's disagreement means, and because it makes the straddle the recurring event
  rather than an incidental one — every four ticks, one value crosses a pooling boundary.
- **What clears the tail of the spoken rim.** [ADR-0003](../adr/0003-action-is-prediction-the-world-clears.md)
  makes a motor edge one the world clears by moving, immediately. That is true of the head slot, which
  the readback answers. It is not obviously true of slot 40, which nothing writes but the shift. The
  spoken buffer is a motor rim entire under this document, and whether a rim's contract is a property
  of the group or of the slot is not settled by anything in the record.
- **A lateral edge in time may be carrying order.** As above, under *Lateral edges transfer*.
- **The interleaved merge is chosen, not derived.** Heard to even core cells and spoken to odd is one
  of several interleavings, and the rule says only that columns spread rather than bunch. The dome's
  precedent — the somatomotor column spreading evenly around L3 — has the same freedom and the same
  absence of an argument for the particular offset.
- **The dyadic alternative is rejected on cell count, which is an argument about this budget.** If the
  boundary budget moved, the argument in *Why 140 and not 255* would have to be made again rather than
  cited.
- **An undifferentiated graph, carved by sparsity rather than designed**, is rejected for this build
  and **not refuted**. Masks close and never re-open, so sparsity prunes and never discovers; and
  `dim H⁰ ≥ Σ_v max(0, n − Σ_e m_e)` gives a high-degree cell zero guaranteed private dimension — the
  home of slow state destroyed before pruning could recover it. It becomes live the moment masks can
  open, which is structural growth and out of scope today. Held in the map's fog with its reactivation
  condition named.
