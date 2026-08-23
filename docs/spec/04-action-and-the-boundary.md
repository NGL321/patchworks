# Action and the boundary

Where the graph stops and the world starts, and what makes acting different from perceiving —
given the cell contract in [`01-cell-and-sheaf.md`](./01-cell-and-sheaf.md) and the tick in
[`02-tick-semantics.md`](./02-tick-semantics.md).

Terms used here are defined in [`CONTEXT.md`](../../CONTEXT.md).

## One graph

Active predictive coding has an upstream and a downstream — world modelling and action
selection, one algorithm running in opposite directions. Patchworks realises that **without two
populations of cells, without two graphs, and without any modification to the cell contract.**

There is one graph, one sheaf, one Laplacian. Every cell does exactly one thing: predict its own
next node stalk. "Generative" and "action-selecting" name no object in this architecture.

**The opposition lives in the edge, in who clears the disagreement:**

| | Sensory edge | Motor edge |
|---|---|---|
| Far endpoint | a sensor boundary cell the world **writes** | an actuator boundary cell the world **reads** |
| Disagreement is cleared by | the cell changing its belief | the world moving |

That is the whole difference. A prediction running out along a sensory edge is world modelling; the
identical prediction running out along a motor edge is action, because the far end of that edge
executes. A third kind, the **drive edge**, is added under *Drives* below; it turns out to be a motor
edge attached far from the rim rather than a new kind of thing. The map anticipated that action cells would need "slight modification"; they need none.
The modification is a property of the edge, and the edge already existed.

### The bet this makes

This is the identity **action is prediction the world clears** — see
[ADR-0003](../adr/0003-action-is-prediction-the-world-clears.md). It is the most
literature-fragile commitment in the spec, and it is deliberately made before reading (per the
map's citation sequencing). A citation pass checks it explicitly; if active predictive coding
turns out to require something structurally two-sided, this section and everything downstream of
it are revised rather than defended.

### What happened to the streams

"Stream" is retired as vocabulary — it named nothing structural. The word was carrying a real
intuition, though: in hierarchical active predictive coding, abstraction *rises* through sensory
processing and *falls* through action selection. That shape survives here as a **path**, not as
two objects. A walk from the sensory rim inward to the centre and back out to the motor rim has
exactly that rising-then-falling profile (see *Abstraction*, below). Nothing needs to be built to
make the two streams two; they are the two halves of one walk.

## The boundary

### Boundary cells

The world appears in the graph as **boundary cells**: degenerate cells obeying the contract, whose
node stalk the environment writes (sensory) or reads (motor). Ordinary cells attach to them by
ordinary edges with ordinary restriction maps, and boundary cells enter and leave through the same
message-passing phase as anything else — as [`02-tick-semantics.md`](./02-tick-semantics.md)
already required of any external edge.

Two consequences, both accepted:

- **The world is one tick away, like any neighbour.** There is a fixed sensorimotor latency: one
  tick out, one tick back. The agent's own body is something it models across a delay rather than
  has direct access to.
- **No arbitration mechanism is needed.** Where several cells attach to the same actuator boundary
  cell, reconciliation on those edges *is* the arbitration. It costs one tick, like everything else.

There is no designated output cell, no privileged read-out map over a set of cells, and no decode
path to torque that bypasses a stalk.

### The sensory boundary is a tiling

The world does not respect `n`. The sandbox's render is 64×64; its action is three torques.

Sensory input arrives **sliced across many boundary cells**, each owning a patch — the visual
boundary is a tiling, not a single cell holding a compressed image. This is the patchwork intuition
applied at the seam: the large-scale problem is divided, and reconciliation on the graph is what
puts it back together. Retinotopy is then not designed; it falls out, because adjacent patches are
adjacent cells.

The actuator boundary is one cell whose stalk the arm reads three components of.

### Efference copy

The actuator boundary cell writes back **what was actually applied** — post-clip, post-saturation —
so a motor edge carries ordinary disagreement between commanded and applied.

Without this, the motor edge would be the only edge in the architecture with no disagreement on it,
and the cell nearest the body would be the one cell getting no local signal from its most important
edge. With it, every edge in the graph is the same kind of object. The arm's torque limits
(3 / 2 / 1 N·m) make the residual real information rather than an echo: what the world does not
clear is the body's refusal, and that is exactly the quantity a cell should learn its own limits from.

### The membership rule

Everything between the world and a cell falls into exactly three categories:

1. **The body** — the embodiment: whatever physically interacts with the world. Actuator limits,
   sensor formatting, the sandbox's own physics. Not ours to design; the graph models it across a
   delay.
2. **The graph** — the judgment centre, the neocortical analogue. *Every* transformation of
   information on the way in belongs here, including relay stopovers and shared broadcast subspaces
   (a thalamic-style unification of the visual edge is a relay cell, not preprocessing: it costs a
   tick and obeys the contract). Where such a stopover goes is a topology question — see
   [#8](https://github.com/NGL321/patchworks/issues/8) — not a boundary question.
3. **A subordinate executor** — outbound only: a cerebellar/medullar-style controller the graph
   *commands* rather than a thing that models the world, taking a body-pose latent and a
   desired-behaviour latent and running the next step of a motor loop. Best understood as an
   architectural constraint on the graph rather than a peer of it.

**Banned, and narrowly:** an out-of-graph module that **compresses across slices on the way in**.
Not because it would be learned, and not because of its direction — because that specific act is
what cells exist to do, and doing it outside would be answering the question this architecture
exists to answer.

**For this proof of concept there is no subordinate executor.** Three torques go straight to the
actuator boundary cell. Such a module earns its place only when motor variety exceeds what direct
torque control can carry, which is past this destination; it is held in the map's fog as the escape
hatch if fine motor control proves beyond the graph.

## Abstraction

**Abstraction is hop distance from the sensorimotor rim.** There is no level attribute on a cell, no
assigned tier. A cell is abstract exactly insofar as it is many hops from the world, which by unit
delay means it also sits in a longer loop and sees staler information.

The measure is from the **sensorimotor** rim specifically, not from any boundary whatsoever.
Boundary cells are alike *mechanically* — all are seams where something outside writes or reads a
stalk — but they are not alike in kind. Sensory and motor boundaries are where the world touches
the graph. Internal faculties (a hippocampal-analogue memory module) attach by the same mechanism at a
**different rim**, and are not thereby concrete: they belong at greater abstraction, and attach
adjacent to the abstract pole rather than next to the arm. The first such faculty is already built:
a **drive boundary cell** at the core (*Drives*, below) is that rim, arriving earlier than expected.

Two things this makes mechanical rather than vague:

- Where a drive attaches (*Drives*, below) becomes a question about hop count. It attaches at the
  core, which is as far from the sensorimotor rim as the graph goes.
- The acceptance demo's "recovered at the appropriate level of the hierarchy" becomes falsifiable:
  you can measure how many hops from the sensorimotor rim the correction originated.

What it does **not** give is differing update *rates* — `01-cell-and-sheaf.md` is explicit that
depth buys horizon, not rate. Multiple timescales remain a real open question
([#7](https://github.com/NGL321/patchworks/issues/7)), not something answered here by accident.

### Shape

The boundary is **one rim with depth away from it**, not two poles with a strip between. Sensory and
motor boundaries sit near each other, and the **shortest sensorimotor loop is a design budget**: with
unit delay, a graph whose sensory and motor boundaries are at opposite ends has a reflexive
correction cost equal to its own diameter, and the acceptance demo is a recovery demo. The working
intuition is a **disk** — external edges around the rim, abstraction toward the centre, sensorimotor
edges sharing one hemisphere separated by exactly the loop budget, internal faculties on the
opposite arc.

The disk is **not fixed here.** Graph shape belongs to
[#8](https://github.com/NGL321/patchworks/issues/8); the constraints this section hands it are the
loop budget and the one-rim reading. Recorded there as notes: the disk itself, the case it makes for
proto-thalamic pass-throughs as chords across it, and explicit partitions — enforced through the
hand-specified structural mask rather than as new mechanism — held in reserve if the
undifferentiated graph needs constraining.

## Drives

### The dark room problem

A pure prediction-error minimiser has no reason to act. If the agent correctly predicts that the puck
stays put and the target zone stays lit, that prediction is **right** — and right predictions are what
this architecture minimises. An unsolved task, watched from a standstill, is a *low-error* state.

So goal-directed behaviour is not something to hope emerges. Something outside the graph has to assert
that an unmet task is uncomfortable, and it is that discomfort, not any planner, that makes the agent
move.

### A drive is a boundary cell in the core

A **drive boundary cell** is written from outside the sheaf — by the human today, by an internal
faculty later — and holds a standing assertion: *satisfied*. Like every boundary cell it runs no body
and holds no chart ([ADR-0006](../adr/0006-boundary-cell-stalks-are-world-shaped.md)). Its **drive
edges** run to a subset of **core** cells and are ordinary edges with ordinary masked linear
restriction maps.

Nothing is overridden. The assertion reaches those cells as ordinary disagreement, pulled by the
ordinary reconciliation gain, and every cell it touches keeps its body and keeps inferring every tick.
It nudges; it does not constrain. See
[ADR-0009](../adr/0009-a-drive-is-a-motor-edge-attached-deep.md), which also records why the earlier
proposal — pinning a cell's node stalk to a goal value — was rejected: a genuine override removes that
cell from inference, leaving a hole in the network dynamics.

**A drive edge is a motor edge.** Sorting by who clears the disagreement, which is the only sort this
spec makes:

| | Sensory edge | Motor edge | Drive edge |
|---|---|---|---|
| Boundary cell is | written by the world | read by the world | written from outside, read by no one |
| Disagreement cleared by | the cell changing its belief | the world moving | the world moving |

The actuator's motor edge is cleared by the world moving *immediately*; a drive edge by the world
moving *eventually*. Because abstraction is hop distance from the sensorimotor rim, **abstract action
is literally a motor edge attached deep** — the same object as a torque command, further in.

### Valence, not specification

The drive stalk is **near-scalar**. It asserts *satisfied* and nothing else. It never names a puck or a
zone, because the render already does: the target zone lights up, and `retarget()` only changes what is
seen ([`03-the-sandbox.md`](./03-the-sandbox.md)). The drive supplies the discomfort; the world supplies
the content. Direction is absent from the signal and comes instead from the graph's own learned model
of what satisfaction looks like.

Low bandwidth is the point. A wide drive channel would smuggle the task specification back in through
the side door, after the sandbox worked to put it in the render.

**One cell is one drive.** For this PoC there is exactly one — the task drive — with several drive
edges into the core. Curiosity, fatigue, or any later drive arrives as an additional boundary cell,
which is an ordinary structural-mask change and needs no new mechanism.

### What follows without being built

- **Release needs no detector.** The cell asserts *satisfied* forever. When the task is met, what is
  sensed agrees and disagreement falls to the floor; pressure vanishes with nothing released. `perturb()`
  knocks the puck out and the disagreement returns by itself. Nothing in the graph ever reads whether
  the goal is met — `info.goal_satisfied` stays privileged, for logging only.
- **Planning is drive propagation.** A drive asserted in the core becomes, some ticks later, a motor
  prediction at the rim, by the same one-step-per-tick machinery as everything else. No rollout, no
  search, no separate planner, no mode in which the agent simulates without acting. Plan depth is graph
  depth; deliberation time is literally ticks.
- **Epistemic action needs no new channel.** A pure disagreement-minimiser's optimal policy is to sit
  still facing a blank wall, and the destination requires an agent that explores a workspace nothing
  rewarded it for visiting — so this is not waved away as emergent. Whatever eventually supplies
  curiosity enters as **another drive boundary cell**, not as a new channel and not as a second error
  signal.
- **Several drives compose by reconciliation**, exactly as several cells driving one actuator do.
  Incompatible drives are standing disagreement, a fourth source alongside static, lag, and settling
  ([ADR-0007](../adr/0007-the-disagreement-floor-is-tolerated-not-represented.md)) — tolerated, not
  represented, and needing no arbitration mechanism.

*Considered and rejected:* **saccades** — restricting the agent to a small moving viewport, as the
active predictive coding papers do. The whole field is visible here. It is visible *as many cells*,
so if attention is ever wanted it arrives as **gating on transport** rather than as an imposed
viewport. That is *not* the change gate of [`05-timescales.md`](./05-timescales.md), which is
outbound and driven by the sender's own rate of change: attention selects which **inbound** evidence
a cell weights, and its likely mechanism is the semi-global reach of the core's broadcast subspace
or of relay cells ([`06-graph-topology.md`](./06-graph-topology.md)) rather than a per-edge
threshold. Neither of those is established, so attention is unspecified here and stays in the map's
fog.

## Known exposure

- **The APC bet.** *Action is prediction the world clears* is strong for reflexive control and
  historically shakier for anything needing lookahead. Flagged for the citation pass; a revision
  sweep through this section and its dependents is pre-accepted.
- **A scalar drive steering a 150-cell graph is unproven.** Low bandwidth is deliberate, but whether
  one dimension of standing disagreement can differentiate behaviour across the whole taper is the
  thing most likely to need widening. Escape hatch: a small learned drive vector, at the cost of the
  one-cell-one-drive reading (ADR-0009).
- **Hallucinating satisfaction.** A core cell can reduce disagreement by *believing* the task is met
  rather than by acting. Bounded rather than eliminated: the sensory edges pull the other way, so the
  cell settles at a compromise, and that compromise is the prediction the motor rim must clear. Leaves
  an observable signature — sensory-side disagreement growing while the motor side stays quiet.
- **No plan comparison.** Nothing in this architecture evaluates counterfactuals. The agent settles
  into a route; it does not compare two. The candidate answer — several drives propagating from
  different abstract regions, colliding, and reconciling — is genuinely promising and genuinely
  unproven, and it is where explicit lookahead would be reopened if it fails. Its own ticket:
  [route selection](https://github.com/NGL321/patchworks/issues/25). Not expected to bite in the
  first testbed.
