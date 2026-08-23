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
executes. The map anticipated that action cells would need "slight modification"; they need none.
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
the graph. Internal faculties (a limbic-analogue reward signal, a hippocampal-analogue memory
module) attach by the same mechanism at a **different rim**, and are not thereby concrete: they
belong at greater abstraction, and attach adjacent to the abstract pole rather than next to the arm.

Two things this makes mechanical rather than vague:

- Which cells a goal may be clamped on ([#9](https://github.com/NGL321/patchworks/issues/9))
  becomes a question about hop count.
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

## Planning and epistemic action

**Planning is clamp propagation.** A goal clamped on a cell deep in the graph becomes, some ticks
later, a motor prediction at the boundary, by exactly the same one-step-per-tick machinery as
everything else. No rollout, no search, no separate planner, and no second mode of operation in
which the agent simulates without acting. Plan depth is graph depth; deliberation time is literally
ticks.

**Epistemic action needs no new channel.** A pure disagreement-minimiser's optimal policy is to sit
still facing a blank wall, and the destination requires an agent that explores a workspace nothing
rewarded it for visiting — so this is not waved away as emergent. The architectural decision is
narrow: whatever drive eventually supplies curiosity (the limbic analogue, still in fog) enters as
**an ordinary clamp on an ordinary cell**, not as a new channel and not as a second error signal.

*Considered and rejected:* **saccades** — restricting the agent to a small moving viewport, as the
active predictive coding papers do. The whole field is visible here. It is visible *as many cells*,
so if attention is ever wanted it arrives as gating on transport
([#20](https://github.com/NGL321/patchworks/issues/20)) rather than as an imposed viewport.

## Known exposure

- **The APC bet.** *Action is prediction the world clears* is strong for reflexive control and
  historically shakier for anything needing lookahead. Flagged for the citation pass; a revision
  sweep through this section and its dependents is pre-accepted.
- **No plan comparison.** Nothing in this architecture evaluates counterfactuals. The agent settles
  into a route; it does not compare two. The candidate answer — several clamps propagating from
  different abstract regions, colliding, and reconciling — is genuinely promising and genuinely
  unproven, and it is where explicit lookahead would be reopened if it fails. Its own ticket:
  [route selection](https://github.com/NGL321/patchworks/issues/25). Not expected to bite in the
  first testbed.
