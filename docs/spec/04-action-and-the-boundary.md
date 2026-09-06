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

**A boundary cell is written by the outside or read by it, never both.** This is a contract-level
ban, not a graph-design preference, and it belongs here rather than being deferred to whoever draws a
graph ([#128](https://github.com/NGL321/patchworks/issues/128)). The tick's ordering has no defined
meaning otherwise: the external write is a tick's **last word**
([`02-tick-semantics.md`](./02-tick-semantics.md)), so a cell that is also *read* by the world would
be read *before* the word it was going to say.

The ban was invisible while the world was a sandbox, because a dome's sensory and motor rims are
different organs and could not have shared a cell. A domain that carries both on one alphabet — say
characters heard and characters spoken — invites the collapse, and separate boundary cells for the two
directions make the loop **longer** than the sandbox's rather than degenerate, which keeps the
sensorimotor loop budget below exactly what it says it is.

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

### Readback

**A motor boundary cell has a readback — what the world made of the command — and a domain that
cannot supply one must say why.** That is the contract requirement, and it is mandatory.

In the sandbox the readback is an **efference copy**: the actuator boundary cell writes back what was
actually applied — post-clip, post-saturation — so a motor edge carries ordinary disagreement between
commanded and applied.

Without a readback, the motor edge would be the only edge in the architecture with no disagreement on
it, and the cell nearest the body would be the one cell getting no local signal from its most
important edge. With it, every edge in the graph is the same kind of object. The arm's torque limits
(3 / 2 / 1 N·m) make that disagreement real information rather than an echo: what the world does not
clear is the body's refusal, and that is exactly the quantity a cell should learn its own limits from.

> **Why the word widened** ([#128](https://github.com/NGL321/patchworks/issues/128)). "Efference copy"
> names a mechanism that works *because the arm clips*. A character emitted to an interlocutor is not
> clipped, so the requirement had to be stated in terms of what it is for rather than how the sandbox
> supplies it.

**A deterministic readback is permitted, and forfeits exactly one thing.** Where the world always
complies, the readback carries no refusal and the **limit-learning** function is gone — an actuator
with no limits has none to learn. Nothing structural breaks: several cells attach to an actuator
boundary cell and reconciliation on those edges *is* the arbitration, so disagreement still exists on
a motor edge, and both learning rules keep their signal. Recorded so that a future compliant actuator
is a documented case rather than a surprise.

**Language is probably not that case.** Once silence is a value and the agent emits every tick,
whether a character was actually taken up — dropped, interleaved, or landing mid-turn — is a real
readback. **Turn-taking is the body's refusal on a language rim.**

### The membership rule

**This rule governs the world boundary.** It sorts what may sit between the *world* and a cell, and it
says nothing about what attaches at any other rim — see *The internal rim*, below, which is a separate
rule with a separate ban. The distinction was invisible until the drive boundary cell existed: written
from outside, inbound, and not the world, it is on no path this rule describes.

On the world path, everything between the world and a cell falls into exactly three categories:

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

### The internal rim

A fourth category, and the reason the rule above needed scoping:

4. **An internal faculty** — something outside the sheaf that writes or reads a boundary cell at a rim
   other than the sensorimotor one. It is not the world, not on the world's path, and not commanded by
   the graph the way a subordinate executor is. The **drive boundary cell** (*Drives*, below) is the
   first and, in this proof of concept, the only one; a limbic-analogue appetite and a
   hippocampal-analogue memory are the fog items that would join it.

Mechanically it needs nothing: it attaches as an ordinary boundary cell by ordinary edges, and the
graph cannot tell a belief that came from a faculty from one that came from a neighbour. What it needs
is a ban, and there are two:

- **An internal faculty may not hold its own model of the world.** This is the graph's job, and the
  whole point of the graph is that its model is answerable to the world through the sensorimotor rim. A
  faculty carrying its own would be imposing a model from outside that nothing corrects — and two
  models with no sheaf between them have no defined way to disagree, so there would be no mechanism by
  which the graph could even discover the imposition. The drive obeys this trivially: it holds a
  constant, not a model.
- **An internal faculty reaches the world only through the graph.** Never directly. This is close to
  implied by the word *internal*, but the two bans are distinct — a faculty could read only what the
  graph gives it and still build a forward model over that — so both are stated. A hippocampal-analogue
  reads abstract frames from core cells, never the render.

Together these are the internal-rim counterpart of the narrow inbound-compression ban above, and they
have the same shape: the graph's job stays in the graph.

**Attention is not in this category.** Its likely mechanism is the semi-global reach of the core's
broadcast subspace or of relay cells ([`06-graph-topology.md`](./06-graph-topology.md)), both of which
are **in-graph** and therefore category 2.

## Abstraction

**Abstraction is hop distance from the sensorimotor rim.** There is no level attribute on a cell, no
assigned tier. A cell is abstract exactly insofar as it is many hops from the world, which by unit
delay means it also sits in a longer loop and sees staler information.

The measure is from the **sensorimotor** rim specifically, not from any boundary whatsoever.
Boundary cells are alike *mechanically* — all are seams where something outside writes or reads a
stalk — but they are not alike in kind. Sensory and motor boundaries are where the world touches
the graph. Internal faculties (a hippocampal-analogue memory module) attach by the same mechanism at
the **internal rim** (*The internal rim*, above), and are not thereby concrete: they belong at greater
abstraction, and attach adjacent to the abstract pole rather than next to the arm. The first such
faculty is already built: a **drive boundary cell** at the apex (*Drives*, below) is that rim,
arriving earlier than expected.

Two things this makes mechanical rather than vague:

- Where a drive attaches (*Drives*, below) becomes a question about hop count. It attaches at the
  **apex**, which is as far from the sensorimotor rim as the graph goes.
- The acceptance demo's "recovered at the appropriate level of the hierarchy" becomes falsifiable:
  you can measure how many hops from the sensorimotor rim the correction originated. What is measured,
  and what counts as passing, is [`08-the-acceptance-demo.md`](./08-the-acceptance-demo.md).

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

### The dark room on a language rim

The problem is general; **its signature is not**, and the language domain's is different enough that
looking for the sandbox's would miss it entirely.

**The direct analogue is gone.** On a language rim silence is a *value*
([#128](https://github.com/NGL321/patchworks/issues/128)), so the agent emits a character every tick
whether or not it has anything to say. There is no standstill available: **the agent cannot fail to
act.**

**The failure takes a new form: constant emission** — the agent settling on whichever single character
is cheapest to keep predicting, plausibly a space or a vowel rather than the idle symbol itself. That
is a degenerate fixed point, it is reachable, it is stable, and it is low-error for exactly the reason
the dark room is.

**Under an idle run yielding the floor, two failures collapse into one object.** Constant emission
means never yielding the floor, so the interlocutor never speaks, so the incoming stream is pure idle
and maximally predictable ([`12-the-interlocutor.md`](./12-the-interlocutor.md), *Half-duplex, and an
idle run yields the floor*). The degenerate fixed point and the failure to yield the floor are the
**same object**, which is why one observable catches both.

**The observable is emission entropy, not idle-symbol fraction.** Idle fraction reads only the
special case where the character the agent settles on happens to be idle; entropy reads the whole
family, and the family is what the failure actually is.

This gives ADR-0009's *Bootstrapping* exposure its language-domain reading: in the sandbox it is no
motion at all, and here it is **emission entropy pinned near zero while drive-edge disagreement is
non-trivial**. The response is the one ADR-0009 already names and is **not** a rung on the width
ladder — a **curiosity drive**, arriving as an ordinary drive boundary cell at the internal rim.
Widening the coherence drive does not reach it, for the reason *Valence, not specification* gives
below.

**That reading is conditional on [#288](https://github.com/NGL321/patchworks/issues/288), and a
drained run yields no reading at all.** The second conjunct has a live confound already in the
register: `open-problems.md` carries *Disagreement can drain under drive, taking the only instrument
that would show it* **(overdue)** — [#288](https://github.com/NGL321/patchworks/issues/288) /
[#324](https://github.com/NGL321/patchworks/issues/324), cutoff `driven_settling` at
`draining_effective_rank < 2`. Under drive, per-edge Dirichlet energy falls while per-edge effective
rank slides toward 1 across the fleet — and if that is what is happening, *entropy near zero* **and**
*drive-edge disagreement trivial* is exactly what a **drained** run looks like. The pre-registered
pair cannot then separate **the dark room was not answered** from **the instrument died**.

So #288 is this reading's **precondition**, not a neighbouring problem: `driven_settling` has to
clear its cutoff before the entropy reading means anything, and a run whose drive-edge disagreement
has drained yields **no reading** — not a negative one. Nothing is minted here
([ADR-0029](../adr/0029-a-problem-is-minted-by-a-human-a-proposal-is-not.md)); the row and its ticket
already exist, and what was missing was the connection to this reading
([#449](https://github.com/NGL321/patchworks/issues/449)).

### A drive is a boundary cell in the core

A **drive boundary cell** is written from outside the sheaf — by the human today, by an internal
faculty later — and holds a standing assertion: *satisfied*. Like every boundary cell it runs no body
and holds no chart ([ADR-0006](../adr/0006-boundary-cell-stalks-are-world-shaped.md)). Its **drive
edges** run to the **apex** — one edge to each of the eight L7 cells — and are ordinary edges with
ordinary masked linear restriction maps. The apex is the most abstract place in the graph and the only
part of the core with private dimension to spare; `06-graph-topology.md` (*Where the drive attaches*)
records the arithmetic.

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

The drive stalk is **scalar** — one dimension, and its edges are `m_e = 1`. It asserts *satisfied* and
nothing else. It never names a puck or a
zone, because the render already does: the target zone lights up, and `retarget()` only changes what is
seen ([`03-the-sandbox.md`](./03-the-sandbox.md)). The drive supplies the discomfort; the world supplies
the content. Direction is absent from the signal and comes instead from the graph's own learned model
of what satisfaction looks like.

Low bandwidth is the point. A wide drive channel would smuggle the task specification back in through
the side door, after the sandbox worked to put it in the render.

**The stalk's width and the lane's are one decision.** A restriction map out of a one-dimensional stalk
has rank at most one, so a wider lane carries nothing more — it only lets the drive assert *zero*
in the surplus directions, which is an arbitrary constraint on a core cell expressed in content the
drive does not have. Matching them is what makes "valence, not specification" exact rather than
approximate.

**Strength is fan-out, not width**, correcting
[ADR-0009](../adr/0009-a-drive-is-a-motor-edge-attached-deep.md), which named mask width as the knob.
Widening a drive edge would raise its share of a cell's pull and, by the same `γ / Σ_e m_e`
normalisation, **turn down every other edge that cell has** — the drive making itself heard by making
the world quieter, which is the wrong trade at any width. More attachment points spread that cost
across cells rather than concentrating it, and never make one cell pay more than the minimum. The
learned drive vector ADR-0009 holds in reserve is the move after that one, not before it.

**The exactness is in the stalk, and it is not established in the edge.** A one-dimensional stalk
asserting a constant — `DRIVE_ASSERTION = 1.0`,
[#137](https://github.com/NGL321/patchworks/issues/137) — has **zero channel capacity**: it cannot
carry which task, how well, or how nearly done, so specification cannot ride on the *signal*, by
arithmetic rather than by measurement. That says nothing about the *edge*. The drive's eight
restriction maps are trained by the same transport rule as every other map — boundary cells held to
gauge exactly 1, otherwise ordinary
([ADR-0032](../adr/0032-the-maps-learn-isometric-transport-and-a-spectral-floor-expresses-it.md)) —
so `F_e` out of the drive stalk is a **learned** `1 → 8` direction into each L7 cell, and *which apex
direction discomfort pushes* is a learned parameter. That is the one place task content can re-enter
after the stalk kept it out, and it is ADR-0009's learned drive vector — held in reserve above —
arriving by the back door rather than by decision
([#449](https://github.com/NGL321/patchworks/issues/449)).

**It is observable, and pre-registered here as a ratio rather than a displacement**, because
[#339](https://github.com/NGL321/patchworks/issues/339) says the transport rule wanders at ~`η`
forever with no fixed point at agreement, so a bare displacement has no zero to be read against:

> `retarget()` changes only what is seen
> ([`03-the-sandbox.md`](./03-the-sandbox.md)), so under *Valence, not specification* the drive edge
> maps are **invariant across a retarget** while behaviour changes. Read `F_e` for the eight drive
> edges across a retarget boundary, against two controls: the same maps' displacement over an equal
> tick count with **no** retarget, and both readings for a matched sample of **non-drive** edges.
> **Confirmed** if retarget-crossing displacement is indistinguishable from the drift control; **it
> bites** if drive maps move more across a retarget than non-drive maps do.

This is written, not run. No ticket has been taken for the run.

**One cell is one drive.** For this PoC there is exactly one — the task drive — with several drive
edges into the core. Curiosity, fatigue, or any later drive arrives as an additional boundary cell,
which is an ordinary structural-mask change and needs no new mechanism.

### What follows without being built

- **The assertion stands because external writes are the tick's last word.** Reconciliation moves the
  drive cell's node stalk like any other, and the write that follows the message-passing phase restores
  it before it next speaks ([`02-tick-semantics.md`](./02-tick-semantics.md), *External writes*). So
  disagreement on a drive edge is never reduced from the drive's side; only the core cell can move.
  That is the motor-edge signature — *cleared by the world moving* — falling out of tick order rather
  than being stipulated, and it is why the drive does not drift toward what the graph already believes.
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
  signal. It has to enter that way: the epistemic term that would otherwise derive novelty-seeking is
  an expectation over candidate futures, so ADR-0003 makes it **architecturally unavailable**, not
  merely unchosen (ADR-0009, *Known exposure*).
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

## Route selection

Two routes reach the same goal. Reconciliation is one local descent step against a delayed
neighbour belief — an averaging operation — and the mean of *go left* and *go right* is a route
through the obstacle. This section says what the architecture does about that, and the short answer
is that it never has two routes to average.

### There is no route

**Route multiplicity is never drive multiplicity.** A drive asserts *satisfied* and nothing else, so
every route that would relieve it is equally satisfying to it; the drive cannot carry two. Nor does
the graph hold a route anywhere else — nothing in this architecture stores a plan, compares plans,
or has a slot a plan could sit in. What looks from outside like a choice between routes is, inside,
several predicting cells holding incompatible predictions about **the next step**, which is ordinary
edge disagreement between ordinary cells.

**The world selects.** A blended prediction is not a decision; it is an unstable state. It still
leaves by the motor edge, the arm still moves *somewhere*, and one tick later the sensory edges come
back disagreeing with the blend. By [ADR-0003](../adr/0003-action-is-prediction-the-world-clears.md)
the world is the only thing entitled to clear a motor edge, so the world is the only thing that
resolves a tie. Selection is closed-loop and one tick wide.

So the architecture does not lack plan comparison. **It declines to have a plan** — and having
declined, has nothing left to compare. Behaviour can still look planned without a plan being stored
anywhere; that is the bet, and it is made deliberately rather than conceded.

### What makes a tie-break stick

The world breaking ties every tick would be dithering, not commitment. Two things already in the
spec stop that, and neither is built for this.

**The folds.** `encode` is piecewise linear and a cell occupies **one activation region at a time**
([`05-timescales.md`](./05-timescales.md), *the regional Jacobian*). Region membership is genuinely
discrete: there is no interpolating between two regions, only being in one. A tie broken is a region
entered. This is not a clean latch — a cell's operating point moves it as much as its biases do
([#41](https://github.com/NGL321/patchworks/issues/41)) — so it is offered as the reason a tie-break
is not re-litigated from scratch every tick, not as a mechanism that guarantees one.

**`H⁰` insulation is the hysteresis.** Reconciliation descends along `im δᵀ` while private features
are `ker δ = (im δᵀ)^⊥`, so the private component of a node stalk is **exactly invariant** under
reconciliation ([`05-timescales.md`](./05-timescales.md)). A commitment held in a cell's private
features is therefore unreachable by a neighbour's contrary belief: **the losing route cannot
re-assert through message passing at all.** Only the cell's own prediction rule, driven by prediction
error, can move it.

Two consequences, both wanted:

- **Commitment deepens with abstraction.** Private dimension is a gradient rising from 0 at the rim
  to ~15 at the apex ([`06-graph-topology.md`](./06-graph-topology.md)), so deep cells commit hard
  and rim cells stay fluid. Strategy persists; torque does not.
- **A bad commitment is protected by the same mechanism.** Nothing distinguishes a well-chosen route
  from a badly-chosen one, and only accumulated prediction error digs either out. This is the price
  of getting hysteresis for free, and it is paid, not avoided.

No new mechanism is introduced here. In particular, **gating on transport is not reached for** —
[`05-timescales.md`](./05-timescales.md) specifies the change gate and does not build it, and route
commitment is not a reason to.

### The test: the workspace is an annulus

The arm is anchored at the centre of its own workspace and **cannot fold through the pedestal**
(`arena.xml`; links 1 and 2 collide with it). The paddle's reachable set is therefore an **annulus**,
inner radius 0.11 (pedestal 0.08 + paddle 0.03), outer radius 0.49. An annulus is not simply
connected: paths between two bearings fall into distinct classes, and there is no continuous
deformation from one to another.

This makes route selection **frequent and unavoidable** rather than occasional. Every repositioning
of the paddle across the arena — which happens several times per task, whenever the arm must get
behind a puck on the far side — is a choice of swing direction. It follows from the arm being
anchored at the centre, not from the pedestal being an obstacle, and it would survive the pedestal's
removal.

**Falsification signature:** the arm **stalls mid-swing** — near-zero commanded torque with standing
motor-side disagreement — instead of committing to a direction. The blend of swing-left and
swing-right is *stay put*, so the failure is unmistakable and needs no instrumentation beyond the
motor-side disagreement readout the demo already carries — the actuator's commanded-outline against
applied-fill bars in [`10-the-demo-surface.md`](./10-the-demo-surface.md), *The somatomotor strip*,
where the signature is an outline near zero beside a disagreement bar that stands.

A second, rarer case is the puck's own route around the pedestal. Measured over the sampler
([`03-the-sandbox.md`](./03-the-sandbox.md), *The world*), it is genuine but shallow — a median 4%
detour — and worth watching rather than testing.

### Horizon

Two questions travel together under "long-horizon planning" and separate cleanly here.

**Horizon as duration** is a measurable quantity, not a hope. Plan depth is graph depth and
deliberation time is ticks; how long a deep cell holds a commitment is the decay rate of its private
component. What sets that rate is a cell's **learned operator** `K`
([ADR-0028](../adr/0028-a-cell-holds-a-spectrum-of-retention-constants.md)). Measured on driven runs
of the built graph, `ρ(K)` gives a **median `τ` of 19–27 ticks** across 150 cells, with an across-cell
p95/p05 ratio of roughly **8–20×** — the median is the reproducible half of that reading and the ratio
is not, for the reason `docs/research/027`'s amendment sets out
(`prototypes/regional-spectra/converted_spread.py`, [#349](https://github.com/NGL321/patchworks/issues/349)).

> **Corrected by #349.** This paragraph quoted #27's **7.7×** as the operative figure. That number was
> measured on a **stand-in** body, under the bias mechanism ADR-0028 retired, and it is withdrawn:
> `docs/research/027-regional-jacobian-spectra.md`'s amendment has the re-run. The correction is not
> only to the number but to its source — duration is now bought by a learned per-cell operator, so
> **too short a horizon is no longer a body-construction defect with a knob.** Whether learning
> produces the gradient is ADR-0028's own pre-registered falsification, and it is
> [#357](https://github.com/NGL321/patchworks/issues/357) that watches it.

**Horizon as detour** — can the agent execute something that gets worse before it gets better? —
dissolves rather than resolves. The drive asserts *satisfied* and supplies no direction; direction
comes from the graph's own learned model of what satisfaction looks like. **"Worse" is not a
quantity this agent computes.** If the model has learned that wrapping is what leads to satisfaction,
the wrapping trajectory *is* the low-disagreement prediction and there is no detour needing
justification. The detour problem is an artifact of imagining a distance-to-goal signal the
architecture does not have.

This is not free. It converts a planning problem into an **exploration** problem: a route the agent
has never experienced is not merely unplanned, it is invisible.

Nothing in the PoC deliberately answers that, and that is a position rather than an omission
([#17](https://github.com/NGL321/patchworks/issues/17)). The account that stood here was that early in
training the model is poor everywhere, so the torques the graph emits are near-arbitrary and the arm
visits configurations it has no model of; the model then sharpens where it acted — exploration as a
consequence of not yet having a model rather than a mechanism, self-extinguishing, the dark room
arrived at late rather than avoided. **That account is measured false, and it is void**
([#154](https://github.com/NGL321/patchworks/issues/154)).

**What an untrained agent actually emits is one constant torque.**
[#120](https://github.com/NGL321/patchworks/issues/120) measured the command sitting at a fixed point
— sd ≤ 3.5e-6 over hundreds of ticks, and identical to four decimal places across two *different*
worlds at the same seed — with every joint pinned against its stop and travel over the last 300 ticks
exactly zero. The untrained emission is a **world-independent constant**. Near-arbitrary is not what
it does, and **any claim resting on exploratory torques appearing for free is void**, including the
bet this section used to make: that discomfort under a standing drive keeps the arm moving through the
window in which the model is still poor.

What the same run does *not* show is a shortage of disagreement — 682 of 682 edges disagree with the
arm pinned, mean 1.66. **The scarce resource is variety, not disagreement:** new directions in the
disagreement, not its existence. And on #154's ruling the steady-state account of where variety comes
from needs **nothing added**. The drive pressures the network to act, action changes the state of the
world, and so the supply of new directions is never short; there is always room for acting and the
senses are never constrained. Every link in that is the **outbound** leg — apex to actuator — which is
exactly what reads zero today (#120: drive → command sensitivity **0**, flat across six decades of
assertion). So the exploration famine is not a second failure owed its own mechanism. It is the
**outbound failure the map already owns**, read by
[#242](https://github.com/NGL321/patchworks/issues/242)'s influence predicate.

That ruling is conditional, and its condition is pre-registered. If #242's outbound clause **passes**
and the world still does not vary — the arm still locks, or per-edge excitation rank stays below that
edge's stalk width `m_e` — then the famine is a failure in its own right and a supplier is owed. Until
that fires, the response stays **gated rather than owed**: the **curiosity drive** held in fog, which
enters as an ordinary drive boundary cell at the internal rim and needs no new channel and no change
to anything here. A drive that produces *no* motion at all remains
[ADR-0009](../adr/0009-a-drive-is-a-motor-edge-attached-deep.md)'s *Bootstrapping* exposure and not its
scalar-width one. Note what is *not* claimed: this says nothing about whether an agent with a converged
model will visit a route it has never taken. It will not, and by then it has no term that would make it
want to. This is the dynamics-exploration argument's limit, stated where it bites —
[`03-the-sandbox.md`](./03-the-sandbox.md), *Dynamics exploration, not spatial exploration*, covers
the other half.

The bet underneath, stated as a bet: that the taper's degrees of separation between abstract and
concrete let general structure — *wrapping*, *routing around* — be learned at depth and reused,
rather than each route being memorised whole. Nothing here proves that; it is the reason the
architecture is shaped this way.

### The escape hatch

If the stall signature persists after the model has demonstrably learned the dynamics, what reopens
is **not** explicit lookahead as new mechanism inside the graph. It is the **hippocampal faculty**
already held in the map's fog and already located at the dome's apex — which turns out to have been
the planner all along.

It costs the one-algorithm claim **nothing**: it attaches as a boundary cell at the abstract rim,
exactly as the drive does, and the graph cannot tell the difference between a prediction that came
from a faculty and one that came from a neighbour. Its rollout is expected to be **retrieval, not
simulation** — asking which abstract frames historically *followed* this one, rather than running
candidates forward — which is what keeps ADR-0003 intact and what makes "must have experienced it"
a property of the mechanism rather than an apology for it. Unspecified here; see the map's fog.

## Known exposure

- **The APC bet.** *Action is prediction the world clears* is strong for reflexive control and
  historically shakier for anything needing lookahead. Flagged for the citation pass; a revision
  sweep through this section and its dependents is pre-accepted.
- **A scalar drive steering a 150-cell graph is unproven.** Low bandwidth is deliberate, but whether
  one dimension of standing disagreement can differentiate behaviour across the whole taper is the
  thing most likely to need widening. **Trigger:** task-invariant behaviour — the arm's trajectory the
  same across tasks that differ only in the render, while the drive edge's disagreement is non-trivial.
  Confirmed, if wanted, by an undifferentiated apex: the eight apex node stalks moving near-identically
  under drive. A drive that produces *no* motion is not this failure — that is bootstrapping, and
  widening the channel is the wrong fix. **Escape hatch, in rungs** (ADR-0009): more attachment points,
  then a second drive cell, then a learned drive vector at `k ≈ 16`, the attested width for a
  directional goal in a learned latent space.
- **Hallucinating satisfaction.** A core cell can reduce disagreement by *believing* the task is met
  rather than by acting. Bounded rather than eliminated: the sensory edges pull the other way, so the
  cell settles at a compromise, and that compromise is the prediction the motor rim must clear. Leaves
  an observable signature — sensory-side disagreement growing while the motor side stays quiet.
- **Plan comparison is declined, not absent.** Nothing here evaluates counterfactuals because
  nothing here holds a plan to evaluate (*Route selection*, above). What remains exposed is narrower
  and sharper than "no lookahead": whether **`H⁰` insulation plus the world's tie-break** is enough
  commitment in practice, when neither was designed for it. The annulus makes this bite in the
  **first testbed**, several times per task — a correction to the earlier expectation that it would
  not — and gives it an unmistakable signature. Escape hatch: the hippocampal faculty in the map's
  fog, at no cost to the one-algorithm claim.
- **A protected commitment is protected whether or not it is right.** `H⁰` insulation cannot tell a
  good route from a bad one, and only prediction error dislodges either. Expected failure: an agent
  that persists at a bad approach longer than a fresh look would warrant.
