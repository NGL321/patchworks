# Patchworks, as an outsider reads it

*External audit, 2026-09-05. This is one reviewer's picture of the project after reading the
vocabulary, the spec, the thirty-two decisions, the registers, the map and about twenty rulings, and
the code that runs. It is written so it can be diffed against the team's own picture: where it is
wrong, the difference is the finding. It goes from one sentence to the mechanism to the theory, and
ends with where my reading probably departs from yours.*

## In one sentence

Patchworks is a bet that a world too big for any one predictor can be modelled by many small
predictors that each keep their own coordinates, agree only at their seams, learn only from what
they can see, and act because a standing assertion makes inaction uncomfortable.

## In one paragraph

Take a hundred and fifty tiny forecasters, each responsible for one square of a scene it never sees
whole. Each keeps private notes in its own shorthand (a twelve-dimensional *chart*), publishes a
bulletin in a larger public vocabulary (a thirty-two-dimensional *node stalk*), and compares bulletins
with its neighbours through narrow shared dialects (three- or four-dimensional *lanes*, reached by
learned linear *restriction maps*). Every tick has two halves: each forecaster advances its own
notes and publishes, then every pair of neighbours takes one step toward agreeing on the lane they
share — one step, never a negotiation to convergence. Two learning signals exist and they never
cross a seam: a forecaster improves its own forecasting on its own error, and improves its dialect
on how much it disagreed with its neighbour. The world enters through *boundary cells* it writes
(pixels, joint angles, touch) and leaves through one it reads (torques). A goal is not a reward and
not a target vector: it is a single number written at the deepest cell asserting that the lit zone
is satisfied while it is not, so the graph carries a disagreement it can only clear by acting. That
is the whole machine. Nothing in it is new alone; the bet is on the conjunction.

## The object, concretely

**A cell** holds three spaces. The chart `z ∈ ℝ¹²` persists across ticks and is advanced by the
cell's own learned linear operator `K`. The node stalk `x ∈ ℝ³²` is what the graph can touch: it is
overwritten each tick by the cell's decoded prediction `D K h + b`, then nudged by reconciliation.
The lane `ℝᵐ` on each edge is shared with exactly one neighbour and belongs to neither. One frozen
nonlinear map `encode` fuses last tick's chart with this tick's stalk into `h`; `K` advances it;
a frozen linear `D` reads it out. The only trained things in a cell are `K` and three bias vectors
(the *adapting surface*); the body's weights are shared by every cell and never move.

**An edge** is a pair of restriction maps, one per end, each `m × 32`, masked at construction and
never re-opened. Their magnitude is gauge-fixed (a Frobenius band, exactly 1 at the world's edge)
and, since ADR-0032, their spectrum is floored flat so each map carries `m` directions rather than
one. The pair is asked to learn *isometric transport*: what one cell says on the lane should be what
the other would have said, up to a rotation.

**A tick** is: predict (no cross-cell traffic), then reconcile (one Jacobi step per cell against
neighbours' last-tick lanes, at a per-cell gain `γ/(g_v²c_v)`), then the world writes its boundary
cells last. Graph distance is literally temporal distance: seven hops is seven ticks.

**Learning** runs after the tick on detached state. The *prediction rule* descends
`½‖prediction − reconciled stalk‖²` on `(b, K)`; the *transport rule* descends the disagreement on a
lane relative to the two beliefs' magnitudes, on that cell's own map only. Locality is enforced by
construction (function transforms scoped to one cell's parameters) and by a test that perturbs one
cell and asserts no other moves.

**Action and drive.** The actuator boundary cell's commanded components are read by the world as
torques and its efference copy written back. The drive is one boundary cell at the apex holding
`1.0`, on eight one-dimensional lanes: valence without specification, since the render already
says what is wanted.

**The graph** is the dome: 256 vision patches (a 64×64 render tiled 4×4) tapering 8×8 → 4×4 →
core levels of 16, 14, 12, 10 → an apex of 8, with a parallel somatomotor column of 6 and 4 cells
carrying proprioception, touch and the actuator into the same core. 414 cells, 682 edges, 150 of
them predicting. The team calls the shape a biological prior and explicitly abandonable.

## The theory underneath, as I reconstruct it

Five commitments, and my reading of what each is for.

1. **Small rigid predictors, narrower inside than out.** A cell that could store its input would not
   have to model it. Compression is forced by shape, not by a penalty.
2. **A cellular sheaf, not an atlas.** The project's own derivation (motivating image, corrected on
   #411) is that compression must be *free abelian*: what arrives at a cell is determined by which
   generators travelled, never by the route. Cells hold different subsets of the generators; the
   sheaf's gluing is what lets overlapping subsets be reassembled. In my reading the sheaf is doing
   **bookkeeping for partial overlap** — who shares which coordinates with whom — more than geometry.
   The geometric claim (overlaps are locally flat, so a linear map loses nothing) is a separate bet,
   and the one I would watch most closely.
3. **Local rules only, sharing no objective.** No error signal crosses a seam. The rules are
   Hebbian-shaped: each is a gradient that factors as an outer product of an error with an input.
   That is what makes them local, and it is also what makes them collapse under low-rank input
   (below).
4. **One relaxation step per tick.** The graph is a medium that relaxes, not a solver that
   converges. The team defends this on purpose (a cognitive system is not a function solver), on
   uniformity of the clock, and on cost, and has measured that the gain of that one step is the
   largest single term in per-hop attenuation.
5. **No reward.** Behaviour is prediction error plus a standing assertion plus a human's hands.

Two things the team claims are without precedent, and I agree they are: getting timescale
separation from *persistence alone* (no clocks, no per-unit rates; a slow cell is one whose content
survives), and wanting the operators' spectra *wide* where the field works to make them narrow.

Above the mechanism sits an image the team wrote down deliberately and marks as motivation rather
than specification: a dissipative chamber in which perturbations enter at a sensory wall, echoes
fade, and what survives is the *structure the ripples assemble* — stable regions of a vector field
that act as channels for later ripples. "Persistence is sustained, not stored." A cell is a filter
allowed to forget; the field is what remembers.

## What is claimed, what is built, what is measured

| | claimed | built | measured |
|---|---|---|---|
| Sheaf, cells, lanes, masks, two-phase tick | yes | yes | tick semantics held by tests |
| Two local rules, locality enforced | yes | yes | perturbation test, kill-tested |
| Drive as motor edge at the apex | yes | yes | reaches the apex (#183); command sensitivity to it 0 |
| Action leaving through the actuator | yes | yes | constant to 10⁻⁵ untrained; arm at its stops from tick 5000 (#120) |
| Retention in `K`'s spectrum, a gradient by depth | yes | mechanism yes, gradient never placed (#276) | τ 2.9–10.3 ticks, no gradient; 7–28 of 150 clear their own loop (#235) |
| Rim-to-core mutual influence | the destination | — | conduction ratio reads 0 both ways; unpinned by #474, unread since |
| Channels (learned aligned subspaces) | yes | yes | real, rank one end to end (#497) |
| Persistent structure in the field | the image's centre | nothing assembles it | coupling contributes no non-normality (#375, twice) |
| Language domain, interlocutor, wedge | specified | rig and builder written | nothing run, no lower bound (#331) |
| Acceptance demo | pre-registered | harness open (#99) | fails today on the conduction clause, by design (#241) |

## How I would explain the current failure to a newcomer

Start with an analogy the team already reached for. An adult can sit in a dark quiet room for hours
because a lifetime of structure inside keeps generating things to think; an infant cannot, and left
in one it does not develop. Patchworks at the cold start is the infant: it has no internal structure
yet, and the only thing that can put structure into it is variation arriving from the world.

Now the loop. Untrained, the graph's command to the arm is a world-independent constant (measured
identical across two worlds to four decimals). A constant torque drives every joint into its stop.
A stopped arm makes proprioception, touch and efference constant. The six cells that read those
boundary cells now see the same evidence every tick. Each learning step on such a cell is the same
rank-one update as the last, so the operator accumulates one direction, and the band that forbids
amplification converts that growth into the shrinking of every other mode: the cell forgets
everything except one direction, which is the direction it was already being told. The apex is in
the same state from the first tick by design: its drive is `1.0` forever. Dead somatomotor cells
output a constant to the actuator; the constant keeps the arm at its stops. The loop closes on
itself. The world is static because the agent made it static, and every instrument that asks how
well the rim reaches the core is measuring a graph driven by a world with one direction of
variation in it.

Everything downstream follows. The maps learn one direction because one direction is excited, so
the composed channel is rank one however many lanes the floor holds open. The retention gradient
never appears because nothing in a one-step prediction objective asks a cell to hold anything, and
under constant evidence it asks the cell to hold nothing. "Persistence is sustained" has nothing
to sustain. The drive answers the incentive half of the dark-room problem (it makes inaction
uncomfortable, and disagreement stays high) and not the exploration half (a constant force on a
saturating system is a fixed point at the stops, which is what was observed).

The one-line version: **the record has been measuring transmission through a graph that nothing
varies, and the graph is collapsing for lack of variation.** The remedy is upstream of every
constraint that has been added: supply variation, then measure.

## Where my picture probably departs from yours

Stated so they can be diffed, not to persuade.

1. **The drive is pressure, not forcing.** The record uses "drive" for a constant assertion and
   "driven field" for a forced dissipative medium, and I think the second word has been quietly
   lending the first its content. A constant scalar supplies zero variation.
2. **The linear map I would worry about is the exchange, not `K`.** The image says nodes are linear
   and the exchange between them cannot be; the build has linear exchange (ADR-0004) and a frozen
   nonlinearity inside the node. Open problem #333 says this; proposal #319 answers it; both are
   parked.
3. **The levels are a cost the diagnostic phase should not pay.** Seven hops was chosen to make a
   measurement legible. The "reducing depth" closure was argued under the amplitude bar that
   ADR-0026 retired; under the time bar depth sets each cell's world loop and so the floor it must
   clear. A two-level graph would answer the mechanism questions faster and cheaper.
4. **The epistemics has outrun the mechanics.** Four redefinitions of the bar in two weeks,
   verdicts flipping on probe, operator, horizon, precision, divisor, population and pooling, ~50
   "write the ruling" tickets, and zero interventions run. The register machinery is admirable and
   is now the largest consumer of the map's capacity.
5. **The sheaf is bookkeeping here, and that is fine.** I do not read the holonomy line as
   load-bearing; the team says the same. Where I would push harder is on what "free abelian" buys
   when the generators are chosen by a frozen random body rather than found.
6. **"Persistence is sustained, not stored" is currently a rescue.** It was adopted to reverse the
   sign of three shortfalls (a contracting median cell, a memory of one, a degenerate spectrum),
   before the re-supply half of the balance existed anywhere in the build. I would hold it as an
   intention until something is supplied and something is seen to be sustained.
7. **The learning rules are the collapse.** Not the band, not the floor, not the taper. Outer
   products under low-rank input concentrate; that is textbook, and it is what #477's four
   statistics say when read together. The companion note derives it and says what would refute it.

## What would change my mind

A dead cell whose operator's row space is unrelated to its mean evidence direction (the note's first
prediction failing). An exogenous-arm run (#496) reading `g ≈ 0`. Or a 100k run in which the arm
keeps moving under its own command after variation is withdrawn — at which point most of this
document becomes history and the interesting questions begin.
