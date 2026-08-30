# Patchworks

An embodied graph architecture in which many small predictors, each working in its own
metric space, are reconciled into a coherent model of a world none of them sees whole.

This glossary defines Patchworks' own vocabulary. It deliberately does not inherit terms
from the sibling project `NGL321/laminar`.

## Language

**Cell**:
One node of the graph: a predictor that experiences the world only through the features it
is given and whose job is to advance them one step in time.
_Avoid_: neuron, unit, agent, module

**Chart**:
The low-dimensional coordinates a cell computes in — the compressed feature set it derives
from its node stalk and advances in time. Its dimension is fixed by construction and below
that of the node stalk; its content is learned and need not correlate with any exposed
feature. The word is used in its strict sense: coordinates on the cell's own piece of the
problem, which is taken to be locally Euclidean and of the chart's dimension. The chart
**persists** across ticks and the cell's own operator advances it; it is not recomputed from the
node stalk each tick, and it is not a lift — the design has no lifted space and no second
dimension besides the piece's (ADR-0014).
_Avoid_: latent, internal representation, hidden state, embedding, lift

**Piece**:
The part of the problem one cell owns — locally Euclidean, and of the chart's dimension. What
makes a cell's chart a chart. The pieces, not the world, are what Patchworks claims to be
manifolds; the sheaf glues them without their union being one.
_Avoid_: patch (reserve for the sensory tiling), subproblem, region, manifold (bare)

**Node stalk**:
A cell's public face — the feature vector it exposes to the graph, and **the cell's own metric
space** in both basis and scale: its restriction maps fix the basis, and its scale is its own,
bounded by construction rather than pinned to its neighbours'. Distinct from the cell's private
internal state, which reconciliation never touches.
_Avoid_: node state, node embedding, activation

**Edge stalk**:
The space shared by two adjacent cells, carrying a belief about a latent variable both are
modelling in common. It carries belief only; error is never a channel in it.
_Avoid_: message, edge feature, edge embedding

**Restriction map**:
The map from a cell's node stalk into one incident edge stalk. Performs transport and change
of basis only; all inference happens inside the cell. Its overall magnitude is **gauge-fixed**,
not learned — no term in the transport rule's objective identifies it, and left free an edge's
joint scale grows without bound, until the maps stop moving.
_Avoid_: projection, encoder, transport map

**Scale gauge**:
The construction-time bound on a restriction map's Frobenius norm: a band `[1/ρ, ρ]` for
interior maps, exactly 1 for boundary-cell maps, restored by projection after each transport
step. The upper face is the working constraint and binds continuously; the lower is a guardrail.
What the band leaves free is an edge's scale ratio, not each map's magnitude.
_Avoid_: normalisation, regularisation, weight decay, orthogonality constraint

**Effective rank**:
How many directions a restriction map is actually transmitting — the participation ratio of its
singular values, read on the diagnostic cadence. Paired with per-edge disagreement energy it
separates parameter collapse from a lag floor draining; neither reading separates them alone.
_Avoid_: rank (bare), sparsity, bandwidth

**Disagreement**:
The difference, measured in an edge stalk, between the two adjacent cells' restrictions of
their node stalks. Patchworks' only edge-level error signal: derived, never carried, and never
fully cleared. Collected across every edge it is the sheaf's coboundary, and its squared sum is
the Dirichlet energy of the sheaf Laplacian. Trains the transport rule; kept distinct from
prediction error, the cell-owned counterpart that trains the prediction rule.
_Avoid_: error, residual, loss, surprise

**Prediction error**:
The difference between what a cell's `decode` predicted last tick and the node stalk it reads
in as evidence this tick — a quantity already shaped by whatever reconciliation did to that
stalk in between. Cell-owned and temporal, where disagreement is edge-owned and spatial. Trains
the prediction rule.
_Avoid_: error (bare), residual, disagreement, loss

**Disagreement floor**:
The part of an edge's disagreement that learning cannot remove. Umbrella over three kinds, static,
lag, and settling. Nothing in the architecture represents it; the learning rule is constrained never
to target zero residual. What is left over is model error, which is reducible and needs no name of
its own — its role is to be what the floors are distinguished from.
_Avoid_: irreducible error, noise floor, bias, residual (bare)

**Static floor**:
Disagreement floor that is a function of *configuration* — curvature the linear restriction map
cannot follow, mask or learned rank deficiency, aleatoric noise. Present at rest. What distinguishes
it from a lag floor is that holding the world still does not drain it.
_Avoid_: curvature error (that is one cause of it, not the term), structural error

**Lag floor**:
Disagreement floor that is a function of *motion* — two adjacent cells whose contents live at
different effective timescales, so the slow end is behind. Velocity-shaped and sign-flipping; drains
to zero when the world is held still. The price of timescale separation, and the mechanism working
rather than failing.
_Avoid_: staleness (reserve for unit delay), timescale error, phase error

**Settling floor**:
Disagreement floor that is a function of *parameter drift* — a predicting cell whose prediction rule
receives ambiguous, sign-flipping prediction error oscillates between activation regions rather than
converging, so its outgoing prediction never stabilises. Bounded by the same construction that bounds
reconciliation (`γ`); `decode` still emits every tick, so it degrades a neighbour's evidence but never
blocks it. Confined to mid-depth predicting cells — boundary cells run no body and are exempt by
construction, deep cells are insulated by `H⁰`.
_Avoid_: instability, confidence, null predictor (describes the cell's state, not the floor itself)

**Compression**:
The lossy, nonlinear, cell-private map from node stalk into chart, performed inside `encode`.
The counterpart to restriction, which is lossy, linear and shared. Naming both by what they
are is why "projection" is not vocabulary here. **"Dictionary" is on the avoid list for a reason
worth keeping:** every published Koopman lift is *larger* than its state, where `encode` compresses,
`k` under `n` — so calling it the EDMD dictionary reproduces exactly the conflation between the
chart's dimension and a lift's that the vocabulary is being kept clean of.
_Avoid_: projection, encoding (bare), bottleneck, dimensionality reduction, dictionary

**Local flatness**:
The claim a linear restriction map rests on: the latent structure two adjacent cells model in common
is locally Euclidean at the scale of their overlap, so transport between their stalks loses nothing a
first-order map could have kept. A claim about the **geometry of the overlap**, and the oldest of the
three linearity claims (ADR-0004). Says nothing whatever about how anything moves in time. Failure
surfaces as a static floor on that edge.
_Avoid_: linear (bare), linearity (unqualified), local linearity, flat (bare)

**Chart linearity**:
The claim `K` rests on: a cell's piece evolves linearly in the coordinates its chart provides, so one
learned linear operator is an honest model of its motion. A claim about **time-evolution**, and
independent of local flatness in both directions — a piece may evolve linearly in its chart while the
overlap it shares with a neighbour stays curved, and an overlap may be flat while the motion across it
is not. Failure surfaces as prediction error no `K` can remove.
_Avoid_: linear (bare), local flatness (that is the geometric claim), Koopman linearity, global
linearity

**Readout gauge**:
The claim a frozen linear `decode` rests on: a cell's node stalk is a linear function of its chart, so
a fixed `D` is an honest readout. A claim about **observability**, distinct from the other two in both
directions — a piece may evolve linearly in its chart while its stalk depends nonlinearly on that
chart, and the reverse. The sharpest falsification of the three, because the subspace is **shared**:
failure surfaces as prediction error confined to `colspan(D)`, the same fixed subspace of every cell's
node stalk, rather than as anything per-edge.
_Avoid_: linear (bare), local flatness, linear decoder, gauge (bare — reserve for the scale gauge)

**Operator band**:
The construction-time bound on a cell operator's **spectral** norm: `σ_max(K) ∈ [1/ρ_K, 1]`, restored
by projection after each prediction step. One global band, not one per level. The upper face is
exactly 1 because what it forbids is *amplification*, and a cell sitting at 1 is non-expansive rather
than divergent — so it permits `ρ(K) = 1` and is **not** the claim `|λ| < 1`. Spectral rather than
Frobenius, and deliberately unlike the scale gauge: rank-deficiency is wanted on a restriction map and
is the failure mode on the body.
_Avoid_: spectral radius bound, stability constraint, scale gauge (that is the sheaf's), spectral
normalisation (bare)

**Sheaf cohomology**:
The cohomology of the cellular sheaf on the graph — coefficients are stalks, the differential
is disagreement. `H⁰` is the configurations no edge disagrees on, which in Patchworks are
exactly the features private to a cell's sub-problem. **Not** the cohomology of Baudot &
Bennequin's information theory, which is taken over a poset of partitions and has no graph in
it; the two share a letter and nothing else, and must never be conflated.
_Avoid_: cohomology (bare), information cohomology (for this object), topological invariant

**Private features**:
The node stalk directions a cell exposes on no edge — masked out everywhere, and therefore exactly
the sheaf's `H⁰`. Reconciliation cannot move them, which is what makes them the home of a cell's
slowly-varying state as well as of its own sub-problem.
_Avoid_: hidden features, internal state (reserve that for the chart), latent

**Effective timescale**:
How slowly a cell's content changes — set by how much private structure it holds and by the
*distribution* of regional spectra its biases select, not by any one of them. A mean rate rather
than a fixed one, and well defined only while the cell's region dwell is long against it. A property
measured from outside, never an input to any computation and **never a criterion anything selects on
at runtime** — though it is exactly the criterion the body's construction selects biases against,
once, before the graph runs. Under the distributional reading there is no constant for a running
cell to read even in principle.
_Avoid_: clock rate, update rate, level, tier, frequency

**Activation region**:
One of the finitely many convex regions of chart values on which `encode` is exactly affine. Its
activation is piecewise-linear, which is what makes these regions exist at all; a **fold** is a
boundary between two of them, and a cell crosses one when its chart moves far enough. **Retained as a
description of `encode` and retired as a mechanism** (#138): `encode` is the body's only nonlinearity,
so this is the one map the vocabulary still has a referent in, and nothing new is built on it — the
folds no longer bound `γ` and no longer carry timescale, which lives in `K`'s spectrum.
_Avoid_: linear region, cell, piece (reserve that for the sub-problem), basin

**Timescale band**:
The range of effective timescales a level of the taper is built to hold. Cells are placed in one by
construction — bias vectors are drawn, measured, and kept if they land in the band — and adjacent
levels' bands **overlap**, so the taper's gradient is continuous and separates levels only as
distributions. A band is where a cell started, not a property it has: nothing stores it, nothing
re-selects, and the biases drift off it as they adapt.
_Avoid_: tier, layer rate, timescale level, clock band

**Realised contraction rate**:
`λ`, the rate at which a cell's private content actually decays along the trajectory it walks —
averaged over every activation region it visits, not read off any one of them. The stability object:
a cell is unstable when `λ ≥ 0`, which is not the same as occupying a region whose spectral radius
exceeds one. `max ρ < 1` over the regions a cell can reach is a *sufficient* condition for `λ < 0`,
cheap enough to check before training and far stronger than necessary.
_Avoid_: stability margin, spectral radius (for this object), Lyapunov exponent (unless the
long-run limit is meant literally)

**Regional spectrum**:
The spectrum of the local Jacobian of whichever activation region of the shared body a cell occupies
on a given tick. A per-tick quantity, re-drawn whenever the cell's chart carries it across a fold —
never a cell attribute. What the biases set is the distribution these are drawn from, which is the
cell's effective timescale.
_Avoid_: the cell's spectrum, its Jacobian, decay rate (unqualified)

**Region dwell**:
How long a cell stays in one activation region of the shared body before its chart carries it across
a fold. The timescale mechanism holds only where dwell is long against the `τ` that region implies;
where dwell is short, a cell still decays at some average rate, but by averaging over unrelated
regions rather than by the mechanism the spec claims. Bounded at construction by the fold margin,
measured at runtime on a driven trajectory.
_Avoid_: region residence, switching rate, region stability

**Fold margin**:
How far a cell sits from the nearest boundary of the activation region it occupies in `encode`. It
had three jobs and keeps one and a half. **Dead:** bounding `γ`, which it never did in practice.
**Falsified premise:** carrying timescale, which now lives in `K`'s spectrum. **Surviving:** it is
still the construction-time proxy for region dwell, and still what makes an expansive region
dangerous rather than a harmless transient. Falls as the body gets wider; read from `encode` alone
since it is the only map with folds, which is why the measured cap rose when `step` was linearised.
Inside a fixed body a cell's margin is uncorrelated with its decay rate.
_Avoid_: slack, headroom, distance to boundary

**Inference phase**:
The half of a tick in which every cell locally advances its own chart and decodes a
prediction, using only its own persisted chart and the node stalk the last message-passing
phase left behind. No cross-cell exchange happens here.
_Avoid_: prediction phase, phase one

**Message-passing phase**:
The other half of a tick: every cell simultaneously exchanges restricted beliefs across its
edges and runs one round of reconciliation. Exactly one simultaneous step per tick — not an
iterative solve run to convergence.
_Avoid_: reconciliation phase, phase two, communication phase

**Reconciliation**:
The disagreement-reducing computation a cell runs during the message-passing phase: a single
local descent step against a neighbour's restricted belief. Penalised rather than enforced —
cells are pulled toward agreement, never projected onto it. Its step size is per cell, normalised so
that degree does not become an accidental rate difference.
_Avoid_: consensus, synchronisation, message passing (bare — reserve that word for the phase)

**Cell-local**:
The sense of "local" in which a cell's learning uses only quantities that cell can see — no
error signal propagated across the graph.
_Avoid_: local (unqualified)

**Graph-local**:
The sense of "local" in which a cell exchanges only with its adjacent cells — no global
aggregation step, no all-to-all read.
_Avoid_: local (unqualified)

**Relay cell**:
A cell whose inference is the identity: it holds stalks and restriction maps but performs no
prediction, existing to provide a shared metric space for distant cells. Costs one tick of
latency like any other cell.
_Avoid_: hub, router, passthrough node

**Boundary cell**:
A cell whose node stalk something outside the sheaf writes or reads directly — the seam between
the graph and everything that is not the graph. Like a relay cell it performs no inference; unlike
one, it is not fed by reconciliation alone. It is **exempt from `n`**: its node stalk has whatever
dimension the thing writing or reading it gives it — the world's shape at the sensorimotor rim, and at
the internal rim whatever the faculty asserts. Its edge stalks are ordinary, and the outside write
lands *after* the message-passing phase, so it is always the last word. Three kinds: sensory,
actuator, and drive — the first two at the sensorimotor rim, the third at the internal rim.
_Avoid_: input node, output node, sensor node, IO cell

**Predicting cell**:
A cell that runs the cell body — every cell that is not a boundary cell or a relay cell. The
population `n` and `k` describe, the population executed batched, and the population the sheaf's
construction diagnostics are computed over.
_Avoid_: interior cell, hidden cell, ordinary cell

**Dome**:
The graph's shape: a taper from the two-dimensional sensorimotor boundary sheet, through
successively coarser levels, to a small deep core. Replaces the disk, whose one-dimensional rim
could not carry the two-dimensional sensory tiling. Held as an intuition-derived shape and
explicitly abandonable.
_Avoid_: pyramid, cone, hierarchy, layers

**Level**:
One stage of the dome's taper, indexed by hop distance from the sensorimotor boundary. Not a tier
a cell belongs to by attribute — a level is a set of cells at a distance, and abstraction is that
distance.
_Avoid_: layer, tier, stage, rank

**Core**:
The dome's deep levels: small, not a lattice, where the modalities first share a cell and where
guaranteed private dimension is largest. The shared space distant regions communicate through, and
the rim at which internal faculties attach.
_Avoid_: apex (reserve for the deepest level alone), hub, centre, bottleneck

**Somatomotor column**:
The parallel taper carrying proprioception, touch, and the actuator, separate from the vision
lattice until the core. What makes the reflex loop three ticks and cross-modal binding a function
of depth.
_Avoid_: motor pathway, proprioceptive stream, sensorimotor branch

**Construction layout**:
The rule that generates the structural mask — cells indexed by level and lattice position. An
index, never a metric embedding: no cell has a coordinate, and nothing about it is the geometry of
the *chart*. It has no runtime role.
_Avoid_: embedding, graph layout, coordinates, positions

**Sensory edge**:
An edge whose far endpoint is a boundary cell the world writes. Disagreement on it is cleared by
the cell changing its belief.
_Avoid_: input edge, afferent edge, observation channel

**Motor edge**:
An edge whose far endpoint is a boundary cell the world reads. Disagreement on it is cleared by the
world moving — this, and not any property of the cell, is what makes a prediction an action.
_Avoid_: output edge, efferent edge, action channel, command

**Drive**:
A standing assertion, written from outside the sheaf, that some state of affairs holds — so that while
it does not, the graph carries disagreement it can only clear by acting. What keeps an unsolved task
uncomfortable, and therefore the answer to the dark room problem. A goal is the particular drive a
human sets; curiosity and any later appetite are others. Not the Hullian drive: no deficit state, no
reduction, no satiation — the disagreement falls away because the world agrees, not because anything is
discharged.
_Avoid_: reward, goal (reserve for the human-set drive), clamp, objective, utility, drive reduction,
tension

**Drive boundary cell**:
The boundary cell a drive is written into: it holds the assertion and nothing else, runs no body, and
is read by nothing. Attached at the **apex**, the deepest core level, not the sensorimotor rim. One
cell is one drive. Its stalk is **scalar** — it carries **valence, not specification**, because the
render already says what is wanted — and how hard it pulls is set by how many cells it attaches to,
never by widening the channel.
_Avoid_: goal cell, reward node, limbic cell, clamp site

**Drive edge**:
An edge from a drive boundary cell to an apex cell, of mask width 1 to match the drive's scalar stalk.
A **motor edge** by the only test that sorts edges — disagreement on it is cleared by the world moving
— differing from the actuator's only in being far from the rim, which is what makes it abstract action
rather than a torque.
_Avoid_: goal edge, reward channel, top-down edge

**Dark room problem**:
That a pure prediction-error minimiser is best off predicting a world it does not disturb: an unsolved
task, watched from a standstill, is a low-error state. The reason drives exist. Named as a problem the
architecture answers by construction, never one it hopes to outgrow.
_Avoid_: exploration problem, motivation problem, the boredom problem

**Dynamics exploration**:
Acting on the world to learn what it does — a puck's mass, its friction, the outcome of a contact,
the orientation of the **eccentric puck** — as opposed to
moving in order to see where things are. The sense in which the sandbox
must be explored to be modelled: its whole arena is visible every frame, so position is given away
and only dynamics has to be earned. A model of where things are in one arena is memorisation of that
arena.
_Avoid_: spatial exploration, search, epistemic action, information gathering

**Efference copy**:
What an actuator boundary cell writes back after the world has read it: the command as actually
applied, post-clip and post-saturation. Makes a motor edge carry ordinary disagreement rather than
being the one edge with none.
_Avoid_: feedback, acknowledgement, actual torque

**Sensorimotor rim**:
The region of the graph where the world touches it — the sensory and motor boundary cells together.
Abstraction is hop distance from this rim; internal faculties attach at other rims without being
concrete.
_Avoid_: input layer, periphery, level zero

**Internal rim**:
Where something outside the sheaf that is *not the world* attaches — the drive boundary cell today, a
limbic-analogue appetite or a hippocampal-analogue memory later. Physically the **apex**, so a faculty
is abstract by attachment point. What attaches here is an **internal faculty**, and it is barred from
two things: holding its own model of the world, and reaching the world by any route but the graph.
Attention is *not* one of these — its likely mechanism is in-graph.
_Avoid_: abstract rim, top-down interface, internal boundary

**Cell body**:
The machinery a cell runs: one set of weights, shared by every cell and frozen. Distinct from the
cell, which is that shared body plus the cell's own adapting surface and its persisted chart.
_Avoid_: the MLP, the network, cell weights

**Adapting surface**:
The parameters that carry a cell's ongoing adaptation — its biases and its restriction maps. The
surface continual learning governs, and the only thing in a cell that ever changes by learning.
_Avoid_: trainable parameters, the readout, fine-tuning surface

**Local learning rule**:
The cell-local mechanism that updates a cell's adapting surface each tick. Splits into the prediction
rule and the transport rule, sharing no objective — see ADR-0008.
_Avoid_: learning rule (bare), the rule

**Prediction rule**:
The half of the local learning rule that updates a cell's own inference parameters — its biases
**and** its operator `K`: a local gradient step through the cell's own forward path, on its
prediction error, followed by projection back into the operator band. The predictive-coding element;
trains inference — the cell's operating point and its chart's dynamics — never transport. Named for
its signal, which is what ADR-0008 splits on; it was the **bias rule** until the Koopman conversion
widened what it trains.
_Avoid_: bias rule (superseded), predictive rule, inference rule

**Transport rule**:
The half of the local learning rule that updates a cell's restriction maps: a local gradient
step on disagreement **relative to the restricted beliefs' own current magnitudes**, composed in
the same step with the sparsity pressure and followed by projection back into the maps' gauge.
Trains transport — the basis a neighbour's features become comparable in — never inference.
Never reads a neighbour's raw node stalk.
_Avoid_: restriction rule, map rule

**Change gate**:
A specified-but-unbuilt amplifier of timescale differentiation: an interior edge holds its
previous transmitted value rather than broadcasting a fresh one when the sender's restricted
belief has not moved, so transmission rate tracks content rate. Outbound only, adds no state,
and exempt on every boundary edge. Distinct from the reconciliation gain (stability of one
descent step), from persistence (commitment), from attention (inbound, differently driven), and
from recurrent-state gating (inside the body, not on the edge).
_Avoid_: gate (unqualified), confidence gate, attention

**Recurrent-state gating**:
The tier, not a mechanism: anything controlling what the chart carries across `K`. Two members,
neither built. The **protected channel** is an ungated subspace the chart passes with unit gain — a
construction choice about the shared body, costing no parameters and breaking no freeze, and the one
to reach for first. Behind it sits a **learned gate** on `encode`'s fusion, reached for only if
deliberate clearing proves necessary, and priced at a third parameter group and therefore a third
learning rule. Distinct from the change gate by tier: inside the body's recurrence, not on the edge.
_Avoid_: gate (unqualified), change gate, LSTM hatch, edge-stalk pass-through

**Cell contract**:
What is uniform across every cell: its interface and the algorithm it runs. Capacity and
schedule may vary per cell; the contract may not. A relay cell is the degenerate instance —
a cell whose inference is the identity.
_Avoid_: cell type, node class

**Route**:
A path the world takes when the agent acts, visible only in hindsight. **Not an object in the
architecture**: no cell holds one, nothing compares two, and there is no slot a route could
occupy. What resembles a choice between routes is several cells disagreeing about the next step.
_Avoid_: plan, policy, trajectory (as something stored), rollout

**Route selection**:
The resolution of that disagreement by the world moving — the only thing entitled to clear a
motor edge. Closed-loop and one tick wide, never a comparison and never in advance. A blended
prediction is not a decision but an unstable state the next tick's evidence destroys.
_Avoid_: arbitration, decision, action selection, planning

**Commitment**:
The persistence of a route already being taken, and the reason selection is not re-litigated
every tick. Identical to `H⁰` insulation: content in a cell's private features is exactly
invariant under reconciliation, so a contrary neighbour belief cannot dislodge it and only
prediction error can. Rises with abstraction, since private dimension does. Distinct from
persistence (the same mechanism serving timescale) by what it is being used for, not by what
it is.
_Avoid_: hysteresis, latching, locking in, decision commitment

**Arm**:
The manipulator in the sandbox: one 3-link planar limb, and the agent's only body. Always the
robot, never an experimental condition — a falsification sweep has **conditions**, not arms. The
word stays singular for the PoC; a second arm is a PoC-2 object.
_Avoid_: arm (as a branch of an experiment), limb, effector, manipulator (as a separate term)

**Friction field**:
The fixed, smooth spatial variation in the sandbox's table friction: a puck's frictionloss scaled by
a function of where the puck is. A property of the world, not a random draw — so the same push at
two places gives two outcomes while snapshot and restore stay bit-exact. The reason repeated
identical pushes in this world are not identical.
_Avoid_: friction noise, roughness map, stochastic friction, domain randomisation

**Eccentric puck**:
Puck 1, whose centre of mass is deliberately offset from its geometric centre. The one object in the
sandbox whose orientation appears in its own equations of motion, which is what makes rotation a
hidden variable worth inferring rather than a decoupled integrator. The eccentricity is invisible in
the render by construction.
_Avoid_: asymmetric puck, weighted puck, the odd puck, off-balance puck

**Precedence depth**:
Of a task: the length of the longest chain of sub-goals that must be reached *in order*, where a
later one is unavailable until an earlier one is done. The measure of how much compositional
structure a world actually contains, as distinct from how wide its observation is or how many
numbers its state holds. Distinct from **Commitment**, which is the architecture's mechanism for
holding to a route; precedence depth is a property of the world, and nothing in the graph
represents it.
_Avoid_: task complexity, horizon, difficulty, planning depth, commitment

**Acceptance demo**:
The single live interaction the proof of concept is judged by: a human disturbs the agent mid-task
and the agent recovers at the appropriate level of its hierarchy. One named protocol with fixed
pass and fail conditions, settled before the run — not a category of demonstration. **Evaluation**
is not a broader thing that contains it: the two are coextensive. The demo's pre-registered readouts
are the whole of what "evaluation" names here, and nothing aggregates a score over runs.
_Avoid_: the demo (bare), benchmark, test run, showcase

**Onset latency**:
Ticks from a disturbance to the first corrective torque. The demo's temporal measure, chosen
because it is a property of the graph — how far a correction had to travel before acting — where a
settling or decay time would be a property of the body's mechanics. Reported per event; a
difference in onset is what "recovered at a different level" means in time rather than in hops.
_Avoid_: reaction time, settling time, response time, recovery time, latency (bare)

**Trial**:
One measured disturbance: a restore to a snapshot tick, one event fired, one onset latency recorded.
The unit the demo's latency ordering is computed over, and the only unit there is — a world with no
episodes has no other. A trial is *valid* or discarded on pre-registered grounds, never scored; goal
satisfaction gates admission and contributes no number.
_Avoid_: episode, run (bare), rollout, attempt, sample

**Restore**:
Rewinding the entire state — world, clock, and the agent's adapting surface — to a snapshot. Kept
sharply distinct from `reset()`: a reset is in-band and the agent lives through it, while a restore
is invisible from inside, since no cell survives it to notice. An experimenter's tool that appears
nowhere in the env's contract, which is what lets evaluation have a defined start without weakening
the reset-free commitment.
_Avoid_: reset, restart, rollback, reload, checkpoint (as a verb)

**Demo surface**:
What a human sees and touches while a run is happening: two windows, the encodings drawn in them,
and the gestures bound to the hands. Display only — it reads privileged state on the same footing
as `info` and no cell reads anything it computes, so switching it off changes no trajectory. Kept
distinct from the acceptance demo, which is the protocol the surface displays.
_Avoid_: the viewer, the visualisation, the UI, the demo (bare), instrumentation
