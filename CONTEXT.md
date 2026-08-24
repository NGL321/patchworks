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
problem, which is taken to be locally Euclidean and of the chart's dimension.
_Avoid_: latent, internal representation, hidden state, embedding

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
not learned — no term in the transport rule's objective identifies it, and left free it drifts
toward zero, where the sheaf couples nothing.
_Avoid_: projection, encoder, transport map

**Scale gauge**:
The construction-time bound on a restriction map's Frobenius norm: a band `[1/ρ, ρ]` for
interior maps, exactly 1 for boundary-cell maps, restored by projection after each transport
step. Excludes the collapsed sheaf `F = 0` without constraining a map's basis or rank.
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
prediction error, the cell-owned counterpart that trains the bias rule.
_Avoid_: error, residual, loss, surprise

**Prediction error**:
The difference between what a cell's `decode` predicted last tick and the node stalk it reads
in as evidence this tick — a quantity already shaped by whatever reconciliation did to that
stalk in between. Cell-owned and temporal, where disagreement is edge-owned and spatial. Trains
the bias rule.
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
Disagreement floor that is a function of *parameter drift* — a predicting cell whose bias rule
receives ambiguous, sign-flipping prediction error oscillates between activation regions rather than
converging, so its outgoing prediction never stabilises. Bounded by the same construction that bounds
reconciliation (`γ`); `decode` still emits every tick, so it degrades a neighbour's evidence but never
blocks it. Confined to mid-depth predicting cells — boundary cells run no body and are exempt by
construction, deep cells are insulated by `H⁰`.
_Avoid_: instability, confidence, null predictor (describes the cell's state, not the floor itself)

**Compression**:
The lossy, nonlinear, cell-private map from node stalk into chart, performed inside `encode`.
The counterpart to restriction, which is lossy, linear and shared. Naming both by what they
are is why "projection" is not vocabulary here.
_Avoid_: projection, encoding (bare), bottleneck, dimensionality reduction

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
How far a cell sits from the nearest boundary of the activation region it occupies in the shared
body. Three jobs: it bounds how much the cell's operating point may be shifted before it lands in a
region with a different decay rate; it is the construction-time proxy for region dwell — a cell
with a small margin has no well-defined effective timescale at all — and, third, it is what makes an
expansive region dangerous rather than a harmless transient. Falls as the body gets wider, and is
read from the narrowest map on the chart's round trip; that trade is global, paid once in the body's
widths, and inside a fixed body a cell's margin is uncorrelated with its decay rate.
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
The cell-local mechanism that updates a cell's adapting surface each tick. Splits into the bias
rule and the transport rule, sharing no objective — see ADR-0008.
_Avoid_: learning rule (bare), the rule

**Bias rule**:
The half of the local learning rule that updates a cell's biases: a local gradient step through
the cell's own frozen forward path, on its prediction error. The predictive-coding element;
trains inference, the body's operating point, never transport.
_Avoid_: predictive rule, inference rule

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
The tier, not a mechanism: anything controlling what the chart carries across `step`. Two members,
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
