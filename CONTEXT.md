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
space**, whose basis its restriction maps fix. Distinct from the cell's private internal state,
which reconciliation never touches.
_Avoid_: node state, node embedding, activation

**Edge stalk**:
The space shared by two adjacent cells, carrying a belief about a latent variable both are
modelling in common. It carries belief only; error is never a channel in it.
_Avoid_: message, edge feature, edge embedding

**Restriction map**:
The map from a cell's node stalk into one incident edge stalk. Performs transport and change
of basis only; all inference happens inside the cell.
_Avoid_: projection, encoder, transport map

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
How slowly a cell's content changes — set by how much private structure it holds and by the decay
rate of the shared body in the activation region its biases select. A property measured from
outside, never an input to any computation and never a criterion anything selects on.
_Avoid_: clock rate, update rate, level, tier, frequency

**Fold margin**:
How far a cell sits from the nearest boundary of the activation region it occupies in the shared
body. It bounds how much its operating point may be shifted before it lands in a region with a
different decay rate — and therefore a different effective timescale.
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
one, it is not fed by reconciliation alone. Its node stalk is **world-shaped**: it has whatever
dimension the world gives it, not `n`. Its edge stalks are ordinary. Three kinds: sensory, actuator,
and drive — the first two at the sensorimotor rim, the third at the core.
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
human sets; curiosity and any later appetite are others.
_Avoid_: reward, goal (reserve for the human-set drive), clamp, objective, utility

**Drive boundary cell**:
The boundary cell a drive is written into: it holds the assertion and nothing else, runs no body, and
is read by nothing. Attached at the **core**, not the sensorimotor rim. One cell is one drive. Its
stalk is near-scalar — it carries **valence, not specification**, because the render already says what
is wanted.
_Avoid_: goal cell, reward node, limbic cell, clamp site

**Drive edge**:
An edge from a drive boundary cell to a core cell. A **motor edge** by the only test that sorts edges
— disagreement on it is cleared by the world moving — differing from the actuator's only in being far
from the rim, which is what makes it abstract action rather than a torque.
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
step on disagreement, composed in the same step with the sparsity pressure. Trains transport —
the basis a neighbour's features become comparable in — never inference. Never reads a
neighbour's raw node stalk.
_Avoid_: restriction rule, map rule

**Cell contract**:
What is uniform across every cell: its interface and the algorithm it runs. Capacity and
schedule may vary per cell; the contract may not. A relay cell is the degenerate instance —
a cell whose inference is the identity.
_Avoid_: cell type, node class
