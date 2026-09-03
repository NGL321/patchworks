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
dimension besides the piece's (ADR-0023).
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

**Incoherence count**:
The effective overlap count `c` — how many of a cell's incident restriction maps load the same input
direction. `c = deg(v)` is fully coherent maps and `c = 1` perfectly incoherent ones. A gauge constant
declared globally alongside `ρ` and held by the same projection (ADR-0010), and the term that makes the
reconciliation gain's denominator a true bound rather than a loose one; applied per cell as
`c_v = min(deg(v), max(c, ⌈deg(v)/n_v⌉))`, where the floor is a fact about stalk dimensions and not a
hedge. Measured at 2.42 at the rim and 1.75–1.98 through the core, against a practical floor of ~1.05
set by effective rank.
_Avoid_: orthogonality, decorrelation, condition number, diversity

**Channel**:
The aligned subspace a chain of restriction maps and cell operators actually carries — the directions a
perturbation survives along, as against the directions it is annihilated in. Learned by the transport
rule rather than nominated by construction, and narrow because the maps are near-rank-1. A hop is an
operator norm along it, never an isotropic average over directions (ADR-0022); reconciliation is fast
along it and under-relaxed off it.
_Avoid_: path, route, pathway, bandwidth, receptive field

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
its own — its role is to be what the floors are distinguished from. It is **not** the quantity
`02-tick-semantics.md`'s bound divides by — see *Standing offset*.
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

**Standing offset**:
The displacement reconciliation leaves on the reconciled component of a node stalk each tick — the
gain multiplied by the disagreement the cell carries. What `02-tick-semantics.md`'s bound weighs
against a cell's fold margin, because a displacement larger than the margin carries the cell into an
activation region with a different regional spectrum. The disagreement floor is one contributor and,
at construction, not the dominant one: model error dominates it, and learning removes model error —
so the offset falls through a run rather than standing at a level. Named for what it is, a shift in
the operating point, rather than for its cause.
_Avoid_: floor, disagreement floor, the floor (as in `γ × floor <` fold margin), reconciliation error

**Bottleneck ratio**:
At one edge, the arriving perturbation over what is already standing there: `[A₀ · Π hop] / floor_e`,
where the numerator is a unit-norm rim deviation carried by the cumulative gain of the edges it has
crossed and the denominator is that edge's quiescent-hold floor — static plus settling, lag excluded.
Dimensionless, and the object that made a dimensionless gain and a magnitude comparable at all
(ADR-0021). Indexed on the **edge**, never on the level: per-edge is a property of the graph, per-level
a property of the shape imposed on it.
_Avoid_: floor (bare), transmission floor, margin, signal-to-noise ratio, headroom

**Rim-to-core detectability**:
The architecture's transmission predicate: over rim-to-apex paths, the max of the min bottleneck ratio
along a path is at least 1 — *there exists a channel that carries the perturbation*, and what fails is
an edge rather than a level. Read per trial, each trial reduced to its peak ratio, reported as a
distribution over trials with the bar at the median. Stated twice, rim→apex and apex→rim, because the
two directions do not share a gain. Detectability rather than magnitude because the floor never
settles: the perturbation must be distinguishable from what stands on the edge, not clear a wall.
_Avoid_: transmission target, reachability, the ~0.37 per hop (retired), sufficiency (bare)

**Conduction ratio**:
A cell's measured retention time over the tick length of the shortest cycle through it that reaches
the rim and returns — `τ̂_c / |loop(c)|`, dimensionless because both halves are in ticks. The
numerator is the e-fold decay time of the paired counterfactual deviation restricted to the cell's
private features; the denominator is a **construction-time** quantity, computed from the mask by
enumeration and never inherited from a hop count (ADR-0026). It is `2 · d(c, rim)` — 14 at the apex of
the default dome, where it was checked rather than assumed. Scale-free in time, so it touches neither
the never-settling floor nor float32.
_Avoid_: margin (`fold margin` is a distance to a boundary; this is a ratio), retention (bare), gain,
timescale ratio

**Loop length**:
`|loop(c)|`: the tick length of the shortest cycle through a cell that reaches the sensorimotor rim
and returns — the conduction ratio's denominator, and the cell's **own round trip**, which is what a
retention time has to beat. A **construction-time** quantity: enumerated from the mask by breadth-
first sweep, `2 · d(c, rim)`, and **never inherited from a hop count** (ADR-0026). It moves with
`DomeSpec` and is recomputed rather than quoted — 14 at the apex of the default dome, where it was
checked rather than assumed. That it coincides with `2 · level` there is a fact about the current
taper's wiring and not a licence to index by level. **Computed nowhere in the tree today**; #99 owes
it.
_Avoid_: round trip (bare), cycle length, depth, hop count, `2 · level`

**Rim-core influence**:
The predicate over the conduction ratio, and **the map's operative bar**: over paths, the max of the
min conduction ratio along a path is at least 1 — the cell still holds what it sent by the time the
answer gets back. Stated twice, and the two directions **do not share a quantifier**: inbound is a
swept single-source **count**, the fraction of rim cells whose loop closes, per stratum and never
averaged across strata, with the bar that every stratum's reaching set is non-empty; outbound is a
**universal** over L1 predicting cells and the actuator boundary cell. **Necessary, not sufficient**:
it says the loop can close, not that anything distinguishable travels it. Distinct from **rim-to-core
detectability**, ADR-0021's predicate over the bottleneck ratio, which is retained as the *sufficient*
diagnostic and is no longer the bar.
_Avoid_: transmission, detectability (that is ADR-0021's), reach

**Collectively**:
Of the rim's influence on the apex: **counted, not summed**. The word was doing two jobs — *many,
counted* and *coherently summed* — and #232 measured the summed sense empty: the collective corner's
2.7x is the predicate's `max` over 256 candidate entry points rather than coherence, the incoherent
contrast reads level with it, and the missing `√K = 16x` says a rim-coherent injection sums as a
random walk downstream. The counted sense is the one the destination means, and the retired one may
not walk back in.
_Avoid_: coherent, aggregated, pooled, summed

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
The claim is made of **every** cell and admits no exception: no cell in this architecture is
autonomous — a fresh node stalk arrives each tick and `encode` fuses it with the persisting chart — so
there is no class of *driven* cells to hold a different operator, and no boundary at which one kind of
cell would give way to another ([#146](https://github.com/NGL321/patchworks/issues/146)).
_Avoid_: linear (bare), local flatness (that is the geometric claim), Koopman linearity, global
linearity, bilinear, bilinear realisation, control-affine (the last three name an exogenous input
multiplying the state; this design has none, and borrowing the word re-imports the control literature
ADR-0023 exited)

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

**Execution clock**:
How often a cell runs: **one tick, uniformly across the graph**. Every cell infers one step ahead on
the same clock, so no cell is frozen while its neighbours run, and ADR-0002's second ground is why. It
is a property of the architecture, identical at the rim and at the apex, and nothing about a cell's
depth, placement or content changes it. A cell with a longer timescale is **not** a cell that runs less
often — that reading was live in the record and is not admissible.
_Avoid_: timescale (bare, for this sense), latency (`05-timescales.md` uses it for unit delay, which is
a phase shift and the wrong ratio), clock rate, update rate, tick rate, schedule

**Retention**:
How much of what a cell held it keeps — the other half of what *timescale* used to name, and the half
that is **meant to be differentiated across the graph**. It is ADR-0005's subject and now
ADR-0028's: persistence in the private features rather than a schedule, and since #138 a per-cell
time constant living in `K`'s spectrum — **a spectrum of them per cell, not one** (#143).
**Retention constant** below is the quantity, and **Effective timescale** the measurement of it. The
graph's retention currently measures flat at about one tick on the chart's **direct** round trip,
and 2.9 to 10.3 ticks with the stalk relay included (`05-timescales.md`, *What the live read
says*) — **flat under either operator**, and a finding about this build rather than a property of
the architecture. **The differentiation is no longer placed by design**: construction assigns no
per-level `τ`, so a depth gradient in retention is learning's to produce, and *nothing guarantees
it appears* is the standing falsification.

*One clock, heterogeneous retention.* The two are separate objects and the record needs both words.
_Avoid_: timescale (bare, for this sense), memory, decay rate, persistence (reserve for ADR-0005's
mechanism), level, tier

**Retention constant**:
`λ(K)`, an eigenvalue magnitude of a cell's own learned operator, and the quantity *effective
timescale* is now read off: `τ = −1/ln|λ(K)|`. **A cell has up to `k` of them at once, one per
eigen-direction** — it can commit in some chart directions while staying reactive in others — so it
has a **spectrum** and not a rate (ADR-0028). Healthy **near 1**: `λ(K) = 0.99` is `τ ≈ 99.5`, a slow
cell, comfortably inside the operator band. **A slow eigenvalue is a memory policy, never a mode of
the piece's physics** (ADR-0023). Distinguish `λ(K)`, the operator's retention and the published
quantity, from `λ(K · J_encode)`, the realised chart retention, which is region-dependent and
per-tick.
_Avoid_: `λ` bare (**Realised contraction rate** is a log-rate with the opposite health direction —
see that entry), timescale (bare), decay rate, mode, eigenvalue of the piece, Koopman eigenvalue

**Effective timescale**:
How slowly a cell's content changes — set by how much private structure it holds (`ker δ`) and by the
**retention constants of its own `K`**, `τ = −1/ln|λ(K)|`. **Not one number:** a cell holds a
spectrum of them, so *the* effective timescale of a cell is a loose way of speaking and the direction
must be said whenever it matters. A property measured from outside, never an input to any computation
and **never a criterion anything selects on at runtime** — a discipline that has to be kept, since
`λ(K)` is a constant and readable in principle, where the retired distributional reading made the
prohibition free by leaving nothing to read. **Nothing places it**: construction assigns no per-level
`τ`, `a` is global, and the depth gradient is learning's to produce or not (#143).
_Avoid_: clock rate, update rate, level, tier, frequency, central tendency, mean rate (the retired
distributional reading), the cell's timescale (where a direction is meant)

**Activation region**:
One of the finitely many convex regions of chart values on which `encode` is exactly affine. Its
activation is piecewise-linear, which is what makes these regions exist at all; a **fold** is a
boundary between two of them, and a cell crosses one when its chart moves far enough. **Retained as a
description of `encode` and retired as a mechanism** (#138): `encode` is the body's only nonlinearity,
so this is the one map the vocabulary still has a referent in, and nothing new is built on it — the
folds no longer bound `γ` and no longer carry timescale, which lives in `K`'s spectrum.
_Avoid_: linear region, cell, piece (reserve that for the sub-problem), basin

**Timescale band**:
**Retired as a mechanism** (#143, ADR-0028) and kept only to name what the record used to do. A level
of the taper was built to hold a range of effective timescales, with bias vectors drawn, measured and
kept if they landed in the band, adjacent bands overlapping to keep the taper's gradient continuous.
**It was built and the graph came out flat** — 0.91 at the apex against 0.99 at the rim on the
direct round trip, and flat under the corrected operator too (`05-timescales.md`) — because the
biases are the adapting surface and drift off their bands with nothing re-selecting. **Construction
now places no per-level `τ` at all**; `a` is global and the gradient is learning's job. Not to be
confused with the **operator band**, which is live, global, and on `σ_max(K)`.
_Avoid_: using this for anything current, tier, layer rate, timescale level, clock band, per-level
band (ADR-0015 rejected that as a second timescale mechanism)

**Realised contraction rate**:
`λ`, the rate at which a cell's private content actually decays along the trajectory it walks —
averaged over every activation region it visits, not read off any one of them. The stability object:
a cell is unstable when `λ ≥ 0`, which is not the same as occupying a region whose spectral radius
exceeds one. `max ρ < 1` over the regions a cell can reach is a *sufficient* condition for `λ < 0`,
cheap enough to check before training and far stronger than necessary.

**`λ` bare is ambiguous and the two senses have opposite health directions.** This entry is a
**log-rate**: unstable at `λ ≥ 0`, healthy negative. A **retention constant** — `λ(K)`,
`λ(K · J_encode)`, always written with its operator — is an **eigenvalue magnitude**: healthy near 1,
so `0.99` is a slow cell here and a violently divergent one under this entry's reading. They are
related by `τ = −1/ln|λ_retention|` and by nothing else. **`λ(K · J_encode)`, *realised chart
retention*, is not this object** despite the near-homonym: it is a magnitude on one tick's realised
recurrence, where this is a log-rate averaged along a trajectory. Read the qualifier, never the bare
letter (#143, #227).
_Avoid_: stability margin, spectral radius (for this object), Lyapunov exponent (unless the
long-run limit is meant literally), `λ` unqualified, retention constant (the opposite convention),
realised chart retention (that is `λ(K · J_encode)`)

**Regional spectrum**:
The spectrum of the local Jacobian of whichever activation region of the shared body a cell occupies
on a given tick. A per-tick quantity, re-drawn whenever the cell's chart carries it across a fold —
never a cell attribute. What the biases set is the distribution these are drawn from — **which is no
longer the cell's effective timescale** (#143). That closing identification is retired with the rest
of the bias mechanism: retention is `K`'s, and this quantity is now the region-dependent half of
`λ(K · J_encode)`, the *realised chart retention*. It still has a referent, because `encode` is still
ReLU and its folds are still real. **It is the chart's *direct* round trip and not the whole
recurrence** (#271, #274): `decode` writes the chart onto the node stalk, reconciliation damps it,
and `encode`'s stalk half returns it next tick, so the full loop is `K (J_chart + J_stalk A_v D)`
and its `ρ` runs 1.70x to 2.12x the quoted one. **It is also not ADR-0026's `τ̂`** — that is the
e-fold decay of a paired counterfactual deviation in private features — and the two are two
instruments, never a stand-in for one another.
_Avoid_: the cell's spectrum, its Jacobian, decay rate (unqualified), the cell's effective timescale
(that is `λ(K)`'s job now), the chart loop (bare — say *direct* or *full*), `τ̂` (that is the
conduction ratio's numerator, a different instrument)

**Region dwell**:
How long a cell stays in one activation region of the shared body before its chart carries it across
a fold. **Demoted by #143 from existence to fidelity**: it used to gate whether a cell's `τ` was a
well-defined object at all, since the rate was a property of the region; under `λ(K)` the operator is
one matrix that does not reset at a fold, so `τ` is defined regardless and dwell gates only how
**faithfully** that rate is realised — the gap between `λ(K)` and `λ(K · J_encode)`. The bar below
stands unrepealed while [#226](https://github.com/NGL321/patchworks/issues/226) rules on whether it
survives the demotion. The residency must express **at least one e-fold of the region's own decay** —
`dwell > τ`; where dwell is short, a cell realises an average over unrelated regions rather than the
rate its operator holds. Nominated at
construction by the fold margin, measured at runtime on a driven trajectory — and since #160 the
runtime measurement is **the verdict**, the construction reading a nomination
(`patchworks.tick.FoldRead`, ADR-0019). Since #208 the verdict is the **median cell's** `dwell/τ > 1`,
the quantity is the **cumulative mean residency to the horizon** (never the windowed one, and the
estimator is named wherever dwell is published), the per-cell count below the floor is **reported,
never asserted**, and `dwell ≥ 2.6 τ` is **reported headroom rather than the bar** — `2.6` is
`DEFAULT_SAFETY_FACTOR`, licensed for a realised-against-regional timescale ratio and never derived
for a residency duration.
_Avoid_: region residence, switching rate, region stability, "clears 2.6 τ" as passing

**Fold margin**:
How far a cell sits from the nearest boundary of the activation region it occupies in `encode`. It
had three jobs and keeps one and a half. **Dead:** bounding `γ`, which it never did in practice.
**Falsified premise:** carrying timescale, which now lives in `K`'s spectrum. **Surviving:** it is
still the proxy for region dwell, and still what makes an expansive region dangerous rather than a
harmless transient. Falls as the body gets wider; read from `encode` alone since it is the only map
with folds, which is why the measured cap rose when `step` was linearised. Inside a fixed body a
cell's margin is uncorrelated with its decay rate. **Read live since #160**, because it moves: the
per-cell biases the prediction rule trains are the *positions* of `encode`'s folds. Weighed against
the standing offset, never against a floor — and since #206 that weighing is an **attribution**
carrying no threshold: it says why a cell lost its region, while region dwell says whether the
mechanism holds. Breaching it is a standing condition of a run, not a fault.
_Avoid_: slack, headroom, distance to boundary, construction-time check

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
that degree does not become an accidental rate difference. **Not** the consensus step of the
sheaf-ADMM literature, which carries this vocabulary in its own senses and whose local updates are
iterated to convergence on a stated global objective — there `k` counts solver iterations and the
state is re-initialised on every input, where here it is one step per tick against a state that
persists in world-time; see ADR-0002.
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
are the whole of what "evaluation" names here, and nothing aggregates a score over runs. Passing is
**one closure and one ordering**: per event, the event's loop closes — the **conduction ratio** holds
along some path from its injection site, read over L1 predicting cells, single-source rather than
swept — and the two hands' onset-latency IQRs do not overlap (ADR-0028). The between-event depth
ordering is **reported, not claimed on**: both its ends are supplied by the injection site, and every
event modifies information at every level it passes through, so what differs is the deepest level
each one reaches.
_Avoid_: the demo (bare), benchmark, test run, showcase, depth ordering (retired as a claim)

**Onset latency**:
Ticks from a disturbance to the first corrective torque. The demo's temporal measure, chosen
because it is a property of the graph — how far a correction had to travel before acting — where a
settling or decay time would be a property of the body's mechanics. Reported per event. **A pass
establishes that a correction travelled a longer path, not that a hierarchy produced it** — every
edge costs one tick, so an onset ordering is a restatement of hop count; the hierarchy claim rests on
the conduction ratio. Distinct from `τ̂`, which is also a decay time but on private features *inside
the graph*, where no joint can supply it.
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

**Conversation**:
The object the language domain models: a **time series of events** — silence, floor-holding, turn
boundaries, and who waits for whom — rather than the text those events produce. The distinction is
load-bearing rather than stylistic. It is what makes the rim tick-indexed instead of event-indexed,
so idle is a value carrying information rather than padding to be skipped; it is why the heard and
spoken rims are aligned on one tick axis rather than kept as two transcripts; and it is why a pause
is world dynamics rather than a gap in the clock. A corpus is the same text with the events deleted,
which is why it is not the unit here.
_Avoid_: dialogue, exchange, transcript, chat, session, episode

**Wedge**:
The graph's shape when the rim is a character stream: a fast taper over a tick-indexed buffer, where
depth is temporal extent, merging into the dome's own core, where depth is abstraction. The two
meanings of depth are kept apart deliberately, and neither region borrows the other's. Sibling of the
*dome*, built by the same construction rule and differing in the number of axes its rim is indexed
on; abandonable on the same terms.
_Avoid_: stack, dilated stack, pyramid, temporal hierarchy, the language dome

**Interlocutor**:
The language domain's world: a small local language model that talks to the agent and, crucially,
**responds because the agent spoke**. Not a data source and not a teacher — it holds an agenda, never
remarks on the babble, and supplies no instruction, so what it gives the agent is a world to act on
rather than a target to match. It is the entire compute cost of the domain, which is why it is small
and local. The counterpart of the *arm and arena* together, not of either alone.
_Avoid_: teacher, oracle, environment model, partner model, LLM (bare), corpus

**Floor**:
Which party currently holds the right to speak. Half-duplex, so exactly one does. Not a lock the rig
enforces on the agent — the agent emits every tick regardless — but a property of the world that
decides whether what it emits is **taken up**. **An idle run yields the floor**, so falling silent is
how the agent is heard, which is what makes silence an action rather than padding.
_Avoid_: turn, lock, mutex, speaking rights, channel

**Uptake**:
Whether a character the agent emitted was actually taken up by the interlocutor, reported back on the
spoken rim as a flag. The language rim's **refusal**: the analogue of the arm's torque clip, and what
makes the motor edge carry real disagreement rather than an echo. A character emitted while the
interlocutor holds the floor is not taken up, tick after tick, and that is the world declining a
command rather than a failure of the rig.
_Avoid_: acceptance, acknowledgement, delivery, success flag

**Coherence readback**:
The interlocutor's next-character surprisal, normalised by the entropy of its own next-character
distribution — carried as a component of the **spoken** rim's readback, never on a sensory cell and
never as the drive's asserted value. It is what the world made of the agent's command, so it is a
readback in exactly the sense efference copy is; a scalar that arrives after the agent acts and
evaluates what it did would otherwise be a reward under another name.
_Avoid_: coherence reward, coherence score, fluency signal, likelihood (bare), feedback

**Topic roster**:
The fixed set of concrete simple subjects the interlocutor talks about, with a sampler that draws one
per conversation — the language domain's counterpart of the sandbox's task sampler, and `reset()`
draws from it the way `reset()` there rearranges the world. A roster rather than a knowledge base, so
that "the partner is talking about something" costs no machinery the contract would have to describe.
Topic identity is privileged and lives in `info`.
_Avoid_: curriculum, prompt bank, knowledge base, corpus, task set
