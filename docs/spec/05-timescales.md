# Timescales

Where more than one timescale comes from, given the cell contract of
[`01-cell-and-sheaf.md`](./01-cell-and-sheaf.md) and the tick of
[`02-tick-semantics.md`](./02-tick-semantics.md).

Terms used here are defined in [`CONTEXT.md`](../../CONTEXT.md).

## What is actually wanted

Not "cells update at different rates." The object is **temporal abstraction**: one step at an
abstract level corresponding to many primitive actions, so a plan is short at the top and long at
the bottom. Differing rates are how the active-predictive-coding literature realises that; they are
the symptom, not the goal.

The precise thing the architecture must supply is **commitment**: an abstract belief that stands
across many primitive steps. Two failure modes motivate it, and two others are explicitly *not*
bought here:

| | |
|---|---|
| **Commitment** — holding an abstract intention steady while the rim churns | this section |
| **Representation** — encoding a variable that only exists over seconds | this section |
| **Memory** — retaining information across hundreds of ticks | recurrent failure modes; escape hatch is recurrent-state gating, a protected channel through `step` before a learned gate |
| **Credit** — associating an abstract belief with a much later outcome | [`#5`](https://github.com/NGL321/patchworks/issues/5) |

Nothing in this section improves memory or credit assignment. Believing otherwise later would be a
mistake worth avoiding now.

## Depth does not supply it

`01-cell-and-sheaf.md` states that depth buys horizon, not rate. The reason, stated here because it
is the argument the rest of this section rests on:

**Unit delay is a phase shift, not a decimation.** It moves a signal in time; it removes no
frequency content from it. Once the graph's pipeline is full — after roughly diameter-many ticks,
once — every cell receives a fresh message on *every* tick, forever. There is no subsequent silence
during which a distant cell integrates. A cell ten hops out has exactly the same input bandwidth as
one at the rim; it is merely looking at older data.

There is a real ratio that depth *does* buy, and it is the wrong one. A correction originating `d`
hops in reaches the motor rim `~2d` ticks later, so a deep cell's influence on behaviour spans many
primitive actions. That is **ratio in latency**. Planning needs **ratio in commitment** — the
abstract level standing behind one decision across many steps — and a deep cell still revises its
belief every tick. It revises it about stale evidence.

Secondary: a loop of length `L` with unit delay has slow resonant modes on the order of `2L` ticks.
At the sandbox's 50 Hz and a graph of diameter ~12 that is ~0.5 s — an order of magnitude short of
task duration, not steerable, and an artifact rather than a mechanism.

## The mechanism: persistence in the private features

Timescale is **not a schedule**. Every cell runs every tick, exactly as
[`02-tick-semantics.md`](./02-tick-semantics.md) specifies; nothing in this section changes the tick.
A cell is slow because its content **persists**, not because it updates rarely. See
[ADR-0005](../adr/0005-timescale-is-persistence-not-a-schedule.md).

Persistence needs two things, and they defend against two different disturbances.

### Insulation from neighbours: `H⁰`

Reconciliation descends on Dirichlet energy, so it moves a node stalk along `im δᵀ`. The private
features are `ker δ` (`01-cell-and-sheaf.md`, *`H⁰` is the private features*), and
`ker δ = (im δᵀ)^⊥`. **Reconciliation therefore leaves the private component of a node stalk exactly
invariant** — orthogonally, not approximately.

This is what makes slow state possible at all despite uniform bandwidth. The bandwidth argument
above applies to the *reconciled* subspace, which is driven every tick. The private subspace is not
driven by neighbours at any rate.

**Slow content lives in the private features.** This is a commitment of the spec, not an
observation about where learning might put things. It is also not an overload of what private
features are for: holding a slowly-varying variable *is* an abstract cell's sub-problem.

The bound `dim H⁰ ≥ Σ_v max(0, n − Σ_{e∋v} m_e)` makes the capacity for slow state a **construction
quantity**, set by the masks and enlarged by sparsity.

### Persistence under the cell's own dynamics: the regional Jacobian

Insulation is from *neighbours*. It is not insulation from the cell itself — the private component
still passes through `encode` → `step` → `decode` every tick, and nothing about `ker δ` makes that
round trip near-identity.

What supplies persistence is bias translation. Per `01-cell-and-sheaf.md`
(*The division of labour between the two adapting surfaces*), the shared frozen body is one
piecewise-linear map whose fold directions are fixed and whose fold offsets are per-cell. Each cell
therefore operates in a **different activation region of the same map**, and each region has its own
local Jacobian with its own spectrum.

**Two quantities, permanently distinguished.** An earlier draft of this section collapsed them, and
[#27](https://github.com/NGL321/patchworks/issues/27)'s measurement found the collapse: at a fixed
operating point, varying only the biases spreads `τ = −1/ln ρ` by 7.7× across 150 cells — but varying
only the operating point, biases fixed, spreads it by 7.3×. The two are statistically
indistinguishable, and the cell's chart moves every tick.

- The **regional spectrum** is a **per-tick** quantity: the spectrum of the local Jacobian of
  whichever region the cell occupies *this tick*. The chart moves, so it is re-drawn as the cell
  crosses folds. It is not a cell attribute.
- A cell's **effective timescale** is the central tendency of the distribution those draws are
  taken from. **That distribution is what the biases select** — a mean rate, not a fixed rate.

So: **private dimension supplies protected state; the distribution of regional spectra supplies its
decay rate.** Neither is new mechanism. Both were already committed to, for other reasons.

### The precondition: region dwell against `τ`

A mean rate is still a rate, and the mechanism survives — but it is only *the mechanism* under a
condition the earlier draft never stated. A cell's **region dwell** — how long it stays in one
activation region before its chart carries it across a fold — must be long against the `τ` that
region implies. Where it is, the cell has a timescale. Where dwell collapses to a tick or two, the
cell still decays at some average rate, but it does so by *averaging over unrelated regions*, which
is a different mechanism from the one specified here and is not what the biases were supposed to
buy.

**The fold margin is the construction-time proxy for dwell.** `02-tick-semantics.md` already checks
`γ × floor <` fold margin per cell, derived there from the disagreement floor shifting the operating
point. That check is doing a second job, and this section is the one that needs it: **the fold margin
is what makes "the cell's region" a well-defined object at all.** The relationship is the same one
`Σ_e m_e` has to the local Laplacian block's spectral radius in `02` — a cheap static quantity
standing in for a dynamic one, checkable before anything runs. Dwell itself is measured, on a driven
trajectory (*What this requires elsewhere*, below).

The bound binds hardest at the apex, since `gain_v = γ / Σ_e m_e` and `Σ_e m_e` falls with depth
(`06-graph-topology.md`) — exactly where the slow cells are meant to live. Failing it there is not
only a reconciliation-stability problem; it is the timescale claim itself failing.

**Body width pulls both ways on this, and the two directions are the same axis.** Hanin & Rolnick
give mean distance to the nearest region boundary as scaling like `1/#neurons`, so a **wider body has
a smaller fold margin** — while narrowness is also what supplies the dispersion (`β = Σ 1/n_j`,
[#27](https://github.com/NGL321/patchworks/issues/27) §4). Wide: stable timescales, little spread.
Narrow: real spread, margins that may not hold. The choice of where to sit on that axis belongs to
the body's construction and is [#42](https://github.com/NGL321/patchworks/issues/42)'s, alongside
the `σ_w²` coupling it already owns; recorded here because this section is what pays for a bad
choice.

### What being a distribution costs, and what it buys

**Nothing in the architecture reads a cell's timescale** (*The clock divisor, as an instrument*,
below). That prohibition was a discipline; under this reading it is **structural**. A per-tick
regional spectrum is not a value anything *could* branch on — there is no constant to read. The
clock divisor and the persistence mechanism stay interchangeable for the same reason as before: no
mechanism can tell them apart.

**The taper's timescale gradient is a gradient in means.** `06-graph-topology.md`'s private-dimension
gradient (0 at the rim, ~16 at the apex) supplies a timescale gradient, and that gradient is now
distributional: cells adjacent in depth overlap on any single tick, and only their distributions
separate. The demo's sharpest falsifiable form — two different depths responding to two different
perturbations (arm-only ~1 hop, puck-moving ~4) — is a behavioural claim over many ticks and is
untouched.

**This is the honest shape of a timescale gradient, not a concession.** Discrete levels, each with
its own rate, were the artificial imposition — a construct built to solve the commitment problem
rather than a thing the substrate does. A graded, overlapping distribution of rates solves the same
problem and is what a shared piecewise-linear body with per-cell fold offsets actually produces.

## What this requires elsewhere

**A body whose regional spectra actually spread.** If the distribution of regional Jacobian spectral
radii — across the bias settings cells occupy — is a spike, every cell lands in the same dynamic
regime and nothing differentiates. Initialisation is a parameter of the body
(`01-cell-and-sheaf.md`), and this is the first thing that freedom is spent on: the body is to be
constructed for spread, not merely assumed to have it. Whether such a construction exists was
[#27](https://github.com/NGL321/patchworks/issues/27), and the answer is **constructible but
coupled**: the spread is available, narrowness supplies it, and the same global knob that buys it
also positions the distribution against the stability boundary
([#42](https://github.com/NGL321/patchworks/issues/42)).

**A cheap go/no-go before anything is built.** This is the falsification condition for
[ADR-0005](../adr/0005-timescale-is-persistence-not-a-schedule.md), so what counts as passing is
specified here and does not drift with the rig. The run must establish three things:

1. **Spread in realised decay across cells**, reported as **quantiles of `τ`** rather than moments of
   `ρ`. `τ = −1/ln ρ` diverges as `ρ → 1`, so moments of the spread are dominated by the tail; a
   quantile ratio is stable where a standard deviation is not.
2. **Measured over a driven trajectory, with the operating point varying as it will at runtime** —
   not at a frozen chart and stalk. A sweep that varies biases at a fixed operating point measures
   roughly half the phenomenon and attributes all of it to the biases. The same run reports **region
   dwell** per cell, which is what makes the `τ` it reports meaningful.
3. **Decay cross-checked against something other than an eigenvalue.** The regional Jacobians are
   non-normal, and `ρ < 1` is not sufficient for a bounded response (Yildiz, Jaeger & Kiebel 2012);
   `ρ` alone will mis-state the rate on the first ticks, which are the ticks reconciliation acts on.

If the `τ` distribution is a spike, this mechanism is dead and the afternoon that established it was
well spent. If it is spread but dwell is short against `τ`, the mechanism is not dead but it is not
this one either — see *The precondition* above.

The estimator is the rig's: `prototypes/regional-spectra/spread_pilot.py` holds the
eigendecomposition, the seeds, the sample sizes (50 cells suffice), and the pseudospectral tooling
for the non-normality gap. Two limits are structural rather than provisional, and the go/no-go is
read as a **shape check** because of them: the body's widths, depth and activation are not yet fixed,
and there is nothing trained to drive the trajectory with, so the sweep runs plausible chart and
stalk sequences rather than real ones. It establishes that the mechanism is available. It does not
produce the body's number.

**Two constraints handed to graph topology** ([#8](https://github.com/NGL321/patchworks/issues/8)),
both hard:

- **Abstract cells need bounded total mask width.** A cell holds slow state only if `Σ_e m_e` is
  meaningfully below `n`. High-degree cells lose the *guarantee* of private dimension, though not
  necessarily the fact of it — the bound is a lower bound, and learned rank-deficiency enlarges `H⁰`
  past it. Low degree buys private dimension by construction; high degree makes it contingent on
  learning.
- **Integration and holding may need to be separate cells.** Nothing is disqualified by the above —
  a relay cell performs no prediction and so has no slow state to hold, which is most of what
  high-degree cells are for. The tension bites only for cells that both *predict* and *integrate
  widely*. Where an abstract slow variable needs wide integration, the shape that satisfies both is
  a **pair**: a high-degree cell that integrates, feeding a low-degree cell that holds.

## The clock divisor, as an instrument

An explicit schedule — a cell updating every `k` ticks, `k` fixed by hand — is **built first, and
not as a fallback.** It is the instrument that establishes the capability depends on timescale at
all. Force `k`, confirm long-horizon behaviour appears, and the variable is isolated before anything
is spent on making it emerge. Then switch it off and see whether persistence reproduces it.

This ordering is what makes failure cheap. If the persistence mechanism does not differentiate, the
divisor is already built and already validated against the same demo, so the system still works —
it is only less interesting.

**Nothing in the architecture reads a cell's timescale.** It is observable from outside and is never
an input to any computation, never a cell attribute another mechanism branches on, never a selection
criterion. A drive attaches by hop distance
([`04-action-and-the-boundary.md`](./04-action-and-the-boundary.md)), which both configurations
respect — and it attaches through ordinary edges, so it never reads a timescale either. This prohibition is what keeps the divisor and the persistence
mechanism interchangeable; the moment anything branches on rate, the cheap fallback is gone.

## The change gate, pre-specified

**Change-gated transport is the named amplifier of this section's mechanism, and it is specified
here but not built.** Suppressing transmission when a restricted belief has not moved makes
transmission rate track content rate, which sharpens whatever differentiation persistence already
produced. It is an amplifier and nothing else: it cannot bootstrap differentiation from nothing
([ADR-0005](../adr/0005-timescale-is-persistence-not-a-schedule.md)), and it does not substitute for
the reconciliation gain or for persistence — the three are permanently distinct
([ADR-0007](../adr/0007-the-disagreement-floor-is-tolerated-not-represented.md)).

It is specified rather than built for the same reason the relay cells are
([`06-graph-topology.md`](./06-graph-topology.md)): it has no job it is *needed* for, and it carries
a standing cost. A future session reaches for it without re-deriving it.

**The trigger.** Reach for the gate when the live private-component readout (*Demonstrating it*,
below) shows persistence differentiation that is **nonzero but below what planning needs**. Nonzero
because an amplifier has nothing to amplify otherwise; below-sufficient because if the ratio already
suffices, the gate buys only lag floor. The numeric cut belongs to the build, not to this spec.
The gate's own accept/reject test is the **quiescent hold** protocol
([#23](https://github.com/NGL321/patchworks/issues/23)): a gated graph held still must still drain
to zero.

**Where it attaches: outbound only.** The gate scales a restriction map's output *before broadcast*.
The two alternatives are the reconciliation gain wearing a hat — scaling the descent step *is* the
gain by definition, and scaling what is received before reconciliation is a per-edge gain
modulation, which reintroduces at the receiver exactly the differentiation that `γ / Σ_e m_e`
was specified to equalise ([`02-tick-semantics.md`](./02-tick-semantics.md)). Only the outbound form
changes what exists on the edge rather than how hard the receiver descends on it, and only the
outbound form makes transmission rate track content rate.

**What suppression means: hold the edge buffer.** A gated edge keeps last tick's transmitted value.
It does not send zero — that would manufacture a spurious disagreement the size of the whole belief
— and "sends nothing" has no meaning on an edge that is read every tick. Because unit delay already
means every edge carries the previous tick's value
([`01-cell-and-sheaf.md`](./01-cell-and-sheaf.md)), a hold is *do not overwrite the buffer this
tick*: **the gate adds no state**.

Neither learning rule needs a special case
([`07-local-learning-rule.md`](./07-local-learning-rule.md)). The sender computes its own
disagreement from its own current restricted belief, which gating does not touch; the receiver
reconciles and trains against whatever the buffer holds, stale or fresh, exactly as it always does.

**The threshold is locally stateless, with one global constant.** Hold when

```
‖ Δ(F_{v→e} x_v) ‖  <  ε · ‖ F_{v→e} x_v ‖
```

relative to the restricted belief's own current magnitude, with a single global `ε`. This
satisfies the real requirement — `m` varies by mask, so one absolute constant cannot mean the same
thing on two edges — without per-edge memory, and `ε` joins `γ` and the learning rate as a
permitted global scalar ([ADR-0008](../adr/0008-the-local-rule-splits-by-parameter-not-by-cell.md)).

*Considered and rejected:* deriving the threshold from a **running average of the edge's recent
scale**, the literal reading of "relative to that edge's own recent scale". It is an auxiliary
per-edge variable with a hand-set time constant, which is ADR-0005 reversed — the same objection
ADR-0007 used to reject the per-edge disagreement baseline. The gate does not get to import what the
baseline was refused.

**Boundary edges are exempt, categorically** — sensory, motor, and drive. A drive is a *standing
constant assertion* ([ADR-0009](../adr/0009-a-drive-is-a-motor-edge-attached-deep.md)), so it is the
maximally-unchanging signal in the graph and a change gate would silence it on the second tick.
Sensory and motor edges are exempt for the reason boundary cells are exempt from the settling floor:
no body, no chart, no timescale to amplify (ADR-0006). The gate is an interior-edge mechanism.
Applying it to a boundary edge deliberately, to probe staleness sensitivity, is permitted as an
**instrument** on the clock divisor's precedent — never as architecture.

**The deadband is accepted and instrumented, not designed away.** A threshold is a deadband, and a
deadband inside a loop with unit delay is a standard recipe for sustained limit cycles rather than
the equilibrium it intuitively suggests. This is a pre-specified risk of building the gate, and the
quiescent hold is what catches it.

*Considered and rejected:* **hysteresis** — two thresholds widen the very deadband at issue.
**A graded gate**, scaling transmission smoothly rather than cutting it — it removes the
discontinuity but transmits every tick, so it decimates nothing in time, which is the one property
the gate exists for. It is a weaker, different object, not a safer version of this one.

**Two neighbours it is not.** *Attention* also arrives as gating on transport
([`04-action-and-the-boundary.md`](./04-action-and-the-boundary.md)), but it is a different object:
it selects which inbound evidence a cell weights, driven by something other than the sender's own
rate of change, and its likely mechanism is the core's broadcast subspace or relay cells rather than
a per-edge threshold. *Recurrent-state gating* — the two-rung escape hatch of
[`01-cell-and-sheaf.md`](./01-cell-and-sheaf.md) (*Known exposure*) — is distinguished by **tier**: it
sits inside the cell body's recurrence, not on the edge. An earlier draft placed it on the edge stalk,
which is not on that loop at all.

## Demonstrating it

The acceptance demo's evidence is a **live private-component readout**: during a perturbation,
display `‖Δ(private component)‖` per cell against hop distance from the sensorimotor rim. The viewer
watches the rim swing while deep private state barely moves.

Cheap — the private component is the node-stalk directions masked out on every incident edge, known
at construction, so it is a fixed projection computed per tick. Falsifiable live: if deep private
state swings with the rim, the mechanism is not working and it is visible in the moment. This is a
requirement on [#11](https://github.com/NGL321/patchworks/issues/11), which must expose the
private/reconciled decomposition rather than raw stalk values.

Behaviour alone is **not** accepted as evidence — a purely reflexive controller produces the same
footage.

The readout is unaffected by timescale being a distribution, and this is not a coincidence: a
measured trace of how far private content actually moved is already an average over whatever regions
the cell passed through. It was never an eigenvalue, and it must not become one.

## Known exposure

- **The biases are over-subscribed.** They now carry three geometrically distinct jobs on one
  per-cell vector: fix the cell's fold offsets (`01-cell-and-sheaf.md`), select a regional spectrum
  slow enough to hold state, and select a region whose Jacobian preserves the private directions
  through **compression**. That third job is the fragile one — `encode` is frozen and shared, so a
  private direction that survives reconciliation still has to survive being compressed into a
  `k`-dimensional chart, and nothing guarantees it does. There is slack (the restriction maps fix
  the node stalk's basis, so a cell can partly align its private directions with what `encode`
  preserves) but the masks are construction-fixed and never re-open, so that alignment is bounded at
  build time. **This is the first concrete argument for pulling the first rung of the flex ladder —
  per-cell adapters — sooner than planned:** adapters are what buys a fourth handle if three jobs
  will not fit on one.
- **Some disagreement is irreducible by design** — a slow cell adjacent to a fast one never agrees
  with it, and that is the mechanism working. **No longer exposure; decided** in
  [ADR-0007](../adr/0007-the-disagreement-floor-is-tolerated-not-represented.md). It is a **lag
  floor**, one of the two kinds of disagreement floor, and it is tolerated rather than represented.
  Reconciliation is unharmed directly — the private component is exactly invariant at any gain — but
  the standing offset a floor leaves on the reconciled component shifts the cell's operating point,
  which is where this section's decay rate comes from. That is bounded by the reconciliation gain
  ([`02-tick-semantics.md`](./02-tick-semantics.md)), whose bound binds hardest at the apex.
- **Change-gated transport is specified and not built.** It is the named amplifier of this
  section's mechanism; see *The change gate, pre-specified* above for its trigger, its shape, and
  what it costs. No longer open exposure: [#20](https://github.com/NGL321/patchworks/issues/20)
  settled it.
- **A confidence gate is not a substitute.** Suppressing transmission when a belief is *uninformative*
  sparsifies the graph but decimates nothing in time — a confident, fast-changing cell still sends
  every tick. Recorded because the two gates are easy to conflate, and only one is a low-pass filter.
