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

Secondary, and corrected: delay-coupled loops **do** produce slow rhythms at second scale, and they
**are** steerable. An earlier draft wrote them off as `~2L`-tick resonances — ~0.5 s at the sandbox's
50 Hz — which is the wrong object as well as the wrong magnitude. The slowness lives in the relative
*phase* of units still firing near their intrinsic rate, and its period is set by **coupling
strength** ([#29](https://github.com/NGL321/patchworks/issues/29)).

The route is closed here anyway, on ground already in the record: **the coupling gain is `γ`, and `γ`
is spoken for.** [`02-tick-semantics.md`](./02-tick-semantics.md) fixes it as `gain_v = γ / Σ_e m_e`
under the bound `γ × floor <` fold margin,
[ADR-0007](../adr/0007-the-disagreement-floor-is-tolerated-not-represented.md) records that it is
explicitly *not* a timescale knob, and *The precondition* below makes that same margin what gives
this section's own mechanism a well-defined region to run in. Buying depth-slowness by raising `γ`
would spend the thing persistence runs on.

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

The design move is a step out from published work rather than a leap: neural sheaf diffusion
engineers `dim ker(Δ_F)` deliberately so that information survives what would otherwise be
oversmoothing, making stalk width the construction quantity that governs how much survives —
structurally this bound. What is unprecedented is the *use*. There the kernel is the space a
diffusion converges to as `t → ∞`; here it is state that persists tick to tick under the cell's own
dynamics, and that second half has no analogue in a formalism with no per-node recurrence in it
([#29](https://github.com/NGL321/patchworks/issues/29)).

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

**Body width sets the fold margin, and that trade is global rather than per-cell.** Hanin & Rolnick
give mean distance to the nearest region boundary as scaling like `1/#neurons`, so a **wider body has
a smaller fold margin** — while narrowness is also what supplies the dispersion (`β = Σ 1/n_j`,
[#27](https://github.com/NGL321/patchworks/issues/27) §4). Wide: stable timescales, little spread.
Narrow: real spread, margins that may not hold. Recorded here because this section is what pays for a
bad choice.

**That choice is now made, and it was cheaper than the axis suggests.**
[`01-cell-and-sheaf.md`](./01-cell-and-sheaf.md)'s *The body's construction* sizes each map at its own
minimum width — 45 / 13 / 33 for `encode` / `step` / `decode` — on the measurement that the wide end
of the axis buys no spread to trade for the margin it costs (τ ratio 2.4 at `[128]`/`[32]` against 2.7
at `[45]`/`[13]`, median fold margin 0.0067 against 0.019). The three maps are also **sized
separately**, and the margin follows the narrowest map on the chart's round trip, so `encode` can meet
a floor `step` never pays for.

[#42](https://github.com/NGL321/patchworks/issues/42) is why that choice costs nothing per cell:
**inside a fixed body a cell's decay rate and its fold margin are uncorrelated** — `corr(log ρ, log
margin) = −0.006` over 20,000 bias draws, with the slowest 1% of cells holding the same median margin
as the fastest 50%. Selecting a cell slow therefore does not cost that cell its region definition:
the trade above is paid **once, in the body's widths**, and never again per cell.

### What "stable" means here

A cell is not unstable because some region it occupies has `ρ ≥ 1`. What has to contract is the
**trajectory**: `λ = lim (1/T) log‖J_T ⋯ J_1‖ < 0`, averaged over the regions the cell actually
visits. [#42](https://github.com/NGL321/patchworks/issues/42) measured 200 cells with biases fixed
and the operating point resampled every tick — the *no-dwell* extreme, the fastest possible
region-hopping — and found **none divergent** even where 19% of the regions drawn were expansive.
The static count of cells past `ρ = 1` that [#27](https://github.com/NGL321/patchworks/issues/27)
reported was counting **region draws, not cells**, and is retired.

`max ρ < 1` survives, demoted. If no region a cell can occupy is expansive then no trajectory
through them can diverge, whatever the dwell — so it is a **sufficient** condition: cheap,
computable from the frozen body before anything is trained, and far stronger than necessary. It is
what the *construction* checks; `λ` is what the go/no-go *measures*, once there is a trajectory to
measure on.

**This is also why spread and stability were never two knobs.** Both arms are the same function of
region dwell. Where dwell is short, the spread averages away — the same cells whose per-region `τ`
spans 7.7× realise a ratio of 1.7–3.5 — and expansive regions are harmless because nothing sits in
them. Where dwell is long, the cell has a genuine timescale *and* a cell parked in an expansive
region genuinely diverges. The fold margin is therefore doing a third job, alongside the two named
above: it is what makes an expansive region dangerous.

### What being a distribution costs, and what it buys

**Nothing in the architecture reads a cell's timescale** (*The clock divisor, as an instrument*,
below). That prohibition was a discipline; under this reading it is **structural**. A per-tick
regional spectrum is not a value anything *could* branch on — there is no constant to read. The
clock divisor and the persistence mechanism stay interchangeable for the same reason as before: no
mechanism can tell them apart.

**The taper's timescale gradient is a gradient in means.** `06-graph-topology.md`'s private-dimension
gradient (0 at the rim, ~8 through L3–L6, 15 at the apex) supplies a timescale gradient, and that
gradient is now distributional: cells adjacent in depth overlap on any single tick, and only their
distributions separate. Note it is a **step rather than a ramp** — degree falls at the apex and
nowhere else in the core — so the structural gradient was never graded through the core either, and
these two facts blunt the same over-reading from different directions. The demo's sharpest falsifiable form — two different depths responding to two different
perturbations (arm-only ~1 hop, puck-moving ~4) — is a behavioural claim over many ticks and is
untouched.

**This is the honest shape of a timescale gradient, not a concession.** Discrete levels, each with
its own rate, were the artificial imposition — a construct built to solve the commitment problem
rather than a thing the substrate does. A graded, overlapping distribution of rates solves the same
problem and is what a shared piecewise-linear body with per-cell fold offsets actually produces.

## What this requires elsewhere

**A body whose regional spectra actually spread, and a construction that places them.** If the
distribution of regional Jacobian spectral radii — across the bias settings cells occupy — is a
spike, every cell lands in the same dynamic regime and nothing differentiates. Initialisation is a
parameter of the body (`01-cell-and-sheaf.md`), and this is the first thing that freedom is spent
on: the body is to be constructed for spread, not merely assumed to have it. Whether such a
construction exists was [#27](https://github.com/NGL321/patchworks/issues/27) — *constructible but
coupled* — and [#42](https://github.com/NGL321/patchworks/issues/42) took the coupling apart. The
construction has four parts, and they are as much a constraint on the body as `n = 32` and `k = 12`:

1. **`σ_w²` is set for containment, and never asked to buy spread.** It is a global, shared,
   frozen quantity; using it to widen the `τ` distribution is what put a material fraction of
   regions past `ρ = 1` in #27's sweep. Its only job is to keep the body's realised contraction
   negative with margin.
2. **The spread is imposed by selection, not by drawing.** Draw candidate bias vectors, measure the
   timescale each one produces, and **keep a set whose timescales cover the target band** —
   discarding the rest. Nothing is added to the architecture: no rate is stored, no parameter is
   introduced, and a cell's rate is still whatever its region gives it, so
   [ADR-0005](../adr/0005-timescale-is-persistence-not-a-schedule.md) is untouched. What changes is
   that the spread stops depending on what an iid draw happens to yield. Measured on the rig: taken
   *as drawn*, 400 cells span a `τ` ratio of 4.5 with the boundary clear; **selected** from that
   same distribution the reachable span is 16×, and 20,000 draws contain bias vectors whose regional
   `τ` is ≥ 100 ticks with `ρ` still under one. Those candidates are *reachable*, not yet *usable*:
   whether one holds up is a `λ` question, which is what the cap below is for.
3. **The target is a range in ticks, derived from the acceptance demo's perturbation horizons** —
   the shape chrono initialisation and S4D both use, where the range comes from the task rather
   than from whatever the initialisation produced. The number is deliberately *not* fixed here: the
   demo is still open ([#10](https://github.com/NGL321/patchworks/issues/10),
   [#17](https://github.com/NGL321/patchworks/issues/17),
   [#30](https://github.com/NGL321/patchworks/issues/30)) and is likely to grow as compositional
   behaviour is asked for. What is fixed is the derivation: **the fastest band must resolve the
   fastest perturbation the demo applies, and the slowest must outlast the longest one.**
4. **The slow end is capped by measured contraction, not by a `ρ` ceiling.** The cap is the slowest
   `τ` for which realised `λ` stays negative by a stated safety factor — a number the construction
   run produces *per body*, not a constant written down here. The factor is not decorative: #27
   measured a 2.6× one-tick non-normal amplification, and the slow-and-stable band is thin (of
   20,000 draws at one candidate width, `ρ ∈ [0.98, 1)` holds 0.15% while `ρ ≥ 1` holds 0.53%), so
   a cell placed at `ρ = 0.99` is one bias update from crossing.

**Selected timescales are assigned by level, in overlapping bands.** The taper's gradient
(`06-graph-topology.md`) is continuous, not two rates: adjacent levels overlap and only their
distributions separate, exactly as *The taper's timescale gradient is a gradient in means* has it.
Banding is a **construction choice and it costs a piece of evidence** — the correspondence between
depth and timescale is now built rather than found, so it can no longer be cited as the mechanism
working. What stays falsifiable is behavioural: recovery at the level matching the perturbation's
horizon. It does not weaken ADR-0005's prohibition, which is about *runtime*: the placement happens
once, at construction, and leaves no rate for anything to consult afterwards.

**A cheap go/no-go before anything is built.** This is the falsification condition for
[ADR-0005](../adr/0005-timescale-is-persistence-not-a-schedule.md), so what counts as passing is
specified here and does not drift with the rig. The run must establish three things:

1. **Reachability of the target band**, reported as **acceptance rate per band** — what fraction of
   drawn bias vectors land in each band of the target range. Spread itself is no longer a
   falsifier, because under *selection* above the spread is constructed rather than observed; what
   can still fail is the body being unable to *reach* a band at any sampling budget, which kills
   the mechanism exactly as a spike would have. This arm is cheap and runs before anything is
   trained. Where `τ` is reported it is reported as **quantiles** rather than moments: `τ = −1/ln ρ`
   diverges as `ρ → 1`, so moments are dominated by the tail.
2. **Measured over a driven trajectory, with the operating point varying as it will at runtime** —
   not at a frozen chart and stalk. A sweep that varies biases at a fixed operating point measures
   roughly half the phenomenon and attributes all of it to the biases. The same run reports **region
   dwell** per cell, which is what makes the `τ` it reports meaningful.
3. **Decay reported as realised contraction `λ`, not as an eigenvalue.** The regional Jacobians are
   non-normal, and `ρ < 1` is not sufficient for a bounded response (Yildiz, Jaeger & Kiebel 2012);
   `ρ` alone will mis-state the rate on the first ticks, which are the ticks reconciliation acts on.
   `λ` is the stability object (*What "stable" means here*); `max ρ < 1` is the construction-time
   sufficient check, and this run is where the sufficient check gives way to the measurement.

If no draw reaches the slow band, this mechanism is dead and the afternoon that established it was
well spent. If the band is reachable but dwell is short against `τ`, the mechanism is not dead but
it is not this one either — see *The precondition* above.

The estimator is the rig's: `prototypes/regional-spectra/spread_pilot.py` and its extension
`selection_sweep.py` — the latter adding separate `encode`/`step` widths, the fold-margin column,
the tail-reachability count and the trajectory `λ` estimator #42 decided on — hold the
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

**The instrument is the prior art's own mechanism.** Active predictive coding's `T1`/`T2` — the
source of this section's goal — *is* a fixed clock divisor, and in APC it is the **mechanism**, not
an instrument; Rao's own paper names the fixed count a limitation and defers termination functions to
future work ([#29](https://github.com/NGL321/patchworks/issues/29),
[#43](https://github.com/NGL321/patchworks/issues/43)). This spec takes APC's goal, declines its
mechanism, and then builds that mechanism as the rig it measures against. So switching the divisor
off and asking whether persistence reproduces the behaviour is a comparison **against APC**, not
against nothing.

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
- **An unbidden slow rhythm, from the graph's own cycles.** Every edge carries unit delay and
  `06-graph-topology.md` commits to lateral edges, so the dome is a delay-coupled network in exactly
  the sense *Depth does not supply it* concedes — a slow phase-difference rhythm may appear whether
  or not it is wanted, and it would be a second source of slowness this spec does not own. Already
  instrumented, though, and not by luck: a coupling phenomenon lives in the **reconciled** component
  (`im δᵀ`) and this section's mechanism lives in the **private** one (`ker δ`), which are orthogonal
  by construction. The *Demonstrating it* readout is a fixed projection computed per tick, so it
  separates the two with no new instrument.
- **Change-gated transport is specified and not built.** It is the named amplifier of this
  section's mechanism; see *The change gate, pre-specified* above for its trigger, its shape, and
  what it costs. No longer open exposure: [#20](https://github.com/NGL321/patchworks/issues/20)
  settled it.
- **The selected spread is an initialisation, and the biases drift off it.** The construction places
  each cell's timescale in its level's band, but the biases *are* the adapting surface
  ([ADR-0001](../adr/0001-continual-learning-applies-to-the-adapting-surface.md)) and the local rule
  moves them every tick — [#33](https://github.com/NGL321/patchworks/issues/33) found it can leave a
  mid-depth cell oscillating between activation regions under ambiguous evidence. So a band is where
  a cell **started**, not where it stays. **Nothing re-selects**, and deliberately: re-selection
  needs a rate to steer toward, which is exactly the runtime parameter ADR-0005 refuses. Recorded
  rather than addressed, and self-announcing — the *Demonstrating it* readout is already a live
  per-cell trace of `‖Δ(private component)‖`, so drift appears in an instrument that exists. This is
  the first place to look if the timescale gradient degrades over a long run.
- **A confidence gate is not a substitute.** Suppressing transmission when a belief is *uninformative*
  sparsifies the graph but decimates nothing in time — a confident, fast-changing cell still sends
  every tick. Recorded because the two gates are easy to conflate, and only one is a low-pass filter.
