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
| **Memory** — retaining information across hundreds of ticks | recurrent failure modes; escape hatch is recurrent-state gating, a protected channel through `K` before a learned gate |
| **Credit** — associating an abstract belief with a much later outcome | [`#5`](https://github.com/NGL321/patchworks/issues/5) |

Nothing in this section improves memory or credit assignment. Believing otherwise later would be a
mistake worth avoiding now.

## Depth does not supply it

**Scoped to a spatially-indexed rim by
[ADR-0024](../adr/0024-depth-decimates-in-time-and-not-in-space.md), and not weakened here.** The
argument below is about **spatial** pooling, where the cells a level covers are all written on the
same tick, and it is sound there — the dome is entirely spatial, so nothing in this section changes
for the sandbox. Where a rim's own axis is *time*, a cell covering four slots covers four ticks, and
that is a genuine decimation. The distinction went unnoticed while there was one domain and the two
readings coincided; see [`11-the-language-graph.md`](./11-the-language-graph.md), where the taper is
taken as **this section's instrument and not its mechanism**, at the cost of a piece of evidence.

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
is spoken for.** [`02-tick-semantics.md`](./02-tick-semantics.md) fixes it as `gain_v = γ / (g_v² · c_v)`
under the bound `gain_v × offset <` fold margin,
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
still passes through `encode` → `K` → `decode` every tick, and nothing about `ker δ` makes that round
trip near-identity.

> **The Koopman conversion moved where persistence comes from, and this section has not been rewritten
> around the successor** ([#138](https://github.com/NGL321/patchworks/issues/138),
> [#140](https://github.com/NGL321/patchworks/issues/140)). What follows describes the mechanism as
> built: persistence supplied by *bias translation*, read off the regional spectra of a frozen
> piecewise-linear map. The conversion replaced the map that carried it. `K` is a per-cell learned
> linear operator, so a cell's decay is now `ρ(K)` composed with `encode`'s regional Jacobian rather
> than `step`'s regional spectrum alone — **twelve per-cell time constants where there were none**.
>
> That is the foreclosure ADR-0005 recorded and this conversion **lifts**: per-cell time constants
> were rejected because they "appeared to be foreclosed by the shared frozen body", and a per-cell `K`
> *is* a per-cell time constant. The four-way choice this document made is therefore **reopened**, and
> its replacement is **not decided here** — reading timescale off `K`'s spectrum is a separate
> question with its own difficulties, and it is deliberately not front-run.
>
> What survives unchanged in the meantime: activation regions and fold margins are still real,
> because `encode` is still ReLU; the construction rig still places every cell's `τ` before anything
> runs; and nothing stores a rate for a running cell to consult.

What supplied persistence in the built mechanism is bias translation. Per `01-cell-and-sheaf.md`
(*The division of labour between the two adapting surfaces*), `encode` is one piecewise-linear map
whose fold directions are fixed and whose fold offsets are per-cell. Each cell therefore operates in a
**different activation region of the same map**, and each region has its own local Jacobian with its
own spectrum.

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
activation region before its chart carries it across a fold — must express **at least one e-fold of
the region's own decay within the residency**: `dwell > τ`. Where it does, the cell has a timescale.
Where dwell collapses to a tick or two, the cell still decays at some average rate, but it does so by
*averaging over unrelated regions*, which is a different mechanism from the one specified here and is
not what the biases were supposed to buy.

**One e-fold is a derivation rather than a constant**
([#208](https://github.com/NGL321/patchworks/issues/208)). Over a residency `D` at rate `1/τ` the
in-region residual is `exp(−D/τ)`, so at `D = τ` it is `1/e`: **63% of the cell's content decays
inside the region it is claimed to decay in.** Below one e-fold the regional spectrum describes a
decay that never happened — which is this section's own named failure, *dwell collapses to a tick or
two*. Above it, every further multiple is a **choice** of how many e-folds to demand, not a
derivation, which is why no larger factor is written here.

**The verdict is the median cell's `dwell/τ > 1`.** Not a per-cell extremum:
[#195](https://github.com/NGL321/patchworks/issues/195) ran four times at one seed and got four
different binding cells, so a bar on the worst cell is a bar on noise. Not a population fraction
either — a fraction needs a level, and a level read off one run's plateau is exactly the constant
[#206](https://github.com/NGL321/patchworks/issues/206) declined a tolerance for. The median needs no
chosen level: the derived `1` is the whole bar. **The per-cell count below the floor is reported,
never asserted**, which keeps the low-dwell cells visible without constituting them as a population
— [#205](https://github.com/NGL321/patchworks/issues/205) read those arrays and found the set is the
base rate.

**Dwell is the cumulative mean residency to the horizon, and the estimator is part of the published
quantity.** #206's *"131 of 150 clear `2.6 τ`"* is a **windowed** dwell over the last 25,000 ticks;
the **cumulative** dwell on the same run, same gate, gives **125 of 150**. Six cells of difference
from a choice nobody had written down — and the window length was itself an unstated constant that
grew 100 → 25,000 across the checkpoint table. A verdict may not rest on a knob the reporting code
picked, so wherever dwell is published the estimator is named with it.

**`dwell ≥ 2.6 τ` is reported headroom, and is not what passing means.** `2.6` is
`bias_selection.py`'s `DEFAULT_SAFETY_FACTOR` — [#27](https://github.com/NGL321/patchworks/issues/27)'s
one-tick non-normal amplification — and it was never derived for a residency duration. Its home use
is sound: `contained()` bounds a *realised* timescale against the *regional* one, a ratio of two
times. The transplant onto a **duration** carried no warrant with it. The count keeps being published,
because it is informative and comparable across runs. It stopped being the bar.

**The fold margin is the proxy for dwell.** `02-tick-semantics.md` checks
`gain_v × offset <` fold margin per cell, derived there from the standing offset shifting the
operating point. That check is doing a second job, and this section is the one that needs it: **the
fold margin is what makes "the cell's region" a well-defined object at all.** It is a cheap static
quantity standing in for a dynamic one.

**Since [#160](https://github.com/NGL321/patchworks/issues/160) the proxy nominates and the
measurement decides** ([ADR-0019](../adr/0019-construction-nominates-the-run-decides.md)). Dwell was
always specified as measured on a driven trajectory (*What this requires elsewhere*, below); what
changed is that it is measured **on the run** rather than only on the construction sweep, and that
this section's precondition is sourced onto that measurement rather than onto the proxy. The proxy
moved for a reason the record had not noticed: the per-cell biases the prediction rule trains *are*
the positions of `encode`'s folds, so the arrangement slides under the operating point for the length
of the run. The live margin-against-offset comparison rides alongside as the **attribution** — dwell
says a cell left its region, and only the comparison says whether reconciliation is what moved it.

**Where the bound binds hardest is not claimed.** This section used to say the apex, since
`gain_v = γ / Σ_e m_e` and `Σ_e m_e` falls with depth — exactly where the slow cells are meant to
live. [#190](https://github.com/NGL321/patchworks/issues/190) made `gain_v` uniform across the
interior and #160 struck the claim without replacing it. What survives is the consequence, which
never depended on *where*: failing the bound is not only a reconciliation-stability problem; it is the
timescale claim itself failing.

**Body width sets the fold margin, and that trade is global rather than per-cell.** Hanin & Rolnick
give mean distance to the nearest region boundary as scaling like `1/#neurons`, so a **wider body has
a smaller fold margin** — while narrowness is also what supplies the dispersion (`β = Σ 1/n_j`,
[#27](https://github.com/NGL321/patchworks/issues/27) §4). Wide: stable timescales, little spread.
Narrow: real spread, margins that may not hold. Recorded here because this section is what pays for a
bad choice.

**That choice is now made, and it was cheaper than the axis suggests.**
[`01-cell-and-sheaf.md`](./01-cell-and-sheaf.md)'s *The body's construction* sized each map at its own
minimum width — 45 / 13 / 32 for `encode` / `step` / `decode` — on the measurement that the wide end
of the axis buys no spread to trade for the margin it costs (τ ratio 2.4 at `[128]`/`[32]` against 2.7
at `[45]`/`[13]`, median fold margin 0.0067 against 0.019).

**After the conversion only `encode`'s 45 remains**: `K` and `decode` are linear and have no hidden
width. The clause about the margin following the narrowest map on the round trip is spent — there is
one map with folds now, so the margin is `encode`'s outright, which is what raised the measured cap
from 0.2600 to 0.3502.

[#42](https://github.com/NGL321/patchworks/issues/42) is why that choice costs nothing per cell:
**inside a fixed body a cell's decay rate and its fold margin are uncorrelated** — `corr(log ρ, log
margin) = −0.006` over 20,000 bias draws, with the slowest 1% of cells holding the same median margin
as the fastest 50%. Selecting a cell slow therefore does not cost that cell its region definition:
the trade above is paid **once, in the body's widths**, and never again per cell.

**The other direction of that decoupling is a cost, and this section only ever wrote the favourable
one.** If `τ` and dwell are uncorrelated then selecting a cell slow also buys it no **residency to be
slow in**: `τ` is placed by the biases, dwell is set by how fast the graph's chatter walks the
operating point across creases, and the precondition above is a ratio of the two. The live run agrees
with the construction sweep — `corr(log τ, log dwell) = −0.110` against #42's construction-time
`−0.006` — so **nothing in the architecture makes the precondition hold.** The mechanism has an
**unenforced precondition**, and what a collapse of it falsifies is
[ADR-0005](../adr/0005-timescale-is-persistence-not-a-schedule.md)'s falsification clause.

**What the live read says today, and why the gate is not slack** (#208, on
`prototypes/live-fold-read-206/206-per-tick.npz`). At the horizon the median cell sits at
`dwell/τ = 9.49` and 147 of 150 cells clear `dwell > τ`; at tick 100 the median sat at 0.96. The graph
**fails the condition at the start and earns it over the run.** It clears comfortably because `τ` is
**flat at about one tick graph-wide** — 0.91 at the apex against 0.99 at the rim, no depth→timescale
gradient in `τ` at all — not because the placement is healthy. Against this section's own target of
`τ ≥ 100` ticks the apex's dwell of 33 gives `dwell/τ ≈ 0.33`: **at the target band the gate fails
outright**, and it fails at the level where the slow cells are meant to live. That is a reading of
these two measured numbers, not a revival of the margin claim #190 struck — it is the *dwell* gate at
a `τ` the run has not reached, not `gain_v` falling with depth. The gate is not slack; it is a
guillotine that has not dropped because the thing it cuts has not arrived. Whether `τ ≈ 1 tick` is
what construction meant to place is
[#143](https://github.com/NGL321/patchworks/issues/143)'s question and is not ruled on here.

**Those counts were read on the post-conversion body.** The Koopman conversion merged as
[PR #161](https://github.com/NGL321/patchworks/pull/161), commit `cd52077`, on 2026-08-29 — before
either read ran. The #202 read is commit `8fa297e` (2026-08-30) and #206's re-run is `bcc0a70`
(2026-08-31), and `cd52077` is an ancestor of both, so each carries `K` and neither has a `step`
left to compose. **The body that ran is the one on `main` today**, stated here so the next reader
need not re-derive it: `K` per [#138](https://github.com/NGL321/patchworks/issues/138), `encode`
still the body's only nonlinearity and still ReLU, `decode` linearised and frozen as a gauge. A
"region" in those reads is therefore a facet of `encode` alone — which is what
[`02-tick-semantics.md`](./02-tick-semantics.md) already records: linearising `step` took its folds
off the round trip, so the partition is `encode`'s own.

**The sandbox has moved under them since, in one respect.**
[#266](https://github.com/NGL321/patchworks/issues/266) put an `mj_forward` into
`PlanarPushSandbox.step`, which until then handed back an observation drawn from kinematics one
integration stale — so both reads were driven by an image and a touch reading that disagreed with the
`qpos` and `qvel` beside them. The stalks feeding these counts move by ~2.8% relative, which **dates
the digits without touching what they say**: `τ` flat at about a tick with no depth gradient is a
shape, and the gate's verdict turns on the shape rather than on a third decimal place. Re-reading is
a 100,000-tick run and has not been done.

The pass condition survives the conversion because it is definitional — one e-fold is one e-fold
whatever the facets are — and **the counts survive it too, by having been read after it**. Nothing
here is rescaled: #206's precedent for `01`'s recorded margins holds in this direction as well,
because rescaling would publish a number nobody ran. Not to be confused with
[#155](https://github.com/NGL321/patchworks/issues/155), which is the *transmission* edit and not
the conversion; it merged later still, as [PR #221](https://github.com/NGL321/patchworks/pull/221)
on 2026-08-31, and gates nothing here.

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
construction has **five** parts, and they are as much a constraint on the body as `n = 32` and
`k = 12`. The fifth arrived with the Koopman conversion:

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
5. **`a`, the scalar in `K = a·I` at construction, is the largest value in the operator band for
   which the cap above still admits the target band.** A number the construction run produces *per
   body*, exactly as the cap in (4) is — the module's existing habit rather than a new one. Read
   plainly: *take the longest memory that still demonstrably forgets.*

   It is not free, and the coupling is why it belongs here rather than in
   [`01-cell-and-sheaf.md`](./01-cell-and-sheaf.md) alone. The realised recurrence this rig measures
   is `K @ J_encode`, and at construction that is `a · J_encode`, so **`a` multiplies every cell's
   placed `τ`**. The two faces guard opposite failures and are easy to get backwards: too small and
   `ρ → 0`, the chart is wiped every tick and the cell collapses toward its bias; too large and the
   cell **never forgets** and stops settling at all — not a zeroing, a never-letting-go.

   **The timescale constraint is the binding one, not transmission.** Nothing rides on transmission
   at initialisation; the construction-time go/no-go must be *valid*.

   **Measured on the default body (#157), and the honest reading is that `a` is not currently the
   binding constraint.** Containment holds for all 2048 candidates at every `a` in the band, so the
   upper face never binds, and the cap simply rises with `a` — 0.90 ticks at `a = 0.5` to 2.50 at
   `a = 1.0`. Both readings of the demo's horizons ask for a slow end of 14 ticks or more, so the
   target is unreachable by roughly 6× whatever `a` does, and the rule returns the **ceiling**. That
   shortfall is a pre-existing property of a body whose effective timescale sits around one tick, and
   it is what arm 1 of the go/no-go below exists to report.

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
`selection_sweep.py` — the latter adding separate `encode`/`step` widths (before the conversion left
one), the fold-margin column,
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
modulation, which makes `gain_v` a per-edge quantity varying at runtime where
[`02-tick-semantics.md`](./02-tick-semantics.md) specifies it per cell as `γ / (g_v² · c_v)`, every
term of it read off the built graph or declared once. Only the outbound form
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
requirement on the demo surface, which exposes the private/reconciled decomposition rather than raw
stalk values. It is drawn as its own panel — deliberately not folded into the dome's marks, so that
it can disagree with the claim it tests: [`10-the-demo-surface.md`](./10-the-demo-surface.md), *The
private-component panel*.

This readout is the *depth* half of the demo's evidence; the other half is **onset latency**, and the
protocol that fixes what passing means is
[`08-the-acceptance-demo.md`](./08-the-acceptance-demo.md).

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
