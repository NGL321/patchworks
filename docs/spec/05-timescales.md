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
| **Compositionality** — one abstract step standing for many primitive ones | a property of the **graph**; declined here ([#143](https://github.com/NGL321/patchworks/issues/143)) |
| **Action selection** — choosing between competing abstract routes | a property of the **reconciliation phase**; declined here (#143) |

The last two rows are new, and they are **declined deliberately rather than left unbooked**. They
arrive from hierarchical active predictive coding as intuition — `T1`/`T2`, micro-steps against
macro-steps — and a later reader reaching for the APC temporal hierarchy to justify a timescale
mechanism is reaching for something this section does not supply. What compositionality *did*
contribute is the axis: it needs a **ratio** rather than a rate, so `τ_deep / τ_rim` is the whole of
the temporal hierarchy, and at ratio 1 there is a deep feedforward reflex arc — which is what
[#120](https://github.com/NGL321/patchworks/issues/120) measured.

**The memory disclaimer is bounded, not blanket** (#143). An earlier draft said *"nothing in this
section improves memory or credit assignment"*, and that is not survivable once the mechanism is a
linear recurrence: **a linear recurrence's retention is its short-term memory.** What this section
buys is retention **up to the chart's capacity**, and that capacity is bounded by `k` — the chart's
double duty, [#166](https://github.com/NGL321/patchworks/issues/166), which this mechanism does not
inherit but **creates**. **Long-horizon** memory and **credit** are still refused, unchanged.

## What timescale enforces

**`τ` is what lets [ADR-0003](../adr/0003-action-is-prediction-the-world-clears.md)'s standing
assertion stand.** Settled in [#143](https://github.com/NGL321/patchworks/issues/143), and stated
before the mechanism because it is what the mechanism is judged against.

ADR-0003 is where this design does its planning: no second faculty, no counterfactual evaluation, no
termination signal — *"a goal is realised by a standing assertion propagating to the boundary."* It
sources commitment entirely on `H⁰` insulation, *"which makes a losing route structurally unable to
re-assert through message passing."* That is half of what an assertion needs, and this section says
in its own words below why it is only half: insulation is from neighbours, and **the private
component still passes through `encode` → `K` → `decode` every tick.**

So there are two guarantees, distinct, and the record had only ever argued one at a time:

- **`H⁰` is commitment against the graph.** Neighbours cannot move the private component.
- **`τ` is commitment against the cell itself.** The cell does not overwrite its own assertion
  between ticks.

**At the measured `τ ≈ 1` tick an assertion cannot stand at all**, and `H⁰` protection is beside the
point, because the threat is not coming from outside. This is stronger and more specific than the
*commitment* row above, and it is the statement
[ADR-0028](../adr/0028-a-cell-holds-a-spectrum-of-retention-constants.md) records.

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
[ADR-0028](../adr/0028-a-cell-holds-a-spectrum-of-retention-constants.md), which supersedes
[ADR-0005](../adr/0005-timescale-is-persistence-not-a-schedule.md) — **the schedule-versus-persistence
choice above is ADR-0005's and is unchanged; what its successor replaces is the source of the
persistence.** ADR-0005 is left standing rather than edited, so the reasoning that reached
*persistence* stays legible.

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

### Persistence under the cell's own dynamics: `K`'s spectrum

Insulation is from *neighbours*. It is not insulation from the cell itself — the private component
still passes through `encode` → `K` → `decode` every tick, and nothing about `ker δ` makes that round
trip near-identity. **That round trip is where retention has to come from**, and since
[#143](https://github.com/NGL321/patchworks/issues/143) it comes from `K`.

`K` is a per-cell **learned linear operator** ([#138](https://github.com/NGL321/patchworks/issues/138)),
so its eigenvalues are **retention constants by construction**: `τ = −1/ln|λ|`. That is arithmetic
for a scalar linear recurrence rather than a convention borrowed from anywhere —
[#167](https://github.com/NGL321/patchworks/issues/167) read the state-space-model literature for the
formula and found the field plainly reasoning this way without any source writing it down.

> **`λ` carries two senses in this document and they have opposite health directions. Read the
> qualifier, never the bare letter.** A **retention constant** — `λ(K)`, `λ(K · J_encode)`, always
> written with the operator it belongs to — is an eigenvalue magnitude, and a cell is **healthy near
> 1**: `λ(K) = 0.99` is a slow cell, which is what this section wants. The **realised contraction
> rate** — `λ = lim (1/T) log‖J_T ⋯ J_1‖`, *What "stable" means here* below — is a log-rate, and a
> cell is **unstable when `λ ≥ 0`**. Under that second reading `0.99` would be violently divergent.
> The two are related by `τ = −1/ln|λ_retention|` and nothing else, they are never interchangeable,
> and `λ(K · J_encode)` (*realised chart retention*) is deliberately not the same object as
> *realised contraction rate* despite the near-homonym. `CONTEXT.md` carries both entries and the
> same warning.

**No eigenvalue of `K` is a mode of the piece's physics.** This is the sharpest constraint on the
vocabulary here, and it is
[ADR-0023](../adr/0023-the-chart-is-not-a-koopman-lift.md)'s permanently:
there is no lift, `K` advances a *persisting* chart, and **a slow eigenvalue says the cell chose to
retain that direction for that long** — a property of its memory policy, not a discovery about the
world. The whole spectral reading survives on that distinction: the certification objections
[#148](https://github.com/NGL321/patchworks/issues/148) raised — no convergence to the approximate
point spectrum in one limit, spectral pollution as the default — are **voided rather than answered**,
because they were damage to a claim the design does not make. Treating a cell as a linear system
whose spectrum estimates something re-imports them in full.

#### Three quantities, named separately

The claim is sourced on the round trip, and `λ(K)` is what gets published.

| quantity | what it is |
|---|---|
| `λ(K)` | the **operator's retention** — per-cell, learned, region-independent, settable. The reported quantity. |
| `λ(K · J_encode)` | the **realised chart retention**. Region-dependent, per-tick. |
| the body round trip on `ker δ` | what actually governs whether an assertion stands. **The precondition.** |

**Naming all three is what discharges the old objection** that `λ(K)` is only the operator's
*contribution* to the realised rate and not the rate itself. That is true, and it is not a refutation
— the same holds of every factor in a product. It was a problem only while the three were conflated.

**The third row is forced by *What timescale enforces*.** What must not decay is the private
component of the *node stalk*, and with `decode` frozen as a gauge
([ADR-0014](../adr/0014-the-linear-readout-is-gauge-fixed.md)) a cell's output is confined to
`im(D)` — so retention reaches the standing assertion only through **`im(D) ∩ ker δ`**. **If that
intersection is thin, no value of `λ(K)` makes an assertion stand**, because the cell cannot write
its retained content back into the protected subspace. That is this mechanism's named precondition
and it is [#225](https://github.com/NGL321/patchworks/issues/225), open and **not** settled here.

#### A cell has a spectrum, not a timescale

The bias mechanism could only ever give a cell **one** number. A learned `K` gives it up to twelve
coexisting retention constants, one per eigen-direction, so **a single cell can hold a commitment in
some chart directions while staying reactive in others** — rather than being assigned wholesale to
the reflex or to the plan.

**The sentence that stood here is retired rather than ported.** *"A cell's effective timescale is the
central tendency of the distribution those draws are taken from"* describes a distribution over draws
of **one** rate; the object now is a **set of simultaneous** rates. Same word, different object, and
porting the phrase would hide the change. What that sentence was answering to remains true of the
frozen body and is now a fact about `J_encode` alone: the **regional spectrum** is still a per-tick
quantity, re-drawn as the cell crosses folds, and still never a cell attribute — it is the
region-dependent half of `λ(K · J_encode)` in the table above.

**What the retirement costs, and what it buys.** It costs the reading that
[#27](https://github.com/NGL321/patchworks/issues/27) made unavoidable — biases and operating point
spreading `τ` by 7.7× and 7.3×, *"statistically indistinguishable"*, so **no per-cell rate was
attributable to the cell at all**. It buys exactly that back: `λ(K)` is a property of a per-cell
operator and needs no distribution over region draws to be defined. Retention is also **learned
rather than drawn** — under the biases `τ` was selected before the cell had modelled anything; under
`K` the prediction rule sets it from the cell's own evidence.

So: **private dimension supplies protected state; `K`'s spectrum supplies its retention.** The first
was already committed to for other reasons. The second is the conversion's, spent here for a second
purpose.

#### `ρ(K)` was never spoken for, and the slow band is reachable

The objection that the conversion already spends `ρ(K)` on transmission does not survive its own
premise. [#140](https://github.com/NGL321/patchworks/issues/140) banded `σ_max(K)`, the spectral
**norm**, and [ADR-0015](../adr/0015-the-cell-operator-band-is-on-the-spectral-norm.md) says so
outright: *"`ρ(K)` survives as the reported spectral quantity — it is what timescale wants — but it
is not the constrained one."* **Transmission spends the norm; retention reads the radius.**
`06-graph-topology.md`'s *"already spoken for three times over"* does not arrive here.

**One real interaction, and it is one-sided.** ADR-0015 restores the band by *rescaling the whole
operator*, which moves every eigenvalue by the same factor — so when the projection fires, **all of
that cell's retention constants shorten together**. Enforcement can make a cell faster and never
slower. Recorded; no mechanism is added for it.

**And the target is reachable for the first time.** ADR-0015's upper face of exactly 1 permits
`ρ(K) = 1`, which it already flags as *"a memory that never fades. For a cell that is arguably
wanted."* A retention constant of `0.99` gives `τ ≈ 99.5`, comfortably inside the band — against this
section's own standing finding that the slow end was out of reach by roughly 6× under bias-drawing.
**The question moves from *can the body draw a slow cell* to *will anything make one*.**

### The precondition: region dwell against `τ`

**Dwell is demoted, and the demotion is what #143 changed here — not the quantity, its job.** Under
the bias mechanism dwell gated whether `τ` was a **well-defined object** at all: the rate was a
property of whichever region the cell occupied, so a cell that left its region too fast had no
timescale to speak of. Under `λ(K)`, **`K` is one matrix and it does not reset when a cell crosses a
fold**, so `τ` is defined regardless of dwell. What dwell now gates is how **faithfully** the
operator's rate is realised — the gap between `λ(K)` and `λ(K · J_encode)` in the table above.
**From existence to fidelity: neither vacuous nor false.**

**Whether the `dwell > τ` bar survives that demotion is
[#226](https://github.com/NGL321/patchworks/issues/226), and it is open.** The bar and its derivation
are left standing below exactly as #208 wrote them, because a demotion is not a repeal and nothing
here licenses reading a fidelity gate as slack. What follows is the bar as it stands, and #226 owns
what becomes of it.

`encode` is still ReLU, so activation regions and fold margins are still real and the object below
still has a referent. A cell's **region dwell** — how long it stays in one activation region before
its chart carries it across a fold — must express **at least one e-fold of the region's own decay
within the residency**: `dwell > τ`. Where dwell collapses to a tick or two, what the cell realises
is an average over unrelated regions rather than the rate its operator holds.

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
operating point. That check is doing a second job, and this section is the one that needs it. **What
that second job is has moved with dwell's demotion**: the margin used to be what made *"the cell's
region"* a well-defined object, and therefore what made a timescale exist; under `λ(K)` the timescale
exists without it, and the margin bounds how faithfully the operator's rate is realised. It is still
a cheap static quantity standing in for a dynamic one, and it is still this section's.

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
[ADR-0005](../adr/0005-timescale-is-persistence-not-a-schedule.md)'s falsification clause —
**stated in the bias mechanism's terms, and read that way**: it falsifies the sufficiency of
*placing* a rate, which is the thing ADR-0028 stopped doing. Under `λ(K)` the corresponding
falsification is not this one; it is *the gradient does not appear*, below.

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
guillotine that has not dropped because the thing it cuts has not arrived.

**Whether `τ ≈ 1 tick` is what construction meant to place was #143's question, and it is now
answered: construction is not what places it.** The flat reading — 0.91 at the apex against 0.99 at
the rim, after a construction that assigned `τ` bands by level — is the measurement the placement
mechanism failed on, and it is why the gradient is handed to learning below. The `dwell/τ` numbers
above are unaffected: they are a ratio of two measured quantities and say what they said.

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

### What a per-cell spectrum costs, and what it buys

**Nothing in the architecture reads a cell's timescale** (*The clock divisor, as an instrument*,
below), and this is unchanged. What changed is **why**. Under the bias mechanism the prohibition had
become structural for free — a per-tick regional draw is not a value anything *could* branch on,
because there was no constant to read. **`λ(K)` is a constant, and it is readable in principle**, so
the prohibition goes back to being a **discipline** that has to be kept rather than a fact that keeps
itself. It is kept: no cell, edge or rule consults a retention constant while the graph runs. The
clock divisor and the persistence mechanism stay interchangeable only while that holds, and the
moment anything branches on rate, the cheap fallback is gone.

**The gradient is not placed, so the taper does not supply it.** `06-graph-topology.md`'s
private-dimension gradient (0 at the rim, ~8 through L3–L6, 15 at the apex) still says how much
protected state a cell has room for — **capacity, not rate**. It is a **step rather than a ramp**
anyway, degree falling at the apex and nowhere else in the core. Where the retention gradient is to
come from is *The gradient is learning's job*, below. The demo's sharpest falsifiable form — two
different depths responding to two different perturbations (arm-only ~1 hop, puck-moving ~4) — is a
behavioural claim over many ticks and is untouched by any of this.

**A cell is not on a rung, and that is the gain the mechanism buys.** Discrete levels, each with its
own rate, were the artificial imposition — a construct built to solve the commitment problem rather
than a thing the substrate does. What replaces them is not a smeared version of the same object: a
cell holds **up to twelve retention constants at once**, so it can commit in some chart directions
while staying reactive in others rather than being assigned wholesale to the reflex or to the plan.
An overlapping distribution of single rates was the best the biases could do; it is not what is being
claimed now.

## What this requires elsewhere

**A body that does not itself destroy retention, and a construction that leaves room for it.** What
this section asks of construction has **narrowed** since #143: the rig no longer has to *place* a
gradient, because it no longer places `τ` at all (*The gradient is learning's job*, below). What
survives is everything that keeps a body from foreclosing the thing learning is now asked to
produce — containment, the target range the demo derives, the slow cap, and `a`.

Initialisation is a parameter of the body (`01-cell-and-sheaf.md`). Whether a construction exists
that spreads regional spectra at all was [#27](https://github.com/NGL321/patchworks/issues/27) —
*constructible but coupled* — and [#42](https://github.com/NGL321/patchworks/issues/42) took the
coupling apart; that history is why parts 1 and 4 read as they do. The construction has **five**
parts, and they are as much a constraint on the body as `n = 32` and `k = 12`. The fifth arrived with
the Koopman conversion, and **the second is spent**:

1. **`σ_w²` is set for containment, and never asked to buy spread.** It is a global, shared,
   frozen quantity; using it to widen the `τ` distribution is what put a material fraction of
   regions past `ρ = 1` in #27's sweep. Its only job is to keep the body's realised contraction
   negative with margin.
2. **~~The spread is imposed by selection, not by drawing.~~ Spent by #143, and kept struck rather
   than deleted.** The rig drew candidate bias vectors, measured the timescale each produced, and
   kept a set covering the target band. **It worked as specified and the graph came out flat
   anyway** — 0.91 at the apex against 0.99 at the rim — because the biases are the adapting surface
   and drift off their bands with nothing re-selecting. That measurement is why *The gradient is
   learning's job* rules the way it does, so the part is left visible: **placement is rejected on
   evidence, not on argument, and this is the evidence.** What the sweep established stays true of
   the body and is now read as reachability rather than as placement — taken *as drawn*, 400 cells
   span a `τ` ratio of 4.5; the reachable span across 20,000 draws is 16×, containing bias vectors
   whose regional `τ` is ≥ 100 ticks with `ρ` under one.
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

### The gradient is learning's job

**`a` stays global. Construction places no per-level `τ`, and learning produces the gradient or
nothing does.** This is #143's ruling and it replaces *Selected timescales are assigned by level, in
overlapping bands*, which stood here.

**Placement is not rejected on argument. It was built, and it failed on measurement.** The
construction assigned `τ` bands by level; the run reports **0.91 at the apex against 0.99 at the
rim** — no gradient at all. The biases *are* the adapting surface
([ADR-0001](../adr/0001-continual-learning-applies-to-the-adapting-surface.md)), they drift off their
bands, and **nothing re-selects** — deliberately, because re-selection needs a rate to steer toward,
which is exactly the runtime parameter ADR-0005 refused and ADR-0028 keeps refusing. Placing by level
a second time buys the same drift and the same cost banding always booked: the depth↔timescale
correspondence built rather than found, leaving only the behavioural claim falsifiable. It would also
cut against [ADR-0015](../adr/0015-the-cell-operator-band-is-on-the-spectral-norm.md)'s **one global
band, not one per level**, and against #127's standing *measure the graph, not the shape imposed on
it*.

**The falsification is pre-registered, and it is this section's own:** nothing guarantees the
gradient appears. Learning may simply not produce it. That is stated as a cost rather than hoped
past, and it is **the first time the depth↔timescale correspondence has been falsifiable at all** —
under banding the correspondence was built in, so its presence could never be evidence. The run is
read for it. If learning cannot produce the gradient, that redirects
[#127](https://github.com/NGL321/patchworks/issues/127) rather than deadlocking it.

**What stays falsifiable behaviourally is unchanged**: recovery at the level matching the
perturbation's horizon. And the **runtime** prohibition is untouched in either direction — nothing
placed a rate, and nothing stores one for a running cell to consult.

**A cheap go/no-go before anything is built.** It was the falsification condition for
[ADR-0005](../adr/0005-timescale-is-persistence-not-a-schedule.md); under
[ADR-0028](../adr/0028-a-cell-holds-a-spectrum-of-retention-constants.md) it is **demoted from the
falsifier to a body check**, because what can now falsify the mechanism is *the gradient does not
appear*, and that is read on the run rather than before it. The run is still worth what it costs —
it can establish that a body forecloses the target before anything is trained — so what counts as
passing is specified here and does not drift with the rig. It must establish three things:

1. **Reachability of the target band**, reported as **acceptance rate per band** — what fraction of
   drawn bias vectors land in each band of the target range. Spread itself is not a falsifier; what
   can still fail is the body being unable to *reach* a band at any sampling budget, which forecloses
   the target exactly as a spike would have. **Read against `a` rather than against a placement**:
   nothing is being selected into a band any more, so this arm reports what the body admits, and it
   is where the ~6× shortfall recorded under (5) above surfaces. Where `τ` is reported it is
   reported as **quantiles** rather than moments: `τ = −1/ln ρ` diverges as `ρ → 1`, so moments are
   dominated by the tail.
2. **Measured over a driven trajectory, with the operating point varying as it will at runtime** —
   not at a frozen chart and stalk. A sweep that varies biases at a fixed operating point measures
   roughly half the phenomenon and attributes all of it to the biases. The same run reports **region
   dwell** per cell, which is what makes the `τ` it reports meaningful.
3. **Decay reported as realised contraction `λ`, not as an eigenvalue.** The regional Jacobians are
   non-normal, and `ρ < 1` is not sufficient for a bounded response (Yildiz, Jaeger & Kiebel 2012);
   `ρ` alone will mis-state the rate on the first ticks, which are the ticks reconciliation acts on.
   `λ` is the stability object (*What "stable" means here*); `max ρ < 1` is the construction-time
   sufficient check, and this run is where the sufficient check gives way to the measurement.

**What a failure of this run now means, restated for the demotion.** If no draw reaches the slow
band, the *body* forecloses the target and the afternoon that established it was well spent — it does
not kill the mechanism, because `λ(K)` is learned and is not a draw off this body. If the band is
reachable but dwell is short against `τ`, the operator's rate is realised unfaithfully rather than
undefined — see *The precondition* above, and #226.

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
display `‖Δ(private component)‖` per cell against hop distance from the sensorimotor rim.

Cheap — the private component is the node-stalk directions masked out on every incident edge, known
at construction, so it is a fixed projection computed per tick. This is a requirement on the demo
surface, which exposes the private/reconciled decomposition rather than raw stalk values. It is drawn
as its own panel — deliberately not folded into the dome's marks, so that it can disagree with the
claim it tests: [`10-the-demo-surface.md`](./10-the-demo-surface.md), *The private-component panel*.

**The picture is kept; the criterion is not the picture.** This readout was also the demo's *depth
criterion*, passing when the trace fell with hop distance, and that clause **cannot fail** — the
channel's own attenuation supplies a falling trace whether or not anything is retained, which
[#214](https://github.com/NGL321/patchworks/issues/214) measured at **8.7e-10** rim→apex. The
criterion is now a **conduction time**: per cell, the e-fold decay time `τ̂_c` of the paired
private-feature deviation over `|loop(c)|`, with the bar at `τ̂_c / |loop(c)| ≥ 1`
([ADR-0026](../adr/0026-rim-core-influence-is-a-conduction-ratio.md) for the quantity,
[ADR-0027](../adr/0027-the-demos-depth-criterion-is-a-conduction-time.md) for the demo's calls
against it, [`08-the-acceptance-demo.md`](./08-the-acceptance-demo.md) for the protocol). The scatter
stays on the panel because it is what a viewer can read; passing is the ratio.

**Falsifiable live, in weakened and stated form.** The old promise was that a flat trace refutes the
mechanism in the moment. The establishing measure is now a **paired** one — two forked runs from a
common state, which `restore` supplies and a human poking a live agent does not — so that promise
does not survive as written, and it is replaced rather than dropped. The panel gains a **live
single-run `τ̂` read**: the excursion above the agent's working baseline after an event marker (which
`10` already owed for onset, and now owes to a second consumer), e-fold time from peak, no fork
needed. It is **noisier** than the paired version and **confounded by the ongoing task** in exactly
the way [ADR-0021](../adr/0021-rim-to-core-detectability-is-a-bottleneck-ratio.md)'s quiescent-hold
floor describes — the baseline it reads against is a working agent's, not a quiet one's. And it can
still fail in the moment, because **`τ̂` flat across depth is a flat scatter**. This is `08`'s own
split — *the live run demonstrates, the repeated runs establish* — applied to the half that never had
it.

**A decay time here is not a decay time in the body.** `08`'s guard on onset latency — *never as
settling or decay time* — is aimed at the body's mechanics and at
[`03-the-sandbox.md`](./03-the-sandbox.md)'s 17.9x joint ladder. `τ̂_c` is a decay time on private
features **inside the graph**, which no joint can supply, and the ladder is deliberately built not to
align with the graph's levels. The two senses do not meet.

This readout is the *depth* half of the demo's evidence; the other half is **onset latency**, and the
protocol that fixes what passing means is
[`08-the-acceptance-demo.md`](./08-the-acceptance-demo.md).

Behaviour alone is **not** accepted as evidence — a purely reflexive controller produces the same
footage.

**The readout survived the mechanism changing under it, twice, and that is not a coincidence.** A
measured trace of how far private content actually moved is agnostic to where the retention came
from: it was already an average over whatever regions the cell passed through, and it is equally an
average over `K`'s eigen-directions weighted by wherever the cell's content actually sits. **It was
never an eigenvalue, and now that the cell has twelve of them it must not become one** — the trace is
what the demo shows, and a spectrum is not displayable in its place.

## Known exposure

- **The biases are over-subscribed — by one job fewer since #143, and the fragile one stayed.** They
  carried three geometrically distinct jobs on one per-cell vector: fix the cell's fold offsets
  (`01-cell-and-sheaf.md`), **select a regional spectrum slow enough to hold state** — *this one is
  gone, retention is `K`'s* — and select a region whose Jacobian preserves the private directions
  through **compression**. Two jobs is not one, and the survivor is the fragile one — `encode` is
  frozen and shared, so a
  private direction that survives reconciliation still has to survive being compressed into a
  `k`-dimensional chart, and nothing guarantees it does. There is slack (the restriction maps fix
  the node stalk's basis, so a cell can partly align its private directions with what `encode`
  preserves) but the masks are construction-fixed and never re-open, so that alignment is bounded at
  build time. **This is the first concrete argument for pulling the first rung of the flex ladder —
  per-cell adapters — sooner than planned:** adapters are what buys a further handle if two jobs will
  not fit on one. The argument is **weakened but not withdrawn** by losing the timescale job, and it
  is not transferred to `K`: `K` acquires the retention duty and the chart's double duty with it
  ([#166](https://github.com/NGL321/patchworks/issues/166)), which is a budget question about `k`
  rather than about the biases.
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
- **~~The selected spread is an initialisation, and the biases drift off it.~~ No longer exposure —
  it happened, and it is why placement was dropped.** The construction placed each cell's timescale
  in its level's band; the biases *are* the adapting surface
  ([ADR-0001](../adr/0001-continual-learning-applies-to-the-adapting-surface.md)), the local rule
  moves them every tick — [#33](https://github.com/NGL321/patchworks/issues/33) found it can leave a
  mid-depth cell oscillating between activation regions under ambiguous evidence — and the run came
  out **flat**. This bullet recorded the risk rather than addressing it, and **#143 addressed it by
  removing the placement**, not by adding re-selection: re-selection still needs a rate to steer
  toward, which is exactly the runtime parameter ADR-0005 refused and ADR-0028 keeps refusing. What
  carries forward is the instrument and the reason to watch — the *Demonstrating it* readout is a
  live per-cell trace of `‖Δ(private component)‖`, and it is still **the first place to look**, now
  for the opposite event: not a placed gradient degrading, but a learned one failing to appear.
- **A confidence gate is not a substitute.** Suppressing transmission when a belief is *uninformative*
  sparsifies the graph but decimates nothing in time — a confident, fast-changing cell still sends
  every tick. Recorded because the two gates are easy to conflate, and only one is a low-pass filter.
