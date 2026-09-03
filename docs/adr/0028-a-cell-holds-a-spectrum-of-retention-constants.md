# ADR-0028: A cell holds a spectrum of retention constants, learned and not placed

**Status:** accepted

**Supersedes** [ADR-0005](./0005-timescale-is-persistence-not-a-schedule.md), which is left standing
and marked superseded rather than edited. The reasoning that got from *not a schedule* to
*persistence* is worth keeping legible, and amending it in place would overwrite the argument while
keeping only its conclusion.

## Context

Settled in [#143](https://github.com/NGL321/patchworks/issues/143) and written here by
[#227](https://github.com/NGL321/patchworks/issues/227).

ADR-0005 chose **persistence** over a schedule and then had to say what supplies the persistence. It
had one mechanism available: the shared frozen body's **activation regions**, each cell placed in a
different one by its bias offsets, each region carrying its own local Jacobian and so its own decay
rate. Per-cell time constants were rejected in that ADR's own *Context* because they *"appeared to be
foreclosed by the shared frozen body of
[ADR-0001](./0001-continual-learning-applies-to-the-adapting-surface.md)."*

**The Koopman conversion lifted that foreclosure**
([#138](https://github.com/NGL321/patchworks/issues/138), which amended ADR-0005 to say so and
deliberately named no successor). `step` became `K`, a per-cell **learned linear operator**, and a
per-cell `K` *is* a per-cell time constant — `k` of them. ADR-0005's four-way choice reopened with
its rejected option now available, and this ADR is the choice made.

**The question was being asked one level too low, and that is the substance of what changed.** *Does
the timescale mechanism become `K`'s spectrum* cannot be answered before *why the architecture needs
a timescale difference at all*, and the record had never written that down in a form a mechanism
could be judged against. It is written here first, and everything else follows from it.

**What the spectral reading survived on the way.**
[#148](https://github.com/NGL321/patchworks/issues/148)'s citation pass attacked it — no algorithm
converges to the approximate point spectrum in one limit (Colbrook, Mezić & Stepanenko), spectral
pollution is the default, a frozen random dictionary is not spectrally competitive. That attack is
**voided rather than answered**, by [#145](https://github.com/NGL321/patchworks/issues/145) and
[ADR-0023](./0023-the-chart-is-not-a-koopman-lift.md): there is no lift, `K` advances a *persisting*
chart, and its eigenvalues are **retention constants by construction** rather than estimates of a
world's spectrum. Nobody asks whether a state-space model's `λ` converges to the true spectrum of
anything. [#167](https://github.com/NGL321/patchworks/issues/167) confirmed the reading at source and
found no certification-style objection anywhere in the SSM literature, for a structural reason: the
question does not arise where the spectrum is *initialised by design*. The escape holds only while
the linear-system reading is declined; treating a cell as a linear system re-imports #148 §5 in full.

## Decision

### What timescale is for: `τ` is what lets ADR-0003's standing assertion stand

[ADR-0003](./0003-action-is-prediction-the-world-clears.md) is where this design does its planning —
no second faculty, no counterfactual evaluation, no termination signal, *"a goal is realised by a
standing assertion propagating to the boundary."* It sources commitment entirely on `H⁰` insulation,
*"which makes a losing route structurally unable to re-assert through message passing."*

That is half of what an assertion needs, and the record has only ever argued one half at a time:

- **`H⁰` is commitment against the graph.** Neighbours cannot move the private component —
  orthogonally, not approximately.
- **`τ` is commitment against the cell itself.** The cell does not overwrite its own assertion
  between ticks. The private component still passes through `encode` → `K` → `decode` every tick,
  and nothing about `ker δ` makes that round trip near-identity.

**At the measured `τ ≈ 1` tick an assertion cannot stand at all**, and `H⁰` protection is beside the
point because the threat is not coming from outside. This is the statement the mechanism is judged
against, and it is stronger and more specific than ADR-0005's *commitment* row.

### Which operator `τ` is read off: three quantities, named separately

| quantity | what it is |
|---|---|
| `λ(K)` | the **operator's retention** — per-cell, learned, region-independent, settable. The reported quantity. |
| `λ(K · J_encode)` | the **realised chart retention**. Region-dependent, per-tick. |
| the body round trip on `ker δ` | what actually governs whether an assertion stands. **The precondition.** |

`τ = −1/ln|λ|` throughout, which is **arithmetic for a scalar linear recurrence and not a borrowed
convention** — #167 read the SSM literature for it and found the field reasoning this way without
any source stating the formula.

The standing-assertion framing forces the third row. What must not decay is the private component of
the *node stalk*, and with `decode` frozen as a gauge
([ADR-0014](./0014-the-linear-readout-is-gauge-fixed.md)) a cell's output is confined to `im(D)` — so
retention reaches the standing assertion only through **`im(D) ∩ ker δ`**. If that intersection is
thin, **no value of `λ(K)` makes an assertion stand**, because the cell cannot write its retained
content back into the protected subspace. That intersection is this decision's named precondition and
is [#225](https://github.com/NGL321/patchworks/issues/225).

### A cell has a spectrum of retention constants, not a timescale

The bias mechanism could only ever give a cell **one** number. A learned `K` gives it up to twelve
coexisting retention constants, one per eigen-direction, so **one cell can hold a commitment in some
chart directions while staying reactive in others**, instead of being assigned wholesale to the
reflex or to the plan.

ADR-0005's *"a cell's effective timescale is the central tendency of the distribution those draws are
taken from"* is **retired, not ported.** It describes a distribution over draws of a single rate; the
new object is a set of simultaneous rates. Same word, different object.

**No eigenvalue of `K` may be read as a mode of the piece's physics.** ADR-0023 gives this up
permanently, and it is the sharpest constraint on the vocabulary: a slow eigenvalue says the cell
**chose to retain that direction for that long** — a property of its memory policy.

### The gradient is learning's job

**`a` stays global. Construction places no per-level `τ`, and learning produces the gradient or
nothing does.**

Placement is not rejected on argument — but **it was not rejected on measurement either, and that
ground is struck (#276).**

> ~~It has already been tried and has already failed on measurement. The construction assigns `τ`
> bands by level; the run reports **0.91 at the apex against 0.99 at the rim**, no gradient at all.~~
> *Superseded by [#276](https://github.com/NGL321/patchworks/issues/276): the construction assigned
> no bands in any run that was measured.* `bias_selection.select()` is invoked only inside
> `go_no_go` and in `tests/test_bias_selection.py`; `Sheaf.__init__` and
> `benchmarks/untrained_fixed_point.build()` both take iid draws. **Placement was never built into
> a running graph, so it never failed on one**, and the flat reading is evidence about neither the
> mechanism nor learning.

**The decision is unchanged, because it never needed that ground.** What carries it is the
argument and the standing rules, none of which depend on the flat reading: the biases *are* the
adapting surface (ADR-0001), they drift off their bands, and nothing re-selects, because re-
selection needs a rate to steer toward — the runtime parameter ADR-0005 refuses and this ADR keeps
refusing. Placing by level a second time buys the same drift and the same cost ADR-0005 already
books: *"the depth↔timescale correspondence is now built rather than found, so only the
behavioural claim remains falsifiable."* It also cuts against [ADR-0015](./0015-the-cell-operator-
band-is-on-the-spectral-norm.md)'s *one global band, not one per level*, and against #127's
standing *measure the graph, not the shape imposed on it*.


**The cost is stated rather than hoped: nothing guarantees the gradient appears.** That is this
decision's **pre-registered falsification**, and it is the first time the depth↔timescale
correspondence has been falsifiable at all. If stage 2 establishes that learning cannot produce the
gradient, that redirects [#127](https://github.com/NGL321/patchworks/issues/127) rather than
deadlocking it.

### `ρ(K)` was never spoken for

The objection that the conversion already spends `ρ(K)` on transmission does not survive its own
premise. [#140](https://github.com/NGL321/patchworks/issues/140) banded `σ_max(K)`, the spectral
**norm**, and ADR-0015 says so outright: *"`ρ(K)` survives as the reported spectral quantity — it is
what timescale wants — but it is not the constrained one."* **Transmission spends the norm; retention
reads the radius.**

**And the target is reachable for the first time.** ADR-0015's upper face of exactly 1 permits
`ρ(K) = 1`, already flagged there as *"a memory that never fades. For a cell that is arguably
wanted."* `λ = 0.99` gives `τ ≈ 99.5`, comfortably inside the band, against the standing finding that
the slow target was out of reach by ~6× under bias-drawing. **The question moves from *can the body
draw a slow cell* to *will anything make one*** — which is what the falsification above watches.

## Consequences

- **Retention up to the chart's capacity is bought, and that capacity is bounded by `k`.** ADR-0005's
  blanket disclaimer — *"nothing in this section improves memory or credit assignment"* — is **not
  survivable as written**: a linear recurrence's retention *is* its short-term memory. It is replaced
  by a bounded claim. The refusal of **long-horizon** memory and of **credit**
  ([#5](https://github.com/NGL321/patchworks/issues/5)) is unchanged.
- **This decision does not inherit the chart's double duty — it creates it.** The retention constants
  are carried in the same twelve dimensions that must also name the piece, which is
  [#166](https://github.com/NGL321/patchworks/issues/166), re-wired to block on #143 rather than on
  transmission. If that budget is short, the *available* spread of `τ` is narrower than `K` alone
  suggests, which bears on the slow end.
- **Compositionality and action selection are declined here, deliberately.** They are properties of
  the graph and of the reconciliation phase. They arrive from hierarchical APC as intuition —
  `T1`/`T2`, micro-steps against macro-steps — and they join memory and credit in the not-bought-here
  column rather than sitting unbooked in neither. **A later reader reaching for the APC temporal
  hierarchy to justify a timescale mechanism is reaching for something this decision does not
  supply.** Compositionality needing a *ratio* rather than a rate is nonetheless what made the
  gradient the axis: `τ_deep / τ_rim` is the whole of the temporal hierarchy, and at ratio 1 there is
  a deep feedforward reflex arc, which is what
  [#120](https://github.com/NGL321/patchworks/issues/120) measured.
- **Dwell is demoted from existence to fidelity.** `encode` stays ReLU, so activation regions survive
  and dwell still matters — but `K` is one matrix that does not reset when a cell crosses a fold, so
  `τ` is **defined regardless**. Under the bias mechanism dwell gated whether `τ` was *well-defined*;
  under `λ(K)` it gates only how **faithfully** the operator's rate is realised. Neither vacuous nor
  false. Whether ADR-0005's `dwell > τ` bar survives the demotion is
  [#226](https://github.com/NGL321/patchworks/issues/226) and is **not** ruled on here.
- **Enforcement is one-sided, and it is the one real interaction with the band.** ADR-0015 restores
  the band by *rescaling the whole operator*, which moves every eigenvalue by the same factor — so
  when the projection fires, **all of that cell's retention constants shorten together**. Enforcement
  can only make a cell faster, never slower. Recorded; no mechanism is added for it.
- **Attributability is now precise rather than rhetorical.**
  [#27](https://github.com/NGL321/patchworks/issues/27) found biases and operating point spreading
  `τ` by 7.7× and 7.3×, *"statistically indistinguishable"*, so under the bias mechanism no per-cell
  rate was attributable to the cell. `λ(K)` is a property of a per-cell operator and needs no
  distribution over region draws to be defined. The caveat that `λ(K)` is the operator's
  *contribution* and not the realised rate is true, is not a refutation — the same holds of every
  factor in a product — and is discharged by naming all three quantities above instead of conflating
  them.
- **Retention is learned rather than drawn.** Under the bias mechanism `τ` was selected from candidate
  biases *before the cell had modelled anything*; under `K` the prediction rule sets it from the
  cell's own evidence. This is the weaker survivor of ADR-0005's *timescale as a consequence of
  content*, which ADR-0023 forbids in its strong form.
- **Two costs arrive with the SSM reading** (#167). **Zucchet & Orvieto:** long memory buys
  **parameter sensitivity** independent of gradient vanishing, and the field's mitigations —
  element-wise or diagonal recurrence, careful parameterisation — are **not available to a dense
  learned `K`**. This is a cost the design pays that the papers being cited do not. **Grazzi et al.:**
  the **sign structure** of the spectrum is expressively load-bearing, and an all-positive spectrum
  cannot represent parity. Construction starts `K = a·I`, every eigenvalue at `+a`. Nothing in
  ADR-0015's band forbids the spectrum from moving off the positive real axis, so this is an
  **initialisation** observation and cheap to check once a graph transmits.
- **Contact cells should read as fast and heavily damped — small `ρ(K)`.** From
  [#149](https://github.com/NGL321/patchworks/issues/149): the Jacobian of the 20 ms tick map with
  respect to a puck's incoming speed is 0.99 in free flight and **0.25–0.50 across a contact**,
  falling to 0.03–0.07 over four ticks. If contact cells do not read that way, that is the surprise
  worth chasing.
- **`H¹(G;F) = 0` does not become a precondition.** The claim made here is a retention **policy** —
  how long a cell chose to hold a direction — not recovery of a local law, which ADR-0023 forbids
  independently. Recorded so that nobody re-derives that the weak claim was chosen deliberately.
- **What is unchanged from ADR-0005, and load-bearing.** Timescale is **not a schedule**: every cell
  runs every tick, and the execution clock is uniform graph-wide. **Nothing in the architecture reads
  a cell's timescale** at runtime — never an input, never a selection criterion — which is what keeps
  the clock divisor interchangeable with the persistence mechanism. And **slow content lives in the
  private features**, `ker δ`, as a commitment of the spec rather than an observation.

## Alternatives considered

- **Keep placing `τ` at construction, by bias selection, now against `K`.** Rejected on measurement,
  not on argument: it was built, it drifted, and the run reports no gradient. Re-selection is what
  would fix the drift, and re-selection needs a stored rate — the one thing both this ADR and
  ADR-0005 refuse.
- **Per-level bands on `a`.** Rejected as a second timescale mechanism, and as a direct conflict with
  ADR-0015's one global band.
- **Reading a single `τ` per cell off `ρ(K)`.** Rejected as throwing away the gain: the spectrum's
  whole advantage over the bias mechanism is that a cell can be slow in some directions and fast in
  others simultaneously. Collapsing to the radius re-imposes the one-number object the bias mechanism
  was limited to.
- **Treating `K` as a linear system and certifying its spectrum.** Rejected — it re-imports #148 §5's
  certification cost in full, in exchange for a claim (that `K`'s eigenvalues estimate something about
  the world) that ADR-0023 forbids the design from making anyway.

## What the literature does and does not give

The field **sets** the spectrum; it does not estimate one. S4D draws `Δ` log-uniformly over
`[1e-3, 1e-1]`; LRU samples eigenvalues on a ring `[r_min, r_max]` and states outright that
long-range reasoning *"need[s] to have magnitude close to 1"*; Mamba's own gloss is *"a large `Δ`
resets the state `h` and focuses on the current input `x`, while a small `Δ` persists the state."*
Detail in `docs/research/167-linear-recurrence-citations.md` §5 and
`docs/research/027-regional-jacobian-spectra.md`.

**Timescale-from-spectrum is not novel** — mrDMD and the delay-embedding line got there first (#148),
and the pre-registered claim to the contrary is false. What remains this project's own is what
ADR-0005's closing section already claimed and this one inherits: obtaining timescale **separation**
from persistence alone, with no hand-set schedule, no learned discrete gate, and no dedicated
per-unit rate parameter — and, now, with the gradient left to learning rather than placed.
