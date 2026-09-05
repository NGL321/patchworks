# ADR-0015: The cell operator band is on the spectral norm

**Status:** accepted

## Context

Settled in [#140](https://github.com/NGL321/patchworks/issues/140), as part of the Koopman conversion
([#127](https://github.com/NGL321/patchworks/issues/127), stage 1).
[ADR-0014](./0014-the-linear-readout-is-gauge-fixed.md) is what makes this ADR's subject exist: with
`decode` frozen, `K`'s scale is identifiable and can therefore be constrained.

The conversion was taken on a stated argument — *`body` is a term in the transmission budget and today
it is not a design variable* — with the promise of **one knob, two purposes**: the same number that
keeps a cell stable would set how much a cell transmits.

**Checked against how the number was actually produced, that is two quantities wearing one name.**
The measured `body = 0.4529` is a random-direction finite-difference gain of the whole forward path,
`‖D·K·J_encode·u‖ / ‖u‖` for random unit `u`. It is not a spectral radius and never was. And the two
candidate constraints are not interchangeable:

- **stability** reads `ρ(K)`, the spectral radius;
- **transmission** reads `σ_max(K)`, the spectral norm.

For a non-normal matrix these come arbitrarily far apart: `ρ = 0.5` is compatible with
`σ_max = 50`. **Written on `ρ(K)` alone, `body` remains an unbounded factor** — precisely the defect
the conversion exists to remove. That would swap an unbounded random draw for an unbounded learned one
and buy only stability.

## Decision

**`σ_max(K)` is bounded in a construction-time band `[1/ρ_K, 1]`, enforced by normalising the
operator inside the forward path.** The *used* operator is the raw `K` rescaled to bring `σ(K)` into
the band — above the upper face, exactly `K / max(1, σ(K))` — while the raw `K` is what the prediction
rule trains. `ρ_K` is a single number for the whole graph, mirroring
[ADR-0010](./0010-restriction-map-scale-is-gauge-fixed.md)'s `ρ = 2`.

*Amended by [#433](https://github.com/NGL321/patchworks/issues/433), 2026-09-04. Until then the band
was **restored by projection after each prediction-rule step**, outside the gradient. **Only the
mechanism moved.** `σ_max(K) ∈ [1/ρ_K, 1]` is the same constraint, two-sided, the lower face `1/ρ_K`
kept; the band, both faces, the `a` rule, the norm choice against ADR-0010, the composed bound, the
contact-cell carve-out and the `ρ(K) = 1` honesty clause are all unchanged. Why it moved, and what
it is bought on, is in* Consequences.

Bounding the norm bounds the radius for free, since `ρ(K) ≤ σ_max(K)`. **`ρ(K)` survives as the
*reported* spectral quantity** — it is what timescale wants — but it is not the constrained one.

### The upper face is exactly 1, and needs no margin

"How much margin below 1" is the wrong question. What is to be forbidden is **amplification**, and
`σ_max(K) ≤ 1` forbids it exactly. A cell at `σ_max = 1` is **non-expansive** (`‖Kz‖ ≤ ‖z‖` always),
not divergent. It is the maximal-transmission end of every band that excludes amplification, which is
what the transmission budget wants, and it is Miyato's setting, so the external precedent is the
strongest available.

**Two honesty clauses, which are part of the decision and not decoration.**

- **It permits `ρ(K) = 1`** — a mode that neither grows nor decays, a memory that never fades. For a
  cell that is arguably wanted, but it is **not** the claim `|λ| < 1` and must never be written as if
  it were.
- **This is per-cell non-expansiveness, not system stability.** The closed loop is nonlinear through
  `encode`, and content crosses restriction maps whose own band licenses gain up to `ρ`. Bounding one
  factor of a product does not bound the product.

### Spectral, not Frobenius — and ADR-0010 is untouched

The mechanism matches ADR-0010; the norm deliberately diverges, and each of ADR-0010's grounds for
Frobenius was checked leg by leg:

| ADR-0010's ground for Frobenius on `F` | for `K` |
|---|---|
| Spectral pins only `σ₁`; other directions may die free — which excludes `F = 0` but leaves `F → rank 1` open | **Reverses.** ADR-0010 *wanted* learned rank-deficiency: it was the mechanism `06-graph-topology.md` relied on to enlarge `H⁰` through a functionally dead but structurally present edge. For `K`, rank-1 is the failure — ADR-0014 put the body's **entire** expressiveness in the `k`-dimensional chart. Miyato's own ground for abandoning Frobenius, that the spectral norm is independent of rank and so lets the matrix use as many features as possible, is the property wanted on the body and refused on the sheaf. *Amended by [ADR-0032](./0032-the-maps-learn-isometric-transport-and-a-spectral-floor-expresses-it.md): **this row's ground is retracted and its conclusion survives.** Rank-1 is now the failure on a map too — ADR-0031 moved `H⁰` off rank-deficiency and ADR-0032 floors the map spectrum flat. What remains true, and is the reason this ADR still differs from that one, is that the two constraints run **opposite in sign on flatness**: a map **transports**, so flat is the target; a cell operator **computes**, and non-normal transient growth is how it moves content, so flat on `K` is the starting state to escape ([#420](https://github.com/NGL321/patchworks/issues/420) §3, and [#357](https://github.com/NGL321/patchworks/issues/357) is whether it must). The band on `σ_max(K)` is untouched.* |
| The provable `λ_max(Σ_e FᵀF) ≤ Σ_e ‖F‖_F²`, giving `02`'s denominator with no SVD | **Inverts.** There, Frobenius is a cheap upper *proxy* for a spectral quantity the spec needs. Here the needed quantity **is** the spectral norm, and a proxy is not wanted when the thing itself is affordable. |
| `F = 0` unrepresentable | **Transfers** — but a lower face does that under either norm. |

The proxy is also loose exactly where it matters: `‖K‖_F ≥ σ_max(K)` with slack up to `√k ≈ 3.46`, so
a Frobenius band on a 12×12 leaves `σ_max` free across a factor of 3.46 — inside the one term the
conversion exists to make settable.

**Cost is not a counterargument.** `K` is `[150, 12, 12]` on the default dome; a batched spectral norm
over that is microseconds, and warm-started power iteration is the cheaper fallback if it ever is not.

**So the two gauges match in mechanism and differ in norm, for a stated reason.** ADR-0010 is not
amended, not weakened, and not cited as precedent for the norm — only for the shape.

*Superseded in its first clause by the 2026-09-04 amendment, and the reason is priced rather than
lost.* ADR-0010's gauge stays a **post-step projection** and the body's is now a **forward
normalisation**, so the two gauges diverge in mechanism as well as in norm. Keeping them the same
kind of object was one of [#318](https://github.com/NGL321/patchworks/issues/318)'s four grounds for
dense-with-projection, and it **expires rather than falls**: it was protecting a reasoning
convenience for an argument that had not yet been made, and
[#423](https://github.com/NGL321/patchworks/issues/423) has since made it. What the ground was for
has been spent. #318's other three grounds survive intact, because the constraint is unchanged — the
enforcement is still **local**, `σ(K)` being a function of the cell's own parameters and nothing
else; a `12x12` normalisation is still cheap, and the warm-started power iteration priced below is
now priced for exactly the per-forward-pass case; and the composed bound still reasons about both
gauges at once, since `σ_max(K) ≤ 1` is untouched. ADR-0010 remains unamended.

### One global band, not one per level

A per-level gauge would be a second timescale mechanism competing with the one the biases already
implement, and it would make the per-hop transmission budget level-dependent. Per-cell freedom lives
**inside** the one band, in `K`'s learned entries.

### `a` is a rule, not a number

`K = a·I` at construction, and `a` is not free. The construction rig places every cell's timescale
from the realised recurrence, which after the conversion is `K @ J_encode`; at `K = a·I` that is
`a · J_encode`, so **`a` multiplies every cell's placed `τ`**.

> **`a` is the largest value in the band for which the selection rig's `slow_cap` still admits the
> target `τ` band.**

A number the rig produces **per body**, exactly as `slow_cap` already is. Read plainly: *take the
longest memory that still demonstrably forgets.*

**Why the timescale constraint is the binding one rather than transmission:** nothing rides on
transmission at initialisation — the untrained graph transmits ~1e-14 whatever `a` is — whereas the
construction-time go/no-go must be **valid**.

**The two faces guard opposite failures**, and it is easy to get backwards. Lower face (`a` too
small): `ρ → 0`, the chart is wiped every tick and the cell collapses toward its bias. Upper face
(`a` too large): the cell **never forgets** and stops settling at all. Not a zeroing — a
never-letting-go. That asymmetry is why, when *no* value in the band admits the target, the rule
returns the **ceiling**: the failure that gets it there is cells forgetting too fast, and the floor is
the fastest `a` available.

## Consequences

**The normalisation is enforcement, not an objective.** It lives in the model's **forward path**,
not in the objective: no term is added to what the prediction rule minimises, and the rule still
trains the raw `K` and nothing else. This is why it is **not** grounds for making `K` a third learning
rule ([ADR-0008](./0008-the-local-rule-splits-by-parameter-not-by-cell.md), as amended). That ADR
ruled *a projection is not an objective*, and the ruling does not need re-arguing here, because the
mechanism is no longer a projection. It still reads nothing the cell did not already own: a cell owns
its own `K` outright and needs nothing from a neighbour to take its norm.

**It enforces the band by rescaling the whole operator**, which moves the norm proportionally and so
needs no SVD reconstruction. What the band restores is magnitude, never structure. The rescale is
**radial** — a scalar multiplying `K` moves every singular value *and every eigenvalue* by the same
factor — and the move does not change that. What the move changes is *when*: the rescale is applied
continuously to the used operator instead of intermittently to the stored one, so **nothing fires**,
and the prediction rule's gradient sees the constraint and optimises the normalised object.

**Why the enforcement moved, and what it is bought on.**
[#422](https://github.com/NGL321/patchworks/issues/422) measured the post-step projection's
correction over a horizon ladder on the real dome: it does not shrink over training, it **grows**,
monotonically, on both clauses and all three seeds, and hardest at the apex — 4.41x the rim's firing
rate at 100k ticks, which is where the architecture needs retention most. The move is bought on the
enforcement being **radial**, not on the band forbidding amplification. Those are two complaints and
have been read as one: a radial rescale shortens all of a cell's retention constants together, which
is [#335](https://github.com/NGL321/patchworks/issues/335)'s failure verbatim and is what moving the
enforcement addresses; the prohibition on amplification is
[#318](https://github.com/NGL321/patchworks/issues/318)'s stated mechanism and survives untouched,
below. It is **not** bought on attribution — whether the excursions are the projection's doing or
the prediction gradient's, the rescale is the same object and does the same damage in both worlds,
so what attribution changes is how much removing it buys, not whether removing it is an improvement.
#335 is therefore untouched, open and unruled by this amendment, and its 4.41 was taken on the build
this supersedes.

**It closes a gap this ADR already had.** The upper face is grounded on Miyato, who normalises in the
**forward pass**; the enforcement was nevertheless implemented post-hoc. It no longer is. No new
source is cited and none is needed.

**ADR-0007 is demoted in one clause**, and this decision is why: with `γ` already at its global
ceiling of 1.0 and timescale no longer living in activation regions, the `γ × floor <` fold margin
bound is neither binding nor motivated. The fold-margin *check* survives as a construction-time
diagnostic on `encode`. See ADR-0007, amended.

### The composed system: bounded above, and the lower side is not a bound

**The deferral is discharged.** This ADR named the spectral radius of a **composition of per-cell
operators through learned restriction maps** and refused to argue it, pending the per-hop budget the
effective-resistance work would produce. That work has reported
([#237](https://github.com/NGL321/patchworks/issues/237)), and the budget was **retracted rather than
delivered** — #237 measured effective *resistance* and diagnosed *rank*, neither of which is a per-hop
spectral gain. The argument was made anyway, on quantities already on the record, in
[#423](https://github.com/NGL321/patchworks/issues/423).

**The named object is two objects, and `ρ` is only literally correct of the second:**

- the **rim→apex chain gain**, a path product over seven hops. It is not square, so what is defined on
  it is `σ_max`. This is what over-squashing sensitivity reads.
- the **dome's tick operator on `C⁰`**, which is square, and whose `ρ` is what *"must not be globally
  contracting"* was about.

**The upper bound, derived.** The hop operator is `M = F_out · gain_v · F_inᵀ` — a transpose, not a
pseudoinverse, whose norm no Frobenius gauge would bound. Every factor is already gauge-fixed on the
record: `‖F‖_F ≤ ρ = 2` ([ADR-0010](./0010-restriction-map-scale-is-gauge-fixed.md)), `gain_v ≤ 1`
with global `γ` at its ceiling of 1.0 ([`02-tick-semantics.md`](../spec/02-tick-semantics.md)), and
`σ_max(K) ≤ 1`, this ADR's own upper face. Submultiplicativity gives `σ_max ≤ ρ²·gain_v ≤ ρ²` per hop,
and over `h = 7` hops rim to apex, **`σ_max(composed) ≤ ρ^{2h} = 4⁷ = 16,384`**. **No constant is
invented** — it is composed entirely of construction-time quantities already declared, so it meets
this ADR's own *derived rather than chosen* test outright.

Proportionality, so this is not read as a present danger. #237's direct read of the composed object
puts the trained seven-hop product at **`σ₁ ≈ 4.5e-17`**, with the most resistive edges at composed
`σ₁ ≈ 5e-4`. The bound is loose against it by some **twenty-one orders of magnitude**, and forbids
nothing that is happening. The composed-gain risk is discharged on this side; the present danger is
the opposite one, and it is the whole map. *(The figure that stood here — the `restrict` factor at
0.1706, the whole hop at 0.001086, the real system ~921× under per hop — was the **isotropic** reading
against near-rank-1 maps that [#142](https://github.com/NGL321/patchworks/issues/142) corrected and
[`docs/research/231`](../research/231-the-record-read-back.md) §3.1 calls "now the thing explicitly
rejected". It is struck rather than restated; the measurement above points the same way, harder.)*

**`ρ(K)` buys none of it.** Submultiplicativity consumes `σ_max`, never `ρ`, and this ADR has already
**spent** `σ_max(K)` at exactly 1 — that is the factor the composed bound uses.
[#420](https://github.com/NGL321/patchworks/issues/420) found `ρ(K)` free *precisely because it does
not enter the composed product at all*. So per-cell control of the radius is not composed control, and
the conversion's claim on the model-dynamics term of over-squashing sensitivity is true and narrower
than it sounds.

**The non-contraction clause is superseded, and demoted rather than deleted.** *"The dome must not be
globally contracting if it is to sustain activity"* is a true description of the **undriven** system.
Under [#144](https://github.com/NGL321/patchworks/issues/144) — persistence is sustained, not stored —
long-time behaviour is a property of the **driven field**, and a driven field needs no `ρ ≥ 1` on its
own tick operator to sustain structure. The clause stands as a description; it does not bind as a
constraint, and no composed bound owes it anything.

**The lower side is not derivable, and the reason is algebra rather than shortfall.** ADR-0010 bounds
a **Frobenius** norm, which cannot floor `σ_min` away from zero, and the structural masks make maps
rank-deficient by construction. That is #237's rank finding restated as a fact about the gauge: the
gauge was never asleep, it simply does not constrain the quantity.

**Pre-registered, and not a bound.** A two-sided composed bound becomes derivable if
[#411](https://github.com/NGL321/patchworks/issues/411)'s per-map spectral floor lands — flat maps fix
`σᵢ = ‖F‖_F/√m` and so floor each map's `σ_min` — **and** an alignment floor is measured alongside it,
since two perfectly flat maps whose carried subspaces are orthogonal still compose to zero. That
second term is what [#315](https://github.com/NGL321/patchworks/issues/315) reads as departure of
holonomy from the identity. This is a pre-registration, not a claim, and it is recorded on
[#415](https://github.com/NGL321/patchworks/issues/415), which owns the floor's consequences.

**Live caveat: apex→rim was never read.** Everything above is rim→apex. #237's rank mechanism is not
directional, so the expectation is the same reading — but expectation is not measurement.

### The falsification, pre-registered

**A band on `σ_max` forbids non-normal transient amplification, which is a real expressive loss.**
If cells prove to need transient growth to move content within a piece, the band is wrong.
`σ_max(K) ≤ 1` means `‖Kz‖ ≤ ‖z‖` for every `z` — no transient growth, ever. **The 2026-09-04
amendment does not repair this and was never bought on it**: a forward normalisation of the same
band forbids exactly what the projection forbade, and a direct parameterisation of the same set
would not have returned it either. What would answer this clause is a contraction in a *learned
metric*, which is declined under *Alternatives considered* with its trigger recorded on
[#357](https://github.com/NGL321/patchworks/issues/357). #357 is whose it is to report on.

**The loss has a price this band makes explicit.** `ρ(K) ≤ σ_max(K) ≤ 1`, with equality exactly when
`K` is normal, so **under this band non-normality is bought with retention**: every unit of it
drives `ρ` below `σ_max`, and `ρ` is what `τ` is read off
([#143](https://github.com/NGL321/patchworks/issues/143),
[ADR-0028](./0028-a-cell-holds-a-spectrum-of-retention-constants.md)). So #335's scarce resource and
#357's are **competitors** rather than one fight seen from two sides, and the band is what makes
them so. [#166](https://github.com/NGL321/patchworks/issues/166)'s near-normal `K` with a sequence
memory of 1 reads differently in that light: not only learning failing to reach non-normality, but
learning sitting at the one corner of the band where retention is cheap. #357 accordingly reads
non-normality as a **trade** and not a rate — a plateau may be learning declining a bad bargain
rather than learning failing to reach.

**The observable this section pre-registered has fired, and was acted on.** It read a standing fight
between the gradient and the projection — deliberately not damped by an additive penalty term — as
the observable that would trigger the fallback from a dense `K` to a structured one (Fan et al.,
[arXiv:2110.06509](https://arxiv.org/abs/2110.06509)).
[#422](https://github.com/NGL321/patchworks/issues/422) measured it firing, and the response was the
enforcement move above rather than the fallback. What replaces it is that amendment's own read,
stated before the build rather than after:

> **Pre-registered.** `rim τ / apex τ` and median apex `λ(K)`, at **100,000 ticks on seeds 0, 1, 2**,
> with **both builds re-run** rather than the new one differenced against stored JSON. The amendment
> is falsified if apex `λ(K)` does not rise materially above **0.529 / 0.415 / 0.289** *and*
> `rim τ / apex τ` does not fall materially below **12.87**. Those three figures and that ratio are
> #422's, taken on the **post-hoc-projection build**, at 100k ticks on three seeds — the surface named
> per [#437](https://github.com/NGL321/patchworks/issues/437).

The horizon is not negotiable downward: #422 established that at the rig's own 3k default this
measurement reads CLEAR and reverses by 100k, so a short read is not a cheap version of this one but a
different and misleading one. **Firing rate is not the metric** — under the amendment nothing fires,
so a firing ratio is undefined rather than improved, and reading it as *0, fixed* would be measuring
the instrument's own removal. And **a falsifying read is not a null result**: it would say the
gradient was never fighting the scale and simply wants a fast apex, which is the strongest attribution
evidence anyone has offered for #335, arriving from the other side.

### Contact cells: the carve-out is discharged

Every spectral claim here was written with a carve-out for the possibility that the sandbox's contact
is *impulsive*, in which case a Koopman operator is not guaranteed to exist on contact cells. The
check ran in parallel ([#149](https://github.com/NGL321/patchworks/issues/149)) and **discharged it**:
the contact is compliant, and the tick map across it is *contracting* rather than ill-conditioned. No
contact-cell exception is needed; the band applies to every predicting cell without qualification, and
contact cells are simply predicted to land at small `ρ(K)` inside it.

## Alternatives considered

**A band on `ρ(K)`.** The ticket's own framing, and rejected on the argument at the head of this ADR:
it leaves `body` unbounded and so fails the one thing the conversion is for.

**Frobenius, matching ADR-0010.** Rejected leg by leg, above. The symmetry is appealing and the
reasons behind ADR-0010's choice reverse on the body.

**A margin below 1 on the upper face.** Rejected: what is to be excluded is amplification, and 1
excludes it exactly. Any margin is transmission given away for a stability the band already has.

**Per-level bands.** Rejected as a second timescale mechanism, above.

**A direct parameterisation of a stable `K`** (Fan et al.,
[arXiv:2110.06509](https://arxiv.org/abs/2110.06509)), which is
[#318](https://github.com/NGL321/patchworks/issues/318)'s proposal. **Declined rather than
refused**, on [#433](https://github.com/NGL321/patchworks/issues/433). Its completeness result does
not transfer — it holds over the embedding and the operator jointly, and `encode`/`decode` are
frozen — so what was on offer was the mechanical property of needing no projection step, and the
forward normalisation above supplies that at no cost. Fan et al.'s *actual* template, contraction
under a learned per-cell metric, is the only construction on the table that answers the expressive
loss above, and it is declined on three grounds: it un-spends the quantity this ADR exists to bound,
since `σ_max(K)` would then be bounded only by the metric's conditioning and `body` by a **learned**
rather than a construction-time quantity — recoverable by banding the conditioning, but that is a
new invented constant; it would consume the composed bound above, which uses `σ_max(K) ≤ 1` as a
factor; and nothing has measured that cells need amplification. **The decline has a trigger,
recorded on [#357](https://github.com/NGL321/patchworks/issues/357)**: if non-normality is reached
*and* is visibly being paid for in retention, it expires and the metric parameterisation is live.

**An additive penalty on `σ_max(K)`.** Refused, and re-read rather than assumed on #433. Two
standing grounds hold unchanged: `docs/research/148` §9 rules it out because it wants a global
objective, and ADR-0008 splits the rule by parameter, so a penalty on `K` is a second term inside a
local rule. #422 supplies a third the earlier rulings did not have — the fight is **depth-graded and
not even uniform within a level** (4.41x apex over rim; on seed 0, cell `408` fires 0.071 against
cell `412`'s 0.511), so one weight is wrong across that spread in exactly the way *measure the
graph, not the shape imposed on it* ([#181](https://github.com/NGL321/patchworks/issues/181))
forbids, and a per-cell weight is an invented constant per cell with nothing deriving it.
