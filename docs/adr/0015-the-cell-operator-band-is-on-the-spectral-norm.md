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

**`σ_max(K)` is bounded in a construction-time band `[1/ρ_K, 1]`, restored by projection after each
prediction-rule step.** `ρ_K` is a single number for the whole graph, mirroring
[ADR-0010](./0010-restriction-map-scale-is-gauge-fixed.md)'s `ρ = 2`.

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

**The projection is enforcement, not an objective.** It runs after the step and outside the gradient
transform, exactly as ADR-0010's does: it is not in the objective, has no gradient, and reads nothing
the cell did not already own — a cell owns its own `K` outright and needs nothing from a neighbour to
take its norm. This is why it was **not** grounds for making `K` a third learning rule
([ADR-0008](./0008-the-local-rule-splits-by-parameter-not-by-cell.md), as amended): a projection is
not an objective.

**It restores the band by rescaling the whole operator**, which moves the norm proportionally and so
needs no SVD reconstruction. What the band restores is magnitude, never structure.

**ADR-0007 is demoted in one clause**, and this decision is why: with `γ` already at its global
ceiling of 1.0 and timescale no longer living in activation regions, the `γ × floor <` fold margin
bound is neither binding nor motivated. The fold-margin *check* survives as a construction-time
diagnostic on `encode`. See ADR-0007, amended.

### The composed system: named here, argued elsewhere

Nothing in the literature addresses the spectral radius of a **composition of per-cell operators
through learned restriction maps**, and the dome must not be globally contracting if it is to sustain
activity. **This ADR names that quantity and records it as an open risk; it does not argue it.** The
argument needs the per-hop budget that the effective-resistance work produced, and deriving a composed
bound before that reported would be *choosing* a number rather than deriving one.

Proportionality, so this is not read as a present danger: the measured `restrict` factor is 0.1706 and
the whole hop 0.001086. The real system is ~921× *under* per hop. The composed-gain risk is a
principled gap in the argument; the present danger is the opposite one, and it is the whole map.

### The falsification, pre-registered

**A band on `σ_max` forbids non-normal transient amplification, which is a real expressive loss.** If
cells prove to need transient growth to move content within a piece, the band is wrong. This is
expected to *show up* as a standing fight between the gradient and the projection, and that fight is
deliberately not damped by an additive penalty term: it is the observable that triggers the fallback
from a dense `K` to a structured one, since the right template for a stability constraint is a direct
parameterisation rather than a penalty (Fan et al.,
[arXiv:2110.06509](https://arxiv.org/abs/2110.06509)).

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
