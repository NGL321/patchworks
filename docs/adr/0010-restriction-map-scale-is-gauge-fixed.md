# ADR-0010: Restriction map scale is gauge-fixed

**Status:** accepted

## Context

Settled in [#37](https://github.com/NGL321/patchworks/issues/37), raised by the citation pass in
[#16](https://github.com/NGL321/patchworks/issues/16) (`docs/research/016-cell-contract-citations.md`,
R1 and R5). Two independent routes reach the same failure — the sheaf stops coupling anything and the
error signal disappears — and the spec excluded neither.

**Route one: parameter collapse.** Di Nino, Barbarossa & Di Lorenzo
([arXiv:2501.19207](https://arxiv.org/abs/2501.19207), §III), the one paper that learns restriction
maps by minimising disagreement directly, states: *"We also need to impose a constraint on the set of
feasible restriction maps `F`, to avoid the trivial solution."* Their own choice is orthonormality —
offered, the citation pass in [#53](https://github.com/NGL321/patchworks/issues/53) found, as *"a
possibility that is theoretically plausible and gives rise to a simple solution"*, the simplicity being
that it collapses to a closed-form orthogonal Procrustes solve. The requirement is that the feasible set
be constrained at all; the orthogonal class is one member of it. Every other learned-sheaf method
constrains its maps by construction (Bodnar: invertible classes; Barbero: `O(d)`). Patchworks imposed
no orthogonality, invertibility, norm, or rank floor: the structural mask constrains *which* entries
may be nonzero and requires none of them to be.

**Route two: state collapse.** Cai & Wang ([arXiv:2006.13318](https://arxiv.org/abs/2006.13318))
state over-smoothing precisely as the Dirichlet energy of the embeddings converging to zero. Since
disagreement **is** the Dirichlet energy and is the only error signal, over-smoothing here is not a
downstream quality loss — it is the vanishing of the quantity both halves of the learning rule are
computed from, and of the only instrument that would show it happening.

**What R1 could not see from where it stood.**
[ADR-0007](./0007-the-disagreement-floor-is-tolerated-not-represented.md) already forbids the zero
target: the transport rule learns on *change* in residual, or on residual relative to that edge's own
recent scale. That constraint was made for the disagreement floor, and it turns out to bear directly
on collapse. This ADR follows the consequence out.

## Decision

### The transport objective is scale-invariant, which leaves magnitude *unidentified*

ADR-0007's two permitted objectives are not equivalent here, and the spec now commits to one. Learning
on **change** in disagreement does not exclude collapse — shrinking the maps produces a negative change
every step, which a change-descending rule can read as progress. Learning on disagreement **relative to
the edge's own scale** does, because the ratio is invariant under scaling both of an edge's maps
together: shrinking the edge buys it nothing. (Scaling *one* map is a different direction and is not
invariant — see *The free magnitude grows* below, where the two are separated.)

The normaliser is **locally stateless** — disagreement relative to `‖F_u x_u‖ + ‖F_v x_v‖`, the
restricted beliefs' own current magnitudes — and not a running average of the edge's recent scale. This
is the same criterion form, adopted for the same reason, as the change gate's threshold in
[`05-timescales.md`](../spec/05-timescales.md): a per-edge running average is an auxiliary variable with
a hand-set time constant, which is the object ADR-0007 rejects under *A per-edge learned baseline*.

The sparsity pressure composed into the same step is **L1 on the normalised map**, which redistributes
weight across the map's directions rather than removing it. This is what
[`06-graph-topology.md`](../spec/06-graph-topology.md)'s "prunes *within* the mask; does not shrink the
stalk" always meant, now stated mechanically.

*Amended by [#89](https://github.com/NGL321/patchworks/issues/89): the term is normalised by the mask
size as well as by the norm.* This ADR said only "L1 on the normalised map", which reads as
`‖F‖₁ / ‖F‖_F`, and the build found that form does not do the job the surrounding argument needs. Its
gradient has norm `√(p − q²) / ‖F‖_F` for a map whose structural mask leaves `p` weights open, so it
**grows with the mask**: one global `λ` — and `λ` is global, one of exactly two permitted signals
([`07-local-learning-rule.md`](../spec/07-local-learning-rule.md)) — pruned a wide map roughly
eightfold harder than a narrow one across the built dome, at `+0.985` correlation with `p`. That is a
per-map pressure varying with a structural accident of the mask, which nothing in this ADR argues for.

The term is therefore

```
‖F‖₁ / (√p · ‖F‖_F)
```

and the `√p` is exact rather than a correction factor: the gradient becomes `√(1 − h²) / ‖F‖_F`, where
`h` is the term's own value, and **`p` is absent from it identically**. Measured correlation with `p`
falls to `+0.071`. The one trace left is that `h` cannot fall below `1/√p`, so the *attainable* ceiling
still varies by 6.8% across this dome's range of mask sizes — at full concentration, which the maps do
not reach.

Three things the amendment deliberately does not disturb. Dividing by a construction-time constant per
map changes nothing **within** a map, so "prunes within the mask" and the rank-concentration pricing
below are untouched, and `06-graph-topology.md` and `01-cell-and-sheaf.md` remain accurate as written.
The term stays blind to magnitude, so the unidentified-parameter argument this section rests on is
unaffected. And `√p` is read off the structural mask, which is fixed at construction and closes
permanently — it is the same kind of object as the `Σ_e m_e` the reconciliation gain divides by, not
the per-edge auxiliary variable [ADR-0007](./0007-the-disagreement-floor-is-tolerated-not-represented.md)
rejects. The normalised quantity is Hoyer's sparseness ratio, which carries the `1/√p` for exactly this
reason: to be comparable across dimensions.

Both terms are therefore blind to the map's overall magnitude. **Nothing in the transport rule has an
opinion about it**, which means the magnitude is not merely unconstrained but *unidentified by the
objective*. This is the whole argument for what follows: fixing an unidentified parameter removes a free
parameter, it does not cap a learned one. The argument is not novel and does not need to be: it is
Arora, Li & Lyu's Definition 2.1 ([arXiv:1812.03981](https://arxiv.org/abs/1812.03981)) and van
Laarhoven's opening move ([arXiv:1706.05350](https://arxiv.org/abs/1706.05350)), restated for
restriction maps.

### The free magnitude grows; it does not collapse

*Corrected against the record by [#57](https://github.com/NGL321/patchworks/issues/57). This ADR as
first written said the free magnitude drifts toward `F = 0`. It drifts the other way, and the correction
moves which end of the band is load-bearing.*

Two directions have to be separated first, because they behave differently and the original sentence ran
them together.

- **An edge's joint scale**, `(F_{u◁e}, F_{v◁e}) ↦ (αF_{u◁e}, αF_{v◁e})`. The relative objective is
  *exactly* invariant here — the normaliser `‖F_u x_u‖ + ‖F_v x_v‖` scales with the numerator — and the
  L1 on the normalised map is invariant under scaling either map alone. This is the direction nothing
  has an opinion about.
- **An edge's scale ratio**, one end up and the other down. This is **not** invariant, and it is the
  residual asymmetry the original sentence reached for.

Along the invariant direction, Arora et al.'s **Lemma 2.4** settles the dynamics: for a scale-invariant
parameter the gradient is *always perpendicular* to it — this follows from scale-invariance alone, by
differentiating `F(w) = F(cw)` in `c` — so `‖w_{t+1}‖² = ‖w_t‖² + η²‖∇‖²`. Both ends of an edge step
in the same tick under the one global learning-rate scalar that
[ADR-0008](./0008-the-local-rule-splits-by-parameter-not-by-cell.md) permits, so the lemma applies to
the pair: **`‖F_{u◁e}‖_F² + ‖F_{v◁e}‖_F²` is non-decreasing at every step, and strictly increasing
whenever the gradient is nonzero.** Salimans & Kingma
([arXiv:1602.07868](https://arxiv.org/abs/1602.07868)) report the same monotone growth for Weight Normalization's shadow parameter `v`.

Along the ratio, the objective points *away* from collapse rather than toward it, which the ADR had no
argument for either way. By the triangle inequality the relative disagreement lies in `[0, 1]`, and
sending either map to zero sends it to **1, its maximum**: one-sided collapse is the worst value the
transport rule can reach, not a trivial solution it can fall into.

So the danger of an unidentified magnitude here is the one the constrained-weight literature actually
documents — **growth with a vanishing effective learning rate**, maps that stop moving while every
norm-based diagnostic reads them as healthy (Ioffe & Szegedy's observation, quoted by Arora et al.: the
growth "has an effect of reducing the learning rate"). The **upper** end of the band closes it, and
closes it by construction rather than by instrumentation: `‖F‖_F ≤ ρ` bounds the effective step size
from below. It is not recorded as a known exposure for that reason.

**What survives unchanged:** the magnitude is unidentified, and gauge-fixing is still the response.
Only the direction of the danger moves.

**Caveat kept honest.** Lemma 2.4 assumes plain gradient descent on an exactly scale-invariant loss.
Patchworks composes a projection into the same step, and the normaliser is exactly invariant only when
both ends scale together. Growth is therefore the expected default, not a theorem about this rule —
which is also why the band's lower end is kept rather than dropped.

### Interior maps carry a band; boundary maps carry the exact gauge

- **Interior restriction maps**: `‖F‖_F ∈ [1/ρ, ρ]`, with `ρ = 2` a **construction-time constant**.
  Nothing reads it at runtime, no cell estimates it, and it never anneals.
- **Boundary-cell restriction maps**: exactly `‖F‖_F = 1`. A boundary cell *runs no body*
  ([`04-action-and-the-boundary.md`](../spec/04-action-and-the-boundary.md)) and its stalk is
  world-shaped by [ADR-0006](./0006-boundary-cell-stalks-are-world-shaped.md), so it holds no metric
  individuality that a band would protect. Where world units and stalk units disagree, the environment
  contract owns the conversion, not the sheaf.

Enforcement is **projection**: take the transport step, then project the map back into the band. This is
van Laarhoven's §5.5 move for move — step, then renormalise, explicitly distinguished there from Weight
Normalization's reparameterisation for the same reason given below.

**The two ends of the band do different jobs**, which the original *guardrail rather than a continuous
force* framing missed. Because an edge's joint scale grows monotonically, the upper face binds
continuously: the larger end of every interior edge rides `‖F‖_F = ρ`, and what the band actually buys
is `ρ²` of freedom in an **edge's scale ratio**, not `ρ²` of slack around each map. The lower bound is
the guardrail — nothing in the objective drives toward it — and it is kept because the residual
asymmetry and finite arithmetic sit outside Lemma 2.4's assumptions, at the cost of one comparison.

Reparameterising as `F = G/‖G‖_F` was rejected: it costs no
more, but it leaves a shadow parameter `G` that the sparsity term can drive toward zero, reintroducing
collapse as numerical ill-conditioning in a parameter no diagnostic watches.

`F = 0` is now **unrepresentable**, not merely disfavoured.

### Frobenius, not spectral — and therefore no rank floor

Unit **spectral** norm pins only the largest singular value; every other direction may shrink to zero at
no cost, so it excludes `F = 0` while leaving `F → rank 1` wide open. Unit **Frobenius** norm pins the
sum of squares across all directions, so concentrating a map onto fewer directions buys per-direction
gain and spending it across more pays.

**Whether that is a price or a reward depends on the objective, and this ADR first claimed it too
strongly** ([#57](https://github.com/NGL321/patchworks/issues/57)). Miyato et al.
([arXiv:1802.05957](https://arxiv.org/abs/1802.05957), §3) run the identical algebra on the identical
budget — `σ₁² + σ₂² + ⋯ + σ_T² = d_o` — and prove that output gain in a fixed direction is *maximised*
when `σ₁ = √d_o` and every other singular value is zero, "which means that `W̄` is of rank one",
corresponding to "using only one feature"; they abandoned Frobenius-style normalisation for spectral on
exactly that ground, since "the spectral norm is independent of rank … allows the parameter matrix to
use as many features as possible." Their objective rewards output gain. Patchworks' divides it out — the
relative normaliser is the same invariance the section above rests on — so the incentive to concentrate
is weaker here. But *weaker* is all the algebra gives, and two pressures still point at concentration:
matching a neighbour on any **single** tick needs only the one direction that tick's stalk occupies, and
the composed **L1 on the normalised map is minimised, at fixed Frobenius norm, by the sparsest map** —
which is the pruning [`06-graph-topology.md`](../spec/06-graph-topology.md) asks for, working as
intended. What resists concentration is not the budget but the fact that a map must serve the whole
distribution of stalk states rather than one draw.

**No rank floor is imposed** — and with the pricing softened, that rests on wanting the outcome rather
than on the budget policing it. Learned rank-deficiency is wanted, not feared: it is
the mechanism `06-graph-topology.md` relies on to enlarge `H⁰` through a functionally dead but
structurally present edge. The degenerate limit — every edge transmitting one direction — is
instrumented rather than excluded.

### The instrument is a pair, because neither half separates the routes alone

- **Per-edge disagreement energy**, already available.
- **Per-edge effective rank**, the participation ratio `(Σσᵢ²)² / Σσᵢ⁴` of a map's singular values,
  reading 1 for a rank-1 map and `m` for a uniform one. One small SVD per map at `m ≈ 4`, on the
  diagnostic cadence, not per tick.

Read against the driven/quiescent contrast the quiescent hold already provides: energy falling **while
the world drives**, with effective rank sliding toward 1 across the fleet, is parameter collapse. Energy
falling **at rest** with effective rank steady is the lag floor draining, which is the hold working.

A map's **norm is not a diagnostic**, and the corrected drift direction is why: the upper face binds
continuously, so an interior edge's larger end reads `ρ` whether learning is healthy or frozen. Both
halves of the pair were chosen to be quantities that still move inside the gauge. The pair also carries
more weight than this ADR first admitted — with the Frobenius budget no longer claimed to price rank
concentration, effective rank is the only thing that says which regime the maps are actually in.

## Consequences

- **The reconciliation gain bound becomes provable, and #33's re-derivation retires.**
  `λ_max(Σ_e F_evᵀF_ev) ≤ Σ_e ‖F_ev‖_F²`, which the band bounds by `ρ² · deg(v)`. The denominator in
  [`02-tick-semantics.md`](../spec/02-tick-semantics.md) becomes `max(Σ_e m_e, ρ² · deg(v))` — at
  `ρ = 2, m = 4` the two are equal, so the gain does not move in practice, but it is written as the max
  so a later change to `ρ` cannot silently loosen `γ × floor < fold margin`. The per-cell spectral
  re-derivation [#33](https://github.com/NGL321/patchworks/issues/33) added to ADR-0007 tracked a drift
  that no longer happens and is **struck** rather than kept just-in-case.
- **A new contributor to the static floor.** With both ends bounded, an edge's representable scale ratio
  is `ρ²` times the `√m` range rank concentration affords. Genuine mismatch beyond that is irreducible
  and appears as static floor. ADR-0007's static-floor list is amended.
- **A cell's own metric space is its own in basis *and* scale** — but the scale that stays private is an
  **edge's ratio**, not a map's magnitude. `CONTEXT.md`'s *Node stalk* previously said only "whose basis
  its restriction maps fix", which a reader could satisfy with the exact gauge everywhere. The band
  exists precisely so scale stays private; since the joint scale rides the ceiling, the `ρ²` of ratio
  freedom is the whole of what it protects, and the previous bullet's `ρ²` scale-ratio figure is the
  load-bearing one rather than an incidental cost.
- **`ρ = 1` is the exact gauge.** If fixing scale turns out to be a beneficial special case rather than
  a restriction, adopting it is tightening a constant, not redesigning anything. Held open deliberately —
  and held open on this ADR's own terms, not on an analogy to attention. Under the only rigorous sheaf
  reading of attention (Hu, [arXiv:2601.21207](https://arxiv.org/abs/2601.21207), Thm 1) the constraint
  is `Σ_j w_ji = 1`, a **per-node budget across incident edges**, structurally Hansen & Ghrist's
  `tr(L_ii)` barrier rather than a per-map gauge; an individual map's norm there is free across its whole
  range. The general case for exact fixing does hold — van Laarhoven §5.5 (it decouples the effective
  learning rate from the penalty, which a band of width `ρ²` only bounds), Miyato's `σ(W) = 1` (an
  equality is what yields the Lipschitz bound; a band would not), and the unit sphere as the intrinsic
  domain of a scale-invariant parameter (Kodryan et al.,
  [arXiv:2209.03695](https://arxiv.org/abs/2209.03695), abstract only). If the per-node form is ever
  wanted, it is a *different* constraint from `ρ = 1` and would be reached for separately.
- **Over-smoothing is named for what it is here** in `01-cell-and-sheaf.md`'s *Known exposure*: the
  error signal vanishing, not a quality loss. Bodnar et al.'s result that a rich harmonic space resists
  collapse is cited as **orientation, not authority** — per `docs/research/015-sheaf-geometry.md` those
  are orthogonal-sheaf theorems, and the band does not put Patchworks in their regime.

## Alternatives considered

- **Orthogonality, Di Nino et al.'s own constraint.** Strictly stronger than what is adopted, and
  rejected on two grounds. It fixes the *basis* as well as the scale, which is precisely the work a
  restriction map exists to do (`01-cell-and-sheaf.md`, *Restriction maps*: transport and change of
  basis). And Patchworks' maps are masked and generally non-square, so orthogonality is not available
  in general without changing what the mask means. **The weaker constraint is known to suffice**
  ([#53](https://github.com/NGL321/patchworks/issues/53)): Hansen & Ghrist (ICASSP 2019), the paper
  Di Nino et al. generalise, exclude the trivial solution with a log barrier on `tr(L_ii)` — and since
  `L_ii = Σ_{e ∋ i} F_{i◁e}ᵀF_{i◁e}`, that quantity *is* `Σ_e ‖F_{i◁e}‖_F²`, a Frobenius floor on a
  node's incident restriction maps with no basis constraint anywhere. They relax **out of** the
  orthogonal class deliberately, "since convex combinations of orthogonal matrices are not orthogonal."
  Norm-without-orthogonality is the field's originating choice, not a Patchworks bet. (Their sparsity
  term runs the other way from Patchworks' — it counterbalances a tendency toward sparsity rather than
  creating one — so they support the floor and not the L1.)
- **A norm floor alone**, with no upper bound. Excludes collapse, but leaves the map magnitudes free to
  grow, so the Laplacian block's spectral radius grows with them and #33's staleness returns
  immediately. **A second and more fundamental reason arrived with the corrected drift direction:** under
  Lemma 2.4 the free magnitude grows by default, so the upper bound is the end that closes the failure
  the dynamics actually produce, and a floor alone would leave the vanishing effective learning rate
  wide open. The band costs one constant and closes the side that binds.
- **The exact gauge everywhere, interior included.** Simpler, one fewer constant, and a cleaner
  statement of the unidentified-parameter argument. Rejected because the scale-relative objective
  normalises by the restricted beliefs' magnitudes, so an edge whose ends carry different stalk scales
  reads near-maximal relative disagreement that pinned maps cannot correct; the pressure then lands on
  the *stalks*, and connectivity chains it into near-uniform stalk scale across the graph. That takes
  scale out of the cell's own metric space by a side effect, which is too large a thing to decide
  accidentally. **The corrected drift direction sharpens this rather than unsettling it:** since the joint
  scale rides the ceiling anyway, the band and the exact gauge differ in exactly one thing — whether an
  edge's ends may sit at different norms — which is the compensation this alternative removes.
- **Recording route one as *Known exposure* only** (R1's option b), with instrumentation and no
  mechanism. This is the posture ADR-0007 takes toward the disagreement floor, and it is the wrong one
  here for a specific reason: the floor leaves its own instrument intact, and this failure **erases the
  instrument that would reveal it**. Tolerating something requires being able to see it.
