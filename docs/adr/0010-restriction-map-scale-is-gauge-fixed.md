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
feasible restriction maps `F`, to avoid the trivial solution."* Their constraint is orthogonality,
solved as an orthogonal Procrustes problem. Every other learned-sheaf method constrains its maps by
construction (Bodnar: invertible classes; Barbero: `O(d)`). Patchworks imposed no orthogonality,
invertibility, norm, or rank floor: the structural mask constrains *which* entries may be nonzero and
requires none of them to be.

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
the edge's own scale** does, because the ratio is invariant under `F ↦ αF`: shrinking the maps buys it
nothing.

The normaliser is **locally stateless** — disagreement relative to `‖F_u x_u‖ + ‖F_v x_v‖`, the
restricted beliefs' own current magnitudes — and not a running average of the edge's recent scale. This
is the same criterion form, adopted for the same reason, as the change gate's threshold in
[`05-timescales.md`](../spec/05-timescales.md): a per-edge running average is an auxiliary variable with
a hand-set time constant, which is the object ADR-0007 rejects under *A per-edge learned baseline*.

The sparsity pressure composed into the same step is **L1 on the normalised map**, which redistributes
weight across the map's directions rather than removing it. This is what
[`06-graph-topology.md`](../spec/06-graph-topology.md)'s "prunes *within* the mask; does not shrink the
stalk" always meant, now stated mechanically.

Both terms are therefore blind to the map's overall magnitude. **Nothing in the transport rule has an
opinion about it**, which means the magnitude is not merely unconstrained but *unidentified by the
objective*. This is the whole argument for what follows: fixing an unidentified parameter removes a free
parameter, it does not cap a learned one. Left free, the magnitude does not sit still — it drifts, and
the direction it drifts under any residual asymmetry is the one route one names.

### Interior maps carry a band; boundary maps carry the exact gauge

- **Interior restriction maps**: `‖F‖_F ∈ [1/ρ, ρ]`, with `ρ = 2` a **construction-time constant**.
  Nothing reads it at runtime, no cell estimates it, and it never anneals.
- **Boundary-cell restriction maps**: exactly `‖F‖_F = 1`. A boundary cell *runs no body*
  ([`04-action-and-the-boundary.md`](../spec/04-action-and-the-boundary.md)) and its stalk is
  world-shaped by [ADR-0006](./0006-boundary-cell-stalks-are-world-shaped.md), so it holds no metric
  individuality that a band would protect. Where world units and stalk units disagree, the environment
  contract owns the conversion, not the sheaf.

Enforcement is **projection**: take the transport step, then project the map back into the band. Inside
the band the projection is inert, so it is a guardrail rather than a continuous force — the same
posture ADR-0007 takes toward the floor. Reparameterising as `F = G/‖G‖_F` was rejected: it costs no
more, but it leaves a shadow parameter `G` that the sparsity term can drive toward zero, reintroducing
collapse as numerical ill-conditioning in a parameter no diagnostic watches.

`F = 0` is now **unrepresentable**, not merely disfavoured.

### Frobenius, not spectral — and therefore no rank floor

Unit **spectral** norm pins only the largest singular value; every other direction may shrink to zero at
no cost, so it excludes `F = 0` while leaving `F → rank 1` wide open. Unit **Frobenius** norm pins the
sum of squares across all directions, so concentrating a map onto fewer directions *buys* per-direction
gain and spending it across more *pays*. Rank concentration becomes a priced trade rather than a free
lunch.

That pricing is why **no rank floor is imposed**. Learned rank-deficiency is wanted, not feared: it is
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
- **A cell's own metric space is its own in basis *and* scale.** `CONTEXT.md`'s *Node stalk* previously
  said only "whose basis its restriction maps fix", which a reader could satisfy with the exact gauge
  everywhere. The band exists precisely so scale stays private; it is bounded by construction, not
  pinned.
- **`ρ = 1` is the exact gauge.** If fixing scale turns out to be a beneficial special case rather than
  a restriction, adopting it is tightening a constant, not redesigning anything. Held open deliberately.
- **Over-smoothing is named for what it is here** in `01-cell-and-sheaf.md`'s *Known exposure*: the
  error signal vanishing, not a quality loss. Bodnar et al.'s result that a rich harmonic space resists
  collapse is cited as **orientation, not authority** — per `docs/research/015-sheaf-geometry.md` those
  are orthogonal-sheaf theorems, and the band does not put Patchworks in their regime.

## Alternatives considered

- **Orthogonality, Di Nino et al.'s own constraint.** Strictly stronger than what is adopted, and
  rejected on two grounds. It fixes the *basis* as well as the scale, which is precisely the work a
  restriction map exists to do (`01-cell-and-sheaf.md`, *Restriction maps*: transport and change of
  basis). And Patchworks' maps are masked and generally non-square, so orthogonality is not available
  in general without changing what the mask means. Whether the weaker constraint is known to suffice is
  the open question handed to [#53](https://github.com/NGL321/patchworks/issues/53).
- **A norm floor alone**, with no upper bound. Excludes collapse, but leaves the map magnitudes free to
  grow, so the Laplacian block's spectral radius grows with them and #33's staleness returns
  immediately. The band costs one constant and closes both sides.
- **The exact gauge everywhere, interior included.** Simpler, one fewer constant, and a cleaner
  statement of the unidentified-parameter argument. Rejected because the scale-relative objective
  normalises by the restricted beliefs' magnitudes, so an edge whose ends carry different stalk scales
  reads near-maximal relative disagreement that pinned maps cannot correct; the pressure then lands on
  the *stalks*, and connectivity chains it into near-uniform stalk scale across the graph. That takes
  scale out of the cell's own metric space by a side effect, which is too large a thing to decide
  accidentally.
- **Recording route one as *Known exposure* only** (R1's option b), with instrumentation and no
  mechanism. This is the posture ADR-0007 takes toward the disagreement floor, and it is the wrong one
  here for a specific reason: the floor leaves its own instrument intact, and this failure **erases the
  instrument that would reveal it**. Tolerating something requires being able to see it.
