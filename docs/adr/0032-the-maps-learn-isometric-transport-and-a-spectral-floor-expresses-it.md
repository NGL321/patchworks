# ADR-0032: The maps learn isometric transport, and a spectral floor expresses it

**Status:** accepted

## Context

Settled in [#411](https://github.com/NGL321/patchworks/issues/411), which
[ADR-0031](./0031-the-sparsity-pressure-is-deleted.md) handed the successor question by name:
deleting the sparsity pressure stopped the architecture spending what the sheaf wants and did not
purchase it. At `λ = 0` the fleet held effective rank **2.913** against a mask ceiling of 4 interior
and 8 boundary ([#393](https://github.com/NGL321/patchworks/issues/393)); at the shipped `0.4` it was
**1.0009** on all three seeds ([#237](https://github.com/NGL321/patchworks/issues/237)) while the
gauge was measured fully intact — maps pinned at Frobenius exactly 1.0000, banded within
`[0.6729, 2.0000]`.

**The gauge was never asleep. It was performing a rank-preserving operation on a collapsing
surface.** [ADR-0010](./0010-restriction-map-scale-is-gauge-fixed.md) bands `‖F‖_F` and says nothing
whatever about how that budget is spread across a map's singular values, and its *Frobenius, not
spectral — and therefore no rank floor* section says so deliberately.

The spread is measured, and it is far from flat.
[#416](https://github.com/NGL321/patchworks/issues/416) read per-map singular values in float64 on
the real dome: `σ_min/σ_max` runs a median **0.40–0.42** at 30,000 ticks and **0.55** at 100,000 —
rising, and nowhere near 1 — with a `|log₂|` p95 on `σ_min` of **17**, which is to say the spectra
are nowhere near flat by orders rather than by percentages.

This ADR records what the maps are for, and the constraint that expresses it. It is not a second copy
of #411's argument; the ruling is the record and this states the decision and its grounds.

## Decision

**The restriction maps are learning isometric transport of a free-abelian feature set, and the
constraint that expresses it is a per-map spectral floor at `σ_min ≥ ‖F‖_F/√m`, enforced by
projection.**

### The isometry is a property of the edge pair, not of one map

*A map is an isometry* is unavailable before any learning: an interior map is `4 × 32` and has a
28-dimensional kernel by arithmetic. The object is one level down. Cell `u` puts `F_u x_u` on the
shared lane and cell `v` reads it back into its own **carried subspace** as `F_v⁺F_u x_u`, so the
demand is that

> **`F_v⁺F_u` be an isometry between the two cells' carried subspaces.**

Per edge, and never per map. A session reaching for *a map is an isometry* has the level wrong.

**It decomposes into two halves.** Writing `F_u = U_uΣ_uV_uᵀ` and `F_v = U_vΣ_vV_vᵀ`, the composite in
the two `V` bases is `Σ_v⁻¹QΣ_u` with `Q = U_vᵀU_u` orthogonal, and edge-isometry is exactly
`Σ_v⁻¹QΣ_u ∈ O(m)`. That needs **flat spectra** — `Σ = σI` on each side — and **matched edge scale**,
`σ_u = σ_v`. This ADR's constraint buys the first. The second is ADR-0010's, is measured, and is
below under *Consequences*.

### Flat is forced, not merely sufficient

Matched *non-flat* spectra also satisfy one edge, if `Q` happens to pair equal singular values.
**That escape does not survive composition, and this graph is a channel.** Relay *through* a cell is
`F_v^{e₂}(F_v^{e₁})⁺ = Σ₂(V₂ᵀV₁)Σ₁⁻¹`, an isometry for non-flat `Σ` only if that cell's two incident
maps are aligned — and `GAUGE_C` pushes them deliberately apart (ADR-0010, *Incoherence is gauge-fixed
too*). **The architecture's existing incoherence cap and per-edge isometry are jointly satisfiable
only when the spectrum is flat**, because flatness is the state with no top direction left to
misalign. Rim-to-apex is seven hops; matched-pairwise does not chain, flat does.

### The constraint, and why it costs no invented constant

**Locus: the map, in `RestrictionMaps.project()`.** Local by ownership — a cell owns its incident maps
and [ADR-0011](./0011-the-locality-guarantee-is-enforced-not-inherited.md) is satisfied without
argument.

**Shape: a floor on the singular values, whose only derivable value is flatness itself.** Since
`Σᵢσᵢ² = ‖F‖²_F`, a floor at the RMS singular value `‖F‖_F/√m` forces `σᵢ = ‖F‖_F/√m` for every `i`,
with equality throughout. **The floor at the derivable value and the projection onto the nearest
scaled co-isometry are the same operation.** Any weaker floor needs a fraction, and the fraction would
be invented — so this satisfies [#127](https://github.com/NGL321/patchworks/issues/127)'s standing
note on invented constants **outright rather than waiving it**, the same standard ADR-0031 held itself
to in preferring *deleted* to *zeroed*.

**A projection and not a pressure, and the reason is
[#339](https://github.com/NGL321/patchworks/issues/339).** The transport term has no fixed point at
agreement and orbits at `~η` permanently. A pressure would be arguing with something that never
settles, and [#356](https://github.com/NGL321/patchworks/issues/356) already showed how that ends — a
coherent 12% bias beat a larger unsettled term over 30,000 steps. A projection holds. It is also
ADR-0010's own idiom, and [#394](https://github.com/NGL321/patchworks/issues/394) reached the same
shape independently from the citation side: VICReg's hinge is *"a projection with a floor instead of a
band"*, the cheapest transfer it found and the one remedy family surviving ADR-0011's locality.

**It is the operation the existing mechanism structurally cannot perform.**
`RestrictionMaps._push_apart` already water-fills a cell's Gram eigenvalues toward `c` — a flattener —
but the fill is gated on `live = eigenvalues > peak * 1e-9`, because scaling zero leaves zero. **So
the cap flattens survivors and can never resurrect a dead direction. A cap exists; a floor does not;
only a floor lifts a dead direction.**

### The target splits into a local half and a global half

Flat spectra plus matched edge scale make every edge composite an isometry, so every *path* composite
is one: **the holonomy lands in `O(m)` instead of `GL(m)`.** It does not land on the identity — a
rotation in `O(m)` preserves every length while permuting *which generator is which* along a route.

| | what it is | who can enforce it |
|---|---|---|
| **metric agreement** | lengths carried undistorted; holonomy ⊂ `O(m)` | **a cell, per map** — local |
| **identification agreement** | which generator is which, route-free; holonomy `= I` | **nobody local** — a cycle is not incident to one cell |

The identification half is not bought by a second constraint. It is bought by **the objective on a
non-degenerate trajectory**: the same environment drives the whole graph, and disagreement descent
pushes toward consistent identification on its own once the trajectory cannot dodge into a subspace.
[#315](https://github.com/NGL321/patchworks/issues/315) is the read that checks whether it did.

### The reframe, which is the ruling's real answer

Minimising `‖F_u x_u − F_v x_v‖` buys **agreement** — both cells land on the same point of the lane.
It does not buy the **metric**. So *the transport rule is already the isometry objective, sampled on
the trajectory*, and what was missing was never a second term. The degenerate optimum is then legible:
**two maps that carry only one direction agree perfectly on the states actually visited and are
unconstrained on the other three.** Rank-1 collapse is agreement achieved by shrinking what has to be
agreed about. The constraint's job is to keep all `m` directions live so the trajectory cannot dodge —
which is a floor, and not an objective.

### The compression is free abelian

`docs/motivating-image.md`'s *"compression must be non-abelian"* was a dictation artifact and is
corrected. Order-free composition means what arrives is determined by *which* generators travelled and
never by the route — that **is** path-independence, and Bodnar et al.'s Lemma 6 is the same sentence
from the other side: `dim H⁰ = d` **iff** transport is path-independent. So free abelian does not sit
politely above the transport layer; it is the demand that holonomy be trivial, and **departure of
holonomy from the identity is the failure measure, not the prize**.

Heterogeneity survives the correction and changes its source: different cells hold **different subsets
of the generators** — different masks, different degree, different `m_e`. The compression is
heterogeneous because the slices differ, and free abelian is what allows overlapping slices to be
reassembled at all, which is what a sheaf's gluing *is*. The motivating image keeps its conclusion —
heterogeneous connections, therefore a sheaf over a graph — and loses only its middle term.

The exponential-versus-logarithmic coverage claim from the compression paper is recorded as **listed,
unattributed influence** and nothing here rests on it; if it is ever to be load-bearing that is a
`/research` ticket.

## The mask-attainability read, taken here rather than pre-registered

ADR-0010's second ground against orthogonality survives its first (below, and on that ADR): **a mask
may not contain a co-isometry.** That is a measurement, and the ADR may not claim the constraint is
attainable without it. Read against `build_graph(DEFAULT_SPEC)`:

| population | endpoints | masks containing a scaled co-isometry |
|---|---|---|
| **banded** (a predicting cell's own maps) | 1091 | **1091 — all of them** |
| **pinned** (a boundary cell's own maps) | 273 | 264 |

Every banded mask permits at least **four times** as many node stalk directions as its lane is wide —
`(m, k)` runs `(4, 32)`, `(4, 24)`, `(8, 32)`, `(4, 28)`, `(4, 20)`, `(4, 17)` and `(1, 17)`, a
minimum margin of `k − m = 13`. On those masks the projection lands **exactly** flat: maximum relative
departure from flat after one projection is `1.3e-15`, and `‖F‖_F` moves by `1.1e-15`. **The
constraint and ADR-0010's gauge do not interact on the maps the projection reaches.**

**The nine that cannot are all pinned, and they are the small-stalk end of the sensorimotor rim.**
Three touch cells (`m = 8`, `k = 1`), three proprioceptive (`m = 8`, `k = 2`) and the actuator's three
maps (`m = 8`, `k = 6`). There `rank(F) ≤ k < m` by construction, so `σ_min = 0` whatever the
projection does; the best attainable `σ_min/flat` is **0.87** at the actuator, **0.50**
proprioceptive, **0.35** touch. On those masks the projection also **shrinks `‖F‖_F` by `√(k/m)`** —
13% to 65% — so it would fight the exact gauge rather than sit beside it.

**This is the same population ADR-0010 already excludes from the projection's reach**, and for the
same reason: a pinned map has no scale freedom to spend. The floor is therefore stated on the banded
maps, where it is attainable everywhere and free, and the nine are a **named exclusion** rather than a
silent one. At `m = 1` — the drive's eight edges — the floor is vacuous, since one singular value is
`‖F‖_F/√1` identically.

## What this costs, priced here rather than discovered later

**The projection preserves `‖F‖_F` and therefore moves `σ_max`.** Setting every `σᵢ = ‖F‖_F/√m` makes
the Frobenius norm exactly preserved, which is why ADR-0010's band is untouched. The **operator** norm
is not:

| | `σ_max` |
|---|---|
| today, trained (effective rank 1.0009, #237) | `≈ ‖F‖_F` |
| after the projection | `‖F‖_F/√m` |

[ADR-0022](./0022-a-hop-is-an-operator-norm-along-a-learned-channel.md) defines a hop as an operator
norm along a learned channel, so that is **`√4 = 2x` down per interior hop**, `√8 = 2.83x` at a
boundary map, and `2⁷ = 128x` across the rim-to-apex seven.

**What is not affected, stated because the naive reading over-charges this.** The isotropic /
Frobenius reading depends only on `‖F‖_F` and is unchanged. What the projection buys is `m − 1`
additional *surviving* directions per map, where today there is one. So the trade is exactly:

> **`√m` of gain on the single best direction, against `m − 1` directions that currently transmit
> nothing at all.**

That is the kernel-versus-rank trade #394 was opened to find priced, and the honest position is that
the shape was ruled without paying for it. The counterweight is #237's reading — sheaf effective
resistance along the channel is **2.8e5x–4.4e5x** the graph's at effective rank 1.0009 against
**5.3x** at 2.85 — so the benefit is plausibly three to four orders against a 128x cost.
**Plausibly is not priced**, and the two are in different units (a resistance ratio against a gain
product), so the comparison is made properly by the read below or not claimed at all.

**And the subtler one.** [#142](https://github.com/NGL321/patchworks/issues/142)'s correction —
*"read along the channel, the hop is ~184x what an isotropic probe reports"* — is a
channel-versus-isotropic ratio measured against near-rank-1 maps, and **that ratio shrinks as the
spectrum flattens, by construction**: a flat map has no preferred direction for the channel to ride.
The isotropic baseline does not move, so the ~1e14 phantom deficit #142 struck does not come back —
but the sentence explaining why it went away stops being true of the post-edit surface, and
[#240](https://github.com/NGL321/patchworks/issues/240)'s gate is what that is for. Flagged, not
amended: the sentence is true of today's surface and stays true until the constraint is built and
read.

## Pre-registrations

Three, and the first is discharged above.

1. **Mask attainability** — taken here. Attainable on all 1091 banded masks; nine pinned masks
   excluded by name.
2. **Falsification, at a horizon longer than 30k.** Flat maps must move effective rank toward `m`
   **and** must not cost the incoherence cap. ADR-0010 pre-registered that a cross-edge alignment
   pressure and `c` pull the same maps in opposite directions, and this is that collision arriving
   from the other side. The horizon is stated because
   [#178](https://github.com/NGL321/patchworks/issues/178) has cost this map the 30k mistake three
   times, most recently on #416, where the interior scale ratio's meaning reversed between 30k and
   100k.
3. **Per-hop gain along the channel, before and after, on the same rig — per edge and per direction**,
   never a graph-wide average (#127's standing rule, and #181's per-edge-not-per-level form). This is
   what makes the 128x-against-three-orders comparison payable rather than plausible.

## Consequences

**ADR-0010 is amended twice, and the two amendments are different in kind.** *Alternatives considered*
rejected orthogonality on a ground that does not hold, and that sentence is what has kept this
constraint out since [#37](https://github.com/NGL321/patchworks/issues/37); its second ground survives
and is the mask-attainability read above. Separately, ADR-0010's free **scale ratio** is recorded as a
second hole, now with #416's measurement against it. Both amendments live on that ADR.

**The edge scale ratio becomes exactly the norm ratio.** Under a flat spectrum an edge's two ends share
one `m`, so `σ_u/σ_v = ‖F_u‖_F/‖F_v‖_F` **identically** — the Frobenius/`σ` identification #416 called
an idealisation today becomes an identity. That makes #416's finding sharper rather than softer: the
`2x` unmatched endpoint scale it measured on all 273 boundary-incident edges *is* the composite's
isotropic distortion, not a proxy for it. [#429](https://github.com/NGL321/patchworks/issues/429) owns
the remedy and this ADR rules nothing about it.

**This is the first thing in the record that could bound a composed product below**
([#423](https://github.com/NGL321/patchworks/issues/423)). ADR-0010 bounds a Frobenius norm, which
cannot bound `σ_min` away from zero, and the structural masks make maps rank-deficient by
construction; flat maps fix `σᵢ = ‖F‖_F/√m` and so floor each map's `σ_min`. **It is explicitly not
sufficient alone**, and that half travels with it: a composed lower bound needs the **alignment of
adjacent cells' carried subspaces**, and two perfectly flat maps whose carried subspaces are
orthogonal still compose to zero. That term is what #315 reads. A two-sided composed bound is
therefore *derivable after this is built and conditional on a measured alignment floor from #315* —
pre-registered, not ruled. An ADR claiming this decision buys composed behaviour would be
overclaiming. [#332](https://github.com/NGL321/patchworks/issues/332) closed `solved` on
[#427](https://github.com/NGL321/patchworks/issues/427) while this was being written, and it closed on
the **upper** side — its *expanding* half forbidden by `σ_max(composed) ≤ ρ^{2h}`, its *contracting*
half reassigned to the driven field by [#144](https://github.com/NGL321/patchworks/issues/144), and
neither by anything here. This ADR supplies the ingredient the **lower** side lacked and does not
close it.

**The floor does not transfer to `K`, and the want inverts.** Ruled on
[#420](https://github.com/NGL321/patchworks/issues/420) §3, and stated here because the two spectra
sit one sentence apart in the record:

> The map spectrum and the cell-operator spectrum take opposite constraints for the same reason. A
> restriction map **transports** — it must not distort what it carries, so its singular values are
> floored flat. A cell operator **computes** — non-normal transient growth is how a linear system
> moves content within a piece, so flatness on `K` is a starting condition to be escaped rather than a
> target to be enforced. Same mechanism (a projection, local by ownership), same standard on invented
> constants, **opposite sign on flatness** — and a session reaching for *the spectral floor applies to
> `K` too* has the level wrong, in the same way this ADR corrects *a map is an isometry*.

Locality by ownership and the arithmetic both transfer cleanly:
[ADR-0015](./0015-the-cell-operator-band-is-on-the-spectral-norm.md) says a cell *"owns its own `K`
outright"*, and `σ_min(K) ≥ ‖K‖_F/√k` forces `Σ = σI` by the identical argument. It is the *want* that
inverts. [#357](https://github.com/NGL321/patchworks/issues/357) — whether the flat within-cell
spectrum must break — is untouched and unanswered by this ADR, and ADR-0015's band stands.

**[#396](https://github.com/NGL321/patchworks/issues/396) is superseded by the local/global split, not
declined.** Its target is right — *buy `H⁰` by aligning incident maps around cycles, leaving each map
full rank* is the direct statement of free abelian. Its lever is unavailable: the locality objection it
filed against itself is upheld, and the half it reaches for is the half no cell can compute. Its stated
lower-ceiling cost also dissolves, because ADR-0031 already ruled `H⁰` a floor rather than a maximand.

**[#315](https://github.com/NGL321/patchworks/issues/315) is promoted to build, with its sign
flipped.** Loops are still what make the reading possible; departure of holonomy from the identity is
now a **defect measure**, not a prize. The instrument is unchanged.

**The `H⁰` bound tightens, and that makes an open problem worse rather than better.** `dim H⁰ ≥ Σ_v
max(0, n − Σ_e m_e)` is a *lower* bound, and both `05-timescales.md` and `06-graph-topology.md` lean on
the slack: *"zero guaranteed private dimension is not zero private dimension … learned rank-deficiency
enlarges `H⁰` past it."* A floored map has rank exactly `m`, so **the per-map source of that slack
closes.** What excess can remain comes from misalignment across a cell's incident maps rather than
from any one map being deficient — and `GAUGE_C` pushes the other way, since incoherence is what makes
the stacked operator's rank *higher*. The honest statement is that privacy near the rim stops being
contingent on learning and becomes what construction says it is, which for the 82 of 150 predicting
cells with `Σ_e m_e ≥ n` is **zero** rather than contingently-nonzero.
[#385](https://github.com/NGL321/patchworks/issues/385) owns that floor and this decision hands it a
harder version of its own question; [#330](https://github.com/NGL321/patchworks/issues/330) reads on
the same quantity. Neither is ruled here.

**[#324](https://github.com/NGL321/patchworks/issues/324) is not closed and not amended.** Its bar is
`draining_effective_rank < 2`; this decision is aimed at it and purchases nothing until the constraint
is built and the long-horizon read returns.

**The gap to the sheaf-diffusion literature narrows and does not close.** `01-cell-and-sheaf.md`'s
*Known exposure* discounts Bodnar et al.'s collapse-resistance result because it is proved for
*orthogonal* sheaves and Patchworks' maps are *"masked, learned, and merely norm-bounded"*. A floored
map is a **scaled co-isometry** — nearer that class than norm-bounded ever was, and still not it,
since the maps stay masked and non-square. The theorems still do not reach this sheaf and the paired
instrument still carries the argument; what changes is that the counterweight now points at something
the design is deliberately steering at.

**Succession from #356, stated so the record does not read as a reversal.** #356 ruled *"nothing new
should police rank upward, and the remedy space this ticket opened is closed."* This does exactly what
that forbids, and it is **spent, not overturned**: #356's stated ground was that the question is *"not
whether to police rank but where `λ` sits, which is a retune of an existing constant rather than an
invented mechanism"* — admissible because a dial existed to carry the decision. ADR-0031 deleted the
dial and hands the successor question here by name. The premise expired; the reasoning did not fail.

**The constraint is specified here and not yet built.** `RestrictionMaps.project()` does not enforce
it today. Three mechanics are open and belong to the build rather than to this decision: where the
flattening sits relative to `_push_apart` (which water-fills eigenvalues and so un-flattens what this
flattens — pre-registration 2 is the read on that collision), how the ragged `m` across edges batches,
and the nine named pinned exclusions.

## Alternatives considered

- **A hinge floor at a fraction of flat.** The natural VICReg transfer, and rejected because the
  fraction is invented where `‖F‖_F/√m` is derived. Taking it would have re-run exactly the failure
  ADR-0031 named — a constant with a magnitude warrant and no purpose.
- **`λ = 0` and nothing further.** #393's **2.913** against a mask ceiling of 4 is the live case, and
  it fails on what it measures: it was read under the collapser's *absence*, not under any pressure
  toward flatness, and at 30k. It is evidence that deletion stopped the bleeding, which ADR-0031
  already states, and no evidence about the ceiling.
- **Enforcing path-independence directly.** No cell can compute it — a cycle is not incident to one
  cell. Not dismissed for the project; ruled unavailable to a *local* rule, which is a different thing,
  and it is why #315 becomes the read.
- **The same floor on `K`.** Arithmetically available and locally enforceable, and rejected on the
  want rather than on the mechanism — see *Consequences*.

**Ceded:** the mechanism was supplied in-session and the ruling's own provenance block marks it — that
flatness sets `Σ = σI` while leaving `U ∈ O(m)` and `V ∈ V_m(Rⁿ)` free, at a cost of 3 degrees of
freedom of 128 per interior map, and that a floor at the RMS singular value forces exact flatness
rather than approximating it. Both are facts about a masked non-square map's decomposition rather than
about the architecture. The want — isometric transport, and the level it sits at — is the user's.
