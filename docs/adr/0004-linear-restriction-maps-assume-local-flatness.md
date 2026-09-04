---
status: accepted
---

# Linear restriction maps assume local flatness

Restriction maps are **linear**, with all nonlinearity inside the cell
([`01-cell-and-sheaf.md`](../spec/01-cell-and-sheaf.md)). That was adopted for what it buys —
a genuine sheaf Laplacian, disagreement as Dirichlet energy, reconciliation that is one cheap
step rather than a nested optimisation. This ADR records the thing it *costs*, which the spec
had not stated: linearity is not an efficiency choice, it is a **geometric commitment**.

A linear restriction map is honest exactly insofar as the latent structure the two endpoint cells
model in common is **locally Euclidean at the scale of the overlap**. Where it is not, the linear
map is a first-order approximation, and the part it cannot follow arrives at the edge as
disagreement indistinguishable from any other.

This is the load-bearing use of the word *manifold* in Patchworks, and the only one — **demoted by [#440](https://github.com/NGL321/patchworks/issues/440) to the *continuous* case's warrant; see *One want, two warrants* below.** The world
is not claimed to be a manifold and does not need to be; the **pieces** are. Each cell owns a
`k`-dimensional, locally Euclidean piece of the problem — its chart is a chart of *that piece*,
in the strict sense — and the sheaf glues pieces of differing dimension without ever requiring
their union to be a manifold. A cellular sheaf is the atlas idea with the constant-dimension
requirement removed, which is why it is the right formalism here and an atlas is not.

`k < n` follows from this rather than being asserted alongside it: **`k` is the dimension of the
piece; `n` is the room needed to talk about it with neighbours.**

**That is the only reading of `k` this design has.** The Koopman conversion appeared to create a
second one — a lift dimension, chosen to make dynamics linear, which the literature puts one to two
orders of magnitude *above* the state rather than below it. It does not:
[ADR-0023](./0023-the-chart-is-not-a-koopman-lift.md) records that the chart persists and `K` is a
linear recurrence rather than a lift, so this sentence stands unamended and unshared. Nothing here
changes; what changes is that nothing else may borrow `k`.

## One want, two warrants

*Amended by [#440](https://github.com/NGL321/patchworks/issues/440).*

This ADR opens by stating a **want** and one **warrant** for it, written as though the warrant were
the want. They are separated here, because the language domain's heard rim
has a **discrete** overlap and the manifold sentence is **inapplicable in its stated form** there —
not vacuous, not violated, and not satisfied, with the want it encodes fully intact.

- **The want.** A linear restriction map is honest exactly insofar as the relation between the two
  endpoint cells' representations of their shared content is realised by a linear map **without
  loss**.
- **Continuous overlap → local flatness.** *Locally Euclidean at the scale of the overlap* is a
  **licence to be first-order**: shrink the patch and a fixed linear rule becomes exact with an
  error term going to zero. **Curvature is the error** — Singer & Wu's mechanism, unchanged — and
  the disagreement floor is its signature.
- **Discrete overlap → injectivity on the situation set, plus isometry of the carried subspaces.**
  There is no limit and no shrinking error, and the underlying question is still sharp: two
  situations either collide or they do not, and distances either survive or they do not.
  [ADR-0032](./0032-the-maps-learn-isometric-transport-and-a-spectral-floor-expresses-it.md)
  already gave that question its statement — `F_v⁺F_u` an isometry between the two cells' **carried
  subspaces**, expressed as a per-map spectral floor, which is a condition on the maps, true or
  false over any point set at all, with no manifold anywhere in it. **Collision is the error**, not
  curvature.

So `manifold` stops being *"the load-bearing use in Patchworks, and the only one"* and becomes **the
continuous case's warrant**. The sentence is demoted, not struck: it still says exactly what it says
wherever the overlap is continuous, which is the whole of the dome.

**The four load-bearing uses of linearity are untouched, and this is stated rather than left to be
inferred** — `L = δᵀδ`, the `E_PC(s) = ½‖δ⁰s‖²` identity, the coupling between pieces, and the
unabsorbable constant offset. None of them ever needed a manifold, so the amendment reaches none of
them. Nor does it reach `k < n`, which keeps one reading and now has two derivations.

**A criterion with a case split is not a domain-specific justification.** The want is one and both
warrants are stated in domain-general terms, so *the same cells run both domains* stands unamended
and the dome remains the only thing on the domain-specific side. The warrant structure is general;
the clause inside it is selected by the overlap's geometry.

### What the sweep reads, and what the chart occupies

[#132](https://github.com/NGL321/patchworks/issues/132) read a language L1 wedge cell by
**configuration sweep** — this ADR's own procedure — and found a finite set: a `97⁴` ceiling, four
pairwise distances `sqrt(2j)` carrying all the mass, and `C(r)` a staircase with no box-counting
dimension. That is a reading of the **situation set**, the configurations a cell must tell apart,
and **not** of the **piece**. The chart **persists**
([ADR-0023](./0023-the-chart-is-not-a-koopman-lift.md)), so the set the chart occupies is the limit
set of a discretely-driven recurrence — bounded under the operator band, filling roughly
`log 97 / log(1/r)` dimensions at retention `r` — and is **neither a finite set nor a manifold**,
only fat enough to be locally Euclidean at high retention. The two terms are separated in
[`CONTEXT.md`](../../CONTEXT.md). In the dome the distinction is inert, because the situation set is
already continuous and reads `d_corr` 1.43; in language it is the whole question, because the sweep
there reads the **drive** rather than the state.

**Language is a mixed domain, and the discreteness reaches one column of it.**
[`11-the-language-graph.md`](../spec/11-the-language-graph.md)'s stalk table gives **heard** 97
dimensions of one-hot, while **spoken** carries a continuous coherence readback
([ADR-0025](./0025-coherence-is-a-motor-readback-not-a-sensory-value.md)) and the drive stalk is a
continuous valence scalar. #132's finding is heard-side, so the criterion is fully live in its
continuous form on the spoken column and on the drive. The design position this rests on is
[`12-the-interlocutor.md`](../spec/12-the-interlocutor.md)'s.

## What this ADR does not claim

*Added by [#141](https://github.com/NGL321/patchworks/issues/141).*

Three distinct claims in this design are reachable by the word *linear*, and this ADR makes exactly
one of them. They are separated in [`CONTEXT.md`](../../CONTEXT.md) as **local flatness** (this ADR —
the geometry of an overlap), **chart linearity** (`K` — time-evolution), and **readout gauge** (`D` —
observability). They are independent in every direction: a piece may evolve linearly in a chart whose
overlap with a neighbour is curved, or sit on a flat overlap while its stalk depends nonlinearly on
its chart.

**The Koopman conversion addresses chart linearity and does nothing for local flatness.** A curved
overlap stays curved, and no property of `K` bears on it. This is recorded here because this is
precisely where a future reader will come looking for permission to assume otherwise.

## Consequences

- **The commitment is falsifiable, and the test is free.** Persistent, structured, irreducible
  disagreement on a particular edge — disagreement that does not fall with learning and is not
  noise-shaped — is the signature of curvature the linear map cannot follow. No instrument needs
  building; the quantity is already computed every tick.

  **Amended by [ADR-0007](./0007-the-disagreement-floor-is-tolerated-not-represented.md): the
  signature has a second cause and is no longer readable on its own.** Timescale separation
  ([ADR-0005](./0005-timescale-is-persistence-not-a-schedule.md)) leaves a **lag floor** with the same
  surface appearance. The quantity is still free; the *reading* now costs two comparisons first —
  `06-graph-topology.md`'s topology-only baseline, then a **quiescent hold** (hold the world still,
  sweep configurations; lag drains, curvature does not). Correlating the residual with the neighbour's
  rate of change does *not* separate them: approximation error grows with displacement along the
  overlap, so both move together.

  **Further amended by [#49](https://github.com/NGL321/patchworks/issues/49): a third cause, read
  forwards rather than backwards.** A stalk too narrow to embed the piece it carries produces
  **self-intersection** — two distinct situations at the same coordinates — and it surfaces as
  persistent structured disagreement exactly like curvature does
  ([ADR-0007](./0007-the-disagreement-floor-is-tolerated-not-represented.md) carries it as a static-floor
  source; this clause carries the reading). It has a distinguishing test the other two lack — widen the
  stalk and the residual falls — but that test is **not available on a running system**: `n`, `k` and `m`
  are fixed at construction and the body is shared and frozen, so "widen" means rebuilding the graph and
  training again. It is a comparison between two builds, not a diagnostic, and it is named here so nobody
  reaches for it mid-run.

  **So this cause is read forwards.** Its criterion is known before anything runs — an embedding is
  generic once the coordinate count exceeds twice the piece's box-counting dimension — which makes
  self-intersection **predictable at construction** rather than diagnosable afterwards, the same shape as
  every other check in this design (`γ × floor <` fold margin, `dim H⁰ ≥ …`, `χ`). The practical
  consequence is that a build satisfying the criterion may *drop* this cause from the reading and
  disambiguate the remaining two by quiescent hold, and a build that does not satisfy it — which
  `06-graph-topology.md` flags as possible at `m = 4` — knows *which edges* to suspect in advance.

  **Further amended by [#138](https://github.com/NGL321/patchworks/issues/138): a fourth cause, which
  arrives with the gauge.** Freezing `decode` ([ADR-0014](./0014-the-linear-readout-is-gauge-fixed.md))
  confines every cell's predictions to `im(D)` — one fixed `k`-dimensional affine subspace of an
  `n`-dimensional stalk, and the *same* subspace in every cell. Wherever a stalk's real content lies
  outside it, the residue is persistent, structured, and arrives at the edge indistinguishable from
  curvature.

  **It has a distinguishing signature the other three lack, and it is cheap.** This cause is *shared
  across cells and fixed at construction*, so it shows as a **common direction in the residual across
  unrelated edges** — where curvature and self-intersection are per-edge and the lag floor is
  per-level. Nothing needs building: the direction is `colspan(D)`, known before the graph runs.
  **Read this one first.** Ruling it out costs one projection and restores the three-cause procedure
  above; failing to rule it out means the gauge is wrong, which is ADR-0014's own pre-registered
  falsification and not a fact about any edge.

  **Further amended by [#440](https://github.com/NGL321/patchworks/issues/440): all four causes keep
  their structure over a discrete overlap, and curvature's applicability becomes a
  measurement.** Read in the order the procedure above reads them:

  - **`colspan(D)`** — unchanged, still read first. It is shared across cells and fixed at
    construction, and it references no geometry at all.
  - **The lag floor** — unchanged in principle. The **quiescent hold is available** in language
    because `idle` is a symbol in the 97-alphabet, so holding the world still is something the
    interlocutor can do. **One instrument detail is unsettled, and is recorded as open rather than
    invented here**: the hold's second half is *sweep configurations while held*, and what a sweep
    is on the **heard** column with the world silent is not obvious. It belongs to stage 5's demo
    work.
  - **Self-intersection** — **transfers free and needs no amendment.** #49's clause above is already
    written as *twice the piece's **box-counting** dimension*, not twice a manifold dimension, and
    box dimension exists for a dust, a filled set, and everything between; the criterion came from
    delay embedding, where fractal attractors are the ordinary case. The ADR's newest clause is the
    one that survives the transfer intact.
  - **Curvature** — **conditional, and this is the finding.** Curvature is a property of something
    locally Euclidean, and whether a language cell's piece is depends on how fat its limit set is.

  Both live questions hinge on the **same number**, so one read settles them.
  **Pre-registered, and not taken:** the box / correlation dimension `d_box` of a **heard L1 cell's
  driven chart limit set**, read once stage 5 trains. `d_box` near 12 → the piece fills, curvature
  keeps its referent, and language reads exactly like the dome. `d_box` low → the piece is a dust,
  curvature is **struck in language** and its mass moves to **collision**, and `n > 2·d_box` is
  comfortably satisfied so self-intersection **drops** from the reading — the same *drop this cause*
  move this ADR already licenses above. It is also the number
  [`docs/research/032-dimensioning-small-predictors.md`](../research/032-dimensioning-small-predictors.md)'s
  capacities are quoted against, so it carries #132's `m = 4` margin (~1.4x, the width with the
  least margin) into language.

- **Linearity is load-bearing a fourth time: it is the whole of the coupling between pieces.**
  *Added by [#141](https://github.com/NGL321/patchworks/issues/141).* Decomposition creates boundary
  conditions — a cell's piece is not closed, and evolves depending on things outside it. The coupling
  that carries that dependence is restrict, average, correct, and it is **linear by construction and
  always was**; it cannot absorb nonlinearity, and nothing removed from inside a piece can hide there.
  After the Koopman conversion the entire cell → edge → cell loop is linear with exactly one
  exception, `encode` — **which is frozen**. So if the coupling between adjacent pieces is genuinely
  nonlinear in the world, the only element of the architecture that could model it is the one element
  nobody trains. The failure appears at the edge rather than inside any cell, and its signature is the
  same static floor this ADR already knows how to read.

  This is the same fact read from the transmission side rather than the injectivity side: `encode`
  being the sole nonlinearity is not only a risk to whether a chart can separate distinct situations,
  it is a risk to whether cross-piece coupling can be modelled at all.
- **Linearity is what keeps this test readable.** A linear map has no constant term, so it cannot
  absorb a persistent offset in a communication lane — and a curvature residual's constant part is
  exactly such an offset. Affine restriction maps would launder the signature away into a learned
  offset while the geometry stayed just as bent. This is a better argument for linearity than
  efficiency, and it was not visible until the floor was named.
- **Linearity is also what makes disagreement and prediction error the same quantity.** Recorded here
  because this is where a future reader comes looking for permission to bend the maps. The
  identification of predictive-coding error with the sheaf coboundary — the result the whole
  single-error-signal design rests on — is **derived for linear networks**; the energy identity
  `E_PC(s) = ½‖δ⁰s‖²` depends on it. Under nonlinear restriction maps that identification stops being
  **true**, not merely harder to compute, and Patchworks would be back to two objects needing to be
  related rather than one quantity read two ways. So linearity is load-bearing three times over: the
  formalism, the geometry recorded above, and this. See
  `docs/research/016-cell-contract-citations.md`.
- **The escape hatch is not "make restriction maps nonlinear."** That would destroy `L = δᵀδ`,
  and with it the Laplacian, the Dirichlet-energy reading of disagreement, and one-step
  reconciliation. The literature leaves nonlinear coboundaries essentially unexplored for exactly
  this reason. If an edge's overlap is genuinely curved, the response is to change what that edge
  carries — the structural mask — not to bend the map.
- **Curvature and disagreement are not separable by inspection.** The architecture has one error
  signal and cannot, on its own, tell a cell that is wrong from an edge that is bent. Diagnosis is
  offline and comparative.

## What the literature does and does not give

Support is **componentwise, and the conclusion is this project's own** — no source states the
argument above.

- Singer & Wu (*Vector diffusion maps*) state that transport-approximation error **is** curvature:
  the subspaces being related "are usually not exactly the same, due to curvature", which surfaces
  as excess rank the low-dimensional map must discard. That is the mechanism this ADR relies on.
- Bodnar et al. (*Neural Sheaf Diffusion*), Lemma 6: flatness / path-independence ⟺ maximal `H⁰`.
  The relation between flatness and the sheaf's structure is real and stated.
- **Do not lean on "the literature shows curvature matters."** A 2026 intervention study finds
  holonomy is *not* what drives sheaf-network performance. The geometric reading here is a design
  commitment with a falsification test, not a result being inherited.
- Patchworks sits **outside the hypothesis class of every learned-sheaf paper**, all of which assume
  invertible or orthogonal restriction maps. Ours are masked, sparse and deliberately
  rank-deficient. Their theorems are orientation, never authority.

See [patchworks#15](https://github.com/NGL321/patchworks/issues/15) and
`docs/research/015-sheaf-geometry.md`.
