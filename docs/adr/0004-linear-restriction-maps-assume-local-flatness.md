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

This is the load-bearing use of the word *manifold* in Patchworks, and the only one. The world
is not claimed to be a manifold and does not need to be; the **pieces** are. Each cell owns a
`k`-dimensional, locally Euclidean piece of the problem — its chart is a chart of *that piece*,
in the strict sense — and the sheaf glues pieces of differing dimension without ever requiring
their union to be a manifold. A cellular sheaf is the atlas idea with the constant-dimension
requirement removed, which is why it is the right formalism here and an atlas is not.

`k < n` follows from this rather than being asserted alongside it: **`k` is the dimension of the
piece; `n` is the room needed to talk about it with neighbours.**

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

- **Linearity is what keeps this test readable.** A linear map has no constant term, so it cannot
  absorb a persistent offset in an edge stalk — and a curvature residual's constant part is exactly
  such an offset. Affine restriction maps would launder the signature away into a learned offset while
  the geometry stayed just as bent. This is a better argument for linearity than efficiency, and it
  was not visible until the floor was named.
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
