# 591 — Path-independence and coherent structure in graphs and sheaves

Context: [B32 / #591](https://github.com/NGL321/patchworks/issues/591), on map
[#532](https://github.com/NGL321/patchworks/issues/532). Opened by
[B27](https://github.com/NGL321/patchworks/issues/576)'s resolution as a **breadth** pass for general
context ahead of designing an objective around a local holonomy term. Builds on
[B24](https://github.com/NGL321/patchworks/issues/573),
[B25](https://github.com/NGL321/patchworks/issues/574),
[B26](https://github.com/NGL321/patchworks/issues/575),
[B29](https://github.com/NGL321/patchworks/issues/585) and
[#396](https://github.com/NGL321/patchworks/issues/396). Does not restate them.

**Reading-depth key** (#148's convention). `[FULL]` — primary text read at source, quotes verbatim
from it. `[ABS]` — abstract/metadata read at source. `[CITE]` — bibliographic existence and claim
confirmed through a secondary description; the primary text was *not* reached, and no quantitative
claim rests on it.

**Provenance discipline.** Every paragraph is tagged `PROVEN` (stated at source, quoted),
`INFERENCE` (my derivation from a quoted source; the source does not say it), or `ABSENT` (searched
for and not found; a negative about the search, not about the world).

---

## Headline

**The approach is not known to fail, but the literature contains three loaded guns, and two of them
are pointed at the exact objective B27 proposes.**

1. **Bodnar's Lemma 6 is a warning as much as a licence.** Read at source, its equality condition
   says `dim H⁰ = d` — the *entire stalk* is global sections — exactly when transport is
   path-independent. A perfectly flat sheaf on a connected graph is one in which nothing is local to
   anything. That is not an interpretation this project supplied: it is what the equality in the
   quoted lemma says. B29's measurement ("a flat sheaf on this construction is one whose lanes have
   stopped being different lanes") is the same statement arrived at empirically.
2. **The one study that has ever pointed at holonomy in a trained sheaf network found it does not
   earn its keep.** Grover & Bourgerie (2026), *Do Sheaf Neural Networks Use Holonomy?* — learned
   loop rotation rises 0.010 → 0.388 rad on the task that should need it, yet a ridge predictor on
   plain graph summaries beats the sheaf model, diagonal maps do as well as rotations, and on
   6-regular graphs rotation approaches a radian while the model stays at the training-mean
   predictor. Their words: *"Nonzero rotation is therefore not sufficient for counting improvement in
   these experiments."* They **measure and ablate; they never optimise.**
3. **Cycle consistency is the same object under a different name, it *has* been optimised for a
   decade, and its named failure mode is precisely a degenerate optimum.** CycleGAN satisfies its
   cycle loss by steganography — hiding a recoverable high-frequency channel invisible to the task.
   Permutation synchronisation prevents its own degenerate optimum with an explicit **rank /
   universe-size constraint**, which is structurally the representation floor B27 already proposes.

And one clean positive: **sparsification by agreement around cycles is a solved, deployed criterion
— in structure-from-motion, not in machine learning.** It has never, as far as this search reaches,
been used as a differentiable or training-time sparsifier. That is the transfer B34 wants, and it
comes with sixteen years of prior art on the *decision rule*, none on the *learning* version.

---

## 1. Path-independence, flatness, holonomy — what is proven

### 1.1 The bound and its equality condition, verified at source

`PROVEN` — Bodnar, Di Giovanni, Chamberlain, Liò & Bronstein, *Neural Sheaf Diffusion*, NeurIPS 2022,
[arXiv:2202.04579](https://arxiv.org/abs/2202.04579) `[FULL, arXiv HTML v3]`. Re-fetched
independently for this ticket rather than inherited from B26:

> **Lemma 6.** "Let ℱ be a discrete O(d) bundle over a connected graph G. Then dim(H⁰) ≤ d and
> dim(H⁰) = d if and only if the transport is path-independent."

The same paper supplies the quantitative slack, which is the part #396 does not quote:

> **Proposition 3.** "If ℱ is a discrete O(d) bundle over a connected graph and
> r := max ‖P^γ_{v→u} − P^γ'_{v→u}‖, then λ₀^ℱ ≤ r²/2"

and the definition of the failure it measures — *"the graph transport is path dependent, meaning
that how the vectors are transported across two nodes depends on the path between them."*

`INFERENCE` — three consequences the paper does not draw:

- **The bound is one-sided in the direction the project cares about.** `λ₀^ℱ ≤ r²/2` upper-bounds the
  spectral gap by path-dependence. It says path-dependence *permits* a small gap; it does not say
  reducing `r` buys anything. A gradient on `r` has no guarantee attached to it in this paper.
- **`r` is a max over pairs of paths, not a sum over cycles.** Any objective built as a mean over a
  cycle basis is optimising a different functional from the one Proposition 3 bounds. Whether they
  agree is unaddressed at source.
- **Equality is the degenerate case.** `dim H⁰ = d` with `d` the stalk width means every direction
  in the stalk is a global section — the sheaf is, cohomologically, the constant sheaf. §5 develops
  this.

### 1.2 Flatness on a graph is *only* holonomy — there is no curvature term

`PROVEN` — Gao, Brodzki & Mukherjee, *The Geometry of Synchronization Problems and Learning Group
Actions*, **Discrete & Computational Geometry** (2021),
[arXiv:1610.09051](https://arxiv.org/abs/1610.09051), DOI
[10.1007/s00454-019-00100-2](https://doi.org/10.1007/s00454-019-00100-2) `[CITE]`. They identify each
synchronisation problem in a topological group `G` on a connected graph `Γ` with a **flat principal
`G`-bundle over `Γ`**, and state that *prescribing an edge potential on a graph is equivalent to
specifying an equivalence class of flat principal bundles, of which the **triviality of holonomy
dictates the synchronizability** of the edge potential*. Their twisted Hodge theory's lowest-degree
Laplacian recovers the graph connection Laplacian.

`INFERENCE` — this is the cleanest available statement of what "path-independence" costs, and it is
worth stating plainly because it is easy to lose. A graph is a 1-complex: it has no 2-cells, so there
is no curvature 2-form to set to zero. **Every** edge-potential assignment on a graph is already flat
in the curvature sense. "Flat" in this project's usage can therefore only mean *trivial holonomy*,
and trivial holonomy over a connected graph is exactly the statement that the bundle is
gauge-equivalent to the product bundle — i.e. that a single global frame exists in which every
restriction map is the identity. **An objective that drives holonomy to the identity is an objective
that drives the sheaf toward being the constant sheaf in disguise.** No source states this about a
learned sheaf; it follows from the definitions Gao et al. and Bodnar et al. give.

Berwick-Evans, Hirani & Schubel, *Discrete Vector Bundles with Connection*,
[arXiv:2104.10277](https://arxiv.org/abs/2104.10277) `[ABS]` build the combinatorial calculus for the
higher-dimensional case where curvature is *not* vacuous — flat discrete connections there determine
a cochain complex computing twisted de Rham cohomology. Relevant only if the architecture ever
acquires 2-cells; it does not have them.

### 1.3 Synchronisation and connection Laplacians — the surrounding machinery

`PROVEN` — the standard chain:

- Singer & Wu, *Vector Diffusion Maps and the Connection Laplacian*, **Comm. Pure Appl. Math.** 65(8)
  (2012), [arXiv:1102.0075](https://arxiv.org/abs/1102.0075) `[CITE]`. Heat kernel for *vector
  fields* rather than functions; VDM's relation to the connection Laplacian is proved in the
  manifold-learning setup. This is where the sheaf-network literature's geometry comes from, and
  §4.1 of `docs/research/015-sheaf-geometry.md` already holds it.
- Bandeira, Singer & Spielman, *A Cheeger Inequality for the Graph Connection Laplacian*, **SIAM J.
  Matrix Anal. Appl.** 34(4):1611–1630 (2013),
  [arXiv:1204.3873](https://arxiv.org/abs/1204.3873), DOI
  [10.1137/120875338](https://doi.org/10.1137/120875338) `[CITE]`. Relates the **minimum frustration
  of a group potential** to the sum of the smallest `d` eigenvalues of the connection Laplacian and
  the second-smallest eigenvalue of the graph Laplacian — a worst-case performance guarantee for
  spectral `O(d)` synchronisation.
- Hansen & Ghrist, *Toward a Spectral Theory of Cellular Sheaves*, **J. Appl. Comput. Topology**
  (2019), [arXiv:1808.01513](https://arxiv.org/abs/1808.01513), DOI
  [10.1007/s41468-019-00038-7](https://doi.org/10.1007/s41468-019-00038-7) `[ABS + partial FULL]`.
  Contains "eigenvalue interlacing, sparsification, effective resistance, synchronization, and sheaf
  approximation". Their §3.5 (read) defines the relevant subclass: *"A subclass of sheaves of
  particular interest are those where all restriction maps are invertible. These sheaves have been
  the subject of significantly more study than the general case, since they extend to locally
  constant sheaves on the geometric realization of the cell complex."*

`INFERENCE` — **the caveat `015-sheaf-geometry.md` §4.2 already records is the binding one and it
gets worse, not better, on inspection.** Every result in §1.1–§1.3 is an `O(d)`- or invertible-map
result. This architecture's restriction maps are rectangular partial isometries (ADR-0032), and #533
established each hop is a principal-angle cosine matrix with `‖C‖₂ ≤ 1`. There is no group, so
`ρ: π₁(G,v₀) → Aut(ℱ(v₀))` is not defined, and the object B29 measures is a *monodromy* of
contractions rather than a holonomy representation. **Lemma 6 does not transfer, and neither does
Proposition 3's bound.** Nothing found in this pass repairs that — see §6.

---

## 2. Has flatness or holonomy ever been a training signal?

### 2.1 In sheaf neural networks — no, and the one study that looked found it does not carry the model

`PROVEN` — Grover & Bourgerie, *Do Sheaf Neural Networks Use Holonomy? A Measure–Intervene–Control
Study*, [arXiv:2607.19514](https://arxiv.org/abs/2607.19514), extended abstract, Geometric
Intelligence @ ECCV 2026 `[FULL, preprint]`. Already cited in `015-sheaf-geometry.md` §4.4; read
again here for the specific question of whether holonomy is *optimised*.

**It is not.** Their intervention is post-training: *"Zeroing the Cayley parameters in both layers
gives T(1)=T(2)=Id. We re-evaluate with all other weights fixed."* No term is added to the loss. What
they find:

- Neural Sheaf Propagation on triangle counting raises mean SO(2) loop rotation from **0.010 to
  0.388 ± 0.078 rad**; community detection, which has direct feature and homophily cues, ends at
  **0.029 ± 0.005 rad**. Holonomy is task-dependent and does grow where the task needs cycles.
- Identity replacement at N=3000 sharply increases error — the learned connection is load-bearing at
  scale.
- **But**: at small data *"the graph-summary ridge predictor reaches MSE 0.110, whereas the best SNN
  reaches 0.71"*; on random 6-regular graphs *"Δconst stays near zero while twist grows"*, rotation
  approaching a radian while models *"stay near the training-mean predictor"*; and diagonal maps
  improve without continuous rotation.
- Their conclusion, verbatim: ***"Nonzero rotation is therefore not sufficient for counting
  improvement in these experiments."***

`PROVEN` — Dong, Peng, Li, Feng & Xia, *Demystifying Oversmoothing in Sheaf Neural Networks: An
Index-Theoretic Criterion*, [arXiv:2608.16180](https://arxiv.org/abs/2608.16180) (17 Aug 2026)
`[FULL, preprint]`. B24 read this at `[ABS]`; read here in full because #396 rests on it. The
holonomy representation is confirmed present and central, contrary to what the abstract alone
suggests: *"The restriction maps determine a holonomy representation ρ: π₁(G,v₀) → Aut(ℱ(v₀))"*, with
fixed subspace *"W := ⋂_{γ∈π₁(G,v₀)} ker(ρ(γ) − I) ⊆ ℱ(v₀)"*. **#396's citation is sound.**

Their **Theorem 5 (Genuine harmonic inclusion)** is the load-bearing find of this pass and §5 returns
to it. Its *dimensional control* hypothesis requires `dim W = k < rank(ℱ)` — verbatim, *"the holonomy
is non-trivial on the full stalk, i.e. k < rank(ℱ)"*. And they are explicit that the framework is
analytical, not an objective: *"our operator-level criterion of Theorem 5 precisely characterises
intrinsic harmonic capacity in the infinite-depth limit, which serves a complementary role to
finite-depth trained behaviour."*

`ABSENT` — no paper found in which a flatness, holonomy, frustration or path-independence term
appears in the **training loss** of a sheaf neural network or any graph network with learned
restriction maps. Searched: sheaf + regulariser/loss/objective + holonomy/flatness/cycle;
gauge-equivariant GNN regularisation; connection-Laplacian training objectives. Every sheaf-learning
objective encountered here and in B24 minimises **smoothness / total variation** (Hansen & Ghrist
ICASSP 2019; Di Nino, Barbarossa & Di Lorenzo, [arXiv:2501.19207](https://arxiv.org/abs/2501.19207)),
which B24 already flagged as an objective whose global optimum is collapse.

**This is a genuine hole in the field, and B27's proposal sits in it.** Stated as a negative about a
search, not a proof of non-existence.

### 2.2 In multi-view correspondence — yes, for a decade, under the name cycle consistency. The lineage is confirmed.

`PROVEN` — the identification is not analogy; one of the founding papers uses this project's exact
phrase. Nguyen, Ben-Chen, Welnicka, Ye & Guibas, *An Optimization Approach to Improving Collections
of Shape Maps*, **Computer Graphics Forum** 30(5) / SGP 2011 `[CITE]`: they *"add the constraint of
global map consistency, requiring that **any composition of maps between two shapes should be
independent of the path chosen in the network**"*, and optimise for a set of consistent compositions.
Path-independence over a network of maps, as an optimisation objective, 2011.

The line from there:

- **Huang & Guibas**, *Consistent Shape Maps via Semidefinite Programming*, **Computer Graphics
  Forum** 32(5) / SGP 2013, DOI [10.1111/cgf.12184](https://doi.org/10.1111/cgf.12184) `[CITE]`. The
  structural theorem this whole field runs on: *if the ground-truth maps are cycle-consistent, the
  matrix storing all pairwise maps in blocks is **low-rank and positive semidefinite***. They recover
  cycle-consistent maps as the nearest PSD matrix, with KKT-derived exact-recovery guarantees under
  bounded input error.
- **Pachauri, Kondor & Singh**, *Solving the multi-way matching problem by permutation
  synchronization*, NeurIPS 2013 `[CITE]` — names permutation synchronisation.
- **Zhou, Zhu & Daniilidis**, *Multi-Image Matching via Fast Alternating Minimization*, ICCV 2015
  `[CITE]`.
- **Zhu, Park, Isola & Efros**, *Unpaired Image-to-Image Translation using Cycle-Consistent
  Adversarial Networks*, ICCV 2017 — cycle consistency as a differentiable **loss term** on learned
  maps. This is the closest published thing to "optimise path-independence with gradient descent",
  and §5.3 records what it does when you do.

`INFERENCE` — **cycle consistency and trivial holonomy are the same predicate, up to the group.** A
composition around a closed loop equalling the identity is `ρ(γ) = Id`; requiring it for all loops is
trivial holonomy; Gao et al. (§1.2) make the identification explicit for group-valued edge
potentials. **The differences that matter for transfer are two, and both cut against us:**

1. In synchronisation and multi-view matching, the **pairwise maps are measured data** and the
   unknowns are the per-node frames. Here **both** the maps and the states are learned. The
   constraint that makes the problem well-posed in the source literature — that you cannot change the
   observations to make them agree — **is absent in the training setting.** Huang & Guibas' recovery
   guarantee is a *denoising* guarantee about recovering a consistent ground truth from noisy
   observations of it; it says nothing about a system free to invent the maps.
2. Huang & Guibas' theorem runs the direction that is unhelpful here: cycle-consistency **implies**
   low rank. Driving toward cycle consistency is, by their own structural result, driving toward a
   low-rank bundle matrix. Rank and consistency are not independent axes in this literature; they are
   the same axis. B29 measured the same inverse on this surface (`p16`/`p24` reach composed rank
   2.9–4.0 with channel return at chance; `shipped`/`p8` return the direction and read rank ≈ 1).

---

## 3. Coherent structure without a global section — machinery and cost

### 3.1 There are two different machineries and they cost wildly different things. Do not conflate them.

`PROVEN` — **Machinery A, contextuality on empirical models.** Abramsky & Brandenburger, *The
Sheaf-Theoretic Structure of Non-Locality and Contextuality*, **New J. Phys.** 13:113036 (2011)
`[CITE, via B24]`: contextuality is exactly the obstruction to a global section — local sections
agreeing on every overlap that admit no global one. Quantified by Abramsky, Barbosa & Mansfield, *The
Contextual Fraction as a Measure of Contextuality*, **Phys. Rev. Lett.** 119:050504 (2017),
[arXiv:1705.07918](https://arxiv.org/abs/1705.07918), DOI
[10.1103/PhysRevLett.119.050504](https://doi.org/10.1103/PhysRevLett.119.050504) `[FULL abstract]`:
*"its value, and a witnessing inequality, can be computed using linear programming."*

`INFERENCE` — **but this machinery is defined on probability tables, not on vector-space sections,
and its LP does not scale the way "linear programming" makes it sound.** The variables of the
contextual-fraction LP range over **global assignments** — outcome assignments to every measurement —
so the program's width is exponential in the number of measurements, not polynomial in the number of
cells. The abstract does not state the size; the exponential is a consequence of the definition of
the empirical model and is my derivation, not theirs. **Verify before budgeting.** Also standing:
B24's caveat that the 2011 cohomological invariant is sufficient but not complete (false negatives),
per [arXiv:1807.04203](https://arxiv.org/abs/1807.04203).

`INFERENCE` — **Machinery B, and it is the one that fits, is linear algebra.** For a cellular sheaf
of vector spaces the global sections are `H⁰ = ker δ` and the obstruction is a rank computation on
`δ`. On a graph of a few hundred cells with stalks of width tens, `δ` is a few thousand square and an
SVD is milliseconds. **The tractability question the ticket asks has a boring answer in the linear
case and a bad answer in the probabilistic case, and the architecture is linear**
(`01-cell-and-sheaf.md`'s "all nonlinearity lives inside the cell"). The relevant vocabulary is
Hansen & Ghrist's relative cohomology `H⁰(G, A, ℱ)` versus local sections `H⁰(A, ℱ)`, which B24
already surfaced from *Opinion Dynamics on Discourse Sheaves*,
[arXiv:2005.12798](https://arxiv.org/abs/2005.12798).

### 3.2 The region machinery

`PROVEN` (via B24, not re-verified here) — Robinson's **consistency radius** and **consistency
filtration**, *Assignments to sheaves of pseudometric spaces*, **Compositionality** 2(2),
[arXiv:1805.08927](https://arxiv.org/abs/1805.08927); Praggastis' **unique maximal consistent
subcomplexes**, [arXiv:1612.00397](https://arxiv.org/abs/1612.00397).

`ABSENT` — **the computational cost of either is still not established.** B24 flagged
*"[Theorem statement, algorithm and complexity NOT read — verify before building]"* for Praggastis;
targeted searches for the complexity of consistency radius and of finding maximal consistent
subcomplexes returned nothing on point in this pass either. **This remains the one number B34 would
need and nobody in this repo has.** It should be read out of the primary texts directly, not searched
for.

---

## 4. Sparsification by a consistency criterion

### 4.1 It exists, it is standard, and it is in computer vision

`PROVEN` — **Zach, Klopschitz & Pollefeys**, *Disambiguating Visual Relations Using Loop Constraints*,
CVPR 2010, pp. 1426–1433, DOI [10.1109/CVPR.2010.5539801](https://doi.org/10.1109/CVPR.2010.5539801)
`[CITE]`. They *chain reversible transformations over cycles in the graph to build statistics for
identifying inconsistent loops*, and infer likely false-positive geometric relations in a Bayesian
framework. **This is edge removal by agreement around cycles, sixteen years old, and it is exactly
the criterion B34 wants.**

The technique is now standard enough to have an acronym. `PROVEN` `[ABS]` —
[arXiv:2608.22054](https://arxiv.org/abs/2608.22054), *Robust Global Structure-from-Motion via View
Graph Pruning* (2026), states the state of the art plainly: *"Cycle consistency inference (CCI) aims
to infer incorrect edges by analyzing the geometric consistency over cycles of the view-graph"*, and
*"wrong edges can be removed by loop closure detection, i.e., by exploiting the property that
relative rotations should compose to the identity over closed loops."* Their own contribution
partitions the view graph into locally consistent subgraphs and RANSACs edges across them. Also:
Shen, Luo, Zhu, Zhang, Fang & Quan, *Graph-Based Consistent Matching for Structure-from-Motion*,
ECCV 2016, DOI [10.1007/978-3-319-46487-9_9](https://doi.org/10.1007/978-3-319-46487-9_9) `[CITE]`.

`INFERENCE` — **note what the criterion is used for there. It removes edges believed to be *wrong* —
outliers from repeated structure — restoring a consistency the true scene has.** It is an outlier
filter against a ground truth, not a capacity allocator. Transferring it to "prune the edges that
disagree" in a learned sheaf inherits none of that justification: nothing here says the disagreeing
edges are the false ones. In a system meant to be **contextual by design** (B29 §5), the
consistently-disagreeing edges may be the informative ones. This is the single most important caveat
on the transfer and no source addresses it.

### 4.2 It does not exist in machine learning

`ABSENT` — searched GNN and sheaf sparsification for a consistency-around-cycles criterion. What is
there instead: **spectral** (Spielman & Srivastava, *Graph Sparsification by Effective Resistances*,
[arXiv:0803.0929](https://arxiv.org/abs/0803.0929); Batson–Spielman–Srivastava), **effective
resistance** (the top-performing sparsifier in Chen et al., *Demystifying Graph Sparsification
Algorithms in Graph Properties Preservation*, **PVLDB** 17), **learnable-mask magnitude** (UGS),
**attention**, **Shapley attribution** ([arXiv:2507.20460](https://arxiv.org/abs/2507.20460)), and
**mixture-of-graphs** ([arXiv:2405.14260](https://arxiv.org/abs/2405.14260), ICLR 2025). Hansen &
Ghrist's own sheaf sparsification (§1.3) is spectral / effective-resistance, not consistency-based.

`INFERENCE` — the transfer is therefore **novel as an ML method and well-trodden as a decision rule**.
That is a good place to be, and it also means there is no published evidence that it works when the
maps are learned rather than measured. `docs/research/150-effective-resistance-and-the-gauge.md` and
`237-the-sheaf-laplacians-effective-resistance.md` already hold the effective-resistance side.

---

## 5. The degenerate optimum — this is the best-documented part of the whole question

### 5.1 The failure mode has a name, and it is the same name in three literatures

`PROVEN` — **dimensional collapse.** Jing, Vincent, LeCun & Tian, *Understanding Dimensional Collapse
in Contrastive Self-Supervised Learning*, ICLR 2022,
[arXiv:2110.09348](https://arxiv.org/abs/2110.09348) `[ABS]`: embedding vectors end up spanning a
lower-dimensional subspace instead of the full embedding space. Distinguished at source from
**complete collapse** (constant output). B24 and B25 already hold the GNN-side names —
oversmoothing, rank collapse (Zhang, Deidda, Higham & Tudisco, ICLR 2026,
[arXiv:2502.04591](https://arxiv.org/abs/2502.04591)), representation degeneracy (Dönmez et al.,
[arXiv:2605.11178](https://arxiv.org/abs/2605.11178)).

**B29's finding — flatness is free at width one — is the extreme case of this, and it is `INFERENCE`
that no source names it in that exact form.** What the literature says is adjacent and strong: an
invariance/agreement objective's optimum is the constant solution unless something forbids it. B24
already recorded the sheaf version (*"the sheaf that minimises disagreement perfectly is the one that
maps everything to a single shared direction — collapse is the global optimum of the objective"*).

### 5.2 What the literature pairs with it — three independent answers, and they agree

`PROVEN` — **(a) A variance floor plus decorrelation.** Bardes, Ponce & LeCun, *VICReg:
Variance-Invariance-Covariance Regularization for Self-Supervised Learning*, ICLR 2022,
[arXiv:2105.04906](https://arxiv.org/abs/2105.04906) `[ABS]`. The invariance term is the agreement
term; collapse is prevented by *"(1) a term that maintains the variance of each embedding dimension
above a threshold, (2) a term that decorrelates each pair of variables."* Notably it needs **neither**
stop-gradient, weight sharing, batch norm, nor memory banks — the two explicit terms suffice.

`INFERENCE` — **this is exactly the shape B27 already chose** (a local holonomy term beside a
representation floor), and it is the canonical shape in the field. The refinement VICReg adds that
B27's formulation does not yet have is the **second** term: a *per-dimension* variance hinge alone is
not enough; decorrelation across dimensions is the other half, because a floor on each dimension's
variance is satisfiable by duplicating one direction across all of them. Given B29's reading that 250
of 260 cycles are rank-1 and training pushes toward positive return on exactly those, the duplication
failure is the live one. **Caveat on the transfer:** VICReg's terms are computed over a *batch* of
embeddings, and this project has a no-batch constraint (#532). A per-cell, per-tick surrogate for the
covariance term is not something VICReg supplies.

`PROVEN` — **(b) An explicit rank / universe-size constraint, in the cycle-consistency literature
itself.** Permutation synchronisation prevents its own degenerate solution structurally: *cycle
consistency across pairwise permutation matrices implies the definition of a **universe** of vertices
… the dimension of this space corresponds to the **rank** of the bulk permutation matrix*, and the
problem is posed *"under constraints imposed on the rank and the permutations"* (Pachauri, Kondor &
Singh 2013; Zhou, Zhu & Daniilidis, ICCV 2015; Huang & Guibas 2013) `[CITE]`. Everything-maps-to-one
is perfectly cycle-consistent; the universe size is what forbids it.

`PROVEN` — **(c) An explicit non-triviality hypothesis, in the sheaf literature.** Dong et al.'s
Theorem 5 (§2.1) requires **`dim W = k < rank(ℱ)`** — the holonomy must be **non-trivial on the full
stalk** — *together with* a capacity condition (`Δ_ind + ΔStr⁽¹⁾ ≥ 1`, or `dim W > dim H⁰(ξ)`) before
it will conclude `dim H⁰(ℱ) > dim H⁰(ξ)` with non-trivial harmonic quotient. And they warn that
*"raw index jump alone does not guarantee genuine anti-oversmoothing."*

`INFERENCE` — **(c) is the sharpest thing this pass found and it should be read carefully, because it
runs against the naive form of the objective.** Dong et al.'s criterion for a sheaf's harmonic space
being *genuinely* larger requires holonomy to be non-trivial. Bodnar's Lemma 6 gets `dim H⁰ = d` from
holonomy being *trivial*. These are not in contradiction — Lemma 6's maximum is the constant sheaf,
which is precisely the *"inflate dim ker ℒ while their harmonic sections remain entirely constant"*
configuration Dong et al. built their criterion to reject. **Read together, the two papers say: the
flat sheaf attains the maximum of `dim H⁰` and it is the worthless maximum.** Neither paper says this
sentence; it is the conjunction of two quoted results. Note also that B29 has already computed the
`dim W` in Theorem 5's hypothesis by hand (its fixed-subspace ladder table), so the criterion is
readable on this surface without new machinery.

### 5.3 The one documented case of optimising cycle consistency end-to-end, and it went wrong in an instructive way

`PROVEN` — Chu, Zhmoginov & Sandler, *CycleGAN, a Master of Steganography*, NIPS 2017 Workshop on
Machine Deception, [arXiv:1712.02950](https://arxiv.org/abs/1712.02950) `[FULL abstract]`:

> "CycleGAN learns to 'hide' information about a source image into the images it generates in a
> nearly imperceptible, high-frequency signal. This trick ensures that the generator can recover the
> original sample and thus satisfy the cyclic consistency requirement, while the generated image
> remains realistic. … the cyclic consistency loss causes CycleGAN to be especially vulnerable to
> adversarial attacks."

`INFERENCE` — the shape of the failure, stated in this project's terms: **a path-independence penalty
is satisfiable by a low-amplitude side-channel that the loop preserves and the task never reads.**
The loop closes; nothing useful travelled. B29's own open question — *"whether the 1-D lateral
path-independence is **useful** (a sign agreement on a one-dimensional channel may be a convention
rather than information)"* — is the same worry, and CycleGAN is the literature's documented instance
of it. A representation floor on *width* does not close this hole, because the steganographic channel
has width; what closes it is a downstream demand that the returned direction be **read**. Nothing
found in this pass supplies a principled form of that demand — which is the same gap B26 recorded
(*"no principled, task-free criterion separating 'should be low-rank' from 'failed to learn'"*).

---

## 6. What is absent, and what would change the picture

`ABSENT`, each a statement about this search and not about the world:

1. **No holonomy/flatness term in any published training objective.** §2.1.
2. **No transfer of Lemma 6 or of the connection-Laplacian Cheeger bounds to non-orthogonal,
   rectangular restriction maps.** Every quantitative result in §1 assumes `O(d)` or invertibility.
   This is the standing caveat from `015-sheaf-geometry.md` §4.2 and this pass did not close it. **A
   version of Lemma 6 for contractions would be the single most valuable thing to find**, and its
   absence means any pre-registered prediction built on Lemma 6 is building on an analogy.
3. **No cost figure for consistency radius or maximal consistent subcomplexes.** §3.2. Read
   Praggastis and Robinson directly.
4. **No consistency-based sparsifier in ML.** §4.2 — an opportunity, not a gap in the search.
5. **No paper naming "flatness is free at width one".** The nearest named object is dimensional
   collapse; the pairing (variance floor + decorrelation, or rank/universe constraint) is standard.

**What would change our minds, stated in advance.** Any of: (a) an `O(d)`-free version of Lemma 6, or
a counterexample showing the equality condition fails for contractions; (b) a paper that puts a
frustration or holonomy term in a loss and reports it; (c) a complexity result making
consistency-radius filtration super-cubic on a few hundred cells; (d) evidence that CCI-style pruning
has been tried on learned rather than measured maps.

---

## Registers consulted

**open-problems** — nothing relevant; nothing in the register names path-independence, holonomy or
consistency-based pruning as an admitted problem, and this pass mints none (only a grilling session
may, per ADR-0029).
**proposed-solutions** — [#396](https://github.com/NGL321/patchworks/issues/396) matched on `@shape
effective rank slides toward 1 under a sparsity pressure` and `@shape private capacity and
transmitted capacity are one budget`; **both of its citations are verified at source by this pass**
(Bodnar Lemma 6 verbatim §1.1; Dong et al.'s holonomy representation confirmed present in the full
text §2.1, notwithstanding its absence from the abstract).
[#315](https://github.com/NGL321/patchworks/issues/315) matched on mechanism (reading holonomy) and is
a prerequisite, now built as `b29_holonomy.py`. No new proposal is minted here: this is a context pass
and every mechanism it surfaces belongs to B27/B34's design decision, not to the shelf.
**dismissed-solutions** — nothing relevant; nothing `refused` or `failed` touches holonomy, cycle
consistency or consistency-based sparsification.

---

## Decision provenance

Resolved by literature search on branch `research/b32-path-independence` off `main`. No rig was run,
no architecture changed, no ADR touched.

Scoping calls made by the agent, not asked for by the ticket:

- **(a) Leading with the three "loaded guns" rather than with a flat five-part answer.** The ticket
  says anything indicating known failure is the most valuable return; nothing found says *fails*, so
  the headline reports the strongest available near-misses and labels them as such rather than
  reporting "no known failure" and burying them.
- **(b) Re-fetching Bodnar Lemma 6 and Dong et al. at source rather than inheriting them from B26 and
  #396.** Both are load-bearing for a design decision and #396's Dong citation could not be confirmed
  from the abstract alone. It is confirmed from the full text.
- **(c) Treating cycle consistency as the same object and reporting it as confirmed lineage.** The
  ticket asked to confirm or rule out; Nguyen et al. 2011's own phrase — *"independent of the path
  chosen in the network"* — is taken as decisive rather than suggestive.
- **(d) Splitting question 3 into two machineries and answering the cost question separately for
  each.** The ticket asks one cost question; the honest answer is two different numbers, and the
  exponential width of the contextual-fraction LP is my derivation from the definition of an
  empirical model, flagged `INFERENCE` and marked *verify before budgeting*.
- **(e) Not minting a register proposal.** Per ADR-0029 a research pass may mint proposals directly;
  I judged that everything here is context for a decision B27 already owns, and a proposal would
  duplicate #396.

Three statements in this document are conjunctions of quoted results that no single source makes, and
each is marked `INFERENCE` at its site: **flat-on-a-graph means gauge-equivalent to the constant
sheaf** (§1.2); **Bodnar's maximum of `dim H⁰` is Dong et al.'s worthless maximum** (§5.2); and **the
source literature's well-posedness comes from the maps being measured, which is exactly what training
removes** (§2.2). They are the three places this pass asserts something the record did not contain.
