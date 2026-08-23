# Citation pass: where the geometry lives — manifolds, charts, and projections (patchworks#15)

Validates (and in three places **corrects**) the sheaf-geometry mechanics asserted while working
issue #15, against `docs/spec/01-cell-and-sheaf.md` and `docs/spec/02-tick-semantics.md`.
Citations validate after the fact per the map's Notes; this does not seed design. Vocabulary
follows `CONTEXT.md`: Patchworks' side of every comparison is in its own terms (chart, node stalk,
edge stalk, disagreement, reconciliation, restriction map); the prior art's side is in its own
field's terms.

**Headline.** Three of the five asserted claims are confirmed as stated. Two are **false as
stated and true only under an extra hypothesis Patchworks does not satisfy** — and in both cases
the hypothesis that fails is *exactly* the masked/sparse restriction maps `01-cell-and-sheaf.md`
commits to. Details in §1.4, §3.1, §3.2. Read those two before reusing the claims.

Primary sources read directly (PDF text extracted, not summaries):

- Hansen, J. & Ghrist, R. (2019). "Toward a spectral theory of cellular sheaves." *Journal of
  Applied and Computational Topology* 3, 315–358. [arXiv:1808.01513](https://arxiv.org/abs/1808.01513)
- Hansen, J. (2020). *Laplacians of Cellular Sheaves: Theory and Applications.* PhD thesis,
  University of Pennsylvania. [PDF](https://www.jakobhansen.org/publications/thesis.pdf)
- Hansen, J. & Ghrist, R. (2021). "Opinion dynamics on discourse sheaves." *SIAM J. Applied
  Mathematics* 81(5). [arXiv:2005.12798](https://arxiv.org/abs/2005.12798)
- Bodnar, C., Di Giovanni, F., Chamberlain, B., Liò, P. & Bronstein, M. (2022). "Neural Sheaf
  Diffusion: A Topological Perspective on Heterophily and Oversmoothing in GNNs." NeurIPS 35.
  [arXiv:2202.04579](https://arxiv.org/abs/2202.04579)
- Barbero, F., Bodnar, C., Sáez de Ocáriz Borde, H., Bronstein, M., Veličković, P. & Liò, P.
  (2022). "Sheaf Neural Networks with Connection Laplacians." TAG-ML @ ICML.
  [arXiv:2206.08702](https://arxiv.org/abs/2206.08702)
- Singer, A. & Wu, H.-T. (2012). "Vector diffusion maps and the connection Laplacian."
  *Comm. Pure Appl. Math.* 65(8). [arXiv:1102.0075](https://arxiv.org/abs/1102.0075)
- Topping, J., Di Giovanni, F., Chamberlain, B., Dong, X. & Bronstein, M. (2022). "Understanding
  over-squashing and bottlenecks on graphs via curvature." ICLR.
  [arXiv:2111.14522](https://arxiv.org/abs/2111.14522)
- Southern, J., Di Giovanni, F., Bronstein, M. & Lutzeyer, J. (2025). "Understanding Virtual
  Nodes: Oversquashing and Node Heterogeneity." ICLR.
  [arXiv:2405.13526](https://arxiv.org/abs/2405.13526)
- Seely, J. (2025). "Sheaf Cohomology of Linear Predictive Coding Networks."
  [arXiv:2511.11092](https://arxiv.org/abs/2511.11092)
- Curry, J. (2014). *Sheaves, Cosheaves and Applications.* PhD thesis, UPenn.
  [arXiv:1303.3255](https://arxiv.org/abs/1303.3255)

---

## 1. The mechanics: what is confirmed, and what is not

### 1.1 Cellular sheaf, restriction maps — CONFIRMED

Hansen & Ghrist define a cellular sheaf on a regular cell complex `X` as an assignment of a vector
space `F(σ)` to each cell and a linear map `F_{σ◁τ}: F(σ) → F(τ)` for each incident pair, subject
to identity and composition; compactly, "a functor `F : P_X → Vect_k`" (§2.2). Bodnar et al. give
the graph special case verbatim in Patchworks' shape — a space `F(v)` per node, `F(e)` per edge,
and "a linear map `F_{v◁e}: F(v) → F(e)` for each incident `v ◁ e` node-edge pair", with the note
that "the vector spaces of the nodes and edges are called **stalks**, while the linear maps are
referred to as **restriction maps**" (§2, Definition 1).

Patchworks' node stalk / edge stalk / restriction map are the standard objects, used with their
standard names. No divergence.

### 1.2 Coboundary `(δx)_e = F_{v◁e}x_v − F_{u◁e}x_u` — CONFIRMED

Hansen & Ghrist give the general graded formula `(δ^k x)_τ = Σ_{dim σ = k} [σ : τ] F_{σ◁τ}(x_σ)`,
where `[• : •] : P_X × P_X → {0, ±1}` is a signed incidence relation supplying orientation (§2.2.2).
On a graph, `δ^0` restricted to an edge `e` with endpoints `u, v` and an orientation choosing `+1`
at `v` and `−1` at `u` is exactly `(δx)_e = F_{v◁e}x_v − F_{u◁e}x_u`. Seely writes it in precisely
that form for network sheaves: `(δ^0 s)_e = ρ_{e←v}s_v − ρ_{e←u}s_u` (his Eq. 8).

**Note the orientation caveat.** The sign is a *choice*; it exists only because a graph edge has
two ends and one must be called positive. Nothing downstream depends on which — `L = δᵀδ` is
orientation-independent. But Patchworks' disagreement is defined in `CONTEXT.md` as "the
difference … between the two adjacent cells' restrictions", which silently fixes an arbitrary
sign per edge. That is fine and standard; it is worth knowing it is a convention, not a fact.

### 1.3 `L = δᵀδ`, Dirichlet energy `xᵀLx = Σ_e ‖disagreement_e‖²` — CONFIRMED

Hansen & Ghrist construct the Hodge Laplacian `Δ = (δ + δ*)² = δ*δ + δδ*`, graded as
`Δ^k = (δ^k)*δ^k + δ^{k-1}(δ^{k-1})*`, and note that at degree 0 the second term vanishes, so the
sheaf Laplacian is `L = (δ^0)*δ^0` (§3.2). Block structure, stated explicitly: diagonal blocks
`Δ^0_{v,v} = Σ_{v◁e} F*_{v◁e}F_{v◁e}`, off-diagonal `Δ^0_{u,v} = −F*_{u◁e}F_{v◁e}` (§3.2).
Bodnar et al. Definition 2 restates this node-wise as
`L_F(x)_v := Σ_{v,u◁e} F^⊤_{v◁e}(F_{v◁e}x_v − F_{u◁e}x_u)` — which is literally "each cell sums,
over its own incident edges, its own restriction-transpose applied to its own disagreement". That
is the graph-local reconciliation gradient `02-tick-semantics.md` runs one step of, with no global
term anywhere in it.

For the energy, Hansen's thesis (§1.7) states it directly:

> `E_0(x) = ⟨x, L_F x⟩ = ⟨δ^0 x, δ^0 x⟩ = ‖δ^0 x‖²`, "which measures how far a 0-cochain `x` is
> from defining a section of `F`."

Bodnar et al. Definition 14 gives the per-edge expansion (in the degree-normalised form):
`E_F(x) := x^⊤Δ_F x = ½ Σ_{e:=(v,u)} ‖F_{v◁e}D_v^{-1/2}x_v − F_{u◁e}D_u^{-1/2}x_u‖²₂`.

So `01-cell-and-sheaf.md`'s claim — "disagreement … is one edge's term of the sheaf's Dirichlet
energy … but no cell ever reads that sum" — is exactly right, and the sum-over-edges decomposition
is the literature's own. Seely closes the loop with predictive coding explicitly: PC energy
"can be written compactly as `E_PC(s) = ½‖δ^0 s‖²`", and PC inference by gradient descent "yields
the gradient flow `ṡ = −Ls` … known as sheaf diffusion" (§3). This is independent confirmation of
`01-cell-and-sheaf.md`'s "predictive coding's error and the sheaf's inconsistency are **the same
quantity**" — someone else derived it from the PC side and landed on the same object.

### 1.4 `H^0 = ker δ = ker L = global sections` — CONFIRMED. `H^1 = coker δ` — CONFIRMED. **"zero for a tree" — FALSE as stated.**

`H^0`: Hansen & Ghrist, §2.2.2: "`H^0(X; F)` is naturally isomorphic to `Γ(X; F)`, the space of
global sections", where a global section is "a choice `x_σ ∈ F(σ)` for each cell `σ` … such that
`x_τ = F_{σ◁τ}x_σ` for all `σ ◁ τ`" (Definition 3). The Laplacian identification is Theorem 1
(Hodge): `ker Δ^k ≅ H^k(C^•)`; at `k = 0`, `ker L = H^0`. Hansen & Ghrist's opinion-dynamics
paper states the graph case as a display equation: `H^0(G; F) = ker L_F` (their Eq. 2.6). Bodnar
et al. restate it: "`H^0(G; F)` and `ker(L_F)` are isomorphic as vector spaces". **Confirmed.**

`H^1 = coker δ`: Seely, Eq. 10: `H^1(G, F) = C^1 / im δ^0 ≅ ker(δ^0)^⊤`, described as "prediction
error patterns (on edges) that cannot arise from any activation choice." On a graph `δ^1 = 0`
(no 2-cells), so `H^1 = C^1 / im δ^0` with no quotient-by-kernel subtlety. **Confirmed.** The
inner-product representative `ker(δ^0)^⊤` is the useful one: it is the subspace of edge-stalk
configurations that **no** assignment of node stalks can produce, i.e. residual disagreement that
reconciliation provably cannot remove no matter how long it runs.

**The tree claim is false.** `H^1 = 0` for a tree holds for the *constant* sheaf, and more
generally whenever every restriction map is **surjective** onto its edge stalk (peel leaves and
induct). It fails otherwise, and the counterexample is one edge wide:

> Graph `u — e — v` (a tree). `F(u) = F(v) = F(e) = R`. Both restriction maps zero.
> Then `δ = 0`, so `H^0 = R²` and `H^1 = R ≠ 0`.

Patchworks does not have the surjectivity hypothesis. `01-cell-and-sheaf.md` makes restriction
maps **masked** and **learned under sparsity pressure**; a sparsified map is rank-deficient by
design, and `m_e` is set by the mask rather than by any rank condition. So a Patchworks graph can
carry irreducible disagreement on a **tree** — with no cycle anywhere. Whatever intuition says
"`H^1` lives on cycles" is the *constant-sheaf* intuition; it does not survive contact with
learned rank-deficient maps.

Seely says this in as many words, and it is the single most useful sentence in his paper for
Patchworks (§3):

> "In our setting the predictive coding sheaf `F` depends on the current network weights, so `H^0`
> and `H^1` should be understood as **weight-dependent linear subspaces** of `C^0` and `C^1`, not
> as topological invariants of the underlying graph."

The right statement: **`H^1` has two independent sources** — cycles (holonomy; §4) and
non-surjectivity of restriction maps (rank; §3). Patchworks' sparsity pressure feeds the second
source continuously, and the second source is invisible to any graph-topological argument.

---

## 2. The Euler characteristic claim — CONFIRMED, and it is the one genuinely topology-free
quantity in the picture

**Claim.** `dim H^0 − dim H^1 = Σ_v dim F(v) − Σ_e dim F(e)`, independent of the restriction maps.

**Verdict: correct, and it is elementary rank–nullity on a two-term complex.** On a graph the
cochain complex is `0 → C^0 --δ--> C^1 → 0`, with `C^0 = ⊕_v F(v)` and `C^1 = ⊕_e F(e)` (Hansen &
Ghrist, §2.2.2: `C^k(X; F) = ⊕_{dim σ = k} F(σ)`). Then

```
dim H^0 − dim H^1 = dim ker δ − dim coker δ
                  = dim ker δ − (dim C^1 − rank δ)
                  = (dim C^0 − rank δ) − dim C^1 + rank δ
                  = dim C^0 − dim C^1
                  = Σ_v dim F(v) − Σ_e dim F(e).
```

`rank δ` — the only place the restriction maps appear — cancels. This is the Euler characteristic
of the complex, and the general statement (`Σ_k (−1)^k dim H^k = Σ_k (−1)^k dim C^k` for any
finite-dimensional cochain complex) is standard homological algebra, in any algebraic-topology
text (Hatcher, *Algebraic Topology*, §2.2, Euler characteristic; Weibel, *An Introduction to
Homological Algebra*, §1.2).

**How it is stated in the literature: it mostly is not.** This is a real finding, and it is worth
recording plainly. Across Hansen & Ghrist (both papers), Hansen's thesis, Bodnar et al., Barbero
et al., and Seely, **I could not find this identity stated for network sheaves at all.** Curry's
thesis discusses Euler characteristic only in the Euler-calculus / constructible-function sense
(§ on higher Euler calculus), not as an index formula for `δ` on a graph. Hansen & Ghrist's
spectral programme is about eigenvalues, interlacing, sparsification and effective resistance —
the index of `δ` is never a subject. The one paper that would naturally have used it, Seely's,
instead makes the *opposite* point (quoted in §1.4): that `H^0` and `H^1` are weight-dependent
and not topological.

Both things are true and they fit together exactly:

- `dim H^0` and `dim H^1` **individually** depend on the restriction maps — Seely's point.
- Their **difference** does not — the index formula.

That is the precise sense in which the Patchworks assertion is right: the structural mask, by
fixing every `m_e` at graph-construction time (`01-cell-and-sheaf.md`: "`m` is therefore not an
independent parameter"), fixes `χ = |V|·n − Σ_e m_e` **permanently**, before any learning. Learning
can trade `dim H^0` against `dim H^1` one-for-one and can do nothing else. `01-cell-and-sheaf.md`
also says the mask "closes and never re-opens" — under closure, `Σ_e m_e` can only shrink if the
mask is read as re-sizing `m_e`, which would make `χ` monotonically *increase*. If instead `m_e` is
frozen at construction and only the map's support inside it sparsifies (the reading the spec's
"`m` is determined by the mask, set at graph construction" supports), `χ` is a hard constant of the
architecture. Worth being explicit about which reading is intended, because they differ.

**Caveat, stated honestly.** Because this identity is elementary and I found no applied-sheaf
source stating it, it is cited here as mathematics, not as literature support. Nothing in the
sheaf-neural-network literature *contradicts* it; nothing *validates* it either, because nobody
there has needed it.

---

## 3. Genericity of trivial `H^0`

### 3.1 The dimension-count claim — **FALSE as stated**; the counting condition is necessary, not sufficient

**Claim.** When `Σ_e m_e > Σ_v n`, `dim H^0` is generically zero.

`Σ_e m_e ≥ Σ_v n` is **necessary** for `ker δ = 0` (a map cannot be injective into a smaller
space), and for a *fully unstructured* generic matrix of that shape it would also be sufficient.
`δ` is not unstructured. Its block-sparsity pattern is the graph's incidence structure, and that
pattern alone can force a kernel however large the global totals are.

**Local obstruction (provable, one line).** Fix a vertex `v`. Take `x` supported only on `v`.
Then `δx` is supported on `v`'s incident edges, and `x ∈ ker δ` iff `F_{v◁e}x_v = 0` for every
`e ∋ v`. The stacked map `R^n → ⊕_{e∋v} F(e)` has kernel of dimension at least
`n − Σ_{e∋v} m_e`. Since distinct vertices contribute in disjoint coordinates,

```
dim H^0  ≥  Σ_v  max(0,  n − Σ_{e∋v} m_e).
```

This holds **for every** choice of restriction maps — generic, learned, orthogonal, anything.
So a single low-degree, thin-edged cell manufactures global sections no matter how fat the rest of
the graph's edges are. Concretely for Patchworks: **a degree-1 cell whose one edge has `m_e < n`
contributes at least `n − m_e` dimensions to `H^0`, permanently.** Sensory and motor boundary
cells (`CONTEXT.md`) are exactly the low-degree cells in the graph, so this is not a corner case.

The correct global statement is a Hall-type condition on the block pattern rather than a scalar
count: generic injectivity requires, for every subset `S ⊆ V`, that `Σ_{e ∩ S ≠ ∅} m_e ≥ |S|·n`.
The `S = {v}` instances give the local obstruction above; `S = V` gives the asserted scalar count.
Taking only `S = V` is the error.

**Corrected claim, safe to use:** *`dim H^0 = 0` requires `Σ_e m_e ≥ |V|·n` **and** the per-vertex
condition `Σ_{e∋v} m_e ≥ n` for every cell; under those, generic restriction maps give `H^0 = 0`.*

I found no source in the sheaf-neural-network literature that states either the scalar claim or
the corrected one. The nearest thing is Hansen & Ghrist's opinion-dynamics discussion of when a
discourse sheaf "has no nontrivial global sections" (§4), which is treated as a *hypothesis to
check*, never as a consequence of dimension counting — consistent with the correction above.

### 3.2 What happens when maps are LEARNED and SPARSE — **sparsity systematically enlarges `H^0`**, and Patchworks has already named the mechanism

Yes, and provably, not merely empirically.

**Masking is the mechanism, and the spec already describes its own effect.** If a node-stalk
direction `w ∈ R^n` lies in `ker F_{v◁e}` for **every** incident edge `e ∋ v` — which is what a
structural mask does when it excludes a feature from every edge that cell touches — then the
cochain supported on `v` with value `w` is a global section, by the §3.1 argument. So it is in
`H^0`, exactly.

`01-cell-and-sheaf.md` already asks for this, without calling it cohomology:

> "**Features private to a cell's sub-problem** — node stalk components that participate in no
> edge."

**Those private features *are* `H^0`.** Every private feature the mask creates is one dimension of
global sections. That is worth stating plainly because the natural reading of "`H^0` is generically
zero" is that global sections are a degenerate accident to be designed away, whereas the spec
deliberately manufactures them. Both readings can be right at once — the `H^0` a mask creates is
"trivial" in the sense of being *invisible to the graph* (it never reaches an edge stalk, so it can
never be reconciled and never contributes disagreement), and *non-trivial* in the sense of being a
real, intended, load-bearing part of a cell's representation. The sheaf formalism does not
distinguish them; the design does.

**Sparsity pressure pushes further in the same direction.** L1-type pressure drives restriction-map
entries to zero, which drives rank down, which enlarges `ker F_{v◁e}` and hence (via §3.1 and via
the Hall condition) enlarges `H^0`, while simultaneously enlarging `H^1` through non-surjectivity
(§1.4). The Euler characteristic keeps them tied: `dim H^0 − dim H^1` is pinned, so sparsification
grows **both** by the same amount. The system does not trade obstruction for consistency under
sparsification; it buys more of each.

**What the literature actually reports.** The learned-sheaf literature constrains restriction maps
to classes that *cannot* be rank-deficient and therefore never encounters this. Bodnar et al.
build their entire hierarchy out of invertibility: `H_sym` (symmetric invertible), `H^d`
(non-symmetric invertible), `H^d_diag` (diagonal invertible), `H^d_orth` (orthogonal) — every class
in the paper is `det(F_{v◁e}) ≠ 0` by definition (§3.2). Barbero et al. likewise force
`F_{v◁e} ∈ O(d)` by construction (SVD-projection onto the orthogonal group). Under invertibility
the local obstruction of §3.1 is vacuous and the only remaining source of `H^0` is holonomy — which
is why the literature's theory of `H^0` is entirely a theory of holonomy (§4).

**Patchworks is outside every hypothesis class in that literature.** Masked, sparsity-pressured,
independently-learned-at-both-ends linear maps are not invertible, not orthogonal, not symmetric,
and not square (`m_e ≠ n` in general). The consequence is not that the formalism breaks — it does
not; Hansen & Ghrist's construction requires nothing of the maps — but that **none of Bodnar's
`H^0`/`H^1` theorems apply**. In particular Bodnar's Lemma 6 (`dim H^0 ≤ d`, with equality iff
transport is path-independent) is stated for discrete `O(d)` bundles and is silent about the
Patchworks case, where §3.1 shows `dim H^0` can exceed any such bound.

---

## 4. Curvature vs linear restriction maps

**The design argument on the table:** a linear restriction map is honest exactly insofar as the
shared latent structure is locally Euclidean at the scale of the overlap; persistent structured
irreducible disagreement is the signature of curvature the linear map cannot follow.

**Verdict: the literature supports this reading, quite strongly and from two independent
directions — with one important caveat about what "curvature" is allowed to mean, and one
empirical result that should temper enthusiasm.**

### 4.1 Direct support: the linear map *is* a discrete parallel transport, and its error term *is* curvature

Barbero et al. build restriction maps by assuming the data lies on a manifold and estimating
parallel transport:

> "If we constrain the restriction maps in the sheaf to belong to the orthogonal group (i.e.
> `F_{v◁e} ∈ O(d)`), the sheaf becomes a discrete `O(d)`-bundle and can be thought of as a
> discretised version of a tangent bundle on a manifold." (§3)

They inherit the construction from Singer & Wu, who state its validity condition and its failure
mode in exactly the terms the design argument uses. On why the alignment map is only an
approximation (§2):

> "The subspaces, however, are usually not exactly the same, **due to curvature**. As a result,
> the matrix `O_i^T O_j` is not necessarily orthogonal."

And on the tangent-space estimate itself:

> "If the neighboring points … are located exactly on `T_{x_i}M`, then `rank X_i = rank B_i = d`,
> and there are only `d` non-vanishing singular values. In practice, however, **due to the
> curvature effect, there may be more than `d` non-zero singular values.**"

Singer & Wu prove `O_ij` approximates parallel transport "whenever `x_i` and `x_j` are nearby" —
i.e. the linear map is a valid model of the shared structure **precisely in the locally-Euclidean
regime**, and the residual is curvature. That is the design argument, stated by the source the
sheaf-network literature borrows its geometry from. Note the direction of the discrepancy: the
overflow shows up as **extra rank that the `d`-dimensional linear map must discard**, which is a
structured, non-noise residual — matching "structured irreducible disagreement", not "noise".

### 4.2 Direct support: holonomy is the discrete curvature, and it is *exactly* the obstruction to `H^0`

Bodnar et al. make the link between path-dependence and consistency into theorems.

- **Proposition 4.** "If `F` is a discrete `O(d)` bundle over a connected graph and
  `x ∈ H^0(G, F)`, then for any cycle `γ` based at `v ∈ V` we have `x_v ∈ ker(P_{γ_{v→v}} − I)`."
- **Lemma 6.** "Let `F` be a discrete `O(d)` bundle over a connected graph `G`. Then
  `dim(H^0) ≤ d` and `dim(H^0) = d` **if and only if the transport is path-independent**."
- **Proposition 3 / 5.** Two-sided Cheeger-type bounds tying the spectral gap `λ^F_0` to the
  deviation of transport from path-independence — "the spectral gap of a sheaf Laplacian is indeed
  related to the deviation of the transport maps from being path-independent."

Path-independence is flatness. Non-trivial holonomy around a loop is the discrete curvature of the
connection. Bodnar's Lemma 6 therefore says, in geometric language: **a flat connection has a full
space of global sections; curvature destroys them, and how much it destroys is measured by the
holonomy's fixed subspace.** That is a very close formal match to the design argument's second
half. Singer & Wu describe the same object on the continuous side: path-dependence of the composed
transports is "analogous to the parallel transport operator from differential geometry that depends
on the path connecting two points **whenever the manifold has curvature** (e.g., the sphere)."

**Caveat.** These are `O(d)`-bundle theorems. Patchworks' restriction maps are neither orthogonal
nor invertible (§3.2), so "holonomy" is not defined for them in the strict sense — there is no
composable transport `F^⊤_{v◁e}F_{u◁e}` that is a group element. The *monodromy* generalisation
survives: Seely's recurrent example defines `Φ = W_2^{FB}W_2` for a general (non-orthogonal) linear
feedback loop, and distinguishes **resonance** (`Φ ≈ I`, "the network has no internal
contradictions to resolve") from **internal tension** (`Φ ≈ −I`, "the feedback loop contradicts
itself … learning thus requires resolving two tasks: match the supervision signal and resolve the
internal contradiction"). So the qualitative claim — loop monodromy away from the identity produces
irreducible disagreement — holds for general linear maps. The clean quantitative statements
(Lemma 6, the Cheeger bounds) do not transfer.

### 4.3 Discrete curvature on graphs, and its use in GNNs — supports the *reading*, but a different notion of curvature

Topping et al. develop an edge-based curvature (Balanced Forman, a sharp lower bound on Ollivier–
Ricci) and prove that **negatively curved edges cause bottlenecks and over-squashing**. Their
geometric picture (§3, Fig. 2): a triangle-rich neighbourhood is discrete-spherical (positive),
a 4-cycle grid is discrete-Euclidean (zero), a tree is discrete-hyperbolic (negative).

This is a *different* curvature from §4.1–4.2. Topping's curvature is a property of the **graph's
combinatorics alone** — it does not see the sheaf, the stalks, or the restriction maps. Bodnar's
holonomy curvature is a property of the **restriction maps alone** — it is invisible to the graph.
They are orthogonal quantities and it is easy to conflate them. For Patchworks:

- **Combinatorial curvature (Topping)** governs how fast information mixes — it is a statement about
  the *latency/mixing* consequences of topology, which under `01-cell-and-sheaf.md`'s unit-delay
  rule is literally about ticks.
- **Connection curvature (Bodnar/Singer–Wu)** governs whether the cells can ever agree — it is a
  statement about the *consistency* consequences of what the maps learned.

The design argument in the issue is about the second. Topping is not evidence for it; it is
evidence for the §5 latency/consistency tension.

### 4.4 What the literature does NOT support

**No source states the design argument.** I found no paper claiming that "linear restriction maps
are honest iff the shared latent is locally Euclidean at overlap scale" as a design principle. The
components are all present (transport-approximation error is curvature; holonomy obstructs `H^0`);
the assembly into a *justification for choosing linear maps* is Patchworks'. Cite the components,
not the conclusion.

**Nonlinear restriction maps: essentially unexplored; the nonlinear work is elsewhere in the
pipeline.** The nonlinear sheaf literature keeps `δ` linear and inserts nonlinearity *on the edge
stalk after restriction*: Hansen & Ghrist's nonlinear sheaf Laplacian is `L^Φ_F = δ^⊤ ∘ Φ ∘ δ`
with `Φ` applied edge-wise ([Ferrari 2024, §4.2.1](https://arxiv.org/abs/2403.00337), quoting
Hansen & Ghrist). This is a **weaker** move than nonlinear restriction maps, and it is chosen for a
reason the spec would recognise: with edge potentials `U_e` and `Ψ(x) = Σ_e U_e(δ_e x)`, the
gradient is `δ^⊤ ∘ ∇U ∘ δ`, so "if the potential functions `U_e` are convex, `Ψ` is a Lyapunov
function that ensures stability of the dynamics." Making `δ` itself nonlinear destroys `L = δᵀδ`,
destroys the Hodge decomposition, and destroys the cohomological reading — which is precisely why
`01-cell-and-sheaf.md`'s "**Linear.** All nonlinearity lives inside the cell. This keeps the
cellular-sheaf formalism real" is the same call the literature made. **Validated.**

The same source reports a **negative empirical result** worth recording: the nonlinear
(bounded-confidence) Laplacian "did not outperform other benchmark GNN models or the standard
linear sheaf models" on synthetic edge-pruning tasks it was designed for. Nonlinearity in the
coupling is not free performance.

**Tempering result.** A 2026 intervention study, ["Do Sheaf Neural Networks Use Holonomy? A
Measure–Intervene–Control Study"](https://arxiv.org/abs/2607.19514), reports that learned sheaves
do develop non-trivial loop rotation (triangle-weighted mean SO(2) loop rotation rising from 0.010
to 0.388 rad) and that ablating it to identity hurts — but also that a ridge predictor on plain
graph summaries beats the sheaf model, that diagonal maps do as well as full rotations, and that
some settings develop rotation without beating a training-mean baseline. Their conclusion is that
holonomy is **not the primary driver of performance** in existing sheaf networks. This does not
refute the design argument — it says nobody has yet shown holonomy earns its keep empirically.
Patchworks should not lean on "the literature shows curvature matters"; it does not.

---

## 5. Hub / wheel topology and `H^1`

**Claim.** A single high-dimensional hub connected to many cells in an otherwise ring-like graph
**enlarges** `H^1` while shortening diameter: harder global consistency, bought latency.

**Verdict: confirmed, and the Euler characteristic makes it exact — with one condition on "high-
dimensional" that Patchworks' fixed global `n` cannot satisfy.**

### 5.1 The exact bookkeeping

Take a ring of `N` cells, then add a hub `h` with stalk dimension `n_h`, joined to every rim cell
by an edge of dimension `m`. By §2, `χ = dim H^0 − dim H^1 = Σ_v dim F(v) − Σ_e dim F(e)`. Adding
the hub adds one vertex (`+n_h`) and `N` edges (`−N·m`):

```
Δχ = n_h − N·m        so       Δ(dim H^1) = N·m − n_h + Δ(dim H^0).
```

Since `dim H^0` cannot increase when edges are added (adding constraints can only shrink the
solution space), `Δ(dim H^0) ≤ 0`, hence

```
Δ(dim H^1)  ≥  N·m − n_h.
```

**So the wheel enlarges `H^1` by at least `N·m − n_h` — the claim, made exact.** And this is fixed
at graph-construction time by the mask, before any training: no choice of restriction maps can
avoid it.

**The caveat is the interesting part.** The bound is only positive when `n_h < N·m`. A hub whose
stalk is *genuinely* high-dimensional — `n_h ≥ N·m`, wide enough to hold everything all its spokes
carry — adds **no** obstruction, and can add none. That is the honest reading of "high-dimensional
hub". Under `01-cell-and-sheaf.md`, `n` is a **global constant**, "fixed and intended to stay
fixed", deliberately excluded from the flex-priority ladder. So a Patchworks hub has `n_h = n`, and
for any hub of appreciable degree `N·m ≫ n`. **Patchworks cannot build the kind of hub that would
be harmless.** The claim holds in Patchworks with room to spare, and it holds *because* of the
fixed-`n` commitment — the two decisions are coupled, which is worth recording.

Sanity check against the topological case: constant sheaf `R^d`, wheel with `N` rim vertices,
`|V| = N+1`, `|E| = 2N`, `m = n = d`. Then `χ = d(N+1) − 2dN = d(1−N)`, and `dim H^0 = d`
(connected), so `dim H^1 = dN`. This matches `d·β_1` with first Betti number
`β_1 = |E| − |V| + 1 = N`. The bare ring has `β_1 = 1`. **The hub multiplies the cycle rank from 1
to `N`** — "many independent cycles", confirmed. Note the wheel adds `N−1` cycles, not `N` edges'
worth minus one; the arithmetic above is the general-sheaf version and reduces correctly.

Diameter: ring `⌊N/2⌋` → wheel `2`. Under `01-cell-and-sheaf.md`'s unit-delay rule, that is a
literal drop from `⌊N/2⌋` ticks to 2 ticks for any pair. **"Buys latency" is confirmed and is, in
Patchworks, a statement about wall-clock ticks rather than an abstraction.**

### 5.2 What the literature says about hubs, virtual nodes, master nodes

**Nothing in the sheaf-neural-network literature.** Searched across Hansen & Ghrist (both papers
and the thesis), Bodnar et al., Barbero et al. (both papers), and the nonlinear/attention follow-ups:
**no treatment of hub nodes, global nodes, virtual nodes, or master nodes, and no analysis of their
effect on sheaf consistency, `H^0`, or `H^1`.** This appears to be a genuine gap. Hansen's thesis
comes closest from an unexpected angle — §8.3 notes that consensus communication cost "is
proportional to `dim C^1(G; F)`", so a hub's `N` spokes cost `N·m` per round; the same quantity
that inflates `H^1` inflates the bandwidth bill. That is one quantity charging twice, and it is the
sharpest argument against hubs I found in the sheaf literature.

**The non-sheaf virtual-node literature is well developed and points the other way — on the
latency axis only.** Southern et al. prove (Theorem 3.1) the exact change in commute time from
adding a virtual node,
`τ_vn(i,j) − τ(i,j) = 2|E| Σ_{ℓ≥1} [1/(λ_ℓ(λ_ℓ+1))]·[(n/|E|)λ_ℓ − 1]·(v_ℓ(i) − v_ℓ(j))²`, and note
that "for many real-world graphs adding a VN reduces the overall commute time" — but not always:
"there are cases, e.g. when the graph is complete, where (6) is positive". They also note VNs are
"often added to alleviate oversquashing, by reducing the diameter of the graph to 2", and prove
(Proposition 4.1) that the resulting influence is **uniform**: "the Jacobian
`∂h_i^{(ℓ+1)}/∂h_k^{(ℓ−1)}` is independent of `k` whenever `k` and `i` are separated by more than 2
hops" — a virtual node gives "a global yet homogeneous update", unable to weight distant cells
differently.

That uniformity result is directly relevant to Patchworks' **relay cell**, which `CONTEXT.md`
defines as "a cell whose inference is the identity … existing to provide a shared metric space for
distant cells" and explicitly warns _Avoid_: hub. A relay cell is **not** a virtual node: it has
its own `n`-dimensional stalk and its own learned restriction map per incident edge, so its
influence on each neighbour is shaped by that map rather than uniform. Southern et al.'s
Proposition 4.1 is therefore an argument *for* the relay-cell design over a classical virtual node
— the sheaf structure supplies exactly the per-neighbour heterogeneity they had to hand-engineer
(their `VN_G` variant). Worth recording as validation of a distinction `CONTEXT.md` already draws
on vocabulary grounds.

### 5.3 The tension, stated cleanly

Combining §4.3 and §5.1, the two curvatures pull in opposite directions on a hub:

| | ring | ring + hub |
|---|---|---|
| Diameter / ticks to cross | `⌊N/2⌋` | `2` |
| Combinatorial curvature (Topping) | ≈0 (cycle) | more positive — spokes + rim edges close triangles |
| Over-squashing / commute time | worse | better (Southern Thm 3.1, typically) |
| Cycle rank `β_1` | `1` | `N` |
| `dim H^1` | small | `≥ dim H^1_ring + N·m − n_h` |
| `dim C^1` (per-round bandwidth, Hansen thesis §8.3) | `N·m` | `2N·m` |

**A hub improves every mixing quantity and degrades every consistency quantity.** That is the
claim, and both halves are supported — the mixing half by Topping and Southern directly, the
consistency half by the index formula. No source states the trade-off; it falls out of putting two
literatures side by side, and it is not, as far as I can find, anywhere in print.

---

## Summary of verdicts

| # | Assertion | Verdict |
|---|---|---|
| 1a | `(δx)_e = F_{v◁e}x_v − F_{u◁e}x_u` | **Confirmed** (H&G §2.2.2; Seely Eq. 8). Sign is an orientation convention. |
| 1b | `L = δᵀδ` | **Confirmed** (H&G §3.2; Bodnar Def. 2). |
| 1c | `xᵀLx = Σ_e ‖disagreement_e‖²` | **Confirmed** (Hansen thesis §1.7; Bodnar Def. 14). |
| 1d | `H^0 = ker δ = ker L = global sections` | **Confirmed** (H&G Def. 3, Thm. 1, Eq. 2.6; Bodnar §2). |
| 1e | `H^1 = coker δ` | **Confirmed** (Seely Eq. 10). |
| 1f | `H^1` supported on cycles, zero for a tree | **FALSE as stated.** True only if all restriction maps are surjective. Patchworks' masked/sparse maps are not. `H^1` has a second, rank-based source invisible to graph topology. |
| 2 | `dim H^0 − dim H^1 = Σ_v dim F(v) − Σ_e dim F(e)`, map-independent, fixed by the mask | **Confirmed** by rank–nullity. Not stated anywhere in the applied-sheaf literature I read — cited as mathematics, not as literature support. |
| 3a | `Σ_e m_e > Σ_v n` ⟹ `H^0` generically zero | **FALSE as stated.** Necessary, not sufficient. Per-vertex condition `Σ_{e∋v} m_e ≥ n` also required; low-degree thin-edged cells force `dim H^0 ≥ Σ_v max(0, n − Σ_{e∋v} m_e)` for *any* maps. |
| 3b | Does sparsity enlarge `H^0`? | **Yes, provably.** Masked-out node-stalk directions are literally global sections — they are the spec's own "features private to a cell's sub-problem". By §2 they enlarge `H^1` by the same amount. Every learned-sheaf paper assumes invertible or orthogonal maps and so never sees this; Patchworks is outside all their hypothesis classes. |
| 4 | Linear map honest iff locally Euclidean; irreducible disagreement = curvature | **Supported, from two independent directions** (Singer & Wu: transport-approximation error *is* curvature; Bodnar Lemma 6: flat ⟺ maximal `H^0`). No source states the design argument itself. `O(d)` theorems don't transfer to non-orthogonal maps; Seely's monodromy generalisation does, qualitatively. Nonlinear-`δ` alternative is unexplored and the nonlinear-`Φ` variant underperforms. A 2026 intervention study finds holonomy is *not* the driver of sheaf-network performance. |
| 5 | Hub enlarges `H^1` while shortening diameter | **Confirmed, exactly:** `Δ(dim H^1) ≥ N·m − n_h`. Fails only for `n_h ≥ N·m`, which Patchworks' globally fixed `n` forbids — so the claim holds in Patchworks *because* `n` is fixed. Ring `β_1 = 1` → wheel `β_1 = N`. **No sheaf literature on hubs/virtual/master nodes exists**; the virtual-node literature (Southern et al.) confirms the latency benefit and, via its uniformity result, argues for relay cells over virtual nodes. |
