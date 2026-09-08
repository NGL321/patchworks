# Discovering what two views share, rather than allocating it (patchworks#574, B25)

Part of map [#532](https://github.com/NGL321/patchworks/issues/532). Opened by
[B17 / #565](https://github.com/NGL321/patchworks/issues/565)'s resolution. Findings feed B27; this
pass **does not rule on the architecture**.

Standing instruction for the batch: *"look at the literature to see what supports the things we've
described, without constraining on those topics we have not described."* Accordingly this document
leads with the places the literature is **better posed than our framing**, and reports negatives
first inside each section.

## Verification key

Each citation is marked with how far it was checked.

- **[V]** — abstract or full text fetched in this pass; claims below are the paper's own.
- **[S]** — existence, authorship and venue confirmed from at least two independent listings
  (publisher page, proceedings page, arXiv listing); content summarised from those listings rather
  than from the full text.
- **[U]** — asserted from background knowledge, not verified in this pass. Treat as a lead, not a
  citation.

Anywhere a paraphrase might be read as stronger than the source, the source's own scope is stated in
a *what it actually says* line.

---

## Headline: our object already has a name, and it is not "two views"

The single most consequential finding of this pass is that the architecture as described in #574 is,
almost line for line, a **cellular sheaf on a graph**, and that the objective we run — descend
disagreement between the two ends of an edge — is **sheaf diffusion / the sheaf Laplacian heat
equation**. This is not an analogy offered here; it is the standing definition in that literature.

- **Hansen & Ghrist, "Opinion dynamics on discourse sheaves", SIAM Journal on Applied Mathematics
  81(5):2033–2060, 2021 (DOI 10.1137/20M1341088).** **[S]** Each agent holds a *private* opinion in
  a node stalk; each edge carries a *discourse space* (edge stalk) in which the two agents' opinions
  are compared; restriction maps express a private opinion into the shared discourse space of that
  edge. The dynamics are gradient descent on total disagreement.
  *What it actually says:* this is the framework and the dynamics; the paper's own targets are
  social phenomena (deception, propaganda, when consensus is or is not reached). It is a modelling
  paper, not a representation-learning result. I did not obtain clean verbatim theorem statements
  (the PDF fetch returned a summary I judged unreliable), so **theorem numbers are deliberately not
  quoted here**; see "what we did not find".
- **Seely, "Sheaf Cohomology of Linear Predictive Coding Networks", arXiv:2511.11092, NeurIPS 2025
  Workshop on Symmetry and Geometry in Neural Representations.** **[V]** States directly that a
  linear predictive-coding network *is* a cellular sheaf: "the sheaf coboundary maps activations to
  edge-wise prediction errors, and PC inference is diffusion under the sheaf Laplacian." Uses
  cohomology to characterise **irreducible error patterns** that inference cannot remove, and Hodge
  decomposition to say when recurrent feedback creates internal contradictions that generate
  prediction error independent of the supervision signal.
  *What it actually says:* linear PC networks only; workshop paper. But it is the closest published
  object to ours — predictive-coding lineage, disagreement as the transported quantity, sheaf as the
  formalism — and it says the fixed points and the unremovable residue are **cohomological**, i.e. a
  property of the restriction maps and the graph, not of the learning rate.

Why this matters more than any individual CCA result: it means the questions in #574 have a
**precise** form, and two of them already have answers in that form.

- "What travels the lane" = a 1-cochain; the edge stalk **is** the lane.
- "The subspace of features that mean the same thing in both cells" = the edge stalk together with
  its two restriction maps.
- "Consistency around cycles" = **flatness / trivial holonomy**, and its failure is measured by
  \(H^1\).
- "What disagreement descent converges to" = \(\ker L_{\mathcal F} \cong H^0(G;\mathcal F)\), the
  space of global sections.

That last equality is the answer to area 4 and is stated in §4.

---

## 1. Shared-versus-private decomposition of two views

### 1.1 The lineage

- **Hotelling, "Relations between two sets of variates", Biometrika 28(3/4):321–377, 1936.** **[U]**
  Canonical correlation analysis. Maximally correlated pairs of directions, one in each view.
- **Bach & Jordan, "A probabilistic interpretation of canonical correlation analysis", UC Berkeley
  Dept. of Statistics Technical Report 688, 2005.** **[U]** CCA as maximum likelihood in a latent
  variable model with a shared latent plus per-view noise. This is the ancestor of every
  shared/private factorisation below. *Not verified in this pass — treat as a lead.*
- **Andrew, Arora, Bilmes & Livescu, "Deep Canonical Correlation Analysis", ICML 2013.** **[S]**
  Nonlinear encoders, CCA objective on the outputs.
- **Wang, Arora, Livescu & Bilmes, "On Deep Multi-view Representation Learning", ICML 2015
  (DCCAE).** **[U]** Adds reconstruction to DCCA. *Lead only.*

### 1.2 Explicit shared/private factorisations

- **Salzmann, Ek, Urtasun & Darrell, "Factorized Orthogonal Latent Spaces", AISTATS 2010, PMLR
  9:701–708.** **[S]** The canonical statement of the problem *as a degeneracy problem*. The paper's
  own framing (from the PMLR abstract) is that it "proposes a method to learn shared and private
  latent spaces that are inherently disjoint by introducing orthogonality constraints", and that
  this improves on "existing shared-private factorizations". The failure modes it exists to fix are
  the ones §2 lists.
  *What it actually says:* orthogonality (plus, in the full method, non-redundancy and energy terms)
  makes the split well posed. It does **not** claim to select the dimensions automatically; a fetch
  of the PDF returned an unreliable paraphrase on that point, so treat dimension selection here as
  **unresolved** — the surrounding literature (below) treats it as a hyperparameter or via ARD.
- **Damianou, Ek, Titsias & Lawrence, "Manifold Relevance Determination", ICML 2012
  (arXiv:1206.4610).** **[S]** ARD priors over latent dimensions, *per view*. A dimension whose
  relevance weight survives for both views is shared; one that survives for only one view is
  private. **This is the cleanest existing answer to "how is the shared dimension chosen": it is not
  chosen, it is switched off.** The shared dimension emerges as the count of dimensions the
  optimiser declines to prune in either view.
- **Klami, Virtanen & Kaski, "Bayesian Canonical Correlation Analysis", JMLR 14:965–1003, 2013.**
  **[S]** Introduces a **group-wise ARD prior** that "automatically identifies variable-specific and
  shared components", which is what made Bayesian CCA usable in high dimensions. Same mechanism as
  MRD, in the linear CCA setting.
- **Märtens & Yau, "Disentangling shared and private latent factors in multimodal Variational
  Autoencoders", arXiv:2403.06338, MLCB 2023.** **[V]** Reports that existing multimodal VAEs (MVAE,
  MMVAE) **fail** in the regime "where modality-specific variation dominates the shared signal", and
  proposes a modification for robustness to that.
  *What it actually says:* a failure regime, characterised by the *ratio* of private to shared
  variance — not by the dimension count. This is a directly relevant negative: the shared subspace is
  hardest to find exactly when the two views are mostly about different things, which is our
  designed condition (heterogeneous cells, deliberately no global agreement).

### 1.3 How the shared dimension is chosen — the honest summary

Across the sources checked, four regimes, in rough order of how principled they are:

1. **ARD / structured sparsity (learned, by pruning).** MRD **[S]**, Bayesian CCA **[S]**, Group
   Factor Analysis **[S]** (§3). You over-provision the latent space and let per-view relevance
   weights kill dimensions. The shared dimension is a *read-out* of the fitted model, not an input.
2. **Grid search / cross-validation.** Widely used; the model is refit for each (shared, private)
   pair and selected on held-out likelihood or validation loss. Confirmed as the dominant practice in
   the applied shared/private literature surveyed **[S]**, e.g. neuroscience latent-dynamics work.
3. **Penalised, by orthogonality and non-redundancy.** FOLS **[S]**. This makes the *split* well
   posed at a fixed total dimension; it does not choose the split point.
4. **Fixed by hand.** The default in most deep multimodal work.

I found **no** source in this pass that anneals the shared dimension during training, and none that
allocates it by a global fair-division rule across many pairs. **Our global allocator has no
counterpart in this literature.** That is a gap in the literature's coverage of our design as much as
it is a criticism of the design — see "what we did not find".

### 1.4 What happens when the shared dimension is wrong

- **Too large.** The classical CCA answer is sharp and is the strongest negative in this section: as
  the per-view dimension approaches or exceeds the sample size, CCA finds **spurious canonical
  correlations that are near-perfect on the sample and near-zero out of sample** — the sample
  cross-covariance can be made to correlate arbitrary data. This is why regularised, sparse and
  reduced-rank CCA exist at all. **[S]** (confirmed across the CCA-in-neuroimaging tutorial
  literature and the sparse/robust CCA papers, which all open with this problem; specific citations:
  the CCA/PLS tutorial and comparative study in *Biological Psychiatry: CNNI* 2022 **[S]**, and the
  high-dimensional CCA literature, e.g. arXiv:2306.16393 **[S]**).
  **Read across to us:** an over-wide lane will be *filled*, and the filling will measure as
  well-correlated in-sample while carrying nothing. This is a candidate explanation for
  #574's third bullet — learned carried subspaces statistically indistinguishable from random frames
  of the same shape is exactly what an over-provisioned, under-constrained shared space looks like.
  It does not by itself prove that is what happened.
- **Too small.** No crisp theorem was found. The nearest is Märtens & Yau **[V]**: when the shared
  capacity is inadequate relative to modality-specific variance, the shared code is captured by the
  dominant modality rather than by what is genuinely common. Also Daunhawer et al. ICLR 2022 (§2).

---

## 2. The degeneracies — where this collapses

This is the section #574 asked to weight most heavily, and the literature is unusually clear.

### 2.1 The invariance term alone is degenerate. This is not controversial.

Every modern two-view method that works states, as motivation, that minimising disagreement between
two views' representations **has a trivial global optimum**: both encoders emit a constant. The
entire design space of joint-embedding SSL is the space of devices for excluding it.

- **Bardes, Ponce & LeCun, "VICReg: Variance-Invariance-Covariance Regularization for
  Self-Supervised Learning", ICLR 2022 (arXiv:2105.04906).** **[S]** Three terms: *invariance*
  (the disagreement term), *variance* (a hinge keeping each embedding coordinate's standard
  deviation above a threshold — this is the explicit anti-collapse device), *covariance* (penalises
  off-diagonal entries of the embedding covariance, decorrelating coordinates). The paper's own
  words for what the last two prevent is **"informational collapse"**.
- **Zbontar, Jing, Misra, LeCun & Deny, "Barlow Twins: Self-Supervised Learning via Redundancy
  Reduction", ICML 2021.** **[S]** Drives the cross-correlation matrix between the two views toward
  the identity: on-diagonal → invariance, off-diagonal → non-redundancy. VICReg explicitly borrows
  the covariance criterion from it. Note the shape: **the identity target is a full-rank target**;
  the anti-collapse mechanism is a rank constraint in disguise.
- **Jing, Vincent, LeCun & Tian, "Understanding Dimensional Collapse in Contrastive Self-supervised
  Learning", ICLR 2022 (arXiv:2110.09348).** **[S]** Distinguishes **complete collapse** (constant
  embedding) from **dimensional collapse** (embeddings span a strict subspace of the available
  space, visible as vanishing singular values of the embedding covariance). Shows contrastive
  methods, which are immune to complete collapse, still suffer dimensional collapse.
  **This is the distinction our instrument needs.** A composed operator's spectrum collapsing to one
  direction is the *operator-side* analogue of dimensional collapse — and #574 already records that
  this instrument never touches the state, so it cannot distinguish "the map can only carry one
  direction" from "the state only ever occupied one".
- **Tian, Chen & Ganguli, "Understanding self-supervised learning dynamics without contrastive
  pairs", ICML 2021.** **[U]** The predictor + stop-gradient + EMA analysis of BYOL/SimSiam.
  *Lead only — not verified in this pass.*

**The load-bearing conclusion for us:** an objective that only descends disagreement is, in the
two-view literature, *the known-degenerate baseline*. Nobody runs it alone. Every working method
pairs it with a term that puts a **floor on the variance or the rank of what each end retains**.

### 2.2 Rank collapse of composed operators

- **Dong, Cordonnier & Loukas, "Attention is not all you need: pure attention loses rank doubly
  exponentially with depth", ICML 2021, PMLR 139 (arXiv:2103.03404).** **[S]** Decomposes the output
  of a stack of self-attention layers into paths; proves that **without skip connections or MLPs the
  output converges doubly exponentially to a rank-1 matrix**, and that skip connections and MLPs are
  what arrest the degeneration.
  *What it actually says:* about self-attention specifically, where each layer is a row-stochastic
  averaging operator. It is **not** a general theorem about composing arbitrary linear maps.
  **Read across to us, carefully:** our seven-hop composition collapsing to a single direction has
  the same *shape* as this result, and the mechanism proposed there — a composition of averaging
  operators has a shared dominant fixed direction, and depth amplifies it — is a live hypothesis for
  our measurement. But our restriction maps are not row-stochastic, so the theorem does not transfer
  as stated. What does transfer is the **remedy pattern**: the fix was not a better objective, it was
  an architectural term (residual) that keeps the composed operator away from the degenerate
  fixed point.

### 2.3 Graph-level: the same collapse, on a network

- **Oono & Suzuki, "Graph Neural Networks Exponentially Lose Expressive Power for Node
  Classification", ICLR 2020.** **[U]** *Lead only.*
- **Bodnar, Di Giovanni, Chamberlain, Liò & Bronstein, "Neural Sheaf Diffusion: A Topological
  Perspective on Heterophily and Oversmoothing in GNNs", NeurIPS 2022 (arXiv:2202.04579).** **[V]**
  See §3–4; this is the paper that ties oversmoothing to the *geometry of the restriction maps*.

### 2.4 What is not solved by any of the above

Nothing in §2 tells you the **right** shared dimension. The anti-collapse machinery keeps the shared
space from vanishing or from becoming a single direction; it does not tell you whether the surviving
rank is the true shared rank. Every method surveyed still fixes total width by hand or by ARD, and
then defends the floor.

---

## 3. More than two views

### 3.1 The classical extension exists but is the *wrong shape* for us

Generalised CCA (Horst 1961; Carroll 1968; Kettenring, "Canonical analysis of several sets of
variables", Biometrika 1971) **[U]** extends CCA to \(>2\) views. Crucially, all these variants seek
**one common space that all views project into** — a single global agreement space. That is
explicitly what #574 says is *not* the goal ("global agreement is explicitly not the goal"). So the
main line of multi-view CCA is a competitor to our design, not a formalisation of it.

- **Klami, Virtanen, Leppäaho & Kaski, "Group Factor Analysis", IEEE TNNLS 26(9):2136–2147, 2015
  (arXiv:1411.5799).** **[S]** The important partial exception. Extends CCA to more than two sets
  "in a way that is more flexible than previous extensions": a latent variable model with
  **structural sparsity** over a two-level hierarchy, where the upper level models the relationships
  *between the groups*. The practical consequence is that a factor can be **active on an arbitrary
  subset of views**, discovered by the sparsity prior.
  *What it actually says:* the sharing *pattern* over views is discovered, not specified. But it is
  still **one global set of factors** with per-view activity — not a per-edge subspace. Two cells
  that share something no one else has would need a factor of their own, which the model permits;
  what it does not give you is a per-edge stalk with its own basis, and it has no notion of a cycle.

### 3.2 The right shape: sheaves, and the cycle question is the flatness question

Cellular sheaves are the formalism where every part of the #574 object is a primitive: node stalks
(a cell's own space), edge stalks (the lane), restriction maps (the translation), disagreement
(the coboundary), and **consistency around cycles as an algebraic invariant**.

- **Bodnar et al., NeurIPS 2022 (above).** **[V]** From the fetched text: the sheaf Laplacian is
  \(L_{\mathcal F}(x)_v := \sum_{v,u \trianglelefteq e} \mathcal F_{v\trianglelefteq e}^\top
  (\mathcal F_{v \trianglelefteq e} x_v - \mathcal F_{u \trianglelefteq e} x_u)\), the space of
  global sections is \(H^0(G;\mathcal F) = \{x : \mathcal F_{v\trianglelefteq e} x_v =
  \mathcal F_{u \trianglelefteq e} x_u\}\), and \(\ker(L_{\mathcal F}) \cong H^0\).
  Its results are about **linear separation power in the infinite-time limit** as a function of the
  class of sheaf allowed, with a hierarchy: diagonal-symmetric \(\subset\) symmetric-invertible
  \(\subset\) orthogonal \(\subset\) general \(d \times d\). Reported results include: symmetric
  invertible maps in \(d=1\) cannot separate classes in the heterophilic case; \(d=1\) general maps
  cannot separate \(C \ge 3\) classes; \(d \ge C\) suffices for the diagonal family; and the
  **orthogonal** family attains \(C \le 2d\) — i.e. **orthogonal restriction maps buy roughly twice
  the separation per unit of stalk width**.
  *What it actually says:* these are statements about asymptotic linear separability of node classes
  under diffusion, for a node-classification task. They are **not** statements about how much signal
  a composed multi-hop map carries. Do not read them as our reach metric.
  **But two things do read across.** (i) The **stalk dimension \(d\) is the binding resource**, and
  the required \(d\) scales with how many distinct things must remain distinguishable at the fixed
  point. (ii) **The family the restriction maps are drawn from changes the answer by a constant
  factor** — orthogonal beats diagonal beats scalar. If our maps are learning toward
  near-singular or near-aligned frames, the relevant lever is a *constraint on the family*, not more
  width.
  On learning: they learn each \(d\times d\) map as \(\mathcal F_{v\trianglelefteq e} =
  \Phi(x_v, x_u)\), a parametric matrix-valued function of **both endpoints' features** — i.e. the
  map is **per-pair by construction**, computed from the pair. That is the mechanism #574 says we
  lack.
- **Hansen & Ghrist, "Learning Sheaf Laplacians from Smooth Signals", ICASSP 2019, pp. 5446–5450.**
  **[S]** Existence and venue confirmed; full text not obtained. Infers the sheaf (topology and
  restriction maps) from data by an optimisation over the sheaf Laplacian, via semidefinite
  programming. This is the *discovery* problem posed at network scale.
- **Di Nino, Barbarossa & Di Lorenzo, "Learning Sheaf Laplacian Optimizing Restriction Maps",
  arXiv:2501.19207, 31 Jan 2025.** **[V]** Abstract verbatim in part: infers "the sheaf Laplacian,
  including the topology of a graph and the restriction maps, from a set of data observed over the
  nodes"; the problem "aims to find the sheaf Laplacian that **minimizes the total variation** of the
  observed data, where the variation over each edge is also locally minimized by optimizing the
  associated restriction maps"; closed-form steps, faster than the SDP alternative. Tested "on data
  consisting of **vectors defined over subspaces of varying dimensions at each node**", and the
  resulting graph is shown to be driven by "the **cross-correlation** and the **dimensionality
  difference** of the data residing on the graph's nodes."
  **This is the closest published thing to per-pair lane discovery over a network that this pass
  found.** Note precisely what it optimises: *total variation*, i.e. our objective, over both the
  state and the maps. Note also the two things it finds control the outcome: pairwise
  cross-correlation, and **dimensionality mismatch between the two ends** — the second is a variable
  our fixed, identical-across-edges face does not have.
- **Grimaldi et al., "A Sheaf-Theoretic Framework for Distributed Multi-Site Channel Charting",
  arXiv:2607.03480, July 2026.** **[V]** A network of independently learned local representations
  (one per base station) stitched by a sheaf: node stalks are each site's embedding, **edge stalks
  are embeddings in the overlapping region between a pair of sites**, restriction maps are orthogonal
  and obtained by Procrustes/Kabsch alignment on the overlap.
  The critical design choice: they do **not** learn independent per-edge maps. They use a **flat
  bundle factorisation**, \(\mathcal F_{b_i \trianglelefteq (b_i,b_j)} = R_{b_j}^\top R_{b_i}\),
  with each node carrying one \(SO(n)\) reference frame. The stated reason: without such a
  factorisation "the sheaf does not guarantee the existence of non-trivial global sections", and the
  factorisation also drops cost from \(O(B^2 n^2)\) to \(O(B n^2)\).
  *What it actually says:* the fetched summary reports the paper does **not** explicitly discuss
  cycle consistency, and fixes \(n=2\) for interpretability. So this is an engineering choice with a
  stated motivation, not a theorem.
  **This is the most architecturally consequential finding in §3.** A flat sheaf is precisely one
  whose holonomy around every cycle is the identity — cycle-consistent by construction. And the
  construction is a *middle term* between our two poles: not "one fixed face shown to everybody", and
  not "an unconstrained free map per edge", but **one per-cell frame, with the per-edge map generated
  as the composition of the two endpoints' frames.** Under that scheme the lane is per-pair (it
  depends on both endpoints) while global sections are guaranteed to exist.

### 3.3 Cycle consistency has its own literature, outside sheaves

The requirement "compositions of maps around a cycle should be the identity" is standard in
multi-view geometry and shape matching, where it is called **cycle consistency** or
**synchronization**, and it is generally enforced by a **low-rank / PSD** condition on the big block
matrix of all pairwise maps.

- **Huang & Guibas, "Consistent shape maps via semidefinite programming", SGP / Computer Graphics
  Forum 2013.** **[S]** Casts cycle consistency as an SDP: the matrix of all pairwise maps is
  cycle-consistent iff it is low-rank PSD; recover by projecting onto that set.
- **Huang, Wang & Guibas, "Functional Map Networks for Analyzing and Exploring Large Shape
  Collections", ACM TOG 33(4), SIGGRAPH 2014.** **[S]** A network of pairwise maps between many
  objects, optimised jointly under cycle consistency; the shared structure across the collection
  emerges as a **latent basis** whose size is a rank parameter.
- **Pachauri, Kondor & Singh, "Solving the multi-way matching problem by permutation
  synchronization", NIPS 2013.** **[S]**
- Related: robust group synchronization via cycle-edge message passing (*Found. Comput. Math.*,
  2021) **[S]**; low-rank nuclear-norm relaxations (MatchALS) **[S]**.

*What this literature actually says, and its limit for us:* consistency around cycles is
**achievable and computationally well understood**, and the standard mechanism is a **global
low-rank constraint** on the ensemble of pairwise maps — equivalently, all the pairwise maps
factoring through a common latent object, which is exactly the flat-bundle trick in §3.2. What this
literature does **not** do is let each pair share a *different* subspace: it wants the maps to be
full correspondences (permutations, orthogonal maps, functional maps) between objects of the same
type. Partial and mismatched-dimension versions exist (partial permutation synchronization **[S]**)
but they are the harder, less settled end.

---

## 4. Does descending disagreement *find* a shared subspace, or degenerate onto one direction?

**Answer: on its own, it degenerates, and this is known in three independent literatures.**

1. **Two-view SSL.** The invariance term alone has a constant global optimum; every deployed method
   adds a variance floor (VICReg **[S]**), a decorrelation/identity-target term (Barlow Twins
   **[S]**), negatives, or a predictor + stop-gradient. Even the methods that escape *complete*
   collapse still suffer **dimensional** collapse (Jing et al. **[S]**).
2. **Sheaf diffusion.** The disagreement objective is the Dirichlet energy of the sheaf; its gradient
   flow is diffusion under \(L_{\mathcal F}\); and each channel is **projected into
   \(\ker(L_{\mathcal F}) \cong H^0(G;\mathcal F)\)** in the infinite-time limit (Bodnar et al.
   **[V]**, verbatim: "each feature channel is projected into ker(Δ_F)").
   **This is a complete answer to the question as posed.** The objective does not *find* a shared
   subspace; it *reveals* the one the restriction maps already define, and drives everything else to
   zero. The dimension of what survives is \(\dim H^0\), a fixed combinatorial-algebraic property of
   (graph, stalks, restriction maps). If our maps give a low-dimensional \(H^0\), then a single
   surviving direction is the **guaranteed** outcome, not a training pathology, and no amount of
   optimisation changes it.
   Bodnar et al. **[V]** are explicit that this is exactly the oversmoothing story, that the trivial
   sheaf (ordinary GNN) has the smallest such kernel, and that a **non-trivial sheaf enlarges it** —
   "we prove that when the sheaf is non-trivial, discretised parametric diffusion processes have
   greater control than GNNs over their asymptotic behaviour."
3. **Joint optimisation of state and maps is worse.** When the restriction maps are *also* trained by
   the same disagreement objective, the objective is minimised by maps that annihilate everything.
   The sheaf-learning literature therefore never optimises total variation unconstrained: Di Nino et
   al. **[V]** locally optimise restriction maps inside a constrained Laplacian-inference problem
   (with closed-form steps and normalisation), and Hansen & Ghrist **[S]** solve an SDP over a
   constrained set. Bodnar et al. **[V]** sidestep it entirely by *not* training the maps on the
   diffusion loss alone — the maps are outputs of a learned function \(\Phi(x_v,x_u)\) trained on a
   downstream task, and they report that restricting \(\Phi\)'s image to **orthogonal** matrices is
   what buys separation power.

**Direct consequence for the architecture.** The pairing of (a) a disagreement-descent objective
with (b) restriction maps trained by that same objective and (c) no variance/rank floor anywhere is
precisely the configuration that all three literatures identify as degenerate. The observed
seven-hop collapse and the "learned carried subspaces indistinguishable from random" measurement are
both consistent with that configuration — though neither is proof, and #574 correctly notes the
spectral instrument does not touch the state.

---

## 5. The "compression is all you need" reference

**I could not identify a single intended reference with confidence.** There are at least five live
candidates and they argue different things. Below is what each actually claims, and whether it bears
on "each cell as a minimal compressed statement of what its features determine."

1. **Aksenov, Bodnia, Freedman & Mulligan, "Compression is all you need: Modeling Mathematics",
   arXiv:2603.20396, 20 March 2026.** **[V]** — *the only exact title match found.* Argues that
   *human* mathematics is the compressible sliver of formal mathematics, distinguished by
   hierarchically nested definitions/lemmas/theorems. Models a deduction as a string in a monoid and
   a theorem as a named macro that compresses it. Result: in the **free abelian** monoid a
   logarithmically sparse macro set gives **exponential** expansion of expressivity; in the **free
   non-abelian** monoid even a polynomially dense macro set gives only **linear** expansion.
   Validated against Lean 4's MathLib: unwrapped length grows exponentially with depth and with
   wrapped length, while wrapped length is roughly constant across depths — consistent with the
   abelian model.
   **Bearing on us: substantial, and not the bearing one would guess.** Two things. (i) It is a
   quantitative claim that *reuse of a small vocabulary of named intermediate results* is what makes
   a domain tractable — which is the compositionality bet, stated as a measurable property of a
   dependency graph. (ii) The abelian/non-abelian split is a claim that **order-insensitivity is what
   buys the exponential gain**; where composition is order-sensitive you get linear returns. Our
   composition of restriction maps along a path is emphatically non-abelian. If one takes the
   analogy seriously it is a *warning*, not a support.
   Caveat: it is a very recent preprint about mathematics libraries, not about representation
   learning. Do not cite it as evidence about neural architectures.
2. **Delétang, Ruoss, Duquenne, Catt, Genewein, Mattern, Grau-Moya, Wenliang, Aitchison, Orseau,
   Hutter & Veness, "Language Modeling Is Compression", ICLR 2024 (arXiv:2309.10668).** **[S]** The
   most-cited thing in this space. Argues prediction and compression are equivalent (arithmetic
   coding), and evaluates foundation models as general-purpose compressors: Chinchilla 70B compresses
   ImageNet patches to 43.4% and LibriSpeech to 16.4%, beating PNG (58.5%) and FLAC (30.3%).
   **Bearing on us: weak.** It is about predictors as compressors of *data streams*. It says nothing
   about a representation being a minimal statement of what a feature set determines.
3. **Ma, Tsao & Shum, "On the Principles of Parsimony and Self-Consistency for the Emergence of
   Intelligence", Frontiers of Information Technology & Electronic Engineering, Sept 2022
   (arXiv:2207.04630).** **[S]** Two principles: *parsimony* (what to learn — a maximally compressed,
   structured representation; realised as rate reduction over a linear discriminative representation)
   and *self-consistency* (how to learn — a closed-loop transcription that checks the representation
   against the world).
   **Bearing on us: the strongest of the five.** "Parsimony" is almost exactly "the minimal statement
   of what can always be known from these features", and "self-consistency" is a closed loop of the
   sort our reconciliation error implements. If the designer's memory is of an argument rather than a
   title, this is the most likely source of the *idea*.
4. **Jack Rae, "Compression for AGI" (Stanford MLSys talk, 2023).** **[U]** A widely circulated talk
   making the "a good LLM is a good compressor" argument. Not a paper; commonly mis-remembered as a
   titled reference. *Not verified in this pass.*
5. **Hutter / Legg's compression-equals-intelligence line, and the Hutter Prize.** **[U]** The
   original source of the slogan. *Not verified in this pass.*

Also encountered and explicitly **not** the reference: "Attention and Compression is all you need for
Controllably Efficient Language Models" (arXiv:2511.05313) **[S]**, an architecture paper; and an SSRN
item titled "Compression is All you Need" by M. Raeini **[S]**, unrelated to this context.

**Recommendation:** do not build on a citation named "compression is all you need". If the intended
content is "a cell holds a minimal compressed statement of what its features determine", cite Ma,
Tsao & Shum (2022) for parsimony, and the information-bottleneck line for the formal object. If the
intended content is "compression is the right lens on prediction", cite Delétang et al. (2024).

---

## 6. What this says about the architecture

Stated as findings, not as rulings. B27 weighs these against the measurements.

1. **We are running a cellular sheaf, and our fixed point is \(H^0\).** Descending disagreement is
   sheaf diffusion. Its infinite-time limit is the projection onto \(\ker L_{\mathcal F}\). The
   dimension of what survives seven hops is therefore, in the limit, a property of the restriction
   maps and the graph — *not* something more training fixes. Bodnar et al. **[V]**.
2. **"One face shown to every neighbour" is the trivial-sheaf end of a hierarchy that is known to be
   the weakest.** The published hierarchy (diagonal ⊂ symmetric ⊂ orthogonal ⊂ general) shows
   separation power increasing strictly along it, with orthogonal maps giving roughly \(2\times\) the
   separation per unit of stalk width **[V]**. A cheap, well-supported intervention that is *not*
   per-pair discovery: **constrain the restriction maps to be orthogonal.**
3. **Per-pair discovery is implemented in the literature and it is cheap.** Bodnar et al. compute the
   restriction map as \(\Phi(x_v, x_u)\), a learned function of *both* endpoints **[V]**. That is a
   direct, published mechanism for the thing #574 says we lack, and it needs no global allocator.
4. **But full per-edge freedom is known to break global sections.** Grimaldi et al. state that
   without a factorisation constraint the sheaf "does not guarantee the existence of non-trivial
   global sections" **[V]**, and adopt a **flat bundle**: one \(SO(n)\) frame per node, edge map =
   \(R_{b_j}^\top R_{b_i}\). This is a genuine third option between our current design and free
   per-edge maps: **the face is per-cell, but the lane is per-pair because it is generated by two
   faces.** It is cycle-consistent by construction (trivial holonomy) and costs \(O(Bn^2)\).
   **This is the single most actionable finding of the pass.**
5. **The objective as stated is the known-degenerate baseline.** No two-view method runs invariance
   alone; no sheaf-learning method optimises total variation over the maps without a normalisation or
   constraint set. If we train the maps on the disagreement loss with no variance or rank floor, the
   literature predicts what we measured. Adding a VICReg-style variance floor per lane, or a
   Barlow-Twins identity target on the cross-correlation between the two ends of a lane, are the
   off-the-shelf fixes **[S]**.
6. **A global fair-division allocator of lane widths has no counterpart anywhere I looked.** The
   literature's answers are ARD/pruning, cross-validation, or a fixed hyperparameter. Nobody
   allocates shared dimension across many pairs by a fairness rule. That is unexplored territory: it
   may be novel, and it may be novel because the objective it optimises (fairness) is unrelated to
   the objective the pairs care about (how much these two actually share). **Di Nino et al. find the
   two things that determine the edge structure are pairwise cross-correlation and dimensionality
   mismatch** **[V]** — both per-pair quantities that a global allocator cannot see.
7. **An over-wide lane is a known failure with a known signature.** Classical CCA in the
   high-dimension regime finds spurious canonical directions that look perfectly correlated in
   sample and are noise out of sample **[S]**. "Learned carried subspaces statistically
   indistinguishable from random frames of the same shape" is consistent with this. Testable: does
   the in-sample edge agreement generalise to held-out episodes?
8. **A framing challenge worth taking seriously.** #574 says global agreement is explicitly not the
   goal — cells modelling gait need no view on car colour. But the disagreement-descent objective
   *is* a global-agreement objective; its fixed point is the global sections. **Reach and
   disagreement-descent may be in tension by construction**: the objective's own optimum is total
   agreement, and everything not in \(H^0\) is what it is built to destroy. Two possible readings:
   (a) the interesting regime is transient rather than asymptotic, and the target is the *rate* of
   propagation rather than the fixed point — in which case the seven-hop spectrum is the wrong
   instrument; or (b) \(H^0\) must be made deliberately large and structured, which is a design
   constraint on the restriction maps, and Bodnar et al.'s hierarchy is the map of how large you can
   make it per unit of stalk width **[V]**. Both readings deserve to be put to B27.
9. **Also worth putting to B27:** Märtens & Yau's failure regime — shared-subspace discovery fails
   when modality-specific variation dominates the shared signal **[V]** — is *our designed
   condition*, since heterogeneity is the central bet. If we adopt per-pair discovery, expect it to
   be hardest exactly where the architecture is most itself.

---

## 7. What we did not find / open

- **No verbatim theorem statements from Hansen & Ghrist (2021).** The PDF fetch returned a
  paraphrase with quoted sentences I could not corroborate, so nothing from that paper is quoted
  here and no theorem numbers are given. The framework attribution **[S]** is safe; any specific
  convergence claim from it is **not yet checked** and should be verified against the SIAM version
  before use. Same caution applies to the FOLS (2010) full text.
- **No source found that anneals a shared dimension during training.** If someone has, this pass
  missed it.
- **No source found that allocates shared dimension across a network by a global rule.** Our
  allocator appears to be unexampled.
- **No result found on whether a network of per-pair shared subspaces of *differing* dimensions can
  be made cycle-consistent.** The synchronization literature assumes maps between like objects; the
  flat-bundle trick assumes one frame per node. Di Nino et al. **[V]** get closest — subspaces of
  varying dimension per node — but the fetch did not confirm any cycle-consistency treatment.
  **This may genuinely be an open problem, and it is exactly our object.**
- **No result found relating \(\dim H^0\) to anything like our reach metric.** The sheaf literature's
  asymptotic results are about linear separability of node classes. Whether "the ripple reaches the
  far shore, distorted" has a formulation in that language is unanswered.
- **Did not investigate**: the information-bottleneck lineage in any depth; partial/inexact
  correspondence over networks; the multi-view VAE literature beyond the two cited; DCCAE and the
  reconstruction-regularised branch; Tian/Chen/Ganguli's collapse dynamics; Oono & Suzuki. All are
  marked **[U]** or absent above and are the obvious next reads.
- **The compression reference is unresolved** — five candidates, §5, with a recommendation to cite
  the argument rather than the remembered title.
