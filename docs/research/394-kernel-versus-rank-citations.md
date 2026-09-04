# Citation pass: the kernel-versus-rank trade, and rank collapse under a sparsity pressure (patchworks#394)

Opened by [#356](https://github.com/NGL321/patchworks/issues/356), part of map
[#127](https://github.com/NGL321/patchworks/issues/127). Sibling: the empirical sweep,
[#393](https://github.com/NGL321/patchworks/issues/393).

**Citation sequencing.** The design came first and is not on trial. `06-graph-topology.md` wanted
learned rank-deficiency in 2026-08-22's first spec commit; ADR-0010 softened its own pricing claim in
[#57](https://github.com/NGL321/patchworks/issues/57); #356 found the mechanism on 2026-09-03. This
pass validates afterwards and seeds nothing. Where a source agrees the document says *corroborates*;
where it disagrees it says so and hands the disagreement to a ticket rather than acting on it.

**Registers consulted:** open-problems — [#324](https://github.com/NGL321/patchworks/issues/324) is
the live match and its `@failure` is this pass's subject in the register's own words (*"per-edge
effective rank — the participation ratio `(Σσᵢ²)² / Σσᵢ⁴` — slides toward 1 across the fleet"*), and
[#325](https://github.com/NGL321/patchworks/issues/325) and
[#330](https://github.com/NGL321/patchworks/issues/330) are adjacent (the gauge's shared subspace, and
core cells carrying agreement *or* slow state); proposed-solutions —
[#319](https://github.com/NGL321/patchworks/issues/319) matches on shape (*"restriction maps collapse
to effective rank ~1.02–1.06 … each edge transmits a single fixed direction"*) and
[#315](https://github.com/NGL321/patchworks/issues/315) matches on mechanism (holonomy), both taken
into account in §5 and neither duplicated; dismissed-solutions — the only row is
[#346](https://github.com/NGL321/patchworks/issues/346), a cross-edge coherence term killed by
`detectability`, and nothing in §5 re-proposes it. Nothing here was foreclosed.

## Reading-depth key

#148's key, used throughout.

- **[FULL]** — paper body read (PDF text or HTML extracted).
- **[ABS]** — authoritative abstract / landing page only.
- **[CITE]** — citation confirmed to exist, text not reached.
- **[UNREACHED]** — existence not confirmed.

**One extraction note, because it bears on trust.** Every fetch in this pass was rendered to markdown
and read through a summarising step, so a passage marked as a quotation is a quotation *as returned by
that step*. Where a quoted string is short, distinctive and load-bearing it was cross-checked against a
second retrieval of the same document or against a search snippet of it; those are the quotations §1.1,
§1.2, §2.1 and §3.3 rest on. One case where two retrievals of the same document **disagreed** is
recorded in §6 rather than used. No claim in this document rests on a passage that was returned once,
uncorroborated, and mattered.

---

## Headline verdict, stated plainly

**Nobody has priced this trade. The sheaf literature does not look at the other end of the dial — but
not because it forgot to: it engineers `dim ker(Δ_F)` by a mechanism that is not rank-deficiency, and
its central results assume the maps are invertible or orthogonal, which excludes Patchworks' mechanism
by hypothesis rather than by oversight.** That is the pass's main finding and it is a genuine
frontier claim, narrower and better founded than "the literature is silent."

Four things follow, and they are separable.

1. **The ancestor's kernel is bought differently.** Bodnar et al.'s Lemma 6 is stated for a *discrete
   `O(d)` bundle* — restriction maps in `O(d)`, hence full rank by construction — and reads
   `dim(H⁰) ≤ d`, with equality *"if and only if the transport is path-independent."* In the source,
   `H⁰` is enlarged by **stalk width and trivial holonomy**, not by rank-deficiency, and it is
   **capped at `d`**. `05-timescales.md`'s sentence — *"neural sheaf diffusion engineers
   `dim ker(Δ_F)` deliberately"* — is accurate, and the mechanism it inherits is not the one this
   project uses. The step out is larger than the spec claims, in the direction of *less* precedent.
2. **The nearest thing to a stated trade is about the *quality* of `H⁰`, never about its price.** Dong
   et al.'s index-theoretic criterion (2026) says outright that *"certain sheaves may possess a large
   harmonic space whose sections are nonetheless constant; conversely, a small harmonic space may
   consist entirely of nonconstant, class-preserving sections"* — the field has learned that a big
   `H⁰` can be worthless, and has **not** learned that a big `H⁰` is paid for out of transmission.
   Dönmez et al. (2026) reach the same place from representation theory. **No source read here writes
   `dim H⁰ = dim C⁰ − rank δ` down as a budget.**
3. **Rank collapse under a concentrating penalty is extremely well known — in four other fields, none
   of them sheaves — and the remedies are real, named, and mostly unavailable here.** §2 lists eight.
   The ones that transfer cheapest are a **hinge/floor stated as a constraint** (VICReg) and
   **nuclear-norm *maximisation*** (BNM), and the reason they are the cheapest is structural: both are
   the *same shape* as the projection ADR-0010 already runs after every transport step.
4. **The guard-built-for-the-wrong-degeneracy failure is written down. Twice, plainly, in
   self-supervised learning — and once already inside this repo.** Jing, Vincent, LeCun & Tian
   (ICLR 2022) and Hua et al. (ICCV 2021) both name it: the mechanism that excludes *complete* collapse
   leaves *dimensional* collapse open. Hua et al.'s wording is the sentence #356 asked for — *"another
   reachable collapse pattern that is usually overlooked, namely dimensional collapse."* And ADR-0010
   already contains the same observation about the sister norm: unit spectral norm *"excludes `F = 0`
   while leaving `F → rank 1` wide open."* **The repo wrote half this sentence in its own voice before
   the pass found the other half.**

**What this does not do.** It does not settle where `λ` should sit. Nothing found bears on the value,
and #393's sweep is unaffected and remains the whole answer to *where*. This pass changes what #393's
curve will be read against, and it supplies two proposals for what could be done if the curve has no
good interior point.

---

## 1. Thread 1 — neural sheaf diffusion and its descendants

### 1.1 The ancestor engineers the kernel, and the mechanism is not rank-deficiency

**Bodnar, Di Giovanni, Chamberlain, Liò & Bronstein, "Neural Sheaf Diffusion: A Topological Perspective
on Heterophily and Oversmoothing in GNNs", NeurIPS 2022, [arXiv:2202.04579](https://arxiv.org/abs/2202.04579).**
[FULL, via ar5iv, two retrievals]

The paper's whole argument turns on what the diffusion converges to. As diffusion progresses *"each
feature channel is projected into `ker(Δ_ℱ)`"*, and `ker(L_ℱ)` and `H⁰(G,ℱ)` are *"isomorphic as vector
spaces."* Where a graph Laplacian's kernel is the constants — one direction per connected component,
which is oversmoothing — a sheaf Laplacian's kernel can be larger and structured. That is the
inheritance `05-timescales.md` names, and it is correctly named.

**The hypotheses are where this pass earns its keep.** The paper's sheaf classes are enumerated by the
determinant condition, not around it:

> **Symmetric invertible:** `ℱ_v◁e = ℱ_u◁e`, `det(ℱ_v◁e) ≠ 0`
> **Non-symmetric invertible:** `det(ℱ_v◁e) ≠ 0`
> **Diagonal invertible:** diagonal `ℱ_v◁e`, `det(ℱ_v◁e) ≠ 0`
> **Orthogonal:** `ℱ_v◁e ∈ O(d)`

Every named class is **full rank by hypothesis.** A rank-deficient restriction map is not a
badly-behaved member of these families; it is outside all four.

And the lemma that sizes the kernel is stated inside the strictest of them:

> "Let `ℱ` be a discrete `O(d)` bundle over a connected graph `G`. Then `dim(H⁰) ≤ d` and
> `dim(H⁰) = d` if and only if the transport is path-independent."

Three readings, in increasing order of consequence.

- **The kernel is bought with holonomy, not with rank.** Under `O(d)` maps `rank δ` cannot move at
  all; what moves `dim H⁰` is whether transport around cycles is trivial. That is a property of how
  the maps *agree with each other*, not of how much each one throws away. `CONTEXT.md`'s
  **incoherence** is the repo's name for the same axis, and
  [#315](https://github.com/NGL321/patchworks/issues/315) is the standing proposal to read it.
- **The kernel has a ceiling, and it is the stalk width.** `dim(H⁰) ≤ d`. In the source, the way to
  more private capacity is a wider stalk — a construction quantity. `05-timescales.md`'s own bound
  `dim H⁰ ≥ Σ_v max(0, n − Σ_e m_e)` is the same instinct: construction-set. The clause that goes
  past it — *"learned rank-deficiency enlarges `H⁰` past it"* — is where the design leaves the
  source, and the source has no ceiling-raising mechanism to lend it.
- **So `05-timescales.md`'s "step out rather than a leap" understates the step in one specific place
  and overstates it in none.** The document already says what is unprecedented is *the use* — the
  kernel as tick-to-tick persisting state rather than a `t → ∞` limit. Correct, and now there is a
  second unprecedented thing beside it: *the route*. This is a candidate revision, §4.

The paper's own stated costs are about parameterisation and numerics, not about capacity: the diagonal
family's *"main disadvantage is that the `d` dimensions of the stalks interact only via the left `W₁`
multiplication"*, and for general matrices *"the sheaf Laplacian is more challenging to normalise
numerically."* **Nothing in the paper says what a sheaf gives up when its kernel grows.** The separation
propositions (8–13) are all statements that a class *has* linear separation power under conditions on
`d` and the class; none is a statement of what is lost.

### 1.2 The nearest thing to a stated trade — and it is a different trade

Two 2026 papers do interrogate whether a large harmonic space is worth having. Both stop one step short
of this ticket's question, and the step they stop short of is the whole question.

**Dong, Peng, Li, Feng & Xia, "Demystifying Oversmoothing in Sheaf Neural Networks: An Index-Theoretic
Criterion", [arXiv:2608.16180](https://arxiv.org/abs/2608.16180) (2026).** [ABS+, HTML partially read]

The criterion is that a raw index jump is not enough — *"trivial configurations (IdentitySheaf,
InverseSheaf) inflate dimension without preserving node-level distinctions"* — and the operative
sentence is:

> "Certain sheaves may possess a large harmonic space whose sections are nonetheless constant;
> conversely, a small harmonic space may consist entirely of nonconstant, class-preserving sections."

This is a real result and it is **orthogonal to the ticket.** It prices `dim H⁰` against *the
informativeness of `H⁰` itself*, and concludes that dimension is the wrong figure of merit for the
private space. It says nothing about what the coboundary can still carry. Their mechanism is again
holonomy — *"the holonomy representation `ρ: π₁(G,v₀) → Aut(ℱ(v₀))`"* — and `Aut` is invertibility
again, so §1.1's observation holds here too.

**Its transferable sting, and it is sharp.** If dimension is a bad figure of merit for the private
space, then `05-timescales.md`'s bound and #356's `dim H⁰ = dim C⁰ − rank δ` are both counting a
quantity whose *usefulness* is not implied by its size. A large `H⁰` bought by zeroing a map's rows is
close to their *IdentitySheaf* case in spirit: dimension inflated, distinctions not preserved. **That is
a second, independent reason not to read a large `dim H⁰` as a win** — and it is the one open problem
[#326](https://github.com/NGL321/patchworks/issues/326) already states in the repo's own terms
(*"private structure held and never consumed"*). Corroborates #326 from outside.

**Dönmez, Mosig, Fritsche & Koch, "Oversmoothing as Representation Degeneracy in Neural Sheaf
Diffusion", [arXiv:2605.11178](https://arxiv.org/abs/2605.11178) (2026).** [ABS+, HTML read]

Defines oversmoothing as *"degeneration toward trivial or low-complexity representation summands whose
global sections fail to preserve discriminative information"*, and proves the harmonic space decomposes
functorially, `H⁰(G,ℱ) ≅ ⨁_k H⁰(G,ℱ^(k))`. Its exclusion device is King (`θ-`) stability. Two things
worth carrying:

- **The failure it names is the failure this project is engineering toward, seen from the other side.**
  "Low-complexity summands whose sections fail to preserve discriminative information" is a fair
  description of what a fleet of rank-1 maps produces.
- **Its guard has a known dead spot, and the dead spot is Patchworks' regime.** In equal-stalk
  architectures (`d_v = d_e`) the trivial subrepresentation *"lies exactly on a stability wall"* where
  admissible `θ` cannot exclude it. Patchworks is not equal-stalk in general (`n = 12`, `m ∈ {4, 8}`),
  so this is not a direct hit — recorded so that `θ`-stability is not later imported as an off-the-shelf
  guard without checking which side of the wall the dome sits on.

**Neither paper reads the other end.** Neither asks what the sheaf still transmits once the kernel is
big. Explicitly: **nothing found.**

### 1.3 What the learned-sheaf line does with low-rank maps

Two data points, both weak, and honestly weak.

**Braithwaite, Borgi, Onorato, Tarantelli, Restuccia, Silvestri & Liò, "Heterogeneous Sheaf Neural
Networks", [arXiv:2409.08036](https://arxiv.org/abs/2409.08036).** [ABS, contested]

It is the one work read here that parameterises restriction maps as explicitly **low rank** —
`AB^T + diag(c)`, parameter-efficient *"when `r < (d−1)/2`"* — and ablates diagonal against low-rank
against general. **Two retrievals of this paper returned contradictory conclusions about which type
wins**, and neither could be pinned to the section (B.2.5, *Effect of Restriction Map Type*). It is
therefore recorded as **not established**, and §6 carries it. What is established is narrower and still
useful: **low rank appears in this literature as a parameter-budget device, never as a capacity
purchase.** Nobody in the sheaf line treats rank-deficiency as buying anything.

**Di Nino, Barbarossa & Di Lorenzo, "Learning Sheaf Laplacian Optimizing Restriction Maps",
[arXiv:2501.19207](https://arxiv.org/abs/2501.19207).** [FULL — **already read**, `docs/research/053`
§1.2; not re-fetched]

053 established that this line constrains maps to avoid a trivial solution and reads the requirement
more weakly than orthogonality. Nothing new is claimed here; noted so the pass is not re-run.

### 1.4 Thread 1 verdict

**Nothing found, stated as the ticket asked.** No source read prices `dim H⁰` against `rank δ`, states a
bound relating them, or notes that enlarging one spends the other. The reason is now specific rather
than vague: **the sheaf literature reaches its kernel through holonomy and stalk width under
invertibility hypotheses, so the trade does not arise in it.** Patchworks reaches the same kernel by a
route the source's theorems exclude by assumption. This puts the project at the frontier on this exact
question, and it means #393's sweep is not duplicating anything.

---

## 2. Thread 2 — rank collapse under a regularisation pressure

### 2.1 The penalty has a name, and its floor is a one-hot vector at source

`normalised_l1(F) = ‖F‖₁ / (√p ‖F‖_F)` is the **Hoyer measure**, rescaled.

**Hoyer, "Non-negative Matrix Factorization with Sparseness Constraints", JMLR 5:1457–1469, 2004.**
[CITE — PDF would not extract; see §6]

The measure is `sparseness(x) = (√n − ‖x‖₁/‖x‖₂)/(√n − 1)`, evaluating to 1 for a vector with a single
non-zero component and 0 when all components are equal in magnitude. The raw ratio therefore runs
`‖x‖₁/‖x‖₂ ∈ [1, √n]`, minimised at a one-hot vector. Patchworks divides by `√p`, so
`normalised_l1 ∈ [1/√p, 1]`, minimised at one-hot — which is exactly what `normalised_l1`'s own
docstring says (*"`h`'s own floor of `1/√p` — a fully concentrated map"*). **The code and the 2004
source agree, and #356's arithmetic is confirmed against a named measure rather than derived
in-house.**

**Yang, Wen & Li, "DeepHoyer: Learning Sparser Neural Network with Differentiable Scale-Invariant
Sparsity Measures", ICLR 2020, [arXiv:1908.09979](https://arxiv.org/abs/1908.09979).** [ABS]

The relevant fact for this repo is not their algorithm but their *reason*: the Hoyer regulariser is
chosen because it is *"both differentiable almost everywhere and scale-invariant"*, unlike `L0` and
`L1`. **Scale-invariance is the property Patchworks selected it for too** (`07`: both terms blind to a
map's overall magnitude, ADR-0010's gauge doing the rest). So the choice of penalty is corroborated as
the standard one for exactly the stated reason. **And the same property is what makes it walk through
the transport rule's scale guard** — §3.3.

**What nobody in the Hoyer line states.** DeepHoyer applies the measure elementwise and in groups, for
*compression*. No source read here applies it to a matrix and asks what happens to the matrix's **rank**.
The rank consequence is arithmetic — a one-nonzero-entry matrix is rank 1 — and it is arithmetic that
this literature never has cause to write down, because a pruned network is not asked to transmit
anything through the pruned layer as a subspace.

### 2.2 The Frobenius-budget half is already in the repo, and only half was carried

**Miyato, Kataoka, Koyama & Yoshida, "Spectral Normalization for GANs", ICLR 2018,
[arXiv:1802.05957](https://arxiv.org/abs/1802.05957).** [FULL — **already read**, `docs/research/053`
§4, and quoted in ADR-0010 since #57; not re-fetched]

Under a fixed `σ₁² + ⋯ + σ_T² = d_o`, output gain in a fixed direction is *maximised* at rank one,
*"which means that `W̄_wn` is of rank one"*, corresponding to *"using only one feature"*. ADR-0010
already carries this, already softened *"a priced trade rather than a free lunch"*, and already names
the two pressures that still point at concentration — a single tick needing one direction, and the
composed L1 minimised at fixed Frobenius norm by the sparsest map.

**The pass's contribution here is not a new source; it is that the record contains the mechanism twice
and the consequence zero times.** ADR-0010 says concentration is wanted. `06-graph-topology.md` says
concentration enlarges `H⁰`. `053` §4 says Frobenius makes rank one the maximiser under a gain-seeking
objective. **What no document says is that `rank δ` is the same number the channel is made of.** That
sentence is #356's, it is arithmetic, and this pass confirms no external source supplies it.

### 2.3 Nuclear norm is the standard rank surrogate — and it points the wrong way by default

**Scarvelis & Solomon, "Nuclear Norm Regularization for Deep Learning", NeurIPS 2024,
[arXiv:2405.14544](https://arxiv.org/abs/2405.14544).** [ABS]

Penalising the nuclear norm of a Jacobian *encourages it to locally behave like a low-rank linear map*.
Their contribution is a tractable estimator: for `f = g ∘ h` one may *equivalently penalize the average
squared Frobenius norm of the component Jacobians.*

**Two things this settles.** First, the obvious remedy name — "add a nuclear-norm term" — is by default
a **rank-reducing** move, so it must be signed correctly or it makes the problem worse. Second, and
sharper: their equivalence says a **Frobenius penalty on composed factors is a nuclear-norm penalty on
the composite.** Patchworks composes restriction maps and cell operators along a path and calls the
composite the channel (ADR-0022). The gauge holds each factor's Frobenius norm *fixed* rather than
penalised, so the equivalence does not fire — but it is the closest thing found to a statement that
per-factor norm control is rank control on the composite, and it is the reason §5's Proposal A is
scoped per map rather than per path.

### 2.4 Remedies that are not "lower the coefficient"

Eight, each with what it claims and what it would cost here. **None comes from the sheaf literature.**

**(a) A per-direction variance *floor*, stated as a hinge — VICReg.**
Bardes, Ponce & LeCun, ICLR 2022, [arXiv:2105.04906](https://arxiv.org/abs/2105.04906). [ABS]
*"a method that explicitly avoids the collapse problem with a simple regularization term on the
variance of the embeddings along each dimension individually."* The term is a hinge: below a threshold
it pushes, above it is silent.
**Cost here:** it is per-dimension on a batch of embeddings, and Patchworks has no batch — but the
*shape* transfers exactly, because ADR-0010 already runs a **projection after every step** and a hinge
is a projection with a floor instead of a band. This is the cheapest transfer found and it is Proposal
A.

**(b) Nuclear-norm *maximisation* — BNM.**
Cui, Wang, Zhuo, Li, Huang & Tian, "Towards Discriminability and Diversity: Batch Nuclear-norm
Maximization under Label Insufficient Situations", CVPR 2020 (oral),
[arXiv:2003.12237](https://arxiv.org/abs/2003.12237). [ABS]
Their decomposition is uncannily this repo's: *prediction discriminability and diversity can be
separately measured by the Frobenius-norm and rank of the batch output matrix*, and since the nuclear
norm is an upper bound of the Frobenius norm and a convex approximation of matrix rank, they maximise
it. **This is the literature's clearest statement that Frobenius and rank are two different things that
have to be controlled separately** — which is precisely the gap ADR-0010's gauge leaves, since it pins
the first and lets the second go.
**Cost here:** an SVD per map per step, on `m × n` maps with `m ∈ {4,8}`, `n = 12` — cheap in principle,
and #324's rig already takes exactly this decomposition in float64.

**(c) Whitening / decorrelation.**
Hua, Wang, Xue, Ren, Wang & Zhao, "On Feature Decorrelation in Self-Supervised Learning", ICCV 2021,
[arXiv:2105.00470](https://arxiv.org/abs/2105.00470). [ABS] — *"connect dimensional collapse with strong
correlations between axes and consider such connection as a strong motivation for feature decorrelation
(i.e., standardizing the covariance matrix)."*
**Cost here:** a covariance over what? The dome has no batch axis and ADR-0011 forbids anything a cell
cannot compute from its own incident edges. The per-cell analogue — decorrelate a cell's *incident maps*
from one another — is `CONTEXT.md`'s **incoherence**, which is already measured (1.75–2.42) and already
proposed as a lever nowhere. Recorded as the shape that would need inventing, not adopting.

**(d) Architectural counterweights — skip connections and MLPs.**
Dong, Cordonnier & Loukas, "Attention is Not All You Need: Pure Attention Loses Rank Doubly
Exponentially with Depth", ICML 2021, [arXiv:2103.03404](https://arxiv.org/abs/2103.03404). [ABS]
*"without skip connections or multi-layer perceptrons (MLPs), the output converges doubly exponentially
to a rank-1 matrix. On the other hand, skip connections and MLPs stop the output from degeneration."*
**Cost here, and it is the useful one:** this is the *adjacent* case the ticket predicted — a linear
transport stage collapsing to rank 1 — and the remedy is **not a penalty at all**. It is a residual path
that the collapsing operator does not sit on. Patchworks' analogue would be a route from a cell's node
stalk to a neighbour that does not pass through the collapsing restriction map, which the architecture
does not have and which ADR-0011's locality does not forbid. Stated as a shape, not recommended: it is
an architectural change and this pass does not own fixes.

**(e) Normalisation that provably floors the rank.**
Daneshmand, Kohler, Bach, Hofmann & Lucchi, "Batch Normalization Provably Avoids Rank Collapse for
Randomly Initialised Deep Networks", NeurIPS 2020, [arXiv:2003.01652](https://arxiv.org/abs/2003.01652).
[ABS] — *"the rank of the intermediate representations in unnormalized networks collapses quickly with
depth"*; batch norm *"is an effective strategy to avoid rank collapse for both linear and ReLU
networks"*, with *"a meaningful lower rank bound in deep linear networks"*.
**Cost here:** batch statistics again, so it does not transfer. Its value is as evidence that a **rank
lower bound is a thing one can actually prove about a normalisation scheme**, which is the standard
Proposal A would eventually be held to.

**(f) A structural property that provably solves it — sum of Kronecker products.**
Roth & Liebig, "Rank Collapse Causes Over-Smoothing and Over-Correlation in Graph Neural Networks",
LoG 2023 (PMLR 231), [arXiv:2308.16800](https://arxiv.org/abs/2308.16800). [ABS]
The headline is the reframing: with depth, *the rank of the node representations collapses*, and
over-smoothing is downstream of that. The paper argues *the importance for future research to focus on
rank collapse rather than over-smoothing*, and proposes a sum of Kronecker products (SKP) as *a general
property that provably solves rank collapse*.
**Why it matters here beyond the remedy:** this is the message-passing field independently arriving at
#324's framing — rank is the primary quantity and smoothing is a symptom. **Corroborates #324's
`@failure` from outside the project.** The SKP remedy itself is about the *aggregation* function, not
the transport map, and does not transfer without an argument nobody has made.

**(g) Effective-rank regularisers, from the one field that treats rank as a first-class training
signal.** Deep RL. Kumar, Agarwal, Ghosh & Levine, "Implicit Under-Parameterization Inhibits
Data-Efficient Deep Reinforcement Learning", ICLR 2021,
[arXiv:2010.14498](https://arxiv.org/abs/2010.14498) [ABS] characterises expressivity loss *"via a drop
in the rank of the learned value network features"* under an implicit pressure nobody asked for, and
correlates it with performance collapse. The survey *Plasticity Loss in Deep Reinforcement Learning*
([arXiv:2411.04832](https://arxiv.org/abs/2411.04832)) [ABS] indexes the family: **Direct Singular Value
Regularization** (penalise dominant singular values to spread the spectrum), **InFeR** (auxiliary
random-target regression heads), **DR3** (penalise dot products between consecutive-state
representations), **BEER** (adaptive rank target from a cosine-similarity bound).
**Cost here:** DSVR is per-matrix and local and is the direct competitor to (b); InFeR needs auxiliary
heads and a second objective, which `07`'s two-rule split would have to absorb; DR3 and BEER are
temporal-difference-specific. **The transferable finding is the framing**: this field treats effective
rank as a *monitored training quantity with a floor*, which is exactly the status #356 declined to give
it (*"effective rank stays an instrument and is not given a bar"*). The literature's practice is against
that ruling; the ruling's reason — that a constant already exists to carry the decision — is not
addressed by any of them.

**(h) The one that is genuinely just "lower the coefficient", named so it is not mistaken for more.**
Every penalty-versus-constraint treatment reduces to it in the limit. See §3.2.

### 2.5 Thread 2 verdict

**Richly known, and not once in this setting.** Rank collapse under a concentrating pressure is
characterised in attention, in graph message passing, in self-supervised embedding, in deep RL value
functions, and in random deep networks. In **none** of them is the collapsing object a sheaf restriction
map, and in none is the collapse *wanted*. The remedies exist, are not exhausted by lowering `λ`, and
divide into three families — **spectral floors** (b, e, g), **decorrelation** (c), and **structural
bypasses** (d, f). Two of the three are batch-shaped and die on ADR-0011's locality. The spectral-floor
family survives, and it survives because it has the same shape as a projection this architecture already
performs.

---

## 3. Thread 3 — a two-term objective whose second term has a degenerate optimum

### 3.1 Is #356's rejection of alternation standard?

**Yes in substance, and nobody states it in these words.** [ABS across the sources below]

The framing the field does have is scalarisation versus multi-objective optimisation. Sener & Koltun,
"Multi-Task Learning as Multi-Objective Optimization", NeurIPS 2018,
[arXiv:1810.04650](https://arxiv.org/abs/1810.04650) [ABS] set out the standard position: a weighted
linear combination of competing losses is *a proxy objective*, valid *only when the tasks do not
compete*, and the principled replacement is to seek a **Pareto-stationary** point — *the lack of a
shared descent direction across all losses* — via MGDA.

Read against #356 this cuts cleanly, and in #356's favour:

- **MGDA replaces the weight with a rule for choosing a descent direction; it does not tell you which
  Pareto point you want.** Where two objectives genuinely compete, every method in this family returns
  *a* point on the front and the choice among them stays a design decision. **That is exactly #356's
  ruling** — *"a real design trade rather than an optimisation artifact"* — arrived at independently,
  and it is the field's own position on what scalarisation can and cannot buy.
- **Alternation is weaker than MGDA, not stronger.** Alternating two terms over the *same* parameters
  is a scalarisation with a time-varying weight. It cannot leave the front; it selects a point on it as
  a function of the schedule. #356's phrasing — it *"would make the equilibrium depend on a schedule
  instead of on a decision"* — is the correct characterisation and this pass found nothing that
  contradicts it.
- **Where alternation *is* standard, the structure is different, and the difference is the one `07`
  already draws.** Alternating optimisation is the tool of choice when there are two *parameter blocks*
  (matrix factorisation, EM, GAN minimax). `07` already separates the rules on exactly that criterion —
  *"prediction error trains how a cell thinks; disagreement trains how it talks"*, different parameter
  groups — and #356 already observed the entanglement is *inside* one rule, between two terms over one
  parameter. **The literature's licence for alternation is a two-block structure the transport rule does
  not have.**

**Nothing found that disentangles a case like this by an optimisation technique.** Explicit negative.

### 3.2 The standard disentanglement is not alternation — it is a constraint

The one technique the literature does offer for "one term's optimum is degenerate" is to stop making it
a term. The penalty/constraint pair — minimise `F(x) + λR(x)` versus minimise `F(x)` subject to
`R(x) ≤ τ` — is textbook, and the textbook caveat is the load-bearing part: **the equivalence is not
automatic.** A penalised problem is an exact reformulation only when a suitable multiplier or exactness
condition holds; otherwise the penalised objective is a *relaxation or scalarised surrogate*.
[ABS/secondary; see §6 — this is standard optimisation and was not traced to a primary text in this
pass.]

**Why this is the thread's most useful output.** It says the choice is not between `λ = 0.4` and
`λ = 0.1`. It is between a **penalty** whose minimiser is rank 1 and a **constraint** — "concentrate as
much as you like, subject to a floor" — whose feasible set never contains rank 1 at all. VICReg's hinge
(§2.4a) *is* that constraint in penalty clothing; ADR-0010's projection *is* that constraint already,
for scale. **The architecture already runs the machinery; it runs it on the wrong quantity.** That is
Proposal A, and it is the only place in this pass where the reading points at something the design
could do differently without inventing a mechanism.

### 3.3 The guard built for scale, walked through in rank — the failure *is* written down

#356 asked for this sentence specifically. It exists, and in three places.

**In self-supervised learning, twice, and the second is nearly verbatim.**

- Jing, Vincent, LeCun & Tian, "Understanding Dimensional Collapse in Contrastive Self-supervised
  Learning", ICLR 2022, [arXiv:2110.09348](https://arxiv.org/abs/2110.09348) [ABS]: methods exist to
  *"solve the collapsing problem where all embedding vectors collapse to a trivial constant solution"*;
  *"non-contrastive methods suffer from a lesser collapse problem of a different nature: dimensional
  collapse, whereby the embedding vectors end up spanning a lower-dimensional subspace instead of the
  entire available embedding space"*; and — the finding — *"we show that dimensional collapse also
  happens in contrastive learning."* **The guard that stops complete collapse does not stop the
  subspace one, and the field discovered this by measuring, after assuming otherwise.**
- Hua, Wang, Xue, Ren, Wang & Zhao, ICCV 2021, [arXiv:2105.00470](https://arxiv.org/abs/2105.00470)
  [ABS]: *"we verify the existence of complete collapse and discover another reachable collapse pattern
  that is usually overlooked, namely dimensional collapse."*

The structural correspondence is close enough to be worth stating flatly. Patchworks' relative
normaliser makes disagreement invariant to an edge's joint scale, so *shrinking* an edge buys nothing —
`07`'s guard, and it excludes the constant/zero solution exactly as an SSL method's normalisation does.
**Both guards are scale guards. Both leave a subspace guard unwritten.** The SSL field's answer is that
a *second, different* term is required — a variance floor, a covariance penalty, a whitening — and that
no amount of tuning the first one substitutes.

**And in this repo, before the pass.** ADR-0010, *Frobenius, not spectral*: unit spectral norm *"excludes
`F = 0` while leaving `F → rank 1` wide open."* That is the identical observation about the sister norm,
written down in 2026 by this project, and then not applied to the relative normaliser standing next to
it. **The near-miss #356 identified is real, and the record contains the sentence that would have caught
it, aimed one object to the left.**

### 3.4 Thread 3 verdict

#356's rejection of alternation is **standard in substance and unstated in form**: the field's reason
is that scalarisation cannot choose a point on a front where objectives genuinely compete, and
alternation is a scalarisation with a schedule. The technique the literature *does* offer is the
penalty-to-constraint move, which the architecture is already tooled for. And the specific failure —
a normalisation guard that excludes the scale degeneracy and leaves the rank one open — is written down
plainly in self-supervised learning and, for a different norm, inside ADR-0010.

---

## 4. What this threatens

**One candidate revision, one corroboration, one thing that is not threatened.**

### R1 — `05-timescales.md` names the ancestor accurately and understates the distance in one place

**Where:** `docs/spec/05-timescales.md`, *Insulation from neighbours: `H⁰`*: *"The design move is a step
out from published work rather than a leap: neural sheaf diffusion engineers `dim ker(Δ_F)` deliberately
… What is unprecedented is the *use*."*

**What the literature says:** Bodnar et al.'s Lemma 6 is stated for `O(d)` bundles and gives
`dim(H⁰) ≤ d`, equality *"if and only if the transport is path-independent"*; the paper's four sheaf
classes are all invertible or orthogonal by hypothesis. The kernel is enlarged there by **stalk width
and trivial holonomy**, under full-rank maps, and is **capped at the stalk width**. Rank-deficiency is
not a mechanism available in the source.

**Why it matters:** the sentence currently reads as though the route is shared and only the use is new.
Two things are new, and the second is the one #356 found the price of. It also removes an implicit
reassurance: there is no published ceiling-raising precedent for `dim H⁰` beyond `d`, so the clause
*"learned rank-deficiency enlarges `H⁰` past it"* is unprecedented rather than inherited.

**Ask:** one sentence, saying the inherited mechanism is stalk width and holonomy under invertible maps,
and that enlarging `H⁰` by rank-deficiency is this project's own step. **This is a spec edit and this
pass does not make it** — it hands off, per #127's standing note.

### Corroborated, not threatened

- **[#324](https://github.com/NGL321/patchworks/issues/324)** — Roth & Liebig independently argue the
  field should *"focus on rank collapse rather than over-smoothing"*. #324's `@failure` is the same
  reframing and needs no amendment.
- **[#326](https://github.com/NGL321/patchworks/issues/326)** — Dong et al.'s *"large harmonic space
  whose sections are nonetheless constant"* is #326's *"private structure held and never consumed"* from
  outside. A second, independent reason not to read `dim H⁰` as a win.
- **ADR-0010** — nothing found contradicts the softened pricing. §2.2 and §3.3 both sit on it and both
  strengthen it. No revision proposed.

### Not threatened

- **#356's ruling.** Every part survives. The alternation rejection is standard (§3.1); the retune
  framing is untouched; the interior-optimum bar is untouched. §2.4(g) records that the deep-RL field's
  *practice* is to put a bar on effective rank, which is against #356's point 4 — but #356's *reason*
  (a constant already exists to carry the decision) is not addressed by any source, so this is noted
  and not raised.
- **[#393](https://github.com/NGL321/patchworks/issues/393).** Nothing found gives a value for `λ`, a
  predicted knee, or a shortcut past the sweep. §1.2 adds one thing #393 may want to read alongside
  `dim H⁰`: dimension is not informativeness, so a curve where `dim H⁰` rises and nothing downstream
  improves is a *predicted* outcome with a citation behind it, not an anomaly.

---

## 5. Proposals minted

Two, both with a source and shapes, per ADR-0029. **A proposal binds nothing**, and neither is
recommended over the other; #393's curve is what should choose between them, or reject both.

Checked against the register first: [#319](https://github.com/NGL321/patchworks/issues/319)
(input-conditioned maps) shares shape A's symptom and proposes a *different* mechanism — making `F` a
function of the stalk — so A is complementary rather than duplicative;
[#315](https://github.com/NGL321/patchworks/issues/315) proposes *reading* holonomy, where B proposes
*using* it, and B should be read as B-on-top-of-#315.

### Proposal A — a rank floor as a constraint, not a smaller `λ`

Filed as a comment on [#324](https://github.com/NGL321/patchworks/issues/324).

```
@proposal A rank floor stated as a constraint on the map, not a smaller sparsity coefficient
@source   VICReg (Bardes, Ponce & LeCun, ICLR 2022, arXiv:2105.04906); BNM (Cui et al., CVPR 2020,
          arXiv:2003.12237); docs/research/394 §2.4, §3.2
@shape    effective rank slides toward 1 under a sparsity pressure
@shape    a scale guard that excludes shrinking a map does not exclude concentrating it
@answers  324
@when     measurement driven_settling draining_effective_rank < 2
@status   open
```

**The argument.** ADR-0010 already runs a projection after every transport step, onto
`‖F‖_F ∈ [1/ρ, ρ]`. A floor on effective rank is the *same act on a second quantity*: project the map's
singular-value vector back into a set whose participation ratio is at least some `r_min`, or add a
VICReg-style hinge that is silent above the floor and pushes below it. Two properties make this cheaper
here than anywhere it is published: the projection machinery exists and is already local — *"a cell owns
its own incident maps and needs nothing from a neighbour"* — and #324's rig already computes the exact
statistic in float64 on the unit-normalised map. BNM supplies the principled version of the same move,
and its decomposition is this repo's situation in another vocabulary: Frobenius norm and rank measure
different things and must be controlled separately, which is precisely the hole the gauge leaves.
**What it costs:** an invented constant (`r_min`), which #127's Notes defer rather than ban; an SVD per
map per step; and it makes explicit a decision `06-graph-topology.md` currently makes implicitly, which
is arguably its main virtue and arguably a violation of the intent.

### Proposal B — buy the kernel with holonomy instead of with rank

Filed as [#396](https://github.com/NGL321/patchworks/issues/396), orphaned.

```
@proposal Buy H^0 with holonomy rather than rank-deficiency: enlarge the kernel by aligning
          incident maps around cycles, leaving each map full rank
@source   Bodnar et al., NeurIPS 2022, arXiv:2202.04579, Lemma 6; Dong et al., arXiv:2608.16180
          (holonomy representation); docs/research/394 §1.1, §1.2
@shape    private capacity and transmitted capacity are one budget
@shape    effective rank slides toward 1 under a sparsity pressure
@when     event 393
@status   open
```

**The argument.** The source `05-timescales.md` already cites gets its kernel from path-independence of
transport under `O(d)` maps — *"`dim(H⁰) = d` if and only if the transport is path-independent"* — and
Dong et al. route the same quantity through the holonomy representation. **Holonomy is an axis on which
`H⁰` grows without `rank δ` falling**, because it is a property of how a cell's incident maps agree with
one another rather than of what each discards. This repo already names that axis — `CONTEXT.md`'s
**incoherence**, measured at 1.75–2.42 — and #315 already proposes reading it. The proposal is to treat
it as the *lever* for `05`'s requirement, with the sparsity pressure retained only for the pruning
`06-graph-topology.md` wants for its own sake. **What it costs:** the axis is not local in the way
sparsity is — a cycle is not incident to one cell — so a locality argument (ADR-0011) has to be made
before this is even coherent, and it may not survive. Filed because the reading supports it, flagged
because that objection is serious and unanswered. Bodnar's ceiling `dim H⁰ ≤ d` also applies: this route
cannot exceed stalk width, where rank-deficiency can, so B is a *smaller* purchase than the current
mechanism makes.

---

## 6. What could not be reached, stated plainly

Ranked by how much it weakens what is above.

**Costliest.**

- **Hoyer (2004), JMLR 5:1457–1469.** The PDF would not extract; the measure's definition and the
  one-hot/uniform endpoints were taken from secondary statements of it and from the arithmetic, which is
  elementary and independently checkable. §2.1's claim — that the penalty is the Hoyer measure and its
  floor is a one-hot vector — is **safe** because #356 already derived the same endpoints from the code,
  and the two agree. What is *not* verified is Hoyer's own wording and the exact normalising convention
  in the 2004 paper.
- **Braithwaite et al., Heterogeneous Sheaf Neural Networks (arXiv:2409.08036), §B.2.5.** Two retrievals
  returned **opposite** conclusions about whether diagonal or general restriction maps win the ablation,
  and the section itself could not be reached in either HTML or PDF. **Nothing in §1.3 rests on it**, and
  the one thing §1.3 does claim — that low rank appears there as a parameter-budget device, not a
  capacity purchase — is visible in the parameterisation itself. This is the pass's one recorded source
  conflict and it is left open rather than resolved by preference.
- **Bodnar et al.'s full proofs.** Lemma 6's statement and hypotheses were read; the proof was not.
  Nothing turns on the proof, but the *scope* claim in §1.1 — that no result in the paper covers a
  rank-deficient map — rests on the enumeration of the four classes rather than on an exhaustive read of
  every proposition's preconditions.

**Moderate.**

- **Dong et al. (arXiv:2608.16180) and Dönmez et al. (arXiv:2605.11178)** were read at abstract-plus-HTML
  depth. The two quotations §1.2 leans on are distinctive and were returned intact, but the
  `θ`-stability wall and the equal-stalk condition were not verified against the theorem statement, so
  the recorded caution ("check which side of the wall the dome sits on") is the right strength and
  should not be upgraded without a full read.
- **DeepHoyer's own text.** The NSF-hosted PDF would not extract and the arXiv landing page returned a
  partly garbled account of the measure's endpoints (it reported the minimum as 0, which is the
  *normalised sparseness* convention, not the raw ratio). §2.1 states the raw-ratio arithmetic and does
  not rely on that summary.
- **spred / Ziyin & Wang (ICML 2023, arXiv:2210.01212)** — the reparameterisation route to `L1`, fetched
  at abstract depth only, and the secondary claim that first surfaced it (*"a node becoming sparse
  implies the learned representation becomes low-rank"*) could **not** be located in the paper. It is
  therefore **not used** anywhere above. If that sentence exists at source it would be the single most
  on-target citation in the pass, and it is the first thing a follow-up should chase.
- **The penalty-versus-constraint statement in §3.2** is textbook optimisation taken from secondary
  sources; it was not traced to a primary text. The claim is uncontroversial and the section says so.

**Not searched at all**, recorded so the ground is not re-covered blindly: matrix-completion and
low-rank-recovery theory (where a rank floor is a *hard* constraint with known geometry); information
bottleneck, which is the other field that treats capacity as a budget and might state the trade in its
own terms; coding theory on the rate of a linear channel, which is the most literal reading of
"transmitted capacity"; and the sheaf-cohomology literature proper, as opposed to its neural
descendants, on whether anyone bounds `H⁰` against a transmission functional.

**And the honest negative, which is the pass's headline.** No source was found that prices a sheaf's
harmonic space against what its coboundary can still carry. Not in neural sheaf diffusion, not in its
2026 descendants, not in learned-sheaf signal processing. Every remedy in §2.4 is a transfer by analogy
from a field where the collapse was unwanted, and each carries a stated cost for the transfer.

---

## 7. Sources

| Source | Depth | Access |
| --- | --- | --- |
| Bodnar, Di Giovanni, Chamberlain, Liò & Bronstein (2022), *Neural Sheaf Diffusion*, NeurIPS, [arXiv:2202.04579](https://arxiv.org/abs/2202.04579) | [FULL] | ar5iv, two retrievals; Lemma 6 and sheaf-class definitions verified across both |
| Dong, Peng, Li, Feng & Xia (2026), *Demystifying Oversmoothing in Sheaf Neural Networks*, [arXiv:2608.16180](https://arxiv.org/abs/2608.16180) | [ABS+] | arXiv HTML, partial |
| Dönmez, Mosig, Fritsche & Koch (2026), *Oversmoothing as Representation Degeneracy in Neural Sheaf Diffusion*, [arXiv:2605.11178](https://arxiv.org/abs/2605.11178) | [ABS+] | arXiv HTML, partial |
| Braithwaite et al., *Heterogeneous Sheaf Neural Networks*, [arXiv:2409.08036](https://arxiv.org/abs/2409.08036) | [ABS] | **contested** — see §6 |
| Di Nino, Barbarossa & Di Lorenzo (2025), [arXiv:2501.19207](https://arxiv.org/abs/2501.19207) | [FULL] | already read, `docs/research/053` §1.2; not re-fetched |
| Hoyer (2004), *NMF with Sparseness Constraints*, JMLR 5:1457–1469 | [CITE] | PDF would not extract |
| Yang, Wen & Li (2020), *DeepHoyer*, ICLR, [arXiv:1908.09979](https://arxiv.org/abs/1908.09979) | [ABS] | landing page; §6 records a garbled endpoint |
| Miyato, Kataoka, Koyama & Yoshida (2018), [arXiv:1802.05957](https://arxiv.org/abs/1802.05957) | [FULL] | already read, `docs/research/053` §4; quoted in ADR-0010 |
| Scarvelis & Solomon (2024), *Nuclear Norm Regularization for Deep Learning*, NeurIPS, [arXiv:2405.14544](https://arxiv.org/abs/2405.14544) | [ABS] | landing page |
| Cui, Wang, Zhuo, Li, Huang & Tian (2020), *BNM*, CVPR (oral), [arXiv:2003.12237](https://arxiv.org/abs/2003.12237) | [ABS] | landing page + CVF abstract |
| Bardes, Ponce & LeCun (2022), *VICReg*, ICLR, [arXiv:2105.04906](https://arxiv.org/abs/2105.04906) | [ABS] | landing page |
| Jing, Vincent, LeCun & Tian (2022), *Understanding Dimensional Collapse in Contrastive SSL*, ICLR, [arXiv:2110.09348](https://arxiv.org/abs/2110.09348) | [ABS] | landing page, abstract verbatim |
| Hua, Wang, Xue, Ren, Wang & Zhao (2021), *On Feature Decorrelation in SSL*, ICCV, [arXiv:2105.00470](https://arxiv.org/abs/2105.00470) | [ABS] | landing page, abstract verbatim |
| Dong, Cordonnier & Loukas (2021), *Attention is Not All You Need*, ICML, [arXiv:2103.03404](https://arxiv.org/abs/2103.03404) | [ABS] | landing page, abstract verbatim |
| Daneshmand, Kohler, Bach, Hofmann & Lucchi (2020), *Batch Normalization Provably Avoids Rank Collapse*, NeurIPS, [arXiv:2003.01652](https://arxiv.org/abs/2003.01652) | [ABS] | landing page |
| Roth & Liebig (2023), *Rank Collapse Causes Over-Smoothing and Over-Correlation in GNNs*, LoG / PMLR 231 | [ABS] | [arXiv:2308.16800](https://arxiv.org/abs/2308.16800) |
| Kumar, Agarwal, Ghosh & Levine (2021), *Implicit Under-Parameterization*, ICLR, [arXiv:2010.14498](https://arxiv.org/abs/2010.14498) | [ABS] | landing page |
| *Plasticity Loss in Deep Reinforcement Learning: A Survey*, [arXiv:2411.04832](https://arxiv.org/abs/2411.04832) | [ABS] | used only to index DSVR / InFeR / DR3 / BEER |
| Sener & Koltun (2018), *Multi-Task Learning as Multi-Objective Optimization*, NeurIPS, [arXiv:1810.04650](https://arxiv.org/abs/1810.04650) | [ABS] | landing page + secondary |
| Ziyin & Wang (2023), *spred: Solving L₁ Penalty with SGD*, ICML, [arXiv:2210.01212](https://arxiv.org/abs/2210.01212) | [ABS] | **not used** — see §6 |

## Context

Written for [#394](https://github.com/NGL321/patchworks/issues/394), on branch
`research/kernel-versus-rank`. Repo documents read before searching: `docs/spec/05-timescales.md`,
`docs/spec/06-graph-topology.md`, `docs/spec/07-local-learning-rule.md`, ADR-0010, ADR-0022, ADR-0029,
`CONTEXT.md`, `docs/agents/registers.md`, all three problem registers, and the prior passes
`docs/research/016`, `053`, `148` and `167`. Nothing in `CONTEXT.md` or the spec is edited by this pass.
