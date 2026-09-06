# Citation pass: what arrests rank collapse in an online, normalised, locally-updated rank-one rule (patchworks#528)

Opened by [#517](https://github.com/NGL321/patchworks/issues/517), the cold-start map, as R-A of its
wave-2 fallout. **Predecessor pass:**
[`docs/research/394-kernel-versus-rank-citations.md`](./394-kernel-versus-rank-citations.md), whose
Proposal A became [ADR-0032](../adr/0032-the-maps-learn-isometric-transport-and-a-spectral-floor-expresses-it.md)'s
per-map spectral floor. **This pass exists because that floor succeeded per map and delivered rank 1 end
to end** ([#497](https://github.com/NGL321/patchworks/issues/497)). #394 is not restated; where a source
was already read there it is cited to that pass and not re-fetched.

**This pass buys options and proposes nothing.** Per the ticket: no codebase change is proposed, no
register row is minted, no ADR is amended. Ranking candidates is not adopting one.

**Registers consulted.** *open-problems* — [#497](https://github.com/NGL321/patchworks/issues/497) is
this pass's subject in its own words (*"a per-map floor that succeeds per map does not spread it"*);
[#335](https://github.com/NGL321/patchworks/issues/335) (the band's projection can only shorten
retention) is §3's mechanism from the other side; [#482](https://github.com/NGL321/patchworks/issues/482)
and [#339](https://github.com/NGL321/patchworks/issues/339) are adjacent and not this. *proposed-solutions*
— [#526](https://github.com/NGL321/patchworks/issues/526) is §3's subject by name;
[#315](https://github.com/NGL321/patchworks/issues/315) (holonomy as a read) is §4's instrument and is
not adopted here; [#319](https://github.com/NGL321/patchworks/issues/319) and
[#318](https://github.com/NGL321/patchworks/issues/318) are noted where they bear. *dismissed-solutions*
— [#346](https://github.com/NGL321/patchworks/issues/346), the cross-edge coherence term, is `failed`
against `detectability`. **It is not reopened**, and #497 has already ruled that it is not this: it was
a uniform per-hop factor in a common rim coordinate judged against a detectability shortfall, and this
is composed rank. §4 says plainly where a candidate would come near it.

## Reading-depth key

#148's key, used throughout.

- **[FULL]** — paper body read.
- **[ABS]** — authoritative abstract / landing page, quoted verbatim.
- **[CITE]** — citation confirmed to exist, text not reached.
- **[UNREACHED]** — existence not confirmed.

---

## Headline verdict, stated plainly

**Nothing in the literature arrests composed rank collapse without an additive identity term, and that
is ADR-0008's forbidden shape exactly.** Every result found that *provably* stops a long composition of
linear maps from converging to rank one does it the same way — a residual path carrying the identity
around the collapsing operator — and the three independent literatures that have met this problem
(transformers, graph neural networks, deep linear networks) converge on that one remedy. The
self-supervised collapse-prevention family is not an alternative: it is batch statistics all the way
down, and its one genuinely local descendant replaces the prediction rule rather than repairing it.

**Five things follow, and they are separable.**

1. **The local collapse is not a pathology; it is Oja's theorem.** A normalised Hebbian-shaped rank-one
   update on a single unit *provably* converges to the top principal direction — rank one is the
   fixed point, not the failure mode (Oja 1982). The classical fixes all work by adding output units
   and a decorrelating term to the update. Every one of them is an **additive term**.
2. **Local rank and composed rank are decoupled, and the literature says why.** Rank is monotonically
   non-increasing under composition as a matter of algebra (Feng et al. 2022), so no per-factor
   constraint can be sufficient for a product. This is not a conjecture and T3's decoupling is its
   expected consequence.
3. **The exact mathematics of §4 is products of projections, not products of random matrices.** Each
   restriction map is projected onto the *nearest scaled co-isometry* every tick — `F ← (‖F‖_F/√m)·UVᵀ`,
   `restriction.py::_flatten` — so every factor is perfectly conditioned by construction and the
   composite's spectrum is set *entirely* by the principal angles between successive carried subspaces.
   In finite dimensions a cyclic product of subspace projections converges **geometrically** to the
   projection onto their **intersection** (von Neumann; Halperin 1962; rate: Kayalar & Weinert 1988;
   N-subspace generalisation: Badea, Grivaux & Müller 2010). Composed rank 1.000 over seven hops is
   what that theory predicts when the intersection is one-dimensional.
4. **The project's own numbers already falsify sufficiency, and they are the cleanest instance of it
   found anywhere.** Per-map flatness `σ_min/σ_max` reads **0.9999995 median** across 1339 of 1364 maps
   at both 30k and 100k on three seeds (`prototypes/spectral-floor-435/read.json`), per-map effective
   rank **4.000 of 4** (#435) — and the seven-hop composite reads **1.000** (#436/#497), *below* its own
   flat-and-chance-aligned null of **1.107**. Each factor is an isometry; the product is a line.
5. **The remedy the literature actually proves is already named in this repo as an ADR-level act.**
   #526's leak `K ← K − μ(K − aI)` is not a local hack: it is the same object as GCNII's identity
   mapping and as the residual connection Chen, Lin, Chen, Polyanskiy & Rigollet (2025) prove mitigates
   oversmoothing *via the multiplicative ergodic theorem* — the exact mechanism of §4. It is the general
   fix, it costs ADR-0008, and #526 is right to be a proposal rather than a change.

**What this does not do.** It does not choose. It does not read anything on the shallow dome. It does
not claim #526 would raise *composed* rank — §3 is explicit that the leak is on `K`, which is not a
factor in the composed transport object at all, and that gap is the pass's sharpest negative.

---

## 0. The object, stated once, from the code

Everything below is matched against this and nothing else.

- **The update.** `PredictionRule.step` (`learning.py`): `b ← b − η∇b`, `K ← Π(K − η_K∇K)`, one tick,
  one cell, gradient of prediction error through the cell's own forward path. The docstring states the
  ADR-0008 constraint in its own voice: *"That fight is not damped by an additive term: the prediction
  rule's objective is prediction error and nothing else."* [FULL, repo]
- **The prior.** `K` is initialised at exactly `a·I` (`body.py::CellOperators.__init__`), so its
  effective rank at construction is maximal and #526's target is the initialisation.
- **The band.** `CellOperators.project` restores `σ_max(K) ∈ [1/ρ_K, 1]` **by rescaling the whole
  operator by one scalar**. A uniform scalar is rank-neutral: it moves no singular-value *ratio*. So the
  band does not itself destroy rank — it pins the growing direction at the ceiling while the directions
  that are not growing fall in absolute terms, which is #477's signature (`σ_max` at 0.9947, `ρ` down
  3–7x) read correctly. ADR-0015's mechanism is now a **forward** normalisation of the same band
  (#466/#433, [ADR-0015](../adr/0015-the-cell-operator-band-is-on-the-spectral-norm.md) as amended);
  the ratio argument is unchanged.
- **The transport factor.** `RestrictionMaps.project` runs `_flatten`: `F ← (‖F‖_F/√m)·UVᵀ` on the
  active block — the projection onto the nearest scaled co-isometry, ADR-0032's floor at its one
  derivable value. **Every reachable factor is a scaled partial isometry.**
- **The composed object.** `prototypes/cold-start/T2/run.py::composed_reads` and
  `benchmarks/floor_price.py`: one chain per rim cell, the graph's shortest edge path to an apex cell,
  hops `F_out · F_inᵀ` composed in float64, effective rank = participation ratio `(Σσ²)²/Σσ⁴`.
  **`K` does not appear in it.** Seven edges, six interior hops on the full dome.

That last line is load-bearing for everything in §3.

---

## 1. Rank collapse under online, normalised, rank-one rules

### 1.1 Rank one is the theorem, not the bug

**Oja, E. (1982), "Simplified neuron model as a principal component analyzer", *Journal of Mathematical
Biology* 15:267–273.** [ABS+, landing page and secondary statements; the UCSD-hosted PDF is the primary
and was not extracted]

A single linear unit under Hebbian update with the normalisation folded in converges to the **first
principal component** of a stationary input stream. The rule is exactly this project's shape — an outer
product with a normalisation that keeps the weight bounded — and its proved asymptotic state is a
**single direction**.

**What this settles.** The ticket asks *when does a normalised rank-one update rule provably lose rank*.
The answer is: **always, for a single unit, by construction, and that is what the rule is for.** The
apex is not malfunctioning; it is doing the one thing a normalised Hebbian-shaped unit is proved to do.
That reframes the question from *what is broken* to *what did the classical work add to get more than
one direction* — which is §1.2, and the answer is uniformly bad news for ADR-0008.

### 1.2 Every classical fix is an additive decorrelating term on the update

The named family — Oja's subspace rule (1989), Sanger's Generalized Hebbian Algorithm (1989), Földiák's
anti-Hebbian lateral inhibition (1990), Rubner–Tavan, the APEX network — all buy `k` directions the same
way: **more than one output unit per cell, plus a term in the update that decorrelates them.**
[CITE for the individual papers; the family structure is standard and is stated here as such rather than
quoted]

Three costs, and they compound:

- **The extra term is additive on the weight update.** GHA's update is the Hebbian outer product *minus*
  a Gram–Schmidt-shaped correction; the subspace rule's is the outer product minus a reconstruction
  term. Under ADR-0008 as written — *"the prediction rule's objective is prediction error and nothing
  else"* — that is the forbidden shape, and it is forbidden for the same reason #526 is a proposal.
- **ADR-0008 already refused this family by name.** Its *Alternatives considered* rejects
  *"Hebbian or forward-forward-style updates"* in favour of gradient descent, on the ground that
  predictive coding already implies a comparison that is a gradient target. So adopting a Hebbian
  decorrelating rule is not one amendment but two: the additive term, and the rule's species.
- **It buys the wrong quantity.** Decorrelation acts on a cell's *output units*, i.e. on local
  excitation rank. T3 has already measured that lever: T1's winner nearly doubles the apex's centred
  participation ratio (1.034 → 1.808) and moves composed rank **not at all** (1.000 → 1.000). §2's
  decoupling result says this was predictable.

### 1.3 What no source supplies

**No source was found that treats the effective rank (participation ratio) of a single operator under a
normalised online rank-one update as a dynamical quantity with a floor.** The Hebbian literature works
in the `t → ∞` limit and reports *which* subspace is reached, not how the spectrum's shape evolves
under a band. #477's signature — `σ_max` pinned while `ρ` falls 3–7x — has no counterpart in it. This
is a genuine gap and it is stated as one rather than papered over.

---

## 2. The self-supervised family, and which parts are local

### 2.1 The decisive question, answered plainly: none of them is local and batch-free

VICReg (Bardes, Ponce & LeCun, ICLR 2022, [arXiv:2105.04906](https://arxiv.org/abs/2105.04906)), Barlow
Twins (Zbontar et al. 2021, [arXiv:2103.03230](https://arxiv.org/abs/2103.03230)), W-MSE whitening
(Ermolov et al. 2021), and the dimensional-collapse analyses (Jing, Vincent, LeCun & Tian, ICLR 2022,
[arXiv:2110.09348](https://arxiv.org/abs/2110.09348); Hua et al., ICCV 2021,
[arXiv:2105.00470](https://arxiv.org/abs/2105.00470)) are **already read in
[`docs/research/394`](./394-kernel-versus-rank-citations.md) §2.4** and are not re-fetched. #394's
verdict stands and this pass sharpens it on the one axis #394 did not need:

**Variance and covariance regularisers are batch statistics by definition.** VICReg's variance term is a
hinge on the standard deviation of each embedding dimension *across a batch*; Barlow Twins' objective is
a cross-correlation matrix *between two views' batch outputs*; W-MSE whitens *a batch*. Patchworks has
one tick, one cell, and **no second view of the same input** — which excludes the twin-network family
twice over, since the redundancy-reduction objective is defined on a pair of views and there is no pair.

**Is an EMA a legitimate substitute?** A streaming exponential-moving-average estimate of a cell's own
output covariance is computable from what the cell already owns, so it does not breach ADR-0011's
locality. But it is a batch statistic in disguise on the axis that matters here: it estimates the same
population quantity, over a *temporal* window instead of a batch axis, and the map's own instrument
already shows what that costs — T0's excitation rank read **1.00 for essentially every cell** uncentred,
because every stream is DC-dominated (ledger row 1). A covariance taken over a window whose leading
component is a constant is measuring the constant. **Recorded as available-but-degenerate on this
surface**, which is a stronger negative than "not local".

**And it would still be the wrong quantity.** Every member of this family regularises the *representation*,
i.e. a cell's own outputs. That is local excitation rank, which §1.2 already showed is decoupled from the
composed object.

### 2.2 The one genuinely local member, and what it costs

**Journé, Rodriguez, Guo & Moraitis (2023), "Hebbian Deep Learning Without Feedback" (SoftHebb), ICLR,
[arXiv:2209.11883](https://arxiv.org/abs/2209.11883).** [ABS]

SoftHebb trains deep networks *"without any feedback, target, or error signals"*, avoiding
*"weight transport, non-local plasticity, time-locking of layer updates, iterative equilibria, and
(self-)supervisory or other feedback signals"* — the strictest locality in the literature, and it
reaches 80.3% on CIFAR-10 and 27.3% on ImageNet with five hidden layers.

**Cost here, and it is total.** Its anti-collapse mechanism is a **soft winner-take-all** competition
among a layer's units — a nonlinearity plus lateral competition. Inside a cell that does not touch
ADR-0004 (nonlinearity is permitted inside the cell). But it is *not a repair to the prediction rule*;
it **replaces** it, it is the Hebbian species ADR-0008 refused by name, and it carries no prediction
error at all — the signal ADR-0008 built the rule on. **It is a different architecture, not a candidate
remedy**, and it is listed so the ranking is honest about having looked.

### 2.3 Neural collapse is a different phenomenon and is not carried

Papyan, Han & Donoho (2020, PNAS) describe terminal-phase collapse of *class means* under a supervised
task loss. There are no classes here and no task loss, and importing the name would be #481's mistake
in a new costume. **Not used.**

---

## 3. Decay toward a prior as a rank-preserving device

### 3.1 The sign matters, and the literature points both ways

**Decay toward zero provably *reduces* rank.** Galanti, Siegel, Gupte & Poggio,
*"SGD and Weight Decay Provably Induce a Low-Rank Bias in Deep Neural Networks"*,
[arXiv:2206.05794](https://arxiv.org/abs/2206.05794) (and the 2025 conference version, PMLR 280). [ABS]
Training with mini-batch SGD and weight decay *"causes a bias towards rank minimization over the weight
matrices"*, more pronounced at smaller batch size, higher learning rate and larger decay, and
*"weight decay is essential for inducing this low-rank bias."*

**So the naive reading of #526 is exactly backwards, and #526 does not make it.** `K ← K − μ(K − aI)`
is decay toward `a·I`, a **full-rank** point, not toward `0`. The two operations have opposite rank
effects and the literature's famous result is about the other one. **This pass's contribution here is
the sign**: a later session reaching for "weight decay is known to be rank-reducing" as an objection to
#526 would be citing a result about a different fixed point.

### 3.2 Decay toward the identity is the general fix, and it is proved

**Chen, Z., Lin, Z., Chen, S., Polyanskiy, Y. & Rigollet, P. (2025), "Residual connections provably
mitigate oversmoothing in graph neural networks", [arXiv:2501.00762](https://arxiv.org/abs/2501.00762).**
[ABS, abstract quoted verbatim]

> *"we analyze the asymptotic oversmoothing rates of deep GNNs with and without residual connections by
> deriving explicit convergence rates for a normalized vertex similarity measure. Our analytical
> framework is grounded in the multiplicative ergodic theorem. Furthermore, we demonstrate that adding
> residual connections effectively mitigates or prevents oversmoothing across several broad families of
> parameter distributions."*

Two things make this the pass's most on-target source. It is the **same object** — a product of many
learned linear maps composed along a graph — and it is the **same mechanism as §4**: the multiplicative
ergodic theorem is Oseledets', the theorem about Lyapunov spectra of long matrix products. And what it
proves works is a residual, i.e. an additive identity.

**Chen, M., Wei, Z., Huang, Z., Ding, B. & Li, Y. (2020), "Simple and Deep Graph Convolutional Networks"
(GCNII), ICML, [arXiv:2007.02133](https://arxiv.org/abs/2007.02133).** [ABS] Two mechanisms, *initial
residual* and **identity mapping**, which together *"effectively relieves the problem of over-smoothing"*
and are what let the architecture go deep at all.

**Dong, Cordonnier & Loukas (2021), *Attention is Not All You Need*, ICML,
[arXiv:2103.03404](https://arxiv.org/abs/2103.03404).** [ABS — **already read**,
`docs/research/394` §2.4(d); not re-fetched] *"without skip connections or multi-layer perceptrons
(MLPs), the output converges doubly exponentially to a rank-1 matrix. On the other hand, skip connections
and MLPs stop the output from degeneration."*

**The convergence is the finding.** Three literatures, three objects, one remedy: an additive path
carrying the identity around the collapsing operator. **#526 is an instance of the general fix**, and
the ticket's question — *local hack or general fix* — has a clear answer.

### 3.3 What it is known to cost, and the one thing this pass cannot say

**Known costs, from the sources.** GCNII's identity mapping is a *plasticity–stability* trade priced as
a hyperparameter: the fraction held back is representational capacity not spent on the layer's own
transform. The residual-GNN analyses report that a residual **slows** the convergence rate rather than
creating a fixed point away from the collapsed set for every parameter family — Chen et al.'s wording is
*"mitigates or prevents"*, and which of the two obtains is family-dependent. Against #526's own claim
that a direction receiving no updates *"relaxes back to `a`"*, the honest reading is that the leak sets a
**balance** between the update and the pull, and where the update is coherent and persistent — which is
exactly the apex's condition, `‖ē‖ ~1e-3` at direction stability 0.95–0.99 — the balance can sit
arbitrarily close to the collapsed state at small `μ`. **It buys a rate, and the rate is a knob.**

**And the thing this pass must say plainly: `K` is not a factor in the composed object.** `composed_reads`
multiplies `F_out·F_inᵀ` hops and nothing else. So even granting #526 everything it claims, **no
mechanism has been stated by which a leak on `K` moves composed rim-to-apex effective rank.** It answers
P1, the apex retention collapse, and #497's failure is a different one. Any session that reads #526 as
an answer to composed rank is making an inference no measurement and no source supports.

---

## 4. The effective rank of a composed product of learned linear maps

The ticket's least-covered area. It is also the one where the project's own instruments already contain
the answer, and where the exact mathematics turns out to be **older and sharper** than the deep-learning
literature that rediscovered it.

### 4.1 Composition cannot increase rank — so no per-factor constraint can be sufficient

**Feng, R., Zheng, K., Huang, Y., Zhao, D., Jordan, M. & Zha, Z.-J. (2022), "Rank Diminishing in Deep
Neural Networks", NeurIPS 35:33054–33065, [arXiv:2206.06072](https://arxiv.org/abs/2206.06072).** [ABS,
abstract verbatim]

> *"We theoretically establish a universal monotonic decreasing property of network rank from the basic
> rules of differential and algebraic composition, and uncover rank deficiency of network blocks and deep
> function coupling."*

The property is *universal* and it comes from *composition itself*. `rank(AB) ≤ min(rank A, rank B)` is
the whole of it, and the effective-rank version inherits the direction. **This is the formal statement of
T3's decoupling**: a constraint that holds each factor open cannot force the product open, because the
product's rank is bounded by an interaction between factors that no factor owns.

### 4.2 The exact object is a product of projections, and the theory is classical

Because `_flatten` makes every reachable map a **scaled partial isometry**, each hop `F_out·F_inᵀ` is
(up to a positive scalar) an isometry from one cell's carried subspace onto the next lane's, and the
composite's singular values are products of cosines of the **principal angles between successive carried
subspaces**. That is the alternating-projections object.

- **von Neumann; Halperin, I. (1962), "The product of projection operators", *Acta Sci. Math. (Szeged)*
  23:96–99.** [CITE] A cyclic product of orthogonal projections converges to the projection onto the
  **intersection** of the subspaces.
- **Kayalar, S. & Weinert, H. L. (1988), "Error bounds for the method of alternating projections",
  *Mathematics of Control, Signals, and Systems* 1:43–59,
  [doi:10.1007/BF02551235](https://doi.org/10.1007/BF02551235).** [CITE — Springer is behind an IdP
  redirect; the bibliographic record and the shape of the bound were confirmed from two independent
  secondary retrievals and are marked accordingly] The sharp rate is a power of the cosine of the
  **Friedrichs angle** between the subspaces — geometric in the number of factors.
- **Badea, C., Grivaux, S. & Müller, V. (2010), "The rate of convergence in the method of alternating
  projections", [arXiv:1006.2047](https://arxiv.org/abs/1006.2047).** [ABS, verbatim]
  *"A generalization of the cosine of the Friedrichs angle between two subspaces to a parameter
  associated to several closed subspaces of a Hilbert space is given. This parameter is used to analyze
  the rate of convergence in the von Neumann–Halperin method of cyclic alternating projections."*
  This is the **N-subspace** form, which is the one a seven-hop chain needs.
- **Kopecká, E. (2019), "When products of projections diverge",
  [arXiv:1901.01921](https://arxiv.org/abs/1901.01921), J. London Math. Soc.** [ABS, verbatim]
  *"in the cyclic case there is a dichotomy: the convergence is fast if and only if
  `L₁^⊥ + ⋯ + L_K^⊥` is closed; otherwise the convergence is arbitrarily slow."*
  **In finite dimensions every such sum is closed**, so on this dome the fast branch is the only branch:
  the collapse is not merely possible, it is **geometric in the hop count**.

**The consequence, stated for this project.** The composite's effective rank is, to a good approximation,
the dimension of the near-**intersection** of the seven carried subspaces along the chain. Seven generic
subspaces of dimension 3–4 inside a 12-wide chart intersect in one direction or none. **Composed
effective rank 1.000 is the generic answer, and getting above 1.5 means engineering a ≥2-dimensional
subspace common to every hop of every chain.**

**Calibration, and it matters.** The exact theorems are about *orthogonal projections*; the hops here are
products of *partial isometries* with differing lane widths (`interior_m = 3`, `boundary_m = 4`) and
mask-restricted active blocks. The mechanism transfers — principal angles between successive subspaces,
geometric in the number of factors — but no theorem quoted above applies to this object verbatim. Stated
as a mechanism match, not a proof.

### 4.3 The same result, rediscovered in the deep-learning literature

- **Oono, K. & Suzuki, T. (2020), "Graph Neural Networks Exponentially Lose Expressive Power for Node
  Classification", ICLR, [arXiv:1905.10947](https://arxiv.org/abs/1905.10947).** [ABS, verbatim]
  *"when its weights satisfy the conditions determined by the spectra of the (augmented) normalized
  Laplacian, its output exponentially approaches the set of signals that carry information of the
  connected components and node degrees only."* This is the closest architectural match in the whole
  pass — linear message passing over a graph, collapsing exponentially in depth to an invariant
  subspace. **And the escape route is scale**: the paper's guideline is a *weight normalisation*, i.e.
  keeping the layer weights' singular values **large enough** relative to the graph spectrum. On this
  surface that route is closed by construction — ADR-0015 caps `σ_max(K) ≤ 1` and ADR-0032's flattening
  moves `σ_max` down by `√m` per hop, which [ADR-0022](../adr/0022-a-hop-is-an-operator-norm-along-a-learned-channel.md)
  prices at **`2⁷ = 128x` across the rim-to-apex seven** (#436). **The project sits deep inside the
  collapse regime, on purpose, for reasons that have nothing to do with rank.**
  *(The precise `s·λ < 1` form of the condition is widely quoted; it was not verified against the
  theorem statement here and is not relied on. The scale direction is from the abstract.)*
- **Dong, Cordonnier & Loukas (2021)** — §3.2 above; *doubly exponentially* to rank one, arrested by
  skip connections and MLPs.
- **The GNN remedy set**, for completeness and for what each costs here: **GCNII** (initial residual +
  identity mapping) — additive identity, ADR-0008; **PairNorm** (Zhao & Akoglu 2020,
  [arXiv:1909.12223](https://arxiv.org/abs/1909.12223)) — keeps *total pairwise embedding distance*
  constant, a statistic over the whole node set, so it is a batch statistic in graph clothing;
  **DropEdge** — stochastic edge removal, which changes the graph and is *"enriching the sandbox"*'s
  structural cousin; it also only *delays* collapse. None is free here.

### 4.4 Is per-factor conditioning sufficient for the product? Measured no, in this repo

This is the ticket's direct question and the answer does not need the literature.

| reading | value | source |
|---|---|---|
| per-map flatness `σ_min/σ_max`, median, 30k **and** 100k, seeds 0/1/2 | **0.9999995** (worst 0.99997; 1339 of 1364 maps reached) | `prototypes/spectral-floor-435/read.json` |
| per-map effective rank | **4.000 of `m` = 4** | [#435](https://github.com/NGL321/patchworks/issues/435) |
| composed rim-to-apex effective rank, 263 chains, 7 edges | **1.000** at 30k and 100k | [#436](https://github.com/NGL321/patchworks/issues/436), [#497](https://github.com/NGL321/patchworks/issues/497) |
| the same, flat maps at **chance** alignment (null) | **1.107** | #436 |
| holonomy channel return `\|⟨u₁,v₁⟩\|`, 260 cycles | **0.9881** floored, against a 0.399 chance null and a 0.457 rewired null | [#453](https://github.com/NGL321/patchworks/issues/453) |
| holonomy flatness `σ_min/σ_max` around a cycle | **1.5e-5** floored, **8.5e-10** unfloored | #453 |

**Every factor is an isometry to six decimal places and the product is a line.** That is as clean a
falsification of *per-factor conditioning is sufficient* as the literature contains, and it is in-house.

**Two refinements that keep the statement honest**, both from #436's own correction and #453:

- **The ceiling is reachable in principle.** #436's ruling: *"holonomy = I composes flat maps into a flat
  composite at effective rank `m`."* So flat factors **plus trivial holonomy** *is* sufficient. What is
  insufficient is flatness alone. The missing ingredient is a **cross-factor** quantity.
- **The failure is not misalignment.** #497 rules this explicitly — *"Alignment is fine; there is one
  thing to align."* #453 measured the transport rule driving cross-edge alignment **toward** the
  identity, at 0.9881 channel return against 0.399 chance, at every cycle length and both horizons.
  The surface aligns **one** direction almost perfectly and sits at chance on the rest; composition then
  suppresses the rest geometrically, which is §4.2's mechanism operating on the residual angles. A
  candidate must therefore make **more than one** direction align consistently along a whole chain —
  not improve alignment in general, which is already near-perfect where it exists.

### 4.5 What nobody supplies

**No source was found that states a constraint computable at one factor which is sufficient for the
effective rank of a product of many such factors.** #497 says the same thing from inside the project —
*"closing it needs something that is not a per-map constraint, which is a design act nobody has taken"* —
and this pass confirms the literature has not taken it either. **That is the pass's frontier claim**, and
it is narrower and better founded than "the literature is silent": the field has met the problem three
times, and each time reached for an architectural additive path rather than a per-factor constraint,
because a per-factor constraint provably cannot do it (§4.1).

---

## 5. The candidates, ranked

Ranked by **(evidence it moves the *composed* quantity) ÷ (constraint cost)**. A candidate that only
moves local rank is ranked below one that moves nothing, because T3 has already shown local rank moving
without the composite following, and a lever that is known to be disconnected is worse than an unknown.

| # | candidate | moves composed rank? | ADR cost | how it reads on the instruments |
|---|---|---|---|---|
| **1** | **Additive identity / residual on the composed path** (GCNII's identity mapping; Dong et al.'s skip connection; Chen et al. 2025's residual) | **Yes — the only family with a proof.** Same object, same mechanism (multiplicative ergodic theorem) | **ADR-0008** (additive term), and ADR-0022 if the path is a second route between cells | composed rim-to-apex effective rank rises off 1.000 toward the null's 1.107 and beyond; holonomy flatness around a cycle rises off 1.5e-5; per-map readings unchanged |
| **2** | **A cross-chain / multi-direction alignment constraint** — make ≥2 directions align route-free along a whole path | Plausibly yes; it is the quantity §4.2 says governs the product. **No source, and no proof** | **ADR-0008** if a term; ADR-0011 locality if it reads a cycle (a cycle is incident to no cell — #453's own point). Comes **near** #346 and is not it (per #497) | holonomy identification departure on the *whole* operator moves off its 0.888 floor; composed rank follows; #315's rig reads it directly |
| **3** | **#526's leak `K ← K − μ(K − aI)`** | **Not shown, and no mechanism stated** — `K` is not a factor in the composed object at all | **ADR-0008** (additive term) | apex `ρ(K)` rises; `‖ē‖` falls; per-cell excitation PR rises; **composed rank predicted unchanged**, and that prediction is the cheap falsifier |
| **4** | **Raising the per-hop scale** (Oono & Suzuki's weight-normalisation guideline) | Yes in their setting — it is *the* stated escape from exponential collapse | **ADR-0015** (the band's ceiling is exactly 1) **and ADR-0022/ADR-0032** (the `√m` per-hop price is the constraint's derivable value). Three ADRs | `σ_max` along the chain rises off the 128x loss; composed rank rises; the incoherence cap starts biting |
| **5** | **Hebbian decorrelating rules** (Oja subspace, GHA/Sanger, anti-Hebbian lateral inhibition) | **No** — buys local excitation rank, which T3 measured decoupled | **ADR-0008 twice**: an additive term, *and* the Hebbian species its *Alternatives considered* refused by name | per-cell excitation PR rises; composed rank unchanged (T3's already-run experiment) |
| **6** | **VICReg / Barlow Twins / W-MSE** | No — regularise a cell's own representation | **Batch statistics**, and Barlow/W-MSE additionally need **a second view**, which does not exist. Plus ADR-0008 for the term | not readable: the instrument that would show the local gain (uncentred excitation PR) reads 1.00 everywhere on this surface (ledger row 1) |
| **7** | **PairNorm** | Delays collapse in GNNs | **Batch statistics** (a statistic over the whole node set) + ADR-0008 | — |
| **8** | **SoftHebb** | Not evaluated on a composed linear chain | Replaces the prediction rule outright; ADR-0008's refused species; no prediction error | — |
| **9** | **DropEdge** | Only delays | Changes the graph; *"no edge is ever removed"* is a construction invariant | — |
| — | **Anything reward-shaped** | — | **Excluded at the door** by #5's bar. No candidate above is task-loss-driven, and none was admitted on those grounds | — |

**The binding constraint is ADR-0008.** Candidates 1, 2, 3 and 5 all cost it and nothing else decisive;
candidate 4 costs three other ADRs instead and is the only route that avoids it. That is the pass's
practical shape: **the choice is between amending ADR-0008 once, or amending ADR-0015/0022/0032
together.**

**And the honest negative, which is the pass's headline restated.** *Nothing in the literature fits the
constraints as written.* Every remedy with a proof behind it is an additive identity term. The ticket
said a finding of that kind is a real result; this is that finding.

---

## 6. What could not be reached, stated plainly

Ranked by how much it weakens what is above.

**Costliest.**

- **Kayalar & Weinert (1988).** Springer's landing page redirects to an identity provider and the AMS
  survey PDF returned 403. The bibliographic record (Math. Control Signals Systems 1:43–59, 1988,
  doi:10.1007/BF02551235) was confirmed from two independent retrievals, and the *shape* of the result
  — a sharp geometric rate in the cosine of the Friedrichs angle — from secondary statements. **§4.2's
  argument does not depend on the constant**, only on the fact that the rate is geometric in the number
  of factors, which Badea et al.'s abstract independently supports at [ABS] depth. Marked [CITE].
- **Oono & Suzuki's theorem statement.** Both the arXiv PDF and the OpenReview PDF failed extraction
  (binary stream; browser check). The abstract was read verbatim and is what §4.3 quotes. The widely
  repeated `s·λ < 1` condition is **not** relied on, and the claim actually made — that the escape
  direction is *larger* weight scale — is from the abstract's own *"principled guideline for weight
  normalization"* plus the paper's title. A follow-up wanting to price candidate 4 properly must read
  the theorem.
- **The classical Hebbian papers** (Oja 1989 subspace rule; Sanger 1989; Földiák 1990; Rubner–Tavan;
  APEX). Marked [CITE] throughout: the family's *structure* — extra output units plus an additive
  decorrelating term — is textbook and is stated as such, but no individual paper's update equation was
  read at source in this pass. §1.2's ADR-0008 cost turns on that structure, so a session that wanted to
  contest the cost should start here.

**Moderate.**

- **Oja (1982)** was read at abstract-plus-secondary depth; the UCSD-hosted PDF was located and not
  extracted. The convergence-to-first-principal-component claim is uncontroversial and multiply
  corroborated.
- **GCNII's identity-mapping cost** is stated from the abstract and from secondary comparisons of the
  oversmoothing remedies; the hyperparameter trade was not read from the paper's own experiments.
- **Chen et al. (2025)** at [ABS]. The distinction its abstract draws — *"mitigates or prevents"* — is
  load-bearing for §3.3's "it buys a rate" and was **not** resolved against the theorem statement. If a
  session wants to claim a residual creates a fixed point rather than a slower rate, this is the read.

**Not searched at all**, recorded so the ground is not re-covered blindly: free-probability treatments of
deep-network spectra (Pennington–Worah, Pennington–Schoenholz–Ganguli) and Hanin & Nica on products of
many large random matrices — all of which bear on §4 and none of which was needed once the
alternating-projections thread proved exact; the reservoir-computing literature on echo-state property
and memory capacity, which is the other field that constrains `σ_max` of a recurrent linear operator and
might state the retention/rank trade in its own terms; and the control-theory literature on products of
sub-stochastic matrices and consensus, which is oversmoothing's older sibling.

---

## 7. Sources

| Source | Depth | Access |
| --- | --- | --- |
| Oja (1982), *Simplified neuron model as a principal component analyzer*, J. Math. Biol. 15:267–273 | [ABS+] | landing page + secondary; primary PDF located, not extracted |
| Oja (1989) subspace rule; Sanger (1989) GHA; Földiák (1990); Rubner–Tavan; APEX | [CITE] | family structure only — see §6 |
| Journé, Rodriguez, Guo & Moraitis (2023), *Hebbian Deep Learning Without Feedback* (SoftHebb), ICLR, [arXiv:2209.11883](https://arxiv.org/abs/2209.11883) | [ABS] | landing page + repo |
| Bardes, Ponce & LeCun (2022), *VICReg*, [arXiv:2105.04906](https://arxiv.org/abs/2105.04906) | [ABS] | **already read**, `docs/research/394` §2.4(a); not re-fetched |
| Zbontar, Jing, Misra, LeCun & Deny (2021), *Barlow Twins*, [arXiv:2103.03230](https://arxiv.org/abs/2103.03230) | [CITE] | — |
| Jing, Vincent, LeCun & Tian (2022), *Understanding Dimensional Collapse in Contrastive SSL*, [arXiv:2110.09348](https://arxiv.org/abs/2110.09348) | [ABS] | **already read**, `docs/research/394` |
| Hua et al. (2021), *On Feature Decorrelation in SSL*, [arXiv:2105.00470](https://arxiv.org/abs/2105.00470) | [ABS] | **already read**, `docs/research/394` |
| Galanti, Siegel, Gupte & Poggio, *SGD and Weight Decay Provably Induce a Low-Rank Bias*, [arXiv:2206.05794](https://arxiv.org/abs/2206.05794) | [ABS] | landing page + PMLR 280 record |
| Chen Z., Lin, Chen S., Polyanskiy & Rigollet (2025), *Residual connections provably mitigate oversmoothing in GNNs*, [arXiv:2501.00762](https://arxiv.org/abs/2501.00762) | [ABS] | abstract verbatim |
| Chen M., Wei, Huang, Ding & Li (2020), *Simple and Deep Graph Convolutional Networks* (GCNII), ICML, [arXiv:2007.02133](https://arxiv.org/abs/2007.02133) | [ABS] | landing page |
| Dong, Cordonnier & Loukas (2021), *Attention is Not All You Need*, ICML, [arXiv:2103.03404](https://arxiv.org/abs/2103.03404) | [ABS] | **already read**, `docs/research/394` §2.4(d) |
| Oono & Suzuki (2020), *GNNs Exponentially Lose Expressive Power*, ICLR, [arXiv:1905.10947](https://arxiv.org/abs/1905.10947) | [ABS] | abstract verbatim; PDFs would not extract — see §6 |
| Zhao & Akoglu (2020), *PairNorm*, ICLR, [arXiv:1909.12223](https://arxiv.org/abs/1909.12223) | [CITE] | — |
| Feng, Zheng, Huang, Zhao, Jordan & Zha (2022), *Rank Diminishing in Deep Neural Networks*, NeurIPS 35:33054–33065, [arXiv:2206.06072](https://arxiv.org/abs/2206.06072) | [ABS] | abstract verbatim |
| Halperin (1962), *The product of projection operators*, Acta Sci. Math. (Szeged) 23:96–99 | [CITE] | via two expository sources |
| Kayalar & Weinert (1988), *Error bounds for the method of alternating projections*, Math. Control Signals Systems 1:43–59, [doi:10.1007/BF02551235](https://doi.org/10.1007/BF02551235) | [CITE] | **contested access** — see §6 |
| Badea, Grivaux & Müller (2010), *The rate of convergence in the method of alternating projections*, [arXiv:1006.2047](https://arxiv.org/abs/1006.2047) | [ABS] | abstract verbatim |
| Kopecká (2019), *When products of projections diverge*, [arXiv:1901.01921](https://arxiv.org/abs/1901.01921), JLMS | [ABS] | abstract verbatim |
| Feshchenko (2019), *On the optimal error bound for the first step in cyclic alternating projections*, [arXiv:1908.00531](https://arxiv.org/abs/1908.00531) | [ABS] | abstract; used only for the `‖P_n···P_1 − P_0‖` object |
| Papyan, Han & Donoho (2020), *Neural collapse*, PNAS | [CITE] | **not used** — see §2.3 |

## Context

Written for [#528](https://github.com/NGL321/patchworks/issues/528), on branch `research/rank-collapse`.
Repo material read before searching: ADR-0004, ADR-0008, ADR-0015, ADR-0022, ADR-0031, ADR-0032;
`src/patchworks/learning.py` (`PredictionRule.step`), `src/patchworks/body.py` (`CellOperators`),
`src/patchworks/restriction.py` (`project`, `_flatten`, `flatness`, `gram_peaks`);
`prototypes/cold-start/T2/run.py` (`composed_reads`, `rim_chains`) on `map/cold-start`;
`prototypes/spectral-floor-435/read.json`; `docs/agents/registers.md`; the three problem registers read
live from the tracker rather than from `docs/registers/*.md`; issues #315, #318, #319, #324, #335, #346,
#435, #436, #453, #477, #497, #517, #518, #520, #521, #522, #524, #526; and the prior passes
`docs/research/020` and `docs/research/394`. **Nothing in `CONTEXT.md`, the spec or any ADR is edited by
this pass, and no code is touched.**
