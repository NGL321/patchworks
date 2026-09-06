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
not adopted here; [#319](https://github.com/NGL321/patchworks/issues/319) (input-conditioned maps) is the
nearest existing row to candidate 1 and is **not** the same mechanism — §5 says where they differ;
[#318](https://github.com/NGL321/patchworks/issues/318) is noted where it bears. *dismissed-solutions* —
[#346](https://github.com/NGL321/patchworks/issues/346), the cross-edge coherence term, is `failed`
against `detectability`. **It is not reopened**, and #497 has already ruled that it is not this: it was
a uniform per-hop factor in a common rim coordinate judged against a detectability shortfall, and this
is composed rank. §5 says plainly where a candidate comes near it.

## Reading-depth key

#148's key, used throughout.

- **[FULL]** — paper body read.
- **[ABS]** — authoritative abstract / landing page, quoted verbatim.
- **[CITE]** — citation confirmed to exist, text not reached.
- **[UNREACHED]** — existence not confirmed.

---

## Headline verdict, stated plainly

**The literature offers exactly one constraint-clean sufficiency result for the rank of a long product —
and this project already implements it, and this project's own numbers already falsify it here.**

That is a sharper and more useful finding than *nothing fits*, and it is the pass's headline. The
sufficiency result is the **degenerate band**: if every factor has all its singular values equal, the
product is exactly a scaled orthogonal matrix at any depth, every Lyapunov exponent is equal, and there
is no spectral gap to grow. ADR-0032's `_flatten` imposes precisely that — `F ← (‖F‖_F/√m)·UVᵀ`, the
projection onto the nearest scaled co-isometry, every tick, measured holding to **0.9999995**. And the
seven-hop composite still reads **1.000**.

**The reconciliation is the finding.** The sufficiency argument holds for **square orthogonal** factors.
This project's maps are **rectangular partial isometries** — `m × k` blocks with `interior_m = 3`,
`boundary_m = 4`, masked — so each hop is an isometry *onto its own subspace* and the composite is a
product of projections onto seven differing subspaces. In finite dimensions that converges
**geometrically** to the projection onto their **intersection** (von Neumann; Halperin 1962;
Kayalar & Weinert 1988; Badea, Grivaux & Müller 2010; Kopecká 2019 for the dichotomy). Seven generic
3–4 dimensional subspaces inside a 12-wide chart intersect in one direction or none. **Composed rank
1.000 is not a failure of the floor; it is what a perfectly-floored rectangular chain is supposed to
do.**

**Six things follow, and they are separable.**

1. **The local collapse is a 1994 theorem about this exact configuration.** Miller & MacKay proved that
   **multiplicative** enforcement of a norm constraint on a linear Hebbian rule converges to the
   *principal eigenvector* of the unconstrained operator, while **subtractive** enforcement does not.
   The band is multiplicative — a single scalar rescale (`body.py::project`). Rank one is the theorem.
2. **Local rank and composed rank are decoupled, and there is a theorem for that too.** Rank is
   monotonically non-increasing under composition (Feng et al. 2022), and Roth & Liebig (2023) prove for
   graph message passing that the dominant subspace *"depends on the aggregation function but not on the
   feature transformations"*. T3's decoupling is the expected consequence, not a surprise.
3. **Per-factor conditioning is not sufficient, with primary sources and a universal lower bound.**
   Atnip, Froyland, González-Tokman & Quas (2026) prove a universal lower bound on Lyapunov gaps under
   perturbation, uniform over the matrix sequence and **with no stationarity assumption** — so it covers
   learned, drifting cocycles. Their own introduction states the point: singular values *"may be far from
   multiplicative"* and the submultiplicative bounds are *"far too weak to control limits"*.
4. **Everything with a proof behind it is an additive identity term.** Skip connections (Dong et al.
   2021), GCNII's identity mapping, and the residual whose oversmoothing mitigation Chen, Lin, Chen,
   Polyanskiy & Rigollet (2025) prove *via the multiplicative ergodic theorem* — the exact mechanism of
   §4. **#526's leak is an instance of that general fix**, not a local hack.
5. **But there is a genuine ambiguity in ADR-0008 worth surfacing rather than deciding.** A residual in
   *transport* is `h ← (I + T)h`: still linear, so ADR-0004 survives, and it is **not a term added to
   the operator update `ΔK`** — it is a structural property of the composition. Whether ADR-0008's
   no-additive-term clause reaches it is a reading this pass does not make. It is the difference between
   *the best-evidenced remedy is banned* and *the best-evidenced remedy is available*.
6. **The self-supervised family is unavailable, and for two independent reasons.** Its anti-collapse
   signal *is* an across-sample statistic, and its invariance signal *is* a second view. In that
   literature **"local" always means the gradient does not travel between blocks; it never means the loss
   is computed from one sample.** Layer-locality and batch-freeness are orthogonal, and the field has only
   ever bought the first.

**What this does not do.** It does not choose. It reads nothing on the shallow dome. And it does not
claim #526 would raise *composed* rank — §3 is explicit that `K` is not a factor in the composed
transport object at all, and that gap is the pass's sharpest negative.

---

## 0. The object, stated once, from the code

Everything below is matched against this and nothing else.

- **The update.** `PredictionRule.step` (`learning.py`): `b ← b − η∇b`, `K ← Π(K − η_K∇K)`, one tick,
  one cell, gradient of prediction error through the cell's own forward path. The docstring states the
  ADR-0008 constraint in its own voice: *"That fight is not damped by an additive term: the prediction
  rule's objective is prediction error and nothing else."* [FULL, repo]
- **The prior.** `K` is initialised at exactly `a·I` (`body.py::CellOperators.__init__`), so its
  effective rank at construction is maximal and #526's target is the initialisation.
- **The band is multiplicative, and that is load-bearing.** `CellOperators.project` restores
  `σ_max(K) ∈ [1/ρ_K, 1]` **by rescaling the whole operator by one scalar**. A uniform scalar is
  rank-neutral — it moves no singular-value *ratio* — so the band does not itself destroy rank. It pins
  the growing direction at the ceiling while the directions that are not growing fall in absolute terms,
  which is #477's signature (`σ_max` at 0.9947, `ρ` down 3–7x) read correctly. **§1.1 is the theorem
  about exactly this choice.** ADR-0015's mechanism is now a **forward** normalisation of the same band
  (#466/#433); the argument is unchanged.
- **The transport factor.** `RestrictionMaps.project` runs `_flatten`: `F ← (‖F‖_F/√m)·UVᵀ` on the
  active block. **Every reachable factor is a scaled partial isometry** — a *degenerate* band, not
  merely a bounded one. This is what makes §4.4 decisive.
- **The composed object.** `prototypes/cold-start/T2/run.py::composed_reads` and
  `benchmarks/floor_price.py`: one chain per rim cell, the graph's shortest edge path to an apex cell,
  hops `F_out · F_inᵀ` composed in float64, effective rank = participation ratio `(Σσ²)²/Σσ⁴`.
  **`K` does not appear in it.** Seven edges, six interior hops on the full dome.

That last line is load-bearing for everything in §3.

**An instrument note.** Effective rank in the sense of Roy & Vetterli (EUSIPCO 2007) is
`exp(−Σ σ̃ᵢ log σ̃ᵢ)`; the participation ratio `(Σσ²)²/Σσ⁴` is a different functional, and by Jensen
`PR ≤ exp(H)`. The two are **not interchangeable** and PR is the conservative one. Every reading in this
document is PR, which is what the rigs compute; a quoted PR of 1.808 is a floor on the entropy-based
number, not an estimate of it.

---

## 1. Rank collapse under online, normalised, rank-one rules

### 1.1 The multiplicative band is the mechanism, and it was proved in 1994

**Miller, K. D. & MacKay, D. J. C. (1994), "The Role of Constraints in Hebbian Learning", *Neural
Computation* 6(1):100–126, [doi:10.1162/neco.1994.6.1.100](https://doi.org/10.1162/neco.1994.6.1.100).**
[ABS, MIT Press landing page; wording below quoted from it]

Correlation-based Hebbian rules are unstable, and the usual repair is a constraint limiting total
synaptic strength. Miller & MacKay distinguish **two** ways to enforce such a constraint, and the
distinction is the whole point:

> *"For otherwise linear learning rules, multiplicative enforcement of a constraint results in dynamics
> that converge to the principal eigenvector of the operator determining unconstrained synaptic
> development."*

Subtractive enforcement instead gives a saturated, winner-take-all-like final state that is generically
**not** the principal eigenvector.

**This is the most on-point citation in the pass and it predates the deep-learning literature by
decades.** `CellOperators.project` rescales the whole operator by a single scalar — the definition of
multiplicative enforcement. Under a rule whose update is a coherent rank-one outer product, the theorem
says the dynamics converge to *one direction*, and the stated mechanism is that growth of the dominant
mode is paid for by proportional shrinkage of every other. **That is #477's signature in a 1994 paper's
own words**, and it means the apex is not malfunctioning: it is doing the one thing a multiplicatively
constrained linear Hebbian rule is proved to do.

### 1.2 The single-unit case is Oja's convergence theorem

**Oja, E. (1982), "Simplified neuron model as a principal component analyzer", *Journal of Mathematical
Biology* 15:267–273.** [ABS+] A single linear unit under a Hebbian update with the normalisation folded
in converges, by a Lyapunov argument, to the **first principal component**. Rank one is the fixed point.

The dome's rule `ΔK ∝ e hᵀ` followed by renormalisation is Oja's shape with `e` in the output role and
`h` in the input role. Power iteration converges at rate `(σ₂/σ₁)^t`, and a persistent error direction
makes `σ₁` grow every tick while the band divides everything by it. **`‖ē‖` direction stability of
0.95–0.99 is therefore not a symptom sitting beside the collapse — under Oja and Miller–MacKay it is the
sufficient condition for it.** That reframes T0's P1 read: the instrument was measuring the cause.

### 1.3 Every classical fix is an additive decorrelating term on the update

The named family all buy `k` directions the same way: **more than one output unit per cell, plus a term
in the update that decorrelates them.**

- **Oja, E. (1989), "Neural networks, principal components, and subspaces", *International Journal of
  Neural Systems* 1:61–68.** [ABS+] `W ← W + η(x yᵀ − W y yᵀ)` with `k` output units. Converges to the
  principal `k`-dimensional **subspace** — but the symmetric rule is rotationally invariant within it, so
  it spans the eigenspace without identifying the axes. Two terms, and the correction `W y yᵀ` is not a
  scalar rescale, so it cannot be folded into a multiplicative normalisation once `k > 1`.
- **Sanger, T. D. (1989), "Optimal unsupervised learning in a single-layer linear feedforward neural
  network", *Neural Networks* 2(6):459–473.** [ABS+] The Generalized Hebbian Algorithm folds a
  **Gram–Schmidt deflation** into the update, recovering the eigenvectors in order. The deflation term
  is subtracted *inside* the update, and unit `i` reads units `1..i−1`, so it is **not local** either.
- **Földiák, P. (1990), "Forming sparse representations by local anti-Hebbian learning", *Biological
  Cybernetics* 64:165–170, [doi:10.1007/BF02331346](https://doi.org/10.1007/BF02331346).** [ABS+]
  Anti-Hebbian **lateral** connections between output units decorrelate them. Genuinely local, online and
  batch-free — but the units are **nonlinear** and every such rule has the form
  `ΔM_ij ∝ (y_i y_j − M_ij)`: an anti-Hebbian term *and* an explicit decay term.

Rubner–Tavan (1989) and APEX (Kung & Diamantaras 1990) are the same shape and nothing rests on them.

**Three costs, and they compound.**

- **The extra term is additive on the weight update.** Under ADR-0008 as written — *"the prediction
  rule's objective is prediction error and nothing else"* — that is the forbidden shape, and it is
  forbidden for the same reason #526 is a proposal.
- **ADR-0008 already refused this family by name.** Its *Alternatives considered* rejects *"Hebbian or
  forward-forward-style updates"* in favour of gradient descent. Adopting one is **two** amendments, not
  one: the additive term, and the rule's species.
- **It buys the wrong quantity.** Decorrelation acts on a cell's *output units*, i.e. local excitation
  rank. T3 has already run this experiment: T1's winner nearly doubles the apex's centred participation
  ratio (1.034 → 1.808) and moves composed rank **not at all**. §4.1 says why that was predictable.

### 1.4 The strongest local family — and it still costs ADR-0008

**Pehlevan, C., Sengupta, A. & Chklovskii, D. B. (2018), "Why do similarity matching objectives lead to
Hebbian/anti-Hebbian networks?", *Neural Computation* 30(1):84–124,
[arXiv:1703.07914](https://arxiv.org/abs/1703.07914)**; and **Pehlevan & Chklovskii (2019),
"Neuroscience-inspired online unsupervised learning algorithms", *IEEE Signal Processing Magazine*,
[arXiv:1908.01867](https://arxiv.org/abs/1908.01867).** [ABS]

Derived from a similarity-matching objective, the online algorithm is strictly **local, online,
single-sample, batch-free and reward-free** — the best-credentialed family in the pass on three of the
four constraints — and with a modified objective it performs online **whitening**, which is precisely
"flatten the spectrum", i.e. maximise effective rank.

**Cost: ADR-0008, and it is structural.** Both update rules have the Hebbian-plus-decay form
`ΔW_ij ∝ (y_i x_j − W_ij)` and `ΔM_ij ∝ (y_i y_j − M_ij)`. The `−W_ij` term **is** `K ← K − μ(K − 0)`,
the forbidden shape with `a = 0`; it falls out of the Lagrangian and cannot be dropped without losing
the convergence proof. *(The update equations above are transcribed from the arXiv rendering and should
be checked against the published version before being quoted anywhere load-bearing — see §6.)*

### 1.5 What no source supplies

**No source treats the effective rank or participation ratio of a single operator under a normalised
online rank-one update as a dynamical quantity with a floor.** Searched for directly; not found. What
exists is (i) definitions without dynamics (Roy & Vetterli 2007), and (ii) convergence theorems that are
the *opposite* of a floor — they are proofs that the floor is 1 (Oja 1982; Miller & MacKay 1994). The
nearest thing in print to rank dynamics is at the singular-value level in a gradient-descent setting:
Jing et al.'s `σ̇₁ᵏ = σ₁ᵏ(σ₂ᵏ)²(v₁ᵏᵀXv₁ᵏ)` — growth **proportional to the current value**, so small
singular values grow strictly slower and the spectrum spreads without bound. That `σ̇ ∝ σ` form is the
mathematical signature of endogenous collapse and it fits this rule, but it is stated for gradient
descent on a linear stack and never for a normalised Hebbian rule.

**This is a genuine gap, and a place this project is positioned to contribute a result rather than adopt
one.** Stated as a gap, not as a plan.

---

## 2. The self-supervised family, and which parts are local

### 2.1 The decisive question, answered plainly: not one of them

VICReg ([arXiv:2105.04906](https://arxiv.org/abs/2105.04906)), Barlow Twins
([arXiv:2103.03230](https://arxiv.org/abs/2103.03230)), W-MSE
([arXiv:2007.06346](https://arxiv.org/abs/2007.06346)), DirectPred
([arXiv:2102.06810](https://arxiv.org/abs/2102.06810)) and DirectCLR
([arXiv:2110.09348](https://arxiv.org/abs/2110.09348)) are blocked twice over, and both blocks are
structural rather than incidental.

**(i) The anti-collapse signal *is* an across-sample statistic.** VICReg's variance term is a hinge on
each embedding dimension's standard deviation *across the batch*, and its covariance term zeroes the
off-diagonals of the *empirical batch covariance*; at batch size 1 the variance term is identically zero
and the covariance is rank one by construction. Barlow Twins' objective is a cross-correlation matrix
*normalised along the batch dimension* — the paper claims only that it does not need *large* batches.
W-MSE is the clearest case: its own framing is that whitening has a *scattering effect on batch samples*
substituting for negatives, and a batch covariance is singular at `n = 1`.

**(ii) The invariance signal *is* a second view.** All of these are joint-embedding architectures whose
learning signal is a distance between two augmented views of one input. **The dome has no second view.**
Deleting the invariance term deletes the only thing the variance/covariance terms are fighting, and the
objective degenerates to "spread out", which is not a learning signal.

**DirectPred is the nearest miss and deserves its name.** Its predictor is set in **closed form** from
input statistics rather than learned — no loss, no backprop through it — and its account of BYOL-style
non-collapse credits predictor, stop-gradient, EMA target and **weight decay** jointly. It still fails on
both counts: the EMA correlation matrix is a running batch statistic, and the online/target asymmetry is
a second view.

### 2.2 Is a streaming EMA covariance a legitimate substitute? No, and it is worse

An EMA `C_t = (1−γ)C_{t−1} + γ h_t h_tᵀ` is a sample covariance with an exponential kernel instead of a
uniform one. It carries the same population information a minibatch carries, and it adds three costs:

- **It is per-cell state that is neither `K` nor `h`** — a second memory the ADRs do not contemplate.
- **Any regulariser derived from it is an additive term on the operator update**, so it breaks ADR-0008
  *as well as* the batch-statistics constraint.
- **It is degenerate on this surface, which is the strongest of the three objections.** T0's ledger row 1
  reads the evidence stream's participation ratio at **1.00 for essentially every cell** uncentred,
  because every stream is DC-dominated. A covariance over that window is measuring the constant, and with
  direction stability 0.95–0.99 the EMA is itself rank-one-dominated — **it would confirm the collapse
  rather than counteract it.**

And it would still regularise the *representation*, which §1.3 and §4.1 both show is decoupled from the
composite.

### 2.3 "Local" in this literature never means batch-free

| method | layer-local? | batch-free? | second view? |
|---|---|---|---|
| Blockwise SSL at scale ([arXiv:2302.01647](https://arxiv.org/abs/2302.01647)) | yes (block gradients) | **no** — per-block Barlow Twins | yes |
| VICRegL ([arXiv:2210.01571](https://arxiv.org/abs/2210.01571)) | **no** — "local" means *spatially* local features | no | yes |
| Layer-wise VICReg losses (*Sci. Rep.* 2025, [doi:10.1038/s41598-025-08504-2](https://doi.org/10.1038/s41598-025-08504-2)) | yes | **no** | yes |
| Forward-Forward ([arXiv:2212.13345](https://arxiv.org/abs/2212.13345)) | yes | **no** — needs negative data and a threshold calibrated across examples | a second stream |
| **SoftHebb** ([arXiv:2209.11883](https://arxiv.org/abs/2209.11883)) | **yes** | **yes** | **no** |

**The generalisable point: in the SSL literature "local" always means the gradient does not travel
between blocks. It never means the loss is computed from one sample.** Layer-locality and batch-freeness
are orthogonal properties and the field has only ever bought the first. SoftHebb is the sole entry that
is both — and it is a Hebbian rule, not an SSL objective, which is why it belongs to §1.

**SoftHebb's cost.** Its anti-collapse mechanism is a **soft winner-take-all** among a cell's units:
Hebbian for the winner, anti-Hebbian for the losers. The nonlinearity is *inside* the cell, so ADR-0004
arguably survives — but the loser term is an additive second term (ADR-0008), it is ADR-0008's refused
species, and it **replaces** the prediction rule rather than repairing it, carrying no prediction error
at all. It is a different architecture, listed so the ranking is honest about having looked.

One transferable detail from Forward-Forward: it must normalise the hidden vector's length between
layers, because otherwise the next layer reads the goodness off the length and learns nothing. **A
normalisation introduced specifically to destroy a scalar that would otherwise dominate** is a useful
precedent for the claim that a normalisation choice is load-bearing for what downstream layers can
represent.

### 2.4 Neural collapse is a different phenomenon and is not carried

Papyan, Han & Donoho (2020, *PNAS* 117(40):24652–24663) describe terminal-phase collapse of **class
means** under a supervised loss, toward a maximally-spread simplex frame of rank `C−1` — not toward rank
one. No classes, no task loss, and the mechanism is a reward-shaped one. **Not used**, and importing the
name would be #481's mistake in a new costume.

---

## 3. Decay toward a prior as a rank-preserving device

### 3.1 The sign matters, and the literature points both ways

**Decay toward zero provably *reduces* rank.**

- **Galanti, Siegel, Gupte & Poggio, *SGD and Weight Decay Provably Induce a Low-Rank Bias in Deep Neural
  Networks*, [arXiv:2206.05794](https://arxiv.org/abs/2206.05794).** [ABS] Minibatch SGD with weight
  decay *"causes a bias towards rank minimization"*, stronger at smaller batch, higher learning rate and
  larger decay; **weight decay is essential** for it, and the analysis assumes nothing about data,
  convergence or architecture.
- **"Weight decay induces low-rank attention layers", [arXiv:2410.23819](https://arxiv.org/abs/2410.23819)
  (NeurIPS 2024).** [CITE] For **multiplicatively interacting** matrices — the dome's composed situation
  exactly — L2 on the factors coincides with **nuclear-norm** regularisation on the product, i.e. the
  convex surrogate for rank, and the rank drop is reported *even in fully online training*.

**So the naive reading of #526 is exactly backwards, and #526 does not make it.** `K ← K − μ(K − aI)` is
decay toward `a·I`, a **full-rank** point — and `K` is initialised at exactly `a·I`. The famous result is
about the other fixed point. **This pass's contribution here is the sign**: a later session reaching for
*weight decay is known to be rank-reducing* as an objection to #526 would be citing a theorem about a
different operation.

### 3.2 Decay toward the identity is the general fix, and three literatures prove it

- **Ledoit, O. & Wolf, M. (2004), "A well-conditioned estimator for large-dimensional covariance
  matrices", *J. Multivariate Analysis* 88(2):365–411.** [CITE] The archetype: `Σ̂ = (1−ρ)S + ρμI` is
  provably better conditioned than `S`, invertible even when `S` is rank-deficient, asymptotically
  optimal under quadratic loss. Shrinking toward `I` **raises the small eigenvalues off zero**. Note it
  shrinks toward `μI` with `μ = tr(S)/p`, the *average* eigenvalue — not toward 1.
- **Chen, M., Wei, Z., Huang, Z., Ding, B. & Li, Y. (2020), *Simple and Deep Graph Convolutional
  Networks* (GCNII), ICML, [arXiv:2007.02133](https://arxiv.org/abs/2007.02133).** [ABS] The
  identity-mapping layer is literally `((1−β_ℓ)I + β_ℓW^(ℓ))` with `β_ℓ ≈ λ/ℓ`, justified by exactly the
  Oono–Suzuki `s^K` argument of §4.3. **`β_ℓ ∝ 1/ℓ` is deliberate**: the identity's share *grows* with
  depth, so shallow layers stay plastic and only deep layers are heavily identity-biased.
- **Bansal, Chen & Wang (2018), *Can We Gain More from Orthogonality Regularizations in Training Deep
  CNNs?*, NeurIPS, [arXiv:1810.09102](https://arxiv.org/abs/1810.09102).** [CITE] Soft orthogonality
  `λ‖WᵀW − I‖_F²` and the stronger **spectral restricted isometry** variant, which penalises
  `|σ(WᵀW − I)|` directly — the closest published thing to "regularise toward isometry", and reported as
  the best of the family.
- **Chen, Z., Lin, Z., Chen, S., Polyanskiy, Y. & Rigollet, P. (2025), "Residual connections provably
  mitigate oversmoothing in graph neural networks",
  [arXiv:2501.00762](https://arxiv.org/abs/2501.00762).** [ABS, verbatim]
  > *"we analyze the asymptotic oversmoothing rates of deep GNNs with and without residual connections by
  > deriving explicit convergence rates for a normalized vertex similarity measure. Our analytical
  > framework is grounded in the multiplicative ergodic theorem. Furthermore, we demonstrate that adding
  > residual connections effectively mitigates or prevents oversmoothing across several broad families of
  > parameter distributions."*

  **The most on-target source in the pass.** Same object — a product of many learned linear maps composed
  along a graph — and the same mechanism as §4: the multiplicative ergodic theorem is Oseledets'.

**#526 is an instance of the general fix.** The ticket's question — *local hack or general fix* — has a
clear answer.

### 3.3 What it is known to cost

1. **A plasticity–stability trade, expressed as a bias toward the identity.** Every unit of `μ` spent
   pulling `K` toward `aI` erases learned structure. GCNII concedes this in its own design by making
   `β_ℓ ∝ 1/ℓ`; a constant `μ` across all seven hops would flatten the deep hops toward the identity,
   buying composed rank by giving up composed *computation*.
2. **It rescales the effective learning rate.** Learned deviation decays with time constant `1/μ`, so
   structure reinforced more slowly than `μ` is never accumulated. **With direction stability 0.95–0.99
   the collapsing direction is reinforced at nearly the maximum possible rate and will survive any `μ`
   that leaves anything else alive.** The prediction is that a leak slows the collapse without stopping
   it, unless `μ` is large enough to erase the signal too.
3. **It buys a rate, and a rate may not be enough.** Chen et al.'s wording is *"mitigates or prevents"*,
   and which obtains is parameter-family dependent. The cautionary parallel is Dong et al.'s MLP result:
   a Lipschitz constant moves the *constant* in front of a doubly-exponential rate and leaves the
   structure untouched. **A multiplicative dial on a doubly-exponential rate is not a fix.**
4. **The value of `a` is not free.** `a = 1` biases toward the identity; `a < 1` toward a contraction. The
   shrinkage literature shrinks toward the *average* eigenvalue, which would put `a` at the band's centre
   rather than its ceiling — and #451's ruling that the band's floor is `world_loop(c)` interacts with
   that choice in a way no source covers.

### 3.4 And the thing this pass must say plainly

**`K` is not a factor in the composed object.** `composed_reads` multiplies `F_out·F_inᵀ` hops and
nothing else. So even granting #526 everything it claims, **no mechanism has been stated by which a leak
on `K` moves composed rim-to-apex effective rank.** It answers P1, the apex retention collapse; #497's
failure is a different one. Any session reading #526 as an answer to composed rank is making an inference
no measurement and no source supports.

---

## 4. The effective rank of a composed product of learned linear maps

The ticket's least-covered area. It is also the one where the exact mathematics turns out to be older and
sharper than the deep-learning literature that rediscovered it.

### 4.1 Composition cannot increase rank, and the composed subspace does not depend on what the cells learn

**Feng, R., Zheng, K., Huang, Y., Zhao, D., Jordan, M. & Zha, Z.-J. (2022), "Rank Diminishing in Deep
Neural Networks", NeurIPS 35:33054–33065, [arXiv:2206.06072](https://arxiv.org/abs/2206.06072).** [ABS,
verbatim]

> *"We theoretically establish a universal monotonic decreasing property of network rank from the basic
> rules of differential and algebraic composition, and uncover rank deficiency of network blocks and deep
> function coupling."*

The property is *universal* and comes from composition itself. **No per-factor constraint can be
sufficient for a product.**

**Roth, A. & Liebig, T. (2023), "Rank Collapse Causes Over-Smoothing and Over-Correlation in Graph Neural
Networks", LoG, [arXiv:2308.16800](https://arxiv.org/abs/2308.16800).** [ABS, verbatim]

> *"we demonstrate that with increased depth, node representations become dominated by a low-dimensional
> subspace that depends on the aggregation function but not on the feature transformations. For all
> aggregation functions, the rank of the node representations collapses."*

Writing a layer as `T = W ⊗ Ã`, their Theorem 5.2 gives the relative amplification of invariant subspaces
as `|λᵢ(Ã)|/|λⱼ(Ã)|` — **a function of the aggregation operator's eigenvalues alone**, with the learned
`W` scaling every subspace equally; Theorem 6.1 bounds the rank by the algebraic multiplicity of `Ã`'s
dominant eigenvalue, for any aggregation and random feature transforms. **This is the literature's
theorem for T3's decoupling**: changing what the cells learn cannot change the composed ratio.

**Calibration, and it is important.** Roth & Liebig assume the *same* aggregation operator `Ã` at every
layer. The dome composes *distinct* learned maps per hop, so it is closer to a deep linear network with
different weights than to a repeated GCN layer, and Theorem 5.2 does **not** transfer verbatim. What
transfers is the shape of the conclusion — the composed collapse is a property of the transport structure
rather than of what the cells learn — and that shape is independently confirmed in-house by T2's
zero-supply control, where composed rank fell to 1.0845 ± 0.0023 with **no exogenous supply at all**.

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
  [doi:10.1007/BF02551235](https://doi.org/10.1007/BF02551235).** [CITE — see §6] The sharp rate is a
  power of the cosine of the **Friedrichs angle** — geometric in the number of factors.
- **Badea, C., Grivaux, S. & Müller, V. (2010), [arXiv:1006.2047](https://arxiv.org/abs/1006.2047).**
  [ABS, verbatim] *"A generalization of the cosine of the Friedrichs angle between two subspaces to a
  parameter associated to several closed subspaces of a Hilbert space is given. This parameter is used to
  analyze the rate of convergence in the von Neumann–Halperin method of cyclic alternating
  projections."* This is the **N-subspace** form a seven-hop chain needs.
- **Kopecká, E. (2019), "When products of projections diverge", JLMS,
  [arXiv:1901.01921](https://arxiv.org/abs/1901.01921).** [ABS, verbatim] *"in the cyclic case there is a
  dichotomy: the convergence is fast if and only if `L₁^⊥ + ⋯ + L_K^⊥` is closed; otherwise the
  convergence is arbitrarily slow."* **In finite dimensions every such sum is closed**, so on this dome
  the fast branch is the only branch: the collapse is **geometric in the hop count**.

**The consequence.** The composite's effective rank is, to a good approximation, the dimension of the
near-**intersection** of the seven carried subspaces. Seven generic 3–4 dimensional subspaces inside a
12-wide chart intersect in one direction or none. **Composed effective rank 1.000 is the generic answer,
and getting above 1.5 means engineering a ≥2-dimensional subspace common to every hop of every chain.**

*Calibration: the theorems are about orthogonal projections; the hops are partial isometries of differing
lane width. The mechanism transfers — principal angles between successive subspaces, geometric in the
number of factors — but no theorem quoted here applies to this object verbatim.*

### 4.3 The Lyapunov thread: separation is generic and its rate is bounded below

- **Furstenberg & Kesten (1960), *Ann. Math. Statist.* 31:457–469; Oseledets (1968), *Trans. Moscow Math.
  Soc.* 19:197–231.** [CITE] The multiplicative ergodic theorem: a product of many matrices has between
  1 and `d` distinct Lyapunov exponents with a filtration by growth rate. **The composed rim-to-apex map
  is an instance of this object.**
- **Newman (1986), *Comm. Math. Phys.* 103:121–126; Isopi & Newman (1992), *Comm. Math. Phys.*
  143:591–598, [doi:10.1007/BF02099267](https://doi.org/10.1007/BF02099267).** [CITE] Exact exponent
  distributions; the **triangle law** for products of large i.i.d. matrices. The spectrum is *spread*,
  not concentrated.
- **Hanin, B. & Jiang, T. (2025), "Global Universality of Singular Values in Products of Many Large
  Random Matrices", [arXiv:2503.07872](https://arxiv.org/abs/2503.07872).** [CITE] As dimension and
  factor count grow in any relative order, normalised squared singular values converge to the **uniform
  distribution on [0,1]**, reconciling the free-probability and multiplicative-ergodic regimes. Read onto
  the dome: `σ₁/σ₂ ~ e^{L(λ₁−λ₂)}`, so at `L ≈ 7` a per-factor gap of a few tenths already puts the ratio
  in the hundreds. **Effective rank 1.000 to three decimals is what this predicts.**
- **Atnip, J., Froyland, G., González-Tokman, C. & Quas, A. (2026), "Universal gap growth for Lyapunov
  exponents of perturbed matrix products", *J. London Math. Soc.* 113(4),
  [doi:10.1112/jlms.70548](https://doi.org/10.1112/jlms.70548).** [CITE] **The strongest form available.**
  Under additive random perturbation, a **universal lower bound on the gaps between consecutive Lyapunov
  exponents**, uniform over all choices of the original matrix sequence and requiring **no stationarity**
  — so it covers sequential, learned, drifting cocycles, which is exactly what the transport rule
  produces. **Spectral separation is generic and its rate is bounded below; it cannot be tuned away.**
  Their introduction also states this pass's per-factor point at source: singular values *"may be far
  from multiplicative"*, and bounds such as `s₁(A)s_d(B) ≤ s₁(AB) ≤ s₁(A)s₁(B)` are *"far too weak to
  control limits"*.
- **Hanin, B. & Nica, M. (2020), *Comm. Math. Phys.* 376:287–322,
  [arXiv:1812.05994](https://arxiv.org/abs/1812.05994).** [CITE] With both factor count and size large,
  `log‖Πv‖` is asymptotically Gaussian with variance growing in the **depth-to-width ratio** — spectral
  spreading in a different coordinate.

### 4.4 Is per-factor conditioning sufficient? The one exception, and why it fails here

**The general answer is no.** A *bounded* condition number per factor does not control the product: the
condition number grows like `e^{L(λ₁−λ₂)}` with the gap bounded below universally (§4.3), and
`κ(AB) ≤ κ(A)κ(B)` is the loose bound the same authors call far too weak.

**The one exception is a *degenerate* band.** If every factor has all singular values equal — `K = cQ`
with `Q` **square orthogonal** — the product is `(Πcᵢ)(ΠQᵢ)`, exactly scaled orthogonal at any depth: all
Lyapunov exponents equal, no gap to grow, Oseledets' filtration trivial. Corroborated from the
deep-learning side by **Pennington, Schoenholz & Ganguli (2017),
[arXiv:1711.04735](https://arxiv.org/abs/1711.04735)** [CITE], whose central negative is that criticality
(`χ = 1`, the *mean squared* Jacobian singular value) prevents explosion on average but **does not give
dynamical isometry** — `s_max²` still grows **linearly in depth** at `χ = 1` — and whose conclusion is
that isometry is achievable *only with orthogonal weights* and is a property of the **end-to-end**
Jacobian, not of each layer. **Saxe, McClelland & Ganguli (2014),
[arXiv:1312.6120](https://arxiv.org/abs/1312.6120)** [CITE] make the same point for deep **linear**
networks specifically — the dome's exact setting.

**And this is where the pass's headline lives: this project already implements the exception, and it does
not save it.**

| reading | value | source |
|---|---|---|
| per-map flatness `σ_min/σ_max`, median, 30k **and** 100k, seeds 0/1/2 | **0.9999995** (worst 0.99997; 1339 of 1364 maps reached, 25 structurally excluded) | `prototypes/spectral-floor-435/read.json` |
| per-map effective rank | **4.000 of `m` = 4** | [#435](https://github.com/NGL321/patchworks/issues/435) |
| composed rim-to-apex effective rank, 263 chains, 7 edges | **1.000** at 30k and 100k | [#436](https://github.com/NGL321/patchworks/issues/436), [#497](https://github.com/NGL321/patchworks/issues/497) |
| the same, flat maps at **chance** alignment (null) | **1.107** | #436 |
| holonomy channel return `\|⟨u₁,v₁⟩\|`, 260 cycles | **0.9881** floored, against 0.399 chance and a 0.457 rewired null | [#453](https://github.com/NGL321/patchworks/issues/453) |
| holonomy flatness around a cycle | **1.5e-5** floored, **8.5e-10** unfloored | #453 |

**Every factor is an isometry to six decimal places and the product is a line.** The reconciliation is
that ADR-0032's floor produces **rectangular partial isometries**, not square orthogonal matrices — `m×k`
blocks with `interior_m = 3`, `boundary_m = 4`, and masks — so each factor is an isometry *onto its own
subspace* and §4.2's product-of-projections mechanism applies in full. **The degenerate-band sufficiency
argument requires squareness, and the architecture's `k < n` commitment (ADR-0004) makes squareness
unavailable by construction.** That is a real and previously unstated interaction between ADR-0004 and
ADR-0032, and it is this pass's sharpest single finding.

**Two refinements that keep the statement honest.**

- **The ceiling is reachable in principle.** #436's own ruling: *"holonomy = I composes flat maps into a
  flat composite at effective rank `m`."* Flatness **plus trivial holonomy** *is* sufficient. What is
  insufficient is flatness alone, and the missing ingredient is a **cross-factor** quantity.
- **The failure is not misalignment**, and #497 rules this explicitly — *"Alignment is fine; there is one
  thing to align."* #453 measured alignment driven **toward** the identity, 0.9881 against a 0.399 chance
  null, at every cycle length and both horizons. The surface aligns **one** direction almost perfectly and
  sits at chance on the rest. **Haas, Gatine, Cosse & Bouraoui (2026),
  [arXiv:2602.12384](https://arxiv.org/abs/2602.12384)** [CITE — a very recent preprint, treated as such]
  is the literature's account of exactly this: spectral separation *forces* singular-vector alignment in
  matrix products, creating an approximately shared singular basis and hence decoupled singular-value
  dynamics — **separation causes alignment, which locks in the separation.** A positive feedback loop,
  which is why the reading sits at exactly 1.000 rather than drifting. Jing et al.'s Theorem 2 is the
  learned counterpart: adjacent matrices align under gradient descent. **A candidate must make more than
  one direction align consistently along a whole chain**, not improve alignment in general.

### 4.5 The same result, rediscovered in the graph literature

**Oono, K. & Suzuki, T. (2020), ICLR, [arXiv:1905.10947](https://arxiv.org/abs/1905.10947).** [ABS,
verbatim] *"when its weights satisfy the conditions determined by the spectra of the (augmented)
normalized Laplacian, its output exponentially approaches the set of signals that carry information of
the connected components and node degrees only."* Theorem 1 gives `d_ℳ(X^(l)) = O((sλ)^l)`, with `s` the
supremum of the weight matrices' maximum singular values and `λ` the largest non-unit eigenvalue modulus
of the augmented normalised Laplacian; **`sλ < 1` implies exponential collapse**, and the paper's stated
guideline is to keep `s > λ^{-1}`. *(Theorem statement at [CITE] depth — see §6.)*

**Read onto this surface, this is the most immediately checkable finding in the pass.** ADR-0015 pins
`σ_max = 0.9947`. If the transport operator's non-unit spectral radius `λ ≤ 1`, then `sλ < 1` and the
system sits **inside the collapse regime by 0.0053** — and ADR-0032's flattening moves `σ_max` down by
`√m` per hop besides, which [ADR-0022](../adr/0022-a-hop-is-an-operator-norm-along-a-learned-channel.md)
prices at **`2⁷ = 128x` across the rim-to-apex seven** (#436). **The project sits deep inside the collapse
regime, on purpose, for reasons that have nothing to do with rank.** Whether `λ` is close enough to 1 for
the 0.0053 to matter is an eigenvalue computation on the existing transport matrices — **no benchmark run
required** — and it either implicates or exonerates the band ceiling immediately.

Corroboration in Dirichlet-energy form: **Cai & Wang (2020),
[arXiv:2006.13318](https://arxiv.org/abs/2006.13318)** [CITE]. The survey that axiomatises oversmoothing
as exponential convergence of a node-similarity measure, and argues Dirichlet energy is the right
instrument while criticising mean-average-distance for failing the axioms: **Rusch, Bronstein & Mishra
(2023), [arXiv:2303.10993](https://arxiv.org/abs/2303.10993)** [CITE].

**Dong, Cordonnier & Loukas (2021), ICML,
[arXiv:2103.03404](https://arxiv.org/abs/2103.03404)** [ABS — **already read**, `docs/research/394`
§2.4(d)]: pure attention converges *doubly exponentially* to rank one, arrested by **skip connections**,
only *slowed* by MLPs. **And a directly transferable negative**: layer normalisation does not mitigate it,
because **right multiplication cannot increase the rank of a matrix**. A fortiori a scalar `σ_max`
rescale cannot raise rank — it can only redistribute. That is `body.py::project` in one sentence.

---

## 5. The candidates, ranked

Ranked by **(evidence it moves the *composed* quantity) ÷ (constraint cost)**. A candidate that only
moves local rank ranks *below* one that moves nothing, because T3 has already run that experiment.

| # | candidate | moves composed rank? | ADR cost | how it reads on the instruments |
|---|---|---|---|---|
| **1** | **Channel-heterogeneous transport** — a sum of Kronecker products, `Σᵢ(Wᵢ ⊗ Aᵢ)`, or message passing over several directed subgraphs (Roth & Liebig [arXiv:2308.16800](https://arxiv.org/abs/2308.16800) Thm 7.1; Roth et al. [arXiv:2409.11504](https://arxiv.org/abs/2409.11504)) | **Yes, provably, in their setting** — and it is the only candidate whose theorem is *about* the composed quantity and *explains* the observed decoupling | **None of the four.** Still linear (ADR-0004 safe); no term on the operator update (ADR-0008 safe); no reward; no batch statistics. **The cost is architectural**, and it is adjacent to [#319](https://github.com/NGL321/patchworks/issues/319) without being it — #319 conditions a map on the node stalk (nonlinear in the input), this diversifies the *aggregation* while every map stays linear | composed rank rises toward `min(p, m)`; **per-cell PR should be unchanged** — which makes it a very clean falsifier: composed up with local flat confirms transport as the whole story |
| **2** | **Additive identity / residual on the composed path** — `h ← (I + T)h`, GCNII's identity mapping, Dong et al.'s skip connection | **Yes — the best-evidenced family in the pass**, three literatures, one of them proving it by the multiplicative ergodic theorem on the same object | **ADR-0008 — and the reading is genuinely contested.** `(I+T)` is linear (ADR-0004 safe) and is **not a term added to `ΔK`**; it is a structural property of the composition. Whether ADR-0008 reaches it is a call this pass does not make. **If it does not, this is candidate 1** | composed rank jumps immediately — the identity path alone carries full rank. Worth reading the bypass and the learned path *separately*, or the jump may mask rather than fix |
| **3** | **Raise the band ceiling above `1/λ`** (Oono & Suzuki's weight-normalisation guideline) | Yes in their setting — it is *the* stated escape from exponential collapse | **ADR-0015** (the ceiling is exactly 1) **and ADR-0022/ADR-0032** (the `√m` per-hop price is the floor's one derivable value). Three ADRs, none of them the four constraints | `σ_max` off the 128x loss; the incoherence cap starts biting; `ρ(K)` may exceed 1, so the chain stops being a contraction — that is the whole risk. **The cheapest read in the pass: one eigendecomposition of the transport operator, no run** |
| **4** | **A degenerate band on `K`** (pin the whole spectrum, not just `σ_max`) | **Already live on transport via ADR-0032, and already falsified there** (§4.4). Untested on `K`, which is not in the composed object anyway | **None of the four** — it is a normalisation swap in the slot the band already occupies | per-cell spectra flatten; apex `ρ(K)` pinned by construction; **composed rank predicted unchanged**, on this project's own measurement |
| **5** | **#526's leak `K ← K − μ(K − aI)`** | **Not shown, and no mechanism stated** — `K` is not a factor in the composed object | **ADR-0008** | apex `ρ(K)` up, `‖ē‖` down, excitation PR up; **composed rank predicted unchanged** — the cheap falsifier. Sweep `μ` and check whether the curve is a step or a slope; a slope means slowing, not stopping |
| **6** | **A cross-chain multi-direction alignment constraint** (make ≥2 directions align route-free along a path) | Plausibly — it is the quantity §4.2 says governs the product. **No source, no proof** | **ADR-0008** if a term; **ADR-0011** locality if it reads a cycle (a cycle is incident to no cell — #453's own point). Comes *near* [#346](https://github.com/NGL321/patchworks/issues/346) and is not it, per #497 | holonomy identification departure off its 0.888 floor; #315's rig reads it directly |
| **7** | **Anti-Hebbian lateral inhibition / similarity matching** (Földiák; Rubner–Tavan; APEX; Pehlevan–Chklovskii) | **No** — buys local rank, measured decoupled | **ADR-0008.** The decay term falls out of the Lagrangian and cannot be dropped without losing the proof. Local, online and batch-free otherwise — the best of the family on the other three | excitation PR up; composed unchanged (already run) |
| **8** | **Oja subspace rule / Sanger's GHA** | **No** — same quantity | **ADR-0008 twice**: an additive term *and* the species its *Alternatives considered* refused by name. GHA is additionally **not local** | as above |
| **9** | **SoftHebb** | Not evaluated on a composed linear chain by anyone | Replaces the prediction rule; ADR-0008's refused species; no prediction error. Local, online, batch-free, single-view — the only method in §2 that is | — |
| **10** | **VICReg / Barlow Twins / W-MSE / DirectPred / Forward-Forward / blockwise SSL** | No — regularise a cell's own representation | **Batch statistics** (definitionally) **+ no second view**; the contrastive members add a task loss | not readable — the local instrument reads 1.00 everywhere (ledger row 1) |
| **11** | **PairNorm / ContraNorm / BatchNorm** | BatchNorm *proves* a lower rank bound (Daneshmand et al. [arXiv:2003.01652](https://arxiv.org/abs/2003.01652)) — so the *category* "a normalisation that provably raises rank" is real | **Batch statistics** for all three; ContraNorm adds a softmax over the node-similarity matrix, so **ADR-0004** too | — |
| **12** | **DropEdge / stochastic perturbation of transport** | Only delays it, **and the theory says it goes the wrong way** | None of the four — but Atnip et al. (2026) prove random perturbation forces a *larger* Lyapunov gap, so it should **increase** separation. Also changes the graph, and *no edge is ever removed* is a construction invariant | **likely counterproductive; ranked last for that reason, not for cost** |
| — | **Anything reward-shaped** | — | **Excluded at the door** by #5's bar. None of the above was admitted on those grounds | — |

**What the ranking actually says.** Two candidates are clean against all four constraints — 1 and 3 (and
4, which is clean and already falsified). **Candidate 1 is the only clean one with a theorem about the
composed quantity, and candidate 2 is the only one with a proof on the same object.** Everything below
line 5 either buys local rank, which is measured decoupled, or breaks a constraint outright.

**The binding constraint remains ADR-0008 for the best-evidenced remedy** — but the honest addition this
pass makes is that **ADR-0008 may not actually reach a residual in transport**, and that reading is worth
settling deliberately rather than by default, because it decides whether the literature's strongest
result is available or banned.

---

## 6. What could not be reached, stated plainly

Ranked by how much it weakens what is above.

**Costliest.**

- **Kayalar & Weinert (1988).** Springer redirects to an identity provider; the AMS survey PDF returned
  403. The record (Math. Control Signals Systems 1:43–59, doi:10.1007/BF02551235) was confirmed from two
  independent retrievals and the *shape* of the result from secondary statements. **§4.2 depends only on
  the rate being geometric in the number of factors**, which Badea et al.'s abstract supports
  independently at [ABS] depth.
- **Oono & Suzuki's theorem statement.** Both the arXiv and OpenReview PDFs failed extraction (binary
  stream; browser check). The abstract was read verbatim; the `sλ < 1` form and the `s > λ^{-1}` guideline
  are carried at [CITE] depth from a second reader's extraction and from GCNII's citation of the same
  `s^K` rate. **Candidate 3 should not be priced without reading the theorem** — and the cheap
  eigenvalue read it suggests does not depend on the constant, only on the direction.
- **Atnip et al. (2026) and Hanin & Jiang (2025)** are [CITE]: records and abstracts confirmed, bodies not
  read. §4.3's load-bearing claim — that the Lyapunov gap is bounded below universally and without a
  stationarity assumption — is the strongest statement in the pass and rests on an abstract-level reading.
  A session that wanted to *rely* on it rather than be oriented by it should read the theorem.
- **The classical Hebbian papers' own equations.** Oja (1989), Sanger (1989) and Földiák (1990) have their
  records confirmed and their mechanisms from authoritative secondary statements; **no update equation was
  read at source**. §1.3's ADR-0008 cost turns on one structural claim — the decorrelating term is
  subtracted inside the update — corroborated for GHA by the Gram–Schmidt wording. The
  Pehlevan–Chklovskii rules in §1.4 are transcribed from an arXiv rendering and should be checked against
  the published *IEEE SPM* version before being quoted load-bearingly.

**Moderate.**

- **Miller & MacKay (1994)** at [ABS]. The quoted sentence is from the MIT Press landing page and is the
  pass's single most load-bearing diagnostic citation; the paper body was not read, so the *conditions*
  on "otherwise linear learning rules" were not checked against this rule.
- **Roth & Liebig's theorem numbering and statements** are [CITE] from a second reader; the abstract is
  verbatim. §4.1 already records the disanalogy (they assume one shared aggregation operator; the dome
  composes distinct maps), which is the thing that would need checking before candidate 1 is priced.
- **Chen et al. (2025)** at [ABS]. The *"mitigates or prevents"* distinction is load-bearing for §3.3's
  "it buys a rate" and was **not** resolved against the theorem.
- **Haas et al. (arXiv:2602.12384)** is a February 2026 preprint, not peer-reviewed, and is used only to
  name a mechanism that #453 independently measured.
- **The per-map and composed readings are paired across two rigs** on the same surface (`main`'s full
  dome) but from different tickets; the pairing is this pass's inference and both numbers are published
  rather than netted.

**Unverified and therefore not cited above**, recorded so nobody inherits them: APPNP's arXiv identifier
(Klicpera, Bojchevski & Günnemann, ICLR 2019 — content confirmed via GCNII and the Rusch et al. survey);
DropEdge's identifier (Rong et al., ICLR 2020); Oja (1989)'s exact pagination.

**Not searched at all**, so the ground is not re-covered blindly: the reservoir-computing literature on
the echo-state property and memory capacity, which is the other field that constrains `σ_max` of a
recurrent linear operator and might state the retention/rank trade in its own terms; consensus and
products of sub-stochastic matrices in control theory, which is oversmoothing's older sibling; and
matrix-completion theory, where a rank floor is a hard constraint with known geometry.

---

## 7. Sources

| Source | Depth | Access |
| --- | --- | --- |
| **Miller & MacKay (1994)**, *The Role of Constraints in Hebbian Learning*, Neural Computation 6(1):100–126, [doi:10.1162/neco.1994.6.1.100](https://doi.org/10.1162/neco.1994.6.1.100) | [ABS] | MIT Press landing page; quoted verbatim |
| Oja (1982), *Simplified neuron model as a principal component analyzer*, J. Math. Biol. 15:267–273 | [ABS+] | landing page + secondary; primary PDF located, not extracted |
| Oja (1989), *Neural Networks, Principal Components, and Subspaces*, Int. J. Neural Systems 1:61–68 | [ABS+] | dblp + Semantic Scholar |
| Sanger (1989), *Optimal unsupervised learning in a single-layer linear feedforward neural network*, Neural Networks 2(6):459–473 | [ABS+] | landing page; Gram–Schmidt wording corroborated |
| Földiák (1990), *Forming sparse representations by local anti-Hebbian learning*, Biol. Cybern. 64:165–170, [doi:10.1007/BF02331346](https://doi.org/10.1007/BF02331346) | [ABS+] | Springer + Scholarpedia |
| Rubner–Tavan (1989); APEX (Kung & Diamantaras 1990) | [CITE] | nothing rests on them |
| Pehlevan, Sengupta & Chklovskii (2018), Neural Computation 30(1):84–124, [arXiv:1703.07914](https://arxiv.org/abs/1703.07914); Pehlevan & Chklovskii (2019), [arXiv:1908.01867](https://arxiv.org/abs/1908.01867) | [ABS] | equations from ar5iv — see §6 |
| Journé, Garcia Rodriguez, Guo & Moraitis (2023), *Hebbian Deep Learning Without Feedback*, ICLR, [arXiv:2209.11883](https://arxiv.org/abs/2209.11883); theory [arXiv:2107.05747](https://arxiv.org/abs/2107.05747) | [ABS] | landing page + repo |
| Krotov & Hopfield (2019), *Unsupervised learning by competing hidden units*, PNAS 116(16):7723–7731 | [CITE] | — |
| Roy & Vetterli (2007), *The effective rank*, EUSIPCO | [CITE] | instrument definition only |
| Bardes, Ponce & LeCun (2022), *VICReg*, [arXiv:2105.04906](https://arxiv.org/abs/2105.04906) | [ABS] | **already read**, `docs/research/394` §2.4(a) |
| Zbontar, Jing, Misra, LeCun & Deny (2021), *Barlow Twins*, [arXiv:2103.03230](https://arxiv.org/abs/2103.03230) | [ABS] | landing page |
| Ermolov, Siarohin, Sangineto & Sebe (2021), *W-MSE*, ICML, [arXiv:2007.06346](https://arxiv.org/abs/2007.06346) | [ABS] | landing page |
| Tian, Chen & Ganguli (2021), *DirectPred*, ICML, [arXiv:2102.06810](https://arxiv.org/abs/2102.06810) | [ABS] | landing page |
| Jing, Vincent, LeCun & Tian (2022), *Understanding Dimensional Collapse in Contrastive SSL*, [arXiv:2110.09348](https://arxiv.org/abs/2110.09348) | [ABS] | **already read**, `docs/research/394`; Thms 2–3 at [CITE] |
| Hua et al. (2021), [arXiv:2105.00470](https://arxiv.org/abs/2105.00470) | [ABS] | **already read**, `docs/research/394` |
| Siddiqui et al. (2023), *Blockwise SSL at scale*, [arXiv:2302.01647](https://arxiv.org/abs/2302.01647); VICRegL [arXiv:2210.01571](https://arxiv.org/abs/2210.01571); Hinton (2022), *Forward-Forward*, [arXiv:2212.13345](https://arxiv.org/abs/2212.13345) | [ABS] | landing pages |
| Papyan, Han & Donoho (2020), *Neural collapse*, PNAS 117(40):24652–24663 | [CITE] | **not used** — see §2.4 |
| Galanti, Siegel, Gupte & Poggio, [arXiv:2206.05794](https://arxiv.org/abs/2206.05794) | [ABS] | landing page + PMLR 280 |
| *Weight decay induces low-rank attention layers*, [arXiv:2410.23819](https://arxiv.org/abs/2410.23819) | [CITE] | — |
| Ledoit & Wolf (2004), J. Multivariate Analysis 88(2):365–411 | [CITE] | — |
| Chen M., Wei, Huang, Ding & Li (2020), *GCNII*, ICML, [arXiv:2007.02133](https://arxiv.org/abs/2007.02133) | [ABS] | landing page |
| Bansal, Chen & Wang (2018), [arXiv:1810.09102](https://arxiv.org/abs/1810.09102) | [CITE] | — |
| Chen Z., Lin, Chen S., Polyanskiy & Rigollet (2025), [arXiv:2501.00762](https://arxiv.org/abs/2501.00762) | [ABS] | abstract verbatim |
| Feng, Zheng, Huang, Zhao, Jordan & Zha (2022), *Rank Diminishing*, NeurIPS 35:33054–33065, [arXiv:2206.06072](https://arxiv.org/abs/2206.06072) | [ABS] | abstract verbatim |
| **Roth & Liebig (2023)**, *Rank Collapse Causes Over-Smoothing and Over-Correlation in GNNs*, LoG, [arXiv:2308.16800](https://arxiv.org/abs/2308.16800) | [ABS] | abstract verbatim; theorems [CITE] |
| Roth, Bause, Kriege & Liebig (2024), LoG, [arXiv:2409.11504](https://arxiv.org/abs/2409.11504) | [CITE] | — |
| Oono & Suzuki (2020), ICLR, [arXiv:1905.10947](https://arxiv.org/abs/1905.10947) | [ABS] | abstract verbatim; theorem [CITE] — see §6 |
| Cai & Wang (2020), [arXiv:2006.13318](https://arxiv.org/abs/2006.13318); Rusch, Bronstein & Mishra (2023), [arXiv:2303.10993](https://arxiv.org/abs/2303.10993) | [CITE] | — |
| Dong, Cordonnier & Loukas (2021), ICML, [arXiv:2103.03404](https://arxiv.org/abs/2103.03404) | [ABS] | **already read**, `docs/research/394` §2.4(d) |
| Zhao & Akoglu (2020), *PairNorm*, [arXiv:1909.12223](https://arxiv.org/abs/1909.12223); Guo et al. (2023), *ContraNorm*, [arXiv:2303.06562](https://arxiv.org/abs/2303.06562) | [CITE] | — |
| Daneshmand, Kohler, Bach, Hofmann & Lucchi (2020), NeurIPS, [arXiv:2003.01652](https://arxiv.org/abs/2003.01652) | [ABS] | **already read**, `docs/research/394` §2.4(e) |
| Halperin (1962), *The product of projection operators*, Acta Sci. Math. (Szeged) 23:96–99 | [CITE] | via two expository sources |
| Kayalar & Weinert (1988), Math. Control Signals Systems 1:43–59, [doi:10.1007/BF02551235](https://doi.org/10.1007/BF02551235) | [CITE] | **contested access** — see §6 |
| Badea, Grivaux & Müller (2010), [arXiv:1006.2047](https://arxiv.org/abs/1006.2047) | [ABS] | abstract verbatim |
| Kopecká (2019), JLMS, [arXiv:1901.01921](https://arxiv.org/abs/1901.01921) | [ABS] | abstract verbatim |
| Feshchenko (2019), [arXiv:1908.00531](https://arxiv.org/abs/1908.00531) | [ABS] | used only for the `‖P_n···P_1 − P_0‖` object |
| Furstenberg & Kesten (1960), Ann. Math. Statist. 31:457–469; Oseledets (1968), Trans. Moscow Math. Soc. 19:197–231; Crisanti, Paladin & Vulpiani (1993), Springer | [CITE] | foundations |
| Newman (1986), CMP 103:121–126; Isopi & Newman (1992), CMP 143:591–598, [doi:10.1007/BF02099267](https://doi.org/10.1007/BF02099267) | [CITE] | — |
| **Atnip, Froyland, González-Tokman & Quas (2026)**, JLMS 113(4), [doi:10.1112/jlms.70548](https://doi.org/10.1112/jlms.70548) | [CITE] | record + abstract; body not read — see §6 |
| Hanin & Jiang (2025), [arXiv:2503.07872](https://arxiv.org/abs/2503.07872); Hanin & Nica (2020), CMP 376:287–322, [arXiv:1812.05994](https://arxiv.org/abs/1812.05994) | [CITE] | — |
| Pennington, Schoenholz & Ganguli (2017), [arXiv:1711.04735](https://arxiv.org/abs/1711.04735); (2018), [arXiv:1802.09979](https://arxiv.org/abs/1802.09979); Saxe, McClelland & Ganguli (2014), [arXiv:1312.6120](https://arxiv.org/abs/1312.6120) | [CITE] | — |
| Haas, Gatine, Cosse & Bouraoui (2026), [arXiv:2602.12384](https://arxiv.org/abs/2602.12384) | [CITE] | **preprint, not peer-reviewed** |

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
