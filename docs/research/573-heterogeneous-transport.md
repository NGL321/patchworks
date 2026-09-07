# B24 (research): learned per-edge translation — what keeps long compositions from collapsing? (patchworks#573)

Part of map [#532](https://github.com/NGL321/patchworks/issues/532). Opened by
[#565](https://github.com/NGL321/patchworks/issues/565)'s resolution. Findings feed B27, where they
get weighed against the measurements. **This pass does not rule on the architecture.**

Standing instruction for the batch: *"look at the literature to see what supports the things we've
described, without constraining on those topics we have not described."* Accordingly the sections
below lead with what contradicts or re-poses the framing, not with what confirms it.

## Reading-depth key

#148's key, used throughout.

- **[FULL]** — paper body read (HTML/ar5iv extracted, or PDF text extracted).
- **[ABS]** — authoritative abstract / landing page only.
- **[CITE]** — citation confirmed to exist, text not reached.

**Preprint caution.** Several of the most on-point items are 2026 arXiv preprints (2602.12384,
2605.11178, 2607.25387, 2608.02558, 2608.16180). They are cited as preprints, not as
peer-reviewed results, and each is marked. Where a summarising fetch reported theorem numbers, those
numbers are the weakest part of the citation and are flagged inline.

---

## Headline verdict, stated plainly

**The literature already contains the result the ticket names under "what would change our minds",
and it is not a sheaf result — it is a random-matrix result. Long products of generic linear maps
collapse to one direction generically, exponentially in path length, at a rate set by the width of
the space they act on. Nothing in the sheaf construction exempts it. Seven-hop collapse at small
stalk width is the expected behaviour of *any* per-edge learned linear transport that has not been
explicitly constrained to be isometric.**

Three consequences follow, and they run against the framing in different directions:

1. **The collapse is structural, not a tuning failure** — but "structural" here means *a property of
   composing linear maps*, not *a property of heterogeneity*. Heterogeneity is not the culprit. §2.1.
2. **The only cure with a proof behind it is isometry** — orthogonal restriction maps, or per-edge
   norm control. Every softer cure (skips, gating, normalisation) is contested in the literature,
   and at least one 2025 result argues skips do *not* prevent collapse. §2.2–2.4.
3. **The object the ticket suspects exists in §3 has a name, two names in fact, and one of them is
   thirty years older than the other.** Robinson's *consistency radius / consistency filtration* and
   Abramsky–Brandenburger's *contextuality* both measure exactly "locally consistent, globally not".
   §3.

And a fourth finding that bears directly on the reported measurement that "the learned carried
subspaces are statistically indistinguishable from random": **that is a known, published outcome for
sheaf GNNs, and at least one preprint argues the learned maps are largely replaceable by random or
shared maps without loss.** §1.4.

---

## §1. Cellular sheaves as learnable objects

### 1.1 The base result, and what it actually says

**Bodnar, Di Giovanni, Chamberlain, Liò, Bronstein (2022), "Neural Sheaf Diffusion: A Topological
Perspective on Heterophily and Oversmoothing in GNNs", NeurIPS 2022.** arXiv:2202.04579. [FULL, via
ar5iv]

This is the closest formal match to the architecture: stalks on nodes, learned linear restriction
maps per node–edge incidence, diffusion by the sheaf Laplacian.

What it *proves* (theorem numbering as extracted from the ar5iv body; treat the numbers as
approximate, the statements as reliable):

- **Prop. 4 / Lemma 6 (harmonic space of an O(d)-bundle).** For a discrete O(d)-bundle over a
  connected graph, a harmonic section `x` satisfies `x_v ∈ ker(P^γ_{v→v} − I)` for every cycle `γ`
  at `v`; and `dim H⁰ ≤ d`, **with equality iff transport is path-independent around every cycle.**
  This is the sharpest statement in the whole pass for our purposes and it cuts both ways. The
  ceiling on how many directions survive is `d`, the stalk dimension — *not* the number of features,
  not the number of cells, not the lane width. And you only reach that ceiling when the composed
  transport is *holonomy-free*: every loop returns the identity.
- **Prop. 11 (the `d = 1` death sentence).** With one-dimensional stalks, `dim ker Δ_F ≤ 1`, and the
  harmonic space "cannot linearly separate the classes" for `C ≥ 3` classes, *for any initial
  conditions*. One-dimensional stalks collapse to one direction by construction.
- **Prop. 12 / 13 (what buys you directions).** Diagonal-invertible sheaves need `d ≥ C` to separate
  `C` classes. Orthogonal sheaves separate `C ≤ 2d` classes — **strictly more dimension-efficient
  than diagonal.**
- **Thm. 15 (Dirichlet energy decay).** For symmetric-positive restriction maps with (Leaky)ReLU,
  `E_F(Y) ≤ λ* ‖W₁‖² ‖W₂ᵀ‖² E_F(X)`. Energy decays exponentially to the kernel whenever the weight
  product is below 1. **So the symmetric-positive class oversmooths just like a GCN.**
- **Prop. 17 (the escape hatch).** *Non-symmetric* sheaves outside those classes can *increase*
  Dirichlet energy with an arbitrarily small linear transform, giving "greater control than GCNs
  over their asymptotic behaviour."

**What it says vs. what we might hope it says.** It says the *harmonic space* — the fixed point of
diffusion — can be up to `d`-dimensional, and gives sufficient conditions (orthogonality, holonomy
freedom) for that. It does **not** say that a learned sheaf will find such conditions, and it does
**not** say anything about the spectrum of a *composed* restriction operator along a path. The
ticket's instrument (spectrum of the composed operator) and this literature's instrument (kernel
dimension of the Laplacian / Dirichlet energy) are different objects. Bridging them is our work, not
theirs.

### 1.2 Orthogonality is the one condition that recurs everywhere

Across §1 and §2 the same condition keeps appearing under different names: **orthogonal restriction
maps (Bodnar et al., Prop. 13), O(d)-bundles / connection Laplacians (Barbero et al.), dynamical
isometry (Pennington et al.), holonomy-freedom (Bodnar et al., Lemma 6).** These are the same
constraint viewed from four fields. An orthogonal map has every singular value equal to 1; a product
of orthogonal maps is orthogonal; the composed operator therefore preserves *every* direction
regardless of path length. **This is the only unconditional anti-collapse mechanism found in this
pass.** Its cost is that it removes the map's ability to be a *lane* — an orthogonal map cannot
discard the directions the two cells do not share, because it discards nothing.

That tension is, as far as this pass can tell, unresolved in the literature and is the sharpest
open question the pass returns. See §5.

### 1.3 A newer result that names the failure mode as *degeneracy*, and blames symmetry

**Dönmez, Mosig, Fritsche, Koch (2026), "Oversmoothing as Representation Degeneracy in Neural Sheaf
Diffusion", arXiv:2605.11178 — preprint, May 2026.** [ABS]

Reframes oversmoothing in NSD representation-theoretically: "learned sheaves may collapse toward
low-complexity summands whose global sections fail to preserve discriminative information."
Direct-sum decompositions of the incidence-quiver representation induce decompositions of the
harmonic space, so degeneration to a simpler summand is an algebraic event, not a numerical one.

The concrete claim of interest: **a structural obstruction exists in *symmetric* architectures where
node and edge stalk dimensions are equal (`d_v = d_e`)** — stability constraints push the trivial
all-object summand onto a stability wall. Their remedy is **non-uniform stalk dimensions** plus
moment-map-inspired regularisers biasing restriction maps toward balanced geometries.

**Negative result, reported as they report it:** improvements appear only "in selected rectangular
settings". This is a narrow, dataset-dependent win, not a general cure. But the diagnostic half —
*equal node and edge stalk dimension is itself an obstruction* — is a cheap thing for us to check
against the implementation, and it is the kind of structural claim that a fixed-at-construction mask
would be very likely to violate silently.

### 1.4 The result that bears hardest on our own measurement

**Yi Liu (2026), "Learned, Relied Upon, or Necessary? Separating Checkpoint Dependence from
Task-Level Value in Sheaf GNNs", arXiv:2607.25387 — preprint, July 2026, single author,
unreviewed.** [ABS]

Tests whether the learned restriction maps in NSD / DNSD / DSNN are *necessary* or merely
*checkpoint-organising*. Method: retrain under controls that break the edge→map assignment (resample
assignment continually) or replace per-edge maps with a single shared, parameter-matched map.

Findings:

- On **four of five** DNSD benchmarks, the controls **recover full performance.** The learned per-edge
  geometry is replaceable.
- Only Roman-Empire retains an advantage for genuinely learned maps: **+0.0675** over the
  continually-resampled-assignment control and **+0.0391** over the parameter-matched shared-map
  control, across ten official splits.
- Framing sentence: "a learned map can govern a fitted computation without constituting indispensable
  edge geometry."

**Why this matters here.** The ticket reports that our learned carried subspaces are *statistically
indistinguishable from random frames of the same shape*. This paper says: in the closest published
analogue, that is the **normal** outcome on four of five benchmarks, and the correct inference is not
"our training is broken" but "per-edge learned geometry is usually not where the task value is."
That is a genuine challenge to the central bet, and it is the second most important thing this pass
returns.

Caveat, stated plainly: single-author unreviewed preprint. The direction of the finding is
corroborated by §1.5, which is a separate group.

### 1.5 The benchmarking pass, and where the variance actually lives

**Fiorini, Coppola, Liò (2026), "Benchmarking Sheaf Neural Networks for Inductive Tasks",
arXiv:2608.02558 — preprint, Aug 2026.** [FULL, HTML]

Twelve datasets, inductive setting. Reported findings:

- **"SNNs do not reach the strongest published baselines on any dataset considered."** Closest on
  PATTERN, MNIST, MalNet-Tiny, CLUSTER (~1.4% off best); worst on ZINC, PascalVOC-SP, ogbg-molpcba,
  ogbg-code2 (~30% off). The gap spans two orders of magnitude across datasets.
- **Variance decomposition:** the surrounding GNN+ architecture explains **26.7%** of performance
  variation; the entire 18-cell sheaf design space explains **21.1%**.
- Within the sheaf design space, **restriction-map parameterisation dominates (12.2% variance)**
  while **the diffusion mechanism is negligible (0.2%)**.

**What this says:** the *family* the restriction maps are drawn from (diagonal / orthogonal /
general / low-rank) is the design choice that matters; how you diffuse along them barely matters.
That is a useful pointer — it says spend effort on the map class, not on the update rule. It also
says, less comfortably, that the sheaf machinery as a whole has not yet earned its keep against
plain GNNs on inductive tasks.

### 1.6 The line that gave up on learning the maps

**Barbero, Bodnar, Sáez de Ocáriz Borde, Bronstein, Veličković, Liò (2022), "Sheaf Neural Networks
with Connection Laplacians", ICML 2022 Workshop on Topology, Algebra, and Geometry in ML (PMLR
v196).** [ABS]

Explicitly motivated by the claim that "learning a sheaf could lead to overfitting and significant
computational overhead." Their alternative computes **orthogonal** restriction maps from the
manifold assumption — maps that optimally align the tangent spaces of neighbouring points — with no
end-to-end learning, and reports competitive results at lower cost.

**Read this as a negative result about learning.** A group that had just published the
learn-the-sheaf paper wrote a follow-up whose premise is that not learning it works about as well
and costs less. Combined with §1.4, the pattern across the sheaf-GNN line is: *the learned maps are
expensive, and it is hard to show they are load-bearing.*

### 1.7 Discovering *what two cells share* — the mask problem

The ticket notes that our mask is fixed at construction, identical across a cell's edges, and never
reopens. The literature does contain machinery for inferring the maps and the shared subspace from
data, though not, as far as this pass found, for *reopening a mask during training*:

- **Hansen & Ghrist (2019), "Learning Sheaf Laplacians from Smooth Signals", ICASSP 2019, pp.
  5446–5450.** [CITE — confirmed to exist and to address exactly this; body not reached.] Learns a
  sheaf from a collection of highly consistent (smooth) vertex signals.
- **Di Nino, Barbarossa, Di Lorenzo (2025), "Learning Sheaf Laplacian Optimizing Restriction Maps",
  arXiv:2501.19207.** [ABS] Infers **both** the topology and the restriction maps from node
  observations by minimising total variation, with the per-edge variation minimised by optimising
  that edge's maps. Closed-form rather than SDP. Reports that **cross-correlation between nodes and
  the dimensionality difference between nodes' data** are the two factors that shape the recovered
  graph. **The abstract does not state identifiability guarantees**; whether the sheaf is recoverable
  and up to what ambiguity was not established in this pass. Flagged as open.
- **"Learning the Structure of Connection Graphs", arXiv:2510.11245.** [CITE — surfaced in search,
  not read.] Adjacent; unverified.

**The honest summary of §1.7:** the objective these papers optimise is *smoothness / low total
variation*, i.e. they learn the sheaf that makes the observed data as consistent as possible. That
is a different objective from ours — and it is worth noticing that an objective which descends
disagreement will, if unconstrained, prefer the sheaf that maps everything to a single agreeing
direction. The degenerate optimum of "minimise disagreement" *is* collapse. None of these papers
addresses that directly, which is itself a finding.

---

## §2. Rank collapse in composed maps — the cures, and their cost

### 2.1 The generic-collapse result the ticket asked for

This is the section that most directly answers "what would change our minds."

**Newman, C. M. (1986), "The distribution of Lyapunov exponents: exact results for random matrices",
Communications in Mathematical Physics 103(1):121–126.** [CITE — existence and content of the
digamma formula confirmed through multiple independent secondary sources; original text not reached.]

For products of i.i.d. matrices from the real Ginibre ensemble (`n × n`, Gaussian entries), **all `n`
Lyapunov exponents are computed exactly and are given by digamma values**, of the form
`λ_i = ½[ψ((n − i + 1)/2) − ψ(n/2)]` (the shift/normalisation convention varies by source; the
*differences* between exponents are convention-independent and are what matters here).

Two consequences, and they are the whole point:

1. **The exponents are distinct.** A long product of generic random matrices does not preserve a
   subspace; the singular values separate exponentially and the product aligns to **one** direction.
   This is Furstenberg–Kesten / Oseledets applied to the Gaussian case, with the constants filled in.
2. **The gap between the top two exponents shrinks with width.** Since `ψ(x + ½) − ψ(x) ≈ 1/(2x)` for
   large `x`, the top gap scales as **≈ 1/(2n)** in the matrix dimension `n`.

Corroboration for the scaling, and its transfer to networks:

**Haas et al. (2026), "Why Deep Jacobian Spectra Separate: Depth-Induced Scaling and ...",
arXiv:2602.12384 — preprint, Feb 2026.** [FULL, PDF; theorem numbers reported by the extractor as
"Theorems 4–5" and should be re-checked before quoting.] States the spectral gap between the largest
and second-largest Lyapunov exponents scales as **~1/(2n)** in width `n`, and that singular values
separate at a rate governed by the exponential of that gap — so convergence to rank one accelerates
with **increasing depth `L`** and with **decreasing width `n`**.

**The operational statement for us.** For a composition of `k` hops of generic (random-like)
`d × d` transport, the ratio of the second to the first singular value of the composed operator
decays roughly as

> `σ₂/σ₁ ~ exp(−k · Δλ)` with `Δλ ≈ 1/(2d)`,

so the characteristic path length at which the composition becomes numerically rank one is on the
order of **`k ≈ 2d` hops**. At small stalk width this is a handful of hops. **Seven-hop collapse is
not an anomaly to be tuned away; at small `d` with maps statistically indistinguishable from random,
it is the predicted behaviour.** This is precisely the "collapse rate a function of path length and
representation width" result the ticket said would be the most useful thing to return, and it exists.

**What it does not say, and this matters.** It is a statement about *random* maps. It says nothing
about maps that have been *constrained* — an orthogonal product has all exponents zero and never
collapses. So the result does not condemn the architecture; it condemns the *unconstrained* version
of it, and says the constraint that saves it is isometry. It also does not say the collapse is a
function of *heterogeneity*; heterogeneity plays no role in it whatsoever.

### 2.2 The classical diagnosis, and the standard cure

**Dong, Cordonnier, Loukas (2021), "Attention is Not All You Need: Pure Attention Loses Rank Doubly
Exponentially with Depth", ICML 2021 (oral), PMLR v139.** [ABS + confirmed repo] Pure self-attention
converges **doubly exponentially** in depth to a rank-1 matrix; **skip connections and MLPs
counteract the decay.** Self-attention has a strong inductive bias toward token uniformity.

**Noci, Anagnostidis, Biggio, Orvieto, Singh, Lucchi (2022), "Signal Propagation in Transformers:
Theoretical Perspectives and the Role of Rank Collapse", NeurIPS 2022.** [ABS] Adds the mechanism by
which collapse hurts: **rank collapse makes the gradients of the queries and keys vanish at
initialisation.** Cure: **depth-dependent scaling of the residual branches.** Cost: a scaling
schedule that must be matched to depth.

### 2.3 The cure that is contested — and this is the important part of §2

**Alman & Song (2025), "Only Large Weights (And Not Skip Connections) Can Prevent the Perils of Rank
Collapse", arXiv:2505.16284.** [ABS] Directly contradicts the received story in §2.2. They define
**layer collapse** — the whole network being well-approximated by a single layer — and prove that
**even with skip connections, small weights still produce collapse.** What prevents it is **weight
magnitude**, not the architectural skip.

The cost they identify is sharp and is worth carrying into any design conversation: **large weights
force quadratic-time attention.** Small-weight models admit almost-linear algorithms precisely
*because* they are representationally collapsed. Expressiveness and cheapness are traded directly
against each other.

**Read this against §2.1 and the picture is coherent:** what stops a long product from aligning to
one direction is control of the *singular values* — either by making them all equal (orthogonality,
§1.2 / §2.4) or by making them large enough that the nonlinearity's re-expansion dominates the
contraction. There is no free architectural trick.

### 2.4 Isometry as a cure, and its known limits

**Pennington, Schoenholz, Ganguli (2017), "Resurrecting the sigmoid in deep learning through
dynamical isometry: theory and practice", NeurIPS 2017.** arXiv:1711.04735. [ABS] Uses free
probability to compute the entire singular value distribution of a deep network's input–output
Jacobian. **Dynamical isometry** — all Jacobian singular values concentrated near 1 — permits very
deep signal propagation.

Two results with direct bearing:

- **Orthogonal weight initialisation achieves dynamical isometry; Gaussian initialisation does not.**
- **ReLU networks are incapable of dynamical isometry** at any initialisation; sigmoidal networks can
  achieve it, but only with orthogonal weights.

**The cost.** This is the same trade as §1.2, arrived at independently: isometry preserves every
direction because it discards nothing. Any per-edge map that genuinely *selects a shared subspace*
is not an isometry on the full stalk, and the moment it is not, §2.1's clock starts.

Lineage note: **Saxe, McClelland, Ganguli (2014), "Exact solutions to the nonlinear dynamics of
learning in deep linear networks", ICLR 2014** is the origin of the orthogonal-initialisation result
in the linear case. [CITE]

### 2.5 The GNN cure literature, and its cost profile

Surveyed via **"Oversmoothing Alleviation in Graph Neural Networks: A Survey and Unified View"
(ATNPA), arXiv:2405.01663** [ABS] and the ETH SAM survey **"A Survey on Oversmoothing in Graph
Neural Networks" (2023-17)** [CITE]. Interventions that restore or preserve rank/energy:

| Cure | Mechanism | Cost as reported |
|---|---|---|
| **PairNorm** (Zhao & Akoglu, ICLR 2020) | holds total pairwise node distance constant per layer | normalisation is *global* — it fixes the aggregate, not any particular direction; can hurt shallow performance |
| **EGNN** (Zhou et al., "Dirichlet Energy Constrained Learning for Deep GNNs", NeurIPS 2021) | constrains layer weights so Dirichlet energy stays within a band; **orthogonal weight init** to set the initial energy | constrains the weights, i.e. buys depth by giving up weight freedom |
| **G²/Gradient Gating** (Rusch, Mishra, Bronstein, Bronstein, "Gradient Gating for Deep Multi-Rate Learning on Graphs", ICLR 2023) | gates each feature by the local graph gradient; features that have converged stop updating | reported to hold Dirichlet energy near constant to **1000 layers**; cost is a per-feature multi-rate gate and the fact that "stop updating" freezes rather than restores |

**Pattern across all three:** every one of them is an *energy-preservation* device. **None of them
is shown to restore rank that has already been lost.** They prevent, they do not cure. This pass
found no intervention in the GNN literature that recovers a direction once the composed operator has
dropped it — which is consistent with the linear-algebraic fact that it cannot be done without
adding information from somewhere else.

### 2.6 A caution about our own instrument — which cuts *in our favour*

**Zhang, Deidda, Higham, Tudisco (2026), "Are We Measuring Oversmoothing in Graph Neural Networks
Correctly?", ICLR 2026.** arXiv:2502.04591. [ABS]

Argues that similarity- and energy-based metrics (Dirichlet energy and friends) "fail to reliably
capture oversmoothing in realistic scenarios" — they only become meaningful for very deep networks,
whereas performance degrades around 10 layers. They propose instead **numerical / effective rank of
the feature representations**, and report that **rank drops track performance degradation even where
energy metrics are flat.** They also **prove that the numerical rank of feature representations
collapses to one for a broad family of GNN architectures.**

**Two things for us.** First, this is an endorsement of the instrument the project already uses:
rank is the *better* measure, not the crude one. Second — and this is a second independent "generic
collapse" result alongside §2.1 — **rank collapse to one is proved for a broad architecture family,
not observed for one model.**

But note the ticket's own caveat stands and is not answered by this paper either: rank of the
*operator* still says nothing about whether signal actually travels the retained directions. Zhang
et al. measure the rank of the **feature representations** (the state), not of the composed operator.
**That difference is a live methodological gap between our measurement and theirs**, and B27 should
know it: the literature's rank instrument touches the state; ours does not.

---

## §3. Regional rather than global consistency — the object has a name

The ticket says: *"We suspect the object exists and we do not know its name."* It exists. It has two
independent names from two literatures, and a third piece of machinery for computing it.

### 3.1 Consistency radius and the consistency filtration — the closest match

**Robinson, M. (2020), "Assignments to sheaves of pseudometric spaces", Compositionality 2(2).**
arXiv:1805.08927. [ABS]

Definitions, in the paper's own terms:

- **Assignment**: a choice of local section over each open set, **"without regard to how these local
  sections are related to one another."** (This is exactly our situation: each cell has its own
  schema, chosen without reference to whether it agrees with its neighbours.)
- **Consistency radius**: quantifies the **agreement between overlapping local sections**. Proved to
  be a **continuous** map.
- **Consistency filtration**: thresholding the consistency radius yields **a filtration of open
  covers** — "a nested set of covers in a structure-preserving way", and the construction is proved
  to be a **functor**. Robustness under perturbation is established, which is what makes it usable on
  noisy data.

**This is the object.** The consistency filtration answers precisely *"over how large a region does
this signal remain consistently expressible, and at what tolerance"* — a family of regions indexed by
tolerance, not a single global yes/no. It never asks whether the whole system agrees. It is exactly
the instrument a "reach, not fidelity" objective wants, and it is *not* the sheaf Laplacian.

Companion, applied: **Robinson, M. (2017), "Sheaves are the canonical data structure for sensor
integration", Information Fusion 36:208–224.** [CITE] — the sensor-fusion framing in which
consistency radius is the working quantity.

Applied, worked example with heterogeneous sources: **Joslyn, Charles, DePerno, Gould, Nowak,
Praggastis, Purvine, Robinson, Strules, Whitney (2020), "A Sheaf Theoretical Approach to Uncertainty
Quantification of Heterogeneous Geolocation Information", Sensors 20(12):3418.** [ABS] — consistency
radius used to fuse genuinely heterogeneous, disagreeing sensors. Note the venue is a real
peer-reviewed journal and the application is heterogeneity-first, which is unusual and useful.

### 3.2 Maximal consistent regions

**Praggastis, B. (2016), "Maximal Sections of Sheaves of Data over an Abstract Simplicial Complex",
arXiv:1612.00397.** [ABS — abstract confirms existence-and-uniqueness proof; the theorem statement,
algorithm, and complexity were **not** reached in this pass and are flagged as unverified.]

Claim: for any vertex assignment there is a **unique set of maximal consistent subcomplexes**, and
the paper proves existence and uniqueness of these maximal sections.

**Why this is the right shape for us.** "Which are the largest regions over which this signal is
coherent" is a well-posed question with a unique answer. That is a regional coherence measure with a
uniqueness theorem behind it, and it is directly implementable as a diagnostic: partition the graph
by where a given signal stays expressible, rather than asking for a global section that we have
explicitly said we do not want.

**Unverified:** algorithm and complexity. Before building on it, read the body.

### 3.3 The oldest and sharpest version — contextuality

**Abramsky, S. & Brandenburger, A. (2011), "The sheaf-theoretic structure of non-locality and
contextuality", New Journal of Physics 13:113036.** arXiv:1102.0264. [ABS]

The central theorem, in their words: **contextuality corresponds exactly to obstructions to the
existence of global sections.** A family of local sections that agree on every overlap, yet admit no
global section, is *the definition* of a contextual model. They give a **linear-algebraic method for
computing these obstructions** and a **strict hierarchy of strengths** (Bell < Hardy < GHZ).

**Why this belongs in the ticket.** The architecture's stated position — *"it is correct and
necessary that the cells modelling gait have no coherent view of the colour of a car"* — is, in this
vocabulary, the claim that **the system should be contextual by design.** That is not a hand-wave;
it is a formally defined property with a cohomological obstruction attached to it and a measure of
degree. If the project wants a number for "how much regional coherence without global coherence do
we have", the contextuality literature has been computing exactly that number since 2011.

Follow-up worth reading: **"Towards a complete cohomology invariant for non-locality and
contextuality", arXiv:1807.04203.** [CITE, not read] — the known caveat is that the cohomological
invariant of the 2011 paper is *sufficient but not complete* (there are "false negatives"), and this
line tries to close that gap. Flagged because anyone using the invariant as a metric needs to know it
is one-sided.

### 3.4 Local vs global agreement inside the sheaf-network framing itself

**Hansen, J. & Ghrist, R. (2021), "Opinion Dynamics on Discourse Sheaves", SIAM J. Applied
Mathematics.** arXiv:2005.12798. [FULL, ar5iv]

This is the closest thing in the literature to the architecture's *objective* — agents descending
disagreement over per-edge translations that need not be the identity:

- Disagreement on edge `e = u ~ v` is `(δx)_e = F_{v◁e} x_v − F_{u◁e} x_u`; global sections
  `H⁰(G, F)` are exactly the states where **"all expressions of opinions are in harmony"**, `δx = 0`.
- **Theorem 5:** diffusion under `L_F = δᵀδ` converges to the orthogonal projection onto `H⁰(G, F)`.
  **The consensus states are the attractors, and there is nothing else.**
- The paper says plainly: **"If this sheaf has no nontrivial global sections, the only stable opinions
  will be everywhere zero: an uninteresting solution."**
- **Theorem 7 (harmonic extension):** with some agents *stubborn* (not updating), the free agents
  converge to the harmonic extension — `L_F x = 0` on free vertices subject to boundary conditions.
- They explicitly distinguish **local sections `H⁰(A, F)`** — agreement within a subgraph `A` — from
  **relative cohomology `H⁰(G, A, F)`**, global sections vanishing on `A`, which measures how
  independent `A` is of the rest of the network.

**Three things fall out of this, and one of them is uncomfortable.**

1. `H⁰(A, F)` is a *third* name for regional consistency, inside the sheaf-network framing itself.
   Cheapest to adopt of the three, because we already have the sheaf.
2. **Harmonic extension (Thm. 7) is a real mechanism for reach without global agreement**: pin some
   cells, let the rest relax, and the boundary's influence propagates as far as the sheaf permits.
   That is a formal version of "drop a rock in a lake."
3. **The uncomfortable one.** Theorem 5 says pure disagreement-descent has *only* the global-section
   space as its attractor. **An architecture whose objective is "descend disagreement" and which has
   no nontrivial global sections converges to zero.** One whose global sections are one-dimensional
   converges to one direction. This is not rank collapse from composition — it is rank collapse from
   *the objective*, and it is a different failure mode from §2.1 with the same symptom. The two are
   separable by measurement (one is a property of the composed operator at initialisation, the other
   only appears at convergence) and B27 should separate them.

Adjacent, unread: **"Selective Adaptation of Beliefs and Communication on Cellular Sheaves",
arXiv:2601.22431.** [CITE]

---

## §4. Imposed hierarchy against emergent sparsity

The finding here is the cleanest positive one in the pass, and it says the bet is reasonable but that
the mechanism is not sparsity as such.

### 4.1 Hierarchy emerges — from a connection cost, not from sparsity

**Mengistu, Huizinga, Mouret, Clune (2016), "The Evolutionary Origins of Hierarchy", PLoS
Computational Biology 12(6):e1004829.** [ABS, peer-reviewed]

The result, and note the negative half is the important half:

- **Networks without a connection cost do not evolve to be hierarchical *even when the task itself is
  hierarchical*.** Task structure alone is not enough. This is a clean negative result and it is the
  single most transferable finding in §4.
- **With a connection cost, networks evolve to be both modular and hierarchical.**
- Conclusion as stated: the same force — the cost of connections — promotes **both** hierarchy and
  modularity, and both drive performance and adaptability.

Predecessor: **Clune, Mouret, Lipson (2013), "The evolutionary origins of modularity", Proc. R. Soc.
B 280:20122863**, arXiv:1207.2743 [CITE] — the modularity half of the same result.

**What this says vs. what we hope it says.** It says hierarchy *can* be discovered rather than
imposed, and names the pressure that discovers it: **an explicit penalty on connection cost**, in an
evolutionary search. It does **not** say gradient-descent training discovers hierarchy, and it does
**not** say generic sparsification does — the pressure has to be *on the wiring*, and it has to be
present. "Sparsify and hope" is not what this result supports. "Penalise connection cost and
hierarchy appears" is.

### 4.2 Pruning and modularity — supportive, with a caveat

- **Filan, Hod, Wild, Critch, Russell (2020), "Pruned Neural Networks Are Surprisingly Modular",
  arXiv:2003.04881**, and the successor **"Clusterability in Neural Networks", arXiv:2103.03386.**
  [ABS] Pruned networks are more clusterable than random networks of the same sparsity; **learning by
  gradient descent *plus* pruning selects for clusterability.**
- **"Neural Sculpting: Uncovering hierarchically modular task structure in neural networks through
  pruning and network analysis", NeurIPS 2023 (OpenReview 1jhmWkZGy6).** [ABS] Uses unit *and* edge
  pruning to reveal **previously unknown hierarchical modularity** underlying a task.

**Caveat, stated because it is the honest reading:** these papers establish *clusterability* — a
graph-partition property of the weight matrix. Whether clusterability is *functional* modularity is a
weaker and more contested claim, and this pass did not find a result establishing that a cluster
found by pruning corresponds to a semantically separable sub-function in general. Treat these as
suggestive, not as proof that sparsification finds the structure you wanted.

### 4.3 Predictive coding does not need imposed hierarchy to train

**Salvatori, Pinchetti, Millidge, Song, Bao, Bogacz, Lukasiewicz (2022), "Learning on Arbitrary
Graph Topologies via Predictive Coding", NeurIPS 2022.** arXiv:2201.13180. [ABS, peer-reviewed]

**PC graphs** perform inference and learning on **arbitrary** topologies, including **cyclic and
backward connections that backpropagation cannot support**. The same network performs different
tasks by stimulating different neurons — query it with partial images, images with labels, images
without. The stated motivation is that the neocortex's connectivity is **heterarchical**, not
hierarchical, and that this may be fundamental to its effectiveness.

**What it says:** the *training machinery* does not require a hierarchy. Layered structure is not a
prerequisite for predictive coding to work on a graph. **What it does not say:** that useful
structure *emerges* on an arbitrary topology. This paper demonstrates trainability, not emergence.
The emergence claim is §4.1's, and §4.1 says you need a wiring-cost pressure to get it.

**The synthesis for §4:** the bet that sparsification would find semi-hierarchical structure is
supported *conditionally*. The condition is an explicit connection-cost term. The dome, as an
imposed constraint, is the shortcut that skips the search; the literature says the search works but
needs a pressure the architecture may not currently have.

---

## §5. What this says about the architecture

Stated as observations for B27, not as rulings.

1. **The seven-hop collapse has an explanation that does not implicate heterogeneity.** §2.1: generic
   linear maps composed `k` times align to one direction at rate `exp(−k/(2d))`. If the learned maps
   are indistinguishable from random (which is our own measurement), the collapse is *predicted*, and
   the number to compare against is `2d` hops, not seven. **This reframes the problem from "why did
   our maps degenerate" to "why would they not."**
2. **Two distinct collapse mechanisms are in play and our current instrument cannot separate them.**
   Composition collapse (§2.1, a property of the operator at any time) and objective collapse (§3.4
   Thm. 5, `H⁰` is the only attractor of disagreement-descent, appearing at convergence). Same
   symptom, different cures. Separating them is a measurement design task.
3. **The only proven anti-collapse mechanism is isometry, and it is in tension with the lane.** §1.2,
   §2.4. An orthogonal restriction map preserves every direction because it discards none; a lane is
   defined by what it discards. This pass found nobody who has resolved that tension. If there is one
   thing to take to B27 as a *design* question rather than a measurement question, it is this.
4. **"Minimise disagreement" has collapse as its global optimum.** §1.7, §3.4. Every sheaf-learning
   paper found optimises smoothness/total variation; the sheaf that minimises disagreement perfectly
   is the one that maps everything to a single shared direction. If nothing in the objective rewards
   *retained dimension*, the objective is actively pushing toward the observed failure.
5. **The mask being fixed and edge-independent is worse than it looks in one specific way.** §1.5
   reports restriction-map *parameterisation* is the dominant design variable (12.2% of variance) and
   the diffusion mechanism is negligible (0.2%). We have frozen the variable that matters and are
   iterating on the one that does not.
6. **`d_v = d_e` may itself be an obstruction.** §1.3. Cheap to check against the implementation.
7. **§3 gives us instruments we do not have.** Consistency radius and the consistency filtration
   (Robinson), maximal consistent subcomplexes (Praggastis), local sections `H⁰(A, F)` and relative
   cohomology (Hansen–Ghrist), and the contextuality obstruction (Abramsky–Brandenburger). All four
   measure regional coherence without requiring global agreement. **`H⁰(A, F)` is the cheapest to
   adopt because the sheaf already exists.**
8. **The framing may be wrong in one place, and it is the central bet.** §1.4 and §1.6: in the
   closest published analogue, learned per-edge geometry is *usually replaceable* by random or shared
   maps with no loss, and the group that introduced learned sheaves published a follow-up arguing you
   should not learn them. Our finding that the carried subspaces look random is, in that light, the
   expected result rather than a bug — and the question becomes whether per-edge *learned* translation
   is carrying anything at all, separately from whether it collapses.

---

## §6. What we did not find / open

Stated as plainly as the findings, because the ticket asked for it.

- **No result on collapse in a *heterogeneous* composition specifically.** Everything in §2 is about
  homogeneous stacks (same layer type repeated) or i.i.d. random products. Whether heterogeneity
  across cells changes the Lyapunov gap — helps, hurts, or is neutral — was not found. This is a real
  gap and it is exactly where the architecture's central bet lives. It may be a genuinely open
  question, or it may be that the answer is "heterogeneity is irrelevant to the spectrum", which
  §2.1 mildly suggests.
- **No mechanism found for *reopening* a per-edge participation mask during training.** §1.7 has
  machinery for inferring restriction maps from data, but all of it is offline structure-learning
  from observed signals, not online mask discovery inside a training loop. If this exists, this pass
  did not find it.
- **No intervention shown to *restore* rank once lost.** §2.5. Every GNN cure is preventive
  (energy-preserving). This is likely a theorem rather than an oversight, but no paper stating it as
  such was found.
- **No resolution of the isometry/lane tension.** §5.3. Nobody found who both selects a proper
  subspace per edge *and* proves non-collapse over long paths.
- **Identifiability of a learned sheaf is unestablished here.** Di Nino et al. (§1.7) infer topology
  and restriction maps but this pass did not reach a statement of what is recoverable and up to what
  gauge ambiguity. Given that a sheaf has an obvious gauge freedom (act on each stalk by an
  invertible map, adjust incident restrictions), "the learned maps look random" and "the learned maps
  are determined up to a random gauge" are different claims and this pass could not separate them.
  **This is a concrete follow-up and it might dissolve finding §5.8.**
- **Praggastis (§3.2): theorem statement, algorithm, and complexity not read.** Verify before
  building on it.
- **Haas et al. (§2.1): theorem numbers as reported by an extraction pass, not verified against the
  body.** The `1/(2n)` scaling is independently corroborated by the Newman digamma formula, so the
  *result* is safe; the *citation precision* is not.
- **Newman (1986) original text not reached** — the digamma formula is confirmed through multiple
  independent secondary sources but not from CMP directly. The exact normalisation convention should
  be checked before any quantitative prediction is made from it.
- **Five 2026 arXiv preprints are load-bearing in §1 and §2.1** (2602.12384, 2605.11178, 2607.25387,
  2608.02558, 2608.16180). None is confirmed peer-reviewed. §1.4 in particular — the "learned maps
  are replaceable" result — is a single-author unreviewed preprint and is the one most worth
  independently reproducing before it changes any decision.
- **Not searched, and probably should be:** the index-theoretic criterion paper (Dong, Peng, Li,
  Feng, Xia, arXiv:2608.16180, Aug 2026) [ABS] proposes a *relative* comparison criterion for when
  one sheaf configuration genuinely preserves discriminative information better than another,
  observing that "certain sheaf configurations inflate the harmonic space dimension while their
  harmonic sections remain entirely constant, without enriching discriminative capacity." **That last
  clause is a direct warning about our instrument** — a nominally high-rank composed operator whose
  retained directions carry nothing is a configuration this criterion is designed to catch. It was
  reached only at abstract depth and deserves a full read.

---

## Sources

- [Bodnar et al. 2022, Neural Sheaf Diffusion (NeurIPS 2022)](https://arxiv.org/abs/2202.04579) · [ar5iv](https://ar5iv.labs.arxiv.org/html/2202.04579)
- [Dönmez et al. 2026, Oversmoothing as Representation Degeneracy in NSD (preprint)](https://arxiv.org/abs/2605.11178)
- [Liu 2026, Learned, Relied Upon, or Necessary? (preprint)](https://arxiv.org/abs/2607.25387)
- [Fiorini, Coppola, Liò 2026, Benchmarking Sheaf NNs for Inductive Tasks (preprint)](https://arxiv.org/html/2608.02558)
- [Dong et al. 2026, Demystifying Oversmoothing in Sheaf NNs: An Index-Theoretic Criterion (preprint)](https://arxiv.org/abs/2608.16180)
- [Barbero et al. 2022, Sheaf Neural Networks with Connection Laplacians (PMLR v196)](https://proceedings.mlr.press/v196/barbero22a/barbero22a.pdf)
- [Di Nino, Barbarossa, Di Lorenzo 2025, Learning Sheaf Laplacian Optimizing Restriction Maps](https://arxiv.org/abs/2501.19207)
- [Dong, Cordonnier, Loukas 2021, Pure Attention Loses Rank Doubly Exponentially (ICML)](https://arxiv.org/abs/2103.03404)
- [Noci et al. 2022, Signal Propagation in Transformers (NeurIPS)](https://proceedings.neurips.cc/paper_files/paper/2022/hash/ae0cba715b60c4052359b3d52a2cff7f-Abstract-Conference.html)
- [Alman & Song 2025, Only Large Weights (And Not Skip Connections) Can Prevent Rank Collapse](https://arxiv.org/html/2505.16284)
- [Pennington, Schoenholz, Ganguli 2017, Dynamical Isometry (NeurIPS)](https://arxiv.org/abs/1711.04735)
- [Zhang, Deidda, Higham, Tudisco 2026, Are We Measuring Oversmoothing Correctly? (ICLR 2026)](https://arxiv.org/pdf/2502.04591)
- [Haas et al. 2026, Why Deep Jacobian Spectra Separate (preprint)](https://arxiv.org/pdf/2602.12384)
- [Akemann, Kieburg, Wei 2014, Universal distribution of Lyapunov exponents for products of Ginibre matrices](https://arxiv.org/abs/1406.0803)
- [Forrester 2013, Lyapunov exponents for products of complex Gaussian random matrices (J. Stat. Phys.)](https://arxiv.org/abs/1206.2001)
- [Newman 1986, The distribution of Lyapunov exponents: exact results for random matrices (CMP)](https://projecteuclid.org/euclid.cmp/1104114627)
- [ATNPA 2024, Oversmoothing Alleviation in GNNs: A Survey and Unified View](https://arxiv.org/html/2405.01663)
- [Rusch et al. 2023, Gradient Gating (ICLR)](https://arxiv.org/pdf/2210.00513)
- [Zhou et al. 2021, Dirichlet Energy Constrained Learning for Deep GNNs (NeurIPS)](https://proceedings.neurips.cc/paper/2021/file/b6417f112bd27848533e54885b66c288-Paper.pdf)
- [Robinson 2020, Assignments to sheaves of pseudometric spaces (Compositionality)](https://arxiv.org/abs/1805.08927)
- [Robinson 2017, Sheaves are the canonical data structure for sensor integration (Information Fusion)](https://www.sciencedirect.com/science/article/abs/pii/S156625351630207X)
- [Joslyn et al. 2020, Sheaf Theoretical Approach to UQ of Heterogeneous Geolocation Information (Sensors)](https://www.mdpi.com/1424-8220/20/12/3418)
- [Praggastis 2016, Maximal Sections of Sheaves of Data over an Abstract Simplicial Complex](https://arxiv.org/abs/1612.00397)
- [Abramsky & Brandenburger 2011, The sheaf-theoretic structure of non-locality and contextuality (NJP)](https://arxiv.org/abs/1102.0264)
- [Hansen & Ghrist 2021, Opinion Dynamics on Discourse Sheaves (SIAM J. Appl. Math.)](https://arxiv.org/abs/2005.12798)
- [Hansen & Ghrist 2019, Learning Sheaf Laplacians from Smooth Signals (ICASSP)](https://semanticscholar.org/paper/282d3539aa17cdfa0da67b30cbb8418b2f496bba)
- [Mengistu, Huizinga, Mouret, Clune 2016, The Evolutionary Origins of Hierarchy (PLoS Comput Biol)](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1004829)
- [Clune, Mouret, Lipson 2013, The evolutionary origins of modularity (Proc R Soc B)](https://arxiv.org/pdf/1207.2743)
- [Filan et al. 2020, Pruned Neural Networks Are Surprisingly Modular](https://arxiv.org/pdf/2003.04881)
- [Clusterability in Neural Networks 2021](https://arxiv.org/pdf/2103.03386)
- [Neural Sculpting (NeurIPS 2023)](https://openreview.net/forum?id=1jhmWkZGy6)
- [Salvatori et al. 2022, Learning on Arbitrary Graph Topologies via Predictive Coding (NeurIPS)](https://arxiv.org/abs/2201.13180)
