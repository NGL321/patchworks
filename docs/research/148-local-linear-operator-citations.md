# Citation pass: local linear operators glued across a decomposition (patchworks#148)

Checks the reasoning behind the Koopman conversion recorded on
[#127](https://github.com/NGL321/patchworks/issues/127) — each **cell** replacing its frozen random
nonlinear `step` with a learned linear operator `K` on its own **chart**, in its own basis, glued to
neighbours by learned **restriction maps** into **edge stalks**, with nothing converging to a global
operator. Citations validate after the fact per [#1](https://github.com/NGL321/patchworks/issues/1)'s
citation-sequencing rule; this document does not revise closed design, it flags where a source
threatens a claim already made. Vocabulary follows `CONTEXT.md`: Patchworks' side in its own terms
(cell, chart, piece, node stalk, edge stalk, restriction map, scale gauge), the prior art's in its
own field's. Where a source could not be reached, that is stated rather than papered over.

The pass had a second job, named in the ticket: the reasoning had been checked **only against people
who agree with it**. SINDy, the invariant-subspaces theorem, deep Koopman, HAVOK and mrDMD all
originate with one group. That orbit was escaped — of the sources below, three are Brunton/Kutz and
none of those three carries a load-bearing critique.

## Reading-depth key

Every source is tagged. The repo's convention requires the distinction and this pass leans on it
heavily, because several of the sharpest claims rest on full-text reads and several of the weakest
do not.

- **[FULL]** — paper body read (PDF or HTML extracted).
- **[ABS]** — authoritative abstract / landing page only.
- **[CITE]** — citation confirmed to exist, text not reached.
- **[UNREACHED]** — existence not confirmed.
- **[BK]** — traces to the Brunton / Kutz orbit.

## Headline verdict, stated plainly

**The conversion survives. The spectral interpretation of it does not, and the stage-1 tickets are
largely unaffected while the long-horizon spectral tickets take real damage.**

That split is the whole finding, and it is a consequence of how
[#127](https://github.com/NGL321/patchworks/issues/127) already chose to argue. The map takes the
conversion on **cost-gate** grounds — `body` is a term in the transmission budget and today it is a
random draw nobody can set; under the conversion it becomes `rho(K)`, settable and bounded.
**Nothing found in this pass touches that argument.** One source independently strengthens it:
Arroyo et al. (2025) find that a state-space reformulation of a GNN alleviates over-smoothing and
over-squashing *"at no extra trainable parameter cost"*, and replacing a frozen random MLP with a
learned linear `K` is exactly that reformulation. The conversion is, in the over-squashing
literature's own terms, the recommended remedy.

What does not survive is the reading of `K`'s eigenvalues as a **spectrum** in the Koopman sense.
Five independent groups, none of them Brunton/Kutz, converge on this:

- **A continuous encoder cannot separate multiple attractors.** Liu, Ozay & Sontag (Automatica 2025)
  prove an obstruction that needs only *continuity* of the encoder, not a linear read-out — closing
  the loophole deep Koopman uses to evade the 2016 invariant-subspace theorem. A frozen random ReLU
  `encode` is continuous. On any piece with more than one ω-limit set, the chart provably cannot be
  injective, **and the learning process pushes it toward merging them**. This is the sharpest single
  finding in the pass.
- **Every published lift is larger than its state; the chart is smaller.** 36, 60 and 100 basis
  functions for *2-dimensional* systems; `N = 330` monomials for a 2-D end-effector. `k = 12` under
  `n = 32` is a compression. **This is a vocabulary collision more than an error**, and
  [#145](https://github.com/NGL321/patchworks/issues/145) already named it `k_piece` vs `k_lift`
  before this pass ran — see §4, which is the most important section here for what to do next.
- **No algorithm gets the spectrum in one limit.** Colbrook, Mezić & Stepanenko prove no algorithm —
  deterministic, or randomised above 50% success — converges to the approximate point spectrum for
  broad well-behaved system classes, at any data volume. One-limit convergence is guaranteed only
  given a known modulus of continuity. Streaming local learning is one limit and the sandbox supplies
  no such modulus.
- **Spectral pollution is the default, not a corner case**, and it persists as the dictionary grows.
- **On contact, the operator is not guaranteed to exist at all** — and the remedy in the literature is
  to change the *physics model*, not the learner. This is a checkable precondition on the sandbox's
  contact solver (§6) and it is the cheapest open item in this document.

**The novelty claim needs rewriting before it is made, not after review.** Pre-registered claim 1 —
that the distributed-Koopman literature does not already do the sheaf version — is **false**, as the
ticket predicted it most likely would be. Per-agent encoders and decoders, a block-sparse operator
supported exactly on graph edges, parallel per-agent learning and no global operator are all in
print (§1). Per the repo's stated position this is a good outcome: *useful prior work located*. What
survives as unclaimed is narrower and more defensible than what was being claimed.

Two findings are cheap and actionable regardless of how the spectral question resolves: the motor
boundary should be **bilinear**, not linear (§7, three independent groups), and **effective resistance
is literally the same object** in Hansen & Ghrist's spectral sheaf theory and in the over-squashing
bound (§10) — which turns [#120](https://github.com/NGL321/patchworks/issues/120)'s measured ~921x per
hop from an observation into a computable diagnosis on the actual dome.

### What this pass did not check

This document checks the *operator* reasoning. It does not re-run
[#31](https://github.com/NGL321/patchworks/issues/31)'s over-squashing diagnosis or
[#32](https://github.com/NGL321/patchworks/issues/32)'s dimensioning work, both of which stand; §4 and
§10 extend them rather than revisiting them. §10 in particular covers the **remedies** literature,
which 031 named but did not investigate.

---

## 1. The nearest prior art: per-subdomain operators without a global one

**Verdict: the distinction holds against the mainstream consensus line and fails against a second,
less-cited line. Pre-registered claim 1 is false.**

### 1.1 The consensus line — the distinction is real here

Every paper in the mainstream "distributed Koopman" line converges to **one** operator, and the
agreement is the entire point:

- Azarbahram, Liu & Incremona, *"Distributed Koopman operator learning from sequential
  observations"*, **European Journal of Control 89** (2026), art. 101497,
  doi `10.1016/j.ejcon.2026.101497` **[ABS]**. Verbatim: *"Each agent estimates a local Koopman
  approximation based on lifted data and collaborates over a communication graph to reach exponential
  consensus on a consistent distributed approximation."*
- Azarbahram, Liu & Incremona, *"Distributed Koopman Operator Learning for Perception and Safe
  Navigation"*, arXiv:2511.22368 **[ABS]**: *"A consensus-based distributed Koopman learning algorithm
  enables multiple computational agents or sensing units to collaboratively estimate the Koopman
  operator without centralized data aggregation."* — definite article; one operator.
- Hao, Lu, Upadhyay & Mou, arXiv:2412.07212 **[ABS]**: *"agents achieve consensus on a global dynamics
  model without sharing their private training trajectories."* Note the motivation is **privacy and
  data locality**, not genuinely different local dynamics — every agent models the *same* system from
  a different slice of trajectory. That is a categorically different problem from a dome where every
  cell owns a different piece.
- Nandanoori, Sinha & Yeung, arXiv:2106.15678 **[ABS]**: *"a phase space stitching result is derived
  that yields the global Koopman operator."* Locality here is in **state space** (invariant
  subspaces), not in space. Worth citing to keep the repo's use of "stitching" from being confused
  with this one.

**Against this line the Patchworks claim is correct.** It is not, however, the whole literature.

### 1.2 Mukherjee et al. — the design's structure, minus the vocabulary

Mukherjee, Nandanoori, Guan, Agarwal, Sinha, Kundu, Pal, Wu, Vrabie & Choudhury (PNNL + Case Western,
**no Brunton/Kutz link**), *"Learning Distributed Geometric Koopman Operator for Sparse Networked
Dynamical Systems"*, **Learning on Graphs Conference (LoG) 2022**, PMLR 198. **[FULL]**

Verified against extracted full text, this is the proposed architecture minus the sheaf vocabulary:

- **Per-agent encoder and decoder** — verbatim, *"both encoder and decoder functions can be
  represented for the ith agent as φᵢ(·), ψᵢ(·)"*. That is a per-cell chart.
- **A block-sparse operator supported exactly on graph edges** — the lifted dynamics split into
  *"neighbor and non-neighbor agents"*, with the block vanishing *"if agent i is not a neighbor of
  agent j"*. A block-sparse operator whose off-diagonal blocks are supported on graph edges **is** a
  sheaf morphism structure; the off-diagonal block is the restriction map in all but name.
- **No global operator** — verbatim, *"the proposed distributed computation of the geometric Koopman
  operator is beneficial for sparse NDS, whereas for the fully connected systems this approach
  coincides with the centralized one."*
- **Parallel local learning** — *"computational advantages can be obtained by incorporating parallel
  learning of each agent."*

### 1.3 Li et al. — parallel Koopman subsystems, confirmed from the body

Li, Bo, Zhang, Qin & Yin, *"Data-driven parallel Koopman subsystem modeling and distributed moving
horizon state estimation for large-scale nonlinear processes"*, arXiv:2404.06746 (eess.SY). **[FULL —
body reached during this pass; the prior partial check had abstract only.]**

The body settles what the abstract only implied, and it settles it against the ticket's framing:

- **Per-subsystem lifting, user-specified and fixed** — e.g. `ϕⁱ = [Tᵢ, CAᵢ, ∛Tᵢ, ∛CAᵢ, eᵀⁱ, eᶜᴬⁱ]ᵀ`.
  Each subsystem `i` has **its own lifted dimension** `zᵢ ∈ ℝⁿᶻⁱ` and its own lifting functions.
- **The subsystem model** is `zᵢ(k+1) = Aᵢᵢzᵢ(k) + Σⱼ∈𝕀ᵢ Aᵢⱼzⱼ(k) + Bᵢũᵢ(k)`, with neighbouring
  lifted states *"treated as known inputs"*.
- **No consensus, averaging, parameter-sharing or global-operator step anywhere.** `m` separate
  subsystem models identified in parallel by EDMD.

So "per-subdomain Koopman operators in their own bases, coupled rather than reconciled" is **already
done**, in process control, on chemical and agro-hydrological plants.

### 1.4 What actually survives as unclaimed

Setting the two lines side by side, the honest residue is:

| Component | Status |
|---|---|
| Per-cell operator in its own basis | **Claimed** — Li et al., Mukherjee et al. |
| No global operator, coupling over graph edges | **Claimed** — Mukherjee et al. |
| Learned restriction maps | **Claimed** — Neural Sheaf Diffusion (Bodnar et al., NeurIPS 2022) |
| Cellular-sheaf formalism with edge stalks and two maps per edge | **Unclaimed in the Koopman setting** |
| Koopman-in-each-stalk | **Unclaimed** (§2) |
| Streaming / online, no global backprop | **Unclaimed** |
| Refusal of a shared encoder | **Unclaimed** — Mukherjee et al. still train theirs jointly through a GNN |
| Magnitude gauge-fixing on the coupling maps | **Unclaimed** |

Note what changed. `A_ij` in Li et al. *is* a learned linear map between two differing bases, so even
the change-of-basis role is not new. What is new is that Patchworks makes the shared space
(the **edge stalk**) an object in its own right that both cells map *into*, rather than mapping one
cell's coordinates directly into the other's — plus the gauge, plus bidirectional reconciliation
rather than one-directional prediction. That is a real difference and a much narrower claim than
"the sheaf version does not exist."

**Peitz et al. supply the counter-question the design must answer.** Peitz, Harder, Nüske, Philipp,
Schaller & Worthmann, *"Equivariance and partial observations in Koopman operator theory for partial
differential equations"*, **J. Computational Dynamics** (2024), doi `10.3934/jcd.2024035` **[ABS]** —
*"a local Koopman matrix is considered that can be applied uniformly across the domain, generating a
global matrix with identical blocks."* They exploit translation equivariance to make every patch
share **one** matrix. That is the exact opposite choice on the load-bearing axis, and it poses the
question plainly: *what does a per-cell `K` buy that one shared `K` plus learned restriction maps
would not?* The dome has no symmetry to tie cells with, which is the beginning of an answer but not
the whole of one.

**Vocabulary warning.** "Localized DMD" in the literature means localized **in time** (Li, Liu & Yang,
arXiv:2503.13093, segments the *temporal* domain) **[ABS]**. An arXiv metadata search for Koopman +
"subdomain" returns **zero** abstracts. Writing "localized Koopman" for a spatial decomposition will
be misparsed; prefer "per-cell".

---

## 2. Sheaf-theoretic Koopman: the gap is real, not a search artefact

**Verdict: confirmed gap, with a stated caveat.** The ticket instructed that the earlier null result
be assumed a search failure. It was not.

- arXiv metadata, `all:"sheaf" AND all:"Koopman"`: **one** hit, an inter-satellite QKD paper where the
  words co-occur incidentally.
- arXiv metadata, `all:"sheaf" AND all:"dynamic mode decomposition"`: **zero**.
- `abs:"cellular sheaf" AND abs:"dynamics"`: 7 hits, all opinion-dynamics / multi-agent / knowledge-graph
  / sheaf-diffusion flavoured. None operator-theoretic.

**Caveat, stated plainly:** these are title+abstract searches. A sheaf-Koopman construction buried in
a paper body would not surface, and no full-text search across arXiv was available.

What *does* exist, and should be cited rather than re-derived:

- **Robinson**, *"Sheaf and duality methods for analyzing multi-model systems"*, arXiv:1604.04647
  **[ABS]** — the charter for the whole frame: *"Sheaves are mathematical objects that manage the
  combination of bits of local information into a consistent whole… complex models can be assembled
  from smaller, easier-to-construct models"*, spanning *"continuous dynamical systems, partial
  differential equations, probabilistic graphical models"*. The single most important citation here.
- **Robinson, Szulczewski & Thorson**, arXiv:2511.04603 (2025) **[ABS]** — "sheaves of dynamical
  systems" done concretely. Per-cell dynamics there are structural-equation models, **not** operators.
- **Schultz, Spivak & Vasilakopoulou**, *"Dynamical Systems and Sheaves"*, arXiv:1609.08086,
  *Applied Categorical Structures* **[ABS]** — the well-posedness machinery for composing interconnected
  machines, with *"sheaf theory, which flexibly captures the crucial notion of time"*. The unit edge
  delay is exactly what makes their totality/determinism conditions bite; cite for whether the glued
  thing is well-defined at all.
- **Hansen & Ghrist**, *"Toward a Spectral Theory of Cellular Sheaves"*, arXiv:1808.01513,
  *J. Applied and Computational Topology* (2019) **[ABS]** — includes *"eigenvalue interlacing,
  sparsification, effective resistance, and sheaf approximation."* See §10: that effective resistance
  is the same object as the over-squashing one.
- **Hansen & Ghrist**, *"Opinion Dynamics on Discourse Sheaves"*, arXiv:2005.12798 **[ABS]** — dynamics
  *on* a sheaf, but the dynamics *are* the sheaf Laplacian's diffusion, not per-node autonomous
  operators glued by it. The **communications** half of a discourse sheaf is a closer analogue to an
  edge stalk "carrying belief only" than anything in the Koopman literature, and may be the better
  vocabulary source.
- **Bodnar, Di Giovanni, Chamberlain, Liò & Bronstein**, *"Neural Sheaf Diffusion"*, arXiv:2202.04579,
  NeurIPS 2022 **[ABS]** — *"we study how sheaves can be learned from data."* **This is the direct prior
  art for learned restriction maps. Patchworks is not first; say so plainly.**

Two follow-ups matter operationally:

- **Dönmez, Mosig, Fritsche & Koch**, arXiv:2605.11178 **[ABS]** identify *"a structural obstruction in
  equal-stalk architectures: when `d_v = d_e`, admissibility for learnable stability parameters forces
  the trivial all-object summand onto a stability wall. Non-uniform stalk dimensions remove this
  obstruction."* Patchworks already has `n = 32` against `m = 4/8`, so `d_v ≠ d_e` and the obstruction
  does not apply. **A free result validating an existing choice**, and it makes the differing
  interior/boundary `m` more defensible than it looked.
- **Bourgerie, Girdzijauskas & Fodor**, arXiv:2605.19021 **[ABS]**: *"as depth increases, the
  disagreement signal of the sheaf Laplacian vanishes, limiting the contribution of deeper layers."*
  A dome of diameter ~9 is deep by that standard. Vanishing disagreement is a live risk for the sheaf
  structure itself, independent of the operator question, and belongs on the risk register.

**Positioning, stated for reuse:** cellular sheaves with learned restriction maps are established;
sheaves of dynamical systems are established; **operator-in-each-stalk is not**. The conjunction is
the contribution, exactly as the README anticipates. Do not claim the sheaf machinery and do not
claim learned restriction maps.

---

## 3. The encoder obstruction — the sharpest finding in the pass

**Liu, Ozay & Sontag**, *"Properties of immersions for systems with multiple limit sets with
implications to learning Koopman embeddings"*, arXiv:2312.17045, IFAC WC 2023, **Automatica** (2025),
doi `10.1016/j.automatica.2025.112220`. **[FULL]** Michigan + Northeastern, control theory, **no
Brunton/Kutz link**.

The 2016 invariant-subspace theorem **[BK]** says no finite-dimensional Koopman-invariant subspace
containing the state exists for systems with multiple fixed points. Deep Koopman routes around it
with a nonlinear decoder — the decoder need not be a linear read-out, so the theorem is arguably
sidestepped. **Liu–Ozay–Sontag close that loophole**, and they do it from outside the lab in question.

- Abstract, verbatim: *"any continuous one-to-one immersion to a class of systems including linear
  systems cannot distinguish different omega-limit sets, and thus cannot be one-to-one."*
- **Theorem 3.16**: under an immersion into a system with closed basins, precompact trajectories and
  countably many ω-limit sets, *"the set {F(Ω) | Ω ∈ W⁺} has exactly one element"* — the immersion
  **collapses every ω-limit set to a single point**.
- **Corollary 3.18**: *"If 𝒳 contains more than one, but at most countably many, ω-limit sets and all
  trajectories are precompact, then a one-to-one linear immersion does not exist."*
- **Theorem 4.27 / Corollary 4.28**, extending to *learned* immersions: *"any immersion candidate that
  can distinguish at least two ω-limit sets would always be ruled out"* as the sampling interval
  decreases and sample size grows; *"approximate linear immersions learned with data converge to
  functions that are not one-to-one."*

**Why this bites Patchworks specifically.** The obstruction is on the **encoder** and requires only
**continuity**. A frozen random ReLU `encode` is continuous. Therefore, on any cell whose piece has
more than one ω-limit set — resting on the ground, in flight, in contact are three — the 12-dimensional
chart cannot be injective on that piece, and the learning process actively pushes it toward merging
those attractors. The nonlinear `decode` does not rescue it: it cannot invert a map that has already
merged two limit sets.

A discrete-time extension exists — *"On the Nonexistence of Continuous Immersions for Discrete-time
Systems"*, arXiv:2605.15161 **[CITE]** — which is the design's actual setting, and it was not reached.

**Two things blunt this, and both should be recorded rather than leaned on.** First, the obstruction
is about *one-to-one*-ness of the immersion. A Patchworks cell is not required to reconstruct its
piece's full state from the chart; it is required to *advance features one step in time*. Whether
non-injectivity across attractors is fatal or merely costly depends on whether two merged limit sets
ever need to be told apart **by that cell**, which is a design question the record does not currently
answer. Second, the sensory rim delivers a **readback** ([#128](https://github.com/NGL321/patchworks/issues/128)),
so a cell straddling contact modes may receive the disambiguating signal from outside rather than
having to encode it. Neither observation is in the literature; both are the kind of re-derivation
[#1](https://github.com/NGL321/patchworks/issues/1)'s "recommendations are input, not instructions"
rule anticipates, and they belong to a revision session, not to this pass.

---

## 4. `k = 12`: chart or lift? The collision #145 already named

**Verdict: the literature is unanimous and the design is not actually in contradiction with it — but
only if a distinction the repo has already ticketed is made explicit and load-bearing.**

The numbers first. Every published lift that works is **larger** than the state it lifts, usually by
one to two orders of magnitude, and usually on toy systems:

| Source | System | State dim | Lift dim |
|---|---|---|---|
| Colbrook & Coote, arXiv:2606.29083 **[ABS]** | pendulum | 2 | **60** (Fourier–Hermite) |
| same | undamped oscillator | 2 | **36** (Hermite tensor) |
| same | Duffing oscillator | 2 | **100** (Chebyshev tensor) |
| Bruder, Gillespie, Remy & Vasudevan, RSS 2019 **[FULL]** | soft robot arm | 2 (+delay, inputs ≈ 6) | **N = 330** (monomials, degree ≤ 4) |

The RSS 2019 figure is verbatim: *"using an N = 330 dimensional set of basis functions consisting of
all monomials of maximum degree 4"* — for a **two-dimensional end-effector position**. Michigan;
independent of Brunton/Kutz.

Mezić (2022) **[FULL]** gives the reason: tensor-product dictionaries grow *"exponentially with the
dimension d"*. And Colbrook, Drmač & Horning **[FULL]** add the caution that matters most here:
a finite-dimensional invariant subspace's *"existence … is not guaranteed, and even if it exists, the
dimension k is typically unknown. (This is a common misconception in papers citing [2], which
explicitly makes this point.)"* — [2] being the 2016 theorem.

**On its face `k = 12` under `n = 32` is a compression, and a compression is not a lift.** But the
repo never claimed it was one. `CONTEXT.md` defines **chart** as *"the low-dimensional coordinates a
cell computes in… Its dimension is fixed by construction and below that of the node stalk"*, and
ADR-0004 states *"`k` is the dimension of the piece; `n` is the room needed to talk about it."* The
chart is a coordinate system on a locally Euclidean piece. A Koopman lift is a dictionary of
observables chosen to make the dynamics linear. **They are opposite operations that happen to share
a symbol.**

This is precisely what [#145](https://github.com/NGL321/patchworks/issues/145), *"`k_piece` vs
`k_lift`: re-defend the chart dimension"*, was opened to settle — before this pass ran. The pass's
contribution is therefore not to discover the problem but to **price it**, and the price is high:

- If `K` is a **reduced-order model on a 12-dimensional piece**, the design is coherent, ADR-0004
  carries it, and the literature above is simply about a different object. **No spectral claim is
  available**, because none of the convergence theory applies to a compression.
- If `K`'s eigenvalues are to be read as a **spectrum**, then `k` is playing the role of `k_lift`, and
  12 is off the bottom of the scale where even *trained* dictionaries were tested on 2-dimensional
  systems.

**These cannot both be true, and [#143](https://github.com/NGL321/patchworks/issues/143) — timescale
as the spectrum of `K` — currently assumes the second while ADR-0004 asserts the first.** That is the
single most important consequence of this pass, and it is a question for #145 and #141, not for this
document to settle.

Two escape routes exist in the literature, recorded as candidates only:

- **Grow the lift adaptively.** Li, Abuduweili, Sun, Chen, Zhao & Liu (CMU), arXiv:2411.14321 **[ABS]**
  grow the lifted space incrementally for legged locomotion. LWPR does the same thing much earlier by
  a different route (§13).
- **Generate the dictionary from delays rather than sampling it.** Mezić **[FULL]**: Krylov/Hankel
  observables *"do not suffer from the curse of dimensionality… the dynamics selects the basis by
  itself."* **This is the cheapest fix available to this architecture specifically**, because a
  streaming embodied system with unit delay on every edge already has the delay line for free, and
  `docs/research/032` already leans on the Sauer–Yorke–Casdagli embedding criterion to defend `k = 12`.
  The two arguments are the same argument, and noticing that is worth a ticket on its own.

---

## 5. The spectral read-out: what the numerical analysts say it costs

**Verdict: an uncertified per-cell spectrum is not a weaker version of a certified one. It is a
different kind of object, and the design does not currently budget for the certification.**

### 5.1 The impossibility result

**Colbrook, Mezić & Stepanenko**, *"Limits and Powers of Koopman Learning"* / *"Adversarial dynamical
systems characterize when data-driven learning succeeds or fails"*, arXiv:2407.06312. **[FULL-HTML]**
Cambridge DAMTP + UCSB — **Mezić is the originator of the modern Koopman programme, disagreeing with
the optimistic reading of his own field.**

For measure-preserving invertible maps of the disk and for smooth interval maps, *"no deterministic
algorithms Γₙ exist that converge to Sp_ap(𝒦_F) for all F in the class Ω as n→∞"*, and *"for any
probabilistic learning algorithms, the probability of convergence cannot exceed 50%."* These hold
**regardless of algorithm, data distribution, or computational model** — unlimited data does not help.

Their Solvability Complexity Index accounting of how many *successive* limits are required:

| System class | Successive limits needed |
|---|---|
| General continuous systems | **three** (snapshots → dictionary size → coherency refinement) |
| Measure-preserving systems | **two** |
| Systems with a **known modulus of continuity** | **one**, with guaranteed convergence |

**Streaming local learning is a one-limit procedure.** The guarantee at one limit requires a known
modulus of continuity for the dynamics, which a contact-rich sandbox does not supply. Independent
corroboration of the SCI framing: Sorg, arXiv:2509.16016 **[ABS]**.

### 5.2 Spectral pollution is the default

**Colbrook, Drmač & Horning**, *"An Introductory Guide to Koopman Learning"*, arXiv:2510.22002.
**[FULL]** On their Duffing figure: *"Most are spurious, illustrating significant spectral pollution
that persists **even as the dictionary size N increases**."* They also flag the correct order of
limits — *"converges in the double limit lim_N lim_M"*, data first then dictionary — so a single joint
limit has no guarantee. And residual pruning does not save you: *"if Algorithm 2 relies solely on
EDMD-generated pairs, parts of the spectrum may remain undetected (spectral invisibility)."*

The underlying references are Davies & Plum (2004, *IMA J. Numer. Anal.* 24(3):417–438) and Lewin &
Séré (2010, *Proc. LMS* 100(3):864–900) — the **numerical analysis** community, wholly outside the
Koopman-ML literature. This is not a Koopman-specific complaint; it is what finite sections of
infinite-dimensional operators do.

### 5.3 Mezić's own counterexample

**Mezić**, *"On Numerical Approximations of the Koopman Operator"*, arXiv:2009.05883, *Mathematics*
10(7):1180 (2022). **[FULL]** Verbatim abstract: *"we … provide an example of a mixing map for which
the finite section method fails."* From the body: *"the finite section method can fail to converge
spectrally for systems with continuous spectrum… This is a mixing map that does not have any
eigenvalues of the Koopman operator on L²(S¹)… U_n has eigenvalue 0 of multiplicity n."*

### 5.4 What certification would cost

The rigorous methods exist, and their dates are informative — everything before them, including every
plain-EDMD spectrum, is unverified:

- **Colbrook & Townsend**, arXiv:2111.14889, *Comm. Pure Appl. Math.*, doi `10.1002/cpa.22125`
  **[ABS]** — ResDMD, *"the **first** scheme for computing the spectra and pseudospectra of general
  Koopman operators from snapshot data without spectral pollution"* (2021).
- **Colbrook**, arXiv:2209.02244, *SIAM J. Numer. Anal.* (2023) **[ABS]** — mpEDMD, *"the first
  truncation method whose eigendecomposition converges to the spectral quantities of Koopman operators
  for general measure-preserving dynamical systems."* **A dissipative contact sandbox is not
  measure-preserving, so this guarantee does not transfer.**
- Statistical side, independent of both labs: **Kostic, Lounici, Novelli & Pontil**, NeurIPS 2023,
  arXiv:2302.02004 **[ABS]** — *"The results shed new light on the emergence of spurious eigenvalues,
  an issue which is well known empirically"*, and *"EDMD suffers from a larger bias."*

**Cost note the design has not budgeted.** ResDMD's residual test requires a Galerkin approximation of
`𝒦*𝒦` as well as `𝒦` — a second, quadratic-order data product **per cell**, every cell, continually.
Without it the per-cell spectrum is uncertified. With a *learned, nonlinear, non-invertible* encoder,
a finite matrix always has eigenvalues, and they are properties of the encoder as much as of the
dynamics.

---

## 6. Contact: a precondition on the simulator, not on the learner

**Verdict: tractable, but only under a condition that is a checkable fact about the sandbox and has
not been checked. This is the cheapest open item in this document.**

**O'Neill, Terrones & Asada** (MIT), *"Koopman global linearization of contact dynamics for robot
locomotion and manipulation enables elaborate control"*, arXiv:2511.06515, **Nature Communications**
(2026), doi `10.1038/s41467-026-72485-7`. **[FULL]** Independent of Brunton/Kutz — and note this is a
*pro*-Koopman paper, which makes the concession more useful, not less.

Verbatim: *"The second fundamental challenge is that **the existence of a Koopman operator cannot be
guaranteed for dynamic systems with discontinuities**."*

Expanded, verbatim: *"Many legged robots are modeled as hybrid systems where ground collisions are
treated as discrete impulses, leading to instantaneous contact dynamics and discontinuous changes in
momenta. **In general, these discontinuities in the governing equations prohibit the application of
Koopman operator theory: they violate one of the key assumptions of the Koopman operator.**"*

Their remedy is **not algorithmic**. It is to change the physics: *"building causal physical models
without discontinuities at mode boundaries. Modeling compliance at the interactions between a robot
and the environment is the key"* — a viscoelastic contact model (stiffness, damping, penetration
depth) so momenta stay continuous.

**So the verdict on contact is a fact about the contact solver:**

- **Compliant contact** (finite stiffness/damping, continuous momenta) → contact pieces are tractable,
  with published precedent including real convex-MPC control through multiple contact changes.
- **Impulsive / rigid contact** → the operator is not guaranteed to exist, and the per-cell spectral
  story on those cells is not noisy but **unfounded**.

MuJoCo's soft-but-stiff default sits between the two and needs looking at rather than assuming.
`docs/spec` should record which, and it is a one-afternoon check.

A second concession from the same paper bears on the motor boundary: *"As the system is lifted, the
number of state variables increases. Therefore, CCK lifted dynamical systems are **often not
controllable**."* They dodge it because MPC does not need controllability. A design that wants to
**act** through the lift does not get that dodge free.

**And one operator per piece may not be enough.** Li et al. (CMU), arXiv:2411.14321 **[ABS]** found a
unified operator could not capture mode-dependent behaviour across stance/swing transitions and
learned **separate operators per contact mode**, growing the lifted space incrementally. This cuts
both ways, and both should be recorded: *for* the design, practitioners independently converge on
per-piece rather than global operators; *against* it, their decomposition is by **contact mode** — a
dynamics-defined partition — while a Patchworks cell owns a **spatial** piece that will straddle
several modes with one fixed `K`. The hybrid-systems paper the ticket named (Govindarajan, Arbabi,
van Blargian, Matchen, Tegling & Mezić, arXiv:1608.08734, IEEE CDC 2016) was **[ABS]** only and
should not be cited for either side of the well-definedness question without reading it.

---

## 7. Bilinearity at the motor boundary — three independent groups

**Verdict: the most actionable finding in the pass after §3, and it is cheap.**

**Bruder, Fu & Vasudevan**, *"Advantages of Bilinear Koopman Realizations for the Modeling and Control
of Systems with Unknown Dynamics"*, arXiv:2010.09961, **IEEE RA-L** 6(3):4369–4376 (2021). **[ABS]**
Michigan — independent.

They give **necessary and sufficient** conditions for a system to admit a valid linear or bilinear
realization over given observables, and show **every control-affine system admits an
infinite-dimensional bilinear realization but not necessarily a linear one.** The clause that hurts:
approximate **bilinear** realizations *tend to improve as the number of basis functions increases,
whereas approximate linear realizations **may not**.*

Read that carefully. It says the failure mode is not "needs a bigger `k`" — it says a linear-in-input
model can **stop improving entirely** as capacity grows, because the object it is converging to does
not exist.

Corroborated from a different continent and school: **Peitz & Klus**, *Automatica* 106 (2019), and
**Peitz, Otto & Rowley**, arXiv:2003.07094, *SIAM J. Appl. Dyn. Syst.* **[ABS]**. Their two remedies
are (i) a finite set of autonomous operators, one per constant input, switched between; or (ii) a
bilinear surrogate by convex combination — with linear interpolation between two **generators**
introducing no error for control-affine systems. Both are admissions that a single input-linear
operator is the wrong object there.

**The motor boundary is exactly a control-affine interface.** The change is small: `z' = A z + Σᵢ uᵢ Bᵢ z`
— one `k×k` per input channel instead of one `k×k`. It is also the natural sheaf-morphism
generalisation, since the restriction map's action becomes input-modulated rather than fixed. This
lands directly on [#146](https://github.com/NGL321/patchworks/issues/146), which already exists and is
blocked; the pass supplies its evidence base and strengthens the case for unblocking it earlier than
planned.

---

## 8. The frozen random dictionary

**Verdict: pre-registered claim 2 is false, with a direct matched-size experiment.**

**Coote & Colbrook**, *"Residual-Guided Dictionary Learning for Spectrally Accurate Koopman
Approximation"*, arXiv:2606.29083 (Cambridge). **[ABS]** This is precisely the experiment: same
architecture, same size, **untrained vs trained** neural dictionary, scored by ResDMD residual — an
operator-level criterion, not a fitting metric.

Untrained, on the pendulum: *"spurious eigenvalues closer to the centre of the disk and little
spectral coverage on the left semicircle"*; on Duffing, *"substantial pollution: many eigenvalues
cluster near z=1, while others appear in the interior of the disk."* Trained: *"substantial reduction
in pollution"*, coverage extends around the unit circle, and *"the trained dictionaries achieve
markedly lower ResDMD residuals across all systems tested."*

Two consequences, the second worse than the first: a random dictionary is **not** competitive with a
learned one at matched size; and their matched sizes were **36–100 for 2-dimensional systems**, so
`k = 12` on a 32-dimensional stalk is off the bottom of the scale where even the trained version was
tested.

**Stated fairly, the countervailing evidence.** The ticket's identification of a frozen random
`encode` with random-feature EDMD is **correct and useful**: DeGennaro & Urban, *"Scalable Extended
Dynamic Mode Decomposition using Random Kernel Approximation"*, arXiv:1710.10256 (Brookhaven —
**not** Brunton/Kutz) **[ABS]** is real and does what the ticket says, with *"random Fourier features,
and the Nyström method."* The error analysis the ticket cites also checks out: **Philipp, Schaller,
Worthmann, Peitz & Nüske**, arXiv:2312.10460 **[ABS]**.

But the identification is looser than it looks, and the looseness matters. Random Fourier features
are **structured** random features approximating a *known* kernel, with theory attached to that
approximation — and, per DeGennaro & Urban's own text, they apply to **translation-invariant kernels
only**. A frozen random **ReLU** layer is closer to a random-feature approximation of an *arc-cosine*
kernel, and inherits none of the RFF guarantee. The Nyström route is data-dependent and carries no
such restriction, but a frozen random MLP is not that either. **The literature behind the frozen
encoder is real but it is not the literature the ticket claims**, and the distinction should be made
before the claim is repeated.

---

## 9. Stability-guaranteed Koopman, and the composition question nobody has answered

**Verdict: `rho(K)` as a settable design variable is well-founded, and the right template is a direct
parameterisation rather than a penalty. The composed system's stability is unclaimed territory.**

**Fan, Yi, Rye, Shi & Manchester**, *"Learning Stable Koopman Embeddings"*, arXiv:2110.06509
(Sydney — **not** Brunton/Kutz). **[ABS]** Verbatim: *"we prove that every discrete-time nonlinear
contracting model can be learnt in our framework… it allows for unconstrained optimization over the
Koopman embedding and operator jointly while enforcing stability of the model, via a **direct
parameterization** of stable linear systems, greatly simplifying the computations involved."*

Two properties make this the right template for Patchworks specifically:

- **Direct parameterisation means no constrained optimiser and no projection step**, which is what a
  purely local learning rule needs. A penalty-based stability constraint would want a global
  objective. Note the contrast with ADR-0010's gauge, which *is* a projection — the two mechanisms
  could be made to match, and whether they should is a real question.
- **The completeness result** — *every* discrete-time contracting model is representable — means
  constraining `K` to a spectral-radius band costs no expressiveness **within the contracting class**.

**The unanswered question, and it is the design's actual one.** Nothing found addresses the spectral
radius of a **composition of per-cell operators through learned restriction maps**. The dome must not
be globally contracting if it is to sustain activity, so per-cell `rho(K)` plus maps in `[1/rho, rho]`
needs an explicit argument about the composed system's spectral radius. That argument does not exist
in the literature and will have to be made here. It belongs to
[#140](https://github.com/NGL321/patchworks/issues/140) and is flagged as an unanswered risk, not a
solved one.

Weaker leads, recorded but **not read and not to be cited**: an orthogonal–diagonal–orthogonal
factorisation with singular values inside the unit circle (`2607.19719`, `2602.02592`). **[UNREACHED]**

---

## 10. Over-squashing and the remedies literature — the transmission ticket

**Verdict: the measured ~921x per hop is the theory's prediction, not an anomaly; the conversion is
half the recommended remedy; and the other half is the thing the design has refused.**

This section extends `docs/research/031`, which established the diagnosis (Alon & Yahav, Topping et
al., Di Giovanni et al., Arroyo et al.) but did not investigate the remedies. It is aimed at
[#142](https://github.com/NGL321/patchworks/issues/142).

### 10.1 The measurement is the bound

**Di Giovanni, Giusti, Barbero, Luise, Liò & Bronstein**, arXiv:2302.02941, **ICML 2023**. **[FULL]**
Theorem 3.2 factorises node-to-node sensitivity into a **model** term `(c_σ w p)^m` and a
**topological** term — a per-hop factor raised to the hop count, which is exactly the shape
[#120](https://github.com/NGL321/patchworks/issues/120)'s decomposition takes. Theorem 5.5 bounds the
Jacobian obstruction above and below by `τ(v,u) / 2|E|`, i.e. by **commute time**, *independent of
depth*. Theorem 4.2: when the per-layer factor is below 1, decay is exponential in layers.

Three consequences land directly on #142:

1. **Depth cannot help** — Theorem 4.2 says more hops are strictly worse. Independent confirmation of
   the map's closure of pre-training waves, by a different route than the arithmetic used there.
2. **The gauge band and the measured attenuation are the same order of magnitude.** `rho = 2` over
   9 hops is at most `2^9 = 512`; the measurement is ~921 per hop. Whether that is coincidence or
   whether the gauge is the dominant term in the model factor is **measurable and has not been
   measured**. This is a concrete, cheap experiment and it bears on candidate 3 of #142.
3. **The right lever is the topological term.** Compute commute time / effective resistance on the
   actual 150-node dome. It is cheap at that size and it names exactly which cell pairs are
   unreachable — turning the attenuation from an observation into a diagnosis.

**The cross-reference that matters most in this document:** effective resistance is *also* one of
Hansen & Ghrist's spectral sheaf results (§2). The sheaf Laplacian's effective resistance and the
over-squashing effective resistance are the same object seen from two sides. Anything done to the
restriction maps has an over-squashing consequence and vice versa, and that link is currently
unexploited in the record.

### 10.2 The recurrent bridge — and why the conversion is half a remedy

**Arroyo, Gravina, Gutteridge, Barbero, Gallicchio, Dong, Bronstein & Vandergheynst**,
arXiv:2502.10818 (2025). **[ABS]** Verbatim: *"a simple state-space formulation of a GNN effectively
alleviates over-smoothing and over-squashing at no extra trainable parameter cost"*, and *"(iii)
Over-squashing is most easily alleviated by a combination of graph rewiring and vanishing gradient
mitigation."*

Patchworks *is* a recurrent state-space GNN. A learned linear `K` with controlled `rho(K)` is
precisely "a state-space formulation". **So the conversion is independently the recommended remedy,
and this should be cited on #138 as support rather than background.** But finding (iii) is a
conjunction: rewiring **and** vanishing-gradient mitigation. `K` supplies the second. The design
refuses the first.

### 10.3 The remedies, and what each costs

| Remedy | What it buys | What it costs |
|---|---|---|
| **Fully-adjacent last layer** (Alon & Yahav, ICLR 2021) **[FULL]** | **42% avg error reduction on QM9** across six GNN types; ENZYMES 59.6%→67.7%; SOTA on VarMisuse — all *without retuning* | A global all-to-all operation. Destroys locality and the local learning rule. |
| **Curvature-based rewiring** (Topping et al., ICLR 2022) **[ABS]** | Provably targets the negatively-curved edges *"responsible for the over-squashing issue"* | Changes the graph. The dome's topology is the problem specification, not a free parameter. |
| **Virtual / master node** | Reduces diameter to 2; decreases effective resistance between distant nodes | A global hub, and a single point of failure for a system with no global backprop. |
| **Local Virtual Nodes** (Karabulut & Baytaş, arXiv:2508.20597) **[ABS]** | *"alleviate the effects of over-squashing without significantly corrupting the global structure"*; positions chosen by centrality | Adds non-physical nodes to a graph whose nodes own real pieces. **Mildest option found, and the one most compatible with a dome** — worth real consideration for the rim→apex path. |
| **Expander / Ramanujan rewiring** **[CITE]** | Reported gains ~12% / ~14% — **from search summaries only; primary papers not read; do not cite these numbers** | Expanders are maximally non-local; nothing about a physical dome is one. |
| **State-space reformulation** (Arroyo et al.) **[ABS]** | Both over-smoothing and over-squashing, *"at no extra trainable parameter cost"* | **Nothing. This is the free one, and it is what stage 1 already proposes.** |

Note how this reads against ADR-0002, whose Consequences tell a future engineer to *"reach for
topology changes, not for reopening this decision to add rounds back."* The literature agrees that
topology is the dominant term — and every topology remedy it offers costs locality. The two cheapest
things on this table that do **not** cost locality are the conversion itself and Local Virtual Nodes.

### 10.4 The spatiotemporal case, unchecked and testable

**Marisca, Bamberger, Alippi & Bronstein**, arXiv:2506.15507, **NeurIPS 2025**. **[ABS]** *"the
temporal dimension amplifies this challenge… counterintuitively, convolutional STGNNs favor
information propagation from points temporally distant rather than close in time"*, and time-and-space
vs time-then-space *"are equally affected"*. Patchworks is a time-and-space spatiotemporal GNN by
construction. The counterintuitive prediction — recent-in-time information is *harder* to propagate —
is specific, testable on the dome, and has not been checked.

---

## 11. Local learning rules vs backprop

**Verdict: the "no global backprop" commitment is not free, and the literature's verdict is
conditional rather than favourable. This is a standing constraint from #1, not a live decision, and
is recorded for the falsification register rather than reopened.**

- **Bartunov, Santoro, Richards, Marris, Hinton & Lillicrap**, NeurIPS 2018, arXiv:1807.04587
  **[ABS]** — on ImageNet, biologically-motivated local algorithms land around **93–99% top-1 error**
  against backprop's **~71.4%**. The gap is **worst for locally-connected, non-weight-shared**
  architectures, which is the family per-cell local rules resemble. Note Hinton is a co-author
  arguing against the optimistic reading.
- **Wang, Ni, Song, Yang & Huang**, ICLR 2021, arXiv:2101.10832; extended as InfoPro, *IJCV* (2024)
  **[ABS]** — greedy layer-wise local learning suffers **information collapse**: it learns features
  benefiting only the local module, so task-relevant information is washed out early and later
  modules cannot use their capacity. **This is the precise risk here**: nothing forces cell *i*'s 12
  dimensions to retain what cell *j* needs three hops on, and the restriction maps are learned
  locally too, so they cannot recover what the encoder already discarded. But the same paper shows
  that with an explicitly information-preserving local loss, *"deep networks with competitive or even
  better performance can be obtained… with gradient-isolated modules."* **The finding is conditional:
  local learning works iff the local objective is designed to preserve information for downstream
  consumers.** A pure per-cell prediction loss is the greedy objective they show fails.
- **Innocenti, Achour & Buckley**, arXiv:2505.13124 **[ABS]** — predictive coding has *"notoriously
  struggled to train very deep networks."* Counterweight, in fairness: **Millidge, Tschantz &
  Buckley**, *Neural Computation* 34(6):1329 (2022), arXiv:2006.04182 **[ABS]** — PC approximates
  backprop along arbitrary computation graphs using only local information, at 100–200 inference
  iterations per update.

The InfoPro finding is the one worth carrying forward: it converts "will local rules work?" into the
sharper "is the per-cell objective information-preserving for downstream cells?", which is a question
the transport rule's design can actually answer.

---

## 12. Known-answer validation: the cylinder wake

**Verdict: numbers obtained, and the canonical Koopman reference is Re = 50, not the Re = 100 usually
quoted. Do not mix them.**

**Bagheri**, *"Koopman-mode decomposition of the cylinder wake"*, **J. Fluid Mech. 726**, 596–623
(2013). **[FULL]** KTH — **not** Brunton/Kutz.

- *"For Reynolds number below the critical value of Re_c = 46.6, the flow consists of two steady
  symmetrical vortices."* Snapshots at **Re = 50**, `t = 700`–`1400`, `Δt = 1`.
- Reference spectrum, Eq. (5.9): `σ = -2λ ≈ -0.03` (Lyapunov exponent of the limit cycle) and
  `ω ≈ 0.79` (fundamental angular frequency). With `D = U = 1`, `St = ω/2π ≈ 0.126`.
- The structural fact that makes this a good test: *"The frequencies and growth rates of the modes
  form a lattice in the lower-half of the complex plane, where the spacing in the vertical and
  horizontal directions correspond, respectively, to the fundamental frequency and the Lyapunov
  exponent of the oscillator."* And *"the Ritz values are collocated on this lattice."*

**Why this is a better instrument than a single frequency:** the correct answer is a *lattice* with
known spacing and known per-mode symmetry, so a per-cell scheme is scored on structure rather than on
matching one number. Two cautions: the lattice is a property of a weakly nonlinear single-Hopf flow
near the bifurcation and will not look like this at Re = 100; and `σ ≈ -0.03` is small, so a short
rollout will not resolve it.

Re = 100 numbers were reached only through a rendered summary of Tu, Rowley, Luchtenburg, Brunton &
Kutz **[BK]**, arXiv:1312.0041, and **should be re-checked against the paper before use**.
Recommendation: use Bagheri Re = 50 — it is the Koopman-theoretic reference, the answer has structure,
and it is outside the orbit.

---

## 13. The opposing bets

### 13.1 SINDy — and the honest framing of the trade

The unifying paper is **Gao, Williams & Kutz**, SINDy-SHRED, arXiv:2501.13329, **PNAS**
123(16):e2508144123 (2026). **[ABS] [BK]** — and it is **structurally incapable of adjudicating the
bet, because it is written by one side of it.** Body not reached (403).

What its abstract does say is uncomfortable: restricting SINDy to a linear model *yields* the
Koopman-SHRED variant — the Koopman branch is presented as a **special case obtained by throwing away
terms** — and all four claimed advantages (symbolic interpretability, physics discovery, provably
robust convergence with a convex loss landscape, superior accuracy and data efficiency) are attributed
to the **nonlinear SINDy** branch. No advantage of the Koopman restriction is named in the abstract.

The genuine trade, from the independent control literature **[ABS-level]**: lifting buys stability
analysis and simulation-free prediction, and generator-based Koopman beats SINDy at **low sampling
rates** where derivative estimates degrade; sparsifying keeps the state small and interpretable in
original coordinates, and there is no general-purpose lifting library that works across systems.

**And the decisive asymmetry for this design: SINDy produces no spectrum.** That is why it is not a
drop-in alternative — but it also means the entire justification for the lift rests on the spectrum
being trustworthy. Per §4 and §5, at `k = 12` with a learned nonlinear encoder on contact dynamics,
it is not. **Stated at its sharpest: if the spectral claim is dropped, the lift is pure cost and
SINDy is the better-supported bet.** The map's cost-gate argument survives this (see the headline verdict) because it never
rested on the spectrum — but [#143](https://github.com/NGL321/patchworks/issues/143) does.

### 13.2 rSLDS — the second established answer to multiple attractors

**Linderman, Miller, Adams, Blei, Paninski & Johnson**, arXiv:1610.08466, AISTATS 2017 **[ABS]**; and
**Nassar, Linderman, Bugallo & Park**, *"Tree-Structured Recurrent Switching Linear Dynamical Systems
for Multi-Scale Modeling"*, arXiv:1811.12386, **ICLR 2019** **[ABS]**. Columbia / Stony Brook
computational neuroscience — a genuinely independent tradition, and the one closest to the embodied
framing.

This is the strongest competitor framing in the pass: *many local linear operators + a learned
switching rule*, against Patchworks' *many local linear operators + a fixed spatial graph*. The
switching rule does the work the graph does here. The honest comparison is that rSLDS partitions
**state space** and switches in time, while Patchworks partitions **the problem** and runs all cells
always.

One concrete, cheap design suggestion falls out. TrSLDS imposes a hierarchical prior — *"partitions
that share a common parent should have similar dynamics"* — and can be *"queried at different levels
of the hierarchy to obtain dynamical descriptions at multiple levels of resolution."* Patchworks has
no analogue, and the dome's spatial adjacency is the obvious place to put one: **neighbouring cells'
`K` should be similar**. That is a regulariser the transport rule could carry, and it is the kind of
thing that would also raise the model term in §10.1.

### 13.3 LWPR — the robotics ancestor, and why its boundary answer does not transfer

**Vijayakumar, D'Souza & Schaal**, *"Incremental Online Learning in High Dimensions"*, *Neural
Computation* **17**, 2602–2634 (2005). **[FULL]**

The premise is identical to ADR-0004's: *"The prerequisite of our approach is that the
high-dimensional learning problems we address have locally low-dimensional distributions."* LWPR's
`R ≪ N` is `k < n` with different names, and each local model runs its own locally-weighted partial
least squares — per-model local dimensionality reduction, i.e. a chart. **This is the ancestor and
should be cited as such.**

The boundary-handling question the ticket asked has a clear answer, and it is that LWPR **dissolves**
boundaries rather than handling them. Output is *"the normalized weighted mean of all K linear models,
ŷ = Σ_k w_k ŷ_k / Σ_k w_k"* with Gaussian receptive-field weights in each model's own learned metric.
Overlap is wanted explicitly: *"the more overlap is permitted, the better the function-fitting
results, without any danger that the increase in overlap can lead to overfitting."*

**That answer does not transfer**, and the reason is instructive. LWPR indexes local models by position
**in input space**, so a query can fall between models and must be blended. Patchworks indexes cells
by position **in a fixed graph**; a cell owns its piece and never answers for a neighbour's. See §14
for the source that speaks to the Patchworks regime instead.

Two transfers that do hold: LWPR fixes receptive-field centres to avoid *"negative interference"*
under non-stationary inputs, which a fixed dome gets free — while inheriting the matching weakness
that **nothing relocates a mis-assigned piece**, and unlike LWPR a fixed dome cannot grow new models
on demand. And LWPR grows `R` adaptively per model by watching `MSE_{r+1}/MSE_r`, which is
ready-made machinery for asking whether every cell's piece actually needs 12 dimensions (§4).

---

## 14. Seams do not break — the one reassuring finding

**Propp, Perego, Cyr, Gruber, Howard, Heinlein, Stinis & Tartakovsky**, *"Domain-Decomposed Graph
Neural Network Surrogate Modeling for Ice Sheets"*, arXiv:2512.01888 (2025). **[FULL]**

Verbatim: *"we adopt **non-overlapping** partitions based on mesh elements, which simplifies the
training pipeline while retaining accuracy."* And on whether seams break: *"our GNN surrogate remains
remarkably robust even with non-overlapping partitions: **we do not see error growth near subdomain
boundaries, nor do we observe the kinds of interface discontinuities one might expect when models are
trained independently.**"* The reason they give is the transferable part: *"because our GNN surrogate
is fully data-driven and optimized under an MSE objective rather than solving a PDE directly, there is
no analogous need to transport nullspace information across subdomains."*

**This removes the most-feared local failure mode.** Independently-trained per-subdomain learned models
do not produce interface discontinuities, and a locally-learning Patchworks cell is in the same
regime — data-driven objective, not a PDE solve. It also says overlapping charts are unnecessary,
which is the opposite of LWPR's advice (§13.3), and the disagreement resolves cleanly: LWPR overlaps
in *input space* where a query can fall between models; a mesh or a dome assigns each location exactly
one owner.

Caveat: their subdomains are large mesh patches, not 12-dimensional charts, and their setting is
time-independent. A ~150-cell recurrent system with unit delay is meaningfully harder, and the result
shifts the prior rather than settling it.

**The convergent pressure worth naming.** The learned-domain-decomposition literature knows that the
**coarse space** — a global, low-rank correction — is what buys scalability (Klawonn, Lanser & Weber,
arXiv:2312.14050, the survey **[CITE]**; Taghibakhshi et al. on learned Robin interface conditions
**[CITE]**). Patchworks has deliberately refused a coarse space, and §10's over-squashing analysis
penalises exactly that refusal. **Two independent literatures arrive at the same recommendation — you
need some global or multi-scale path — and the design has declined it in both.** That is the strongest
external pressure on the design found anywhere in this pass, and it is aimed at #142 rather than at
the conversion.

A pleasing incidental: a Patchworks **restriction map** into an edge stalk is structurally the same
object as a *learned, subdomain-specific Robin interface parameter*. The DD literature has design
guidance the repo is not currently drawing on.

---

## Verdicts on the five pre-registered vulnerable claims

Ordered as pre-registered.

### Claim 0 — "timescale-from-spectrum is new"

**FALSE, confirmed. High confidence.** Already believed false; nothing rehabilitates it. Reading
timescales off a spectrum, at multiple resolutions, is the explicit content of mrDMD and the
Hankel/delay line **[BK]**, and the rigorous version — spectral measures, continuous spectra — is a
developed subfield with its own convergence theory.

**The pass adds something worse than unoriginality.** The timescale read-out is the *specific
quantity* Colbrook–Mezić–Stepanenko prove is uncomputable in one limit for this system class (§5.1).
Being second is the lesser problem.

### Claim 1 — "the distributed-Koopman literature does not already do the sheaf version"

**FALSE, near-certainly. Correctly pre-registered as most likely wrong.** Mukherjee et al. (LoG 2022,
PNNL) have per-agent encoders and decoders, a block-sparse operator supported on graph edges, parallel
per-agent learning and no global operator (§1.2). Li et al. (§1.3, body confirmed this pass) have
parallel per-subsystem operators with per-subsystem lifted dimensions and no consensus step.

Per the ticket's own framing this is **good news operationally** — prior art to build on. What
survives as unclaimed is in §1.4 and is much narrower than what was being claimed. **Rewrite the
novelty claim before it is made.**

### Claim 2 — "a frozen random dictionary is competitive with a learned one at `k ~ 12`"

**FALSE, with a direct matched-size experiment. High confidence.** Coote & Colbrook (§8): untrained
dictionaries show substantial spectral pollution, trained ones *"achieve markedly lower ResDMD
residuals across all systems tested"* — and their matched sizes were 36–100 for 2-dimensional systems.

Qualification that matters: this is a verdict about **spectral accuracy**, scored by an operator-level
residual. It is not a verdict about whether a frozen random `encode` is adequate as a *chart*, which
is a different claim measured differently, and which §4's distinction keeps separate. The frozen
encoder's real defence is [#13](https://github.com/NGL321/patchworks/issues/13)'s reservoir precedent,
not the RFF literature (§8).

### Claim 3 — "contact pieces are tractable at all"

**TRUE, BUT ONLY UNDER AN UNMET PRECONDITION. Medium-high confidence.** Contact is not categorically
out of reach — O'Neill et al. demonstrate real convex-MPC control through multiple contact changes in
one horizon. But the operator's existence is not guaranteed under discontinuous momenta, and their
contribution is **removing the discontinuity from the physics**, not tolerating it (§6).

**This verdict is a checkable fact about the sandbox's contact solver and should be settled before any
spectral claim is made about contact cells.**

### Claim 4 — "`k = 12` is anywhere near enough for a lift"

**FALSE as stated. Highest confidence of the five — and the framing is what needs fixing, not
necessarily the number.** Four independent groups agree no published working lift is smaller than its
state (§4).

But the design does not claim `k` is a lift. `CONTEXT.md` and ADR-0004 define the chart as
coordinates on a locally Euclidean piece — the opposite operation. **The verdict is therefore: `k = 12`
is fine as `k_piece` and indefensible as `k_lift`, and the record currently uses it as both.**
That is [#145](https://github.com/NGL321/patchworks/issues/145)'s question exactly, and this pass
supplies its evidence base.

---

## Corrections to the ticket's own citations

Three, and the first is not cosmetic.

1. **`10.1016/j.ejcon.2026.100050` does not exist.** 404 at doi.org, *"Resource not found"* from the
   Crossref API, no match in *European Journal of Control*. **[UNREACHED]** The paper it appears to
   garble is Azarbahram, Liu & Incremona, *EJC* **89** (2026), art. 101497,
   doi `10.1016/j.ejcon.2026.101497`. **This matters beyond bookkeeping:** the ticket files the dead
   DOI under "localized EDMD", but the real paper is a **consensus** paper (§1.1) — cited for close to
   the opposite of what it says. Correct or strike.
2. **The Römer & Breitenhuber quote is unverified.** *"the price that we pay for the desirable property
   of linearity is the infinite dimensionality of the function space"* — PAMM 24(3),
   doi `10.1002/pamm.202400187`. Wiley returned 403 on both routes. **[CITE]** The citation exists;
   the verbatim quote was not confirmed by this pass. Mark it unverified or have someone with Wiley
   access check it. (The *claim* is not in doubt — Mezić (§5.3) and Otto & Rowley say the same thing
   from reachable sources — only this rendering of it.)
3. **arXiv:1608.08734** (Govindarajan et al., hybrid pendulum) was **[ABS]** only. Do not cite it as
   authority for either side of the hybrid-well-definedness question until the body is read.

**One process note, recorded because the pass exists to be trustworthy.** During this work a fetch of
arXiv:2508.18954 returned a **fabricated verbatim quote** claiming learned Koopman representations
generalise poorly off-distribution. The actual PDF reports the **opposite**: *"Koopman embeddings
outperform both standard and physics-informed PCA baselines… fixing the pre-trained transformer
weights during fine-tuning leads to no performance degradation."* The quote does not exist in the
paper. It is recorded here, against the pass's own thesis, because suppressing it would be exactly
the bias this pass was opened to correct — and as a standing reason to prefer **[FULL]** over
**[ABS]** on any load-bearing claim.

---

## Where the evidence is thin

Stated plainly, because the pass's credibility depends on it.

**Target 3 — off-distribution generalisation — is the weakest section and could not be supported.**
No multi-lab reproduction study of Koopman autoencoders, no reproducibility-challenge report on Lusch
et al., and no benchmark paper where deep Koopman is reported losing to an MLP on the same data were
found. The literature is overwhelmingly composed of success reports. **The absence is itself
informative** — it is the signature of a field that does not publish failures — but it is not
evidence, and no claim in this document rests on it.

What was found is indirect and belongs in the falsification register rather than in an argument:

- **Fathi, Gehring, Pilault, Kanaa, Bacon & Goroshin**, arXiv:2310.15386 (Mila / DeepMind) **[ABS]**
  *"discover several limitations of predicting future states in the latent space"* and need **Periodic
  Reencoding** at inference to stop trajectory drift — i.e. naive latent rollout of a learned linear
  operator drifts, and the fix is to stop trusting the operator. **Patchworks' cells roll forward
  continually on a stream with no such correction**, though the sensory rim arguably supplies one.
- **Zhang & Gilpin**, arXiv:2505.11349 **[ABS]**: a baseline that simply copies the most similar
  segment of its own context beats leading time-series foundation models on low-dimensional chaos
  *"at a tiny fraction of the computational cost"*; and Gilpin's dysts line finds *"many complex, deep
  learning methods fail to outperform a linear, autoregressive baseline."*

**The operational consequence is a pre-registration requirement, not a finding:** any claim that a
learned linear `K` beats the frozen random `step` needs a copy-the-nearest-past-segment baseline and a
plain linear-AR baseline alongside it, or the result will not be believable. That belongs on
[#147](https://github.com/NGL321/patchworks/issues/147), the falsification register.

Also thin, and flagged: no dedicated polemic against deep Koopman's nonlinear decoder was found in
print (§3 rests on Liu–Ozay–Sontag's theorem instead, which is stronger anyway); the objection exists
mostly *implicitly*, in people rebuilding the architecture to remove the nonlinear decoder
(invertible-NN and symplectic Koopman variants, both **[CITE]**).

---

## Recommended revision tickets (recommendations only; none opened)

Per [#1](https://github.com/NGL321/patchworks/issues/1): a research doc's recommendations are **input,
not instructions**. Each of these is one candidate fix; a revision session re-derives and owns the
conclusion. They are ordered by what they cost to act on.

**R1 — Correct the dead DOI on [#148](https://github.com/NGL321/patchworks/issues/148) and re-file it.**
`task`. The citation is dead and the paper it names says the opposite of the use it was put to.
Minutes of work; see Corrections (1).

**R2 — Check the sandbox's contact model for continuity of momenta.** `task`. The verdict on claim 3
is a fact about the contact solver, not a decision. If contact is impulsive, the spectral story on
contact cells is unfounded and several long-horizon tickets change shape. One afternoon; see §6.

**R3 — Compute commute time / effective resistance on the actual dome.** `task`. Cheap at 150 nodes,
and it converts [#120](https://github.com/NGL321/patchworks/issues/120)'s ~921x per hop from an
observation into a diagnosis naming which cell pairs are unreachable. Feeds
[#142](https://github.com/NGL321/patchworks/issues/142) directly. The same quantity appears in
Hansen & Ghrist's sheaf spectral theory, so it is one computation serving both halves of the
architecture. See §10.1.

**R4 — Measure whether the gauge band is the dominant term in the model factor.** `task`. `rho = 2`
over 9 hops gives `2^9 = 512` against a measured ~921 per hop. Coincidence or cause is measurable, and
the answer bears on candidate 3 of #142. See §10.1.

**R5 — Settle `k_piece` vs `k_lift` explicitly, and make the answer load-bearing.** This is
[#145](https://github.com/NGL321/patchworks/issues/145), already open; the recommendation is to
**unblock it and put it before [#143](https://github.com/NGL321/patchworks/issues/143)**, because #143
assumes a spectral reading that #145's answer may withdraw. See §4.

**R6 — Make the motor boundary bilinear.** This is
[#146](https://github.com/NGL321/patchworks/issues/146), already open and blocked; the recommendation
is that the evidence is now strong enough (three independent groups, plus a necessary-and-sufficient
condition) to unblock it earlier than planned. The change is one `k×k` per input channel. See §7.

**R7 — Record the ω-limit-set obstruction against the chart, and decide what it costs.** `grilling`.
Liu–Ozay–Sontag is a theorem, not an opinion, and it applies to any continuous encoder. The open
question is genuinely open: whether a cell ever needs to tell two merged limit sets apart, given that
the rim supplies a readback. Belongs with #141 or as its own ticket. See §3.

**R8 — Add trivial baselines to the falsification register.** Belongs on
[#147](https://github.com/NGL321/patchworks/issues/147). Copy-nearest-segment and linear-AR, per
"Where the evidence is thin".

**R9 — Note the delay-dictionary convergence.** `grilling`, low urgency but high leverage. Mezić's
argument that the right dictionary is delay-generated (*"the dynamics selects the basis by itself"*)
and `docs/research/032`'s Sauer–Yorke–Casdagli defence of `k = 12` are **the same argument reached
from two directions**, and the architecture already has a delay line on every edge. If the spectral
reading is to be kept at all, this is the cheapest route to it. See §4.

**Nothing to revise on these.** The cost-gate rationale for the conversion (see the headline
verdict); ADR-0004's
local-flatness commitment, which §4 *strengthens* by giving the chart a defence independent of the
Koopman literature; the differing interior/boundary `m`, which Dönmez et al. independently validate
(§2); and the no-relays and topology decisions, which §10 pressures but which
[#142](https://github.com/NGL321/patchworks/issues/142) already owns and prices.

---

## Sources

Grouped by role. Reading depth and lab-orbit flags as tagged above.

**The critiques (§3, §5, §6, §7, §11)**
- Liu, Ozay & Sontag, arXiv:2312.17045, *Automatica* (2025), doi `10.1016/j.automatica.2025.112220` **[FULL]**
- Colbrook, Mezić & Stepanenko, arXiv:2407.06312 **[FULL-HTML]**
- Colbrook, Drmač & Horning, *An Introductory Guide to Koopman Learning*, arXiv:2510.22002 **[FULL]**
- Mezić, arXiv:2009.05883, *Mathematics* 10(7):1180 (2022) **[FULL]**
- Colbrook & Townsend, arXiv:2111.14889, *Comm. Pure Appl. Math.*, doi `10.1002/cpa.22125` **[ABS]**
- Colbrook, arXiv:2209.02244, *SIAM J. Numer. Anal.* (2023) **[ABS]**
- Kostic, Lounici, Novelli & Pontil, NeurIPS 2023, arXiv:2302.02004 **[ABS]**
- O'Neill, Terrones & Asada, arXiv:2511.06515, *Nature Communications* (2026), doi `10.1038/s41467-026-72485-7` **[FULL]**
- Bruder, Fu & Vasudevan, arXiv:2010.09961, *IEEE RA-L* 6(3) (2021) **[ABS]**
- Bruder, Gillespie, Remy & Vasudevan, RSS 2019 **[FULL]**
- Coote & Colbrook, arXiv:2606.29083 **[ABS]**
- Bartunov et al., NeurIPS 2018, arXiv:1807.04587 **[ABS]**
- Wang et al., ICLR 2021, arXiv:2101.10832; InfoPro, *IJCV* (2024) **[ABS]**
- Innocenti, Achour & Buckley, arXiv:2505.13124 **[ABS]**; Millidge, Tschantz & Buckley, arXiv:2006.04182 **[ABS]**

**The prior art (§1, §2, §13, §14)**
- Mukherjee et al., LoG 2022, PMLR 198 **[FULL]**
- Li, Bo, Zhang, Qin & Yin, arXiv:2404.06746 **[FULL]**
- Azarbahram, Liu & Incremona, *EJC* 89 (2026) art. 101497, doi `10.1016/j.ejcon.2026.101497` **[ABS]**; arXiv:2511.22368 **[ABS]**
- Hao, Lu, Upadhyay & Mou, arXiv:2412.07212 **[ABS]**
- Nandanoori, Sinha & Yeung, arXiv:2106.15678 **[ABS]**
- Peitz, Harder, Nüske, Philipp, Schaller & Worthmann, doi `10.3934/jcd.2024035` **[ABS]**
- Robinson, arXiv:1604.04647 **[ABS]**; Robinson, Szulczewski & Thorson, arXiv:2511.04603 **[ABS]**
- Schultz, Spivak & Vasilakopoulou, arXiv:1609.08086 **[ABS]**
- Hansen & Ghrist, arXiv:1808.01513 **[ABS]**; arXiv:2005.12798 **[ABS]**
- Bodnar, Di Giovanni, Chamberlain, Liò & Bronstein, arXiv:2202.04579, NeurIPS 2022 **[ABS]**
- Dönmez, Mosig, Fritsche & Koch, arXiv:2605.11178 **[ABS]**; Bourgerie, Girdzijauskas & Fodor, arXiv:2605.19021 **[ABS]**
- Linderman et al., arXiv:1610.08466 **[ABS]**; Nassar, Linderman, Bugallo & Park, arXiv:1811.12386, ICLR 2019 **[ABS]**
- Vijayakumar, D'Souza & Schaal, *Neural Computation* 17:2602–2634 (2005) **[FULL]**
- Propp et al., arXiv:2512.01888 **[FULL]**
- Klawonn, Lanser & Weber, arXiv:2312.14050 **[CITE]**; Taghibakhshi et al. **[CITE]**

**Over-squashing and remedies (§10)**
- Alon & Yahav, arXiv:2006.05205, ICLR 2021 **[FULL]**
- Topping, Di Giovanni, Chamberlain, Dong & Bronstein, arXiv:2111.14522, ICLR 2022 **[ABS]**
- Di Giovanni, Giusti, Barbero, Luise, Liò & Bronstein, arXiv:2302.02941, ICML 2023 **[FULL]**
- Arroyo et al., arXiv:2502.10818 (2025) **[ABS]**
- Marisca, Bamberger, Alippi & Bronstein, arXiv:2506.15507, NeurIPS 2025 **[ABS]**
- Karabulut & Baytaş, arXiv:2508.20597 **[ABS]**

**Supporting and secondary**
- Fan, Yi, Rye, Shi & Manchester, arXiv:2110.06509 **[ABS]**
- Bagheri, *J. Fluid Mech.* 726:596–623 (2013) **[FULL]**
- DeGennaro & Urban, arXiv:1710.10256 **[ABS]**; Philipp, Schaller, Worthmann, Peitz & Nüske, arXiv:2312.10460 **[ABS]**
- Amini, Zheng, Sun & Motee, arXiv:2207.07755 **[ABS]** — Carleman finite-section bounds, *"convergence is indeed exponential with respect to the finite-section order"*, with a constructive rule for truncation length
- Li, Abuduweili, Sun, Chen, Zhao & Liu, arXiv:2411.14321 **[ABS]**
- Shang, Haseli, Cortés & Zheng, arXiv:2602.14537 **[ABS]**
- Fathi et al., arXiv:2310.15386 **[ABS]**; Zhang & Gilpin, arXiv:2505.11349 **[ABS]**, arXiv:2409.15771 **[ABS]**
- Peitz & Klus, *Automatica* 106 (2019) **[ABS]**; Peitz, Otto & Rowley, arXiv:2003.07094 **[ABS]**
- Otto & Rowley, *Annu. Rev. Control Robot. Auton. Syst.* 4:59–87 (2021) **[ABS, secondary extraction]**
- Hjikakou, Cardenas Cartagena & Sabatelli, arXiv:2508.18954 **[FULL]** — reports a *positive* result; recorded against this pass's thesis
- Li, Liu & Yang, arXiv:2503.13093 **[ABS]** — "localized DMD" is temporal
- Sorg, arXiv:2509.16016 **[ABS]**

**Brunton / Kutz orbit [BK]** — three, none carrying a load-bearing critique
- Brunton, Brunton, Proctor & Kutz, *PLOS ONE* 11(2):e0150171 (2016) — the target of §3, not a critic
- Lusch, Kutz & Brunton, *Nature Communications* 9:4950 (2018) — likewise
- Gao, Williams & Kutz, arXiv:2501.13329, *PNAS* 123(16):e2508144123 (2026) **[ABS]** — cannot adjudicate the SINDy/Koopman bet, being one side of it
- Also cited in passing: Kutz, Fu & Brunton (mrDMD), arXiv:1506.00564; Brunton et al. (HAVOK), arXiv:1608.05306; Tu et al., arXiv:1312.0041 — all **[ABS]**

**Could not be reached**
- `10.1016/j.ejcon.2026.100050` **[UNREACHED]** — does not resolve; see Corrections (1)
- Römer & Breitenhuber, PAMM 24(3), doi `10.1002/pamm.202400187` **[CITE]** — Wiley 403; quote unverified
- Govindarajan et al., arXiv:1608.08734 **[ABS]** — body not read
- Sinha, Nandanoori & Yeung, IEEE CDC 2021, doc 9682872 **[CITE]**
- Colbrook, Ayton & Szőke, *JFM*, arXiv:2205.09779 **[CITE]**
- arXiv:2605.15161 (discrete-time immersions) **[CITE]** — the design's actual setting
- PNAS 123(16) body **[CITE]**; Otto & Rowley full text **[CITE]**
- Expander/Ramanujan rewiring quantitative claims **[CITE]** — do not cite the numbers
- Stable-Koopman ODO leads `2607.19719`, `2602.02592` **[UNREACHED]** — leads only
