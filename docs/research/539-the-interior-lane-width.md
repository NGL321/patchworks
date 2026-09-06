# Citation pass: does the interior lane width `m` have a derivation? (patchworks#539)

Part of map [#532](https://github.com/NGL321/patchworks/issues/532). Opened by
[#533](https://github.com/NGL321/patchworks/issues/533)'s resolution at the user's instruction (their
Q10), which moved this map's load from `k` to `m`. Siblings: the principal-angle read,
[#537](https://github.com/NGL321/patchworks/issues/537), and the lever choice,
[#540](https://github.com/NGL321/patchworks/issues/540).

**Citation sequencing.** The design came first and is not on trial. This pass validates after the
fact in [#32](https://github.com/NGL321/patchworks/issues/32)'s idiom — it flags where a source
threatens a claim already made, and says plainly where nothing was found. **It sets no value.** The
lever choice is #540's and the value is downstream of it.

**Registers consulted.** The live match is
[#497](https://github.com/NGL321/patchworks/issues/497) (open-problems), whose `@failure` is this
pass's subject in its own words — *"composed rim-to-apex transport carries exactly one direction […]
no matter how many the maps are held open to carry."* **That last clause is the one this pass
contradicts**, and §1.4 says how. [#439](https://github.com/NGL321/patchworks/issues/439) (the
incoherence cap unenforced where a cell's incidence is pinned) is adjacent and is taken into account
in §1.5 without being duplicated or resolved.

## Reading-depth key

#148's key, used throughout.

- **[FULL]** — paper body read (PDF text or HTML extracted).
- **[ABS]** — authoritative abstract / landing page only.
- **[CITE]** — citation confirmed to exist, text not reached.

---

## Headline verdict, stated plainly

**`m` has no derivation, and never did — but it is no longer true that nothing bounds it. The
governing law exists in closed form, it has been in the literature since 2015, and it was not found
because nobody had identified what kind of object a hop is.**

A hop is `F_out · gain_v · F_inᵀ` with both maps `m × n`, orthonormal-rowed under ADR-0032's band, and
incident on the same relay cell. Writing each map's carried subspace as an orthonormal basis, the hop
is `V_outᵀ V_in` — **the top-left `m × m` sub-block of an `n × n` Haar-orthogonal matrix**, which is
the textbook definition of a **truncated orthogonal matrix**. That object's Lyapunov spectrum is known
exactly (Forrester 2015, eq. 3.3), so this map's mechanism — a product of seven principal-angle cosine
matrices — is not a qualitative argument. It is arithmetic.

Four findings, in order of weight.

1. **The measured collapse is what generic geometry predicts. It needs no pathology and indicts no
   ADR.** At today's `interior_m = 3`, `boundary_m = 4`, `n = 32`, a seven-hop rim-to-apex chain of
   *independent, uniformly random* lanes gives a composed participation ratio of **median 1.011, mean
   1.079** (20,000 trials, §1.2). The record's construction reading is **1.025**. The architecture is
   sitting exactly where a random architecture of these dimensions sits. **The collapse is a
   consequence of the lane widths and the hop count, and of nothing else.**
2. **`m = 3` and `m = 4` are both on the wrong side of the line, and so were the pre-#474 widths.**
   The same simulation at `(interior 4, boundary 8)` gives median **1.082**. Widening the lane back to
   where it was would not have bought composed rank either. To reach a *median* composed
   participation ratio of 2 at `n = 32` over seven hops takes **`m ≈ 12`** (§1.2).
3. **That target collides head-on with the construction invariant, so `m` cannot be the lever.**
   `Σ_e m_e ≤ n − 1` caps the lane at `m ≤ (n − 1)/deg`; at the binding cell — L1 vision at degree 9 —
   that is `m ≤ 3.4`, about **a third of what composed rank needs** at the same `n`. Holding `m` at
   the invariant's own ceiling, only two things reach composed rank 2 at seven hops: **`n` to ~128 at
   today's degree, or the binding cell's degree down to ~3 at today's `n`** (§1.4). Degree is the
   map's own untouched third lever, and the arithmetic promotes it.
4. **The delay-embedding ceiling this ticket was opened to test has no literature standing for a
   *shared* object.** The dimension of an overlap, as a dynamical quantity with its own embedding
   theorem, **does not exist in the literature** (§2). #32's `m` argument, and every inheritance of it
   since, is an analogy — a defensible one, but it must be written as an analogy. It is also, per §1,
   no longer the sharpest thing available.

**The honest null the ticket asked for is real and stands: nothing licenses 4, 3, or any number.** But
"the map is choosing `m` on measurement alone" is now too weak a statement. The map is choosing `m`
against a constraint it can compute, and the current value fails it.

---

## 1. The hop is a truncated orthogonal matrix, and its spectrum is closed-form

### 1.1 Sources, and the identification

- **P. J. Forrester** (2015). *Asymptotics of finite system Lyapunov exponents for some random matrix
  ensembles.* J. Phys. A **48**, 215205. [arXiv:1501.05702](https://arxiv.org/abs/1501.05702).
  **[FULL]**
- **A. Ahn & R. Van Peski**. *Lyapunov exponents for truncated unitary and Ginibre matrices.*
  [arXiv:2109.07375](https://arxiv.org/abs/2109.07375). **[ABS]**
- **Q. Dong** (2025). *Lyapunov exponents for products of truncated orthogonal matrices.*
  [arXiv:2505.04928](https://arxiv.org/abs/2505.04928). **[ABS]**
- **P.-A. Absil, A. Edelman, P. Koev** (2006). *On the largest principal angle between random
  subspaces.* Linear Algebra Appl. **414**(1), 288–294.
  [DOI](https://doi.org/10.1016/j.laa.2005.10.004). **[ABS]** — ScienceDirect returned 403 and the
  preprint PDF would not extract. **Nothing below rests on it.**

Forrester's Proposition 3.1 fixes the ensemble in these words:

> "Let the matrices A_i in (1.1) be chosen as the top d×d sub-block of a (d+n)×(d+n) Haar distributed
> random unitary matrix with real (β=1), complex (β=2) or real quaternion (β=4) entries."

**That is Patchworks' hop.** `benchmarks/alignment_read.py::hop_operator` builds
`M = F_out · gain_v · F_inᵀ`; `gain_v` is a scalar and cannot move any singular-value *ratio*, so the
participation ratio is gain-invariant and the identification is exact. Under ADR-0032's band both maps
have orthonormal rows, so each carries an `m`-subspace of the relay's `n = 32` stalk and `M` is that
pair's matrix of principal-angle cosines — an `m × m` truncation of a `32 × 32` orthogonal matrix, with
`β = 1`, `d = m`, `d + n_F = 32`.

Forrester's eq. (3.3) gives the Lyapunov exponents:

> "μᵢ = (1/2)(Ψ((β/2)(d−i+1)) − Ψ((β/2)(n+d−i+1)))"

which instantiates here as `λᵢ = ½[ψ((m−i+1)/2) − ψ((33−i)/2)]`, and eq. (3.10) gives the variance
`σᵢ² = (1/4N)(Ψ′((β/2)(d−i+1)) − Ψ′((β/2)(n+d−i+1)))`.

**Reproduced independently before use.** `λᵢ` was checked against direct simulation — QR-reorthogonalised
products of 400 hops, 20 replicates, `n = 32` — and agrees to **max absolute error 0.018** at every
`m ∈ {2, 3, 4, 8, 12}`. At `m = 4`: formula `(−1.1591, −1.3359, −1.6258, −2.3014)` against simulated
`(−1.1555, −1.3361, −1.6229, −2.3080)`. The formula is being used because it was verified, not because
it was found.

### 1.2 The sizing table, in the repo's own statistic

Lyapunov exponents are an `L → ∞` statement and this map's question is at `L = 7`, so the table below
is **simulated directly at seven hops** rather than extrapolated. Effective rank is the participation
ratio `(Σσᵢ²)² / Σσᵢ⁴` — `diagnostics.py:234`'s definition, the one `composed_reads` and
`alignment_read.py` report, so the numbers are commensurate with the map's readings. Lanes are
independent and uniformly random; ambient `n = 32`; 20,000 trials.

| `m` | participation ratio, median | mean | p05 | p95 | median σ₂/σ₁ |
|---|---|---|---|---|---|
| 2 | 1.000 | 1.014 | 1.000 | 1.061 | 0.003 |
| **3** | **1.010** | **1.076** | 1.000 | 1.436 | 0.070 |
| **4** | **1.062** | **1.160** | 1.002 | 1.694 | 0.172 |
| 6 | 1.284 | 1.376 | 1.032 | 1.990 | 0.351 |
| 8 | 1.599 | 1.654 | 1.135 | 2.347 | 0.480 |
| 12 | 2.390 | 2.402 | 1.626 | 3.231 | 0.644 |
| 16 | 3.561 | 3.557 | 2.653 | 4.445 | 0.757 |

And at the **actual chain shape** — a rim-to-apex chain enters on a boundary edge and then runs on
interior edges, so hop 1 is `(interior × boundary)` and hops 2–7 are `(interior × interior)`:

| widths (interior, boundary) | median | mean | p05 | p95 | max of 20,000 |
|---|---|---|---|---|---|
| (4, 8) — before #474 | 1.082 | 1.183 | 1.003 | 1.725 | 2.566 |
| **(3, 4) — after #474** | **1.011** | **1.079** | 1.000 | 1.442 | 2.135 |

### 1.3 This reproduces the record, and that is the finding

The map's three collapse readings, against the generic prediction at the widths each was taken at:

| reading | measured | generic prediction |
|---|---|---|
| full dome, **at construction** ([#520](https://github.com/NGL321/patchworks/issues/520) ledger row 6) | **1.025** | median 1.011, mean 1.079 |
| full dome, 100k, both arms ([#524](https://github.com/NGL321/patchworks/issues/524)) | 1.000 ± 0.000 | p05 1.000 |
| max over chains, 100k winner | 1.641 | p95 1.442, max 2.135 |

**T3's readings are at today's widths.** #520 and #524 were opened 2026-09-05 19:24Z; #474's commit
`dabb9ea` landed 2026-09-05 03:40Z. So the comparison above is like for like.

Every row lands inside the generic distribution. **The composed collapse is fully accounted for by
seven hops of width-3 lanes in a 32-dimensional stalk, with no appeal to training, supply, drive,
reward, or any ADR.** This corroborates the map's own position that the collapse is endogenous and
prior to learning, and it sharpens it: the collapse is prior to learning because it is prior to the
*maps* — it is a property of the dimensions alone.

**It also disarms one candidate.** The agent pass that sourced Forrester reported the measured 1.000
as *worse* than generic and inferred a training pathology on top of the width. **That inference does
not survive the change of statistic**: it was computed with spectral entropy, where the repo's number
is the participation ratio. In the repo's own metric the measured values sit inside the generic
distribution and there is no residual to explain. This pass records the disagreement rather than the
conclusion, because the conclusion was the more interesting of the two and it is the one that is wrong.

### 1.4 The collision with the construction invariant — and #497's clause

To reach a median composed participation ratio of 2 at `n = 32` over seven hops takes **`m ≈ 10`**
(§1.2, and the sweep below). The construction invariant `Σ_e m_e ≤ n − 1`, which
`06-graph-topology.md` now owns and from which #474 derived both widths, caps the lane at
`m ≤ (n − 1)/deg`. At the binding cell — **L1 vision at degree 9** — that is `m ≤ 31/9 = 3.4`.

**The width composed rank needs is about 3x the width the invariant permits, at the same `n`.** The
feasible frontier #474 found — `(3,4)`, `(2,5)`, `(1,6)` — lies entirely in the collapsed region of the
table above, and so does every point the invariant admits at `n = 32`. This is not a near miss that a
rung on #14's ladder closes.

**So the honest question is what the invariant's own ceiling on composed rank is.** Below, `m` is held
at the largest value the invariant permits for each `(degree, n)` — the best case available — and the
seven-hop composed participation ratio is read off generic lanes (1,500 trials; the `deg 9, n = 32`
row reproduces §1.2's 1.011 at 1.010, which cross-checks the two samplers):

| degree | `n` | `m = ⌊(n−1)/deg⌋` | median | mean | p95 |
|---|---|---|---|---|---|
| **9** (today's binding cell) | **32** | **3** | **1.010** | 1.077 | 1.434 |
| 9 | 64 | 7 | 1.367 | 1.453 | 2.101 |
| 9 | 128 | 14 | 2.195 | 2.231 | 3.113 |
| 6 | 32 | 5 | 1.172 | 1.274 | 1.881 |
| 4 | 32 | 7 | 1.428 | 1.503 | 2.133 |
| **3** | **32** | **10** | **1.967** | 1.992 | 2.765 |
| 2 | 32 | 15 | 3.256 | 3.239 | 4.112 |

**Two levers reach the same place, and they are not equally priced.** Holding degree at 9 and
quadrupling the stalk to `n = 128` buys 1.010 → 2.195; holding `n = 32` and cutting the binding cell's
degree from 9 to 3 buys 1.010 → 1.967. The first multiplies every restriction map's parameter count by
sixteen and moves a stipulated constant the whole spec is written against; the second is a change to
`06`'s levels. **The arithmetic does not choose between them — but it does say they are the only two
that reach, and that `m` is not a third.**

**This is where #497's `@failure` needs amending.** Its clause *"no matter how many the maps are held
open to carry"* reads as though width were irrelevant. It is not irrelevant — it is decisive, and the
table prices it. What is true is the narrower and more useful statement that **width cannot be moved
far enough on its own while `n` and degree are held**, because the invariant binds it. Amending a
register row is not this pass's to do; it is flagged for a ticket.

**So the lever ordering changes.** `m` is not a lever that can be pulled alone to any useful distance.
The reachable quantities are `n`, **degree**, and hop count `L` — and degree is the map's own
*Not yet specified* entry, *"the third lever, and the only one nobody has touched."* This pass's
contribution to #540 is that the arithmetic now selects that lever rather than leaving it third.

### 1.5 What this pass deliberately does not read

The model assumes the two subspaces at a relay are **independent and uniformly random**. Under
ADR-0032 they are partial isometries, which is the assumption's other half; the untested half is
uniformity. Two things in the record bear on it and **both are #537's to read, not this pass's**:

- **ADR-0010's `c = 2`** bounds `λ_max(Σ_e F_evᵀF_ev) ≤ g_v²·c_v` — a cap on the coherence of a cell's
  incident maps, which is adjacent to the principal angles this mechanism runs on without being the
  same quantity (`c` constrains *top singular directions*; the decay is set by the *whole* subspace
  pair). Both extremes are bad in a way worth stating: subspaces pushed fully apart decay fastest,
  and subspaces made fully coherent share a single direction and compose to rank 1 outright. Generic
  is between them, and generic is what §1.2 measures.
- **#439** reads the actuator's three pinned incident maps at **99.6% of the fully-coherent ceiling
  `g_v²·deg`** — the second extreme, at a cell the projection cannot reach.

`benchmarks/alignment_read.py` already reports per-hop operator effective rank (median **2.226**, p05
1.433, p95 3.637, σ₁² share 0.591 over all 7,122 directed hops at construction) and it reads
`dome.edges[edge_in].m` live, so it **re-runs at today's widths without modification**. That reading is
at the pre-#474 widths and has aged with `main` (#455). Handing it to #537 rather than re-running it
here.

---

## 2. The delay-embedding ceiling has no standing for a *shared* object

### Sources

- **T. Sauer, J. A. Yorke, M. Casdagli** (1991). *Embedology.* J. Stat. Phys. **65**, 579–616.
  [DOI](https://doi.org/10.1007/BF01053745). **[CITE]** — Springer paywall, redirected to an IdP
  login. **Full text not reached in this pass.**
- **J. Stark** (1999). *Delay Embeddings for Forced Systems. I. Deterministic Forcing.* J. Nonlinear
  Sci. **9**, 255–332. [DOI](https://doi.org/10.1007/s003329900072). **[CITE]** — Springer paywall.
  **Full text not reached.**
- **G. Sugihara et al.** (2012). *Detecting Causality in Complex Ecosystems.* Science **338**, 496–500.
  [DOI](https://doi.org/10.1126/science.1227079). **[ABS]**
- **P. Mattila**. *A survey on the Hausdorff dimension of intersections.*
  [arXiv:2301.13478](https://arxiv.org/abs/2301.13478). **[ABS]**

### Findings

**The explicit null, and it is the answer to the ticket's question 1.** Nothing was found defining a
box-counting, fractal or intrinsic dimension of the *shared* part of two coupled attractors as a
distinct quantity with its own embedding theorem. Stark's forced-system theorems extend Takens to
non-autonomous systems — bibliographically, they prove *"two versions of Takens Theorem relevant to
forced systems: one applicable to the case where the forcing is unknown, and the other to the
situation where the forcing is known"* — but the dimension budget in every version is the box-counting
dimension of the **whole** reconstructed object. Convergent cross mapping likewise sizes its embedding
dimension against the full attractor; there is no cross-map dimension condition stated in terms of an
intersection.

The one genuine body of mathematics for "dimension of an overlap" is the **generic intersection
formula**, and it is geometric measure theory rather than dynamics: for Borel sets with
`dim A + dim B > n` the expected answer is `max(dim A + dim B − n, 0)`, which per Mattila *"happens
when smooth surfaces meet in a general position"* and is *"only known if one of the sets has dimension
bigger than (n+1)/2."* It does not transfer, for two reasons: the two cells' latent objects are **not**
in general position in a common ambient — they are coupled through the very lane being sized, which is
precisely the excluded non-generic case — and the theorem concerns Hausdorff dimension of a set
intersection, not the information a channel carries between coupled subsystems.

**What this costs the record.** #32 wrote that an interior lane of 4 *"supported a shared piece of box
dimension under 2"*, and `06-graph-topology.md`, the architecture register and `graph.py:206` all
inherit it. The Takens/Sauer machinery it borrows is real and the reasoning is defensible; what it is
**not** is a cited result about a shared object. It should be written as an explicit analogy. This
does not make it wrong, and §1 supersedes it as a bound in any case.

---

## 3. The over-squashing literature yields no width

### Sources

- **U. Alon & E. Yahav** (2021). *On the Bottleneck of Graph Neural Networks and its Practical
  Implications.* ICLR. [arXiv:2006.05205](https://arxiv.org/abs/2006.05205). **[ABS]**
- **J. Topping, F. Di Giovanni et al.** (2022). *Understanding over-squashing and bottlenecks on
  graphs via curvature.* ICLR. [arXiv:2111.14522](https://arxiv.org/abs/2111.14522). **[ABS]**
- **F. Di Giovanni et al.** (2023). *On Over-Squashing in MPNNs: The Impact of Width, Depth, and
  Topology.* ICML, [PMLR v202](https://proceedings.mlr.press/v202/di-giovanni23a.html).
  [arXiv:2302.02941](https://arxiv.org/abs/2302.02941). **[ABS]**
- **M. Black, Z. Wan, A. Nayyeri, Y. Wang** (2023). *Understanding Oversquashing in GNNs through the
  Lens of Effective Resistance.* ICML. [arXiv:2302.06835](https://arxiv.org/abs/2302.06835). **[ABS]**

### Findings

Alon & Yahav state the phenomenon without a quantity — *"This bottleneck causes the over-squashing of
exponentially growing information into fixed-size vectors"* — and propose no width. Topping et al.
locate the cause in topology: *"we introduce a new edge-based combinatorial curvature and prove that
negatively curved edges are responsible for the over-squashing issue"*, with **rewiring** as the
remedy. Black et al. likewise *"propose to use total effective resistance as a bound of the total
amount of oversquashing."*

Di Giovanni et al. is the one paper that studies width, and its verdict runs against using it:

> "(i) Neural network width can mitigate over-squashing, but at the cost of making the whole network
> more sensitive; (ii) Conversely, depth cannot help mitigate over-squashing: increasing the number of
> layers leads to over-squashing being dominated by vanishing gradients; (iii) The graph topology
> plays the greatest role."

Its Theorem 4.1 bounds sensitivity at distance `r` by `C_k γ_{r+k}(v,u) (2c_σ w p / d_min)^r`, and the
authors are explicit that width acts globally rather than where the problem is: taking larger `c_σ, w,
p` *"affects the model globally and does not target the sensitivity of specific node pairs induced by
the topology."*

**Explicit null.** Nothing in this literature calibrates a per-edge channel width against depth or
against composed fidelity. These are **upper bounds on a Jacobian norm with unpinned constants**, not
design formulas, and they admit no inverse of the form "given `L = 7`, take `m ≥ …`". The field's
proposed fix is uniformly graph rewiring — which is to say, **degree and topology**, independently
arriving at §1.4's conclusion from a different direction.

One thing worth carrying: this literature bounds a *Jacobian norm*, where §1 gives an *exact spectrum*.
Where the two speak to the same question, §1 is the stronger instrument and should be preferred.

---

## 4. Does anything vindicate 4, 3, or any number?

**Clean null on the direct question, and it is worth stating without hedging.** No source in any of
these literatures names a per-edge width, endorses 3 or 4, or forbids one. Over-squashing work treats
width as a free hyperparameter and aims its remedies at topology. Embedding theory sizes a
reconstruction dimension for a whole attractor, never a channel. The random matrix literature studies
the truncation ratio without ever recommending one. **`m` is chosen, not derived, and the record's own
architecture register already types both `DomeSpec.interior_m` and `DomeSpec.boundary_m` as
`stipulated`.** That typing is correct and this pass does not disturb it.

**But the null is narrower than "nothing bounds it."** Forrester (3.3) is a *conditional* derivation:
fix the ambient `n`, the chain length `L` and a target composed rank, and it returns the `m` that
delivers it. That is not the literature blessing a number — it is the literature converting this map's
own targets into one. Any claim that `m = 3` is licensed by the literature is false; any claim that `m`
is unconstrained by it is now also false.

**Two things the current value genuinely has going for it**, neither of which was found countered:

- **`boundary_m` is defensible on exactly the argument that condemns `interior_m`.** A boundary lane
  sees far fewer hops — a patch cell's edges are the only route its information takes, and that route
  is short. The seven-hop decay law is a statement about composition depth, and it does not bite on a
  lane that is composed once or twice.
- **Nothing forbids a narrow lane if the composed operator is *meant* to carry one direction.** If
  interior lanes are intended as single-channel relays, `m = 3` at composed rank 1.01 is doing exactly
  what it says. **That is a design question the literature cannot answer, and it is the question these
  numbers push toward #540** — which is why this pass sets no value.

---

## 5. Three places the record is stale, and one it is wrong

Flagged, not edited — a citation pass does not amend a spec, an ADR or a register.

1. **ADR-0004 lines 145 and 190 still read `m = 4`**, and line 190 still quotes #132's *"~1.4x"*
   margin, both superseded by #474's move to `interior_m = 3`.
2. **`06-graph-topology.md`, `graph.py:206` and `docs/registers/architecture.md` all say
   [#440](https://github.com/NGL321/patchworks/issues/440) *"left open whether the piece has a box
   dimension at all."* #440 closed ruling that it does** — the piece is *"the limit set of a
   discretely-driven recurrence […] filling roughly `log 97 / log(1/r)` dimensions at retention `r`
   […] neither a finite set nor a manifold."*
3. **#474 closed against an escape hatch that had already shut.** #440 closed 2026-09-04 19:34Z;
   `dabb9ea` landed 2026-09-05 03:40Z, about eight hours later, still citing #440 as having *re-opened*
   the question. The ruling that narrowed the lane leaned on a get-out that no longer existed. Nothing
   in §1 depends on this — the truncated-orthogonal bound is indifferent to the piece's dimension — but
   the record should not keep saying it.
4. **The delay-embedding margin is worse than anywhere in the record admits, on the record's own
   numbers.** #132 measured `d_corr` quartiles **1.26 / 1.43 / 1.53** over the 25 of 52 L1 vision cells
   with a certified scaling region, and tabled the interior lane's margin at ~1.4x *at `m = 4`*. At
   `m = 3` the ceiling is box dimension **< 1.5**, which is **below the measured upper quartile of
   1.53** and about **1.05x** the median. Nothing in the repo compares those two numbers. Two caveats
   keep it from being a contradiction, and both are real: `m` carries the **overlap**, whose dimension
   is bounded above by the piece's (#132's own note), and §2 has just found that the overlap's
   dimension is not a quantity the literature defines. ADR-0004's pre-registered read — `d_box` of a
   heard L1 cell's driven chart limit set — **is still not taken.**

---

## 6. What this pass could not reach

Stated plainly rather than papered over.

- **Sauer–Yorke–Casdagli, *Embedology* (1991)** — Springer paywall, full text not read.
- **Stark, *Delay Embeddings for Forced Systems* (1999)** — Springer paywall, full text not read.
- **Absil–Edelman–Koev (2006)** — ScienceDirect 403; the preprint PDF would not text-extract. Its
  result is in any case the *largest* principal angle, the wrong tail for this question.

**No conclusion in §1 depends on any of these three.** §1 rests on Forrester
[arXiv:1501.05702](https://arxiv.org/abs/1501.05702), read in full and reproduced numerically before
use. §2's finding is a null, and a null is not weakened by a paywall — but it is fair to say that the
two unread papers are the two most likely places a shared-overlap dimension result would hide, and
this pass did not eliminate them.

---

## 7. Sources

| Source | Depth | Access |
| --- | --- | --- |
| Forrester (2015), *Asymptotics of finite system Lyapunov exponents*, J. Phys. A 48, 215205, [arXiv:1501.05702](https://arxiv.org/abs/1501.05702) | [FULL] | ar5iv; Prop. 3.1, eqs. (3.3) and (3.10) quoted; **reproduced by simulation to 0.018** |
| Ahn & Van Peski, *Lyapunov exponents for truncated unitary and Ginibre matrices*, [arXiv:2109.07375](https://arxiv.org/abs/2109.07375) | [ABS] | not needed — finite-`L` regime is Forrester's |
| Dong (2025), *Lyapunov exponents for products of truncated orthogonal matrices*, [arXiv:2505.04928](https://arxiv.org/abs/2505.04928) | [ABS] | landing page |
| Absil, Edelman & Koev (2006), Linear Algebra Appl. 414(1) 288–294 | [ABS] | **ScienceDirect 403**; preprint would not extract; nothing rests on it |
| Sauer, Yorke & Casdagli (1991), *Embedology*, J. Stat. Phys. 65, 579–616 | [CITE] | **Springer paywall** |
| Stark (1999), *Delay Embeddings for Forced Systems I*, J. Nonlinear Sci. 9, 255–332 | [CITE] | **Springer paywall** |
| Sugihara et al. (2012), *Detecting Causality in Complex Ecosystems*, Science 338, 496–500 | [ABS] | landing page |
| Mattila, *A survey on the Hausdorff dimension of intersections*, [arXiv:2301.13478](https://arxiv.org/abs/2301.13478) | [ABS] | arXiv abstract |
| Alon & Yahav (2021), ICLR, [arXiv:2006.05205](https://arxiv.org/abs/2006.05205) | [ABS] | abstract verbatim |
| Topping, Di Giovanni et al. (2022), ICLR, [arXiv:2111.14522](https://arxiv.org/abs/2111.14522) | [ABS] | abstract verbatim |
| Di Giovanni et al. (2023), ICML, [arXiv:2302.02941](https://arxiv.org/abs/2302.02941) | [ABS] | abstract + PMLR page; Thm 4.1 statement |
| Black, Wan, Nayyeri & Wang (2023), ICML, [arXiv:2302.06835](https://arxiv.org/abs/2302.06835) | [ABS] | abstract verbatim |

## 8. Reproducing the numbers

Every simulated figure in §1 is generic geometry — it involves no dome, no spec and no checkpoint, so
it needs no container and ages with nothing. Three scripts, `numpy` and `scipy` only:

- **Forrester (3.3) against simulation** — QR-reorthogonalised products of 400 hops, 20 replicates,
  `n = 32`, `m ∈ {2,3,4,8,12}`. Max absolute error 0.018.
- **The `m` sweep at seven hops** — 20,000 trials per width, participation ratio
  `(Σσᵢ²)²/Σσᵢ⁴`, `n = 32`. Reproduced at 3,000 trials with agreement to 0.01 on every median.
- **The rim-to-apex chain shape** — 20,000 trials, hop 1 `(interior × boundary)`, hops 2–7
  `(interior × interior)`, at `(4, 8)` and `(3, 4)`.
- **The invariant sweep** — `m = ⌊(n−1)/deg⌋` for each `(degree, n)`, 1,500 trials, drawn with the
  cheaper sampler (QR of an `n × m` Gaussian, a uniform `m`-frame) rather than truncating a full
  `n × n` Haar draw.

**The sweep was run twice, by two samplers, and the tabled numbers are the weaker of the two runs.**
The independent replication — full `n × n` Haar truncation, 4,000 trials — returns `deg 9, n = 32`
**1.009** against the tabled 1.010, `deg 9, n = 64` **1.368** against 1.367, `deg 9, n = 128` **2.230**
against 2.195, `deg 3, n = 32` **1.975** against 1.967, and `deg 2, n = 32` **3.232** against 3.256.
Every row agrees inside Monte-Carlo noise and **no conclusion in §1.4 moves**; the largest discrepancy,
0.035 at `deg 9, n = 128`, is in the row furthest from any claim this pass makes. The `deg 9, n = 32`
row also reproduces §1.2's independently-computed 1.011, so the three ways of getting at today's
configuration agree to 0.002.

The scripts are short enough to restate from the definitions above and were not committed, since the
repo keeps rigs under `benchmarks/` for surfaces that read the dome and these read nothing. **If any of
these numbers is to be leaned on for a decision, it should be re-derived rather than trusted from
here** — that is the same standard #533 applied to R-A, and it is the standard this pass applied to the
agent report it was given.

## Context

Written for [#539](https://github.com/NGL321/patchworks/issues/539) on branch
`research/interior-lane-width`. Repo documents read before searching: `docs/spec/06-graph-topology.md`,
`docs/spec/05-timescales.md`, ADR-0004, ADR-0010, ADR-0032, `docs/research/032`, `docs/research/244`,
`docs/registers/architecture.md`, `src/patchworks/graph.py`, `src/patchworks/diagnostics.py`,
`benchmarks/alignment_read.py`, and issues #32, #132, #440, #474, #497, #533. Nothing in the spec, the
ADRs or the registers is edited by this pass; §5's four items are handed on as flags.
