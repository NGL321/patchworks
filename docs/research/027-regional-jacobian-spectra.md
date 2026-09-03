# Constructing a body with spread regional Jacobian spectra (patchworks#27)

An **inverted citation-sequencing pass**, run *ahead* of the decision it serves rather than after it
— precedent [patchworks#13](https://github.com/NGL321/patchworks/issues/13). The decision waiting on
it is [#7](https://github.com/NGL321/patchworks/issues/7) / `docs/spec/05-timescales.md` /
[ADR-0005](../adr/0005-timescale-is-persistence-not-a-schedule.md), which makes a cell's effective
timescale the spectral radius of the Jacobian of the activation region its biases select.

Terms used here are defined in [`CONTEXT.md`](../../CONTEXT.md).

> **How this pass ran.** Its first agent was killed by an API spend limit about forty steps in,
> having written no file; a second session reconstructed its trail from the transcript as an
> explicitly incomplete partial. This document completes it. Sources the first agent read are marked
> **[verified at source]**; everything added since is verified the same way or marked otherwise. The
> paper corpus is cached at
> `~/.claude/projects/-Users-angl-Documents-patchworks/research-cache-027/`.
>
> One thing the earlier passes could not do, and this one did: **the go/no-go rig `05-timescales.md`
> asks for was built and run** (`prototypes/regional-spectra/spread_pilot.py`). Its numbers are
> reported in §6 and are load-bearing for the verdict. They are a *pilot on a stand-in body*, not a
> measurement of the body — see §6's caveats before quoting them.

## Amendment, 2026-09-03: the mechanism this pass measured is retired, and the numbers are re-run

Added on the resolution of [#349](https://github.com/NGL321/patchworks/issues/349), which was opened
because this document's own caveat — *"the body is a **stand-in** … This establishes the shape of the
answer and the sensitivity of the rig, not the body's number"* — was never discharged, while its
**7.7×** was quoted onward as operative headroom. `docs/research/032` already carried a Koopman-era
amendment; this one carried none.

**Everything below §6 is a measurement of a mechanism the design no longer uses.** The Koopman
conversion ([#138](https://github.com/NGL321/patchworks/issues/138)) made `step` a **linear `K`**, and
[ADR-0028](../adr/0028-a-cell-holds-a-spectrum-of-retention-constants.md) moved timescale onto
`λ(K)`. This pass measured the spectral radius of a *regional* Jacobian of a piecewise-linear ReLU
`step` — and a linear `K` **has one region globally**. The object §6 sweeps does not exist in the
converted design, so `spread_pilot.py` cannot be re-pointed at the body; it can only be replaced.

**The re-run was nonetheless possible, and it was done.**
`prototypes/regional-spectra/converted_spread.py` (#349) is the replacement. What ADR-0028 split into
three separately named quantities is measured in **this document's own spread statistic** — `τ` p95/p05
with `τ = −1/ln ρ`, over the cells admitting a finite `τ`, alongside `sd(log₁₀ ρ)` — so the comparison
against 7.7× is like-for-like rather than a change of units. The two realised readings come from
[#274](https://github.com/NGL321/patchworks/issues/274)'s nine driven seeds
(`prototypes/driven-rho-274/`); `ρ(K)` is read off driven runs here, because #274 did not record it.

| quantity (ADR-0028's rows) | `τ` p95/p05, across seeds | `τ` median | `sd(log₁₀ ρ)` | expansive |
|---|---|---|---|---|
| `ρ(K)` — the **operator's retention**, per-cell, learned | **9.0–18.0** (med 10.3) | 19–26 ticks | 0.023–0.073 | 0/150 |
| `ρ(K · J_chart)` — the **realised chart retention** (#206's object) | **1.8–2.8** (med 2.1) | 1.0–1.3 ticks | 0.077–0.107 | 0/150 |
| `ρ(K · (J_chart + relay))` — the **full chart loop** (#274's correction) | **9.6–49.6** (med 12.9) | 2.9–7.6 ticks | 0.063–0.089 | 0–33/150 |

`ρ(K)` at 5 seeds × 2,000 ticks; the realised readings at 9 seeds, one to 100,000 ticks.

**Four things follow, and the first is the one that matters for anyone quoting this document.**

1. **7.7× is not a number about the body, and there is no single successor to it.** It fragments by
   which of the three quantities is meant. The nearest object to what §6 actually measured — the
   *realised* per-tick retention of the chart, region included — reads about **2×**, not 7.7×: the
   stand-in overstated the realised spread by roughly a factor of four. **Any citation of 7.7× as
   headroom, reachable spread, or "what the body gives" is withdrawn.**
2. **The headroom claim survives, but at the operator and not at the region.** `ρ(K)` spreads
   9–18×, which brackets 7.7× — so a spread of that order is real in the converted design. It is a
   property of a *learned per-cell operator*, which is precisely what this pass's §7 said the bias
   mechanism could not deliver, and what ADR-0028's *Attributability* consequence books.
3. **§6's headline — "spread and stability are the same knob" — is a fact about the bias mechanism
   and does not transfer.** Every configuration here with a usable ratio put 5–39% of cells past
   `ρ ≥ 1`; `ρ(K)` shows **no expansive cell at any seed**, because
   [ADR-0015](../adr/0015-the-cell-operator-band-is-on-the-spectral-norm.md)'s band on `σ_max(K)`
   bounds the radius for free. The conflict §7.3 recorded is dissolved by the conversion rather than
   resolved by a construction choice. (It was already once reframed, on the stand-in, by
   [#42](https://github.com/NGL321/patchworks/issues/42) — see below.)
4. **The retention the bias mechanism could not reach is now reached.** §6's `τ` medians sat under
   two ticks; `ρ(K)` gives 19–26. ADR-0028 stated the bar — *"at the measured `τ ≈ 1` tick an
   assertion cannot stand at all"* — and this is the measurement that it clears.

**Reconciling [#42](https://github.com/NGL321/patchworks/issues/42)'s 4.5× and 16× against this
document's 7.7×: they measure different things, and all three are now superseded.** #42 ran
`selection_sweep.py` on **the same stand-in body**, so its numbers are not a re-run of anything here.
It changed two things: the population (400 candidate bias draws, not 150 occupied cells) and the
construction (**draw-then-select** — 4.5× as drawn, 16× after selecting a covering set). So 7.7×,
4.5× and 16× are one stand-in read three ways, not three readings of the body. **The construction
that produced 16× is itself retired:** ADR-0028 leaves the gradient to learning and places no `τ`,
and [#276](https://github.com/NGL321/patchworks/issues/276) found `bias_selection.select()` was never
invoked in any graph that ran. Nothing downstream should cite 4.5× or 16× either.

**One internal inconsistency in §6, recorded rather than silently fixed.** Its `ρ` p05/med/p95 of
`0.32 / 0.57 / 0.98` and `τ` of `0.86 / 1.76 / 5.49` are **not the same population**:
`spread_pilot.py` takes `τ` over `rho[rho < 1.0]` and its `ρ` quantiles over every cell. `−1/ln(0.98)`
is 49.5, not 5.49 — the `τ` p95 is the 95th percentile of the *contracting* cells. The figures are
each correct and the pairing is misleading. The re-run keeps the same convention deliberately, and
`tests/test_converted_spread.py` pins it, so the comparison is not a change of population dressed as
a change of body.

**One standing caveat travels with every `τ` in the table, and it is not this ticket's to discharge.**
The cheap spectral reading of retention is licensed only where a cell dwells in one activation region
long enough for its operator's rate to express — `dwell > τ`, ADR-0005's precondition re-pointed by
[#226](https://github.com/NGL321/patchworks/issues/226). That licence is not enforced anywhere:
[#344](https://github.com/NGL321/patchworks/issues/344) measures the median `dwell/τ` at **2.00**,
with **57 of 150 cells breaching** it on the corrected full-loop operator. So the realised readings
above are quoted subject to that open problem, and the `ρ(K)` row is the one least exposed to it —
`K` is region-independent, which is exactly ADR-0028's *dwell is demoted from existence to fidelity*.

**What is *not* withdrawn.** §1–§5's citation work stands entirely — it is about the literature, not
about the body, and #167 and ADR-0028 both build on it. §7.4's argument against reporting `ρ` alone
survives the conversion and is if anything strengthened by the `ρ(K)`-versus-realised gap in the
table above. And **ADR-0028's citation of this pass for the *attributability* argument is a different
use and survives**: that argument needs only §6's finding that bias and operating point were
statistically indistinguishable, which is a true statement about the mechanism that was replaced.

## The question, restated

One shared frozen piecewise-linear cell body. Weights fix fold *directions*; per-cell biases fix fold
*offsets*, so each cell sits in a different activation region of the same map, with its own local
Jacobian and its own spectrum. That differentiates cells only if the **distribution of regional
Jacobian spectral radii, across the bias settings cells actually occupy, has real spread**. If it is
a spike, every cell lands in the same dynamic regime and #7's mechanism is dead on arrival.

The ticket's five sub-questions are answered in §1–§5; §6 is the measurement; §7 is the verdict.

---

## 1. Is it a solved problem? — Not as posed, and nobody has tried to *widen* this distribution

**[verified at source]** Schoenholz, Gilmer, Ganguli & Sohl-Dickstein, *Deep Information Propagation*
([arXiv:1611.01232](https://arxiv.org/abs/1611.01232)). The order-to-chaos control parameter is
`χ₁ = σ_w² ∫ 𝒟z [ϕ'(√q* z)]²` (Eq. 5), with the fixed point `q*` satisfying
`q* = σ_w² ∫ 𝒟z φ²(√q* z) + σ_b²`, and the critical line `χ₁ = 1` drawn in the **(σ_w², σ_b²)
plane**. `σ_b²` does not enter `χ₁` directly, but it enters `q*` additively and `χ₁` depends on `q*`
— so **bias variance shifts the local dynamical regime**, weights held fixed. Structurally that is
this ticket's claim, and it is the closest thing in the literature to a positive answer.

**[verified at source]** Pennington, Schoenholz & Ganguli, *Resurrecting the sigmoid…*
([arXiv:1711.04735](https://arxiv.org/abs/1711.04735)): "controlling the entire distribution of
Jacobian singular values is an important design consideration in deep learning." And
*The Emergence of Spectral Universality in Deep Networks*
([arXiv:1802.09979](https://arxiv.org/abs/1802.09979)): free probability gives "a detailed analytic
understanding of how a deep network's Jacobian spectrum depends on various hyperparameters including
the nonlinearity, the weight **and bias** distributions, and the depth."

**But the objective is inverted, and that is the finding.** This entire literature exists to achieve
**dynamical isometry** — to *concentrate* the Jacobian spectrum around one. Patchworks wants the
opposite: deliberate spread. Nothing found in this pass shows anyone attempting to widen an
across-region spectral distribution on purpose. The analytic machinery transfers; the design goal
does not, and there is no recipe to copy.

**Answer to sub-question 1: no, it is not a solved problem in the form asked.** What exists is (i)
theory that says the knobs are real and (ii) two published instances of spreading timescales across
units by construction, in architectures where the bias→timescale route is analytic rather than
regional:

- **Chrono initialization.** **[verified at source]** Tallec & Ollivier, *Can recurrent neural
  networks warp time?* ([arXiv:1804.11188](https://arxiv.org/abs/1804.11188), ICLR 2018), Eq. 16:
  `b_f ∼ log(𝒰([1, T_max − 1]))`, `b_i = −b_f`, with `T_max` "the expected range of long-term
  dependencies to be captured". Gate biases spread **log-uniformly over a task-set range**.
- **S4D's timescale parameter.** **[verified at source]** Gu, Gupta, Goel & Ré,
  *On the Parameterization and Initialization of Diagonal State Space Models*
  ([arXiv:2206.11893](https://arxiv.org/abs/2206.11893)), Listing 1:
  `log_dt = np.random.rand() * (np.log(dt_max) - np.log(dt_min)) + np.log(dt_min)` with
  `dt_min=1e-3, dt_max=1e-1`, commented "**Geometrically uniform timescale**". Confirms what the
  earlier pass flagged **unverified — the claim is now verified at source and citable**, in S4D
  rather than in S4.

Two independent modern architectures spread per-unit timescales **log-uniformly over a range set by
the task horizon**. If #7's spread has to be imposed rather than inherited, that is the published
shape to impose.

**The transfer is not free.** Both set a parameter whose map to a time constant is analytic — a
gate's sigmoid *is* a retention coefficient, `Δ` *is* a step size. Patchworks has no gate: its route
from bias to time constant runs through *which activation region the cell lands in*. That step is the
unestablished one, and §6 is the first evidence on it.

## 2. Is it provably hard? — Hardness attaches to *exact* control only

**[verified at source]** Scaman & Virmaux, *Lipschitz regularity of deep neural networks*
([arXiv:1805.10965](https://arxiv.org/abs/1805.10965)): "even for two layer neural networks, the
exact computation of this quantity is NP-hard."

**[verified at source]** Jordan & Dimakis, *Exactly Computing the Local Lipschitz Constant of ReLU
Networks* ([arXiv:2003.01219](https://arxiv.org/abs/2003.01219)) sharpens it: they "establish strong
inapproximability results showing that it is hard to even approximate Lipschitz constants of
scalar-valued ReLU networks, for `ℓ₁` and `ℓ∞` norms", then give LipMIP, a mixed-integer program that
computes the quantity **exactly** and does "not run in polynomial time in the worst-case."

**[verified at source]** Fazlyab et al. ([arXiv:1906.04893](https://arxiv.org/abs/1906.04893))
supplies the tractable side: an SDP yielding "guaranteed upper bounds."

**Answer to sub-question 2: the hardness does not bind here, for two independent reasons.** First,
it is about *exact* computation of a global constant, while #7 needs the *distribution* of a local
quantity. Second, Patchworks' regional Jacobian is a **12 × 12 matrix** (`k = 12`,
`06-graph-topology.md`): its spectrum is an exact `eigvals` call costing microseconds. The
intractability results are about networks whose region structure cannot be enumerated; they say
nothing against sampling regions and reading each one's spectrum exactly, which is what the rig in
§6 does.

## 3. Statistical vs exact — statistical shaping is available, and it is what #7 needs

**Answer to sub-question 3: yes, in the affirmative, and this is the pass's cheapest firm result.**
The literature supports shaping the distribution statistically — via `σ_w²`, `σ_b²`, width and depth
(Schoenholz; Pennington; Hanin below) — even where per-region construction is intractable (§2). #7
asks for spread, not a designed spectrum per region, so the statistical route is sufficient.

**[verified at source]** Hanin & Nica, *Products of Many Large Random Matrices…*
([arXiv:1812.05994](https://arxiv.org/abs/1812.05994)) says what shape to expect: "the norm of the
vector `M^(d) u` is approximately **log-normal** distributed." The natural summary statistic for the
spread is therefore a spread of `log ρ`, not of `ρ` — which also matches chrono and S4D both being
log-uniform. **Report the distribution in log space.**

## 4. Does architecture matter? — Yes, and narrowness is the lever

**[verified at source]** Hanin, *Which Neural Net Architectures Give Rise to Exploding and Vanishing
Gradients?* ([arXiv:1801.03744](https://arxiv.org/abs/1801.03744)): the empirical variance of the
squares of the input-output Jacobian entries is **exponential in `β = Σ_j 1/n_j`**, the sum of
reciprocals of the hidden-layer widths. "When `β` is large, the gradients computed by `N` at
initialization vary wildly," and the paper's own advice is to *minimize* `β` — equal, wide layers.

**Patchworks is narrow by construction** (`k = 12`, `n = 32`), so `β` is large, so the Jacobian
statistics fluctuate wildly. **What that literature calls a pathology is, for #7, exactly the spread
the mechanism needs.** The low-dimensionality constraint buys the dispersion for free. §6 measures
this and it holds.

**[verified at source]** Pennington et al. (1711.04735) adds one hard architectural constraint:
ReLU networks "are incapable of dynamical isometry," while sigmoid networks achieve it under
orthogonal initialization. Read in the inverted direction this is *favourable*: the activation that
cannot be made isometric is the one that cannot suppress spread.

**Orthogonal / unitary RNN parameterisations point the wrong way, with one exception.** uRNN, scoRNN,
expRNN and AntisymmetricRNN all exist to *pin* the transition spectrum to the unit circle — the
isometry objective again, and the opposite of what is wanted. The exception is worth naming:
**[verified at source, abstract]** Kerg et al., *Non-normal Recurrent Neural Network (nnRNN)*
([arXiv:1905.12080](https://arxiv.org/abs/1905.12080)), which argues orthogonal constraints come "at
the cost of reduced expressivity due to the limited variety of orthogonal transformations" and
deliberately uses **non-normal** matrices to gain "transient dynamics" while keeping stability. That
is the nearest published statement of Patchworks' actual preference: keep the spectrum controlled,
but do not make the operator normal. §6 finds the body's regional Jacobians are non-normal by a
factor of ~2.6, which per nnRNN is a feature, and per §5 is a measurement hazard.

## 5. What the empirical check should be — naive sampling is fine; the statistic is what needs care

The earlier pass died mid-search on stochastic Lanczos quadrature. **The answer is that it does not
apply.** SLQ (Ubaru, Chen & Saad, *Fast estimation of tr(f(A)) via stochastic Lanczos quadrature*,
SIMAX 38(4), 2017; analysed by [Chen, Trogdon & Ubaru, ICML 2021](https://proceedings.mlr.press/v139/chen21s.html))
estimates the cumulative empirical spectral measure of a **large** symmetric matrix under matrix-vector
access, with cost scaling in the dimension `n`. Patchworks' regional Jacobian is `12 × 12` and
explicitly available, and it is **not symmetric**. Exact eigendecomposition is free. There is no
better-conditioned estimator to reach for at the matrix level.

**The variance that matters is in the sampling over regions, and it is small.** §6 measures it:
**50 sampled cells already pin the spread statistic to ±0.011 in `sd(log₁₀ ρ)`, and 150 to ±0.006.**
The go/no-go run `05-timescales.md` specifies is adequately powered at the population size the
architecture already has. **Answer to sub-question 5: no better estimator is needed. Four changes to
the protocol are.**

1. **Report `τ = −1/ln ρ` in quantiles, not `ρ` in moments.** `τ` is the quantity the mechanism cares
   about and it diverges as `ρ → 1`, so means and standard deviations of the spread are dominated by
   the tail. §6 shows `sd(log₁₀ ρ)` behaving erratically (0.98 in one narrow configuration) purely
   from a heavy left tail of near-zero radii. Quantile ratios are stable where moments are not.
2. **Sample the operating point as well as the bias.** §6's control finds the operating point
   contributes *as much* spread as the bias does. A sweep that varies biases at a fixed chart and
   stalk measures roughly half the phenomenon and attributes all of it to the wrong cause.
3. **Cross-check `ρ` against the actual decay.** These Jacobians are non-normal: §6 measures
   `σ_max/ρ ≈ 2.6`, so a cell with `ρ = 0.5` still amplifies some perturbation directions on the
   first tick. The `ρ`-only reading is the error Yildiz, Jaeger & Kiebel warn about
   (*Re-visiting the echo state property*, Neural Networks 35, 2012, **[verified at source]**):
   spectral radius below unity is **not sufficient** for the echo state property. Pair `ρ` with
   `‖J^t‖` at the horizon of interest; the standard tooling for the gap is pseudospectra and the
   Kreiss constant (Trefethen & Embree, *Spectra and Pseudospectra*, Princeton, 2005).
4. **Measure a run, not only a Jacobian.** See §7's first defect: the one-step regional Jacobian is
   only a cell's timescale if the cell stays in its region.

### Region counts are not the binding constraint

The earlier partial flagged region-counting as the strongest negative evidence found. **On this
pass's reading it does not bind.**

**[verified at source]** Hanin & Rolnick, *Complexity of Linear Regions in Deep Networks*
([arXiv:1901.09021](https://arxiv.org/abs/1901.09021), ICML 2019): "the average number of regions
along any one-dimensional subspace grows **linearly** in the total number of neurons, far below the
exponential upper bound," the average distance to the nearest region boundary scales like the inverse
of the number of neurons, and both stay "roughly constant during training." Companion result,
**[verified at source]** Hanin & Rolnick, *Deep ReLU Networks Have Surprisingly Few Activation
Patterns* ([arXiv:1906.00904](https://arxiv.org/abs/1906.00904)): in a 2-D cross-section the count
"starts at approximately `(#neurons)²/2`", is "independent of the depth", and "changes little during
training." Both are far below the exponential-in-depth maximum of Montúfar, Pascanu, Cho & Bengio
([arXiv:1402.1869](https://arxiv.org/abs/1402.1869)), whose bound rests on depth letting a network
"re-use pieces of computation exponentially often" (**[verified at source, abstract]**; the exact
formula was not read).

**Why it does not bind: Patchworks needs ~150 distinct regions, not many.** A body with even a
hundred hidden units offers thousands of regions on the pessimistic count. The count was never the
scarce resource; the **dispersion of `ρ` across the occupied regions** is, and that is a different
quantity, governed by §4's `β` rather than by any region count. §6 confirms the separation directly:
every sampled cell landed in a live region (no dead cell in any configuration), and the spread came
from `β` and `σ_w²`.

**One genuine caution survives, and it is not about counting.** **[verified at source]** Sattelberg,
Cavalieri, Kirby, Peterson & Beveridge, *Locally linear attributes of ReLU neural networks*
(*Frontiers in Artificial Intelligence* 6:1255192, 2023) find that regions are almost all distinct —
in MNIST networks "even simple networks… only have overlap on < 1% of the training inputs" — but that
the local maps are highly **redundant**: "there is potentially a great deal of redundancy or
similarity among them," and clustering to as few as ten centres preserved accuracy on Inception.
Distinct matrices need not be dynamically distinct. That is the right worry and it is about
*similarity*, not *count* — which is precisely what §6 measures and finds nonzero.

### DeepESN vs #7's rejection of depth: they are about different mechanisms

**[verified at source]** Gallicchio & Micheli, *Deep Echo State Networks*
([arXiv:1712.04323](https://arxiv.org/abs/1712.04323)): "the structured state space organization with
multiple time-scales dynamics in deep RNNs is **intrinsic to the nature of compositionality** of
recurrent neural modules," with timescales "naturally ordered along the network's hierarchy."
`05-timescales.md` rejects depth-gives-timescale on the grounds that unit delay is a phase shift, not
a decimation.

**Adjudication: both are right, and they are not in contact.** DeepESN's layers are each *themselves
recurrent reservoirs*, with their own leaky-integrator state and their own spectral radius; the
hierarchy of timescales is a hierarchy of **per-layer recurrences**, tuned per layer. Patchworks'
depth is a chain of **singly-recurrent cells separated by a unit-delay edge**; there is no second
recurrence per hop to slow anything down. `05-timescales.md`'s argument is untouched — and DeepESN is
in fact evidence *for* #7's actual mechanism rather than against it, since what produces DeepESN's
hierarchy is per-module recurrent dynamics differing in spectral radius, which is exactly what #7
proposes to obtain from bias-selected regions instead of by hand.

**[verified at source]** Jaeger, Lukoševičius, Popovici & Siewert, *Optimization and applications of
echo state networks with leaky-integrator neurons* (Neural Networks 20(3), 2007) remains the
precedent for the **fallback**, not the mechanism: heterogeneous per-unit time constants exist there
as a hand-set "global control parameter… a leaking rate has to be optimized" — `05-timescales.md`'s
clock divisor, one field over. It strengthens the fallback and does not support the claim.

## 6. The measurement

`prototypes/regional-spectra/spread_pilot.py` builds the go/no-go rig `05-timescales.md` asks for and
runs it. Per cell it forms `J = ∂chart_{t+1}/∂chart_t` — the chart's round trip through `encode` then
`step` — for one frozen shared body and 150 cells differing in bias, and reads `ρ(J)` exactly.

**Caveats, before any number below is quoted.** The body is a **stand-in**: iid Gaussian ReLU MLPs at
`k = 12`, `n = 32`, because nothing else about the body is specified yet — not its widths, not its
depth, not its activation. The biases are iid Gaussian, not biases a cell would learn. It is a
one-step Jacobian, not a simulated trajectory. **This establishes the shape of the answer and the
sensitivity of the rig, not the body's number.**

> **Discharged by the amendment above (#349).** The re-run against the converted body was
> done, and no number in this section is the body's. The nearest successor to the spread swept
> here reads **~2×**, not 7.7×; see the amendment's table for all three of ADR-0028's quantities.

**Spread exists. It is not a spike.** In every non-degenerate configuration the across-cell `ρ` spans
a factor of 2–8, and `τ = −1/ln ρ` a factor of 3–46. At `widths=[64,64], σ_w²=1.7`: `ρ` p05/med/p95 =
0.32 / 0.57 / 0.98, `τ` = 0.86 / 1.76 / 5.49 ticks.

**Narrow layers widen the dispersion, as Hanin's `β` predicts** (`σ_w²=1.7, σ_b²=0.5`):

| widths | `sd(log₁₀ ρ)` | `ρ` p05 → p95 |
|---|---|---|
| `[12,12]` | **0.32** | 0.064 → 0.693 |
| `[32]` | 0.12 | 0.440 → 1.113 |
| `[64,64]` | 0.13 | 0.354 → 0.883 |
| `[128,128,128]` | 0.13 | 0.344 → 0.862 |

**Biases alone reproduce essentially the whole spread — #7's mechanism is the one doing the work.**
Holding the operating point fixed and varying only the per-cell biases (`[12,12], σ_w²=1.7`) gives
`ρ` p05/med/p95 = 0.147 / 0.463 / 0.871, a `τ` p95/p05 ratio of **7.7**; at `[64,64], σ_w²=1.9` the
ratio is **46**. The regional route from bias to timescale — the step §1 called unestablished —
**works in the pilot**.

**But the operating point moves it just as much.** The same sweep varying only the chart and incoming
stalk, biases fixed, gives a `τ` ratio of **7.3** (and 42 in the wide configuration): statistically
indistinguishable from the bias effect. See §7's first defect.

**`σ_b²` is the weak knob; `σ_w²` is the strong one, and `σ_w²` is not per-cell.** At frozen weights,
sweeping `σ_b²` over three orders of magnitude (0.01 → 8) moved median `ρ` only 0.261 → 0.309, while
`σ_w²` from 1.2 → 2.0 moved it 0.198 → 0.943. This **qualifies the Schoenholz hit**: `σ_b²` does move
the operating point, as mean-field says, but in a finite `k = 12` body its leverage over the *median*
is small. Its leverage over the *spread* — which is what #7 needs — is the part that survives.

**Spread and stability are the same knob.** The `σ_w²` sweep at `[64,64]`:

| `σ_w²` | `τ` p95/p05 | cells with `ρ ≥ 1` |
|---|---|---|
| 1.2 | 2.0 | 0.00 |
| 1.5 | 3.2 | 0.01 |
| 1.7 | 6.4 | 0.05 |
| 1.9 | 19.0 | 0.24 |
| 2.0 | 35.1 | 0.39 |

Every configuration with a usable timescale ratio has a **material fraction of cells above the
stability boundary**, and every configuration with no unstable cells has every `τ` under one tick.
The median crosses `ρ = 1` at `σ_w² ≈ 2`, the known ReLU criticality point — the pilot reproduces it.
**This is the pass's headline and it is not in the literature found:** a single global `σ_w²` cannot
buy spread and unconditional stability at once, because the spread lives in the tail that crosses the
boundary.

**Non-normality is real but mild**: `σ_max/ρ` median **2.62**, while `‖J⁸‖^(1/8)/ρ` median 1.12 — i.e.
a one-tick transient amplification of ~2.6× that decays at the eigenvalue rate thereafter.

**Estimator power**: `sd(log₁₀ ρ)` measured across 8 seeds is stable to ±0.021 at 20 cells, **±0.011
at 50**, ±0.006 at 150. Naive sampling at the architecture's own population size is sufficient (§5).

---

## 7. Verdict

**The mechanism is not dead on arrival. Sub-question 1's "is it constructible" answers *yes,
statistically*, with two named knobs and one unresolved conflict between them.**

- **Spread is available and biases are what produce it.** Measured, at fixed shared frozen weights
  and a fixed operating point, a 7.7×–46× spread in `τ` across 150 bias settings (§6).
- **The construction recipe is two knobs, and only one is per-cell.** `σ_w²` (global, shared, frozen)
  positions the distribution against the stability boundary; **narrowness — large `β` — supplies the
  dispersion**, and Patchworks is narrow by construction, so it gets the dispersion for free (§4,
  §6). `σ_b²` is a weak third knob.
- **Statistical shaping is enough and is not blocked by hardness** (§2, §3).
- **If the emergent spread proves thin, impose it in the published shape:** log-uniform over a range
  set by the task horizon, as chrono initialization and S4D both do (§1).
- **The go/no-go run is cheap, correctly specified, and adequately powered** — with the four protocol
  corrections in §5.

**"Not constructible" was pre-authorised as a clean result. It is not the result.** The result is
*constructible but coupled*: the same quantity that buys the spread is the one three other parts of
the spec need bounded.

### Defects and revisions warranted

**7.1 — A cell's timescale is not a per-cell constant, and `05-timescales.md` reads as if it were.**
The measurement (§6) finds the operating point contributes as much spread as the bias does. The spec
says "a cell's **effective timescale is the spectral radius of its region's Jacobian**, selected by
its biases"; `CONTEXT.md` defines it as a property of "the activation region its biases select."
Both are true only if the cell *stays in its region*, and the cell's chart moves every tick. What the
biases set is the **distribution** a cell's per-tick regional spectrum is drawn from — its mean
timescale — not a fixed rate. **This is a wording-and-substance revision to `05-timescales.md`
(*Persistence under the cell's own dynamics*), `CONTEXT.md` (*effective timescale*), and ADR-0005's
decision paragraph.** It does not break the mechanism; a mean rate is still a rate. It does change
what the go/no-go run measures and what the acceptance demo's readout means.

**7.2 — This makes the fold margin structural, not just a stability side-condition.** ADR-0007's
bound `γ × floor < fold margin` was framed as protecting the operating point from the disagreement
floor. Given 7.1 it is doing a second job: **the fold margin is what makes "the cell's region" a
well-defined object at all.** A cell whose margin is small re-draws its timescale every tick. The
bound binds hardest at the apex (`gain_v = γ/Σ_e m_e`, `Σ_e m_e` falling with depth,
`02-tick-semantics.md`) — which is exactly where slow cells are supposed to live. **Recommend
`05-timescales.md` state the fold margin as a precondition of the timescale claim, not only as a
reconciliation constraint.** Hanin & Rolnick supply the quantitative handle: mean distance to the
nearest region boundary scales like `1/#neurons`, so **a wider body has a smaller fold margin** — the
same axis as §4's `β`, pulling the same direction. Wide bodies: stable timescales, little spread.
Narrow bodies: real spread, margins that may not hold.

**7.3 — Spread and stability cannot both come from one global `σ_w²` (§6).** Configurations with a
useful `τ` ratio put 5–39% of cells past `ρ ≥ 1`. Three ways out, none yet chosen, all cheap to test
on the existing rig: (i) accept a truncated distribution — clip the body's construction so the
realised maximum `ρ` is below one, and take the smaller ratio; (ii) spread in the *slow* direction
only, choosing `σ_w²` so the median is fast and the tail reaches toward one rather than through it;
(iii) impose the spread (chrono/S4D shape, §1) instead of drawing it. **Recommend this be recorded as
a construction question against the body, not left implicit in "constructed for spread."**

**7.4 — `ρ` is the wrong single statistic to build the acceptance readout on.** Non-normality of ~2.6×
(§6) plus Yildiz et al.'s result that `ρ < 1` is not sufficient for the echo state property means the
demo's per-cell timescale readout should be a measured decay (`‖J^t‖`, or the live private-component
trace `05-timescales.md` already specifies) rather than an eigenvalue. The spec's chosen demo
evidence — `‖Δ(private component)‖` per cell against hop distance — is already the right object.
**No change needed to the demo; the change is to the go/no-go run, which should not report `ρ`
alone.**

### Measurements to run

- ~~Re-run `spread_pilot.py` against the body once its widths, depth and activation are fixed. The
  pilot's numbers are a stand-in (§6 caveats) and should not be quoted as the body's.~~
  **Done, by replacement rather than re-pointing (#349).** The Koopman conversion left this rig
  without its object — a linear `K` has one region — so
  `prototypes/regional-spectra/converted_spread.py` measures ADR-0028's three quantities in this
  document's own statistic instead. See the amendment at the top.
- Add a **fold-margin** column to the same sweep — `min |z(x) − b_z| / ‖∇z(x)‖` (Hanin & Rolnick,
  verified) — and check it against `γ × floor` per cell across the taper, as #28 and #9 both asked.
  One sweep answers both tickets; the apparatus is already written.
- Measure `τ` over a **driven trajectory**, not a single Jacobian, to size how often a cell changes
  region (7.1).

### Honest gaps

- **The pilot is not the body.** iid Gaussian ReLU MLPs, iid Gaussian biases, one-step Jacobians. It
  establishes shape and sensitivity, not the body's number.
- **Nobody has tried to widen this distribution on purpose** (§1). Read the whole positive case as
  *the machinery permits it*, not as *it has been done*.
- **Sattelberg et al.'s redundancy result (§5) was not tested here.** Distinct regions with similar
  local maps is the failure mode the pilot would not have caught if the similarity were in `ρ`
  specifically; the pilot measures dispersion in `ρ` and finds it nonzero, which is evidence against
  the worry but not a direct test of it.
- **Not read in full:** Montúfar et al.'s exact region bound (abstract only); Fazlyab et al. beyond
  the abstract; Ubaru, Chen & Saad's SIMAX text (paywalled — its scope was taken from the ICML 2021
  analysis paper and the method's own description); Serra et al. on region counting (not searched,
  judged redundant to Hanin & Rolnick); Hanin & Nica's finite-`k` regime beyond the log-normal
  statement.
- **The two Scholar Gateway result sets the first agent fetched and never read have now been read.**
  Both are reservoir-computing hardware and applications papers (memristive reservoirs, photonic
  reservoirs, ESN forecasting) with no bearing on regional Jacobian spectra. **Closed, not a gap.**
- **`arXiv:2508.02882`** (*Deep Network Trainability via Persistent Subspace Orthogonality*) was
  opened and is the isometry objective again, relaxed to a subspace — noted, not load-bearing.
