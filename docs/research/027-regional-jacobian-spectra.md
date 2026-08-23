# Constructing a body with spread regional Jacobian spectra (patchworks#27)

> **STATUS: INCOMPLETE — RECOVERED PARTIAL.** The agent running this pass was killed mid-flight by an
> API spend limit and never wrote a file. This document is a reconstruction of its research trail from
> its transcript, assembled so the pass can be resumed rather than restarted. **No verdict has been
> reached on any sub-question.** Everything below is evidence; the synthesis is the part that is
> missing.
>
> **The paper corpus has been rescued out of `/tmp` and is durable at
> `~/.claude/projects/-Users-angl-Documents-patchworks/research-cache-027/`** — text extractions of
> `chrono`, `leaky`, `deepesn`, `s4d`, `hr2019`, `1801.03744`, `1802.09979`, `1812.05994`,
> `1906.00904`, `2003.01219`, `2111.00396`, plus both unread Scholar Gateway result sets. Re-download
> is unnecessary. The original `.pdf` files remain in the dead session's scratchpad at
> `/private/tmp/claude-501/-Users-angl-Documents-patchworks/c3c36451-fe6e-4cd3-b707-d90c4195de24/scratchpad/`
> and will be lost whenever `/tmp` is cleaned; the extracted text is what matters and it is safe.
>
> The two Scholar Gateway result sets (480 and 484 lines) were fetched and **never read** — they blew
> the tool's output cap. A spot-check of titles suggested mostly low-relevance reservoir-hardware
> papers, but neither was read in full.

Terms used here are defined in [`CONTEXT.md`](../../CONTEXT.md). The question is set by
[`05-timescales.md`](../spec/05-timescales.md) and
[ADR-0005](../adr/0005-timescale-is-persistence-not-a-schedule.md).

## The question, restated

One shared frozen piecewise-linear cell body. Weights fix fold *directions*; per-cell biases fix fold
*offsets*, so each cell sits in a different activation region of the same map, with its own local
Jacobian and its own spectrum. #7 makes a cell's effective time constant the spectral radius of *its*
region's Jacobian. That differentiates cells only if the **distribution of regional Jacobian spectral
radii, across the bias settings cells actually occupy, has real spread**. If it is a spike, every cell
lands in the same dynamic regime and #7's mechanism is dead on arrival.

This is a **feasibility** question asked *ahead* of the decision, not a validation pass after it —
the inverted sequencing precedent is [#13](https://github.com/NGL321/patchworks/issues/13).

---

## What was established, with sources

Each item is marked **[verified at source]** where the agent read the paper's own text (PDF or ar5iv
full text), or **[secondary]** where the claim rests on a search-result summary or an abstract page
and still needs confirming.

### The single most on-point result: bias variance moves the operating point along the order-to-chaos axis

**[verified at source]** Schoenholz, Gilmer, Ganguli & Sohl-Dickstein, *Deep Information Propagation*
([arXiv:1611.01232](https://arxiv.org/abs/1611.01232)), read via ar5iv.

The order-to-chaos control parameter is

> "χ₁ = σ_w² ∫ 𝒟z [ϕ'(√q\* z)]²" (Eq. 5)

and the fixed point q\* satisfies

> "q\* = σ_w² ∫ 𝒟z φ²(√q\* z) + σ_b²"

with the critical line χ₁ = 1 separating an ordered from a chaotic phase, drawn in the
**(σ_w², σ_b²) plane**.

**Why this is the load-bearing hit.** σ_b² does not appear in χ₁ *directly* — but it appears
additively in q\*, and χ₁ depends on q\*. So **bias variance shifts χ₁ indirectly, by moving q\***.
That is structurally the Patchworks claim: hold the weights fixed, vary the biases, and the local
dynamical regime moves. This is the closest thing found to a positive answer on Q1, and the resumed
pass should build its answer around it.

**What it does not yet settle.** Mean-field theory describes an *ensemble* at initialization, in the
infinite-width limit, in terms of variances σ_w², σ_b² — not the spread of spectral radii across the
*specific, learned, finite* bias vectors ~150 cells actually occupy in a `k = 12` body. The gap
between "σ_b² moves the mean operating point" and "the across-cell distribution of regional spectral
radii has usable spread" is exactly what the pass still has to close, and it is not a small gap.

### Controlling the Jacobian spectrum is an established design activity — and the activation choice matters

**[verified at source]** Pennington, Schoenholz & Ganguli, *Resurrecting the sigmoid in deep learning
through dynamical isometry* ([arXiv:1711.04735](https://arxiv.org/abs/1711.04735)): "controlling the
entire distribution of Jacobian singular values is an important design consideration in deep
learning." Critically for Q4: **ReLU networks "are incapable of dynamical isometry"**, whereas sigmoid
networks can achieve it — but only under orthogonal weight initialization.

**[verified at source]** Pennington, Schoenholz & Ganguli, *The Emergence of Spectral Universality in
Deep Networks* ([arXiv:1802.09979](https://arxiv.org/abs/1802.09979)): free probability theory gives
"a detailed analytic understanding of how a deep network's Jacobian spectrum depends on various
hyperparameters including the nonlinearity, the weight **and bias** distributions, and the depth."

**Note the direction of the tension.** This literature is overwhelmingly about achieving *isometry* —
concentrating the spectrum tightly around one. Patchworks wants the **opposite**: deliberate spread.
The machinery for reasoning about the spectrum transfers; the objective is inverted. The resumed pass
should say plainly whether anyone has ever tried to *widen* this distribution on purpose, because
nothing found so far suggests they have.

### Finite width is what produces spread — and Patchworks is narrow

**[verified at source]** Hanin, *Which Neural Net Architectures Give Rise to Exploding and Vanishing
Gradients?* ([arXiv:1801.03744](https://arxiv.org/abs/1801.03744)): the empirical variance of the
squares of the entries of the input-output Jacobian is **exponential in β = Σ_j 1/n_j**, the sum of
reciprocals of hidden-layer widths. "When β is large, the gradients computed by N at initialization
vary wildly." The paper's own advice — via the power-mean inequality — is to *minimize* β, and hence
to keep widths equal and large.

**This may be the pass's most interesting finding, and it cuts both ways.** Patchworks' body is
narrow by construction (`k = 12`), so β is large, so Jacobian statistics fluctuate wildly. What that
literature calls a pathology to be minimized is, for #7, **the very spread the mechanism needs**. The
low-dimensionality constraint may be buying the timescale spread for free.

The obvious caution, which the pass never got to state: wild Jacobian variation is also exactly what
ADR-0007's `γ × floor < fold margin` bound and #33's settling floor are trying to keep bounded. Spread
and stability are being asked of the same quantity. That tension deserves to be the resumed pass's
headline, not a footnote.

**[verified at source]** Hanin & Nica, *Products of Many Large Random Matrices and Gradients in Deep
Neural Networks* ([arXiv:1812.05994](https://arxiv.org/abs/1812.05994)): the log of the ℓ² norm of
such a product is asymptotically Gaussian, giving "a quantitative measure of the extent to which the
exploding and vanishing gradient problem occurs." A candidate analytic handle on the *shape* of the
distribution, not just its width — worth pursuing, unexamined so far.

### Do cells actually land in different regions? Region counts are far below the exponential bound

**[verified at source]** Hanin & Rolnick, *Complexity of Linear Regions in Deep Networks*
([arXiv:1901.09021](https://arxiv.org/abs/1901.09021)): for networks at initialization, "the average
number of regions along any one-dimensional subspace grows **linearly** in the total number of
neurons, far below the exponential upper bound," and "the average distance to the nearest region
boundary at initialization scales like the inverse of the number of neurons." Both quantities stay
"roughly constant during training."

Also recorded, exact: `distance(x, B_N) = min |z(x) − b_z| / ‖∇z(x)‖`.

**Two unexamined consequences.** (i) Far fewer distinct regions than the folklore suggests means the
spread available across ~150 bias settings may be much smaller than a naive count implies — this is
the strongest *negative* evidence found, and it was never weighed against the positive evidence above.
(ii) That distance formula is the same object as ADR-0007's fold margin, and the 1/#neurons scaling is
a quantitative handle on it. Nobody has connected those yet.

### Hardness: the intractability is specifically about *exact* control

**[verified at source]** Scaman & Virmaux, *Lipschitz regularity of deep neural networks*
([arXiv:1805.10965](https://arxiv.org/abs/1805.10965)): "even for two layer neural networks, the exact
computation of this quantity is NP-hard" — the quantity being the Lipschitz constant itself.

**[verified at source]** Fazlyab, Robey, Hassani, Morari & Pappas
([arXiv:1906.04893](https://arxiv.org/abs/1906.04893)): recasts the problem as a semidefinite program
yielding "guaranteed upper bounds," the most accurate in the literature. The abstract does not discuss
NP-hardness.

**Where this points, unstated so far:** hardness attaches to *exact* computation, while tractable
methods deliver *bounds* and *statistics*. #7 needs spread, not a designed spectrum per region — so
sub-question 3 (statistical vs exact) looks answerable in the affirmative, and this is probably the
cheapest remaining win in the pass.

### Reservoir computing: heterogeneous timescales exist, but as an explicit parameter

**[verified at source]** Jaeger, Lukoševičius, Popovici & Siewert, *Optimization and applications of
echo state networks with leaky-integrator neurons* (Neural Networks 20(3), 2007;
[PDF](https://www.ai.rug.nl/minds/uploads/leakyESN.pdf)): leaky-integrator ESNs have "one more global
control parameter than standard sigmoid unit ESNs… a leaking rate has to be optimized," with the paper
"managing very slow timescales by adjusting the leaky neurons' time constants" on the "lazy eight"
task, and a gain γ dialled from 1.0 down to 0.02 and back to speed the dynamics up and down at
runtime.

**The relevance is precise and unfavourable to the interesting claim.** This is a real precedent for
heterogeneous per-unit time constants — but as a **hand-set parameter per unit**, which is exactly
#7's *clock divisor* fallback, not its emergent-from-biases mechanism. It strengthens the fallback; it
does not support the claim. Note the paper assumes a uniform leaking rate "for simplicity," so even
the heterogeneous case is less explored than it first appears.

**[verified at source]** Yildiz, Jaeger & Kiebel, *Re-visiting the echo state property* (Neural
Networks 35, 2012): spectral radius below unity is **not sufficient** for the echo state property. A
direct caution against reading a spectral radius as a time constant too naively — which is what #7's
mechanism does.

**[verified at source]** Gallicchio & Micheli, *Deep Echo State Networks*
([arXiv:1712.04323](https://arxiv.org/abs/1712.04323)): "the structured state space organization with
multiple time-scales dynamics in deep RNNs is **intrinsic to the nature of compositionality** of
recurrent neural modules," with timescales "naturally ordered along the network's hierarchy."

**This one deserves a fight, and the pass never had it.** DeepESN says depth *does* produce a
timescale hierarchy for free — while #7 rejected depth-produces-timescale for Patchworks on the
grounds that unit delay is a phase shift, not a decimation. Either the two claims are about different
mechanisms (stacked recurrent layers each with their own recurrence, versus a delay line between
single-recurrence cells) or #7 gave up something it did not need to. The resumed pass should
adjudicate this explicitly.

### Biases setting timescales: the exact analogue exists, and it is called chrono initialization

**[verified at source]** Tallec & Ollivier, *Can recurrent neural networks warp time?*
([arXiv:1804.11188](https://arxiv.org/abs/1804.11188), ICLR 2018): learnable gates give
"quasi-invariance to general time transformations," recovering part of the LSTM from first principles,
and this "leads to a new way of initializing **gate biases** in LSTMs and GRUs… this new chrono
initialization is shown to greatly improve learning of long term dependencies, with minimal
implementation effort." Confirmed present in the PDF at line 324 ("Hereafter, we refer to this as the
chrono initialization") and used with `T_max = 3T/2` on the copy task.

The formula the agent never reached (Eq. 16, recovered here from the downloaded PDF):

```
b_f ∼ log(𝒰([1, T_max − 1]))
b_i = −b_f
```

with `𝒰` the uniform distribution and `T_max` "the expected range of long-term dependencies to be
captured."

**Why this matters most for the *shape* of the answer.** Chrono initialization is a real, published
instance of **biases spanning a range of timescales by construction** — the same move #7 wants, one
architecture over — and note the form: the spread is **log-uniform over a range**, not Gaussian, and
`T_max` is set from the task's known horizon. If #7's mechanism has a published ancestor, this is it.

**The transfer is not free, and the pass should say so.** Chrono initialization sets the biases of an
explicit *gate* whose sigmoid output is directly a retention coefficient — the bias maps to a time
constant analytically. Patchworks has no gate; its biases translate folds in a shared frozen
piecewise-linear body, and the route from bias to effective time constant runs through *which
activation region the cell lands in*, which is exactly the unestablished step. Chrono is the right
precedent to argue from and does not settle the question.

### Unconfirmed and unfinished

- **S4 / S4D timescale initialization.** The agent tried to confirm that the Δ timescale parameter is
  initialized log-uniformly over a range ([arXiv:2111.00396](https://arxiv.org/abs/2111.00396), and
  S4D at [arXiv:2206.11893](https://arxiv.org/abs/2206.11893), `s4d.pdf` downloaded). Both greps came
  back empty and the abstract pages do not carry it. **Unverified — do not cite until confirmed.** If
  it holds, it is a second modern instance of deliberately spreading timescales across units.
- **Sub-question 5 — a better-conditioned estimator than naive bias sampling.** Never reached. The
  agent was searching for stochastic Lanczos quadrature for spectral density estimation (Ubaru, Chen &
  Saad 2017, SIMAX) at the moment it was killed. That is the thread to pick up first, and it is the
  one #7 explicitly asked for.
- **Two papers surfaced but never opened:** *Locally linear attributes of ReLU neural networks*
  (Frontiers in AI, 2023, `10.3389/frai.2023.1255192` — the fetch died on the spend limit) and
  *Neural Networks with Orthogonal Jacobian* ([arXiv:2508.02882](https://arxiv.org/abs/2508.02882)).
  The first states the Jacobian as `J = S(x₀)W` per activation region, which is exactly this pass's
  object.
- **Montúfar et al. and Serra et al.** on linear-region counting were named in the ticket and never
  searched; Hanin & Rolnick covers adjacent ground and may make them redundant.
- **Orthogonal/unitary RNN parameterisations** (uRNN, expRNN, antisymmetric RNNs) were named in the
  ticket and never searched at all.

## How to resume

1. ~~Pull the chrono-initialization formula.~~ **Done during recovery** — see above. What remains is
   judging how far it transfers to a gateless frozen body.
2. Pick up sub-question 5 where it was cut off: stochastic Lanczos quadrature as an estimator for the
   spectral-radius distribution, versus #7's naive bias sampling.
3. Weigh Hanin's β (narrow width → wild Jacobian variation → **spread**) against Hanin & Rolnick
   (region counts far below exponential → **less spread than hoped**). These are the two strongest
   results found and they point opposite ways. The pass's verdict lives here.
4. State the tension between spread and ADR-0007's fold margin / #33's settling floor. The same
   narrowness that buys the spread threatens the stability bound.
5. Adjudicate DeepESN's depth-gives-timescales against #7's rejection of depth.
6. Confirm or drop the S4/S4D claim.
7. Then write the verdicts, and the *Candidate revisions* section in the house style of
   [`018-sandbox-citations.md`](./018-sandbox-citations.md) — including, if it comes to it, the
   ticket's own pre-authorised outcome: **"not constructible"** is a clean result, absorbed by #7's
   fallback (random non-degenerate init plus an empirical spread check, backed by the clock divisor as
   an already-built rig). Nothing downstream breaks; the interesting claim just does not get made.
