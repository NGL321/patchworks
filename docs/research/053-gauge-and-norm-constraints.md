# Citation pass: the gauge argument, and whether norm-without-orthogonality suffices (patchworks#53)

Validates the design closed in [#37](https://github.com/NGL321/patchworks/issues/37)
([ADR-0010](../adr/0010-restriction-map-scale-is-gauge-fixed.md), `docs/spec/07-local-learning-rule.md`
*The transport rule*), raised by R1 of
[`016-cell-contract-citations.md`](./016-cell-contract-citations.md) §4. Citations validate after the
fact per the map's sequencing rule; this document does not reopen the closed design — it records where
a source confirms a claim and flags where one threatens it. Scope is exactly the ticket's three
questions and nothing else. Vocabulary follows [`CONTEXT.md`](../../CONTEXT.md): Patchworks' side of
every comparison in its own terms, the prior art's in its field's.

## Headline verdict, stated plainly

**All three questions came back positive, and the one the ticket flagged as "constructed, not read"
turns out to be a named definition and a proved lemma in the literature it pointed at — but the same
lemma contradicts a sentence of ADR-0010's rhetoric about which way a free magnitude drifts.**

- **Q1 — norm-without-orthogonality is not a Patchworks bet.** It is the *founding* choice of the
  sheaf-learning literature. Hansen & Ghrist (ICASSP 2019), the paper Di Nino et al. generalise,
  minimise sheaf Dirichlet energy over unconstrained restriction maps and exclude the trivial solution
  with a **log barrier on `tr(L_ii)`**, which they themselves describe as "putting a barrier on the
  nuclear norm of the diagonal blocks." Since `L_ii = Σ_e F_{i◁e}ᵀF_{i◁e}`, that quantity **is**
  `Σ_e ‖F_{i◁e}‖_F²` — a Frobenius-norm floor on a node's incident restriction maps, with no
  orthogonality anywhere. They reject the orthogonal class explicitly, on convexity grounds. The
  same pattern is standard one level down in graph learning (Dong et al.'s `tr(L) = n`; Kalofolias'
  role-(1) requirement on `f(W)`).
- **Q1 corollary — Di Nino et al. are weaker than R1 read them as being.** They require *a* constraint
  on the feasible set; orthonormality is offered as "**a possibility** that is theoretically plausible
  and gives rise to a simple solution," i.e. chosen for closed-form Procrustes convenience, not
  necessity. ADR-0010's alternatives section can drop the hedge.
- **Q2 — the unidentified-parameter argument is known, and formalised.** Arora, Li & Lyu give it
  Definition 2.1 (scale-invariance), the statement "the scale of each weight does not affect loss
  values," and Lemma 2.4. van Laarhoven states the L1/L2 half: "the parameter `λ` has no impact on the
  optimum, since the weights can be scaled to compensate." **van Laarhoven §5.5 is ADR-0010's
  enforcement mechanism, move for move** — project to unit norm after the step, explicitly
  distinguished from Weight Normalization's reparameterisation, for the same reason ADR-0010 gives.
- **Q2 threat — the drift direction in ADR-0010 is wrong.** Arora et al.'s Lemma 2.4 proves that under
  a purely scale-invariant objective the gradient is *perpendicular* to the weight and
  `‖w_{t+1}‖² = ‖w_t‖² + η²‖∇‖²` — the norm **grows monotonically**, never shrinks. Salimans & Kingma
  observe the same. ADR-0010 says the free magnitude "drifts, and the direction it drifts under any
  residual asymmetry is the one route one names." The literature says the opposite direction. The
  *decision* survives untouched — the band's upper end closes the failure the literature actually
  predicts — but the sentence is a candidate revision (R1 below).
- **Q3 — the recalled claim does not exist as recalled.** The nearest real source is Hu
  ([arXiv:2601.21207](https://arxiv.org/abs/2601.21207)), Theorem 1: a **GAT** — not a transformer — is
  a cellular sheaf whose restriction maps are "scalar multiplication by `w_ij`". No zero coboundary;
  a sheaf with zero coboundary computes nothing. The transformer step is a separate informal one
  (Joshi 2020, restated by Barbero 2022), and Barbero lists sheaf transformers as *future work*.
- **Q3, on `ρ = 1` — the attention analogy does not hold, in an instructive way.** Attention's
  normalisation is `Σ_j w_ji = 1`: a **per-node budget across incident edges**, not a per-map norm.
  It is structurally Hansen & Ghrist's `tr(L_ii)` barrier, not ADR-0010's per-map band. Evidence that
  exact fixing *buys* something exists (van Laarhoven §5.5; Miyato's `σ(W) = 1`; the unit sphere as
  "the intrinsic domain" of scale-invariant parameters), but none of it is attention.
- **One further threat, unasked but load-bearing.** Miyato et al. prove that a *fixed Frobenius norm*
  makes rank one the **maximiser** of output gain. ADR-0010 argues fixed Frobenius makes rank
  concentration "a priced trade rather than a free lunch." Same algebra, opposite sign on the price.
  See §4 and R2.

---

## 1. Q1 — Is norm-without-orthogonality known to exclude the trivial solution?

**Verdict: YES, and in the closest possible literature — the sheaf-Laplacian learning paper that
Di Nino et al. extend. ADR-0010's constraint is not a bet; it is closer to the field's default than
Di Nino et al.'s orthogonality is.**

### 1.1 Hansen & Ghrist: a Frobenius floor on a node's incident maps, no orthogonality

*Source: Hansen, J. & Ghrist, R. (2019). "Learning Sheaf Laplacians from Smooth Signals," ICASSP 2019,
pp. 5446–5450. Read from the authors' PDF at
[jakobhansen.org/publications/learningsheaves.pdf](https://www.jakobhansen.org/publications/learningsheaves.pdf).*

Their problem (2) is Patchworks' objective almost exactly — minimise sheaf Dirichlet energy over the
sheaf Laplacian, i.e. over the restriction maps:

> "min_{L ∈ L_sheaf} tr(Xᵀ L X) + f(L) … where `f` enforces connectivity and sparsity. Here we
> consider different functional forms for `f`. We will write `f(L) = α f_c(L) + β f_s(L)`, with `f_s`
> controlling sparsity and `f_c` encouraging connectivity."

The term that excludes the collapsed sheaf is `f_c`, and it is a **norm barrier**:

> "A common way to encourage connectivity of a graph is to put a lower barrier on the degrees of
> vertices. However, the cone of sheaf Laplacians contains block diagonal matrices, so this constraint
> does not perfectly enforce connectivity for sheaves. We let `f_c(L) = − Σ_i log(tr(L_ii))`, thus
> requiring that each diagonal block have at least one nonzero diagonal entry. Note that since the
> diagonal blocks are positive semidefinite, the trace is equal to the nuclear norm, and we can view
> this term as **putting a barrier on the nuclear norm of the diagonal blocks**."

**Why this is decisive for ADR-0010.** For a sheaf Laplacian, the diagonal block at a node is
`L_ii = Σ_{e ∋ i} F_{i◁e}ᵀ F_{i◁e}`, so

`tr(L_ii) = Σ_{e ∋ i} tr(F_{i◁e}ᵀ F_{i◁e}) = Σ_{e ∋ i} ‖F_{i◁e}‖_F²`.

Hansen & Ghrist's connectivity term is therefore **a lower barrier on the sum of squared Frobenius
norms of a node's incident restriction maps** — the same quantity ADR-0010 bounds below, aggregated
per node rather than imposed per edge. Nothing in their formulation constrains the maps' basis. This
is a published, primary-source precedent for the exact structural move: *scale floor, free basis,
excludes the trivial solution*.

They also state why the orthogonal class is unattractive, which is a second, independent line of
support for ADR-0010's alternatives section:

> "The set of connection Laplacians is not a convex cone, since convex combinations of orthogonal
> matrices are not orthogonal. However, a simple convex cone that contains the set of connection
> Laplacians is the cone of sheaf Laplacians whose diagonal blocks are scalar multiples of the
> identity."

That is: the field's originating paper **deliberately relaxed out of orthogonality** into a larger
class, and paid for the trivial solution with a norm barrier. Di Nino et al.'s return to orthogonality
in 2025 is a step back toward tractability, not a step forward in necessity.

One asymmetry worth recording honestly: their `f_s` is a Frobenius penalty
(`f_s(L) = Σ ‖L_ij‖_F²`) added to **counterbalance** the objective's tendency toward sparsity —

> "The objective without `f_s(L)` already encourages sparsity of `L`, since we are minimizing a linear
> function within a particular cone. … Thus, rather than adding a term to encourage sparsity, we add a
> regularization term to counterbalance this tendency toward sparsity."

Patchworks composes an L1 **for** sparsity in the same step. So the two designs push opposite ways on
that term while agreeing on the floor. Not a contradiction — Patchworks wants the pruning
`06-graph-topology.md` relies on, and prices it on the normalised map — but it means Hansen & Ghrist
support the *floor* and not the *sparsity pressure*, and the ADR should not claim more than that.

### 1.2 Di Nino et al. state the requirement more weakly than R1 read it

*Source: Di Nino, L., Barbarossa, S. & Di Lorenzo, P. (2025). "Learning Sheaf Laplacian Optimizing
Restriction Maps," [arXiv:2501.19207](https://arxiv.org/abs/2501.19207) — read from the PDF.*

R1 (016 §4.1) quoted the requirement correctly:

> "We also need to impose a constraint on the set of feasible restriction maps `F`, to avoid the
> trivial solution."

What R1 did not carry forward is the **next paragraph**, which frames orthonormality as one option
among several rather than as the constraint:

> "The feasible set to be used in the search of the restriction maps in the global problem **can be
> chosen in different ways. A possibility that is theoretically plausible and gives rise to a simple
> solution** is that of assuming the restriction maps to be orthonormal. In this way, the restriction
> maps represent isometries."

And the reason it is "simple" is stated outright — it collapses to a closed form:

> "This problem can be solved as an orthogonal Procrustes problem, so that each local problem can be
> solved in closed form solutions via singular value decomposition."

**Reading.** The paper's requirement is on the *feasible set* `F` being constrained at all. Its choice
of orthonormality is motivated by closed-form solvability, which is a property Patchworks does not
need — the transport rule is a local gradient step with a projection, not a per-edge closed-form
solve. ADR-0010 adopts a different member of the same family of admissible feasible sets. That is a
weaker claim than "strictly weaker than the literature and unvalidated."

### 1.3 Graph learning, one level down: the same move is standard

*Sources: Dong, Thanou, Frossard & Vandergheynst (2016), "Learning Laplacian Matrix in Smooth Graph
Signal Representations," [arXiv:1406.7842](https://arxiv.org/abs/1406.7842); Kalofolias (2016), "How
to Learn a Graph from Smooth Signals," AISTATS,
[proceedings.mlr.press/v51/kalofolias16.pdf](http://proceedings.mlr.press/v51/kalofolias16.pdf). Both
read from PDF.*

Dong et al. minimise `tr(YᵀLY)` — the scalar-stalk special case of the sheaf objective — subject to:

> "`s.t. tr(L) = n`, `L_ij = L_ji ≤ 0, i ≠ j`, `L · 1 = 0` … The first constraint (the **trace
> constraint**) in Eq. (16) **acts as a normalization factor and permits to avoid trivial solutions**,
> and the second and third constraints guarantee that the learned `L` is a valid Laplacian matrix that
> is positive semidefinite. Furthermore, the trace constraint essentially **fixes the L1-norm of `L`**."

A pure scale constraint. No orthogonality; no invertibility; nothing about basis.

Kalofolias makes the requirement a numbered role of the regulariser:

> "This means that `f(W)` has to play three roles: **(1) prevent `W` from going to the trivial solution
> `W = 0`**, (2) allow `W` to obtain zero values, and (3) impose further structure using prior
> information."

Note the shape of that list. Excluding the trivial solution (1) and permitting individual entries to
go to zero (2) are stated as **separate, simultaneously satisfiable** requirements — which is exactly
Patchworks' combination of a norm floor with an L1 that prunes within the mask, and Kalofolias treats
it as unremarkable.

### 1.4 Constrained-weight literature generally: floors are rare, and the reason is instructive

The general deep-learning practice is **upper** norm bounds — max-norm constraints, weight clipping,
`σ(W) = 1`. This pass found no general-purpose deep-learning method that imposes a norm **floor**,
and the reason appears to be that ordinary supervised objectives are not minimised at `W = 0`, so
there is nothing to exclude. The floor is standard precisely where the objective *is* trivially
minimised by collapse — graph and sheaf structure learning from smoothness — which is Patchworks'
situation. **The absence of a general precedent is therefore not evidence against; the relevant
literature is the one where the trivial solution exists, and there the floor is the norm.**

---

## 2. Q2 — Is the unidentified-parameter argument known?

**Verdict: YES. It is Definition 2.1 and Lemma 2.4 of Arora, Li & Lyu, and the opening move of
van Laarhoven. ADR-0010 reconstructed a known result. One consequence of the known result
contradicts a sentence of the ADR.**

### 2.1 The argument, formalised

*Source: Arora, S., Li, Z. & Lyu, K. (2019). "Theoretical Analysis of Auto Rate-Tuning by Batch
Normalization," ICLR 2019, [arXiv:1812.03981](https://arxiv.org/abs/1812.03981) — read from PDF.*

The definition is ADR-0010's premise, stated as a definition:

> "**Definition 2.1. (Scale-invariance)** Let `F(w, θ′)` be a loss function. We say that `w` is a
> scale-invariant parameter of `F` if for all `c > 0`, `F(w, θ′) = F(cw, θ′)`; if `w` is not
> scale-invariant, then we say `w` is a scale-variant parameter."

And the consequence is ADR-0010's conclusion, stated as a fact:

> "Thanks to the scale-invariant properties, **the scale of each weight `w(i)` does not affect loss
> values.** However, the scale does affect the gradients."

The degenerate-optimisation observation is also there, and is worth carrying into Patchworks'
vocabulary because it names a second way a scale-free objective misleads a descent rule:

> "To make `‖∇_{w(i)} F_z(W; g)‖₂` to be small, one can just scale the weights by a large factor.
> **Thus there are ways to reduce the norm of the gradient that do not reduce the loss.**"

*Corroborating source: van Laarhoven, T. (2017). "L2 Regularization versus Batch and Weight
Normalization," [arXiv:1706.05350](https://arxiv.org/abs/1706.05350) — read from PDF.* He states the
penalty half — that a scale-invariant objective makes a norm-driving penalty stop meaning what it
looks like it means:

> "This means that the L2 penalty term forces the weights to become small, as expected, but that this
> has **no regularizing effect** of making the computed function simpler. The function `y` is exactly
> the same, regardless of the scale of the weights. And **the parameter `λ` has no impact on the
> optimum, since the weights can be scaled to compensate.**"

> "While **the scale of weights `w` has no effect on the objective value**, with first order
> optimization methods such as stochastic gradient, the scale of `w` does influence the updates that
> are performed."

**Verdict on ADR-0010's central claim.** *"A scale-invariant objective plus a scale-invariant penalty
leaves the map's magnitude unidentified by the objective, so fixing it removes a free parameter rather
than capping a learned one"* is **the standard reading of this literature**, restated for restriction
maps. The ticket's worry that it was constructed rather than read is answered: it was constructed, and
it happens to be right, and it has a name.

The specific thing the ticket asked to check — an L1 that no longer shrinks anything — has its exact
analogue in Kalofolias' eq. (4), which shows an L1 term on a scale-free graph-learning objective
degenerates into a constant offset rather than a sparsifier:

> "`tr(XᵀLX) + γ‖W‖_{1,1} = ‖W ∘ (2γ + Z)‖_{1,1}`"

> "As we showed with eq. (4), adding an ℓ-1 norm term for this reason is not very useful: **it just
> adds the same constant to all pairwise squared distances.** Adding a Frobenius norm is a wiser
> choice in this case."

This does **not** transfer directly — Patchworks' L1 is on the *normalised* map, where it redistributes
across a map's directions rather than adding a global offset, and 07's "redistributes weight across
the map's directions rather than removing it" is the correct description of what survives. But it is
the nearest published instance of the failure mode the ticket named, and it lands on the same
conclusion Patchworks reached from the other side: on a scale-free objective, an L1 does not do the
job an L1 looks like it is doing.

### 2.2 van Laarhoven §5.5 is ADR-0010's enforcement mechanism, including its rejection of reparameterisation

This is the closest match found anywhere in the pass, and it is close enough to quote in full:

> "**5.5 Normalizing Weights.** A brute force approach to avoid the interaction between the
> regularization parameter and the learning rate is to **fix the scale of the weights**. We can do this
> by rescaling the `w` to have norm 1: `w̃_{t+1} ← w_t − η∇L_λ(w_t)`; `w_{t+1} ← w̃_{t+1}/‖w̃_{t+1}‖₂`.
> With this change, the scale of the weights obviously no longer changes during training… **Note that
> this weight normalizing update is different from Weight Normalization, since there the norm is taken
> into account in the computation of the gradient, but is not otherwise fixed.**"

Three things ADR-0010 arrived at independently are all in that paragraph: (i) take the step, then
project the norm back — the ADR's "Enforcement is projection"; (ii) fixing the scale by projection is
the remedy for a scale-invariant objective, not a cap on a learned quantity; (iii) it is **not** the
same as reparameterising, and the distinction matters. ADR-0010 rejected `F = G/‖G‖_F` because it
"leaves a shadow parameter `G` that the sparsity term can drive toward zero." Salimans & Kingma's own
analysis confirms the shadow parameter is real and does not sit still — though it moves the other way:

*Source: Salimans, T. & Kingma, D. P. (2016). "Weight Normalization," NeurIPS 2016,
[arXiv:1602.07868](https://arxiv.org/abs/1602.07868) — read via ar5iv.*

> "`Δv` is necessarily orthogonal to the current weight vector `w` since we project away from it when
> calculating `∇_v L`."

> "the norm of `v` **grows monotonically** with the number of weight updates"

So the shadow parameter is a real object with its own uncontrolled dynamics — which is the ADR's
reason for rejecting the reparameterisation, confirmed. The *direction* is the next section's problem.

### 2.3 The threat: the drift direction in ADR-0010 is contradicted

ADR-0010 says:

> "Left free, the magnitude does not sit still — it drifts, and the direction it drifts under any
> residual asymmetry is the one route one names."

Route one is collapse toward `F = 0`. The literature proves the opposite direction. Arora et al.'s
key lemma:

> "**Lemma 2.4.** For any scale-invariant weight `w(i)` in the network `Φ`, we have: 1. `w_t(i)` and
> `∇_{w_t(i)} F_{z_t}(θ_t)` are always **perpendicular**; 2.
> `‖w_{t+1}(i)‖₂² = ‖w_t(i)‖₂² + η_{w,t}² ‖∇_{w_t(i)} F_{z_t}(θ_t)‖₂²`."

Perpendicularity is not approximate and not conditional on the loss — it follows from scale-invariance
alone, by differentiating `F(w) = F(cw)` in `c`. The Pythagorean consequence is that the squared norm
is **non-decreasing at every step**, strictly increasing whenever the gradient is nonzero. Salimans &
Kingma state the same for their `v`. Ioffe & Szegedy's original observation, quoted by Arora et al.,
is that this is the *generic* behaviour:

> "even though the scale of weight parameters of a linear layer proceeding a BatchNorm no longer means
> anything to the function represented by the neural network, **their growth has an effect of reducing
> the learning rate**."

**What this means for Patchworks.** If both terms of the transport rule are scale-invariant — the
relative disagreement objective is, by ADR-0010's own construction, and the L1 on the *normalised* map
is too — then Lemma 2.4 applies to `F` directly. The free magnitude does not drift toward zero; it
**grows**, and what it produces is not collapse but a vanishing effective learning rate: maps that
stop moving while every diagnostic reports a healthy nonzero norm. That is a distinct failure from
route one, arguably harder to see, and the *upper* end of the band is what closes it.

This does not unsettle the decision. It arguably strengthens it: ADR-0010 lists "a norm floor alone,
with no upper bound" under alternatives considered and rejects it on a staleness argument; the
literature supplies a second and more fundamental reason to reject it, namely that the upper bound is
the one the dynamics actually need. But the ADR's stated reason for *why* the free parameter is
dangerous points the wrong way, and one sentence should be corrected. See R1.

A caveat that keeps this honest: Lemma 2.4 assumes plain gradient descent on an exactly scale-invariant
loss. Patchworks' transport rule is composed in the same step as a projection, and the disagreement
normaliser `‖F_u x_u‖ + ‖F_v x_v‖` is exactly scale-invariant only when both ends scale together. Under
a one-sided rescaling the ratio is not invariant, so a residual asymmetry does exist and the ADR's
instinct that *something* moves is right. What is not supported is the claim that the asymmetry points
downward by default.

---

## 3. Q3 — The transformer-as-sheaf framing

**Verdict: the claim as recalled — "a transformer is a sheaf on a GNN with zero coboundary" — does
not exist in the literature. Two real claims stand near it, and neither is that.**

### 3.1 What actually exists

**(a) The GAT-as-sheaf theorem.** *Source: Hu, C.-S. (2026). "A Sheaf-Theoretic and Topological
Perspective on Complex Network Modeling and Attention Mechanisms in Graph Neural Models,"
[arXiv:2601.21207](https://arxiv.org/abs/2601.21207) — read from PDF.*

> "**Theorem 1.** Let `(G, (s_vi), W)` denote a triple defined as in Equation (6) … Then, a cellular
> sheaf `F : (G, ⊴) → Vect_R` is defined as follows: (a) `F_v := R^d` for each `v ∈ V`; (b)
> `F_e := R^d` for each edge `e ∈ E`; and (c) `F_{v_i ⊴ {v_i,v_j}} : F_{v_i} → F_{{v_i,v_j}}` is the
> **scalar multiplication by `w_ij`** for every edge `{v_i, v_j} ∈ E`."

The restriction maps are scalar multiples of the identity, with the scalar being the trained attention
weight. The coboundary is emphatically **not** zero — if it were, the model would compute nothing, and
the paper's whole apparatus is built on the coboundary being informative. What the paper does have is
a notion the recollection may have compressed into "zero coboundary":

> "**Definition 1.** An edge `e ∈ E` with endpoints `v, w ∈ V` is called a **harmonic edge** if
> `t_e = 0`, that is, if `F_{v,e} s_v = F_{w,e} s_w`."

`t = C⁰s`, so a harmonic edge is one where the coboundary vanishes **on the current signal** — i.e.,
in Patchworks' terms, an edge with zero disagreement. That is a property of a section, not of the
sheaf. A second candidate for the compression is the standard fact that on a graph there are no
2-cells so `δ¹ = 0` identically, which is true of every sheaf on every graph and says nothing about
transformers.

**(b) The transformer↔GAT identification, which is informal and not sheaf-theoretic.** *Source:
Barbero, F. (2022). "Attention-based Sheaf Neural Networks," MPhil dissertation, Cambridge MLMI —
read from the
[departmental PDF](https://www.mlmi.eng.cam.ac.uk/files/2021-2022_dissertations/attention-based-sheaf-neural-networks.pdf).*

> "**Remark.** Perhaps one of the most important breakthroughs in the machine learning world … has
> been the proposal of the Transformer (Vaswani et al., 2017). The Transformer and GAT models happen
> to be close cousins. **One may in fact imagine a transformer as a graph attention network operating
> over a fully connected graph** (Joshi, 2020)."

The citation is to Joshi, C. (2020), "Transformers are Graph Neural Networks," *The Gradient* — a blog
essay, not a theorem. And Barbero is explicit that the sheaf-transformer composition **had not been
done**:

> "Effectively, our models are sheaf generalisations of GAT, which in turn is a graph generalisation of
> a Transformer. As a consequence, **exploring sheaf Transformer models may prove to be very fruitful**
> in a lot of Transformer based tasks."

**Conclusion.** The recalled claim is a chain of two steps that no single source takes: Hu's GAT sheaf
(rigorous, 2026) composed with Joshi's transformer-as-GAT-on-a-complete-graph (informal, 2020), with
"zero coboundary" apparently imported from Hu's *harmonic edge* or from the trivial `δ¹ = 0`. **No
source asserts the composite.** Recorded as *no precedent found*, which is a result and not a failure.

### 3.2 Does any of it bear on whether `ρ = 1` is a beneficial special case?

**On the attention analogy specifically: no — and the reason is worth keeping.** Under the only
rigorous sheaf reading of attention that exists, the normalisation attention performs is:

> "the coefficients `w_ji` are typically assumed to lie within the closed interval `[0,1]` and satisfy
> **the normalization condition `Σ_j w_ji = 1`**."

That is a **per-node budget over incident edges** — an L1 simplex constraint on a node's whole
neighbourhood — not a norm fixed on each map. An individual restriction map's Frobenius norm under
Theorem 1 is `|w_ij| · √d` and is free anywhere in `[0, √d]`; a node may put nearly all its budget on
one edge and leave the rest near zero. So softmax is structurally **Hansen & Ghrist's `tr(L_ii)`
barrier**, not ADR-0010's per-map band, and certainly not `ρ = 1`. If Patchworks ever wants the thing
attention actually does, the per-node form is the one to reach for — and it is a different constraint
from the one held open in the ADR's consequences.

**On exact fixing being beneficial more generally: yes, with three independent supports.**

1. **It removes a coupling, which is van Laarhoven's stated motivation** (§2.2 above): "A brute force
   approach to avoid the interaction between the regularization parameter and the learning rate is to
   fix the scale of the weights … the effective rate no longer depends on the regularization parameter
   `λ`." Exact fixing buys a decoupling that a *band* only bounds — inside a band of width `ρ²` the
   effective learning rate still varies by `ρ²`.
2. **Exact fixing is what buys a guarantee, in the one place a norm constraint is used for a
   theorem.** *Source: Miyato, T. et al. (2018), "Spectral Normalization for Generative Adversarial
   Networks," ICLR 2018, [arXiv:1802.05957](https://arxiv.org/abs/1802.05957) — read via ar5iv.* Their
   constraint is an equality, `σ(W) = 1`, and it is the equality that yields the Lipschitz bound. A
   band would not.
3. **The unit sphere is the canonical domain for a scale-invariant parameter.** *Source: Kodryan, M.
   et al. (2022), "Training Scale-Invariant Neural Networks on the Sphere Can Happen in Three
   Regimes," NeurIPS 2022, [arXiv:2209.03695](https://arxiv.org/abs/2209.03695) — abstract read at
   source.* "**The intrinsic domain of such parameters is the unit sphere**, and therefore their
   gradient optimization dynamics can be represented via spherical optimization with varying effective
   learning rate (ELR)." They then find "three regimes of such training depending on the ELR value:
   convergence, chaotic equilibrium, and divergence" — i.e. fixing the norm exactly does not
   trivialise the dynamics; it *reveals* them, by moving all the remaining behaviour into one
   interpretable scalar.

**Net.** There is real evidence that exact fixing buys something — decoupling, provable bounds, and a
cleaner dynamical picture. There is **no** evidence that it buys what the ADR's `ρ = 1` note gestures
at, because attention's normalisation is a different constraint entirely. ADR-0010's reason for
keeping `ρ > 1` (scale stays private to a cell; pinning it chains near-uniform stalk scale across the
graph via connectivity) is untouched by anything read here, and remains the live consideration. The
`ρ = 1` door stays open on the ADR's own terms, not on an attention analogy.

---

## 4. Unasked but load-bearing: Miyato's rank result cuts against ADR-0010's Frobenius argument

The ADR's section *Frobenius, not spectral — and therefore no rank floor* argues:

> "Unit Frobenius norm pins the sum of squares across all directions, so concentrating a map onto
> fewer directions *buys* per-direction gain and spending it across more *pays*. Rank concentration
> becomes a priced trade rather than a free lunch."

Miyato et al. run the identical algebra and reach the opposite normative conclusion. Their §3 proves
that weight normalisation implies

> "`σ₁(W̄_wn)² + σ₂(W̄_wn)² + ⋯ + σ_T(W̄_wn)² = d_o`"

— a fixed Frobenius budget — and then:

> "the norm `‖W̄_wn h‖₂` for a fixed unit vector `h` **is maximized** at `‖W̄_wn h‖₂ = √d_o` when
> `σ₁(W̄_wn) = √d_o` and `σ_t(W̄_wn) = 0` for `t = 2,…,T`, which means that **`W̄_wn` is of rank one.**"

> "Using such `W` corresponds to **using only one feature** to discriminate the model probability
> distribution from the target."

And their reason for choosing spectral instead:

> "Note that the Lipschitz constant of a linear operator is determined only by the maximum singular
> value. In other words, **the spectral norm is independent of rank** … our spectral normalization
> allows the parameter matrix to **use as many features as possible** while satisfying local
> 1-Lipschitz constraint."

**Reading.** The ADR and Miyato agree on the mathematics — under a Frobenius budget, concentration
buys per-direction gain. They disagree on whether that is a price or a reward, and the disagreement is
resolved by what the objective wants. Where the objective rewards large output gain in *some*
direction, a Frobenius budget makes rank one the **maximiser**, not a priced trade, and Miyato et al.
abandoned Frobenius for exactly that reason. Where the objective is genuinely scale-invariant —
Patchworks' case, if the relative normaliser holds — the incentive to concentrate is weaker, because
gain in the numerator is divided out of the denominator.

The ADR is not wrong to prefer Frobenius: it explicitly *wants* learned rank-deficiency
(`06-graph-topology.md`'s functionally dead but structurally present edge) and it already instruments
for the degenerate limit with per-edge effective rank. But "a priced trade rather than a free lunch"
is stronger than the algebra supports, and the one paper that examined this exact budget concluded the
opposite. Worth softening and citing. See R2.

**Note the pleasing symmetry with the ADR's own spectral argument.** ADR-0010 rejects spectral norm
because it "excludes `F = 0` while leaving `F → rank 1` wide open." Miyato rejects Frobenius because
it drives toward rank 1. Both are right about their own objective. The instrument ADR-0010 already
built — per-edge effective rank on the diagnostic cadence — is the only thing that settles which
regime Patchworks is actually in, which is a good argument that the instrument was the right call.

---

## Candidate revisions to the spec

Two, both to ADR-0010, both **corrections to stated reasoning rather than to the decision**. The
decision — a Frobenius band on interior maps, the exact gauge on boundary maps, enforced by projection
— survives this pass intact and better supported than it was.

### R1 — ADR-0010 states the wrong drift direction for the free magnitude

**Where:** `docs/adr/0010-restriction-map-scale-is-gauge-fixed.md`, *The transport objective is
scale-invariant, which leaves magnitude unidentified*, final sentence: "Left free, the magnitude does
not sit still — it drifts, and the direction it drifts under any residual asymmetry is the one route
one names."

**What the literature says:** Arora, Li & Lyu ([arXiv:1812.03981](https://arxiv.org/abs/1812.03981)),
Lemma 2.4: for a scale-invariant parameter the gradient is *always perpendicular* to the weight, and
`‖w_{t+1}‖² = ‖w_t‖² + η²‖∇‖²`. The norm is non-decreasing at every step. Salimans & Kingma
([arXiv:1602.07868](https://arxiv.org/abs/1602.07868)) observe the same for `v`: "the norm of `v` grows
monotonically with the number of weight updates." The known failure of an unconstrained scale-invariant
parameter is **growth with a vanishing effective learning rate**, not shrinkage.

**Why it matters:** the sentence is the ADR's bridge from "unidentified" to "dangerous," and it points
at route one. If the drift is upward, the danger is frozen maps that every norm-based diagnostic reads
as healthy — a failure the ADR's instrument pair does not currently name. The band closes it via its
*upper* bound, so no mechanism changes.

**Ask:** replace the sentence with the correct direction, cite Arora et al. Lemma 2.4 and Salimans &
Kingma, and add one line to *The instrument is a pair* or to `01-cell-and-sheaf.md`'s *Known exposure*
naming vanishing-effective-learning-rate as what the upper bound buys. Optionally strengthen
*Alternatives considered* → "A norm floor alone": the literature supplies a second reason the upper
bound is not optional.

### R2 — "Rank concentration becomes a priced trade" is stronger than the algebra supports

**Where:** `docs/adr/0010-restriction-map-scale-is-gauge-fixed.md`, *Frobenius, not spectral — and
therefore no rank floor*.

**What the literature says:** Miyato et al. ([arXiv:1802.05957](https://arxiv.org/abs/1802.05957), §3)
prove that under a fixed sum of squared singular values, output gain in a fixed direction is
**maximised** at rank one, and abandoned Frobenius-style normalisation for spectral on that ground:
"the spectral norm is independent of rank … allows the parameter matrix to use as many features as
possible."

**Ask:** keep the decision and the reasoning's shape, but soften "priced trade rather than a free
lunch" to reflect that the price is only positive when the objective does not reward concentration,
and cite Miyato et al. as the counter-case. This also strengthens the ADR's existing justification for
per-edge effective rank as an instrument: it is the measurement that distinguishes the two regimes.

### Non-revisions worth folding in opportunistically

- **`Alternatives considered` → "Orthogonality, Di Nino et al.'s own constraint"** currently closes
  "Whether the weaker constraint is known to suffice is the open question handed to #53." That question
  is now answered — Hansen & Ghrist (ICASSP 2019) impose a Frobenius-norm floor on a node's incident
  restriction maps with no basis constraint, for exactly this purpose, and relax *out of* the
  orthogonal class deliberately. The sentence can be replaced with the citation.
- **The ADR's `ρ = 1` consequence** can note that attention's normalisation is not the precedent it
  might look like — `Σ_j w_ji = 1` is a per-node budget, not a per-map gauge — while the general case
  for exact fixing (van Laarhoven §5.5; Miyato's `σ(W) = 1`; the unit sphere as intrinsic domain) does
  hold.
- **`07-local-learning-rule.md`** needs no change. Its description of the rule is accurate throughout.

## Sources read

| Source | Access |
| --- | --- |
| Hansen & Ghrist (2019), *Learning Sheaf Laplacians from Smooth Signals*, ICASSP | [PDF](https://www.jakobhansen.org/publications/learningsheaves.pdf), read in full |
| Di Nino, Barbarossa & Di Lorenzo (2025), [arXiv:2501.19207](https://arxiv.org/abs/2501.19207) | PDF, §§III–IV read |
| Dong, Thanou, Frossard & Vandergheynst (2016), [arXiv:1406.7842](https://arxiv.org/abs/1406.7842) | PDF, §IV read |
| Kalofolias (2016), *How to Learn a Graph from Smooth Signals*, AISTATS | [PDF](http://proceedings.mlr.press/v51/kalofolias16.pdf), §§3–4 read |
| Arora, Li & Lyu (2019), [arXiv:1812.03981](https://arxiv.org/abs/1812.03981) | PDF, §§1–2 and §4 read |
| van Laarhoven (2017), [arXiv:1706.05350](https://arxiv.org/abs/1706.05350) | PDF, read in full |
| Salimans & Kingma (2016), [arXiv:1602.07868](https://arxiv.org/abs/1602.07868) | ar5iv, §§2–3 read |
| Miyato et al. (2018), [arXiv:1802.05957](https://arxiv.org/abs/1802.05957) | ar5iv, §3 and Appendix E read |
| Kodryan et al. (2022), [arXiv:2209.03695](https://arxiv.org/abs/2209.03695) | abstract only — full text not read, and nothing beyond the abstract is claimed here |
| Hu (2026), [arXiv:2601.21207](https://arxiv.org/abs/2601.21207) | PDF, §§ on GAT sheaf and harmonic sets read |
| Barbero (2022), *Attention-based Sheaf Neural Networks*, Cambridge MPhil | [PDF](https://www.mlmi.eng.cam.ac.uk/files/2021-2022_dissertations/attention-based-sheaf-neural-networks.pdf), §2 remark and conclusions read |
| Joshi (2020), *Transformers are Graph Neural Networks*, The Gradient | not read directly; cited only as Barbero's cited source for the transformer↔GAT remark |
