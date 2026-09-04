# Citation pass: the cell contract and its sheaf (patchworks#16)

Validates `docs/spec/01-cell-and-sheaf.md` — the **cell** half, and the four sheaf commitments
`015-sheaf-geometry.md` did not reach. Citations validate after the fact per the map's Notes; this
document does not revise the closed design. Patchworks' side of every comparison is described in
its own terms (`CONTEXT.md`); the prior art's side in its own field's terms. Where a source could
not be reached, that is stated rather than papered over.

**Inherited from [`015-sheaf-geometry.md`](./015-sheaf-geometry.md), not rediscovered here:**
Patchworks sits **outside the hypothesis class of every learned-sheaf paper**. All of them assume
invertible or orthogonal restriction maps; Patchworks' are masked, sparse, and deliberately
rank-deficient. Their `H⁰`/`H¹` theorems do not apply, and their results are orientation, not
authority. §4 below finds the *reason* the literature imposes that hypothesis, which 015 did not
have, and it is load-bearing.

---

## Headline

Three findings worth reading before the rest.

1. **§4 — the literature states, in one sentence, the reason every learned-sheaf paper constrains
   its restriction maps: without a constraint, minimising disagreement has a trivial global
   minimum at `F = 0`.** Patchworks' transport rule descends on disagreement *and* adds a sparsity
   pressure pushing the same weights toward zero. Both terms point at the same degenerate solution.
   This is the sharpest thing found in this pass and it is a candidate revision, not a footnote.
2. **§6 — the canonical-microcircuit rationale for a fixed uniform `n` does not survive contact
   with the literature.** The specific proposition the spec leans on — that a cortical column has
   an efficient, roughly fixed size — is contradicted in print: minicolumns "cannot have constant
   size and number of neurons across different brain areas or across species." What *does* survive
   is a different and better claim the spec already makes elsewhere (uniform circuit, per-area
   variation), which supports uniform *contract*, not uniform *dimension*.
3. **§2 — the predictive-coding/Dirichlet-energy identification is in print, independently
   derived, and it lands on exactly Patchworks' object.** The spec's strongest sentence is its
   best-supported one.

---

## 1. The three-tier split: private chart (`k`) → node stalk (`n`) → communication lane (`m`)

**Verdict: KNOWN VARIANT in two neighbouring literatures, UNUSUAL in the sheaf-network literature
specifically. No sheaf paper has a third, reconciliation-proof tier below the node stalk; two
other fields have exactly that split and have it for Patchworks' stated reason.**

### 1.1 The sheaf-network literature is two-tier, uniformly

Every sheaf neural network surveyed identifies the node stalk **with** the node's representation.
Bodnar et al.'s node stalk `F(v) = R^d` carries the feature channels directly, and diffusion
`X_{t+1} = X_t − σ(Δ_F(t)(I⊗W₁)X_t W₂)` rewrites those same features
([arXiv:2202.04579](https://arxiv.org/abs/2202.04579), Eq. 6) — there is no space underneath that
the Laplacian cannot touch. Hansen & Gebhart's Sheaf Neural Networks
([arXiv:2012.06333](https://arxiv.org/abs/2012.06333)) and Barbero et al.'s connection-Laplacian
variant ([arXiv:2206.08702](https://arxiv.org/abs/2206.08702)) are the same shape. Patchworks'
commitment that "reconciliation edits the node stalk only… never reaches into the chart" has **no
counterpart** in this literature, because there is nothing for it to not reach into.

Read alongside 015 §3.2, this is the same structural fact from a second angle: the sheaf-network
literature has no private tier and no masked directions, so it has neither the mechanism that
creates `H⁰` nor the space that `H⁰` would insulate.

### 1.2 The sheaf literature's *one* private/public split — Hansen & Ghrist's discourse sheaves

The nearest thing in print is Hansen & Ghrist, "Opinion Dynamics on Discourse Sheaves"
([arXiv:2005.12798](https://arxiv.org/abs/2005.12798)), which does split private from public — but
one tier up from where Patchworks does. Each agent's **opinion space is the vertex stalk**; the
**discourse space is the lane**; and the restriction map is explicitly the act of making a
private view public: agents "represent their opinions on the topics of discussion by formulating
stances as a linear combination of existing opinions on personal basis topics." Consistency is
defined only on the public side — "Agents have expressed consensus when `F_{u⊴e}(x_u) =
F_{v⊴e}(x_v)`."

**This validates the *lower* boundary of Patchworks' split, not the split itself.** It is precedent
for "the node stalk is private relative to the lane, and the restriction map is the act of
going public." It is not precedent for a third space beneath the node stalk. Patchworks'
`chart → node stalk` boundary is a second, deeper privacy boundary the discourse-sheaf paper does
not have.

### 1.3 Where the split *does* have precedent: bandwidth-limited multi-agent communication

Foerster et al., "Learning to Communicate with Deep Multi-Agent Reinforcement Learning"
([arXiv:1605.06676](https://arxiv.org/abs/1605.06676), NIPS 2016), builds the split explicitly.
Each agent is a recurrent network (their C-Net) with a **private hidden state** and **two output
heads**: the Q-values of environment actions, and a **separate real-valued message** emitted into a
limited-bandwidth channel. The agent solves in its recurrent state; it communicates in the message
space; the two are different spaces of different dimension, and the message channel is narrow by
construction. That is Patchworks' chart/stalk division with the same motivation — the space you
compute in and the space you are allowed to say things in are not the same space.

This is a **known variant**, then, but from multi-agent RL rather than from geometry: the design
move "separate the space a unit solves in from the space it communicates in" is established, and
the reason given there (bandwidth) is a cousin of the reason given here (recomposition is a
different operation from solving).

### 1.4 What is genuinely without precedent

The specific conjunction — *three* tiers, with the innermost one **structurally immune to the
consistency operator** — was not found anywhere. In Foerster et al. the hidden state is trained by
gradients flowing back through the channel; in the sheaf networks there is no inner tier at all.
Patchworks' "the correction arrives as evidence on the next tick, never as an edit to internal
state" is, as far as this pass can determine, its own.

**Relay cells.** Also without direct precedent in the sheaf literature — 015 §5.2 already recorded
that no sheaf paper treats hub, virtual, or master nodes at all, and the relay cell (own `n`-stalk,
own per-edge learned map, identity `step`) is the degenerate instance of that gap. Southern et
al.'s virtual-node uniformity result ([arXiv:2405.13526](https://arxiv.org/abs/2405.13526),
Prop. 4.1) remains the argument *for* relay cells over virtual nodes, per 015 §5.2.

---

## 2. Disagreement as the sole error signal: is PC error = sheaf Dirichlet energy in print?

**Verdict: SUPPORTED, and the identification is made explicitly, independently, and in print —
by a paper that derived it from the predictive-coding side and landed on Patchworks' exact object.**

**Source:** Seely, J. (2025). "Sheaf Cohomology of Linear Predictive Coding Networks."
[arXiv:2511.11092](https://arxiv.org/abs/2511.11092) (submitted 14 Nov 2025).

The paper states that linear predictive-coding networks "admit a natural formulation as cellular
sheaves: the sheaf coboundary maps activations to edge-wise prediction errors, and PC inference is
diffusion under the sheaf Laplacian." The energy identity is written out: PC energy
"can be written compactly as `E_PC(s) = ½‖δ⁰s‖²`", and PC inference by gradient descent "yields
the gradient flow `ṡ = −Ls` … known as sheaf diffusion" (§3). Sheaf cohomology then "characterizes
irreducible error patterns that inference cannot remove."

Three things follow for the spec, stated precisely:

- **`01-cell-and-sheaf.md`'s "Predictive coding's error and the sheaf's inconsistency are the same
  quantity, not two objects that need relating" is exactly Seely's thesis**, arrived at
  independently and from the opposite direction. This is the single best-supported sentence in the
  document.
- **`H¹` as "irreducible disagreement" is Seely's own reading**, not a Patchworks coinage — and it
  matches ADR-0007's *static floor* (specifically its mask/rank-deficiency component) rather than
  its lag component, which has no counterpart in Seely because his networks have no delay.
- **The identification is for the *linear* case.** Seely's title says "Linear"; the sheaf reading
  is exact because the PC generative maps are linear. Patchworks' restriction maps are linear too
  (`01-cell-and-sheaf.md`), so the identification transfers — but it transfers *because of* the
  linearity commitment, and would not survive nonlinear restriction maps. This makes ADR-0004 and
  the disagreement-is-Dirichlet-energy claim the same commitment wearing two hats, which the spec
  does not currently say.

One nuance carried over from `019-tick-semantics-citations.md`'s flagged inconsistency, which
applies verbatim here: Dirichlet energy is a **global scalar sum over edges**; one cell's
disagreement on one edge is a **summand** of it. Seely's `E_PC(s) = ½‖δ⁰s‖²` is likewise global.
The spec is careful about this in one clause ("one edge's term of the sheaf's Dirichlet energy")
and loose in another ("its squared sum is the Dirichlet energy", CONTEXT.md, which is correct).
No new defect.

**Nothing else in print makes this identification.** Searches across the sheaf-network and
predictive-coding literatures returned Seely as the only source. Bogacz's free-energy tutorial and
Rao & Ballard treat error as a transported quantity with its own neural population and never reach
a Laplacian; the sheaf papers reach the Laplacian and never mention predictive coding. Seely is the
bridge, and he is the only one.

---

## 3. Penalised vs. enforced consistency

**Verdict: KNOWN VARIANT — and the literature splits cleanly along a line the spec is on the
correct side of, for reasons the spec correctly states. But the sheaf literature's own
optimization branch enforces the constraint *exactly*, and names what the penalty term is for,
which is not what Patchworks uses it for.**

### 3.1 The soft branch: diffusion, and it is the neural-network branch

Hansen & Ghrist's discourse-sheaf dynamics are gradient flow on the Dirichlet energy —
`dx/dt = −αL_F x`, i.e. gradient descent on `Ψ(x) = ½‖δx‖²`
([arXiv:2005.12798](https://arxiv.org/abs/2005.12798)). Bodnar et al. discretise the same flow as a
residual update run for `T` layers ([arXiv:2202.04579](https://arxiv.org/abs/2202.04579), Eq. 6).
Neither ever projects onto the consistent subspace in one shot. **Every sheaf neural network is on
the penalty side of this line.** Patchworks' "penalised, not enforced" is therefore conventional
for the neural-network branch of the field.

The divergence is not softness but *count*: their flows converge (Hansen & Ghrist's Theorem 5 has
trajectories converging to `H⁰`) or run `T` tuned layers; Patchworks runs exactly one step per tick
and never converges by construction (ADR-0002; `019-tick-semantics-citations.md` covers this).
Asymptotically, a converging soft flow **is** enforcement. Patchworks' one-step rule is what keeps
"penalised, never cleared" true rather than true-in-the-limit-only, and that is a sharper statement
than the spec currently makes.

### 3.2 The hard branch exists, is also Hansen & Ghrist's, and names the penalty's real job

Hansen, J. & Ghrist, R. (2019). "Distributed Optimization with Sheaf Homological Constraints."
*57th Allerton Conference*, 565–571.
[PDF](https://jakobhansen.org/publications/distopt.pdf) — read directly.

Here consistency is a **hard constraint**. The problem is stated as

```
min_x  Σ_{v∈V} f_v(x_v) + xᵀ L_F x        s.t.   L_F x = 0
```

and the paper notes `x ∈ H⁰(G;F)` "is equivalent to `L_F x = 0`". It is solved by Lagrangian
saddle-point dynamics with a dual variable `z`:

```
ẋ = −∇f(x) − 2L_F x − L_F z
ż = L_F x
```

— "gradient descent in the primal variable `x` and gradient ascent in the dual variable `z`."

**The load-bearing sentence, quoted verbatim, is about the penalty term:**

> "Although this is not strictly necessary, we add the term `xᵀ(L⊗I)x` to the objective function,
> **to improve stability and convergence** of the resulting algorithm **while leaving the minimum
> and minimizer undisturbed**."

So in the sheaf literature's optimization branch, the Dirichlet-energy penalty is a *stabiliser*,
and the thing that actually makes agreement hold is the **dual variable**. Patchworks keeps the
penalty and discards the dual variable. That is a coherent choice and the spec's reason for it is
sound — a projection "would zero out the quantity the architecture runs on" — but the precise
statement of what is given up is: **Patchworks has no mechanism that drives residual disagreement
to zero even asymptotically, and this is deliberate.** The spec says this in prose ("never clears
it"); the citation makes it exact by exhibiting the machinery that would.

Worth recording: the hard branch is also the *distributed* branch. Hansen & Ghrist's saddle-point
dynamics are graph-local — each node needs only `L_F x` and `L_F z`, both computable from
neighbours. So the enforcement alternative is **not** ruled out by graph-locality; it is ruled out
only by the design's wish to keep disagreement non-zero and by ADR-0002's one-step rule. The spec's
sentence "would drag a global solve into a system whose thesis is locality" **overstates the case**
— the exact-enforcement algorithm in this literature is not a global solve. It is a candidate
revision (§ Candidate revisions, R3).

---

## 4. Learned linear restriction maps with structural masks

**Verdict: SUPPORTED for "learned" and for "linear" (015 §4.4 already settled linearity).
CONTRADICTED — or at minimum, seriously exposed — on the conjunction of *learning maps by
descending on disagreement* with *a sparsity pressure on the same maps*. The literature names the
degenerate solution that pairing runs at.**

### 4.1 The finding: every learned-sheaf method constrains its maps *to avoid a trivial solution*

**Source:** Di Nino, L., Barbarossa, S. & Di Lorenzo, P. (2025). "Learning Sheaf Laplacian
Optimizing Restriction Maps." [arXiv:2501.19207](https://arxiv.org/abs/2501.19207) — read directly
from the PDF.

They infer the sheaf Laplacian, restriction maps included, by minimising total variation (the
Dirichlet energy) over the maps. And they state the obstruction plainly (§III):

> "We also need to impose a constraint on the set of feasible restriction maps `F`, **to avoid the
> trivial solution.**"

Their chosen constraint is orthogonality — the local problem is
`min_{F_{u◁e}} ‖F_{u◁e}D_uS_u − D_vS_v‖²  s.t. F_{u◁e}ᵀF_{u◁e} = I`, "solved as an orthogonal
Procrustes problem … in closed form via singular value decomposition."

**This retro-explains 015's headline.** 015 recorded *that* Bodnar (`H_sym`, `H^d`, `H^d_diag`,
`H^d_orth`), Barbero (`O(d)` by SVD projection) and the rest all force invertibility or
orthogonality, and that Patchworks is outside every one of those hypothesis classes. Di Nino et al.
say **why** the constraint is there, and the reason is not geometric elegance: *disagreement
minimised over the restriction maps is trivially and globally solved by `F = 0`.* Set every
restriction map to zero and every edge agrees perfectly, forever, at zero cost.

### 4.2 Why this bites Patchworks specifically

`CONTEXT.md` defines the **transport rule** as "a local gradient step on disagreement, composed in
the same step with the sparsity pressure," and `01-cell-and-sheaf.md` makes restriction maps
"**Learned**, under a sparsity pressure." Those are the two terms of the degenerate objective:

- the disagreement term is globally minimised at `F = 0`;
- the sparsity term is *also* globally minimised at `F = 0`;
- and unlike every cited method, Patchworks imposes **no** orthogonality, invertibility, norm, or
  rank constraint that would exclude it. The structural mask restricts *which entries may be
  nonzero*; it does not require any of them to be.

Nothing in the pass suggests this is fatal — a slow rule, a bounded step, and the bias rule's
independent pressure on prediction error may well keep the maps away from zero in practice, and
the spec's own reading of `H⁰` says shrinking the maps is *partly desirable*. But the design
currently has **no stated mechanism** that makes `F = 0` unreachable, and the one literature that
has tried to learn restriction maps by exactly this objective says a constraint is needed. This is
the pass's strongest candidate revision (R1).

Note also the interaction with 015 §3.2's result: sparsification enlarges `H⁰` *and* `H¹` together
(the Euler characteristic pins the difference). `F → 0` is the limit of that process: `H⁰` becomes
everything, `H¹` becomes everything, and the sheaf stops coupling anything at all. The spec's
"`H⁰` is large by construction and enlarged by sparsity, so consistency is not scarce" is true and
is also the first half of a slope whose bottom is a disconnected graph.

### 4.3 Learned restriction maps: is *learning* them established?

**Yes, and it is now the norm.** Bodnar et al.'s central contribution is learning `F_{v◁e}` from
node features rather than hand-specifying them ([arXiv:2202.04579](https://arxiv.org/abs/2202.04579));
Barbero et al. learn them via local PCA and SVD projection onto `O(d)`
([arXiv:2206.08702](https://arxiv.org/abs/2206.08702)); Di Nino et al. learn them jointly with graph
topology ([arXiv:2501.19207](https://arxiv.org/abs/2501.19207)). Patchworks' "learned" is
conventional. Its **mask** is not — no surveyed paper imposes a hand-specified structural mask on
which node-stalk features may participate on which edge, fixed at construction and never re-opened.
That is Patchworks' own, and 015 already recorded it as the reason the field's `H⁰` theorems lapse.

**Tempering result, worth carrying forward.** A 2026 study, "Learned, Relied Upon, or Necessary?
Separating Checkpoint Dependence from Task-Level Value in Sheaf GNNs"
([arXiv:2607.25387](https://arxiv.org/abs/2607.25387)), argues that learned restriction maps in
sheaf GNNs are "often treated as proof that the model has discovered useful edge geometry" and
examines that claim critically. Together with the holonomy-intervention study 015 §4.4 already
records, the field has two recent papers saying the learned sheaf structure may not be doing the
work it is credited with. Patchworks should not cite "learned restriction maps work" as settled.

### 4.4 Is linearity known to be limiting?

**Settled in 015 §4.4; not re-run.** Summary of what stands: nonlinear restriction maps are
essentially unexplored; the nonlinear sheaf work keeps `δ` linear and inserts nonlinearity on the
lane *after* restriction (`L^Φ_F = δᵀ∘Φ∘δ`), for the stated reason that convex edge
potentials give a Lyapunov function; making `δ` itself nonlinear destroys `L = δᵀδ`, the Hodge
decomposition, and the cohomological reading. The one empirical trial of a nonlinear coupling did
not outperform linear sheaf models. **No paper argues for nonlinear restriction maps.** The spec's
linearity commitment is the same call the literature made, for the same reason.

---

## 5. Unit delay: every edge costs exactly one tick

**Verdict: SUPPORTED, and it is a KNOWN VARIANT with two distinct precedents — one that
*derives* the rule from physical communication constraints (and gets Patchworks' rule exactly),
and one that adopts it as a deliberate architectural device and reports what it buys.**

### 5.1 Derived from physics: delayed-aggregation GNNs in decentralized control

**Source:** Tolstaya, E., Gama, F., Paulos, J., Pappas, G., Kumar, V. & Ribeiro, A. (2019).
"Learning Decentralized Controllers for Robot Swarms with Graph Neural Networks." *CoRL 2019*.
[arXiv:1903.10527](https://arxiv.org/abs/1903.10527) — read via the HTML full text.

Their aggregation sequence at node `i` is `z_{in} = [[y_{0n}]_i; [y_{1n}]_i; …; [y_{(K−1)n}]_i]`
(their Eq. 6), where "the `k`th element is `y_{kn} = (S_n S_{n−1} … S_{n−k+1}) x_{n−k}`."

Read that off: the `k`-hop term is built from the graph shift applied at `k` successive **time
steps**, acting on data `x` from **`k` time steps ago**. Information from `k` hops away is exactly
`k` time steps old. **That is Patchworks' unit-delay rule, character for character** — arrived at
independently, and derived rather than chosen: in a real robot swarm each hop *is* one round of
radio exchange, so the delay is not a modelling decision but a fact about the hardware.

This is the strongest validation in the pass for "graph distance is literally temporal distance."
It also validates the spec's framing that the delay makes transport "a real channel with its own
structure": in a swarm, predicting what a neighbour will have sent is genuine modelling for exactly
the same reason.

What it buys, in their terms: decentralized execution with only one-hop radio, at the cost that
"explicit multi-hop message passing between all team members … incurs a superlinear growth in
communications with team size." They report empirically that accepting the delay beats a strictly
local (zero-hop) controller substantially, and that the value of multi-hop information rises as the
communication radius shrinks and speeds increase — the analogue of Patchworks' "depth buys
horizon."

### 5.2 Adopted as a device: DRew

**Source:** Gutteridge, B., Dong, X., Bronstein, M. & Di Giovanni, F. (2023). "DRew: Dynamically
Rewired Message Passing with Delay." *ICML 2023*.
[arXiv:2305.08018](https://arxiv.org/abs/2305.08018) — read via the HTML full text.

`νDRew` aggregates from distance-`k` neighbours using their state from `τ_ν(k) = max(0, k − ν)`
layers earlier:

```
a_{i,k}^(ℓ) = AGG_k^(ℓ)({ h_j^(ℓ − τ_ν(k)) : j ∈ N_k(i) }),   1 ≤ k ≤ ℓ+1
```

The paper states the principle directly — "nodes that are closer should interact earlier in the
architecture" and "distant nodes always interact with a fixed delay given by their distance" — and
that at `ν = 1` node `i` sees "the state of `j` as it was `k−1` layers ago." **`ν = 1` DRew is
Patchworks' unit-delay rule** (up to DRew adding direct long-range edges that Patchworks does not
have; Patchworks relays instead).

What DRew says delay buys and costs, verbatim in substance:

> "Larger delay (i.e. smaller value of `ν`), means that `i` aggregates the features from `j` before
> they are (significantly) 'smoothed' by repeated message passing. Conversely, a smaller delay
> (i.e. larger value of `ν`), implies that when `i` communicates with `j`, it also leverages the
> structure around `j`."

So delay **buys freshness against over-smoothing** and **costs contextualisation** — a neighbour's
delayed state has not yet been mixed with its own neighbourhood. The paper positions delay as a
remedy for over-smoothing in deep MPNN architectures, and reports it helps on long-range tasks.

**This is a real addition to the spec's account.** `01-cell-and-sheaf.md` argues delay is a phase
shift that "removes no frequency content" and therefore costs nothing but staleness. DRew's
trade-off is a *second* axis the spec does not name: what a delayed value has *not yet been mixed
with*. On DRew's reading that is a benefit (less smoothing) rather than a cost, which cuts in
Patchworks' favour — but the spec should not claim delay is cost-free on the information axis when
the literature that studies the knob describes it as a genuine trade.

### 5.3 What was not found

No source states unit delay as a *locality guarantee* — the spec's argument that with per-edge
delay "there is no 'now' spanning the graph, so a global aggregation step is not expressible." That
inference is Patchworks'. It is sound (Tolstaya et al. rely on precisely this to get decentralized
execution) but it is not stated as a principle anywhere read.

Jacobi vs. Gauss–Seidel is covered in `019-tick-semantics-citations.md` and was not re-run. Spiking
GNNs and systolic-array formulations were searched but nothing was found that adds to the two
sources above; that is a limit of this pass's reading, not a claim of absence.

---

## 6. Canonical microcircuit / Mountcastle columnar uniformity

**Verdict: The spec leans on this in TWO places and they have different standings.
"A cortical column has an efficient, roughly fixed size" — CONTRADICTED as a settled fact; it is a
live dispute at best, and the word "efficient" is supported by nobody.
"A uniform circuit repeats across cortex" — SUPPORTED, in a qualified form that supports uniform
*contract*, which is what the spec argues elsewhere, and NOT uniform *dimension*, which is what the
spec uses it to justify.**

The sentence under test, `01-cell-and-sheaf.md`:

> "`n` is fixed on a canonical-microcircuit rationale: a cortical column has an efficient size, and
> cells are the analogue."

### 6.1 The column concept is in serious trouble, and this is the mainstream position

- **Horton, J.C. & Adams, D.L. (2005). "The cortical column: a structure without a function."**
  *Phil. Trans. R. Soc. B* 360, 837–862.
  [PMC1569491](https://pmc.ncbi.nlm.nih.gov/articles/PMC1569491/). Abstract, verbatim: "This year,
  the field of neuroscience celebrates the 50th anniversary of Mountcastle's discovery of the
  cortical column. In this review, we summarize half a century of research and come to the
  disappointing realization that **the column may have no function**." And: "Although the column is
  an attractive concept, **it has failed as a unifying principle for understanding cortical
  function**."
- **Rakic, P. (2008). "Confusing cortical columns."** *PNAS* 105(34), 12099–12100.
  [doi:10.1073/pnas.0807271105](https://www.pnas.org/doi/10.1073/pnas.0807271105). Its opening
  premise: "The term cortical 'column' is used in so many ways that it can be very confusing to the
  nonspecialist." Rakic's point is that "column" names several non-equivalent objects
  (ontogenetic, minicolumn, macrocolumn, hypercolumn) with different sizes and different evidence.
- **da Costa, N.M. & Martin, K.A.C. (2010). "Whose cortical column would that be?"**
  *Front. Neuroanat.* 4:16. [doi:10.3389/fnana.2010.00016](https://doi.org/10.3389/fnana.2010.00016).
  Abstract, verbatim: "no one has 'seen' the anatomy of a column… **There is no obvious
  morphological analog for this functional architecture, in fact much of the anatomical data seems
  to challenge it.**"
- **Haueis, P. & Margulies, D. (2026). "Developing Concepts for Neuroscience: A Philosophical
  Toolkit."** *Eur. J. Neurosci.* 63(2). [doi:10.1111/ejn.70403](https://doi.org/10.1111/ejn.70403).
  A conceptual audit that lays out three unresolved problems — missing anatomical boundaries,
  non-columnar responses, inter- and intra-species variation — and concludes: "'cortical column'
  persistently fails its central epistemic goal: to identify an anatomically discrete, functionally
  modular building block that executes the same computation across areas and species… the three
  problems also make it **highly unlikely that columnar structures are species-invariant units that
  compute the same function across varying inputs**."

**The spec's rhetorical move is exactly the one this literature rejects**: taking the column as an
anatomically discrete, uniformly sized building block that runs the same computation everywhere,
and reading `n` off it.

### 6.2 The "efficient size" claim, specifically: unsupported by any source read

The nearest thing found in print is Bosman, C.A., Lansink, C.S. & Pennartz, C.M. (2014),
*Eur. J. Neurosci.* 39(11), 1982–1999, [doi:10.1111/ejn.12606](https://doi.org/10.1111/ejn.12606),
which offers the defence: "neuronal computations generated by this architecture may be **more
efficient** than computations performed by the same number of neurons connected randomly, as
suggested by recent neuronal models of cortical columns (Haeusler & Maass; Bastos et al.)."

Read that carefully. It is a claim that the columnar **wiring pattern** is more efficient than
random wiring *at fixed neuron count*. It is not a claim that there is an efficient **size**, and
it does not license reading a dimension off it. **No source read states or implies that a column
has an optimal or efficient size.** That phrase in the spec is an overreach.

### 6.3 The uniformity of *neuron number* is a genuine live dispute, not a settled fact either way

This is the honest state of the quantitative literature, and it splits:

**For uniformity** — Carlo, C.N. & Stevens, C.F. (2013). "Structural uniformity of neocortex,
revisited." *PNAS* 110(4), 1488–1493.
[doi:10.1073/pnas.1221398110](https://www.pnas.org/doi/10.1073/pnas.1221398110). They re-test
Rockel et al.'s classic count and report "statistically the same number of neurons underneath a
square millimeter of neocortical surface" across areas and species — **with one exception**:
primate primary visual cortex has roughly **twice** as many.

**Against uniformity** — Herculano-Houzel, S., Collins, C.E., Wong, P., Kaas, J.H. & Lent, R.
(2008). "The basic nonuniformity of the cerebral cortex." *PNAS* 105(34), 12593–12598.
[doi:10.1073/pnas.0805417105](https://www.pnas.org/doi/abs/10.1073/pnas.0805417105), and Collins,
C.E. et al. (2010). "Neuron densities vary across and within cortical areas in primates." *PNAS*
107(36), [doi:10.1073/pnas.1010356107](https://www.pnas.org/doi/10.1073/pnas.1010356107). The
summary as it reaches the methods literature (Rafati, A. et al. (2016), *J. Microscopy* 261(1),
115–126, [doi:10.1111/jmi.12321](https://doi.org/10.1111/jmi.12321)) is blunt:

> "it is clear from studies of the number of cortical neurons below a fixed unit area of pial
> surface, that **the putative minicolumns cannot have constant size and number of neurons across
> different brain areas or across species**."

The same source records the classical numbers the spec's intuition is presumably drawing on —
minicolumn diameter 35–60 µm, "typically 80–100 neurons," 60–80 minicolumns to a macrocolumn of
300–600 µm — and immediately notes it is "still unclear to what degree minicolumns observed by
various definitions truly reflect elementary functional units, **or if they are just packing
artefacts or reminiscences of the radial growth of the cortex during embryogenesis**."

Note the shape of the surviving pro-uniformity result: **even at its strongest it has a 2× outlier
at the sensory rim.** A Patchworks reader should find that uncomfortable, because the graph's
sensory rim is exactly where the spec already grants an exemption (boundary cells, ADR-0006) — and
because a 2× spread in the biological analogue is a poor advertisement for a single global `n`.

### 6.4 What DOES survive, and it is the better claim

- **da Costa & Martin's own conclusion** is that the **canonical microcircuit**, not the column, is
  the object worth keeping: it "respects the known connectivity of the neocortex, and it is
  flexible enough to transiently change the architecture of its network in order to perform the
  required computation." Critically, the canonical microcircuit is defined **without a spatial
  boundary or a size** — Haueis & Margulies note it "avoids the problem of missing boundaries by
  representing circuit structure without specifying a discrete spatial boundary at the mesoscale."
  **A canonical microcircuit is a uniform *circuit*, and explicitly not a uniform *size*.** The
  spec cites it to fix a dimension, which is precisely the thing the concept was refined to stop
  claiming.
- **Harris, K.D. & Shepherd, G.M.G. (2015). "The neocortical circuit: themes and variations."**
  *Nature Neuroscience* 18, 170–181. [doi:10.1038/nn.3917](https://www.nature.com/articles/nn.3917).
  The defence in its strongest current form: the connections "constitute a basic circuit pattern
  that appears to be repeated across neocortical areas, **with area- and species-specific
  modifications**," and the properties of cortical neuron classes "are remarkably similar across
  areas." Cortex has "a serially homologous organization, featuring area- and species-specific
  variations on a basic theme."

That last is a strikingly good match to what `01-cell-and-sheaf.md` *actually argues* two sections
later — cells "uniform in contract: same interface, same algorithm," with individuality carried by
"what its features mean," fixed by restriction maps and biases. **Harris & Shepherd's "themes and
variations" is a real neuroscientific warrant for uniform machinery plus per-cell specialisation.**
It is a warrant for the *shared frozen body plus adapting surface* design, which the spec currently
justifies on other grounds (reservoir computing), and it is **not** a warrant for a fixed global
`n`. The spec has the right citation attached to the wrong claim.

---

## 7. Recurrent failure modes in per-node recurrent GNNs

**Verdict: SUPPORTED that the exposure is real, documented, and central to the field's history.
The gating remedy is SUPPORTED as a known fix — but the specific form the spec names, a
pass-through subset of the *lane*, is NOT the known fix, which lives in the recurrent state
itself. That is a candidate revision.**

### 7.1 The failure mode is the founding constraint of the field

Scarselli, F. et al. (2009), "The Graph Neural Network Model," *IEEE TNN* 20(1), 61–80, made the
per-node state update a **contraction map** so that iterating it converges to a unique fixed point
by Banach. That is not an incidental choice — it is the price of having a recurrent per-node state
at all, and the whole design is organised around paying it.

Li, Y., Tarlow, D., Brockschmidt, M. & Zemel, R. (2016). "Gated Graph Sequence Neural Networks."
*ICLR 2016*. [arXiv:1511.05493](https://arxiv.org/abs/1511.05493) — read via HTML full text. They
state the constraint and its cost: parameters "must be constrained so that the propagation step is
a contraction map"; this "may limit the expressivity of the model"; and, in their intuition,
contraction maps have exponentially decaying information flow, so it is "difficult to model long
range dependencies." Their fix is twofold: **GRU gating on the per-node hidden state**, and
"unroll[ing] the recurrence for a fixed number of steps `T` and us[ing] backpropagation through
time," which "requires more memory than the Almeida–Pineda algorithm, but **it removes the need to
constrain parameters to ensure convergence**."

**Patchworks is exposed on both halves and has taken the harder branch of each.** It has neither a
contraction guarantee (nothing constrains the shared body's spectral properties) nor BPTT (learning
is cell-local, one step per tick, no unrolling). GGNN's trade — give up the contraction constraint,
buy stability back with gating and BPTT — is available to Patchworks only in its gating half.

### 7.2 Over-smoothing, and the sharp thing it means *here*

Oono, K. & Suzuki, T. (2020). "Graph Neural Networks Exponentially Lose Expressive Power for Node
Classification." *ICLR 2020*. [arXiv:1905.10947](https://arxiv.org/abs/1905.10947): when weights
satisfy conditions set by the spectrum of the augmented normalized Laplacian, GCN output
exponentially approaches a set of signals carrying only connected-component and degree information.

Cai, C. & Wang, Y. (2020). "A Note on Over-Smoothing for Graph Neural Networks." *ICML 2020 GRL
workshop*. [arXiv:2006.13318](https://arxiv.org/abs/2006.13318): they recast the same result in
exactly Patchworks' currency — **the Dirichlet energy of the embeddings converges to zero**,
"resulting in the loss of discriminative power."

**This is the observation worth carrying out of this pass.** Over-smoothing, stated in the
literature's own sharpest form, is *the Dirichlet energy going to zero*. Patchworks' only error
signal **is** the Dirichlet energy (§2). So in Patchworks, over-smoothing is not a degradation of
representation quality that shows up as worse accuracy somewhere downstream — it is **the error
signal vanishing**, and with it both halves of the transport rule's gradient. The failure mode and
the loss of the ability to detect the failure mode are the same event. Combined with §4's
`F = 0` trivial solution — which is the *same* endpoint reached from the parameter side — the
architecture has two independent routes to one degenerate state.

**The counterweight, and it is real.** Bodnar et al.'s Neural Sheaf Diffusion
([arXiv:2202.04579](https://arxiv.org/abs/2202.04579)) is *about* this: the paper's thesis is that a
sheaf Laplacian, unlike a graph Laplacian, has a rich enough harmonic space that diffusion does not
collapse to a constant, which is why its subtitle names over-smoothing. Patchworks inherits that
protection, and 015 §3.2 showed its masks make `H⁰` larger than any paper's — so the collapse
target is a large subspace rather than a point. But per 015's headline, those are theorems about
invertible/orthogonal sheaves; the protection is orientation, not authority, and it is silent about
the `F → 0` route, which destroys the harmonic-space argument by destroying the sheaf.

### 7.3 The escape hatch the spec names — is it the known remedy?

`01-cell-and-sheaf.md`: "The known escape hatch is a designated pass-through subset of **the edge
stalk** carrying state across the recurrence — an LSTM-shaped fix."

**Partly.** Gating is genuinely the known remedy and GGNN is the proof (§7.1). But the mechanism it
invokes is misplaced by one tier:

- LSTM's constant-error carousel (Hochreiter & Schmidhuber, 1997) is a **linear self-connection on
  a cell's own internal state**. It carries state across *time*, within one unit.
- GGNN's GRU is likewise applied to the **per-node hidden state** `h_v`, not to the messages.
- Patchworks' recurrence is `chart(t) → chart(t+1)` via `step`. The lane is not on that loop
  at all: it is on the *spatial* path, and its contents reach the chart only indirectly, one tick
  later, after reconciliation has edited the node stalk and `encode` has read it.

So a pass-through subset of the **lane** would be an ungated skip on the *inter-cell* path,
not on the recurrence. **The LSTM-shaped fix for a recurrence lives in the recurrence** — here, a
protected linear channel through `step` inside the chart, or a gate on `encode`'s fusion of
persisted chart with incoming stalk. There is a real design tension behind this: putting it in the
chart means gating parameters inside the **frozen shared body**, which the freeze forbids; putting
it in the lane keeps it on the adapting surface where the design can reach it. That tension
is worth stating; the current sentence hides it by naming the wrong tier.

Note also that a designated pass-through subset of a lane has a second reading the spec may
not intend: a lane direction that is *always* passed through is a direction on which
disagreement is meant to be small, which makes it a hand-placed `H⁰`-like channel rather than an
error-carrying one. Under §2 that is fine; it is worth being explicit that it is a channel the
architecture deliberately does not learn from.

---

## 8. Bonus check: the Euler-characteristic identity in the applied literature

**Verdict: ABSENCE CONFIRMED. Nothing found.**

015 §2 reported that `χ = Σ_v dim F(v) − Σ_e dim F(e) = dim H⁰ − dim H¹` is elementary rank–nullity
on the two-term complex `0 → C⁰ → C¹ → 0`, and that it could not be located anywhere in the applied
sheaf literature. This pass searched again, across the network-sheaf and sheaf-GNN literature and
the applied-topology side. **Still nothing.** The Euler characteristic appears in the applied-sheaf
world only in the Euler-calculus / constructible-function sense (Ghrist's Euler-calculus work,
[arXiv:1202.0275](https://arxiv.org/pdf/1202.0275)) and in the algebraic-geometry sense
(`χ(F) = h⁰ − h¹` for coherent sheaves on a variety), never as an **index formula for `δ⁰` on a
network sheaf with heterogeneous stalk dimensions**.

The absence is now confirmed twice by independent searches. `015-sheaf-geometry.md`'s handling —
cite it as mathematics, not as literature support — stands, and this pass adds nothing to it except
that a second look did not find it either.

---

## Candidate revisions to the spec

Stated precisely enough to file verbatim as wayfinder revision tickets. Ordered by severity.
None of these is a demand; each is a place where the literature contradicts the spec, or where the
spec claims more than its citation supports.

### R1 — The transport rule's objective has a trivial global minimum at `F = 0`, and nothing excludes it

**Where:** `01-cell-and-sheaf.md`, *Restriction maps* ("**Learned**, under a sparsity pressure");
`CONTEXT.md`, *Transport rule*; and by inheritance `07-local-learning-rule.md`.

**What the literature says:** Di Nino, Barbarossa & Di Lorenzo
([arXiv:2501.19207](https://arxiv.org/abs/2501.19207), §III), the one paper that learns restriction
maps by minimising disagreement directly, states: "We also need to impose a constraint on the set
of feasible restriction maps `F`, to avoid the trivial solution." Every other learned-sheaf method
imposes the same kind of constraint by construction (Bodnar: invertible classes; Barbero: `O(d)`) —
015 §3.2 documented the fact; this pass found the reason.

**Why it bites here:** Patchworks' transport rule descends on disagreement *and* composes a sparsity
pressure in the same step. Both terms are globally minimised at `F = 0`, at which every edge agrees
perfectly and the sheaf couples nothing. Patchworks imposes no orthogonality, invertibility, norm,
or rank floor. The structural mask constrains *which* entries may be nonzero and requires none of
them to be.

**Ask:** either (a) name the mechanism that keeps the restriction maps away from zero and state it
in the spec — a norm constraint, a rank floor, a bounded step plus a stated argument, or a
demonstration that the bias rule's independent pressure suffices; or (b) record explicitly, as
*Known exposure*, that `F → 0` is an unexcluded global optimum of the transport rule's objective
and that the proof-of-concept must instrument for it. Cite Di Nino et al. either way.

### R2 — The canonical-microcircuit rationale for fixed uniform `n` is unsupported and should be replaced or dropped

**Where:** `01-cell-and-sheaf.md`: "`n` is fixed on a canonical-microcircuit rationale: a cortical
column has an efficient size, and cells are the analogue."

**What the literature says:** (i) The column is widely held not to be an anatomically discrete,
uniformly sized, functionally modular unit — Horton & Adams 2005 ("the column may have no
function"; "it has failed as a unifying principle"), Rakic 2008 (the term names several
non-equivalent objects), da Costa & Martin 2010 ("no obvious morphological analog… much of the
anatomical data seems to challenge it"). (ii) The uniformity of neuron count under unit cortical
area is a live dispute — Carlo & Stevens 2013 for, with a 2× exception at primate V1;
Herculano-Houzel et al. 2008 and Collins et al. 2010 against; the methods literature (Rafati et al.
2016) states flatly that "minicolumns cannot have constant size and number of neurons across
different brain areas or across species." (iii) **No source read asserts that a column has an
efficient or optimal size.** (iv) The canonical microcircuit — the concept da Costa & Martin offer
as the column's replacement — is defined *without* a spatial boundary or a size, so it cannot
license a dimension.

**Ask:** delete "a cortical column has an efficient size, and cells are the analogue." Replace with
one of:
- an **engineering** rationale for fixed `n` (batched execution needs one operator shape; a uniform
  `n` is what makes the sheaf's construction diagnostics comparable across cells) — this is honest
  and the spec already has the ingredients; or
- the **surviving** neuroscience, attached to the claim it actually supports: Harris & Shepherd
  2015's "basic circuit pattern… repeated across neocortical areas, with area- and species-specific
  modifications" is a genuine warrant for **uniform contract with per-cell specialisation** — i.e.
  for the shared frozen body plus adapting surface, not for a fixed dimension.

**Second-order:** `01-cell-and-sheaf.md` lists `n` as "deliberately absent" from the Flex priority
ladder and "intended to stay fixed." If R2 is accepted, the *reason* for that absence changes from
a biological one to an engineering one. The decision may well stand; the justification should not
be left pointing at a citation that does not support it.

### R3 — "A hard projection… would drag a global solve into a system whose thesis is locality" overstates the case

**Where:** `01-cell-and-sheaf.md`, *Disagreement, and what is done about it*.

**What the literature says:** Hansen & Ghrist, "Distributed Optimization with Sheaf Homological
Constraints" (Allerton 2019, [PDF](https://jakobhansen.org/publications/distopt.pdf)) enforce
`L_F x = 0` — i.e. `x ∈ H⁰` — **exactly**, via Lagrangian saddle-point dynamics
`ẋ = −∇f(x) − 2L_F x − L_F z`, `ż = L_F x`. Those dynamics are **graph-local**: each node needs
only `L_F x` and `L_F z`, both computed from its own incident edges. Exact enforcement therefore
does *not* require a global solve.

**Ask:** replace the locality argument with the argument that actually holds — that exact
enforcement requires a **dual variable per edge** and would drive the residual to zero, "zeroing out
the quantity the architecture runs on," which the spec already says and which is sufficient on its
own. Optionally add the sharper fact this pass turned up: in Hansen & Ghrist's formulation the
Dirichlet-energy penalty is present only "to improve stability and convergence… while leaving the
minimum and minimizer undisturbed" — Patchworks keeps the stabiliser and discards the enforcer, on
purpose.

### R4 — The LSTM-shaped escape hatch is named on the wrong tier

**Where:** `01-cell-and-sheaf.md`, *Known exposure*: "a designated pass-through subset of the edge
stalk carrying state across the recurrence — an LSTM-shaped fix."

**What the literature says:** the constant-error carousel (Hochreiter & Schmidhuber 1997) is a
linear self-connection on a unit's **own internal state**; Gated Graph Neural Networks
([arXiv:1511.05493](https://arxiv.org/abs/1511.05493)) apply GRU gating to the **per-node hidden
state**, and that is what removes Scarselli et al.'s contraction requirement. In both, the gate is
on the recurrence.

**Why it bites here:** Patchworks' recurrence is `chart(t) → chart(t+1)` through `step`. The edge
stalk is not on that loop; its contents reach the chart only one tick later, via reconciliation and
`encode`. A pass-through subset of the lane is a skip on the *spatial* path, not the
recurrent one, and does not do what the cited fix does.

**Ask:** restate the escape hatch as acting on the recurrence — a protected linear channel through
`step`, or a gate on `encode`'s fusion of persisted chart with incoming node stalk — and **name the
tension the current phrasing hides**: a gate in the chart means gating parameters inside the frozen
shared body, which the freeze forbids, whereas the lane is on the adapting surface where the
design can reach it. That tension is a genuine consequence of the shared-frozen-body bet and
belongs in *Known exposure* alongside it.

### R5 — Over-smoothing, in Patchworks, is the error signal vanishing; say so

**Where:** `01-cell-and-sheaf.md`, *Known exposure* (recurrent failure modes).

**What the literature says:** Cai & Wang ([arXiv:2006.13318](https://arxiv.org/abs/2006.13318))
state over-smoothing precisely as **the Dirichlet energy of the embeddings converging to zero**,
"resulting in the loss of discriminative power"; Oono & Suzuki
([arXiv:1905.10947](https://arxiv.org/abs/1905.10947)) prove the exponential-collapse result it
recasts.

**Why it bites here:** §2 establishes that Patchworks' only error signal *is* the Dirichlet energy.
So over-smoothing here is not a downstream quality loss — it is the disappearance of the signal both
halves of the local learning rule are computed from, and of the only instrument that would reveal
it. It reaches the same endpoint as R1 by an independent route (state collapse rather than parameter
collapse).

**Ask:** add one line to *Known exposure* naming this identity, and pair it with the counterweight —
Bodnar et al.'s sheaf-diffusion result that a rich harmonic space resists collapse, with 015's
caveat that those are orthogonal-sheaf theorems and are orientation, not authority, for Patchworks.
This is a candidate instrumentation requirement for the proof-of-concept, not a design change.

### R6 — Unit delay is not free on the information axis

**Where:** `01-cell-and-sheaf.md`, *Unit delay*: "delay is a phase shift, not a decimation: it
removes no frequency content."

**What the literature says:** DRew ([arXiv:2305.08018](https://arxiv.org/abs/2305.08018)) studies
exactly this knob and describes a two-sided trade: larger delay means a node "aggregates the
features from `j` before they are (significantly) 'smoothed' by repeated message passing," while
smaller delay means that when `i` communicates with `j` "it also leverages the structure around
`j`."

**Why it bites here:** the frequency-content argument is correct as stated and is about the temporal
axis. DRew's axis is different: what a delayed value has *not yet been mixed with*. On DRew's
reading the trade cuts in Patchworks' favour (less smoothing), so this is a cheap win — but the spec
currently reads as though delay has exactly one consequence (staleness), and the one paper that
tunes the parameter says it has two.

**Ask:** add a sentence acknowledging the second axis and claiming the favourable side of it, citing
DRew. Low severity; strengthens the section rather than correcting it.

### R7 — Linearity of restriction maps is load-bearing twice, and the spec says so once

**Where:** `01-cell-and-sheaf.md`, *Restriction maps* (Linear) and *Disagreement* (PC error =
sheaf inconsistency).

**What the literature says:** Seely's identification of predictive-coding error with the sheaf
coboundary is derived for **linear** PC networks ([arXiv:2511.11092](https://arxiv.org/abs/2511.11092));
the energy identity `E_PC(s) = ½‖δ⁰s‖²` depends on it.

**Why it bites here:** the spec justifies linearity on two grounds — keeping the sheaf formalism
real, and ADR-0004's local-flatness geometry. It does not say that the *predictive-coding
identification itself* is a third thing linearity buys. Under nonlinear restriction maps, "PC error
and sheaf inconsistency are the same quantity" would stop being true, not merely become harder to
compute.

**Ask:** one clause in the *Linear* bullet noting that the disagreement-is-PC-error identification
depends on linearity too, citing Seely. Documentation, not a design change.

### Nothing to revise on these

- **The three-tier split (§1).** Unusual in the sheaf literature, established in bandwidth-limited
  multi-agent RL, contradicted by nobody. The spec's own justification (solving and recomposing are
  different operations) is not stated in any source but is not opposed by any either. It survives.
- **Disagreement as the sole error signal (§2).** Supported, and independently derived elsewhere.
- **Penalised-never-cleared (§3), as a choice.** Conventional for the neural-network branch of the
  sheaf literature. Only the *reason given* needs correcting (R3).
- **Learned and masked restriction maps, as such (§4.3).** Learning them is now the norm; the mask
  is Patchworks' own and nothing contradicts it. It is the *sparsity pressure composed with the
  disagreement objective* that R1 flags, not the mask.
- **Linearity (§4.4).** The same call the literature made, for the same reason. No paper argues for
  nonlinear restriction maps.
- **Unit delay (§5).** Independently arrived at in two literatures, one of which derives it from
  physics and gets Patchworks' rule exactly. Strongly supported.
- **The Euler-characteristic identity (§8).** Absence re-confirmed; 015's handling stands.

---

## Sources

- Barbero, F., Bodnar, C., Sáez de Ocáriz Borde, H., Bronstein, M., Veličković, P. & Liò, P. (2022).
  Sheaf neural networks with connection Laplacians. TAG-ML @ ICML.
  [arXiv:2206.08702](https://arxiv.org/abs/2206.08702)
- Bodnar, C., Di Giovanni, F., Chamberlain, B., Liò, P. & Bronstein, M. (2022). Neural sheaf
  diffusion: a topological perspective on heterophily and oversmoothing in GNNs. *NeurIPS 35*.
  [arXiv:2202.04579](https://arxiv.org/abs/2202.04579)
- Bosman, C.A., Lansink, C.S. & Pennartz, C.M. (2014). Functions of gamma-band synchronization in
  cognition. *Eur. J. Neurosci.* 39(11), 1982–1999. [doi:10.1111/ejn.12606](https://doi.org/10.1111/ejn.12606)
- Cai, C. & Wang, Y. (2020). A note on over-smoothing for graph neural networks. *ICML 2020 GRL
  workshop*. [arXiv:2006.13318](https://arxiv.org/abs/2006.13318)
- Carlo, C.N. & Stevens, C.F. (2013). Structural uniformity of neocortex, revisited. *PNAS* 110(4),
  1488–1493. [doi:10.1073/pnas.1221398110](https://www.pnas.org/doi/10.1073/pnas.1221398110)
- Collins, C.E. et al. (2010). Neuron densities vary across and within cortical areas in primates.
  *PNAS* 107(36). [doi:10.1073/pnas.1010356107](https://www.pnas.org/doi/10.1073/pnas.1010356107)
- da Costa, N.M. & Martin, K.A.C. (2010). Whose cortical column would that be? *Front. Neuroanat.*
  4:16. [doi:10.3389/fnana.2010.00016](https://doi.org/10.3389/fnana.2010.00016)
- Di Nino, L., Barbarossa, S. & Di Lorenzo, P. (2025). Learning sheaf Laplacian optimizing
  restriction maps. [arXiv:2501.19207](https://arxiv.org/abs/2501.19207)
- Foerster, J., Assael, Y.M., de Freitas, N. & Whiteson, S. (2016). Learning to communicate with
  deep multi-agent reinforcement learning. *NIPS 2016*.
  [arXiv:1605.06676](https://arxiv.org/abs/1605.06676)
- Gutteridge, B., Dong, X., Bronstein, M. & Di Giovanni, F. (2023). DRew: dynamically rewired
  message passing with delay. *ICML 2023*. [arXiv:2305.08018](https://arxiv.org/abs/2305.08018)
- Hansen, J. & Gebhart, T. (2020). Sheaf neural networks.
  [arXiv:2012.06333](https://arxiv.org/abs/2012.06333)
- Hansen, J. & Ghrist, R. (2019). Distributed optimization with sheaf homological constraints.
  *57th Allerton Conference*, 565–571. [PDF](https://jakobhansen.org/publications/distopt.pdf)
- Hansen, J. & Ghrist, R. (2021). Opinion dynamics on discourse sheaves. *SIAM J. Appl. Math.*
  81(5). [arXiv:2005.12798](https://arxiv.org/abs/2005.12798)
- Harris, K.D. & Shepherd, G.M.G. (2015). The neocortical circuit: themes and variations.
  *Nature Neuroscience* 18, 170–181. [doi:10.1038/nn.3917](https://www.nature.com/articles/nn.3917)
- Haueis, P. & Margulies, D. (2026). Developing concepts for neuroscience: a philosophical toolkit.
  *Eur. J. Neurosci.* 63(2). [doi:10.1111/ejn.70403](https://doi.org/10.1111/ejn.70403)
- Herculano-Houzel, S., Collins, C.E., Wong, P., Kaas, J.H. & Lent, R. (2008). The basic
  nonuniformity of the cerebral cortex. *PNAS* 105(34), 12593–12598.
  [doi:10.1073/pnas.0805417105](https://www.pnas.org/doi/abs/10.1073/pnas.0805417105)
- Hochreiter, S. & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation* 9(8),
  1735–1780.
- Horton, J.C. & Adams, D.L. (2005). The cortical column: a structure without a function.
  *Phil. Trans. R. Soc. B* 360, 837–862. [PMC1569491](https://pmc.ncbi.nlm.nih.gov/articles/PMC1569491/)
- Li, Y., Tarlow, D., Brockschmidt, M. & Zemel, R. (2016). Gated graph sequence neural networks.
  *ICLR 2016*. [arXiv:1511.05493](https://arxiv.org/abs/1511.05493)
- Oono, K. & Suzuki, T. (2020). Graph neural networks exponentially lose expressive power for node
  classification. *ICLR 2020*. [arXiv:1905.10947](https://arxiv.org/abs/1905.10947)
- Rafati, A. et al. (2016). Detection and spatial characterization of minicolumnarity in the human
  cerebral cortex. *J. Microscopy* 261(1), 115–126. [doi:10.1111/jmi.12321](https://doi.org/10.1111/jmi.12321)
- Rakic, P. (2008). Confusing cortical columns. *PNAS* 105(34), 12099–12100.
  [doi:10.1073/pnas.0807271105](https://www.pnas.org/doi/10.1073/pnas.0807271105)
- Scarselli, F., Gori, M., Tsoi, A.C., Hagenbuchner, M. & Monfardini, G. (2009). The graph neural
  network model. *IEEE Trans. Neural Networks* 20(1), 61–80.
- Seely, J. (2025). Sheaf cohomology of linear predictive coding networks.
  [arXiv:2511.11092](https://arxiv.org/abs/2511.11092)
- Southern, J., Di Giovanni, F., Bronstein, M. & Lutzeyer, J. (2025). Understanding virtual nodes:
  oversquashing and node heterogeneity. *ICLR*. [arXiv:2405.13526](https://arxiv.org/abs/2405.13526)
- Tolstaya, E., Gama, F., Paulos, J., Pappas, G., Kumar, V. & Ribeiro, A. (2019). Learning
  decentralized controllers for robot swarms with graph neural networks. *CoRL 2019*.
  [arXiv:1903.10527](https://arxiv.org/abs/1903.10527)
- *Learned, relied upon, or necessary? Separating checkpoint dependence from task-level value in
  sheaf GNNs* (2026). [arXiv:2607.25387](https://arxiv.org/abs/2607.25387) — cited from its
  abstract only; authorship not verified in this pass.
