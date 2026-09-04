# Citation pass: linear recurrences, the nonlinearity's position, and who disagrees (patchworks#167)

Checks the reassurance recorded in
[ADR-0023](../adr/0023-the-chart-is-not-a-koopman-lift.md) — that having moved the design off the
Koopman path, the **state-space-model and linear-RNN literature is a better-trodden path than the one
it left**, making a 12-dimensional persisting **chart** per **cell** unremarkable. ADR-0023 says in
its own text that this was argued from recall; only S4D's timescale initialisation was verified at
source, in `docs/research/027-regional-jacobian-spectra.md`. This pass reads the rest.

Citations validate after the fact per [#1](https://github.com/NGL321/patchworks/issues/1)'s
citation-sequencing rule. **This document does not revise closed design.** Where a source threatens a
claim already made it is flagged, per artifact, in *What this threatens*. Vocabulary follows
`CONTEXT.md`: Patchworks' side in its own terms (cell, chart, piece, node stalk, communication
lane, restriction map, scale gauge), the prior art's in its own field's — and the two are deliberately not
blurred, because §1's central finding is exactly that the design has been described in the wrong
field's vocabulary. Where a source could not be reached, that is stated rather than papered over; see
*What could not be reached*.

The pass had the same second job [#148](https://github.com/NGL321/patchworks/issues/148) had: check
the reasoning against people who **disagree**. §6 is that section, and it is the longest one here.
Two of this document's three sharpest findings come out of it.

## Reading-depth key

Every source is tagged, using #148's key. The pass leans on the distinction heavily: the two findings
that change what other tickets may do both rest on full-text reads, and the source that most
threatens #166 was reached only at abstract depth.

- **[FULL]** — paper body read (PDF text or HTML extracted).
- **[ABS]** — authoritative abstract / landing page only.
- **[CITE]** — citation confirmed to exist, text not reached.
- **[UNREACHED]** — existence not confirmed.

One extraction note, because it bears on trust: Jaeger's GMD Report 152 is distributed as a PDF whose
embedded font uses non-standard glyph names, so ordinary text extraction returns nothing readable. It
was decoded glyph-by-glyph in this pass and the decoding was validated against the paper's known
title, authors, report number and date before any quotation was taken from it. Quotations from it
below are verbatim from that decoding.

## Headline verdict, stated plainly

**Three findings, in descending order of what they cost.**

**1. ADR-0023 names the wrong comparison class, and its own Consequences section already knows.**
The literature ADR-0023 reaches for — S4/S4D, LRU, Mamba — is the literature of a linear recurrence
with the nonlinearity **between stacked layers**. Patchworks re-applies `encode` **inside the loop,
every tick**. That is not a variant of the SSM class; it is the thing the SSM class was defined by
excluding. Merrill, Petty & Sabharwal state the distinction in as many words and prove it is not
cosmetic: *"adding a nonlinearity to the output of an SSM layer (as in Mamba) is not the same thing
as an RNN-SSM. Rather, an RNN-SSM applies the nonlinearity at each recurrent update"* — and the
second is strictly more expressive (§1.3). ADR-0023's Consequences section says the same thing
correctly (*"`K . encode` is a **nonlinear** recurrence"*); its Decision section then hands the design
to a comparison class that excludes it. **So the answer to question 1 is: yes, moving the
nonlinearity out of the recurrence is a recognised architecture class with a universality theorem
behind it — and Patchworks is not in it.** The direction of the error is favourable (the design sits
on the *more* expressive side of the line), but the licence for `k = 12` cannot come from where
ADR-0023 says it comes from. §1.4 gives the class the design **is** in, by a change of variables that
takes ten lines and lands the cell squarely in reservoir computing.

**2. The memory bound #166 wants exists, is stronger than #166 states, and its most-cited authors
have since published a paper saying the metric is worthless.** Jaeger's `MC ≤ N` is real and was read
at source, with both hypotheses named in the original: *"Both conditions (i.i.d. input and linear
output units) are necessary for this bound."* Dambre et al. strengthen it to **total** capacity —
linear and nonlinear functionals together — bounded by the number of linearly independent state
variables, which is the exact form #166's double-duty question needs. But Ganguli, Huh & Sompolinsky
show the bound is attained only under stringent design: *"the capacity of networks with normal
connectivity matrices is exactly 1 and that of any network of N neurons is, at most, N"* — and
Patchworks constructs `K = a·I`, which is **normal**. And Ballarin, Grigoryeva & Ortega (2025)
conclude *"the existing definition of MC in linear and nonlinear cases has no practical value."*
**#166 may cite the bound, with the hypotheses attached and the Ganguli refinement carried; it may
not cite "~12 taps" as a clean number.** §2.

**3. Question 3 does not return the loud "yes" it was written to catch, but it comes closer than
#148's Claim 1 did, and the residue is now one property wide.** Per-cell persisting linear state,
advanced by a linear recurrence with the nonlinearity outside it, glued spatially over graph edges,
is **in print and current** — Ceni, Gravina, Gallicchio, Bacciu, Schönlieb & Eliasof's
Message-Passing State-Space Models, verbatim *"A key distinction from such models lies in the use of
a purely linear recurrent equation"*, with per-node state, edge propagation and an MLP outside the
loop. Sheaf-theoretic learning **with restriction maps and a backpropagation-free local rule** is
also in print — Bosca & Ghrist, from the group that owns the sheaf vocabulary the design borrows.
Local linear filters reconciled across a graph by a purely local rule is fifty years old under
another name (Kalman-consensus filtering). **What is not found, in any single source, is the
conjunction: a learned per-cell operator, glued by learned restriction maps into communication
lanes, trained
by a local rule with no global objective.** Every *pair* of those three is claimed. That is a
narrower residue than the record currently assumes, and it should be stated that way. §3.

Two smaller results:

- **Question 5 comes back clean.** No certification-style objection of #148 §5's kind exists in the
  SSM literature, because the question does not arise there: `λ` and `Δ` are **initialised by
  design** and read as retention constants. S4D draws `Δ` log-uniformly over `[1e-3, 1e-1]`; LRU
  samples eigenvalues on a ring `[r_min, r_max]` and says outright that long-range reasoning
  *"need[s] to have magnitude close to 1"*; Mamba's own gloss is *"a large Δ resets the state h and
  focuses on the current input x, while a small Δ persists the state."* Nobody asks whether these
  converge to a true spectrum. **#143's reading is confirmed at source.** But §5.3 carries two
  objections of a *different* kind that #143 does need. §5.
- **ADR-0023's "mechanism transfers, results do not" is half right, and the half that is wrong is the
  expensive half.** The *expressivity* results are indeed about deep stacks trained by backprop and
  do not transfer. The *capacity* results transfer completely, because they are facts about how much
  a state of a given dimension can hold, indifferent to how the operator got there — Dambre et al.
  state theirs for arbitrary dynamical systems, and the entire reservoir-computing field is the
  shallow, non-backprop case they were built for. **The results that transfer are the unfavourable
  ones.** §4.

### What this pass did not check

This document checks the *recurrence* reasoning. It does not revisit #148's Koopman findings, #32's
dimensioning, or #27's regional-spectra work, all of which stand. It does not re-derive #148 §11's
local-learning verdict; §4 leans on it. It reads `src/patchworks/body.py` for structure only, and
runs nothing.

---

## 1. A linear recurrence with the nonlinearity outside it

**Verdict: a recognised class with a universality theorem, in which the cost of moving the
nonlinearity out is genuinely close to zero for the architectures studied — and Patchworks is not one
of them, because Patchworks did not move it out.**

### 1.1 The class exists and the claim is made in its own words

**Orvieto, Smith, Gu, Fernando, Gulcehre, Pascanu & De**, *"Resurrecting Recurrent Neural Networks
for Long Sequences"*, **ICML 2023**, PMLR 202, arXiv:2303.06349. **[FULL — body read via ar5iv]**

This is the paper ADR-0023's reassurance is really about, and it does say what the reassurance
attributes to it. Verbatim: *"linear RNN layers can be surprisingly expressive when coupled with
nonlinear MLP or GLU blocks, outperforming tuned nonlinear RNN variants."* The ablation is explicit:
the recurrence `x_k = σ(A x_{k-1} + B u_k)` has `σ` removed to give `x_k = A x_{k-1} + B u_k`, and
their Table 1 has the linearised variant **improving** on the ReLU and tanh recurrences in the same
architecture. Their own summary of the trade: *"While dropping the nonlinearity does not seem to harm
expressivity, it leads to several advantages, from the ability to directly control how quickly the
gradients might vanish or explode, to allowing us to parallelize training."*

Two caveats, both from the paper and both load-bearing here. The architecture is *"a network of 6
layers with residual connections and layer/batch normalization"* — the nonlinearity is **between
layers**, and there are six of them. And the evidence is one benchmark: the **Long Range Arena**,
with the authors stating their goal is *"not to surpass the performance of S4-based models, but
rather to demonstrate that simple RNNs can also achieve strong performance on long range reasoning
tasks when properly initialized."* This is an existence result on a benchmark, not a general theorem
that in-loop nonlinearity is redundant.

### 1.2 The theorem behind it

**Orvieto, De, Gulcehre, Pascanu & Smith**, *"Universality of Linear Recurrences Followed by
Non-linear Projections: Finite-Width Guarantees and Benefits of Complex Eigenvalues"*, arXiv:2307.11888,
**HLD 2023 workshop**. **[ABS]**

Verbatim: *"combining MLPs with both real or complex linear diagonal recurrences leads to arbitrarily
precise approximation of regular causal sequence-to-sequence maps"*, resting on *"a separation of
concerns: the linear RNN provides a lossless encoding of the input sequence, and the MLP performs
non-linear processing on this encoding."*

Read the hypothesis carefully, because it is where the transfer to Patchworks breaks. The theorem is
about a **stack**: linear recurrent layers *interleaved with* position-wise MLPs. The linear
recurrence's job in the proof is to be a lossless encoder of history; all nonlinear processing
happens **after** it, outside the loop, in a separate layer. One cell of Patchworks is not a stack.
Whether a sheaf over a graph, with `encode` between hops and unit delay on every edge, is a stack in
space is an interesting question and this pass found nobody who has asked it — see *What could not be
reached*.

### 1.3 Where the position of the nonlinearity is proven to matter

**Merrill, Petty & Sabharwal**, *"The Illusion of State in State-Space Models"*, **ICML 2024**,
arXiv:2404.08819. **[FULL — body read via arXiv HTML]**

This is the single most important source in §1, because it is the one that draws the exact line
Patchworks sits on the other side of, and it draws it deliberately rather than in passing.

- The limitation: SSMs *"cannot express computation outside the complexity class TC⁰"* and so cannot
  solve permutation composition (the `S₅` word problem) or track chess state. Their expressive power
  is *"limited very similarly to transformers"* — the recurrence buys nothing.
- The contrast: *"RNNs can easily express S₅ via standard constructions that encode finite-state
  transitions into an RNN"* (their §4.4).
- The remedy, and the sentence that matters most here (their §5.1): *"One extension to the SSM is to
  add a nonlinearity, effectively making it an RNN"* — defined as `h_i = sgn(Ā h_{i-1} + B̄ x_i)`,
  the nonlinearity applied at **each recurrent update**. Their Theorem 5.1: *"For any regular
  language L ⊆ Σ* (including the word problem for S₅), there exists a one-layer log-precision RNN-SSM
  with k = |Σ| that recognizes L."*
- And the disambiguation, verbatim: *"adding a nonlinearity to the output of an SSM layer (as in
  Mamba) is not the same thing as an RNN-SSM. Rather, an RNN-SSM applies the nonlinearity at each
  recurrent update."*

**Patchworks applies the nonlinearity at each recurrent update.** `chart_{t+1} = K · encode(chart_t,
stalk_t)` is an RNN-SSM in Merrill et al.'s sense, not an SSM. So the answer to question 1's second
half — *did anyone establish that moving the nonlinearity out costs nothing?* — is: **somebody
established that it costs something specific and nameable, and the design has not paid it.** Note
also Theorem 5.1's shape: one layer suffices, and `k` scales with the alphabet, not with sequence
length. A shallow per-cell recurrence with in-loop nonlinearity is exactly the object that theorem is
about.

Corroborating, from a different direction: **Muca Cirone, Orvieto, Walker, Salvi & Lyons**,
*"Theoretical Foundations of Deep Selective State-Space Models"*, **NeurIPS 2024**, arXiv:2402.19047.
**[ABS]** Verbatim: *"if the linear recurrence powering SSMs allows for multiplicative interactions
between inputs and hidden states (e.g. GateLoop, Mamba, GLA), then the resulting architecture can
surpass in both accuracy and efficiency attention-powered foundation models"*, because
input-controlled transitions make the hidden state *"a low-dimensional projection of … the signature
of the input — capturing non-linear interactions between tokens at distinct timescales."* Selectivity
is the SSM field's partial move back toward in-loop nonlinearity: it lets the input modulate the
transition, but keeps the map **linear in the state**. `encode` is nonlinear in the state as well, so
even selective SSMs are a weaker structure than a Patchworks cell in this specific respect.

### 1.4 What class the cell is actually in — a change of variables

This subsection is **this pass's own derivation**, not a citation, and is marked as such. It is
included because §2 and §4 both depend on knowing which literature owns the object.

From `src/patchworks/body.py`: `encode` is one hidden ReLU layer, `E_out · ReLU(E_hid [chart; stalk] +
b)`, with `E_hid` of shape `[45, 44]` and `E_out` of shape `[12, 45]` at `n = 32, k = 12`, both frozen
random buffers. So the cell's tick is

```
chart_{t+1} = K · E_out · ReLU( E_hid [chart_t ; stalk_t] + b )
```

Substitute `u_t = E_hid [chart_t ; stalk_t] + b ∈ R^45`, the pre-activation. Then `chart_t = K E_out
ReLU(u_{t-1})`, and

```
u_t = ( E_hid^chart · K · E_out ) · ReLU(u_{t-1})  +  E_hid^stalk · stalk_t  +  b
```

which is an ordinary 45-unit recurrent network — random input weights, random bias, ReLU, a linear
readout — whose **recurrent weight matrix is `W_rec = E_hid^chart K E_out`, of rank at most 12, with
the only learned degrees of freedom in the 12×12 factor `K`**. That is an echo state network with a
low-rank, partially-learned reservoir. The composition order that looked unusual (`linear ∘
nonlinear`, where an ESN is `nonlinear ∘ linear`) is an artefact of where the loop is cut; it is the
same loop.

Three consequences, all of which the rest of this document uses:

1. **The matched literature is reservoir computing and nonlinear RNN theory**, not deep SSMs. Every
   memory-capacity result in §2 was derived for exactly this object.
2. **The state that carries memory across ticks is 12-dimensional**, not 45: `rank(W_rec) ≤ 12`, so
   the recurrent subspace has dimension at most `k`. #166's identification of the chart as the memory
   budget is structurally correct, and this is why.
3. The classical control name for a linear block closed around a static nonlinearity is a **Lur'e
   system**, `x_{k+1} = A x_k + B φ(C x_k)`; Patchworks is the degenerate case `A = 0`. Absolute
   stability theory (Popov, circle criterion, Zames–Falb multipliers) is stated for exactly this
   structure, and **Suykens, Vandewalle & De Moor**, *"Lur'e systems with multilayer perceptron and
   recurrent neural networks: absolute stability and dissipativity"*, *IEEE Trans. Automatic Control*
   44(4), 1999 **[CITE — abstract and KU Leuven technical-report landing page located, body not
   reached]** is the paper that connects the two. This is a **route not currently used** by the
   design, noted and not pursued: it is a family of stability certificates for `K · encode` as a
   whole, where ADR-0015's band certifies only `K`.

---

## 2. Memory capacity bounds for linear recurrences

**Verdict: the bound is real, was read at source, and its hypotheses are more specific than the
record assumes. One hypothesis (linear readout) Patchworks satisfies. One (i.i.d. input) it does not,
and its failure loosens rather than tightens the bound. A third result nobody has cited yet says a
normal `K` — which is what the design constructs — gets capacity 1, not 12.**

### 2.1 The original result, with both hypotheses in the author's own words

**Jaeger**, *"Short term memory in echo state networks"*, **GMD Report 152**, GMD –
Forschungszentrum Informationstechnik GmbH, 2002. **[FULL — glyph-decoded from the PDF; decoding
validated against title, author, report number and date before quotation]**

Abstract, verbatim: *"A quantitative measure MC of short-term memory capacity is introduced. The main
result is that MC ≤ N for networks with linear output units and i.i.d. input, where N is network
size."*

**Proposition 2**, verbatim: *"The memory capacity for recalling an i.i.d. input by a N-unit RNN with
linear output units is bounded by N."* Immediately after: *"Both conditions (i.i.d. input and linear
output units) are necessary for this bound."*

Jaeger then demolishes each condition himself, and this matters more than the bound does:

- **Drop i.i.d. input** and the bound is not merely loosened but destroyed: for constant input *"any
  linear-output network … would 'recover' all (identical) delayed versions of the input perfectly,
  formally yielding an infinite memory capacity."*
- **Drop linear readout** and *"infinite memory capacity can be obtained even with a single-unit
  network"*, by his base-2-digit-extraction construction. His own comment: *"This is of course an
  extreme and theoretical example, but it underlines the necessity of the conditions in Proposition
  2."*

Two further propositions are what actually govern whether a given network reaches `N`. **Proposition
3**: *"The k-delay STM capacity for an i.i.d. input of a N-unit linear RNN decreases monotonically
with k."* **Proposition 4**: *"The STM memory capacity of a linear network is N iff the matrix M_N =
(W¹w^in ⋯ W^N w^in) has full rank."* That is the Kalman controllability matrix, and it is the exact
form Hermans & Schrauwen and later Grigoryeva–Ortega restate.

**How the hypotheses land on a cell.** The readout condition holds: `decode` is a frozen **linear**
gauge ([ADR-0014](../adr/0014-the-linear-readout-is-gauge-fixed.md)), and what neighbours read
of a cell passes through **linear restriction maps** into **communication lanes**. Patchworks is
unusually
well-placed here — the linear-readout hypothesis is not an idealisation for this design, it is a
construction commitment. The input condition fails: `stalk_t` is written by reconciliation from
neighbouring cells and is strongly autocorrelated in time and across the graph. Jaeger's own
counterexample is the *correlated-input* one, and it runs in the direction of **more** apparent
memory, not less. So `MC ≤ 12` is **not** directly citable for a cell embedded in a transmitting
sheaf; what is citable is §2.2's version.

### 2.2 The result #166 actually wants

**Dambre, Verstraeten, Schrauwen & Massar**, *"Information Processing Capacity of Dynamical
Systems"*, ***Scientific Reports* 2, 514 (2012)**, doi `10.1038/srep00514`. **[FULL — body read via
the publisher's HTML]**

This is the strongest form of the bound and the one that matches the chart's *double duty*, because
it counts nonlinear functionals of the input history in the same budget as linear ones.

- **Theorem 4**, as stated in the body: *"the sum of the capacities for these functions is bounded by
  the number N of output functions."*
- **Theorem 7** gives when the bound is met: under fading memory and linear independence of the
  internal variables, *"the sum of the capacities for the sets Y_L tends towards the number N of
  output functions."*
- Their framing of it: capacity is *"bounded by the number of linearly independent state
  variables"*, and can be read as *"the total number of linearly independent functions of its stimuli
  the system can compute."*
- Hypotheses, from the body: fading memory; inputs *"independent and identically drawn from some
  probability distribution p(u)"*; a linear estimator readout; and an orthonormal basis, *"products
  of normalized Legendre polynomials for each time step."*
- And the finding that names #166's question precisely: the paper *"uncovers universal trade-offs
  between the non-linearity of the computation and the system's short-term memory."*

**This is the citation #166 was blocked on, and it is a better one than the ticket expected.** #166
frames naming and memory as two jobs with independent budgets that happen to share twelve numbers.
Dambre et al. prove they are the *same* budget, with a trade-off between them, and that the budget is
the number of linearly independent state variables — which §1.4 shows is `rank(W_rec) ≤ k = 12`. The
i.i.d. hypothesis still applies and still fails for a cell in a sheaf, so this is a bound on the cell
considered in isolation, driven by an idealised input.

### 2.3 The refinement nobody in the record has

**Ganguli, Huh & Sompolinsky**, *"Memory traces in dynamical systems"*, ***PNAS* 105(48):18970–18975
(2008)**, doi `10.1073/pnas.0804451105`. **[FULL — body read via PMC2596211]**

They measure memory by Fisher information under Gaussian noise rather than by correlation, and get a
sharper and much less comfortable picture. Verbatim: *"the capacity of networks with normal
connectivity matrices is exactly 1 and that of any network of N neurons is, at most, N."* Attaining
the upper bound is not generic: *"A nonnormal network achieving this bound is subject to stringent
design constraints: It must have a hidden feedforward architecture that superlinearly amplifies its
input for a time of order N"*, and *"the input connectivity must optimally match this architecture."*

And, directly answering question 2's last clause — does it apply to a nonlinearly-fused recurrence?
— verbatim: *"The memory capacity of networks subject to saturating nonlinearities is further
limited, and cannot exceed N."* The remedy they identify is structural: *"This limit can be realized
by feedforward structures with divergent fan out that distributes the signal across neurons, thereby
avoiding saturation."*

**`K = a·I` at construction is normal.** So is any symmetric or orthogonal `K`. On Ganguli et al.'s
measure a cell at construction has total memory capacity **1**, not 12, and reaching 12 requires `K`
to become strongly non-normal in a specific, matched way. This does not contradict Jaeger or Dambre —
different measures, different noise assumptions — but it is the difference between "12 is the
ceiling" and "12 is reachable", and the record currently only has the first. It also gives #166 a
cheap tier-0 instrument: **measure the non-normality of learned `K` (e.g. `‖K†K − KK†‖_F`) alongside
the numerical rank the ticket already plans to take.** Rank saturation and non-normality answer
different halves of the same question.

Corroborating, on the linear side: **White, Lee & Sompolinsky**, *"Short-Term Memory in Orthogonal
Neural Networks"*, ***Phys. Rev. Lett.* 92, 148102 (2004)**, arXiv:cond-mat/0402452. **[ABS]**
Verbatim: *"We study the ability of linear recurrent networks obeying discrete time dynamics to store
long temporal sequences that are retrievable from the instantaneous state of the network… We show
that the memory capacity of these networks scales with system size."* Orthogonal matrices are normal,
and the apparent tension with Ganguli et al. is the noise model: White et al. are noise-free, Ganguli
et al. are not.

### 2.4 The non-i.i.d. extensions

Two lines address the hypothesis that fails for a cell in a sheaf. Neither was reachable in full.

- **Gonon, Grigoryeva & Ortega**, *"Memory and forecasting capacities of nonlinear recurrent
  networks"*, ***Physica D* 414 (2020)**, arXiv:2004.11234. **[ABS]** They generalise to
  **dependent, stationary** inputs and to nonlinear networks, with bounds *"formulated in terms of
  the number of neurons of the nonlinear recurrent network and the autocovariance function or the
  spectral density of the input"*, and settle the linear case: *"the memory capacity is given by the
  rank of the associated controllability matrix."* This is the right family for a driven cell, and
  the shape of the answer — capacity depends on the *input's* correlation structure, not only on the
  state dimension — is what a sheaf's correlated stalks would make binding.
- **Goudarzi, Marzen, Banda, Feldman, Teuscher & Stefanovic**, *"Memory and Information Processing in
  Recurrent Neural Networks"*, arXiv:1604.06929. **[ABS]** They note prior analysis was *"only for
  the case of orthogonal networks, and only under annealed approximation, and uncorrelated input"*
  and compute *"the memory capacity for arbitrary networks with exponentially correlated input."*
- **Hermans & Schrauwen**, *"Memory in linear recurrent neural networks in continuous time"*,
  ***Neural Networks* 23(3):341–355 (2010)**, doi `10.1016/j.neunet.2009.08.008`. **[CITE — PubMed
  and publisher records confirmed; neither abstract page nor body reached, both behind cookie or
  paywall gates]** Cited by the Grigoryeva–Ortega line as the continuous-time counterpart.

### 2.5 The SSM-era restatement

**Gu, Dao, Ermon, Rudra & Ré**, *"HiPPO: Recurrent Memory with Optimal Polynomial Projections"*,
**NeurIPS 2020**, arXiv:2008.07669. **[ABS]** HiPPO frames memory as *"online compression of
continuous signals and discrete time series by projection onto polynomial bases"* and *"produces an
optimal solution to a natural online function approximation problem."* That is the same bound in
approximation-theoretic clothing: `N` coefficients, `N`-term projection, fidelity set by `N`. One
detail is worth #143's attention: HiPPO-LegS is sold as *"avoiding priors on the timescale"* — the
opposite design stance from S4D and LRU in §5. The abstract does not quantify the error bound in
terms of `N`, and the body was not reached.

---

## 3. Has anyone glued linear recurrences spatially, under a local rule?

**Verdict: not the conjunction. But every pair of its three properties is in print, two of them in
work published since #148 ran, and one of them by the group whose sheaf vocabulary the design uses.
The residue is narrower than the record assumes and should be restated as one property, not three.**

### 3.1 The closest object: per-node linear recurrence, nonlinearity outside, glued over edges

**Ceni, Gravina, Gallicchio, Bacciu, Schönlieb & Eliasof**, *"Message-Passing State-Space Models:
Improving Graph Learning with Modern Sequence Modeling"*, arXiv:2505.18728. **[FULL — body read via
arXiv HTML]**

This is the nearest published object to a Patchworks dome read as a recurrence, and it is nearer than
anything #148 found for the Koopman reading.

- **The recurrence is linear and the paper knows that is the point.** Verbatim: *"A key distinction
  from such models lies in the use of a purely linear recurrent equation."* The update is
  `X_{t+1} = A X_t W + U_{t+1} B` — a per-node state matrix `X`, a graph operator `A` mixing across
  edges, a learned `W` advancing the state.
- **The nonlinearity is outside the loop, exactly as in §1.1.** Verbatim: *"Each block is composed of
  k iterations of the linear recurrence… followed by a learnable graph-agnostic nonlinear mapping."*
- **The linearity buys an exact sensitivity statement**, which is the same currency ADR-0023 spends:
  *"The Jacobian of the linear recurrent equation… can be computed exactly, and it has the following
  form: ∂X_t^(i)/∂X_s^(j) = (A^{t−s})_{ij} (Wᵀ)^{t−s}"*, used for lower bounds on information flow
  and over-squashing.
- **Trained end-to-end by backpropagation**, with residual connections, normalisation and dropout.

So *"a spatially glued set of linear recurrences works"* is **claimed**. ADR-0023's sentence *"Nobody
has shown a spatially glued set of locally-trained linear recurrences works"* survives only on the
word **locally-trained**, and it should be read that narrowly from here on.

Second instance, temporal rather than depth-indexed: **Li, Wu, Jin, Ma, Chen & Zheng**, *"State Space
Models on Temporal Graphs: A First-Principles Study"* (GraphSSM), arXiv:2406.00943.
**[FULL — body read via arXiv HTML]** Each node maintains a latent state evolving through a linear
dynamical system, `U_l = U_{l−1} e^{Δ_l A} + Δ_l X̂_l Bᵀ`, with the diffused features `X̂` carrying
neighbourhood information via learned GNNs. Also end-to-end backprop; no local rule.

Third, the one #148 already knew, confirmed here for the recurrence framing: **Arroyo, Gravina,
Gutteridge, Barbero, Gallicchio, Dong, Bronstein & Vandergheynst**, *"On Vanishing Gradients,
Over-Smoothing, and Over-Squashing in GNNs: Bridging Recurrent and Graph Learning"*, arXiv:2502.10818,
**NeurIPS 2025**. **[ABS]** Verbatim: *"We propose an interpretation of GNNs as recurrent models and
empirically demonstrate that a simple state-space formulation of a GNN effectively alleviates
over-smoothing and over-squashing at no extra trainable parameter cost."*

### 3.2 The local-rule half, with the sheaf vocabulary attached

**Bosca & Ghrist**, *"Neural Networks as Local-to-Global Computations"*, arXiv:2603.14831 (March
2026). **[ABS]**

Verbatim: *"We construct a cellular sheaf from any feedforward ReLU neural network by placing one
vertex for each intermediate quantity in the forward pass and encoding each computational step —
affine transformation, activation, output — as a restriction map on an edge."* The forward pass is
*"the unique harmonic extension of the boundary data"*, and the training is *"training through local
discrepancy minimization without a backward pass"*, driven by sheaf heat equations rather than
backpropagation.

**This is Ghrist**, co-author of the spectral sheaf theory #148 §10 leans on. Cellular sheaves +
learned restriction maps + a strictly local, backpropagation-free training rule is therefore **in
print, from the originating group**. What it does not have is a recurrence in world-time: the
"dynamics" is a relaxation to a fixed point that computes one forward pass, not a persisting state
advanced by evidence. That is a real difference and it is the one that keeps Patchworks distinct
here — but it is a *smaller* difference than "nobody does local rules on sheaves", which is what the
record's tone implies.

Adjacent, sheaf-and-spatiotemporal but backprop-trained: **"Dynamic Sheaf Diffusion Networks with
Adaptive Local Structure for Heterogeneous Spatio-Temporal Graph Learning"** (ST-Sheaf GNN),
arXiv:2604.11275. **[ABS]** It *"embeds graph topology into sheaf-based vector spaces connected by
learned linear restriction maps"* and *"learns dynamic restriction maps that evolve over time."* The
abstract does not settle whether node state persists across time steps or how it is trained; the body
was not reached.

Also located, not read: **"Learning Multi-Agent Coordination via Sheaf-ADMM"**, arXiv:2605.31005.
**[CITE]** Cellular sheaves used to formalise a low-dimensional agreement space for distributed
agents under a decomposition method that is local by construction. If any single source is going to
close the residue in §3.4, this family is where it will come from, and it was not reachable in this
pass.

### 3.3 The oldest analogue: local linear filters reconciled by a local rule

Two lines predate all the machine-learning work and are the strongest evidence that the *idea* is
not new, only the vocabulary.

- **Kalman-consensus filtering.** Olfati-Saber, *"Distributed Kalman filtering for sensor networks"*,
  **IEEE CDC 2007**, and *"Distributed Kalman Filter with Embedded Consensus Filters"*, **CDC 2005**.
  **[CITE — landing pages and downstream restatements confirmed; bodies behind IEEE paywall]** Every
  node runs its own linear estimator over its own measurements, exchanges only with graph
  neighbours, and reconciles by a local consensus term. That is per-node linear recurrence + spatial
  gluing + a purely local update rule — **all three of Patchworks' properties**, with the operator
  *given* rather than learned. What Patchworks adds to this line is that `K` is learned; what the
  line has that Patchworks does not is fifty years of stability analysis.
- **Local module identification in dynamic networks.** Van den Hof, Ramaswamy, Dankers & Bottegal,
  *"Local module identification in dynamic networks with correlated noise: the full input case"*,
  **IEEE CDC 2019**, arXiv:1809.07502. **[ABS]** *"an algorithm that, based on the given network
  topology and disturbance correlation structure, selects an appropriate set of node signals as
  predictor inputs and outputs"*, with *"selected output node signals … predicted based on all of
  their in-neighbor node signals in the network."* One local **linear** module identified from
  neighbour signals only, with no global model — which is the estimation half of a cell's local
  prediction rule, in control-theory dress. This literature also carries the identifiability
  conditions Patchworks' rule has never been checked against; noted, not pursued.

### 3.4 Reservoirs on graphs — the untrained precedent

**Gallicchio & Micheli**, *"Graph Echo State Networks"*, **IJCNN 2010**, doi
`10.1109/IJCNN.2010.5596796` **[CITE — bibliographic record and multiple downstream descriptions
confirmed; body not reached]**, and **Tortorella & Micheli**, *"Dynamic Graph Echo State Networks"*,
**ESANN 2021**, arXiv:2110.08565 **[CITE]**. Per-node reservoir state, a contractive update computed
purely from a node's own state and its neighbours', the recurrence left **untrained** after random
initialisation, and only a linear readout fitted. DynGESN extends this to temporal graphs with
per-node state persisting across time steps.

This is the closest thing to "no backprop through the graph" in the graph-recurrence literature, and
it makes the point precisely: the field's way of avoiding a global objective is to **not learn the
recurrence at all**. Patchworks learns `K` under a local rule, which is a third position neither the
reservoir line nor the graph-SSM line occupies.

### 3.5 The honest residue

| Property | Status |
|---|---|
| Per-node persisting linear state advanced by a linear recurrence | **Claimed** — MP-SSM, GraphSSM, GraphESN/DynGESN |
| Nonlinearity outside the per-node recurrence | **Claimed** — MP-SSM (and see §1: Patchworks does not have this) |
| Glued spatially over graph edges, coupled not consensused | **Claimed** — MP-SSM, GraphSSM; and #148 §1.3 for the Koopman framing |
| Learned linear restriction maps into communication lanes | **Claimed** — Neural Sheaf Diffusion; ST-Sheaf GNN |
| Cellular sheaf + restriction maps + backpropagation-free local rule | **Claimed** — Bosca & Ghrist |
| Local linear filters reconciled by a purely local rule over a graph | **Claimed** — Kalman-consensus filtering |
| Per-node **learned** operator, glued by learned restriction maps, under a **local rule**, with a **persisting state in world-time** | **Not found in any single source** |

The last row is the claim, and it is a conjunction of four properties each of which is individually
old. That is the same shape as #148's verdict and the same shape #127's Notes said was the good
outcome. **It is not a loud "yes", but it is a quieter "yes" than the record currently allows for**,
and the sentence in ADR-0023 should be read as resting entirely on the local rule.

---

## 4. What the transfer does not cover

**Verdict: ADR-0023's boundary is drawn in the right place and labelled backwards. Mechanism
transfers, as it says. But so do the capacity results — and they are the unfavourable ones. What does
not transfer is the reassurance.**

ADR-0023: *"The state-space-model literature is about deep stacks trained by backprop on sequence
benchmarks. Patchworks is a shallow-per-cell sheaf trained by local rules. Linear recurrence,
timescale as an eigenvalue, and stability by norm bound transfer. **No result does.**"*

**Confirmed, for the expressivity results.** Everything in §1.1–§1.2 is conditioned on a stack. LRU's
evidence is six layers on one benchmark, by the authors' own framing. The universality theorem needs
interleaved MLPs. Merrill et al.'s TC⁰ result is about layer counts and precision. None of these say
anything about one shallow cell trained by a local prediction rule, and #148 §11 already priced what
local training costs (Bartunov et al.'s 93–99% top-1 on ImageNet against backprop's ~71.4%; InfoPro's
information collapse under greedy local objectives, with its conditional escape). Nothing found in
this pass changes that.

**Challenged, for the capacity results, and this is the more valuable half.** Jaeger, Dambre et al.,
Ganguli et al., White et al. and Gonon et al. are **not** statements about training procedures. They
are statements about how much of an input history a state of dimension `N` can hold, given a readout
class and an input process. Three reasons they transfer intact:

1. **Dambre et al. state theirs for arbitrary dynamical systems**, not for trained networks — the
   paper's framing is *"Many dynamical systems, both natural and artificial, are stimulated by time
   dependent external signals"*, and the whole point of the metric is to compare physical substrates
   nobody trains at all.
2. **Reservoir computing is the shallow, non-backprop case, and it is where these results were
   born.** A cell with a frozen `encode`, a rank-≤12 learned recurrent factor and a frozen linear
   `decode` (§1.4) is closer to Jaeger's object than to Orvieto's.
3. **The chart's dimension is a construction constant.** No amount of local-vs-global training moves
   `rank(W_rec) ≤ 12`.

So the ADR's *"No result does"* is too strong, and the correction is not comforting: the results that
survive the shallow, locally-trained setting are the ones that **bound** the design, and the ones
that would have licensed it are the ones that need the stack. This is worth more than the reassurance
was, exactly as the ticket predicted, and it lands on #166 rather than on ADR-0023's Decision.

One thing that would transfer favourably if checked, and was not: Merrill et al.'s Theorem 5.1 is a
**one-layer** construction. A shallow in-loop-nonlinear recurrence recognising any regular language
at `k = |Σ|` is a positive expressivity result for a single cell. Whether the constant it needs is
near 12 for anything a cell must do was not established here.

---

## 5. Timescale from eigenvalues as a designed constant

**Verdict: #143's reading is confirmed at source and is the field's actual practice, not a charitable
gloss. No certification-style objection of #148 §5's kind exists, and the reason it does not is
structural rather than accidental. But two objections of a different kind do exist and #143 needs
both.**

### 5.1 The field sets the spectrum; it does not estimate one

- **S4D.** Gu, Gupta, Goel & Ré, *"On the Parameterization and Initialization of Diagonal State Space
  Models"*, **NeurIPS 2022**, arXiv:2206.11893. **[FULL — body read via ar5iv]** `Δ` is drawn
  log-uniformly: `log_dt = rand() * (log(dt_max) − log(dt_min)) + log(dt_min)`, with `(dt_min,
  dt_max) = (1e-3, 1e-1)` and `(1e-4, 1e-2)` for Path-X. On the eigenvalues: *"the real part of Aₙ
  controls the decay rate of the function"*, and *"Aₙ = −1/2 is a good default that bounds the basis
  functions by the envelope e^{−t/2}, giving a constant timescale."* A **designed** constant timescale,
  chosen for a bound, in the paper's own words. This corroborates
  `docs/research/027-regional-jacobian-spectra.md`'s verified-at-source reading and extends it from
  the code comment to the paper's argument.
- **LRU.** Orvieto et al. **[FULL]** Eigenvalues are parameterised `λⱼ = exp(−νⱼ + iθⱼ)` and
  initialised on a ring: for `0 ≤ r_min ≤ r_max ≤ 1`, `ν = −½ log( u₁(r²_max − r²_min) + r²_min )`,
  `θ = 2π u₂`. The design intent is stated outright: *"for reasonings requiring consideration of
  interactions between distant tokens, eigenvalues in the recurrence need to have magnitude close to
  1."* Magnitude **is** retention length, chosen, not measured. Their Proposition 3 prices it — the
  forward-pass norm scales as `1/(1 − r²)` as `r_max → 1`, which is why LRU adds `γ` normalisation.
- **Mamba.** Gu & Dao, arXiv:2312.00752. **[FULL — §3.5.2 read via ar5iv]** *"In general, Δ controls
  the balance between how much to focus or ignore the current input x_t."* And: *"Mechanically, a
  large Δ resets the state h and focuses on the current input x, while a small Δ persists the state
  and ignores the current input."* `Δ` is initialised to `τ_Δ^{-1}(Uniform([0.001, 0.1]))`. This is
  the *retention-policy* reading of a timescale in the clearest possible terms, and it is exactly
  ADR-0023's *"A slow eigenvalue says the cell **chose to retain** that direction for that long."*

**No source found in this pass treats a trained SSM's `λ` as an estimate of a true spectrum of
anything, and none asks whether it converges.** The reason is structural: an SSM's `A` is a design
parameter of a model of the *task*, whereas a Koopman operator's spectrum is a property of a
*system* that exists independently of the estimator. #148 §5's impossibility results are about the
second kind of object. **#143's inherited reading is correct and citable.** The countervailing note
from §2.5 stands: HiPPO markets *"avoiding priors on the timescale"*, so "set it by construction" is
the dominant practice but not a unanimous one.

### 5.2 A gap worth naming

The `τ = −1/ln|λ|` formula itself was **not located as a stated convention in any SSM paper read
here.** What the field states is the equivalent facts — decay as `O(|λ|^k)` (LRU), the real part
controlling decay rate (S4D), `Δ` as focus/persist (Mamba). The conversion to a time constant is
arithmetic and uncontroversial, but if the record wants a citation *for the formula*, this pass did
not find one, and the honest move is to present it as arithmetic rather than as an inherited
convention.

### 5.3 The two objections #143 does need

Neither is a certification objection. Both are sharper for being about design rather than estimation.

- **Constraining the eigenvalue range is provably not free.** Grazzi, Siems, Zela, Franke, Hutter &
  Pontil, *"Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues"*, **ICLR 2025**,
  arXiv:2411.12537. **[ABS]** They *"prove that finite precision LRNNs with state-transition matrices
  having only positive eigenvalues cannot solve parity"*, trace Mamba's failure to *"restricting the
  value range of their diagonal state-transition matrices to [0, 1]"*, and show extending to `[−1,
  1]` fixes it. **`K = a·I` at construction has all eigenvalues equal to `+a`.** ADR-0015's band is on
  `σ_max(K)`, a norm, and constrains nothing about eigenvalue sign or phase — so nothing forbids `K`
  from learning negative or complex eigenvalues, and this is an observation about the *initialisation*,
  not a defect in the band. It is worth #143 knowing that the field has a theorem saying the sign
  structure of the spectrum is expressively load-bearing, since #143 is the ticket proposing to read
  meaning off that spectrum.
- **Long timescales are expensive in a way that is not vanishing gradients.** Zucchet & Orvieto,
  *"Recurrent neural networks: vanishing and exploding gradients are not the end of the story"*,
  **NeurIPS 2024**, arXiv:2405.21064. **[ABS]** Verbatim: *"as the memory of a network increases,
  changes in its parameters result in increasingly large output variations."* They identify
  element-wise recurrence plus careful parameterisation as the mitigations SSMs and LSTMs use. A dense
  learned `K` with eigenvalues placed near the unit circle is precisely the configuration this
  sensitivity result is about, and Patchworks has no element-wise structure to fall back on. This is a
  **learning-dynamics** cost of slow modes that the record has not priced.
- Related, located but not read: **Li, Han, E & Li**, *"On the Curse of Memory in Recurrent Neural
  Networks: Approximation and Optimization Analysis"*, **ICLR 2021**, arXiv:2009.07799. **[ABS]** For
  linear RNNs, *"when there is long term memory in the target, it takes a large number of neurons to
  approximate it"* and *"the training process will suffer from slow downs"*, with *"both of these
  effects becom[ing] exponentially more pronounced with memory."* The abstract does not settle the
  polynomial-decay case; the body was not reached. If it applies to `K` at `k = 12`, it is a second
  argument that slow modes are dear.

---

## 6. Escaping the orbit: the sources that disagree

The ticket required this and it changed the verdict twice. Note the shape of what came back: the
strongest dissent about **memory capacity** comes from the authors who proved the modern form of the
bound, and the strongest dissent about **linear recurrence** comes from the group that built the
architectures.

### 6.1 The memory-capacity metric, attacked by the people who formalised it

**Ballarin, Grigoryeva & Ortega**, *"Memory Capacity of Nonlinear Recurrent Networks: Is it
Informative?"*, arXiv:2502.04832 (Feb 2025, rev. Sep 2025). **[ABS]**

Verbatim abstract: *"The total memory capacity (MC) of linear recurrent neural networks (RNNs) has
been proven to be equal to the rank of the corresponding Kalman controllability matrix, and it is
almost surely maximal for connectivity and input weight matrices drawn from regular distributions.
This fact questions the usefulness of this metric in distinguishing the performance of linear RNNs in
the processing of stochastic signals. This work shows that the MC of random nonlinear RNNs yields
arbitrary values within established upper and lower bounds depending exclusively on the scale of the
input process. This confirms that the existing definition of MC in linear and nonlinear cases has no
practical value."*

**This is the sharpest single threat in the pass, and it is aimed at #166.** Two distinct blows.
First, for linear networks MC is *almost surely maximal* for random matrices — so `MC = N` is generic
and discriminates nothing. Second, for nonlinear networks (which §1.4 shows is the case that applies)
MC takes *arbitrary* values inside its bounds, set **only by the input's scale**. Patchworks' input
scale is a design variable: the **scale gauge** and ADR-0010's `ρ` on the restriction maps set the
magnitude of what arrives on a node stalk. If Ballarin et al. are right, a measured per-cell MC would
be reporting the gauge, not the chart.

The same group's earlier paper, **Ballarin, Grigoryeva & Ortega**, *"Memory of recurrent networks: Do
we compute it right?"*, ***JMLR* 25(243):1–38 (2024)**, arXiv:2305.01457 **[ABS]**, is the
constructive companion: *"Numerical evaluations of the memory capacity (MC) of recurrent neural
networks reported in the literature often contradict well-established theoretical bounds"*, traced to
*"ignor[ing] the Krylov structure of the linear MC"*, with corrected estimators whose curves *"fully
agree with the theory."* If #166 ever measures MC rather than citing it, this paper is the method
note, and skipping it is how the reported numbers end up contradicting the bound the ticket is
invoking.

### 6.2 State dimension is the binding constraint, and it binds harder than "12 taps" suggests

- **Jelassi, Brandfonbrener, Kakade & Malach**, *"Repeat After Me: Transformers are Better than State
  Space Models at Copying"*, **ICML 2024**, arXiv:2402.01032. **[ABS]** They prove *"a two layer
  transformer can copy strings of exponential length while GSSMs are fundamentally limited by their
  fixed-size latent state"*, and find empirically that transformers *"dramatically outperform state
  space models at copying and retrieving information from context"*, concluding there is *"a
  fundamental gap between transformers and GSSMs on tasks of practical interest."* The fixed-size
  latent state is the binding constraint, named as such.
- **Arora, Eyuboglu, Zhang, Timalsina, Alberti, Zinsley, Zou, Rudra & Ré**, *"Simple linear attention
  language models balance the recall-throughput tradeoff"*, **ICML 2024**, arXiv:2402.18668.
  **[ABS]** *"a key tradeoff between a model's state size and recall ability"*; models that *"maintain
  a fixed-size recurrent state … struggle at recall"*; and the remedy is to *"dial the state size"*
  along *"the pareto frontier of the recall-memory tradeoff curve."* The lever the field reaches for
  when recall is short is **width**, which is the move #14's constraint ladder puts last.
- **Merrill, Petty & Sabharwal** (§1.3) is dissent as well as structure: the recurrence *"cannot
  express computation outside the complexity class TC⁰"* and the apparent state is an illusion. This
  is the strongest published statement that removing in-loop nonlinearity costs something exact.
- **Grazzi et al.** (§5.3): parity, the simplest state-tracking task, is unsolvable in one forward
  pass by linear RNNs with positive eigenvalues, *"which non-linear RNNs can handle effectively."*
- **Sarrof, Veitsman & Hahn**, *"The Expressive Capacity of State Space Models: A Formal Language
  Perspective"*, **NeurIPS 2024**, arXiv:2405.17394. **[ABS]** The balanced view, and the one worth
  keeping: *"SSMs and transformers have overlapping but distinct strengths"*; SSMs *"implement
  straightforward and exact solutions to problems that transformers struggle to represent exactly"*
  in star-free state tracking and can *"model bounded hierarchical structure with optimal memory even
  without simulating a stack"*, while the paper also identifies *"a design choice in current SSMs
  that limits their expressive power."*

### 6.3 What the dissent does not reach

The dissent is uniformly about **linear** recurrences and fixed state size under **long-context
language and copying** workloads. Two limits on how far it should be carried:

- Patchworks' recurrence is not linear (§1.3–§1.4), so the TC⁰, parity and star-free results do not
  bind it. They bind the object ADR-0023 says the design is, which is a reason to fix the description
  rather than a reason to worry about the design.
- Copying and multi-query associative recall are not a cell's job. A cell advances a chart of its own
  **piece** under evidence; nothing in the spec asks it to retrieve a token emitted 4,000 ticks ago.
  The recall-throughput frontier is real and is about a workload Patchworks does not have.

What the dissent *does* reach is the **conjunction of §2.3 and §6.1**: on the noisiest reading the
chart's capacity at construction is 1, and on the most recent reading the metric that says otherwise
is not informative. Both bear on #166 and neither bears on ADR-0023's Decision.

---

## What this threatens

### ADR-0023 — one wording defect, one over-strong sentence, no structural damage

The **Decision** stands: the chart is not a Koopman lift, `K` is not advancing a dictionary of
observables, and nothing found here re-opens the EDMD framing. Three flags:

1. **"The comparison class is deep state-space models and linear RNNs" is wrong as written, and the
   ADR's own Consequences section contradicts it.** The SSM class is defined by the nonlinearity
   sitting *between layers*; `K · encode` puts it *inside the loop*, which Merrill et al. §5.1
   distinguishes explicitly and Theorem 5.1 shows is a strict expressivity gain. The matched class is
   **nonlinear RNN / reservoir computing** — see §1.4's change of variables, which lands the cell as a
   45-unit ESN with a rank-≤12 partially-learned reservoir. **The direction of the error is
   favourable**, and the ADR's Consequences sentence *"`K . encode` is a **nonlinear** recurrence"* is
   already the correct statement. **Recommendation only, and it is a wording repair, not a decision
   change: the two sections should be made to agree, and the licence for `k = 12` should be sourced
   from reservoir dimensioning rather than from SSM stacking results.** No ticket is opened here.
2. **"No result does [transfer]" is too strong** (§4). The capacity results transfer completely,
   because they are architecture-level and training-method-independent, and they are unfavourable.
   The ADR's boundary is in the right place; its label needs the exception.
3. **`k = 12` "is unremarkable in that class"** now has a support and a counterweight. Support:
   Dambre's bound is on `rank(W_rec) ≤ 12` and there is nothing anomalous about a 12-dimensional
   recurrent state. Counterweight: §2.3's normal-matrix result and §6.2's recall-throughput frontier
   both say that width is the lever the field reaches for when memory is short, and #14 puts it last.

### #143 — reading confirmed, two new costs to carry

- **The core reading is confirmed at source** (§5.1): S4D, LRU and Mamba all treat `λ` and `Δ` as
  **designed retention constants**, and no certification-style objection of #148 §5's kind exists in
  the SSM literature, for a structural reason. #143 may rewrite `05-timescales.md` around the spectrum
  without importing #148 §5's attack.
- **Carry Zucchet & Orvieto**: long memory buys parameter sensitivity independent of gradient
  vanishing, and the field's mitigations (element-wise recurrence, careful parameterisation) are not
  available to a dense learned `K`.
- **Carry Grazzi et al.**: the sign structure of the spectrum is expressively load-bearing, and
  `K = a·I` starts with every eigenvalue at `+a`. Nothing in ADR-0015's `σ_max` band forbids the
  spectrum from moving; this is an initialisation observation, and a cheap one to check.
- **Note the citation gap** (§5.2): `τ = −1/ln|λ|` was not located as a stated convention in any SSM
  paper read here. Present it as arithmetic.
- The ADR's own caveat that `λ(K)` is the operator's *contribution* to realised timescale, `encode`
  being interleaved, is untouched and is strengthened by §1.4 — the realised recurrent matrix is
  `E_hid^chart K E_out`, and its spectrum is not `K`'s.

### #166 — unblocked, with conditions, and one new instrument

**The block lifts.** The bound exists, was read at source, and is stronger than the ticket states.
Four conditions on citing it:

1. **Cite Dambre et al. (2012), not Jaeger alone.** Dambre's Theorem 4 bounds **total** capacity —
   linear and nonlinear functionals in one budget, with a proven trade-off between memory and
   nonlinearity — which is the exact object #166 calls "double duty". Jaeger is the ancestor and gives
   the hypotheses in the clearest form.
2. **State the hypotheses, and state which one fails.** Linear readout: **holds** (frozen linear
   `decode`, linear restriction maps). i.i.d. input: **fails** — `stalk_t` is written by
   reconciliation and is strongly autocorrelated. Jaeger's own counterexample for correlated input
   yields *infinite* MC, so the failure loosens the bound rather than tightening it. Gonon,
   Grigoryeva & Ortega (2020) is the dependent-input generalisation and should be named as the
   version that would actually apply.
3. **Do not write "12 taps" as a number.** Write "at most 12, and 1 for a normal `K`". Ganguli, Huh &
   Sompolinsky: *"the capacity of networks with normal connectivity matrices is exactly 1 and that of
   any network of N neurons is, at most, N"*, with the bound attainable only under stringent
   nonnormal design. `K = a·I` is normal.
4. **Carry Ballarin, Grigoryeva & Ortega (2025)** — *"the existing definition of MC in linear and
   nonlinear cases has no practical value"*, with the nonlinear case's MC set by the input scale.
   Patchworks sets the input scale by gauge. A measured per-cell MC may be reporting the scale gauge.
   If #166 measures rather than cites, it needs the same group's JMLR (2024) estimator note, or the
   numbers will contradict the bound the ticket is invoking.

**New instrument, cheap, available with tier 1.** Alongside numerical rank of each learned `K`,
measure **non-normality** — `‖K†K − KK†‖_F`, or the departure-from-normality of the Schur form.
Rank saturation says the width is spent; non-normality says whether the width is *usable* as memory
on Ganguli et al.'s measure. `K = a·I` scores zero on the second, so a population that stays near
normal after training is a positive read that the memory job is not being served at all — which is
tier 2's signature arriving early and for free.

### Nothing threatens ADR-0015 or the transmission argument

`σ_max(K)` is a norm bound. Everything in this pass is about spectra, capacity and expressivity.
#148's observation that the spectral attack does not reach a norm bound holds here too, and §1.4's
Lur'e reading offers an *additional* certificate family for `K · encode` as a whole rather than a
threat to the existing one.

---

## What could not be reached

Stated rather than papered over.

- **Ganguli, Huh & Sompolinsky (PNAS 2008)** — body reached via PMC and quotations are verbatim, but
  the publisher's PDF returned HTTP 403 and the derivation of the `≤ N` bound and its noise model were
  not read in full. The quoted statements are from the paper's own summary of its results.
- **Hermans & Schrauwen (*Neural Networks* 2010)** — **[CITE]**. Publisher and PubMed records
  confirmed; abstract page blocked by a cookie gate and body paywalled. Its result (`MC` = rank of the
  controllability matrix, continuous time) is reported here only via the Grigoryeva–Ortega line's
  restatement of it.
- **Olfati-Saber's Kalman-consensus filtering papers (CDC 2005, CDC 2007)** — **[CITE]**. IEEE
  paywall. §3.3's characterisation rests on multiple downstream restatements, not on the originals.
- **Gallicchio & Micheli, *Graph Echo State Networks* (IJCNN 2010)** — **[CITE]**. Bibliographic
  record and several independent downstream descriptions confirmed; body behind IEEE paywall.
- **Suykens, Vandewalle & De Moor (IEEE TAC 1999)** on Lur'e systems with neural nonlinearities —
  **[CITE]**. Landing pages located, body not reached. §1.4's Lur'e framing is the pass's own and rests
  on the standard textbook form of a Lur'e system, not on this paper.
- **ST-Sheaf GNN (arXiv:2604.11275)** — **[ABS]**. Whether node state persists across time steps and
  how the model is trained were not settled; the abstract does not say and the body was not read.
- **"Learning Multi-Agent Coordination via Sheaf-ADMM" (arXiv:2605.31005)** — **[CITE]**. Title and
  framing located only. This is the most likely place for §3.5's residue to shrink further, and it is
  the largest single gap in this pass.
- **Li, Han, E & Li, curse of memory (ICLR 2021)** — **[ABS]**. Whether the exponential penalty
  applies to polynomially-decaying memory, and at what `N`, was not established.
- **"Task-Level Insights from Eigenvalues across Sequence Models" (arXiv:2510.09379)** —
  **[UNREACHED]**. Surfaced in search as an empirical study of eigenvalue spectra across sequence
  models; neither abstract nor body reached. Potentially relevant to #143.
- **Dambre et al.'s proof of Theorem 4** — the statement and hypotheses were read in the body; the
  proof itself was not followed. The claim that the bound survives non-i.i.d. input was **not** made
  by that paper and is not asserted here.
- **Not asked of anyone**: whether a sheaf over a graph, with `encode` between hops and unit delay on
  every edge, satisfies the *stacking* hypothesis of §1.2's universality theorem in space rather than
  in depth. This pass found nobody who has posed the question. It is the single most interesting open
  question the pass surfaced and it is not a citation gap — it is an unclaimed result.

---

## Context

Opened and specified by
[patchworks#167](https://github.com/NGL321/patchworks/issues/167), *"Citation pass: linear recurrences
with the nonlinearity outside, and who disagrees"*, part of
[#127](https://github.com/NGL321/patchworks/issues/127). Audits
[ADR-0023](../adr/0023-the-chart-is-not-a-koopman-lift.md), raised by
[#145](https://github.com/NGL321/patchworks/issues/145). Unblocks
[#166](https://github.com/NGL321/patchworks/issues/166) and reports to
[#143](https://github.com/NGL321/patchworks/issues/143). Companion to
`docs/research/148-local-linear-operator-citations.md`, whose form, reading-depth key and per-source
discipline this document follows.

---

# Follow-up pass — 2026-08-30 (patchworks#173): the two bodies §3 could not reach

Opened by [patchworks#173](https://github.com/NGL321/patchworks/issues/173) as a targeted gap-closing
pass, not a survey. §3 above left the design's novelty residue **one property wide** and named two
sources it could not reach as the most likely places that residue would shrink. This pass reads both
bodies and follows the citation graph **one hop** out of the closer of the two. It does not re-run
§3, and it revises no closed design; new threats are flagged in §7.5 below.

**Both arXiv identifiers resolve.** `2605.31005` and `2604.11275` are real, current papers, and this
pass reached **both bodies in full** via arXiv's HTML renderings. The two `[CITE]`/`[ABS]` entries in
*What could not be reached* are hereby discharged and upgraded to `[FULL]`.

## 7.1 Verdict, stated plainly

**Neither source claims the conjunction, and the residue survives — but it moved, and it moved by
more than one property's worth of ground.**

Sheaf-ADMM is much closer than §3.2's one-line `[CITE]` entry allowed for. It is the **first single
source in this record to hold three of the conjunction's properties at once**: a learned per-agent
operator, learned restriction maps into communication lanes, and an update rule that is local by
construction. §3.5's table had those three claimed only across *separate* sources. That is a real
narrowing and the record should stop implying otherwise.

What Sheaf-ADMM does **not** have is precisely the pair Patchworks is built on, and it fails on both
in the loudest possible way:

- It has **two** global objectives stacked on each other — an inner one the ADMM iterations are
  provably solving, and an outer task loss that is backpropagated through the whole unrolled solver.
- It has **no world-time at all**. The agent state is re-initialised to zero on every input and the
  iterations relax toward a fixed point inside a single forward pass. Coordination *over time* is
  listed in the paper's own Future Directions.

ST-Sheaf GNN does not come close. Both discriminating questions §3.2 flagged resolve against it: node
state does **not** persist across time steps, and it is trained by ordinary end-to-end backpropagation
under a global forecasting loss.

## 7.2 Sheaf-ADMM, read in full

**Seely, Cupiał & Jones** (Sakana AI, Tokyo; Cupiał also University of Warsaw), *"Learning Multi-Agent
Coordination via Sheaf-ADMM"*, arXiv:2605.31005v1 **[cs.LG], 29 May 2026, CC BY 4.0.
[FULL — body read via arXiv HTML]**

The four discriminating questions #173 asked, answered from the body.

### (1) Are the per-agent operators learned? — **Yes, but amortised, and they are not a recurrence.**

Each agent solves a convex subproblem whose *parameters* are emitted by a shared learned encoder:
*"An encoder network `Enc_θ(d_i)` produces the parameters of `f_i`, shared across agents."* For the
quadratic family this collapses to a per-agent **linear solve**:

> `x_i^{k+1} = (Q_i + ρI)^{-1}(ρ(z_i^k − u_i^k) − q_i)`
>
> *"This is a single linear solve per agent. The encoder outputs `(Q_i, q_i)`."*

So there is a learned, per-agent, linear operator — nearer to a **cell**'s `K` than anything in §3.
Two differences, both structural. First, it is not the agent's own parameter: `Q_i` is produced by a
network **shared across all agents** from that agent's local view, so what is learned is one map from
view to operator, not `N` operators. Second, and more importantly, it is applied to the *consensus
target* `z_i − u_i`, not to the agent's own previous state — it is a proximal step, not a recurrence.

### (2) Are the restriction maps learned? — **Yes, unambiguously, and this is the paper's headline contribution.**

> *"Our framework retains the ADMM semantics from Hanks et al. (2025b) but learns the sheaf structure
> and local agent subproblem parameterizations end-to-end from data."*

> *"Restriction maps `F_{i→e}` determine what information agents share. On grid graphs, edges have
> spatial directions (up, down, left, right for 4-connectivity). We learn a base restriction map per
> direction, shared across all agents. The encoder outputs low-rank updates `ΔF_i = U_i V_i^ᵀ` that
> modulate the base maps: `F_{i→e} ← F_{i→e} + ΔF_i`."*

And in the training list: *"Beyond encoder weights `θ` and decoder weights `ψ`, we learn the base
restriction maps `{F_{i→e}}` (shared by direction) and the penalty `ρ` (positivity enforced via
softplus)."*

The vocabulary is also the same vocabulary, not a paraphrase of it — see §7.5.

### (3) Is there a global objective the local updates are provably solving? — **Yes. Twice.**

This is the decisive answer and it decides in Patchworks' favour.

**Inner objective.** The ADMM iterations are a solver for the consensus problem

> `minimize_x Σ_{i=1}^{N} f_i(x_i)  subject to  x ∈ C`   (Eq. 6)

with the sheaf supplying the constraint set, `C = {x | x ∈ ker(F)}`. The method section says so
directly: *"Agents then run `K` iterations of ADMM toward solving (6)."* The z-update is gradient
descent on the sheaf Dirichlet energy, and the paper states the convergence property it is relying
on: *"Starting from an arbitrary initial state `x^0`, gradient descent on `xᵀL_F x` converges to the
projection of `x^0` onto `ker(L_F) = ker(F)`."* Convexity is deliberate: *"In Sheaf-ADMM, the encoder
and decoder handle non-convexity but the latent iterations are convex. Convexity allows for
convergence guarantees but is not a strict requirement."*

**Outer objective.** The whole solver is a differentiable layer inside a backprop graph:

> *"All operations are differentiable. We unroll `K` ADMM iterations and minimize a task loss between
> the global prediction `Ŷ` and the target `Y`. Gradients flow through decoder, ADMM iterations, and
> encoder."*

and in the experimental setup, *"Models are trained end-to-end with cross-entropy loss."*

Locality here is the locality of a **decomposition**, not of a learning rule. The update is local
because ADMM decomposes a global problem into per-agent pieces — the global problem is still there,
named, and is the thing the locality is in service of. Patchworks has no such object at any level:
there is no functional the transport rule descends on globally and no loss backpropagated through the
graph. **This is exactly the property #173 predicted would keep the residue alive, and it does.**

### (4) Is there a state persisting in world-time? — **No. It is a relaxation to a fixed point.**

Algorithm 1, verbatim from the paper's own pseudocode:

```
def forward(d, F, K, T):
    for i in agents:
        params[i] = encoder(d[i])
    z, u = zeros(), zeros()
    for k in range(K):
        ...
```

The consensus and dual variables are **zeroed at the start of every forward pass**; the loop index
`k` is a solver iteration, not a tick. The output is read off the final iterate: *"After `K` ADMM
iterations, a shared decoder maps each agent's final state and local view to a local prediction."*
The tasks — MNIST, maze pathfinding, Sudoku — are all static structured prediction. Convergence
behaviour is characterised as a fixed-point approach (*"Performance improves with the number of ADMM
iterations `K` up to a task-dependent saturation point"*; *"local proposals repeatedly revise against
the consensus before converging"*), and the paper puts world-time squarely in its Future Directions:
*"dynamic environments requiring coordination over time."*

So Patchworks' **tick** has no counterpart here. Sheaf-ADMM's `K` is the analogue of a
**message-passing phase** iterated to convergence — which is exactly the reading CONTEXT.md forbids
for Patchworks (*"Exactly one simultaneous step per tick — not an iterative solve run to
convergence"*). The two designs differ on that sentence.

## 7.3 ST-Sheaf GNN, read in full

**Mostafa, Younis & Ahmadi**, arXiv:2604.11275. **[FULL — body read via arXiv HTML]** Note a title
discrepancy worth recording: the arXiv landing page carries *"Dynamic Sheaf Diffusion Networks with
Adaptive Local Structure for Heterogeneous Spatio-Temporal Graph Learning"* (as cited in §3.2), while
the HTML rendering of v1 is titled *"Sheaf Diffusion with Adaptive Local Structure for Spatio-Temporal
Forecasting"*. Same paper, same authors, same abstract.

Both questions §3.2 left open are now settled, and both settle against the conjunction.

**(a) Does node state persist across time steps? — No, and the paper says so in one sentence.**

> *"The sheaf diffusion will operate independently on each timestep `t ∈ {1,…,T}`."*

Time is handled *before* the sheaf, by attention over a fixed window: *"To model temporal
dependencies, we apply self-attention along the time dimension independently for each node."* The
whole input window is ingested at once (12 steps for the traffic sets, 48 for Air Quality), each
frame is diffused separately, and *"After `L` diffusion layers, the final representations are decoded
to forecasts `Ŷ` … via a linear projection."* The `ℓ` index is depth, not world-time. There is no
state carried from one world-time step to the next — which is the property §3.5's last row turns on.

**(b) How is it trained? — Ordinary end-to-end backpropagation under a global loss.**

> *"We utilize Adam optimizer with an initial learning rate set to 0.01, a mini-batch size of 12 (to
> match sequence length), and the MAE loss function."*

No local rule anywhere. What it does have, and what §3.5 already credits it with, is genuinely
learned restriction maps, and they are *input-conditioned* rather than merely learned:
`[r_{u⊴e}, r_{v⊴e}] = MLP([h_u ‖ h_v])`, diagonal, *"conditioned on the node representations at a
given time step."* That is a stronger form of "learned restriction map" than Patchworks uses.

**Net effect on the residue: none.** ST-Sheaf GNN was already `[ABS]`-credited in §3.5's
"learned linear restriction maps" row and it stays there; it acquires no other row.

## 7.4 One hop out: the sheaf-ADMM / distributed-optimisation-on-sheaves neighbourhood

Followed one hop from Sheaf-ADMM, the source closest to the conjunction. The neighbourhood is a
single research line (Hanks, Riess, Hale, Fairbanks, and collaborators) and it is uniformly a
**control-theory** line: sheaves given, dynamics given, guarantees proved.

- **Hanks, Riess, Cohen, Gross, Hale & Fairbanks**, *"Distributed Multi-agent Coordination over
  Cellular Sheaves"*, **IEEE CDC 2025**, pp. 3057–3064, doi `10.1109/CDC57313.2025.11312066`,
  arXiv:2504.02049. **[CITE]** Sheaf-ADMM's own characterisation, verbatim: it *"develop[s]
  sheaf-constrained ADMM for multi-agent problems with fixed, hand-specified sheaves."* Restriction
  maps **not** learned. This is the direct parent of §7.2 and it is one property further away.
- **Zhao, Hanks, Riess, Cohen, Hale & Fairbanks**, *"Asynchronous Nonlinear Sheaf Diffusion for
  Multi-Agent Coordination"*, arXiv:2510.00270. **[ABS]** Relaxes synchrony — of interest to
  Patchworks' tick semantics — but the result is a **global-objective** theorem: under bounded
  delays, nonlinear sheaf diffusion converges to a minimizer of the Dirichlet energy of the
  coordination sheaf at a linear rate. Same shape as everything else in this line: locality in
  service of a functional. Body not reached.
- **Anwer, Riess & Hale**, *"Multi-Agent System Identification with Nonlinear Sheaf Diffusion"*,
  arXiv:2605.11204 **[eess.SY], 11 May 2026. [FULL — body read via arXiv HTML]** The one neighbour
  that both *learns* something and has *state persisting in real time*, so it was read rather than
  skimmed. It is nonetheless not the conjunction: the sheaf is **given**, verbatim *"Suppose the
  sheaf `F` is fixed and known to the observer"*, the estimation is an offline global least squares
  over collected trajectories, and the thing recovered is a shared edge potential, not a per-agent
  operator. The paper draws its own boundary against the learning literature exactly where this
  pass would: sheaf-Laplacian graph-learning work consists of *"works that learn the sheaf operator
  for a downstream task, in contrast to our inverse problem of recovering a fixed edge potential
  from trajectory data."* It carries a warning for Patchworks all the same — see §7.5.
- **Hanks, Nino, Bou Barcelo, Copeland, Dixon & Fairbanks**, *"Heterogeneous multi-agent multi-target
  tracking using cellular sheaves"*, arXiv:2512.24886. **[ABS]** Harmonic extension on a sheaf with a
  decentralised control law and Lyapunov stability analysis. Dynamics given, sheaf given.
- Also surfaced, **[UNREACHED]** beyond a search hit and not pursued: *"A Sheaf Framework for
  Strategic Multi-Agent Systems: From Consensus to Nash Equilibria"*, arXiv:2606.01663. Framing
  suggests game-theoretic equilibria — another global solution concept — but neither abstract nor
  body was read and nothing here rests on it.

Stopping here, per #173's one-hop bound. The line is consistent enough that a second hop is unlikely
to pay: every member solves a stated global problem by local means.

## 7.5 What this threatens

Flags only. No closed design is revised here, and nothing in `docs/adr/`, `CONTEXT.md` or `src/` is
touched by this pass.

- **The vocabulary is no longer unoccupied, and the collision is exact.** Sheaf-ADMM uses *edge
  stalk* — the name Patchworks used for the communication lane until #411 §6 — in that exact sense
  (*"Each edge `e ∈ E` is assigned its own edge stalk `R^{d_e}` — the space in which neighboring
  agents compare their states"*), *restriction map* in Patchworks' sense,
  *vertex stalk dimension* for what CONTEXT.md calls the node stalk, and *disagreement* for what
  CONTEXT.md calls disagreement — with the sheaf Laplacian quadratic form written out as the sum of
  squared edge disagreements, the same object CONTEXT.md names. It also calls its cells **agents**,
  a word CONTEXT.md's *Cell* entry explicitly avoids. This is a **naming-collision flag for
  CONTEXT.md's owners**, not a design problem: the terms are used compatibly, but a reader arriving
  from this paper will import *consensus* (full agreement, iterated to convergence) where
  Patchworks means *reconciliation* (one penalised step per tick). Worth an explicit disambiguation
  the way the *Sheaf cohomology* entry already disambiguates Baudot & Bennequin.

  **Narrowed by [#415](https://github.com/NGL321/patchworks/issues/415), which #414's sweep handed the
  call to: the collision is no longer exact, and the surviving half is the half that always mattered.**
  The rename defused *edge stalk* outright — the word is Sheaf-ADMM's alone now, and `communication
  lane` collides with nothing. What still collides is **`restriction map`** and **`disagreement`**, and
  both are used compatibly, so they are a reading aid rather than a hazard. The hazard the bullet
  actually names is untouched by any rename: a reader arriving from this paper imports *consensus*
  where Patchworks means *reconciliation*. Kept, rather than struck, for that reason and one other —
  it is the record of what the rename bought, which is one of three terms.
- **Globally-shared learned restriction maps were empirically not enough on two of three tasks.**
  Sheaf-ADMM's ablation, verbatim: *"Fixed identity maps recover partial performance on MNIST but
  fail on Maze and Sudoku. Learned shared maps suffice on Sudoku (92.5%) but not on Maze (8.9%);
  LoRA modulation closes the Maze gap (99.8%)."* The gap that input-conditioning closed is
  8.9% → 99.8%. ST-Sheaf GNN reaches the same conclusion independently: *"we find that dynamic
  signal-conditioned maps are essential for spatio-temporal forecasting tasks."* Patchworks'
  restriction maps are per-edge and learned but **not input-conditioned** — they are parameters, not
  functions of the node stalk. Two independent 2026 papers now report that this distinction mattered
  on their tasks. Neither task resembles a Patchworks dome and neither result transfers directly,
  but this is the first evidence in the record bearing on a capacity question the design has never
  been asked. **Flagged for whichever ticket owns restriction-map parameterisation.**
- **Dense restriction maps as a named failure mode.** Sheaf-ADMM's Limitations, verbatim: a failure
  occurs when *"the learned restriction maps become effectively dense/high-dimensional potentially
  removing any benefit of sparse communication."* That is the same hazard the **effective rank**
  diagnostic exists to watch, seen from the other side — Patchworks watches for rank *collapse*,
  this names rank *saturation*. Both readings come off the same instrument. Noted as a second thing
  the existing diagnostic may already be able to see.
- **Identifiability, and `H¹` doing work the design has never assigned it.** §3.3 noted that the
  local-module-identification literature carries identifiability conditions Patchworks' rule has
  never been checked against, and left it. Anwer, Riess & Hale now state the condition **in sheaf
  cohomology**: *"recovery is determined by the sheaf cohomology of the system: when `H¹(G;F) = 0`,
  the local law is uniquely recoverable from trajectory data; when `H¹(G;F) ≠ 0`, recovery requires
  both a restricted parametric class and sufficiently diverse data."* And the sting: *"accurate node
  rollout does not certify recovery of the underlying local law — prediction and system
  identification are distinct objectives, and a learned model is interpretable only when the
  identifiability conditions are met."* CONTEXT.md gives `H⁰` a load-bearing role (private features)
  and says nothing about `H¹`. Patchworks does not claim to recover a true local law, so this is not
  a defect — but any future claim that a cell's learned `K` *means* something, as opposed to
  predicting well, now has a named precondition on the sheaf's first cohomology. **Flag only;
  pursuing it is a separate ticket.**
- **Nothing here threatens ADR-0023, §1's finding, or #166.** This pass touched none of those
  arguments.

## 7.6 The residue, restated

**It changed. It did not close.**

The move is that three of the four properties are now claimed **in a single source** for the first
time, where §3.5 had them claimed only across separate ones. Sheaf-ADMM holds a learned per-agent
linear operator, learned restriction maps into communication lanes, and updates that are local by
construction, all at once. What it cannot supply — and what the whole surrounding literature is
built *not* to supply, because a proved global guarantee is the point of that literature — is the
absence of a global objective and the presence of a state that persists in world-time.

So §3.5's instruction to read ADR-0023's sentence as *"resting entirely on the local rule"* needs
tightening: the weight rests on **local rule read strictly as *no global objective anywhere above
it***, together with **world-time**. "Local rule" alone is now ambiguous, because Sheaf-ADMM's rule
is local in every mechanical sense and still descends on a stated global functional.

**New residue, in one sentence, in §3.5's form:**

> A per-cell **learned** linear operator, glued by **learned** restriction maps into communication
> lanes, advanced by a rule that is local **with no global objective anywhere above it** — neither an inner
> problem it provably solves nor an outer loss backpropagated through it — carrying a state that
> **persists in world-time** rather than relaxing to a fixed point within one input: **not found in
> any single source**.

**Updated last row of §3.5's table**, with one row inserted above it to record what Sheaf-ADMM now
claims:

| Property | Status |
|---|---|
| Per-node **learned** operator **and** learned restriction maps into communication lanes **and** updates local by construction — all three in one source | **Claimed** — Sheaf-ADMM (§7.2), but only ever as an unrolled solver for a stated global objective, trained by backpropagation, with state re-initialised to zero on every input |
| Per-node **learned** operator, glued by learned restriction maps, under a local rule with **no global objective above it**, with a **persisting state in world-time** | **Not found in any single source** — Sheaf-ADMM has two global objectives and no world-time; ST-Sheaf GNN has neither world-time state nor a local rule; the sheaf-ADMM neighbourhood (§7.4) is uniformly locality *in service of* a global functional |

The conjunction stands, one property narrower and considerably more precisely stated. Per #173, a
pass returning *"this is known"* would have been a good outcome; it is not the outcome, and the
reason it is not is specific and quotable rather than a failure to reach the right paper.

## 7.7 Reading-depth ledger for this pass

- **Seely, Cupiał & Jones, *Learning Multi-Agent Coordination via Sheaf-ADMM*, arXiv:2605.31005v1** —
  **[FULL]** (arXiv HTML v1). Upgrades §3.2's **[CITE]**.
- **Mostafa, Younis & Ahmadi, ST-Sheaf GNN, arXiv:2604.11275** — **[FULL]** (arXiv HTML v1).
  Upgrades §3.2's **[ABS]**.
- **Anwer, Riess & Hale, arXiv:2605.11204** — **[FULL]** (arXiv HTML v1). New to the record.
- **Hanks et al., CDC 2025 / arXiv:2504.02049** — **[CITE]**. Bibliographic record confirmed and
  characterised from Sheaf-ADMM's own description of it; body not reached.
- **Zhao et al., arXiv:2510.00270** — **[ABS]**. Abstract-level only; body not reached.
- **Hanks et al., arXiv:2512.24886** — **[ABS]**. Abstract-level only; body not reached.
- **arXiv:2606.01663** — **[UNREACHED]**. Surfaced in search; neither abstract nor body read. Nothing
  above depends on it.

**What could not be reached in this pass**, stated plainly rather than papered over: the bodies of
Hanks et al. (CDC 2025, arXiv:2504.02049), Zhao et al. (arXiv:2510.00270), Hanks et al.
(arXiv:2512.24886), and anything at all of arXiv:2606.01663. §7.4's characterisation of the first
rests on Sheaf-ADMM's description of it, not on the original. Since §7.4 is one-hop context rather
than load-bearing evidence, and since the residue is decided by §7.2 and §7.3 — both **[FULL]** —
nothing in §7.6 depends on an unread body.
