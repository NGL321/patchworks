# ADR-0023: The chart is not a Koopman lift, and `K` is a linear recurrence

**Status:** accepted

**Cross-references [ADR-0004](./0004-linear-restriction-maps-assume-local-flatness.md)**, which
defines `k` as the dimension of the piece. That definition is correct and unamended; this ADR records
that it is the *only* reading of `k` the design has, and what follows from there being no second one.

## Context

Raised by [#145](https://github.com/NGL321/patchworks/issues/145), which opened on a distinction it
named `k_piece` vs `k_lift`:

- **`k_piece`** — how many coordinates faithfully *name* the states of the piece. Defended by
  [#32](https://github.com/NGL321/patchworks/issues/32) and
  `docs/research/032-dimensioning-small-predictors.md` on the fractal generalisation of Takens, and
  not challenged.
- **`k_lift`** — how many observables are needed for the piece's dynamics to be *linear* in them.
  Never asked, because the question could not arise while `step` was nonlinear.

[#138](https://github.com/NGL321/patchworks/issues/138) made the question urgent by linearising
`decode` as well as `step`, putting the body's entire expressiveness in the 12-dimensional chart, and
the [#148](https://github.com/NGL321/patchworks/issues/148) citation pass priced it. Its §4 verdict:
*"`k = 12` is fine as `k_piece` and indefensible as `k_lift`, and the record currently uses it as
both."* Every published lift that works is **larger** than the state it lifts — 330 monomials for a
two-dimensional end-effector position (Bruder et al., RSS 2019); 60 to 100 basis functions for
two-dimensional oscillators (Colbrook & Coote). On that scale `k = 12` under `n = 32` is not a
marginal lift. It is a **compression**, and a compression is not a lift.

## Decision

**The design has no `k_lift`, because it has no lift. `k = 12` is `k_piece` and nothing else.**

The premise the collision rested on is false. `K` does not advance a dictionary of observables of an
instantaneous state. It advances a **persisting chart**:

```
chart_{t+1} = K · encode(chart_t, stalk_t)
```

`encode` is `R^k x R^n -> R^k` — *"Fuse the persisted chart with the node stalk into a single
chart"* (`src/patchworks/body.py`) — and `02-tick-semantics.md`'s advance phase already ran on
*"its own persisted chart and the node stalk"* before the conversion existed. Two properties follow,
and each independently puts the cell outside EDMD's hypotheses:

- **The chart is recurrent, not a function of the instantaneous state.** A Koopman dictionary is
  memoryless by construction; `psi(x)` depends on `x` alone. A chart that persists depends on the
  whole history that wrote it.
- **The cell is driven, not autonomous.** A new `stalk_t` arrives every tick from reconciliation.
  Koopman theory is about the evolution of observables under an autonomous flow.

So `K` is a **linear recurrence driven by a nonlinear input map**, and the comparison class is
**nonlinear RNNs and reservoir computing** — not EDMD, and not deep state-space models either.

> **Corrected by [#172](https://github.com/NGL321/patchworks/issues/172) on
> [#167](https://github.com/NGL321/patchworks/issues/167)'s citation pass.** This sentence originally
> named *"deep state-space models and linear RNNs"*, and sourced `k = 12` from stacking practice in
> that class. It was wrong, and the Consequences section below already said so. The correction is a
> repair to **what the design is compared to**, not to what it is: the Decision itself is untouched.

The SSM class is **defined by** placing the nonlinearity *between stacked layers*. `K · encode`
re-applies it **inside the loop, at every recurrent update**, which is the thing that class excludes.
Merrill, Petty & Sabharwal, *"The Illusion of State in State-Space Models"* (ICML 2024,
arXiv:2404.08819) **[FULL — body read]** draw the line in as many words — *"adding a nonlinearity to the output
of an SSM layer (as in Mamba) is not the same thing as an RNN-SSM. Rather, an RNN-SSM applies the
nonlinearity at each recurrent update"* — and their Theorem 5.1 makes it a strict expressivity gain
rather than a matter of taste: a **one-layer** RNN-SSM recognises any regular language, including the
`S_5` word problem that SSMs provably cannot express. **The design sits on the more expressive side
of that line**, and a one-layer construction is exactly the shape a shallow per-cell recurrence has.

`docs/research/167-linear-recurrence-citations.md` §1.4 places the cell on the matched shelf by a
change of variables rather than by analogy — its own derivation, marked as such. Substituting the
pre-activation `u_t = E_hid[chart_t; stalk_t] + b` into the tick gives

```
u_t = ( E_hid^chart · K · E_out ) · ReLU(u_{t-1}) + E_hid^stalk · stalk_t + b
```

an ordinary 45-unit recurrent network with random input weights, a random bias and a linear readout,
whose recurrent matrix `W_rec = E_hid^chart K E_out` has **rank at most 12** and whose only learned
degrees of freedom sit in the 12x12 factor `K`. That is an **echo state network with a low-rank,
partially-learned reservoir**, and `rank(W_rec) <= k` is the reason the chart — not the 45-wide
hidden layer — is the state that carries memory across ticks.

So the licence for `k = 12` comes from **reservoir dimensioning**, where a 12-dimensional recurrent
state is unremarkable and Dambre's capacity bound is stated on exactly `rank(W_rec)`. It does not
come from SSM stacking practice, which is a shelf this design is not on. The licence arrives with a
counterweight the SSM reading concealed, and it is carried here rather than left in the research
doc: §2.3's normal-matrix result (Ganguli, Huh & Sompolinsky, *PNAS* 2008 **[FULL — body read]**) gives memory
capacity **exactly 1** for a normal connectivity matrix, and `K = a·I` at construction is normal;
§6.2's recall-throughput frontier records that **width is the lever the field reaches for when
memory is short**, which is the move [#14](https://github.com/NGL321/patchworks/issues/14)'s
constraint ladder puts last. Both bear on the chart's double duty and neither reaches this Decision;
they are [#166](https://github.com/NGL321/patchworks/issues/166)'s to price.

**What the cell learns is the dynamics of its own belief under evidence, not the dynamics of the
world's state.** That sentence is the decision, stated in the vocabulary of the thing rather than of
the thing it is not.

## Consequences

### The dynamics problem is not linearised. One factor of one tick is.

`K . encode` is a **nonlinear** recurrence, because `encode` re-applies nonlinearity every tick
between every application of `K`. The design does not claim, and must not claim, that a piece's
dynamics are linear in the chart's coordinates. What is linear is the **advance of an already-fused
chart**, and what that buys is per-tick tractability, a settable `sigma_max(K)`, and readable
retention constants — not a linearisation of the control problem.

### Linearity was never bought with width. It is paid for in nonlinearity-in-the-loop and in time.

The trade the record implied — linearity purchased by dimensional increase, the cost then amortised
across a sheaf of small pieces — is not the trade that was made. There is no dimensional increase.
The price is `encode` running every tick, the chart persisting, and unit delay on every edge.
[#148](https://github.com/NGL321/patchworks/issues/148) §4 recorded this as an escape route to buy —
Mezic's Krylov/Hankel observables, which *"do not suffer from the curse of dimensionality… the
dynamics selects the basis by itself"* — and called it *"the cheapest fix available to this
architecture specifically."* It is not a fix. It is what the architecture already does.

### The decomposition keeps its own justification.

The sheaf over a graph of small pieces is **not** amortising a lift's dimensional cost, and never
was. ADR-0004 justified the decomposition on locality and local flatness — *"the world is not claimed
to be a manifold and does not need to be; the **pieces** are"* — before the conversion existed. The
structure does not wobble when the Koopman reading goes, because it never stood on it.
[#148](https://github.com/NGL321/patchworks/issues/148) §1.4 and §2 found that structure is where the
design's unclaimed novelty actually lives.

### Three things are given up, deliberately and permanently.

1. **Modes stop being *of the world*.** No eigenvalue of `K` may be read as a mode of the piece's
   physics. A slow eigenvalue says the cell **chose to retain that direction for that long** — a
   property of its memory policy. Any claim of the form *"the architecture discovered the system's
   frequency"* is closed to this design.
2. **No error bound tied to invariance.** Nothing bounds prediction error by how nearly `im(D)` is
   invariant under the true dynamics. No such bound was ever in reach — it needs dictionary
   completeness and unbounded data, and this is a 12-wide chart under a local rule — but the
   aspiration is now formally surrendered.
3. **The word, unqualified.** This cannot be called a Koopman operator method without immediately
   caveating that its "lift" compresses. Naming it a linear recurrence up front is cheaper than
   defending that.

[#138](https://github.com/NGL321/patchworks/issues/138) anticipated all three: it struck the *"EDMD
with a fixed dictionary"* framing and recorded linearity as taken for tractability and for
`rho(K)`'s meaning, **explicitly not** to buy the Koopman literature's prior work. This ADR makes
that consistent rather than aspirational.

### What survives, because it was never Koopman-dependent.

- **The fitting.** Regression of a linear map onto observed transitions is linear algebra. The
  Koopman reading only supplied an interpretation of it, and
  [#139](https://github.com/NGL321/patchworks/issues/139) had already made this a local prediction
  rule rather than batch EDMD.
- **The stability band.** [#140](https://github.com/NGL321/patchworks/issues/140) put it on
  `sigma_max(K)`, a **norm**. Non-expansiveness is an exact fact about a matrix.
  [#148](https://github.com/NGL321/patchworks/issues/148) §5's attack — impossibility results,
  spectral pollution as the default, the cost of certification — is aimed at *spectra*, and does not
  reach a norm bound.
- **`rho(K)` as the transmission knob.** [#142](https://github.com/NGL321/patchworks/issues/142)'s
  budget is about Jacobian magnitude and is indifferent to interpretation.
- **`docs/research/032`'s defence of `k = 12`.** Takens / Sauer–Yorke–Casdagli defends the chart as
  an injective *naming of states from delay coordinates*. A persisting chart **is** a delay
  embedding, so the reframe strengthens that argument rather than weakening it.
- **The spectral read-out for timescale.** Eigenvalues of `K` remain readable as
  `tau = -1/ln|lambda|` — as **retention constants of a designed recurrence**, which is what they are
  by construction. Nobody asks whether a state-space model's `lambda` converges to the true spectrum
  of anything. This is handed to [#143](https://github.com/NGL321/patchworks/issues/143), whose
  independent caveat is unaffected: `encode` is interleaved, so `lambda(K)` is the operator's
  *contribution* to realised timescale, not the realised timescale.

### The escape holds only while no cell runs without evidence.

*Repaired by [#151](https://github.com/NGL321/patchworks/issues/151), which read the discrete-time
extension at source. The verdict is unchanged; the reason given for it was wrong, and a wrong safety
argument is worse than none.*

**Struck**: *“Both the spectral certification cost and Liu–Ozay–Sontag's omega-limit-set obstruction
are stated over embeddings **into linear systems**. `K . encode` is not one.”* The discrete-time
extension — Ristich, Sontag & Ozay, *On the Nonexistence of Continuous Immersions for Discrete-time
Systems*, arXiv:2605.15161v2 — walks straight through that. Its **Corollary 4** needs only a target
that is **continuous with closed basins**; Lemma 3 supplies linear systems as an *instance*, and
Corollary 2 says so in its own parenthesis, *“(e.g., a linear system)”*. Linearity is an example, not
a hypothesis. As written the clause was a **trapdoor**: checked once by a future reader against a
hypothesis that was never the binding one, and passed.

**The escape rests on two hypotheses this ADR has already established, one section above** — under
*The chart is recurrent* and *The cell is driven*:

- **No cell is autonomous.** The theorem's object is `x_{k+1} = f(x_k)`, wound up and left alone, and
  that is *how* omega-limit sets come to exist at all. A fresh node stalk arrives from reconciliation
  every tick and `encode` fuses it before `K` runs; `body.py` composes the two in a single call and
  there is no code path that iterates `K` without evidence. There is no `f: X -> X` here to have
  limit sets.
- **No semiconjugacy is claimed.** Definition 7 fixes an immersion as a mapping `F: X -> Z` with
  `F . f = g . F` — a *function of state*, and nothing weaker. The chart persists, so it is not a
  function of the instantaneous state and no such `F` exists. The only thing one would have bought is
  already surrendered above: *“No error bound tied to invariance.”*

**The standing cost survives verbatim, and is now a trigger rather than a test.** Any future move
that lets a cell **run without evidence** re-imports both obstructions in full — that is one trade
arriving from two directions, and it is the standing cost of this ADR. A reader can recognise that
condition without holding the proof, which the linearity test never was.
[#144](https://github.com/NGL321/patchworks/issues/144), which reads disagreement at horizon `h` as
`K^h z` with no `encode` between steps, is **the only place in the design that currently proposes
one**; the constraint on how its output may be read is pre-registered in its own body.

**A third door, recorded and not leaned on.** The paper's **Assumption 1** requires `X` to be path
connected — which the authors themselves flag as *“more restrictive in discrete-time dynamical systems
than continuous-time systems”*, because non-constant discrete trajectories have disconnected images.
Nobody here has checked it for a piece. It is written down so it is not rediscovered as an escape,
and it stays shut and unused because the first two suffice.

**Invoked once, and discharged.** [#146](https://github.com/NGL321/patchworks/issues/146) asked
whether cells adjacent to a motor edge need a **bilinear** operator, `z' = A z + (B + sum_i u_i C_i) z`,
on the ground that such a cell models a *driven* system where Koopman theory wants an autonomous one.
The premise does not survive this ADR: **no cell here is autonomous.** A fresh node stalk arrives every
tick from reconciliation, so driven-ness is universal and sorts nothing. Nor does closed-loop
self-causation sort cells — every cell's prediction reaches the world through the motor rim and
returns, differing only in latency, and every cell causes its own next input directly through the
persisting chart at a latency of one tick, which is *shorter* than a motor-adjacent cell's two-tick
loop out through the world and back by efference copy. **There is therefore no boundary at which a
linear cell would give way to a bilinear one**, and the question is all cells or none. Bruder et al.
(2021, arXiv:2010.09961) is stated over Koopman lifts of control-affine systems and does not reach
`K . encode`, which is a lift of nothing — the escape's second hypothesis, above. What survives is
not about motor edges at all: whether a cell's linear maps should be **conditioned on the evidence it is currently seeing**, which is fog on
[#127](https://github.com/NGL321/patchworks/issues/127) and is asked of the restriction maps and `K`
together.

*Footnote, from [#167](https://github.com/NGL321/patchworks/issues/167) §1.4, recorded and not
taken.* The classical control name for a linear block closed around a static nonlinearity is a
**Lur'e system**, `x_{k+1} = A x_k + B φ(C x_k)`, of which a cell is the degenerate `A = 0` case.
Absolute stability theory — Popov, the circle criterion, Zames–Falb multipliers — is stated for
exactly this structure, and so offers a certificate family for `K · encode` **as a whole**, where
[ADR-0015](./0015-the-cell-operator-band-is-on-the-spectral-norm.md)'s band certifies only `K`. It
is a route this design does not currently use, and nothing here takes it.

### The transfer is of mechanism, not of results — with one exception, and it is the unfavourable one.

The state-space-model literature is about deep stacks trained by backprop on sequence benchmarks.
Patchworks is a shallow-per-cell sheaf trained by local rules. Linear recurrence, timescale as an
eigenvalue, and stability by norm bound transfer.

**No *expressivity* result does**, and
[#167](https://github.com/NGL321/patchworks/issues/167) §4 confirms that half at source: LRU's
evidence is six layers on one benchmark by its authors' own framing, the universality theorem needs
MLPs interleaved between recurrent layers, and Merrill et al.'s `TC^0` result is about layer counts
and precision. None of them says anything about one shallow cell trained by a local prediction rule.

> **Amended by [#172](https://github.com/NGL321/patchworks/issues/172): the original *"No result
> does"* was too strong, and the exception is the expensive one.** The **capacity** results transfer
> **completely**. Jaeger, Dambre et al., Ganguli et al. and Gonon et al. are not statements about
> training procedures; they are statements about how much history a state of a given dimension can
> hold, given a readout class and an input process. Dambre states his for *arbitrary dynamical
> systems*; reservoir computing is the shallow, non-backprop case these results were born in, and by
> §1.4's change of variables a cell is closer to Jaeger's object than to Orvieto's; and
> `rank(W_rec) <= 12` is a construction constant that no choice of local or global training moves.
> The boundary this section draws is in the right place. Its label needed the exception, and the
> results that survive into this setting are the ones that **bound** the design, while the ones that
> would have licensed it are the ones that need the stack.

Nobody has shown a spatially glued set of locally-trained linear recurrences works, and that novelty
is this project's either way.

## Falsification, pre-registered

`k_piece = 12` is defended here by argument, because the measurement
[#145](https://github.com/NGL321/patchworks/issues/145) was written to demand — a sweep against a
*predicting* cell's stalk — cannot bear on a structural claim about what the code computes. What
would overturn it is stated in advance, in two tiers, both readable once
[#155](https://github.com/NGL321/patchworks/issues/155) makes a graph transmit:

1. **Cheap early read: rank saturation.** Take the numerical rank of each learned `K` across the
   population. If they saturate at 12, the width is binding.
2. **The real signature: error that scales with history, not with disagreement.** Per-cell prediction
   error bounded away from zero, scaling with the cell's evidence history length rather than with its
   disagreement, on cells whose piece is **not** under-charted by
   [#32](https://github.com/NGL321/patchworks/issues/32)'s box-dimension criterion. That is error a
   wider `k` fixes and more training does not.

Both point at the same suspect: the chart's **double duty**. Twelve dimensions must now name the
piece *and* carry the cell's memory — and those are **one budget with a trade-off inside it**, not
two budgets that happen to share twelve numbers. Dambre, Verstraeten, Schrauwen & Massar,
*Information Processing Capacity of Dynamical Systems*, *Sci. Rep.* **2**, 514 (2012), Theorem 4:
*"the sum of the capacities for these functions is bounded by the number N of output functions"*,
carrying a *"universal trade-off between the non-linearity of the computation and the system's
short-term memory"* — read at source by
[#167](https://github.com/NGL321/patchworks/issues/167) §2.2. #32's headroom argument covers the
naming job alone and says nothing about how the one budget is split. That was a separate ticket,
[#166](https://github.com/NGL321/patchworks/issues/166).

> **Resolved by [#166](https://github.com/NGL321/patchworks/issues/166); the pre-registration above
> is history and is annotated here, not rewritten.** The paragraph originally read the duty as two
> jobs with **independent** budgets and sourced the memory bound from recall rather than at source;
> the framing above is the corrected one.
>
> **Tier 1 cannot fire, and no later session should re-run it.** `CellOperators.__init__` builds
> `K = a·I` — rank 12 — and `project()` only rescales, one common factor across every singular value,
> which can send none of them to zero. **Numerical rank 12 is the null construction hands out for
> free**, not the signal the width is binding, and it measured 12.0 at 150 of 150 cells, at every
> checkpoint, on every seed. Rank *deficiency* is the only thing the statistic could ever have
> reported. **Tier 2 is deferred rather than struck**, on the ground that there is no ceiling to
> attribute. The ruling — the double duty is **not a contest for width** — and its rig, numbers and
> consequences are on [#166's resolution](https://github.com/NGL321/patchworks/issues/166#issuecomment-5520717326).

*Note added by [#172](https://github.com/NGL321/patchworks/issues/172).* Tier 1's rank read has a
companion the pre-registration did not know about:
[#167](https://github.com/NGL321/patchworks/issues/167) §2.3 gives memory capacity **1** for a normal
`K`, so **non-normality** — `‖K†K − KK†‖_F`, or departure from normality of the Schur form — is
readable at the same cost and answers the other half. Rank saturation says the width is spent;
non-normality says whether the width is *usable* as memory at all. This note records the instrument;
registering it belongs to the double-duty ticket
([#166](https://github.com/NGL321/patchworks/issues/166)), not to this pre-registration.

*The instrument was taken, and it carried the answer the rank read could not.* #166 measured a
population non-normality of ~0.05 against 0 at construction, which leaves `K` overwhelmingly normal
and its recoverable **sequence** memory at ~1 rather than 12 — Ganguli, Huh & Sompolinsky, *PNAS*
**105**(48):18970–18975 (2008), #167 §2.3. The memory shortfall is the operator's **shape**, not its
size, which is why no `k` fixes it. See [#166's resolution](https://github.com/NGL321/patchworks/issues/166#issuecomment-5520717326).

## What the literature does and does not give

Support here is **structural, and the conclusion is this project's own.**

- **[#148](https://github.com/NGL321/patchworks/issues/148) §4 is the evidence base**, read at source
  across four independent groups, and its verdict is adopted rather than disputed. What this ADR adds
  is the observation the pass could not make from outside the code: it priced `k` against EDMD
  because it was reading the design's own framing, and that framing was wrong.
- **S4D's geometrically-uniform timescale initialisation** is verified at source in
  `docs/research/027-regional-jacobian-spectra.md`, and
  [#167](https://github.com/NGL321/patchworks/issues/167) §5.1 confirms at source that S4D, LRU and
  Mamba all treat `λ` and `Δ` as **designed retention constants** rather than estimates of anything.
  It is cited for that *practice*, which is the design's own and transfers as mechanism — **not** as
  the comparison class, which is the error the Decision above corrects.
- **Do not lean on "state-space models show this works."** This was filed as a research ticket
  rather than assumed, and **the ticket has run**:
  [#167](https://github.com/NGL321/patchworks/issues/167), with reading-depth tags, in
  `docs/research/167-linear-recurrence-citations.md`. It returned the two corrections applied above —
  the comparison class, and the transfer exception — and answered the three named supports:

  - *Linear recurrence with the nonlinearity moved outside it* is a real class with a universality
    theorem (Orvieto et al., HLD 2023 **[ABS — abstract only]**) and an existence result (LRU, ICML 2023 **[FULL — body read]**),
    **and this design is not in it** — the reason the Decision moved.
  - *Memory-capacity bounds* exist, were read at source, and are **stronger and less comfortable**
    than the record assumed. They belong to
    [#166](https://github.com/NGL321/patchworks/issues/166), which the pass unblocks with four
    conditions on citing them.
  - *Has anyone glued such recurrences spatially?* Closer than assumed. Per-node linear recurrence
    with the nonlinearity outside, glued over edges, is in print and current (Message-Passing
    State-Space Models); so is sheaf learning with restriction maps under a backprop-free local rule
    (Bosca & Ghrist). What is **not** in any single source is the conjunction of all three — a
    learned per-cell operator, glued by learned restriction maps, trained by a local rule with no
    global objective. Every *pair* is claimed. The residue is one property wide, and the record
    should say so rather than claim more.
