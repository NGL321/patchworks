# ADR-0014: The chart is not a Koopman lift, and `K` is a linear recurrence

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

So `K` is a **linear recurrence driven by a nonlinear input map**, and the comparison class is deep
state-space models and linear RNNs — not EDMD. In that class 12 dimensions of persisting state per
unit is unremarkable, and `docs/research/027-regional-jacobian-spectra.md` already cites S4D at
source for exactly this kind of object.

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

### The escape holds only while the linear reading is declined.

Both the spectral certification cost (§5 of #148) and Liu–Ozay–Sontag's omega-limit-set obstruction
([#151](https://github.com/NGL321/patchworks/issues/151)) are stated over embeddings **into linear
systems**. `K . encode` is not one. Any future move to treat a cell as a linear system re-imports
both in full. That is one trade arriving from two directions, and it is the standing cost of this
ADR.

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
`K . encode`, exactly as this ADR's escape says. What survives is not about motor edges at all: whether
a cell's linear maps should be **conditioned on the evidence it is currently seeing**, which is fog on
[#127](https://github.com/NGL321/patchworks/issues/127) and is asked of the restriction maps and `K`
together.

### The transfer is of mechanism, not of results.

The state-space-model literature is about deep stacks trained by backprop on sequence benchmarks.
Patchworks is a shallow-per-cell sheaf trained by local rules. Linear recurrence, timescale as an
eigenvalue, and stability by norm bound transfer. **No result does.** Nobody has shown a spatially
glued set of locally-trained linear recurrences works, and that novelty is this project's either way.

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
piece *and* carry the cell's memory, and a linear recurrence's recoverable short-term memory is
bounded by its state dimension — the same bound `docs/research/032` already invokes when it caps one
cell's linearly recoverable memory at `n = 32` delay taps. #32's headroom argument covers the naming
job alone and says nothing about two jobs sharing one budget. That is a separate ticket, blocked on
transmission.

## What the literature does and does not give

Support here is **structural, and the conclusion is this project's own.**

- **[#148](https://github.com/NGL321/patchworks/issues/148) §4 is the evidence base**, read at source
  across four independent groups, and its verdict is adopted rather than disputed. What this ADR adds
  is the observation the pass could not make from outside the code: it priced `k` against EDMD
  because it was reading the design's own framing, and that framing was wrong.
- **S4D's geometrically-uniform timescale initialisation** is verified at source in
  `docs/research/027-regional-jacobian-spectra.md` and is the nearest published object to a
  per-unit-timescale linear recurrence.
- **Do not lean on "state-space models show this works."** The nearest supports — linear recurrence
  with the nonlinearity moved outside it, memory-capacity bounds for linear recurrences, and whether
  anyone has glued such recurrences spatially under a local rule — are **named from recall and not
  read at source at the time of writing**. A citation pass owes this ADR its reading-depth tags, and
  is filed as a research ticket rather than assumed.
