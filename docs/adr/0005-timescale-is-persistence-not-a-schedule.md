# ADR-0005: Timescale is persistence in the private features, not a schedule

**Status:** superseded by
[ADR-0028](./0028-a-cell-holds-a-spectrum-of-retention-constants.md)

> **Superseded by [#143](https://github.com/NGL321/patchworks/issues/143), written as ADR-0028 by
> [#227](https://github.com/NGL321/patchworks/issues/227). Superseded, not edited, and left standing
> in full.** The reasoning that got from *not a schedule* to *persistence* is worth keeping legible;
> amending it in place would overwrite the argument and keep only its conclusion. Read this ADR for
> how the choice was reached and ADR-0028 for what is in force.
>
> **What changed is the source of the persistence, and nothing above it.** *A cell is slow because its
> content persists, not because it updates rarely* survives intact, as does the uniform execution
> clock, the runtime prohibition on reading a timescale, the location of slow content in `ker δ`, and
> the clock divisor as instrument and fallback. What is retired is the **mechanism** underneath:
> persistence supplied by *bias translation*, read off the regional spectra of a frozen
> piecewise-linear map, with a cell's rate the central tendency of the distribution its biases select
> and the spread placed by level at construction.
>
> **In its place:** a cell's retention lives in the spectrum of its own learned `K`, so a cell has a
> **spectrum** of retention constants rather than one rate; `a` stays global and the depth gradient is
> **learning's job**, placed nowhere, with *nothing guarantees the gradient appears* as ADR-0028's
> pre-registered falsification. Two amendments recorded below are consequently spent: the #41
> distributional reading of the regional spectrum, and the construction-time placement of `τ` by level
> in overlapping bands. **Region dwell is demoted rather than dropped** — it gated whether `τ` was
> *well-defined*, and under `λ(K)` it gates only how *faithfully* the operator's rate is realised.
> **The `dwell > τ` bar survives that demotion and its rank changes**
> ([#226](https://github.com/NGL321/patchworks/issues/226)): it is the **licence for the cheap
> spectral instrument**, not a bar on the architecture, and **this ADR's falsification clause is
> retired** with it. See the amendment under *Consequences*.
>
> The #138 amendment below is what lifted the foreclosure this ADR rejected per-cell time constants
> on. It named no successor, deliberately. ADR-0028 is that successor.

## Context

The proof-of-concept's central claim is compositional planning across timescales, and the acceptance
demo requires a slow plan to survive a fast correction. Nothing in the architecture supplied more
than one timescale.

Four mechanisms were available. Depth was already rejected in `01-cell-and-sheaf.md` ("depth buys
horizon, not rate") but without an argument; the argument is that **unit delay is a phase shift, not
a decimation** — it removes no frequency content, so every cell has the same input bandwidth
regardless of hop distance, and depth's genuine ratio is in *latency* rather than in *commitment*.
Per-cell time constants appeared to be foreclosed by the shared frozen body of
[ADR-0001](./0001-continual-learning-applies-to-the-adapting-surface.md).

> **Amended by [#138](https://github.com/NGL321/patchworks/issues/138): that foreclosure is lifted.**
> The Koopman conversion replaced the frozen nonlinear `step` with `K`, a per-cell learned linear
> operator — and **a per-cell `K` *is* a per-cell time constant**, `k` of them. The premise on which
> this ADR's four-way choice rejected one of its options no longer holds, so the choice is **reopened**.
>
> This ADR is **amended rather than superseded**, and the distinction is deliberate. Dropping it would
> leave the proof of concept's central claim — compositional planning across timescales — with no
> mechanism at all for the whole of the next stage. What is recorded here is that the foreclosure is
> lifted and the replacement is **open**; no successor mechanism is named, because reading timescale
> off `K`'s spectrum is a separate question with real difficulties of its own and is deliberately not
> front-run.
>
> What is *not* affected: the runtime prohibition. The placement still happens once, at construction,
> and still leaves no rate for a running cell to consult. Activation regions and fold margins also
> survive, because `encode` is still ReLU — the mechanism this ADR built is described in
> `05-timescales.md` as built, with the conversion's consequences marked around it.

One divergence is recorded here because a reader who checks the source will find it. Active
predictive coding's `T1`/`T2` — the source of the goal this decision serves — *is* a fixed clock
divisor, and in APC it is the **mechanism**, not an instrument. This decision takes APC's goal,
declines its mechanism, and rebuilds that mechanism as the instrument it measures against.

## Decision

**A cell is slow because its content persists, not because it updates rarely.** Every cell runs
every tick; the tick contract of `02-tick-semantics.md` is untouched. Persistence has two halves,
each already committed to for other reasons:

- **Insulation.** Reconciliation moves a node stalk along `im δᵀ`, and the private features are
  `ker δ = (im δᵀ)^⊥`. The private component is therefore exactly invariant under reconciliation.
  Slow content lives there. This is a commitment, not an observation about where learning might put
  things.
- **Decay rate.** The shared frozen body is one piecewise-linear map with per-cell fold offsets
  ([ADR-0004](./0004-linear-restriction-maps-assume-local-flatness.md)'s companion finding), so each
  cell occupies a different activation region with its own local Jacobian.

  *Amended by [#41](https://github.com/NGL321/patchworks/issues/41), correcting the sentence that
  stood here:* the **regional spectrum is a per-tick quantity**, re-drawn as the cell's chart carries
  it across folds, and [#27](https://github.com/NGL321/patchworks/issues/27) measured the operating
  point contributing as much spread in `τ` as the biases do (7.3× against 7.7×). What the biases
  select is therefore the **distribution** those draws come from; a cell's effective timescale is
  that distribution's central tendency — a **mean rate, not a fixed rate**. The mechanism is
  unaffected: a mean rate is still a rate, and the across-cell spread is real. It holds under one
  condition, now stated in `05-timescales.md`: a cell's **region dwell** must be long against the
  `τ` its region implies. The **fold margin** was the construction-time proxy for it; since
  [#160](https://github.com/NGL321/patchworks/issues/160) dwell is **measured on the run**
  ([ADR-0019](./0019-construction-nominates-the-run-decides.md)), and the margin is the proxy that
  nominates rather than the quantity that decides.

  *Its premise, recorded by [#49](https://github.com/NGL321/patchworks/issues/49):* all of this
  presupposes a **piecewise-linear** body. Activation regions, folds, the regional spectrum and dwell
  are objects only a piecewise-linear network has — under a smooth activation the Jacobian varies
  continuously and none of them has a referent. The commitment (ReLU, and why the class rather than the
  instance is what matters) is in `01-cell-and-sheaf.md`, *The body's construction*; it is noted here
  because this is the decision that would fall if it were ever swapped away.

**Nothing in the architecture reads a cell's timescale.** It is observable from outside only, never
an input to any computation and never a selection criterion. Under the amendment above this is
**structural rather than disciplinary** — a per-tick draw is not a value anything could branch on.

*Sharpened by [#42](https://github.com/NGL321/patchworks/issues/42):* the prohibition is about
**runtime**, and needs saying that way, because the body's construction now *does* select on
timescale. Cells are built by drawing candidate bias vectors, measuring the timescale each produces,
and keeping a set that covers a target band (`05-timescales.md`, *What this requires elsewhere*).
That is a construction-time criterion applied once, from outside, leaving no stored rate behind; no
cell, edge or rule can consult a timescale while the graph is running, which is the property this
decision protects and the property the clock divisor has to be interchangeable with.

**An explicit clock divisor is built first, as an instrument** — the rig that establishes the
capability depends on timescale at all — and thereby becomes an already-validated fallback.

## Consequences

- Slow-state capacity becomes a **construction quantity**, `dim H⁰ ≥ Σ_v max(0, n − Σ_{e∋v} m_e)`,
  so graph topology gains two hard constraints: abstract cells need bounded total mask width, and
  wide integration may have to be a separate cell from holding.
- The body must be **constructed** for spread in its regional spectra, not assumed to have it.
  [#27](https://github.com/NGL321/patchworks/issues/27) answered *constructible but coupled* — the
  spread is available and narrowness supplies it, but the same global knob positions the distribution
  against the stability boundary. [#42](https://github.com/NGL321/patchworks/issues/42) resolved the
  coupling: **`σ_w²` is set for containment only, and the spread is imposed by selecting bias vectors
  rather than drawing them**, then assigned to levels in overlapping bands whose range is derived
  from the demo's perturbation horizons. Spread and stability were never two knobs — both are
  functions of region dwell — and the stability object is the trajectory's realised contraction `λ`,
  with `max ρ < 1` demoted to a cheap sufficient check available before training. The cost recorded
  with it: the depth↔timescale correspondence is now built rather than found, so only the
  *behavioural* claim remains falsifiable.
- **`02-tick-semantics.md`'s `gain_v × offset <` fold margin bound is now a precondition of this
  decision,** not only a reconciliation-stability check: the margin is what makes a cell's region —
  and therefore its timescale — a well-defined object. *Amended by
  [#160](https://github.com/NGL321/patchworks/issues/160)* on both halves of what stood here. The
  precondition **re-sources onto measured dwell**: the margin bounds dwell and the bound is checked on
  the run, so the well-definedness this decision needs is read rather than nominated
  ([ADR-0019](./0019-construction-nominates-the-run-decides.md)). And *it binds hardest at the apex,
  where the slow cells live* is **struck** — [#190](https://github.com/NGL321/patchworks/issues/190)
  made `gain_v` uniform across the interior, and nothing replaces the claim.
- **The go/no-go run's passing criterion is specified rather than left to the rig** (`05-timescales.md`):
  reachability of the target band as an acceptance rate, measured over a driven trajectory with the
  operating point varying, with dwell reported alongside and decay reported as realised `λ` rather
  than an eigenvalue. Per #42 the first arm is *reachability*, not spread — spread is constructed
  now, so it can no longer falsify anything; a body that cannot reach the slow band still can.

  *Amended by [#208](https://github.com/NGL321/patchworks/issues/208), which gave the dwell
  precondition a bar and named what failing it kills.* The bar is **one e-fold of the region's own
  decay expressed within the residency** — `dwell > τ`, read on the **median cell**, with dwell stated
  as the **cumulative mean residency to the horizon** and the per-cell count below the floor reported
  rather than asserted. `dwell ≥ 2.6 τ` is demoted to reported headroom; it was
  `DEFAULT_SAFETY_FACTOR` transplanted onto a duration it was never derived for. `05-timescales.md`
  carries the derivation.

  *Amended by [#226](https://github.com/NGL321/patchworks/issues/226) on the referent, and on what
  the bar is a bar on.* The bar stays, with the same derived `1`, and two things about it change.

  **The referent is re-pointed and the number is unchanged.** `dwell > τ` is one e-fold of **the
  operator's own retention** rather than of the region's: `exp(−D/τ)` is the fraction of the cell's
  retained content that decays while the observables hold still. The arithmetic never depended on `τ`
  being the region's, and it is cleaner under `λ(K)`, not weaker. Two things are now stated that
  previously read as open: `τ` is **the slowest of the cell's twelve retention constants** — already
  what `read.py` computes, `eigvals(...).abs().amax()` — read off the **full loop**
  `ρ(K · (J_chart + J_stalk · A_v · D))` and not the chart half
  ([#271](https://github.com/NGL321/patchworks/issues/271),
  [#274](https://github.com/NGL321/patchworks/issues/274)).

  **`dwell > τ` is a validity condition on the spectral instrument, not an architectural bar.** There
  are two instruments for retention: the cheap `τ = −1/ln ρ`, available at construction, and
  [#242](https://github.com/NGL321/patchworks/issues/242)'s measured `τ̂`, available only from a run.
  `encode` is piecewise linear, so realised retention over `N` ticks is a product of `N` operators
  `K · J_encode(region_t)`; the cheap reading stands in for the expensive one only while that operator
  holds still, and dwell is how long it holds still. So the bar is **published wherever the spectral
  `τ` is published**, and is **reported, never asserted**, in #206's language. It composes with #242
  as `world_loop(c) ≤ τ_c < dwell_c` — **one architectural bar (#242, on `τ̂`) plus the licence for
  the proxy**, not two bars — whose `τ`-free consequence, `dwell_c > world_loop(c)`, is what is
  readable today. *The floor was `|loop(c)|` until
  [#404](https://github.com/NGL321/patchworks/issues/404):* the lower end **is** #242's bar,
  [#383](https://github.com/NGL321/patchworks/issues/383) moved that bar to `world_loop(c)`, and this
  composition moved with it — on the strength of the *one bar plus a licence* sentence above, which
  leaves the floor nowhere else to stand. `|loop(c)|` is kept and demoted, and the consequence is a
  **harder** comparison than it was, by 1 to 7 ticks at every cell.

  ~~**A collapse of median `dwell/τ` below 1 falsifies the claim that placing biases is *sufficient*
  to buy a timescale.**~~ ***Retired by #226. This ADR's falsification clause is withdrawn, not
  re-pointed.*** Two independent reasons. **Placement stopped being the mechanism** at
  [#143](https://github.com/NGL321/patchworks/issues/143), which moved retention onto `λ(K)`, and
  [#276](https://github.com/NGL321/patchworks/issues/276) found `select()` never ran in any Sheaf that
  ticks — so there was no sufficiency-of-placement claim left for a dwell reading to falsify. And more
  fundamentally: **no dwell reading falsifies *timescale is persistence, not a schedule*.** What would
  is retention achieved and conduction still absent — `τ̂` raised to the loop and the rim still not
  reaching the core — and that experiment lives on #242's measured quantity, in
  [ADR-0026](./0026-rim-core-influence-is-a-conduction-ratio.md), not here. A breach invalidates the
  **instrument**, not the design.

  **What survives the retirement is the observation, and it is filed as a problem rather than as a
  falsifier.** `τ` and dwell are measurably independent — `corr(log τ, log dwell) = −0.110` in the
  live run against #42's construction-time `corr(log ρ, log margin) = −0.006` — so the mechanism has
  an **unenforced precondition**: `05-timescales.md` books the decoupling as a win, having written
  only the favourable direction, while the other direction is that **selecting a cell slow buys it no
  residency to be slow in**. That is [#344](https://github.com/NGL321/patchworks/issues/344). It is
  **live today rather than latent**: on the corrected operator, with dwell and `τ` read **off the
  same run** ([#361](https://github.com/NGL321/patchworks/issues/361), seed 42 at 100,000 ticks), the
  median `dwell/τ` is **3.923** and **19 of 150** cells fail the licence, where the chart-only
  reading published 9.49 and 3. *The 2.00 and 57 of 150 that stood here were #226's mismatched
  pairing — seed-42 dwell against another run's `τ` — and were pessimistic by about 2x; #361
  supersedes them rather than amending them.* Across nine seeds at 30,000 ticks the median runs
  **2.165 to 10.984** and **every seed clears the bar at the median**, so the licence is breached at
  cells rather than in the aggregate. Individual cells below the floor are a placement finding owned
  by [#205](https://github.com/NGL321/patchworks/issues/205).
- **The biases become over-subscribed** — three geometrically distinct jobs on one per-cell vector,
  the third being to preserve private directions through a frozen `encode`. First concrete argument
  for pulling per-cell adapters off the flex ladder early.
- **Some disagreement is now irreducible by design**, which reconciliation, the local learning rule,
  and ADR-0004's falsification signature all assume it is not
  ([#28](https://github.com/NGL321/patchworks/issues/28)).
- The prohibition on reading timescale is what keeps the divisor and the persistence mechanism
  interchangeable. Reversing it costs the cheap fallback.

## Alternatives considered

- **Emergence from depth.** Rejected on the phase-versus-frequency argument above. The secondary
  dismissal that stood here — its slow modes are unsteerable `~2L` artifacts, an order of magnitude
  short — was wrong, and is corrected in `05-timescales.md`: delay-coupled loops reach second-scale
  rhythms, in relative phase, steerable by coupling gain. The route is closed here because that gain
  is `γ`, which ADR-0007 fixed for stability and recorded as explicitly not a timescale knob. *The
  clause that stood here — and which the fold margin binds — is struck by #160:* the margin bounds
  the standing offset, not `γ`, and `γ` is held at 1.0 by ADR-0019 rather than by any margin. The
  route stays closed on the first ground, which never depended on the second.
- **An explicit per-cell clock divisor as the answer.** Retained as an instrument and fallback, not
  as the mechanism: `k` is arbitrary, it aliases, and it hard-codes hierarchy at construction time in
  a spec that has worked to make hierarchy emergent.
- **Termination-driven abstract steps** — a cell holding until the disagreement it asserts is
  cleared. The most faithful to the active-predictive-coding framing and the most expensive: it needs
  a termination signal, which is new mechanism, and it breaks the synchronous single-step reconciliation
  of [ADR-0002](./0002-message-passing-is-one-step-not-a-solve.md). Held as the named escape hatch.
- **Drive dwell alone** — treating a held goal as the slow variable. Rejected because it would pass
  the acceptance demo with a *human* supplying the abstraction, and because it would require the agent
  to assert drives on itself for agent-originated plans. Under this decision, persistence *is* holding,
  so no self-assertion mechanism is needed.

  *Amended by [ADR-0009](./0009-a-drive-is-a-motor-edge-attached-deep.md):* this alternative was
  written when a goal was thought to be a **clamp** on an ordinary cell. It is now a
  [drive boundary cell](../spec/04-action-and-the-boundary.md) writing across ordinary edges. The
  rejection is unaffected — a human-held drive still supplies the abstraction the demo is supposed to
  show — and the wording is updated only because "clamp" is retired.
- **A probabilistic sheaf**, so that low-confidence messages fail to pass. Rejected as the tail
  wagging the dog: it would change what a communication lane is, what restriction maps map, what
  disagreement means, and what `decode` outputs — a rewrite of the foundational cell contract — in
  service of a gate that needs only a magnitude threshold on a vector difference. The probabilistic
  sheaf stays in the map's fog on its own merits.

## What the literature does and does not give

No prior system obtains timescale separation from persistence alone: every architecture surveyed sets
it with a hand-set schedule, a learned discrete gate, or a dedicated per-unit rate parameter. This
decision takes none of them — the novelty claim, and the reason the divisor is built first. The
phase-shift argument and the commitment-versus-latency distinction are this project's own: stated
nowhere, contradicted nowhere. See `docs/research/029-timescale-citations.md`.
