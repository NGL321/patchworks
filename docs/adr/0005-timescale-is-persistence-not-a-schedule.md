# ADR-0005: Timescale is persistence in the private features, not a schedule

**Status:** accepted

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
  `τ` its region implies, for which the **fold margin** is the construction-time proxy.

**Nothing in the architecture reads a cell's timescale.** It is observable from outside only, never
an input to any computation and never a selection criterion. Under the amendment above this is
**structural rather than disciplinary** — a per-tick draw is not a value anything could branch on.

**An explicit clock divisor is built first, as an instrument** — the rig that establishes the
capability depends on timescale at all — and thereby becomes an already-validated fallback.

## Consequences

- Slow-state capacity becomes a **construction quantity**, `dim H⁰ ≥ Σ_v max(0, n − Σ_{e∋v} m_e)`,
  so graph topology gains two hard constraints: abstract cells need bounded total mask width, and
  wide integration may have to be a separate cell from holding.
- The body must be **constructed** for spread in its regional spectra, not assumed to have it.
  [#27](https://github.com/NGL321/patchworks/issues/27) answered *constructible but coupled* — the
  spread is available and narrowness supplies it, but the same global knob positions the distribution
  against the stability boundary ([#42](https://github.com/NGL321/patchworks/issues/42)).
- **`02-tick-semantics.md`'s `γ × floor <` fold margin bound is now a precondition of this decision,**
  not only a reconciliation-stability check: the margin is what makes a cell's region — and therefore
  its timescale — a well-defined object. It binds hardest at the apex, where the slow cells live.
- **The go/no-go run's passing criterion is specified rather than left to the rig** (`05-timescales.md`):
  `τ` in quantiles, measured over a driven trajectory with the operating point varying, with dwell
  reported alongside and decay cross-checked against something other than an eigenvalue.
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
  is `γ`, which ADR-0007 fixed for stability and recorded as explicitly not a timescale knob, and
  which the fold margin binds.
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
  wagging the dog: it would change what an edge stalk is, what restriction maps map, what disagreement
  means, and what `decode` outputs — a rewrite of the foundational cell contract — in service of a
  gate that needs only a magnitude threshold on a vector difference. The probabilistic sheaf stays in
  the map's fog on its own merits.

## What the literature does and does not give

No prior system obtains timescale separation from persistence alone: every architecture surveyed sets
it with a hand-set schedule, a learned discrete gate, or a dedicated per-unit rate parameter. This
decision takes none of them — the novelty claim, and the reason the divisor is built first. The
phase-shift argument and the commitment-versus-latency distinction are this project's own: stated
nowhere, contradicted nowhere. See `docs/research/029-timescale-citations.md`.
