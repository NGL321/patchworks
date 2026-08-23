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
  cell occupies a different activation region with its own local Jacobian. A cell's effective
  timescale is its region's spectral radius, selected by its biases.

**Nothing in the architecture reads a cell's timescale.** It is observable from outside only, never
an input to any computation and never a selection criterion.

**An explicit clock divisor is built first, as an instrument** — the rig that establishes the
capability depends on timescale at all — and thereby becomes an already-validated fallback.

## Consequences

- Slow-state capacity becomes a **construction quantity**, `dim H⁰ ≥ Σ_v max(0, n − Σ_{e∋v} m_e)`,
  so graph topology gains two hard constraints: abstract cells need bounded total mask width, and
  wide integration may have to be a separate cell from holding.
- The body must be **constructed** for spread in its regional spectra, not assumed to have it. Whether
  that is possible is open ([#27](https://github.com/NGL321/patchworks/issues/27)); a cheap sampling
  check settles it before anything is built.
- **The biases become over-subscribed** — three geometrically distinct jobs on one per-cell vector,
  the third being to preserve private directions through a frozen `encode`. First concrete argument
  for pulling per-cell adapters off the flex ladder early.
- **Some disagreement is now irreducible by design**, which reconciliation, the local learning rule,
  and ADR-0004's falsification signature all assume it is not
  ([#28](https://github.com/NGL321/patchworks/issues/28)).
- The prohibition on reading timescale is what keeps the divisor and the persistence mechanism
  interchangeable. Reversing it costs the cheap fallback.

## Alternatives considered

- **Emergence from depth.** Rejected on the phase-versus-frequency argument above. Its slow resonant
  modes (~`2L` ticks, ~0.5 s here) are an artifact, unsteerable and an order of magnitude short.
- **An explicit per-cell clock divisor as the answer.** Retained as an instrument and fallback, not
  as the mechanism: `k` is arbitrary, it aliases, and it hard-codes hierarchy at construction time in
  a spec that has worked to make hierarchy emergent.
- **Termination-driven abstract steps** — a cell holding until the disagreement it clamps is
  cleared. The most faithful to the active-predictive-coding framing and the most expensive: it needs
  a termination signal, which is new mechanism, and it breaks the synchronous single-step reconciliation
  of [ADR-0002](./0002-message-passing-is-one-step-not-a-solve.md). Held as the named escape hatch.
- **Clamp dwell alone** — treating a held goal as the slow variable. Rejected because it would pass
  the acceptance demo with a *human* supplying the abstraction, and because it would require inventing
  self-clamping for agent-originated plans. Under this decision, persistence *is* holding, so no
  self-clamping mechanism is needed.
- **A probabilistic sheaf**, so that low-confidence messages fail to pass. Rejected as the tail
  wagging the dog: it would change what an edge stalk is, what restriction maps map, what disagreement
  means, and what `decode` outputs — a rewrite of the foundational cell contract — in service of a
  gate that needs only a magnitude threshold on a vector difference. The probabilistic sheaf stays in
  the map's fog on its own merits.
