# ADR-0007: The disagreement floor is tolerated, not represented

**Status:** accepted

## Context

[ADR-0005](./0005-timescale-is-persistence-not-a-schedule.md) made some disagreement **irreducible by
design**: a slow cell adjacent to a fast one will never agree with it, and that is the mechanism
working rather than failing. Three parts of the architecture had been written assuming disagreement
is in principle reducible — reconciliation, the local learning rule
([#5](https://github.com/NGL321/patchworks/issues/5)), and
[ADR-0004](./0004-linear-restriction-maps-assume-local-flatness.md)'s falsification signature.

Two facts about the shape of the problem, established before anything was decided:

- **The floor is a gradient across the whole graph, not a pathology on a few edges.**
  `06-graph-topology.md` gives guaranteed private dimension as a taper — zero at the rim, ~16 at the
  apex — so every vertical edge is a slow↔fast edge to some degree. There is no clean subset of
  "timescale edges" to special-case.
- **Reconciliation cannot erode slow content directly.** Slow content is `ker δ`; reconciliation
  moves along `im δᵀ`. The pull is exactly orthogonal to the thing it would erase.

## Decision

### Disagreement divides into a reducible part and a floor

The **disagreement floor** is the part of an edge's disagreement that learning cannot remove. It has
three kinds:

- **Static floor** — a function of *configuration*. Curvature the linear restriction map cannot
  follow (ADR-0004), mask or learned rank deficiency, and aleatoric noise. Present at rest. *Amended by
  [#37](https://github.com/NGL321/patchworks/issues/37):* it has a fourth source, added deliberately.
  [ADR-0010](./0010-restriction-map-scale-is-gauge-fixed.md) bounds every restriction map's norm, so an
  edge's representable **scale ratio** between its two ends is bounded too — `ρ²` times the range rank
  concentration affords. Where two cells' stalks genuinely differ in scale by more than that, the
  remainder cannot be transported away and sits here. It is the price of excluding `F = 0`, and it is
  paid in the currency this ADR already tolerates.
- **Lag floor** — a function of *motion*. The two endpoints' contents live at different timescales,
  so the slow end is behind. Drains at rest.
- **Settling floor** — a function of *parameter drift*, added by
  [#33](https://github.com/NGL321/patchworks/issues/33). A mid-depth predicting cell whose bias rule
  ([#5](https://github.com/NGL321/patchworks/issues/5)) receives ambiguous, sign-flipping prediction
  error oscillates between activation regions instead of converging, so its outgoing prediction never
  stabilises and its incident edges never fully clear. Bounded, not cumulative, by the same
  construction that already bounds reconciliation — see "Simultaneous learning does not need its own
  bound" below.

Everything else is **model error**: the cell is simply wrong, the residual is reducible, and this is
what the local learning rule feeds on. It gets no name of its own; its whole role is to be the thing
the floors are distinguished from.

### Nothing represents the floor

No per-edge state estimates it, no channel carries it, no cell is told a floor exists. What a floor
produces is a **bounded standing offset** on the reconciled component of a node stalk — bounded, not
cumulative, because disagreement is re-derived every tick after both ends re-predict.

The risk this creates is not erosion of slow content but a shift in the cell's **operating point**,
and per ADR-0005 the operating point is where timescale comes from. A large enough standing offset
pushes a cell across a fold into an activation region with a different spectral radius. **Timescale
separation would then be erased by its own reconciliation** — not by the slow value being dragged,
but by the slowness being dragged.

This makes the governing quantity a product of gain and floor, which forced a gap in the spec into
the open: **reconciliation's step size had never been named.** ADR-0002 forbids a round count, but a
descent step needs a gain regardless. It is now specified in `02-tick-semantics.md` as
`γ / Σ_e m_e` per cell, with the bound that `γ × floor` must stay below the cell's fold margin,
checked at construction. Because `Σ_e m_e` falls with depth, the gain is largest at the apex and the
bound **binds hardest exactly where timescale matters most**.

*Amended by [#41](https://github.com/NGL321/patchworks/issues/41): the fold margin has a second job,
and it is structural.* This ADR framed the margin as protecting the operating point *from the floor*.
Since a cell's regional spectrum is re-drawn every tick as its chart moves
([ADR-0005](./0005-timescale-is-persistence-not-a-schedule.md), amended), the margin is also what
makes "the cell's region" a well-defined object at all: it is the construction-time proxy for
**region dwell**, so a cell with a small margin has no well-defined timescale even at zero floor.
The bound is unchanged and the check is the same check — but it is now a precondition of the
timescale claim as well as a stability condition, and `02-tick-semantics.md` says so at the point
where a build would be tempted to relax `γ`.

### The learning rule tolerates the floor; it does not subtract it

The rule may never take zero residual as its target. It learns on *change* in residual, or on
residual relative to that edge's own recent scale — a constraint on the objective
[#5](https://github.com/NGL321/patchworks/issues/5) has to choose anyway, not new mechanism. The
cost is accepted: the floor consumes some of the rule's dynamic range.

### The two floors are separated by a quiescent hold, offline

Hold the world still and watch what drains. The **lag floor decays toward zero** — nothing is moving,
so there is nothing for the slow end to be behind. The **static floor does not** — it is a property
of that configuration and it sits there. Swept across several configurations, since a static floor is
positional and one pose reports on one point of the overlap.

The sandbox supports this without a special mode: `reset()` rearranges the world and never the agent,
the clock is monotonic, and there are no episodes, so a quiescent hold is an ordinary state of the
environment. Diagnosis is offline and comparative, which ADR-0004 already conceded it must be, and it
composes with `06-graph-topology.md`'s topology-only baseline rather than replacing it — the baseline
says what disagreement a cycle produces with no curvature at all, the hold says how much of the
remainder is lag.

### Simultaneous learning does not need its own bound

[#33](https://github.com/NGL321/patchworks/issues/33) asked whether the local learning rule
([#5](https://github.com/NGL321/patchworks/issues/5)) — every cell running both the bias rule and the
transport rule every tick, off signals that shift as neighbours update simultaneously — needs a bound
of its own, the way reconciliation needed `γ × floor < fold margin`, or whether it inherits stability
from what is already decided. The answer splits along tick-locality.

**Within a tick, nothing new is needed.** Each rule is already a single local gradient step, applied
after that tick's signals (prediction error, disagreement) are read — the same shape ADR-0002 requires
of reconciliation, already satisfied by construction. Simultaneity across cells creates no new locality
problem, because no cell's update this tick depends on another cell's update this tick.

**Across ticks, the risk splits by parameter group, and neither half needs new mechanism — one needed
naming, the other needed this bound extended.**

- **The bias rule can produce the settling floor** above: a mid-depth cell with ambiguous evidence
  oscillates between activation regions rather than settling. This is not the "standing offset dragging
  the operating point" failure this ADR already guards against — that failure is a *cell being pushed*
  by something external; this is a cell's own gradient step being genuinely undecided. It cannot diverge
  (bounded by the same `γ` that already bounds reconciliation; `decode` still emits every tick, so a
  neighbour's evidence degrades but is never blocked) and it is not a stability defect to fix — it is a
  third floor, tolerated the same way the other two are.
- **The transport rule does not need a new bound; it makes an existing one go stale.** `gain_v = γ /
  Σ_e m_e` treats `Σ_e m_e` as a proxy for the local Laplacian block's largest eigenvalue, accurate when
  it was checked. The transport rule trains the restriction-map *magnitudes* that proxy stands in for,
  so as training proceeds the proxy can drift away from the block's true spectral radius, silently
  loosening `γ × floor < fold margin` without anything re-checking it. The fix was not a cap on what a
  restriction map may learn — that would impose a geometric constraint the maps' job (`01-cell-and-sheaf.md`,
  ADR-0004) doesn't call for. It was cheaper and stayed local: each cell recomputing its own actual
  spectral estimate from its own incident maps, on the same schedule the global learning-rate and
  sparsity anneals already use ([`07-local-learning-rule.md`](../spec/07-local-learning-rule.md)).

  *Superseded by [#37](https://github.com/NGL321/patchworks/issues/37).* That fix was correct and is no
  longer needed. [ADR-0010](./0010-restriction-map-scale-is-gauge-fixed.md) **fixes the magnitudes the
  transport rule was found not to identify**, so the proxy stops drifting because there is nothing left
  to drift: `λ_max ≤ ρ² · deg(v)` holds for as long as the run does, and
  [`02-tick-semantics.md`](../spec/02-tick-semantics.md) takes the max of the two bounds. The periodic
  re-derivation is **struck** rather than kept just-in-case. Note what did *not* happen — the gauge is
  still not a cap on what a map may learn. It fixes a magnitude no term in the objective has an opinion
  about, which is why it closes this hole without imposing the geometric constraint declined above.

Neither half is [#20](https://github.com/NGL321/patchworks/issues/20)'s change gate or the probabilistic
sheaf. The gate amplifies differentiation on an edge; it has no purchase on whether a cell's own
parameter update oscillates or whether a spectral-radius proxy has gone stale. The probabilistic sheaf
was declined a third time on the same structural grounds as its first two declines (ADR-0005, #7): a
distribution over stalk values doesn't change how many things are moving at once or how far, which is
what this question is actually about.

## Consequences

- **ADR-0004's falsification signature is no longer readable on its own.** It has a second cause, and
  reading it now requires the topology-only baseline and the quiescent hold first. ADR-0004 is
  amended accordingly.
- **`01-cell-and-sheaf.md`'s "penalised, never cleared" survives, its justification does not.**
  "Residual disagreement *is* the signal the rule consumes" was written when all residual was
  informative. Some now is not.
- **Reconciliation gain enters the spec**, along with a per-cell construction check folded into
  [#27](https://github.com/NGL321/patchworks/issues/27)'s sampling rig. Per
  [#33](https://github.com/NGL321/patchworks/issues/33) that check could not stay construction-time
  only — and per [#37](https://github.com/NGL321/patchworks/issues/37) it does after all, because
  ADR-0010 bounds the map magnitudes the `Σ_e m_e` proxy stood in for. One construction-time check, no
  running re-derivation.
- **The gain is not a timescale knob and must not become one.** Normalising by `Σ_e m_e` tracks the
  local Laplacian block's largest eigenvalue; its job is to *equalise* the effective step across the
  taper, removing a degree artifact. A gain deliberately graded by depth would be ADR-0005's rejected
  clock divisor wearing a different name.

## Alternatives considered

- **A per-edge learned baseline**, subtracted from the residual before the rule consumes it — cheap,
  strictly local, one vector per edge. This is **the same mechanism as affine restriction maps**, only
  relocated: either way it is one learned offset per edge.

  Affine maps are a real object and cost less than assumed. `‖δx + b‖²` is still quadratic, its
  gradient is `Lx + δᵀb`, reconciliation is still one cheap linear step, `H⁰ = ker δ` is unchanged and
  the private-features identity survives; only the consistent set becomes an affine subspace rather
  than a linear one.

  Rejected anyway, on two grounds. **It catches the static floor's constant part and nothing else** —
  the lag floor is velocity-shaped, sign-flipping, zero at rest, and a fixed offset does nothing to
  it. To catch that, the baseline would have to *track*, which means a hand-set time constant: a new
  free parameter that is itself a timescale, in a system whose entire timescale story is meant to fall
  out of biases. That is ADR-0005 quietly reversed inside a per-edge auxiliary variable.

  Retained as the named escape hatch, under the name **affine restriction maps**, with the note that
  it buys the static floor only.

- **Correlating the residual with the neighbour's rate of change**, to separate a timescale-induced
  residual from curvature. Rejected: first-order approximation error grows with displacement along the
  overlap, so a *curvature* residual correlates with rate of change too. Both move together and the
  test separates nothing. Replaced by the quiescent hold, which probes the axis the two kinds
  genuinely differ on — driven versus positional.

- **Giving reconciliation a notion of a floor**, so it stops descending once it reaches one. Rejected:
  it needs a per-edge estimate of where the floor is, which is the rejected baseline again, and the
  standing offset it would prevent is bounded and cheaply bounded further by `γ`.

## Relation to change-gated transport

[#20](https://github.com/NGL321/patchworks/issues/20)'s change gate and this decision meet head-on and
the collision is recorded here because it was not visible from either side alone.

Three objects are easy to conflate and are distinct. **Gain** governs the stability of one descent
step. **Persistence** (`H⁰` insulation plus the regional Jacobian, ADR-0005) supplies commitment.
**The change gate** amplifies whatever differentiation persistence already produced; it cannot
bootstrap from nothing.

Neither of the first two does the gate's job. The gain cannot: normalisation deliberately equalises,
and even a differentiating gain would only change how fast a cell accepts neighbour evidence —
latency — while what planning needs is ratio in **commitment** (`05-timescales.md`). Reconciliation
never touches `ker δ` at any gain.

And the gate has a cost this decision names: on a slow cell's **outbound** edge it suppresses
transmission, so the fast neighbour reconciles against an ever-staler slow belief. **The gate
manufactures lag floor.** `05-timescales.md` calls it "positive feedback on exactly this mechanism,"
which cuts both ways — positive feedback on differentiation is positive feedback on the floor. The
quiescent hold is the instrument that catches it: a gated graph held still should still drain to zero.

Not a veto. [#20](https://github.com/NGL321/patchworks/issues/20)'s question was narrowed from *how
to gate* to *whether the gate is ever reached for, and on what observed trigger*, and **it settled
there: the gate is specified and not built**, with its trigger, its outbound-only attachment, its
stateless threshold and its boundary exemptions written into
[`05-timescales.md`](../spec/05-timescales.md) (*The change gate, pre-specified*).

Two consequences land back here. First, the threshold `05-timescales.md` required to be *derived*
cannot be derived from a running average of the edge's recent scale: that is an auxiliary per-edge
variable with a hand-set time constant, which is the same object, and the same objection, as the
per-edge baseline this decision rejects above. The gate uses a locally stateless criterion instead,
relative to the restricted belief's own current magnitude. Second, the register of easily-conflated
objects this section opens is now **four**, not three — gain, persistence, the change gate, and
**recurrent-state gating**, which is distinguished from the change gate by *tier*: it sits inside the
cell body's recurrence, not on the edge. Its shape is settled and its two rungs — an ungated protected
channel through `step`, and behind it a learned gate on `encode`'s fusion — are specified in
[`01-cell-and-sheaf.md`](../spec/01-cell-and-sheaf.md) (*Known exposure*). Neither is built.

See [patchworks#28](https://github.com/NGL321/patchworks/issues/28) and
[patchworks#33](https://github.com/NGL321/patchworks/issues/33).
