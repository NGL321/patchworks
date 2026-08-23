# Timescales

Where more than one timescale comes from, given the cell contract of
[`01-cell-and-sheaf.md`](./01-cell-and-sheaf.md) and the tick of
[`02-tick-semantics.md`](./02-tick-semantics.md).

Terms used here are defined in [`CONTEXT.md`](../../CONTEXT.md).

## What is actually wanted

Not "cells update at different rates." The object is **temporal abstraction**: one step at an
abstract level corresponding to many primitive actions, so a plan is short at the top and long at
the bottom. Differing rates are how the active-predictive-coding literature realises that; they are
the symptom, not the goal.

The precise thing the architecture must supply is **commitment**: an abstract belief that stands
across many primitive steps. Two failure modes motivate it, and two others are explicitly *not*
bought here:

| | |
|---|---|
| **Commitment** — holding an abstract intention steady while the rim churns | this section |
| **Representation** — encoding a variable that only exists over seconds | this section |
| **Memory** — retaining information across hundreds of ticks | recurrent failure modes; escape hatch is the LSTM-shaped edge-stalk pass-through |
| **Credit** — associating an abstract belief with a much later outcome | [`#5`](https://github.com/NGL321/patchworks/issues/5) |

Nothing in this section improves memory or credit assignment. Believing otherwise later would be a
mistake worth avoiding now.

## Depth does not supply it

`01-cell-and-sheaf.md` states that depth buys horizon, not rate. The reason, stated here because it
is the argument the rest of this section rests on:

**Unit delay is a phase shift, not a decimation.** It moves a signal in time; it removes no
frequency content from it. Once the graph's pipeline is full — after roughly diameter-many ticks,
once — every cell receives a fresh message on *every* tick, forever. There is no subsequent silence
during which a distant cell integrates. A cell ten hops out has exactly the same input bandwidth as
one at the rim; it is merely looking at older data.

There is a real ratio that depth *does* buy, and it is the wrong one. A correction originating `d`
hops in reaches the motor rim `~2d` ticks later, so a deep cell's influence on behaviour spans many
primitive actions. That is **ratio in latency**. Planning needs **ratio in commitment** — the
abstract level standing behind one decision across many steps — and a deep cell still revises its
belief every tick. It revises it about stale evidence.

Secondary: a loop of length `L` with unit delay has slow resonant modes on the order of `2L` ticks.
At the sandbox's 50 Hz and a graph of diameter ~12 that is ~0.5 s — an order of magnitude short of
task duration, not steerable, and an artifact rather than a mechanism.

## The mechanism: persistence in the private features

Timescale is **not a schedule**. Every cell runs every tick, exactly as
[`02-tick-semantics.md`](./02-tick-semantics.md) specifies; nothing in this section changes the tick.
A cell is slow because its content **persists**, not because it updates rarely. See
[ADR-0005](../adr/0005-timescale-is-persistence-not-a-schedule.md).

Persistence needs two things, and they defend against two different disturbances.

### Insulation from neighbours: `H⁰`

Reconciliation descends on Dirichlet energy, so it moves a node stalk along `im δᵀ`. The private
features are `ker δ` (`01-cell-and-sheaf.md`, *`H⁰` is the private features*), and
`ker δ = (im δᵀ)^⊥`. **Reconciliation therefore leaves the private component of a node stalk exactly
invariant** — orthogonally, not approximately.

This is what makes slow state possible at all despite uniform bandwidth. The bandwidth argument
above applies to the *reconciled* subspace, which is driven every tick. The private subspace is not
driven by neighbours at any rate.

**Slow content lives in the private features.** This is a commitment of the spec, not an
observation about where learning might put things. It is also not an overload of what private
features are for: holding a slowly-varying variable *is* an abstract cell's sub-problem.

The bound `dim H⁰ ≥ Σ_v max(0, n − Σ_{e∋v} m_e)` makes the capacity for slow state a **construction
quantity**, set by the masks and enlarged by sparsity.

### Persistence under the cell's own dynamics: the regional Jacobian

Insulation is from *neighbours*. It is not insulation from the cell itself — the private component
still passes through `encode` → `step` → `decode` every tick, and nothing about `ker δ` makes that
round trip near-identity.

What supplies persistence is bias translation. Per `01-cell-and-sheaf.md`
(*The division of labour between the two adapting surfaces*), the shared frozen body is one
piecewise-linear map whose fold directions are fixed and whose fold offsets are per-cell. Each cell
therefore operates in a **different activation region of the same map**, and each region has its own
local Jacobian with its own spectrum. A cell's **effective timescale is the spectral radius of its
region's Jacobian**, selected by its biases.

So: **private dimension supplies protected state; the regional spectrum supplies its decay rate.**
Neither is new mechanism. Both were already committed to, for other reasons.

## What this requires elsewhere

**A body whose regional spectra actually spread.** If the distribution of regional Jacobian spectral
radii — across the bias settings cells occupy — is a spike, every cell lands in the same dynamic
regime and nothing differentiates. Initialisation is a parameter of the body
(`01-cell-and-sheaf.md`), and this is the first thing that freedom is spent on: the body is to be
constructed for spread, not merely assumed to have it. Whether such a construction exists is
[#27](https://github.com/NGL321/patchworks/issues/27).

**A cheap go/no-go before anything is built:** sample bias vectors, measure the regional spectral
radius of each, plot the distribution. If it is a spike, this mechanism is dead and the afternoon
that established it was well spent.

**Two constraints handed to graph topology** ([#8](https://github.com/NGL321/patchworks/issues/8)),
both hard:

- **Abstract cells need bounded total mask width.** A cell holds slow state only if `Σ_e m_e` is
  meaningfully below `n`. High-degree cells lose the *guarantee* of private dimension, though not
  necessarily the fact of it — the bound is a lower bound, and learned rank-deficiency enlarges `H⁰`
  past it. Low degree buys private dimension by construction; high degree makes it contingent on
  learning.
- **Integration and holding may need to be separate cells.** Nothing is disqualified by the above —
  a relay cell performs no prediction and so has no slow state to hold, which is most of what
  high-degree cells are for. The tension bites only for cells that both *predict* and *integrate
  widely*. Where an abstract slow variable needs wide integration, the shape that satisfies both is
  a **pair**: a high-degree cell that integrates, feeding a low-degree cell that holds.

## The clock divisor, as an instrument

An explicit schedule — a cell updating every `k` ticks, `k` fixed by hand — is **built first, and
not as a fallback.** It is the instrument that establishes the capability depends on timescale at
all. Force `k`, confirm long-horizon behaviour appears, and the variable is isolated before anything
is spent on making it emerge. Then switch it off and see whether persistence reproduces it.

This ordering is what makes failure cheap. If the persistence mechanism does not differentiate, the
divisor is already built and already validated against the same demo, so the system still works —
it is only less interesting.

**Nothing in the architecture reads a cell's timescale.** It is observable from outside and is never
an input to any computation, never a cell attribute another mechanism branches on, never a selection
criterion. Clamping selects by hop distance ([#9](https://github.com/NGL321/patchworks/issues/9)),
which both configurations respect. This prohibition is what keeps the divisor and the persistence
mechanism interchangeable; the moment anything branches on rate, the cheap fallback is gone.

## Demonstrating it

The acceptance demo's evidence is a **live private-component readout**: during a perturbation,
display `‖Δ(private component)‖` per cell against hop distance from the sensorimotor rim. The viewer
watches the rim swing while deep private state barely moves.

Cheap — the private component is the node-stalk directions masked out on every incident edge, known
at construction, so it is a fixed projection computed per tick. Falsifiable live: if deep private
state swings with the rim, the mechanism is not working and it is visible in the moment. This is a
requirement on [#11](https://github.com/NGL321/patchworks/issues/11), which must expose the
private/reconciled decomposition rather than raw stalk values.

Behaviour alone is **not** accepted as evidence — a purely reflexive controller produces the same
footage.

## Known exposure

- **The biases are over-subscribed.** They now carry three geometrically distinct jobs on one
  per-cell vector: fix the cell's fold offsets (`01-cell-and-sheaf.md`), select a regional spectrum
  slow enough to hold state, and select a region whose Jacobian preserves the private directions
  through **compression**. That third job is the fragile one — `encode` is frozen and shared, so a
  private direction that survives reconciliation still has to survive being compressed into a
  `k`-dimensional chart, and nothing guarantees it does. There is slack (the restriction maps fix
  the node stalk's basis, so a cell can partly align its private directions with what `encode`
  preserves) but the masks are construction-fixed and never re-open, so that alignment is bounded at
  build time. **This is the first concrete argument for pulling the first rung of the flex ladder —
  per-cell adapters — sooner than planned:** adapters are what buys a fourth handle if three jobs
  will not fit on one.
- **Some disagreement is irreducible by design** — a slow cell adjacent to a fast one never agrees
  with it, and that is the mechanism working. **No longer exposure; decided** in
  [ADR-0007](../adr/0007-the-disagreement-floor-is-tolerated-not-represented.md). It is a **lag
  floor**, one of the two kinds of disagreement floor, and it is tolerated rather than represented.
  Reconciliation is unharmed directly — the private component is exactly invariant at any gain — but
  the standing offset a floor leaves on the reconciled component shifts the cell's operating point,
  which is where this section's decay rate comes from. That is bounded by the reconciliation gain
  ([`02-tick-semantics.md`](./02-tick-semantics.md)), whose bound binds hardest at the apex.
- **Change-gated transport is the named amplifier, and is not taken here.** Suppressing transmission
  when a restricted belief has not moved makes transmission rate track content rate, which sharpens
  whatever differentiation exists — positive feedback on exactly this mechanism. It cannot bootstrap
  from nothing, so it is worth reaching for only if the persistence mechanism shows weak but nonzero
  differentiation. Handed to [#20](https://github.com/NGL321/patchworks/issues/20) with two cautions:
  a threshold gate is a **deadband**, and a deadband inside a loop with unit delay is a standard
  recipe for sustained limit cycles rather than the equilibrium it intuitively suggests; and any
  threshold must be **derived** (relative to that edge's own recent scale) rather than hand-set,
  since `m` varies by mask and one absolute constant cannot mean the same thing on two edges.
- **A confidence gate is not a substitute.** Suppressing transmission when a belief is *uninformative*
  sparsifies the graph but decimates nothing in time — a confident, fast-changing cell still sends
  every tick. Recorded because the two gates are easy to conflate, and only one is a low-pass filter.
