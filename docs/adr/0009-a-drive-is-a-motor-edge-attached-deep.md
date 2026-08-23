# ADR-0009: A drive is a motor edge attached deep

**Status:** accepted

## Context

A pure prediction-error minimiser has no reason to act. If the agent correctly predicts that the puck
stays where it is and the target zone stays lit, that prediction is *right*, and right predictions are
exactly what the architecture minimises. **An unsolved task, observed from a standstill, is a
low-error state.** This is the **dark room problem**, and it is not a detail: without something that
makes an unmet goal *uncomfortable*, the acceptance demo has nothing driving it and the agent watches
the arena forever.

So some externally-injected "this should be true" pressure is structurally required. The question this
ADR settles is its shape.

**The citation pass on the boundary (`docs/research/026-action-boundary-citations.md`) bears on this
directly, and favourably.** The dark-room problem has two resolutions in the literature and they are
not equivalent. Friston et al. (2012) resolve it **definitionally, through priors**: surprise is
`H(S|m)`, conditioned on the agent's model, so "a dark room will afford low levels of surprise if, and
only if, the agent has been optimized… to predict and inhabit it" — an agent whose priors expect a
structured world finds the dark room *itself* surprising. No exploration bonus, curiosity term, or
information-gain quantity appears in that argument. The now-standard resolution — epistemic value as an
explicit summand of expected free energy (Da Costa et al. 2020) — is a **later, separate** development
that computes and compares information gain **per candidate policy, before acting**.

That second resolution is architecturally unavailable here: ADR-0003 refuses counterfactual evaluation
outright. A drive lands squarely on the **first** horn — it is a prior, written from outside, asserting
a world the agent should expect to inhabit. The resolution this ADR reaches for is the one with a
primary source *and* the only one compatible with the rest of the design, which is a happier
coincidence than it had any right to be.

The working proposal inherited from [ADR-0003](./0003-action-is-prediction-the-world-clears.md) was a
**clamp**: pin a chosen cell's node stalk to a goal value, overriding what the message-passing phase
would have written. That word entered the design as shorthand and was never examined.

Examined, it fails. A clamp that genuinely overrides is not a nudge — it removes the clamped cell from
inference for the clamp's duration, leaving a hole in the network dynamics where a predicting cell used
to be. It also demands a target value expressed in that cell's own learned basis, which nothing outside
the graph can know and which would have to be recovered by calibration.

## Decision

**A drive is a boundary cell attached deep in the core, and its edges are motor edges.** Nothing is
overridden and nothing is clamped.

A **drive boundary cell** is written from outside the sheaf — by the human today, by an internal
faculty later — and holds a standing assertion. Like every other boundary cell it runs no body and
holds no chart ([ADR-0006](./0006-boundary-cell-stalks-are-world-shaped.md)). Its **drive edges** run
to a subset of core cells and are ordinary edges with ordinary masked linear restriction maps. The
assertion reaches those cells as ordinary disagreement, pulled with the ordinary reconciliation gain
`γ / Σ_e m_e`. Every core cell it touches keeps its body and keeps inferring every tick.

Placed in ADR-0003's taxonomy, which sorts edges by **who clears the disagreement**, it is a motor
edge:

| | Sensory edge | Motor edge | Drive edge |
|---|---|---|---|
| Boundary cell is | written by the world | read by the world | written from outside, read by no one |
| Disagreement cleared by | the cell changing its belief | the world moving | the world moving |

The pairing *written from outside, cleared by the world moving* is new, but it is a new combination of
properties the architecture already has, not new machinery. The actuator's motor edge is cleared by the
world moving **immediately**; a drive edge is cleared by the world moving **eventually**. Since
abstraction is hop distance from the sensorimotor rim (`04-action-and-the-boundary.md`), this makes
**abstract action literally a motor edge attached deep** — the same object as a torque command, further
from the rim.

**The drive stalk carries valence, not specification.** It is near-scalar: it asserts *satisfied*, and
nothing else. It never says which puck or which zone, because the render already does
(`03-the-sandbox.md`: the target zone lights up, and `retarget()` only changes what is seen). The drive
supplies the discomfort; the world supplies the content. Direction is not in the signal — it comes from
the graph's own learned model of what satisfaction looks like.

**For the PoC: one drive boundary cell, several drive edges.** One cell is one drive. The sandbox wants
a single (puck, zone) task at a time, so a second buys nothing yet; curiosity, fatigue, or any later
drive arrives as an additional cell, which is an ordinary structural-mask change.

## Consequences

- **Release needs no mechanism, and no detector.** The drive cell asserts *satisfied* forever. When the
  task is actually met, what is sensed agrees, and disagreement falls to the floor on its own —
  pressure vanishes without anything being released, and with no teleology trained into the graph.
  `perturb()` knocks the puck out and the disagreement returns by itself. Nothing anywhere reads
  "is the goal met"; `info.goal_satisfied` stays privileged, for logging only.
- **The internal-drive transfer is a no-op.** Replacing the human writing the cell with a limbic-analogue
  faculty writing it changes nothing else — same cell, same edges, same rim. The map's fog item for
  intrinsic reward acquires its attachment point, and it is this object.
- **Strength is not a new axis.** How hard a drive pulls is set by things that already exist: the drive
  edges' mask width `m_e`, and the existing per-cell gain. There is no new "how hard to clamp" parameter.
- **Multiple simultaneous drives compose by reconciliation**, the same way several cells driving one
  actuator do (ADR-0003). Genuinely incompatible drives are standing disagreement, which is a fourth
  source alongside static, lag, and settling
  ([ADR-0007](./0007-the-disagreement-floor-is-tolerated-not-represented.md)) — tolerated, not
  represented, and needing no arbitration mechanism.
- **"Clamp" is retired as vocabulary**, in `04-action-and-the-boundary.md`, `05-timescales.md`,
  ADR-0003, and ADR-0005. Every claim those passages made survives — planning really is a deep
  assertion propagating to the rim by ordinary machinery — but the noun named the mechanism this ADR
  rejects. A hard clamp is retained in exactly one role, as an **instrument**: pinning a stalk to debug,
  and to isolate whether the drive edge is doing the work. The same status ADR-0005 gives the clock
  divisor. Instrument, never mechanism.
- **Graph topology gains one cell and its mask entries** — a revision against `06-graph-topology.md`,
  small because it is an ordinary boundary cell, not a new category.

## Known exposure

- **One dimension of standing disagreement steering a 150-cell graph is unproven.** Low bandwidth is
  deliberate — a wide drive channel would smuggle the task specification back in through the side door,
  after `03-the-sandbox.md` worked to put it in the render — but whether a scalar suffices to
  differentiate behaviour across the taper is exactly the thing most likely to need widening. The
  escape hatch is a small learned drive vector, at the cost of the one-cell-one-drive reading.
- **Bootstrapping.** The drive's meaning arrives through a restriction map the transport rule must
  learn, so early in training a drive edge is noise. This is the same cost every sensory edge already
  pays and the architecture accepts everywhere else, but the drive edge is the one place where paying
  it delays the behaviour the demo exists to show.
- **Hallucinating satisfaction.** A core cell can reduce disagreement by *believing* the task is met
  rather than by acting — the failure GLean documents for goal-conditioned forward models
  (`docs/research/018-sandbox-citations.md`). Under a drive edge it is bounded rather than eliminated:
  the sensory edges pull the other way continuously, so the cell settles at a compromise, and that
  compromise is the prediction the motor rim must clear. It leaves an observable signature —
  sensory-side disagreement growing while the motor side stays quiet — which the demo can watch for.
