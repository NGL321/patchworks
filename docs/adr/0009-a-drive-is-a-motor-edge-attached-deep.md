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
a world the agent should expect to inhabit.

That is a stronger position than it first reads as. The burden the priors route usually carries is
explaining where world-expecting priors *come from* without appealing to epistemic value, which is why
its defenders reach for evolution and neurodevelopment. Patchworks writes the prior in from outside, so
it never incurs that burden. Someone has already walked the horn without the epistemic summand: Baltieri
& Buckley (2019) derive PID control as active inference with the setpoint as a prior, full stop, and no
information-gain or expected-free-energy term anywhere in the formulation. Its scope is narrow — one
regulated variable, a linear model, no hierarchy — so it proves the horn is walkable, not that it scales
to a ~150-cell graph. And Sun & Firestone's sharpest objection, that optimism only escapes the dark room
by smuggling desires back in, turns on predictive processing's claim that prediction is the *only*
state. Patchworks never made that claim: the desire is declared, as a motor edge, in the open. See
`docs/research/045-drives-citations.md`.

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
to a subset of core cells — fixed by [#36](https://github.com/NGL321/patchworks/issues/36) as the
**apex level, entire** — and are ordinary edges with ordinary masked linear restriction maps. The
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

**The drive stalk carries valence, not specification.** It is near-scalar — fixed by #36 as literally
one dimension, with `m_e = 1` on its edges to match. It asserts *satisfied*, and nothing else. It
never says which puck or which zone, because the render already does
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
- **Strength is not a new axis.** How hard a drive pulls is set by things that already exist. There is
  no new "how hard to clamp" parameter.

  **Amended by [#36](https://github.com/NGL321/patchworks/issues/36): the knob is fan-out, not mask
  width.** This ADR named `m_e` and the gain. `m_e` turns out not to be usable. A restriction map out
  of a `d`-dimensional stalk has rank at most `d`, so with a scalar drive stalk an `m_e` above 1
  carries nothing extra — it only lets the drive assert *zero* in the surplus directions, an arbitrary
  constraint on a core cell in content the drive does not have. Worse, the per-cell gain is
  `γ / Σ_e m_e`, so widening a drive edge **turns down every other edge that cell has**: the drive
  making itself heard by making the world quieter. The usable knob is the **number of attachment
  points** — not free either, since each new one costs that cell a dimension of privacy and the same
  dilution, but it spreads the cost instead of concentrating it and never makes one cell pay more than
  the minimum. The learned drive vector below remains the move after that one.
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
  small because it is an ordinary boundary cell, not a new category. **Done in
  [#36](https://github.com/NGL321/patchworks/issues/36):** the drive attaches at the **apex level,
  entire** — one edge to each of the eight L7 cells — with a **scalar** stalk and `m_e = 1`. The apex is
  the most abstract place in the graph and the only part of the core with private dimension to spare;
  the cost is one dimension of privacy per apex cell (16 → 15) and 8 off `χ`.
- **The assertion needs the tick to be ordered.** ADR-0009 has the drive standing forever, which is only
  true if reconciliation cannot erode it. #36 settled this without an exemption: external writes land
  **after** the message-passing phase, as the tick's last word
  (`02-tick-semantics.md`, *External writes*). Reconciliation moves the drive stalk and the write
  restores it before it next speaks, so a drive edge's disagreement can only ever fall by the *cell*
  moving — the motor-edge property of this ADR, derived rather than stipulated.

## Known exposure

- **One dimension of standing disagreement steering a 150-cell graph is unproven.** Low bandwidth is
  deliberate — a wide drive channel would smuggle the task specification back in through the side door,
  after `03-the-sandbox.md` worked to put it in the render — but whether a scalar suffices to
  differentiate behaviour across the taper is exactly the thing most likely to need widening.
  Confirmed real by `docs/research/045-drives-citations.md`: nothing found runs a single scalar as the
  sole directional drive of a large graph, biology runs four distinct diffuse scalars with four
  distinct jobs (Doya 2002), and the nearest comparable object — FeUdal's directional goal — is a
  vector, "the dimensionality of the embedding vectors, w, … set as k = 16" (Vezhnevets et al. 2017).
  Abel et al. (2021) prove a scalar cannot express some task specifications, which lands softer here
  because the drive is not the specification: the render is. What survives is the width question, and
  it is now pre-costed.

  **Trigger.** Task-invariant behaviour: the arm's trajectory the same across tasks differing only in
  the render, while the drive edge's disagreement is non-trivial. The mechanism-level confirmation is
  an undifferentiated apex — the eight apex node stalks moving near-identically under drive. A drive
  that produces no motion at all is **not** this failure; that is *Bootstrapping* below, and a wider
  channel does not fix it.

  **The hatch, in rungs.** (1) **More attachment points** — the strength knob already named above, and
  the cheapest thing to try. (2) **A second drive cell** with a distinct job, which is the reading the
  biology actually supports: Doya's answer to needing more directional influence is another diffuse
  scalar channel, not a wider one. Near-empty in this PoC, where there is one task and so nothing for a
  second drive to *be* — it is real for an internal-rim faculty later. (3) **A learned drive vector at
  `k ≈ 16`**, the attested width, at the cost of the one-cell-one-drive reading. The rungs are ordered
  by price, and only (3) changes the design's shape.
- **A derived account of curiosity is forfeited — inside the graph.** Expected free energy's epistemic
  term is what makes long-horizon agents intrinsically novelty-seeking, and Seth, Millidge, Buckley &
  Tschantz (2020) insist it "arise[s] naturally out of the mathematical formalism, instead of being
  bolted on." It is an expectation over futures under candidate policies, and
  [ADR-0003](./0003-action-is-prediction-the-world-clears.md) has no counterfactual evaluation, so it
  is not computable in the sheaf. Exploration is not thereby lost: curiosity enters as an ordinary
  drive boundary cell like any other, and what is written from outside is never *what a drive means* —
  the transport rule learns that, as it does for every edge — but only whether and how hard it is
  written. What the graph cannot do is *become* novelty-seeking on its own. If that is ever derived
  rather than asserted, it is derived at the **internal rim**, in a faculty, and arrives here as an
  ordinary drive. Live multiplicity among apex cells is not the missing evaluation: they disagree about
  the *actual* next step, not about candidate futures (`04-action-and-the-boundary.md`, *Route
  selection*).
- **Bootstrapping.** The drive's meaning arrives through a restriction map the transport rule must
  learn, so early in training a drive edge is noise. This is the same cost every sensory edge already
  pays and the architecture accepts everywhere else, but the drive edge is the one place where paying
  it delays the behaviour the demo exists to show.

  This exposure carries a second job it was not filed with. The PoC's whole account of exploration is
  that an unconverged model emits near-arbitrary torques and sharpens where it acted
  (`04-action-and-the-boundary.md`, *Route selection*), so the window in which the drive edge is noise
  is the *same* window that account depends on: the agent has to keep moving through it. **No motion
  at all is therefore both failures at once**, and only one response addresses it — a **curiosity
  drive**, an ordinary drive boundary cell at the internal rim, which is the fog item and not a rung
  on the ladder above. Widening the task drive does not reach it, and neither does attaching it more
  widely.
- **Hallucinating satisfaction.** A core cell can reduce disagreement by *believing* the task is met
  rather than by acting — the failure GLean documents for goal-conditioned forward models
  (`docs/research/018-sandbox-citations.md`). Under a drive edge it is bounded rather than eliminated:
  the sensory edges pull the other way continuously, so the cell settles at a compromise, and that
  compromise is the prediction the motor rim must clear. It leaves an observable signature —
  sensory-side disagreement growing while the motor side stays quiet — which the demo can watch for.
