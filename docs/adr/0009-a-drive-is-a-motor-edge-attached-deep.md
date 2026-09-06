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
`gain_v = γ / (g_v² · c_v)` ([`02-tick-semantics.md`](../spec/02-tick-semantics.md),
*Reconciliation gain*). Every core cell it touches keeps its body and keeps inferring every tick.

What stood here named a different denominator, quoted rather than deleted because a consequence below
was built on it:

> pulled with the ordinary reconciliation gain `γ / Σ_e m_e`

*Superseded by [#190](https://github.com/NGL321/patchworks/issues/190)*, which struck `Σ_e m_e` from
the denominator outright — as a **bound** it was never derived, and as a **normalisation** the
*equalising* property it was separately defended for was measured and never held (`gain_v · λ_max`
spreads **3.57x at initialisation and ~2.6x taught, graded by depth and largest at the apex**).
[#155](https://github.com/NGL321/patchworks/issues/155) published the replacement in the spec.
**This ADR's decision does not turn on which denominator it is** — a drive edge is pulled with
whatever the ordinary gain is, and that is the whole content of the sentence. *Strength is not a new
axis* below did turn on it, and is amended there.

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

  **Amended by [#36](https://github.com/NGL321/patchworks/issues/36), and re-grounded by
  [#188](https://github.com/NGL321/patchworks/issues/188): `m_e > 1` is still not taken, but neither of
  #36's two reasons survives.** The conclusion stands and its entire support has been replaced. What
  stood here, quoted rather than deleted because the reasoning that produced it is worth keeping
  legible:

  > This ADR named `m_e` and the gain. `m_e` turns out not to be usable. A restriction map out
  > of a `d`-dimensional stalk has rank at most `d`, so with a scalar drive stalk an `m_e` above 1
  > carries nothing extra — it only lets the drive assert *zero* in the surplus directions, an
  > arbitrary constraint on a core cell in content the drive does not have. Worse, the per-cell gain
  > is `γ / Σ_e m_e`, so widening a drive edge **turns down every other edge that cell has**: the
  > drive making itself heard by making the world quieter. The usable knob is the **number of
  > attachment points** — not free either, since each new one costs that cell a dimension of privacy
  > and the same dilution …

  *Superseded by [#188](https://github.com/NGL321/patchworks/issues/188)* on both grounds:

  - **The gain ground is void.** [#190](https://github.com/NGL321/patchworks/issues/190) struck
    `Σ_e m_e` from the denominator entirely and
    [#155](https://github.com/NGL321/patchworks/issues/155) published
    `gain_v = γ / (g_v² · c_v)` ([`02-tick-semantics.md`](../spec/02-tick-semantics.md),
    *Reconciliation gain*). **Widening a drive edge turns down nothing**, and the cost argument goes
    with the formula — including the *"and the same dilution"* half of the fan-out pricing, which was
    that same `Σ_e m_e` under another name.
  - **The content ground is false as the code is built.** A boundary cell's own maps are
    **scale-pinned, not frozen**: `TransportRule.step` steps every pair, and only `RestrictionMaps`'
    projection knows about pinning, restoring *scale* alone — which is
    [#356](https://github.com/NGL321/patchworks/issues/356)'s *"`grep pinned
    src/patchworks/learning.py` returns nothing."* At `m_e = 1` out of a one-dimensional stalk the
    drive-side map's admissible set is `{±1}`, **zero degrees of freedom**; at `m_e = w` it is
    `S^(w−1)`, **`w − 1` learnable degrees of freedom the transport rule already trains**, needing no
    gauge change and no unpinning. The surplus directions carry a **learned unit vector**, not zero.
    *Valence, not specification* is untouched: the drive still asserts one number, and the graph still
    chooses the direction.

  **What closes the axis instead is the transport rule's sparsity pressure.** #188 read the drive
  edges' apex-side maps over three seeds at 30k ticks: mean squared-mass share in a single apex-stalk
  coordinate **0.9904–1.0000**, Hoyer `h` **0.2470–0.2539** against a floor of `1/√17 = 0.2425` —
  **1.02x–1.05x the pressure's exact global minimum**, where the interior maps on the same apex cells
  sit at **1.17x–1.20x** their own floor. So `rank(F_apex) = 1` whatever `m_e` is; the composition
  through a drive edge is one coordinate of the apex stalk at `m_e = 1` and one coordinate of it at
  `m_e = 16`, and widening cannot widen what the drive constrains. The ruling is stated at
  `λ = DEFAULT_SPARSITY_PRESSURE`, the operating point the build runs, and its reversal condition is
  [#393](https://github.com/NGL321/patchworks/issues/393).

  **#393 has since reported, and the reversal condition is met in the sense it named.** The `λ` sweep
  found the collapse is the constant's own doing: at `λ = 0` the fleet's effective rank **holds at
  2.913** over 30k ticks where the default drives it to **1.002**. The width axis is therefore inert at
  `0.4` and **not** inert at `0` — which is this amendment's ground behaving as stated, not against it.
  The ruling above stands **because the build runs at 0.4**, and not because widening is impossible.
  Whether `λ` stays there is [#406](https://github.com/NGL321/patchworks/issues/406), open at the time
  of writing. If it moves, what this amendment is re-read against is two ceilings #188 recorded for
  exactly that case: **`m_e` is bounded above by the apex-side map's achievable effective rank**, which
  even at `λ = 0` is a construction-time fleet **3.66**
  ([#356](https://github.com/NGL321/patchworks/issues/356)), so anything past `m_e ≈ 4` is unusable at
  any `λ` this design has measured — a derived bound with a definition site, not an invented constant
  — and the `p_v` collision recorded under *The hatch, in rungs* below, which binds before it.

  **#406 has since closed, and `λ` did not move — it was deleted.**
  [#406](https://github.com/NGL321/patchworks/issues/406) removed the sparsity term from the transport
  rule entirely ([ADR-0031](./0031-the-sparsity-pressure-is-deleted.md)), so
  `DEFAULT_SPARSITY_PRESSURE` no longer exists and *"the operating point the build runs"* is now
  `λ = 0` by construction rather than by configuration. The clause above is therefore live rather than
  hypothetical: **the ruling's stated operating point is gone, and the re-read this amendment
  pre-specified is due.** Nothing here re-rules it — the two ceilings named above are exactly what the
  re-read runs against, and what the maps should be learning in the term's place is
  [#411](https://github.com/NGL321/patchworks/issues/411)'s. What is recorded is that the gate's
  subject no longer exists, so no later reader mistakes *inert at 0.4* for a standing finding about
  the build as it now runs.

  The usable knob therefore remains the **number of attachment points**, at the cost of a dimension of
  privacy per cell touched — spreading that cost instead of concentrating it, and never making one
  cell pay more than the minimum. It is gated too, in *The hatch, in rungs* below. The learned drive
  vector below remains the move after that one, and is gated in the same place.
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

  **Extended from a held-still pin to a schedule across a learning run**, on
  [#495](https://github.com/NGL321/patchworks/issues/495), 2026-09-04. The clause above was written
  with a *counterfactual* probe in view — run to the fixed point, hold the world still, re-run from the
  same state under an altered write, difference the settled node stalks, **rules off**. That is the
  licence `benchmarks/untrained_fixed_point.py` already exercises by name, and `prototypes/drive-reach/`
  with it. #495's probe is different in kind: it varies the assertion **while both local rules run**,
  across a whole learning run, so the write enters the evidence stream the transport rule learns from.
  The clause names no horizon and no rules-off condition, so it *could* have been read as already
  covering that. It is extended on purpose instead, because a generous reading discovered later is
  worth less than a stated one.

  **What carries the extension is the three-part test that tells a drive from a reward**, made here
  rather than assumed. A reward (a) enters the learning rule, (b) needs credit assignment, and (c)
  needs a satisfaction detector. **A task-blind schedule has none of the three.** It needs no credit
  assignment. Nothing reads whether the goal is met, so there is no detector —
  `info.goal_satisfied` stays privileged and for logging only, exactly as *Release needs no mechanism,
  and no detector* above already has it. And it enters the learning rule the way the render does, as
  **evidence**, rather than as a term modifying how an update is computed. What disqualifies a reward
  is the triple, not mere presence in the rule.

  **The condition is `exogenous drive`, and its content is task-blindness by construction.** The
  schedule is drawn from a seed *before* the run and reads **nothing**: not `info.goal_satisfied`, not
  prediction error, not `travel`, not the reconciliation residual. A metronome and a coach both make
  noise at you; only one is telling you whether you are doing well. It also never visits zero, since
  zero makes the drive inert and would periodically delete the dark-room answer — a different
  experiment wearing this one's clothes. **The bright line, so no later session re-derives it: a
  schedule contingent on satisfaction is [#5](https://github.com/NGL321/patchworks/issues/5)**, a
  second architecture rather than a channel swap, and not an instrument at all.

  **What the extension costs, on the record rather than absorbed.** The standing *no reward channel*
  inspection argument has a capacity-zero half — a constant carries no bits, so specification cannot
  ride the signal — and a schedule carries bits, so that half does not survive as stated. What survives
  is narrower, and is the claim to make: **the architecture's drive is a constant and therefore carries
  no bits**, while the probe is an instrument on a diagnostic run and not part of the build. A
  thermometer in the soup is not an ingredient. `DRIVE_ASSERTION` is **not** retuned — it stays `1.0`,
  which is what keeps this an instrument — and this is neither the ramp declined under *Known exposure*
  below nor a reopening of [#137](https://github.com/NGL321/patchworks/issues/137)'s fixing of the
  constant. A probe is not a ramp: the ramp was rejected as a **mechanism** that would supply
  something, and this supplies nothing.
- **Graph topology gains one cell and its mask entries** — a revision against `06-graph-topology.md`,
  small because it is an ordinary boundary cell, not a new category. **Done in
  [#36](https://github.com/NGL321/patchworks/issues/36):** the drive attaches at the **apex level,
  entire** — one edge to each of the eight L7 cells — with a **scalar** stalk and `m_e = 1`. The apex is
  the most abstract place in the graph and the only part of the core with private dimension to spare;
  the cost is one dimension of privacy per apex cell (16 → 15) and 8 off `χ`. **That dimension has
  since acquired a second buyer**: [ADR-0026](./0026-rim-core-influence-is-a-conduction-ratio.md) reads
  the operative bar's `τ̂` *in private features*, so `p_v` is the substrate the bar is measured on and
  not only a privacy budget. See the ladder's gate under *Known exposure*, where that is what makes
  rung 3 self-defeating rather than merely expensive.
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
  it is now pre-costed — and, at today's operating point, **answered**: see the ladder's gate below,
  where [#188](https://github.com/NGL321/patchworks/issues/188) closes both halves of it. The exposure
  itself is not thereby retired; the scalar is still unproven, and what is ruled is that widening is
  not the hatch out of that while `λ` stands where it does.

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

  **The ladder carries a dependency it was written without, and rungs 1 and 3 are both gated on it**
  ([#188](https://github.com/NGL321/patchworks/issues/188)). The gate is the transport rule's sparsity
  pressure at `λ = DEFAULT_SPARSITY_PRESSURE` — a gate on a **constant**, and
  [#393](https://github.com/NGL321/patchworks/issues/393) has since confirmed it is the constant doing
  the work. **[#406](https://github.com/NGL321/patchworks/issues/406) has since deleted that
  constant**, so both rungs' gate names something that no longer exists; see *Strength is not a new
  axis* above for what that does and does not move.

  - **Rung 1** already bore negatively on its own evidence:
    [#183](https://github.com/NGL321/patchworks/issues/183) measured coherent fan-out at **0.94x**
    across the eight drive edges that exist.
  - **Rung 3 is measured shut at today's `λ`.** The drive edges' apex-side maps sit at
    **1.02x–1.05x** the sparsity pressure's exact global minimum over three seeds at 30k ticks — mass
    share **0.9904–1.0000** in a single apex-stalk coordinate, Hoyer `h` **0.2470–0.2539** against a
    floor of `1/√17 = 0.2425` — where the interior maps on the same apex cells sit at
    **1.17x–1.20x** their own floor. `rank(F_apex) = 1` whatever `m` is, so the composition through a
    drive edge is one coordinate of the apex stalk and widening cannot widen what the drive
    constrains. See *Strength is not a new axis* above.

  This is a **gate, not a re-ordering**: #188 declined to re-price the rungs against each other on a
  single operating point.

  **Rung 3 is additionally self-defeating on the map route, and the pricing above predates the reason.**
  Apex cells are `n = 32` with `Σ_e m_e = 17`, so `p_v = 15`; widening the drive edge to `w` gives
  `p_v = 16 − w`. **At `k ≈ 16` the apex's private width reaches zero** — and
  [ADR-0026](./0026-rim-core-influence-is-a-conduction-ratio.md) reads `τ̂` **in private features**, so
  the operative bar would read **zero at the apex by construction**, adding the eight apex cells to the
  82 [#385](https://github.com/NGL321/patchworks/issues/385) is open about. This ADR records that rung's
  cost as *"one dimension of privacy per apex cell"* under *Graph topology gains one cell* above, which
  was written before the bar existed: on the map route the rung is not merely expensive, it is
  self-defeating against the bar it would be widened to clear.

  **The ramp is a fourth axis, and it is declined.**
  [#137](https://github.com/NGL321/patchworks/issues/137) asked whether the asserted scalar should stay
  constant or ramp under a ceiling, on this exposure's own premise that a scalar drive may prove too
  weak. It should not, and the ladder above is not missing a rung.
  [#183](https://github.com/NGL321/patchworks/issues/183) measured the axis: the drive **reaches the
  apex** at 0.100 per hop, settled from 30k to 100k ticks, a `1 -> 10` assertion landing **0.90** on an
  apex stalk. The assertion scales that deposit **linearly**, so a ramp buys **apex-local pressure and
  nothing else** — merely making the rim displacement representable in float32 would need the assertion
  scaled ~5.2e3. And it cannot reach this exposure's own trigger: an **undifferentiated apex** is a
  failure of *direction*, and with `m_e = 1` there is no direction to choose, so scaling moves all eight
  apex stalks in the same proportions and differentiates nothing. Coherent fan-out was measured at
  0.94x for the same reason. Width was the axis with leverage, and
  [#188](https://github.com/NGL321/patchworks/issues/188) has since resolved it: **no**, at
  `λ = DEFAULT_SPARSITY_PRESSURE`. The two halves of "width" were never one question. On the **map**
  route, `m_e > 1` is buildable and free of the contract cost this ADR priced — and inert, because the
  sparsity pressure has already taken the apex-side map to a single coordinate (*Strength is not a new
  axis* above, and the ladder's gate above). On the **stalk** route, `drive_stalk > 1` is refused
  outright, and on [`docs/motivating-image.md`](../motivating-image.md)'s own ground rather than on
  this ADR's: **the render supplies content**, so a wider drive stalk moves task specification out of
  the world and into the drive. [#393](https://github.com/NGL321/patchworks/issues/393) — the `λ`
  sweep — is the **reversal condition** for the map half, and was deliberately not made a blocker: the
  finding does not depend on `λ`'s future value, only on its current one. **#393 has since reported
  and the condition is met**; see *Strength is not a new axis* above for what that does and does not
  move, and [#406](https://github.com/NGL321/patchworks/issues/406) for whether `λ` stays where the
  ruling is stated. The stalk half is unaffected either way — it is refused on a ground that has
  nothing to do with `λ`.

  This is *Strength is not a new axis* above, restated where a reader will actually reach for it. It has
  a second job: **the ramp is the wrong answer to *Bootstrapping* below.** A drive edge that is noise
  early in training, with the arm not moving, is the one moment where cranking the assertion up looks
  like the fix. It is not — no motion at all is *Bootstrapping* and not this exposure, its answer is a
  **curiosity drive** (gated rather than owed, per *Bootstrapping* below), and that is a fog item
  rather than a rung. The constant stays at `1.0`, typed as
  **chosen** in the constants register with #183's linearity as its stated flexibility. No ceiling is
  derived and none is derivable today: `gamma x floor <` fold margin is under audit
  ([#158](https://github.com/NGL321/patchworks/issues/158),
  [#160](https://github.com/NGL321/patchworks/issues/160),
  [#178](https://github.com/NGL321/patchworks/issues/178),
  [#181](https://github.com/NGL321/patchworks/issues/181)), the drive's hop is a dimensionless gain
  against a margin that is a magnitude, and
  [#138](https://github.com/NGL321/patchworks/issues/138) retired folds as a mechanism. A constant needs
  no ceiling; nothing varies for one to bound.
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

  This exposure carries a second job it was not filed with, and the account it leaned on for that job
  is gone. It read: the PoC's whole account of exploration is that an unconverged model emits
  near-arbitrary torques and sharpens where it acted, so the window in which the drive edge is noise is
  the *same* window that account depends on. **That account is measured false.**
  [#120](https://github.com/NGL321/patchworks/issues/120) found the untrained emission to be **one
  world-independent constant** — sd ≤ 3.5e-6, identical to four decimal places across two different
  worlds at the same seed — and `04-action-and-the-boundary.md`, *Route selection* now carries that
  measurement in the account's place ([#154](https://github.com/NGL321/patchworks/issues/154)).

  **The exposure is not retracted by that; it is sharpened.** The window in which the drive edge is
  noise is still the window in which the agent has to keep moving, and untrained the agent does not
  move at all — which the old account merely predicted against and #120 has now observed.

  What #154 changes is the **standing of the response**. *Only one response addresses it — a curiosity
  drive*, an ordinary drive boundary cell at the internal rim, which is the fog item and not a rung on
  the ladder above; widening the task drive does not reach it, and neither does attaching it more
  widely. That stands, but **conditionally**. #154 rules that nothing must be added, conditional on the
  **outbound clause** of [#242](https://github.com/NGL321/patchworks/issues/242)'s influence predicate:
  the drive pressures action, action varies the world, and the supply of new directions is never short
  — but every link in that is the outbound leg, apex to actuator, which is what reads zero today. The
  excitation famine is therefore the outbound failure the map already owns. This exposure was right
  that the two must not be **conflated**, and the finding is not that they are the same failure but
  that they are **one failure and its symptom**.

  So the curiosity drive is **gated rather than owed**, and its gate is #154's pre-registered
  falsification: #242's outbound clause **passes** and the world still does not vary — the arm still
  locks, or per-edge excitation rank (the participation ratio read on the disagreement time-series)
  stays below that edge's stalk width `m_e`. Until that fires, this exposure names its response without
  claiming it is due.
- **Hallucinating satisfaction.** A core cell can reduce disagreement by *believing* the task is met
  rather than by acting — the failure GLean documents for goal-conditioned forward models
  (`docs/research/018-sandbox-citations.md`). Under a drive edge it is bounded rather than eliminated:
  the sensory edges pull the other way continuously, so the cell settles at a compromise, and that
  compromise is the prediction the motor rim must clear. It leaves an observable signature —
  sensory-side disagreement growing while the motor side stays quiet — which the demo can watch for.
