# The acceptance demo

The destination's acceptance criterion: *a live interaction in which a human perturbs the world
mid-task and the agent recovers at the appropriate level of its hierarchy.*

This file owns the demo. Until now it lived in four places — the hands in
[`03-the-sandbox.md`](./03-the-sandbox.md), the two-depths criterion in
[`06-graph-topology.md`](./06-graph-topology.md), the private-component readout in
[`05-timescales.md`](./05-timescales.md), the drive's release behaviour in
[`04-action-and-the-boundary.md`](./04-action-and-the-boundary.md). Those files keep their
mechanisms and point here for the protocol.

The demo is not the architecture and not the environment. It exists inside the environment without
being it, and it tests the architecture without being part of it.

## The run

**Three events, one unbroken run, ascending in the depth each one reaches.** That ascent is the
run's shape and its narrative; it is **reported and not claimed on** — see *Pass and fail*, below,
for why an ordering supplied by the injection site cannot test the graph. The agent is mid-task;
nothing is reset
between events, because there is nothing to reset — physics time is monotonic and there are no
episodes ([`03-the-sandbox.md`](./03-the-sandbox.md), *The Gymnasium contract, made continual*).

1. **Nudge the arm.** `disturb_arm(joint, impulse)` — an impulse to one joint, a ctrl-drag on a link
   in the live viewer. The world moves the body; nothing about the task changes.
2. **Teleport the puck.** `perturb(puck, xy)` on the *target* puck. The world changes; what is wanted
   does not.
3. **Retarget to the puck just moved.** `retarget(goal_puck, goal_zone)`, naming the same puck the
   previous event displaced. What is wanted changes; the world does not.

Events 2 and 3 share a referent deliberately: the viewer watches the same object addressed twice, at
two different levels, seconds apart. To a non-expert the whole run reads *you messed with it three
times, harder each time, and it figured it out each time.*

**The headline run uses a clean layout and does not cross the annulus.** Both of those are difficulty
dials, and both get their own run below. Turning them up in the headline would make a failure
uninterpretable — the demo would be answering two questions with one number.

## What is measured

Two quantities, both live, neither of them behaviour. How they are *displayed* —
and how each of the near-misses below is told apart on screen — belongs to
[`10-the-demo-surface.md`](./10-the-demo-surface.md).

**Depth.** The **conduction ratio**, per cell: the e-fold decay time `τ̂_c` of the paired
private-feature deviation, over `|loop(c)|` — the tick length of the shortest cycle through `c` that
reaches the rim and returns. Both halves are in ticks, so the ratio is dimensionless. The quantity,
its derivation and `|loop(c)|`'s enumeration on the default dome are
[ADR-0026](../adr/0026-rim-core-influence-is-a-conduction-ratio.md)'s and are not restated here; what
this file fixes is what the *demo* does with them, which is
[ADR-0027](../adr/0027-the-demos-depth-criterion-is-a-conduction-time.md)'s.

Behaviour alone is not accepted as evidence and never has been — a purely reflexive controller
produces the same footage.

This **replaces** `05-timescales.md`'s `‖Δ(private component)‖`-against-hop-distance reading as the
depth *criterion*. The amplitude picture is kept on the demo surface, because a scatter is what a
viewer can read; it is no longer what passing means. The reason is registered below under *The
graph's own attenuation with depth*, and it is that the old clause could not fail.

**The population is L1 predicting cells, and the partition is forced rather than chosen.** A boundary
cell is not masked (`graph.py:660`) — its stalk is world-shaped rather than `n`-shaped — so it has no
`H⁰` and cannot carry a conduction ratio at all. The demo therefore cannot re-borrow ADR-0026's
outbound universal verbatim, which includes the actuator. But the demo already reads the actuator:
that is what onset latency *is*, and ADR-0026 calls it *"the acceptance demo's own instrument."* So
the depth measure runs over **L1 predicting cells** and the **actuator is read by onset latency**.
The two halves partition the graph with no overlap and no gap, and **this is why the demo has two
measures and not one**.

**Two senses of "decay time", and they must not be confused.** *Onset latency*, below, is guarded
*never as settling or decay time* — that guard is aimed at the **body's mechanics**, at
[`03-the-sandbox.md`](./03-the-sandbox.md)'s 17.9x ladder in passive joint decay. `τ̂_c` is a decay
time on **private features inside the graph**, which no joint can supply, and the ladder is
deliberately built not to align with the graph's levels. The misalignment guard, built for the old
clause, earns its keep a second time here. The guard on onset latency is untouched.

**Onset latency.** Ticks from the event to the **first corrective torque**. Not settling time, and
this is the load-bearing distinction in the whole protocol.

Settling time is dominated by physics: how long the arm takes to stop ringing after an impulse, or to
cross the arena after a retarget. Those are properties of the body and the geometry.
Onset is the quantity the depth claim actually predicts — [`06-graph-topology.md`](./06-graph-topology.md)
puts the somatomotor reflex loop at **three ticks**, a correction needing visual context at four hops
out and back, and a drive asserted at the apex further still. Measuring onset makes the latency
ordering a claim about the graph at every rung.

It also keeps the demo clear of the ladder in the body.
[`03-the-sandbox.md`](./03-the-sandbox.md)'s *A timescale ladder in the body* deliberately gives the
arm heterogeneous joint timescales — a 17.9x spread in passive decay, bought with rotor inertia —
under the constraint that they must not align with the graph's levels. A demo that read recovery off a joint's decay would be reporting a mechanical time constant
and calling it a hierarchy. Onset cannot be supplied by the joint's mechanics, so it cannot be
confused for them.

## The nudge's impulse is not sized, and #214 has since closed the question it left

`disturb_arm(joint, impulse)` leaves the impulse a **free parameter**, deliberately — this file fixes
a protocol, not a magnitude, and onset latency is behavioural and reports nothing about how large the
arriving deviation was.

[ADR-0021](../adr/0021-rim-to-core-detectability-is-a-bottleneck-ratio.md) needed an amplitude
convention for the rim perturbation and **created one rather than adopting this parameter**: a
unit-norm deviation on the L1 boundary stalk, `A₀ = 1`. It refused to weld a claim about the graph to
a demo parameter, because doing so re-opens the sufficiency verdict every time the parameter moves.

The residue lands here, and it was a real question: **is the impulse this demo actually fires large
enough for the disturbance to be detectable where the demo claims it acts?** ADR-0021's
pre-registered arithmetic gave the shape of the answer — at ~0.2 per hop, each 5x of rim amplitude
buys roughly one more hop — and the number that matters was always going to be measured.

**[#214](https://github.com/NGL321/patchworks/issues/214) has since delivered, and the answer is
no.** Read along the channel in float64, both directions fail: rim→apex at a median bottleneck ratio
of **8.7e-10**, apex→rim at **1.3e-8**, with the hops graded **9x-240x**. The shortfall is 1.15e9x,
so no impulse this arena can deliver closes it — *each 5x buys one hop* prices the fix at more rim
amplitude than the world has. **On the amplitude reading, the disturbance does not arrive where the
demo claimed it acted, at any impulse.**

What that settles and what it does not:

- **It settles the free parameter as a non-question for this protocol.** There is no size at which
  the old depth clause becomes an honest test, which is one of the reasons the criterion above is no
  longer an amplitude.
- **The pass conditions below are unaffected by impulse size, and now for a stronger reason than
  before.** They were orderings, and an ordering does not depend on absolute size. The depth clause
  is now a **ratio of times**, which is scale-free outright: `τ̂_c / |loop(c)|` has no amplitude in
  it. The impulse must still be large enough to produce a deviation the paired fork can see at all —
  the falsification named in ADR-0026 is a deviation *bit-identical* between branches — and above
  that floor it does not enter the criterion.
- **The event stays exactly as specified.** Nothing in the protocol changes on this account.

## Pass and fail, pre-registered

Fixed **before** the live run, so that no result is narrated after the fact.

**Passing is one closure and one ordering.**

- **Depth: the event's loop closes.** For each event, `τ̂_c / |loop(c)| ≥ 1` along **some path from
  that event's injection site** — read **inbound** for the arm nudge (entering at proprioception and
  touch) and the puck teleport (entering at vision), **outbound** for the retarget, which is injected
  at the drive boundary cell attached to the apex. **Single-source, not a sweep**: the demo pokes the
  sources a human actually pokes, and ADR-0026's swept per-stratum read stays where it lives, on
  [`benchmarks/detectability.py`](../../benchmarks/detectability.py). Reported as the **median over
  the 40 paired trials**, with p05 / p25 / p75 / p95 alongside — ADR-0021's precedent and
  [#202](https://github.com/NGL321/patchworks/issues/202)'s reason for it.
- **Latency ordering.** `perturb` and `retarget` onset latencies have **non-overlapping interquartile
  ranges** over the 40 paired trials specified in *The repeated runs*, below.

**What a latency PASS establishes, and what it does not.** A PASS here establishes that **a
correction travelled a longer path, not that a hierarchy produced it** — on this graph every edge
costs one tick, so an onset ordering is a restatement of hop count (see *Unit edge delay*, below).
**The hierarchy claim rests on the depth clause.** Onset latency remains the temporal measure and the
settling-time distinction is untouched; only the reading of a pass is corrected.

**The between-event ordering is reported, not claimed on.** The file previously ordered the three
events *arm nudge shallowest, puck teleport intermediate, retarget deepest* and made it load-bearing.
That clause leaves the pass condition on two grounds, the second the stronger. **Both ends are
supplied by the injection site** — `retarget()` writes the drive boundary cell at the apex,
`disturb_arm` enters at the somatomotor rim — so neither needs the graph to do anything, in a world of
any depth; that is *What this demo does not show*, below, applied to the clause that was resting on
it. And **the ordering was never accurate as a description**: the events do not differ in *which*
level they address, because there is information modification at every level each one passes through.
What differs is the **deepest level each one reaches**. Under the conduction reading that is a
measured quantity rather than a stipulated one — **how deep an event reaches is the length of the
path along which the ratio holds** — so it is reported, and it is the same observation with the graph
put back in it. The precedent is in this file already: the arm nudge's onset is reported and not
claimed on, for the reason given below.

The latency claim is confined to the two hands, and the confinement is deliberate on two counts.
It is what [#30](https://github.com/NGL321/patchworks/issues/30) handed down. And the arm nudge's
onset — reported, and expected near three ticks — is left out of the load-bearing claim so that the
shallow rung cannot be attacked as a mechanical artefact. That guard has since earned its keep: the
ladder measurement did come back wide, at 17.9x
([#60](https://github.com/NGL321/patchworks/issues/60)).

**On the latency half, nothing sharper than non-overlapping IQRs is claimed**, because nothing
sharper is available. A cell's timescale is a distribution and the taper's gradient is a gradient in
**means**, with adjacent depths overlapping per tick
([#41](https://github.com/NGL321/patchworks/issues/41)). A ratio threshold there would be a number
invented before anything was trained. So: **the live run demonstrates, the repeated runs establish.**

**And the depth half does carry a threshold, which is admissible for a reason worth stating.** The
sentence above is about the latency ordering and about **invented** constants. The depth bar is `1`,
and `1` is the loop's own length in ticks — the cell still holds what it sent by the time the answer
gets back, and nothing more. It is **derived**, in the way ADR-0021 derived `k = 1` and in the way
[#142](https://github.com/NGL321/patchworks/issues/142)'s inherited ~0.37/hop was **not**, which is
why that one was struck. There is no multiplier and no safety factor. Stated here because, left
unsaid, a reader arriving at a threshold two paragraphs after a refusal of thresholds will think the
rule was forgotten.

**Failing.**

- **A loop that does not close is a failure**, including the case where every recovery looks
  perfect. If no path from the event's injection site holds `τ̂_c / |loop(c)| ≥ 1`, the graph does not
  hold what it sent long enough for the answer to get back, and no amount of convincing footage
  changes that. **This is the criterion that fails today, and it fails on every reading of `τ` the
  record holds.** On the chart's **direct** round trip `τ` is flat at about one tick graph-wide —
  0.91 at the apex against 0.99 at the rim, no depth→timescale gradient and slightly inverted. With
  the stalk relay included ([#274](https://github.com/NGL321/patchworks/issues/274), nine driven seeds) the
  apex's `τ` is **1.6 to 13.1 ticks** and the inversion is *larger*, the apex decaying faster than
  the rim rather than slower.
  Against `|loop|` of 2 at L1 and 14 at the apex, that is a ratio of **0.12 to 0.93 at the apex** —
  short on every seed, which is why the criterion's verdict is unchanged and only its magnitude
  moves. Both figures are read on `05`'s regional `τ` and **neither is `τ̂_c`**, the paired
  counterfactual decay this criterion is actually written over; they are stand-ins, the corrected
  one is the better stand-in, and #99 owes the real instrument
  ([`05-timescales.md`](./05-timescales.md), *What the live read says*). Pre-registering a condition
  the architecture currently fails is the point of pre-registering it.
- **A private-feature deviation bit-identical between the paired branches is a failure**, and the
  starkest one: no counterfactual dependence at all, so there is nothing to time. ADR-0026 names this
  as the falsification and it is inherited here unchanged.
- **Overlapping latency IQRs is a failure**, for the same reason: the demo has not shown two levels,
  whatever the footage looks like. This half can still fail cleanly — nothing transmitting means no
  corrective torque, both hands record the ceiling, the IQRs overlap.

Three near-misses are named in advance, because each of them produces convincing footage:

- **Restart.** The arm returns to a home pose and re-approaches. A reflex controller does this and it
  looks like recovery.
- **Stall.** The arm stops mid-swing. This is [#25](https://github.com/NGL321/patchworks/issues/25)'s
  annulus signature — the blend of left and right is *stay put* — and it is a predicted failure with
  a known cause, not a surprise.
- **Task-invariant behaviour.** The trajectory is the same across tasks differing only in the render,
  while the drive edge's disagreement is non-trivial. This is
  [#50](https://github.com/NGL321/patchworks/issues/50)'s trigger and it **falsifies the drive, not
  the demo** — the scalar valence channel is what failed. Labelled as such wherever it is reported,
  and it has its own ladder of responses in
  [ADR-0009](../adr/0009-a-drive-is-a-motor-edge-attached-deep.md).

## Confounds, registered in advance

A confound is not a near-miss. A near-miss is behaviour that *looks* like a pass; a confound is a
mechanism that would **supply** a pass without the architecture doing the work. Registered here so
that a reader can check the criteria above against them rather than take the criteria on trust.

- **The graph's own attenuation with depth.** The largest one, and until now this file had no entry
  for it. `‖Δ(private component)‖` falls with hop distance whether or not anything is being retained,
  because the channel attenuates: [#214](https://github.com/NGL321/patchworks/issues/214) measured
  **8.7e-10** rim→apex with hops graded **9x-240x**, so a graph transmitting nothing renders as one
  bright rim row above a flat baseline. **That made the old depth clause unfailable**, and it is why
  the criterion now reads a time. The principle was already this spec's own —
  [`10-the-demo-surface.md`](./10-the-demo-surface.md) refuses to drive the trail from
  `‖Δ private‖` because *"that would make the display's decay and the claim the display tests the
  same number, so the panel could never contradict the thesis."* #214 made the graph's attenuation
  and the claim the same number; only the application was missing, because this file and `05` predate
  the measurement. **The conduction ratio is immune**: attenuation is an amplitude and `τ̂ / |loop|`
  is a ratio of times, so shrinking the deviation does not lengthen its decay.
- **Unit edge delay.** Every edge costs exactly one tick
  ([`01-cell-and-sheaf.md`](./01-cell-and-sheaf.md), *Unit delay*), so on this graph **an onset
  ordering is a restatement of hop count**, and no guard separates hierarchy from delay — because
  delay *is* hop count here. **There is no fix available on this graph**, which is why this is an
  entry and not a repair: it is the reason a latency PASS is read as *a correction travelled a longer
  path* and not as *a hierarchy produced it*.
- **The body's timescale ladder.** `03`'s arm carries a 17.9x spread in passive joint decay
  ([#60](https://github.com/NGL321/patchworks/issues/60)), which a recovery time read off a joint
  would report as a hierarchy. Guarded twice over: the ladder is deliberately built **not to align
  with the graph's levels**, and neither measure is a body quantity — onset is *before* the mechanics
  act, and `τ̂_c` is on private features inside the graph, which no joint can supply.
- **Behaviour alone.** A purely reflexive controller produces the same footage. Refused as evidence
  throughout, which is why neither measured quantity is behavioural.
- **Agent drift across the sample.** Weights never freeze, so an agent sampled for one hand and then
  the other is not the same agent, and the drift could manufacture the very latency ordering being
  claimed. Guarded by the pairing at each snapshot; see *The repeated runs*.

## The repeated runs

*The live run demonstrates, the repeated runs establish.* This section fixes what a repeat **is**,
because in a world with no episodes it is not obvious, and because a trial definition chosen after
the data exists is the thing pre-registration exists to prevent.

**A trial starts from a restore, not a reset.** There is no episode boundary to start from and
`reset()` deliberately never touches the agent, so the only defined start available is
`03-the-sandbox.md`'s snapshot/restore — which is not a reset in disguise: it rewinds the agent's
adapting surface along with the world and is invisible from inside. The cited reset-free literature
all reintroduces a start distribution for evaluation specifically and says so; this is that move,
made without an in-band reset the agent could learn to anticipate.

**Structure: 40 snapshots down one continual run, three events at each.** The snapshot ticks are
spaced across the run rather than clustered, and at each one the state is restored three times — once
for `disturb_arm`, once for `perturb`, once for `retarget`.

The pairing is the load-bearing part. Weights never freeze, so an agent sampled for one hand and
then for the other is not the same agent, and the drift between the two sample sets could
**manufacture the very latency ordering being claimed**. Paired at each snapshot, both hands see an
identical agent at an identical moment, and any ordering between them can only come from depth. The
arm nudge rides along free on the same restore; it is reported, not claimed on, for the reason given
above.

**What varies, and why it has to.** MuJoCo is deterministic, so a fixed snapshot with a fixed event
yields a single number rather than a distribution. Spread comes from two sources deliberately, and
both belong in it: the **agent's drift** across the 40 snapshot points, and the **event's
parameters** — where `perturb()` teleports the puck to, which held-out pair `retarget()` names.

An interval was rejected here. Firing events down one continuous run without restoring is more
faithful to the reset-free contract, but the agent adapts to being poked: onset latency shrinks
across repeats and the IQR ends up measuring learning rather than depth.

**Retarget names a held-out pair.** The event draws from the **pair** axis of
[`03-the-sandbox.md`](./03-the-sandbox.md)'s held-out slice, which is what gives that slice a reader
at all and puts the compositional claim inside the pre-registered pass rather than beside it. The
pair axis specifically: it withholds compounds while leaving every puck, zone and region seen, so a
recovery there is attributable to composition and not to a position the agent has never been shown.
The sector axis stays out of the load-bearing claim for exactly that reason.

**Two seeds, fixed before the run**: the snapshot schedule (the 40 tick indices) and a per-trial
event seed. Reproducibility of a trial once started is snapshot/restore's job, not a seed's.

### What disqualifies a snapshot

Pre-registered, because discarding trials after the fact is narration.

- **Satisfied at the snapshot tick.** The goal is already met, so there is no task to be mid-way
  through.
- **Not engaged.** No torque directed at the target puck in the preceding **`K` = 25 ticks**
  (0.5 s at `03-the-sandbox.md`'s 50 Hz control rate).

**`K`'s warrant, and it is stipulated long.** The number is a stipulation, not a derivation; what a
warrant owes is the reason for *this* stipulation rather than a neighbouring one. `K` has to exceed
the longest quiet interval an agent can leave while still being engaged, and this graph fixes what
that is: every edge costs exactly one tick (*Unit edge delay*, above), so the deepest correction the
graph admits is the apex's own round trip, `|loop| = 14` ticks (*Failing*, above). A `K` below that
disqualifies snapshots for the silence a deep correction spends **in transit** — deleting precisely
the trials the demo exists to collect, and biasing what survives toward the shallow rung. 25 ticks
clears 14 with room, and it is not a fresh number: it is the same half second
[`03-the-sandbox.md`](./03-the-sandbox.md) already spends deciding that a goal is *held* rather than
momentarily touched. That is this project's existing convention for *long enough that an instant is
not mistaken for a state*, which is the same question `K` asks.

**Erring long is the cheap direction, and the pairing is why.** A short `K` deletes deep trials
selectively, which is a bias with a direction. A long `K` admits the occasional idling snapshot,
whose onset latency is inflated by however long the agent takes to re-engage — but the snapshot is
restored three times and both hands meet that same idling agent at that same tick, so the inflation
lands on `perturb` and `retarget` **together**. It can widen the two distributions; it cannot
separate them, and the separation is the entire latency result.

**A stalled agent is a valid trial and its latency counts.** Stall is a *predicted* failure with a
known cause — #25's annulus signature, the blend of left and right being *stay put* — and
disqualifying it would quietly delete the demo's most likely honest negative. It is named in *Pass
and fail* as a near-miss precisely so it can be recognised, not so it can be dropped.

For the same reason there is a **ceiling, not a discard**: if no corrective torque arrives within
**100 ticks (2 s)** of the event, the trial records 100 and enters the distribution at that value.
Failures leave the distribution nowhere.

**The ceiling's warrant, and it is set from the failure side.** This number is load-bearing on the
**honest negative** specifically, which is what decides how it is chosen. The trials it exists to
keep are the ones that do not recover: the stalled agent above, whose latency counts; and the case
this file already anticipates in *Failing* — nothing transmitting, no corrective torque, **both
hands record the ceiling and the IQRs overlap**. That failure mode only lands as a clean overlap
because the ceiling gives a non-recovery a number to be. There is a floor beneath it that is not
this protocol's to set: below ADR-0026's bit-identical deviation the paired branches differ nowhere
and there is nothing to time at any window (*the impulse must still be large enough to produce a
deviation the paired fork can see at all*, above). The ceiling handles the case above that floor
where a deviation exists and no correction follows it.

So the window is not sized to sit tightly around the expected onsets; it is sized so that reaching
it **means** non-recovery. At 100 ticks it stands at about **7x** the apex round trip of 14 and 4x
`K`, wide enough that a censored trial reads as a correction that did not come rather than one
truncated on its way. Note what is ceilinged: **onset**, the first corrective torque — expected near
three ticks at the shallow rung — and not settling time. Two seconds is generous against that
quantity; it is not a budget for crossing the arena.

**Censoring is reported, and it cannot be what carries the result.** A ceiling value sits above
every uncensored one, so ceilings accumulating in one hand push that hand's IQR up on their own. The
count of ceilinged trials is therefore reported **per hand, beside the IQRs**, and: **if dropping
the ceilinged trials from both hands collapses the non-overlap, the latency half has not passed.**
A separation carried by censoring would be reporting how often the agent fails to recover, not how
deep the correction that recovered it was. Pre-registered here rather than decided at the run, for
the reason the whole section exists.

**Nothing here is aggregated into a score.** Goal satisfaction gates whether a trial is valid and
that is its whole job; no success rate is computed, over splits or otherwise. The map rules degree
of success out of scope, and the loop closure and the latency ordering above are the entire result.

## One secondary run

Run and reported whichever way it comes out. Suppressing it would make the demo a highlight reel.

- **Crossing.** The headline, with `perturb()` placing the puck across the arena so that repositioning
  forces a swing-direction choice through the annulus
  ([#25](https://github.com/NGL321/patchworks/issues/25)). This is where the stall signature lives,
  and it bites in this testbed rather than a later one.

A third run — the headline in a **route-blocking layout**, for perturbation recovery and a forced
ordering together — was specified here and is **withdrawn**. Route-blocking layouts were built and
measured inert ([#60](https://github.com/NGL321/patchworks/issues/60);
[`03-the-sandbox.md`](./03-the-sandbox.md), *Route-blocking layouts*), so the run would have
reported a forced ordering that the world does not impose.

## What this demo does not show

Two levels out of eight, and both supplied by the **interface** rather than discovered by the graph.
`perturb()` and `retarget()` hit two different levels by construction, in a world of any depth. That
is the honest limit, and it is [#30](https://github.com/NGL321/patchworks/issues/30)'s: a demo can
prevent a false success, but it cannot manufacture depth. The world's thinness is stated where it
belongs, in [`03-the-sandbox.md`](./03-the-sandbox.md)'s *What this sandbox does not exercise*, and
what guards the claim there is the falsification sweep, not this protocol.

**Ruled out as demo material, with reasons, so neither is re-proposed:**

- **Structural damage** — knocking over a half-built result. Nothing is built in this world. Its
  nearest translation is teleporting the target puck back out of its zone, which is event 2 wearing a
  hat: mechanically identical, and to a viewer a repeat rather than a new claim.
- **A hand-made detour** — teleporting a non-target puck into the path mid-task. Originally ruled out
  as redundant against *Route-blocking layouts*; now ruled out on the stronger ground that killed
  those. A blocker does not obstruct anything in this arena — nothing here is concave, so every
  blocker is displaceable and the straight route was never the route
  ([#60](https://github.com/NGL321/patchworks/issues/60)). A teleported one would be theatre, and
  would announce itself besides.
