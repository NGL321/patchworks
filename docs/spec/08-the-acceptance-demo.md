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

**Three events, one unbroken run, ascending in depth.** The agent is mid-task; nothing is reset
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

**Depth.** [`05-timescales.md`](./05-timescales.md)'s private-component readout: `‖Δ(private
component)‖` per cell against hop distance from the sensorimotor rim, displayed during each event.
Behaviour alone is not accepted as evidence and never has been — a purely reflexive controller
produces the same footage.

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

## The nudge's impulse is not sized, and that is an open question

`disturb_arm(joint, impulse)` leaves the impulse a **free parameter**, deliberately — this file fixes
a protocol, not a magnitude, and onset latency is behavioural and reports nothing about how large the
arriving deviation was.

[ADR-0021](../adr/0021-rim-to-core-detectability-is-a-bottleneck-ratio.md) needed an amplitude
convention for the rim perturbation and **created one rather than adopting this parameter**: a
unit-norm deviation on the L1 boundary stalk, `A₀ = 1`. It refused to weld a claim about the graph to
a demo parameter, because doing so re-opens the sufficiency verdict every time the parameter moves.

The residue lands here, and it is a real question with no answer yet: **is the impulse this demo
actually fires large enough for the disturbance to be detectable where the demo claims it acts?**
ADR-0021's pre-registered arithmetic gives the shape of the answer — at ~0.2 per hop, each 5x of rim
amplitude buys roughly one more hop — but the number that matters is measured, not assumed, and the
measurement is [#214](https://github.com/NGL321/patchworks/issues/214)'s. Until it is read, no
statement in this file about where a disturbance reaches is doing more than naming the level the
*interface* addressed.

Nothing in the protocol changes on this account. The event stays as specified, and the pass and fail
conditions below are unaffected — they are orderings, and an ordering does not depend on the
impulse's absolute size.

## Pass and fail, pre-registered

Fixed **before** the live run, so that no result is narrated after the fact.

**Passing is two orderings.**

- **Depth ordering.** For each event, the private-component trace falls with hop distance, and the
  three events order: the arm nudge shallowest, the puck teleport intermediate, the retarget deepest.
- **Latency ordering.** `perturb` and `retarget` onset latencies have **non-overlapping interquartile
  ranges** over the 40 paired trials specified in *The repeated runs*, below.

The latency claim is confined to the two hands, and the confinement is deliberate on two counts.
It is what [#30](https://github.com/NGL321/patchworks/issues/30) handed down. And the arm nudge's
onset — reported, and expected near three ticks — is left out of the load-bearing claim so that the
shallow rung cannot be attacked as a mechanical artefact. That guard has since earned its keep: the
ladder measurement did come back wide, at 17.9x
([#60](https://github.com/NGL321/patchworks/issues/60)).

Nothing sharper than non-overlapping IQRs is claimed, because nothing sharper is available. A cell's
timescale is a distribution and the taper's gradient is a gradient in **means**, with adjacent depths
overlapping per tick ([#41](https://github.com/NGL321/patchworks/issues/41)). A ratio threshold would
be a number invented before anything was trained. So: **the live run demonstrates, the repeated runs
establish.**

**Failing.**

- **Wrong depth is a failure**, including the case where every recovery looks perfect. If the traces
  are flat across hop distance — deep private state swinging as far as the rim — the mechanism is not
  working, and that is visible in the moment.
- **Overlapping latency IQRs is a failure**, for the same reason: the demo has not shown two levels,
  whatever the footage looks like.

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
- **Not engaged.** No torque directed at the target puck in the preceding K ticks, K fixed in
  advance.

**A stalled agent is a valid trial and its latency counts.** Stall is a *predicted* failure with a
known cause — #25's annulus signature, the blend of left and right being *stay put* — and
disqualifying it would quietly delete the demo's most likely honest negative. It is named in *Pass
and fail* as a near-miss precisely so it can be recognised, not so it can be dropped.

For the same reason there is a **ceiling, not a discard**: if no corrective torque arrives within a
fixed window, the trial records the ceiling. Failures leave the distribution nowhere.

**Nothing here is aggregated into a score.** Goal satisfaction gates whether a trial is valid and
that is its whole job; no success rate is computed, over splits or otherwise. The map rules degree
of success out of scope, and the two orderings above are the entire result.

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
