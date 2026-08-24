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

Two quantities, both live, neither of them behaviour.

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
[`03-the-sandbox.md`](./03-the-sandbox.md)'s *Per-joint gearing* deliberately gives the arm
heterogeneous joint timescales, under the constraint that they must not align with the graph's
levels. A demo that read recovery off a joint's decay would be reporting a mechanical time constant
and calling it a hierarchy. Onset cannot be supplied by the joint's mechanics, so it cannot be
confused for them.

## Pass and fail, pre-registered

Fixed **before** the live run, so that no result is narrated after the fact.

**Passing is two orderings.**

- **Depth ordering.** For each event, the private-component trace falls with hop distance, and the
  three events order: the arm nudge shallowest, the puck teleport intermediate, the retarget deepest.
- **Latency ordering.** `perturb` and `retarget` onset latencies have **non-overlapping interquartile
  ranges** over N repeated runs. N and the seeds belong to
  [The evaluation protocol](https://github.com/NGL321/patchworks/issues/23).

The latency claim is confined to the two hands, and the confinement is deliberate on two counts.
It is what [#30](https://github.com/NGL321/patchworks/issues/30) handed down. And the arm nudge's
onset — reported, and expected near three ticks — is left out of the load-bearing claim so that the
shallow rung cannot be attacked as a mechanical artefact even if the gearing measurement comes back
wide.

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

## Two secondary runs

Both run, both reported, whichever way they come out. Suppressing either would make the demo a
highlight reel.

- **Crossing.** The headline, with `perturb()` placing the puck across the arena so that repositioning
  forces a swing-direction choice through the annulus
  ([#25](https://github.com/NGL321/patchworks/issues/25)). This is where the stall signature lives,
  and it bites in this testbed rather than a later one.
- **Blocked.** The headline, in a **route-blocking layout** — a non-target puck across the target's
  route, precedence depth 2 ([`03-the-sandbox.md`](./03-the-sandbox.md), *Route-blocking layouts*).
  Perturbation recovery and a forced ordering in the same run.

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
- **A hand-made detour** — teleporting a non-target puck into the path mid-task. Bought instead by
  *Route-blocking layouts*, and bought better: a teleported blocker announces itself, a sampled one
  was always there, and the sampled version is what the *Blocked* run above already exercises.
