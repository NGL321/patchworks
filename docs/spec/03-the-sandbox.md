# The sandbox

The world the agent lives in, and the surfaces through which it meets that world. Independent of
the architecture: nothing here assumes a sheaf, a cell, or a local learning rule.

Terms used here are defined in [`CONTEXT.md`](../../CONTEXT.md).

Every number in this section was chosen by building the thing and watching it, not by argument.
See [issue #3](https://github.com/NGL321/patchworks/issues/3) for what broke on the way.

## Planar by construction

The world is planar because **gravity is zero and every joint is a hinge about `z` or a slide in
`x`/`y`** — not because objects rest on a floor. Table friction is modelled as joint frictionloss
(`μmg`, with `μ ≈ 0.3`), not as contact with a supporting surface.

This is what makes the dimensionality commitment real. Going 3D means *adding joints and turning
gravity on* — removing constraints — rather than porting an engine. Nothing in the observation
contract, the action contract, or the cell I/O shapes names a plane; they name joints and a camera.

## The body

A **3-link planar arm**, torque-controlled, mounted at the centre of the arena.

| | Link 0 | Link 1 | Link 2 |
|---|---|---|---|
| length (m) | 0.20 | 0.16 | 0.10 |
| torque limit (N·m) | 3.0 | 2.0 | 1.0 |
| joint range (rad) | ±π | ±2.6 | ±2.6 |

Reach is 0.46 m to the tip site; the tip is a **rounded paddle** (a cylinder of radius 0.03), so
the arm's outer extent is 0.49 m. The paddle is the only part of the body intended to meet the
world.

Three links rather than two: the redundancy means posture carries information beyond tip position,
which is a real thing for the agent to model. A gripper was rejected — grasping adds contact
physics to debug without adding to the argument, and pushing already composes.

## The world

A **circular arena** bounded by a ring wall at radius 0.52 m, containing **three sliding pucks** and
**three target zones** (radius 0.075 m) at radius 0.30 m.

| | Puck 0 | Puck 1 | Puck 2 |
|---|---|---|---|
| radius (m) | 0.035 | 0.045 | 0.055 |
| mass (kg) | 0.05 | 0.10 | 0.20 |
| colour | red | green | blue |

Pucks differ in size and mass, so *which* puck is being pushed changes what happens —
the dynamics are not one model with a colour attached.

Two radii are load-bearing and were both found by watching the thing fail:

- **The ring wall sits at 0.52, outside the arm's 0.49 extent.** The arm therefore never touches
  the boundary, while a puck pinned against it still has its centre inside the workspace. A square
  arena was tried first and was wrong: its corners sat at radius 0.76, so a puck pushed into one
  was gone for good.
- **The pedestal at the centre is a real obstacle**, radius 0.08. The arm is singular at its own
  shoulder and has no leverage there; without the pedestal, a puck nudged toward the centre reaches
  a place where the push-line standoff point lies inside the base and no push can recover it. The
  pedestal walls that region off. It also makes the workspace **non-convex**.

  The non-convexity matters, but **not** in the way first claimed. Measured over the sampler
  (`prototypes/route-geometry`), the pedestal is a graze: 37–41% of tasks have a straight puck→zone
  line that clips it, but going around costs a **median 4%** extra distance (max 17%), and the
  failure the pedestal was added to fix — the standoff point inside the base — now fires on **1–2%**
  of tasks. Routing the *puck* around it is real but shallow.

  What is not shallow is what the pedestal does to the **arm**: links 1 and 2 collide with it, so the
  paddle's reachable set is an **annulus** (inner 0.11, outer 0.49) rather than a disk. That is where
  the workspace's topology actually bites, and it bites several times per task — see
  [`04-action-and-the-boundary.md`](./04-action-and-the-boundary.md), *Route selection*.

Pucks spawn in the annulus **0.15–0.36 m**, clear of each other, of the zones, and of the arm's
current configuration.

## Sensory surface

```
qpos    (3,)            joint angles
qvel    (3,)            joint velocities
touch   (3,)            contact force, one sensor per link
image   (64, 64, 3)     top-down render, uint8
```

**No object pose is ever given to the agent.** Everything about the world arrives through the
render, unlabelled: the agent must learn that the coloured blobs are objects, that they have
different masses, and how they respond to being hit.

**The goal is perception, not a scalar.** The target zone lights up in the render. There is no goal
vector, no task id, and no reward. Retargeting mid-task is therefore just a change in the world's
appearance.

This is what lets the agent's drive carry **valence without specification**
([ADR-0009](../adr/0009-a-drive-is-a-motor-edge-attached-deep.md)): because the render already says
*which* puck and *which* zone, the only thing left for a drive to assert is *satisfied*. The two
statements are not in tension — the goal's **content** is perception, and the discomfort at not having
reached it is the scalar.

### Known exposure: what 64×64 does and does not resolve

Each puck carries an orientation marker. **At 64×64 the marker is not resolvable**, so puck
rotation is a genuine hidden variable — inferable only from contact dynamics, never read off. This
is recorded as a consequence, not a decision: raising the resolution would change it.

### Dynamics exploration, not spatial exploration

The whole arena is visible in every frame, so **spatial** exploration is not forced — and that is a
decision, not a leftover. The destination's *a sandbox it must explore to model* is honoured by
**dynamics exploration**: mass and contact outcome are nowhere in the render and are recoverable only
by acting. An agent that has looked at this arena and never touched it knows almost nothing that
predicts the next tick; the heaviest puck does not move at all below ~2 N at the tip (*Motor
surface*), so even *whether a push works* is knowledge only action buys.

The argument deliberately rests on mass and contact outcome alone.
[#21](https://github.com/NGL321/patchworks/issues/21) has the hidden rotation and the per-puck
friction difference both under revision — rotation is currently dynamically decoupled, and all three
pucks share one μ — so neither is leaned on here. Whichever way that ticket lands, it adds epistemic
pressure or leaves it unchanged; it cannot remove the pressure this section needs.

That is the reading worth having. A model of where things are in one arena is memorisation of that
arena; a model of what things do is the part that would survive being moved to another. Position is
the cheaper variable, and it is the one this world gives away.

**Hiding position would not raise the difficulty — it would break the instrument.** The acceptance
demo's load-bearing measure is onset latency
([`08-the-acceptance-demo.md`](./08-the-acceptance-demo.md)): ticks from the event to the first
corrective torque, which needs the event to be perceivable at a known tick. From a hidden event the
interval becomes event → search → discovery → correction, and search duration is a property of where
a fovea happened to be pointing. It would swamp the one-to-four-hop difference the depth ordering
rests on, and the teleported puck is the *intermediate* rung precisely because a displacement
arrives through vision at ~4 hops.

So the runner-up sensory surface — a short-range sensor rigidly attached to the arm tip, exploration
becoming a sweep of the fovea over the workspace — is **out of scope** rather than deferred: it is a
different world, and it belongs to the second PoC with the rest of the enrichment
[#30](https://github.com/NGL321/patchworks/issues/30) ruled out. Raising the render resolution is
ruled out in the same place and points the wrong way regardless — it would *remove* this world's one
hidden variable rather than add pressure. Occluders are rejected on the onset-latency argument
above.

Exploration in the other sense — acting somewhere the agent has no model of at all — is a different
question with a different owner: [`04-action-and-the-boundary.md`](./04-action-and-the-boundary.md),
*Route selection*.

## Motor surface

Three torques, normalised to `[-1, 1]` and scaled to the per-joint limits. Control runs at **50 Hz**
(physics at 500 Hz, 10 substeps per tick).

Measured: a clean push moves a puck **0.12–0.17 m**, so a puck crosses the arena in two or three
pushes. The heaviest puck needs **more than 2 N at the tip** to break static friction at all; below
that the arm and the puck sit in a static equilibrium, touching but not moving. Torque limits are
sufficient for all three pucks.

### Per-joint gearing: a timescale ladder in the body

Joint damping is `0.8 / 0.5 / 0.3` and torque limits `3 / 2 / 1 N·m`, so the arm is heterogeneous in
*strength*; all three motors currently run `gear="1"`, so it is not deliberately heterogeneous in
*timescale*. It is very likely heterogeneous incidentally — the shoulder swings all three links and
the wrist swings one — and nobody has measured the spread.

The build therefore owes a measurement before it owes a change: impulse each joint, fit the decay,
report the ratio of effective time constants. **If the incidental spread is already wide, the ladder
is free and gearing is redundant.** If it is flat, gear the shoulder slow and strong and the wrist
fast and weak, so that a task must traverse the ladder — a coarse reposition, then a fine adjustment,
in that order.

This is realism the sandbox currently lacks, not scaffolding: every real limb has shoulder inertia
dwarfing wrist inertia. One constraint on it, and it is the one a sceptical reader will reach for
first — **the ladder must not be aligned with the graph's levels by construction.** Three joint
timescales must never be built to correspond to three levels of the core, or "recovered at the
appropriate level" degenerates into a lookup. [ADR-0005](../adr/0005-timescale-is-persistence-not-a-schedule.md)
has the graph's timescales emerging from persistence; the world's job is to give that something to
find, not to tell it what to find. The acceptance demo respects the same constraint from the other
side, by measuring onset rather than decay
([`08-the-acceptance-demo.md`](./08-the-acceptance-demo.md)).

Known exposure: this buys a **timescale ladder**, which is not the same object as precedence depth.
The claim that traversing the ladder induces ordered sub-goals is plausible and unproven.

## The sampler

A task is a **(layout, target puck, target zone)** triple. Layout is 3 puck poses in the spawn
annulus; the goal is one of 3×3 puck-zone pairs.

### Route-blocking layouts

Left to itself the sampler makes one puck matter and leaves the other two as scenery that may or may
not be in the way. That gives the task set a **precedence depth of 1**: every push is locally
correctable, nothing must happen before anything else, and the pedestal — the one feature that could
impose an order — was measured as a graze (59–63% of tasks have a straight puck→zone line anyway, and
the rest detour a median 4%).

So a fixed fraction of sampled layouts **place a non-target puck across the target's route**, where
"across" is `prototypes/route-geometry`'s existing homotopy check applied to a puck instead of the
pedestal. Clearing the blocker first is then forced, and the order is not recoverable after the fact:
push the target into the blocker and the two are worse placed than they started. That is precedence
depth 2, bought with a layout constraint — no new bodies, no physics change, no change to the action
space, and no exposure to 3D.

Deliberately modest. Depth 2 is not depth 8, and the sandbox does not pretend otherwise: see *What
this sandbox does not exercise*, below.

The **held-out slice** is defined along two axes at once, so generalisation is tested
combinatorially and spatially:

- **Pairs**: `(puck 0 → zone 2)` and `(puck 2 → zone 0)` are never sampled in training.
- **Sector**: training never places the *target* puck in the wedge between 30° and 75° at radius
  above 0.22.

`split="train"` samples the complement of both; `split="heldout"` samples the union; `split="any"`
ignores the distinction.

## The Gymnasium contract, made continual

The env is a literal `gymnasium.Env`, with three deliberate deviations. They are the contract, not
placeholders:

- **`reset()` rearranges the world; it never resets the agent.** The arm keeps whatever
  configuration it was left in, physics time is monotonic across the entire run, and no observation
  component announces that anything happened. The agent finds out the world changed the way it
  finds out anything else: its predictions stop working. `reset(options={"reset_arm": True})` exists
  for setup and is not used in normal operation.
- **`reward` is always `0.0`; `terminated` and `truncated` are always `False`.** There is no reward
  channel and there are no episodes. A separate consequence of the same contract, load-bearing
  elsewhere and worth naming here: **experience is generated by interaction, so the sample budget is
  unbounded by construction.** There is no dataset, and no fixed number of examples to be spent —
  which is what lets `06-graph-topology.md` decline the sample-efficiency objection to a locally
  connected architecture. This is *not* the same claim as "no episodes"; a bounded replay of a
  non-episodic recording would satisfy that and not this.
- **Privileged truth lives in `info`** — puck poses, goal identity, goal distance, whether the goal
  is satisfied. It is for logging and evaluation only. Feeding it to the agent defeats the sandbox.

Because there is no episode boundary to restart from, reproducibility comes from
**snapshot/restore** of the full state (`qpos`, `qvel`, `ctrl`, clock, task, RNG), consistent with
[ADR-0001](../adr/0001-continual-learning-applies-to-the-adapting-surface.md).

### The human's hand

Three entry points, which together are the acceptance demo's interface:

- `disturb_arm(joint, impulse)` — an impulse to one joint mid-task. In the live viewer this is a
  ctrl-drag on a link.
- `perturb(puck, xy)` — teleport a puck mid-task. In the live viewer this is a ctrl-drag.
- `retarget(goal_puck, goal_zone)` — change what is wanted without touching the world.

The distinction matters for the acceptance demo: disturbing the body should be absorbed lowest of
all, perturbing the world low, and retargeting high. Having all three means the demo can show the
*level* at which recovery happens, not merely that it happens.

**The arm is disturbed by an impulse, never by a teleport.** Displacing `qpos` directly would have
the world rewrite the arm's configuration, which is the one thing this env never does — `reset()`
rearranges the world and leaves the agent where it is. An impulse is the world moving, which is
exactly what a motor edge is defined as being cleared by
([ADR-0003](../adr/0003-action-is-prediction-the-world-clears.md)), so proprioception reports it the
way it reports everything else and no new observation path exists.

Recovery from that impulse is read as **onset latency** — ticks to the first corrective torque — and
never as a decay or settling time. A settling time here would be a joint's mechanical time constant,
which is precisely what *Per-joint gearing* above is building a deliberate spread in; reading
recovery off that ladder is what the non-alignment constraint forbids. See
[`08-the-acceptance-demo.md`](./08-the-acceptance-demo.md) for the protocol.

## Achievability

A deliberately dumb scripted controller (Jacobian-transpose reaching, no learning, reading the
privileged `info`) solves **15 of 48** sampled tasks within 60 s of sim time each, across both
splits and all three pucks. That is a **lower bound**: it establishes that the geometry, torque
limits, and friction admit the tasks the sampler generates. It is not a baseline, and no agent
should be compared against it.

**The same number read the other way, which is the less flattering and equally valid reading:** a
controller with no model, no learning, no hierarchy and no planning takes 31% of the task set. So
15/48 also bounds how much of this task set requires anything the architecture provides. A build
session should meet that fact at the start rather than rediscover it, and the route-blocking layouts
above exist partly to move it.

## What this sandbox does not exercise

Stated plainly, because the alternative is discovering it in the acceptance demo. This world is
**thin in precedence depth** — 2 with route-blocking layouts, against a graph eight levels deep. The
architecture is built for long-horizon compositional planning at multiple scales, and this sandbox
does not contain a task that demands it. The risk that follows is real and accepted: the demo may
succeed with a few cells doing the work.

That is a deliberate scoping decision, not an oversight. **This proof of concept shows that the
architecture functions; a second one shows that it functions well**, in a world enriched along axes
this one declines — a second arm, articulated or more numerous objects, more links, higher render
resolution, task breadth, and three dimensions. Those belong to that effort. What guards the claim
here is not the world's richness but the two falsification conditions in `06-graph-topology.md` and
the demo's obligation to exercise both of the human's hands at two latencies.

The env runs at roughly **400 ticks/s** with rendering on a single laptop CPU core.
