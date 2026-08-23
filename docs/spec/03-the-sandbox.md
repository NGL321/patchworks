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

Pucks differ in size, mass, and friction, so *which* puck is being pushed changes what happens —
the dynamics are not one model with a colour attached.

Two radii are load-bearing and were both found by watching the thing fail:

- **The ring wall sits at 0.52, outside the arm's 0.49 extent.** The arm therefore never touches
  the boundary, while a puck pinned against it still has its centre inside the workspace. A square
  arena was tried first and was wrong: its corners sat at radius 0.76, so a puck pushed into one
  was gone for good.
- **The pedestal at the centre is a real obstacle**, radius 0.08. The arm is singular at its own
  shoulder and has no leverage there; without the pedestal, a puck nudged toward the centre reaches
  a place where the push-line standoff point lies inside the base and no push can recover it. The
  pedestal walls that region off. It also makes the workspace **non-convex**, which is a feature:
  routing around it is a real subproblem.

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

More generally, the whole arena is visible in every frame, so **spatial** exploration is not
forced. The epistemic pressure in this world is over **dynamics** — mass, friction, contact
outcome, and the hidden rotation — not over where things are. If the acceptance demo needs
exploration for *position*, the sensory surface has to change (a tip-mounted short-range sensor
was the runner-up design, and would restore it).

## Motor surface

Three torques, normalised to `[-1, 1]` and scaled to the per-joint limits. Control runs at **50 Hz**
(physics at 500 Hz, 10 substeps per tick).

Measured: a clean push moves a puck **0.12–0.17 m**, so a puck crosses the arena in two or three
pushes. The heaviest puck needs **more than 2 N at the tip** to break static friction at all; below
that the arm and the puck sit in a static equilibrium, touching but not moving. Torque limits are
sufficient for all three pucks.

## The sampler

A task is a **(layout, target puck, target zone)** triple. Layout is 3 puck poses in the spawn
annulus; the goal is one of 3×3 puck-zone pairs.

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
  channel and there are no episodes.
- **Privileged truth lives in `info`** — puck poses, goal identity, goal distance, whether the goal
  is satisfied. It is for logging and evaluation only. Feeding it to the agent defeats the sandbox.

Because there is no episode boundary to restart from, reproducibility comes from
**snapshot/restore** of the full state (`qpos`, `qvel`, `ctrl`, clock, task, RNG), consistent with
[ADR-0001](../adr/0001-continual-learning-applies-to-the-adapting-surface.md).

### The human's hand

Two entry points, which together are the acceptance demo's interface:

- `perturb(puck, xy)` — teleport a puck mid-task. In the live viewer this is a ctrl-drag.
- `retarget(goal_puck, goal_zone)` — change what is wanted without touching the world.

The distinction matters for the acceptance demo: perturbing the world should be absorbed low in the
hierarchy, while retargeting should be absorbed high. Having both means the demo can show the
*level* at which recovery happens, not merely that it happens.

## Achievability

A deliberately dumb scripted controller (Jacobian-transpose reaching, no learning, reading the
privileged `info`) solves **15 of 48** sampled tasks within 60 s of sim time each, across both
splits and all three pucks. That is a **lower bound**: it establishes that the geometry, torque
limits, and friction admit the tasks the sampler generates. It is not a baseline, and no agent
should be compared against it.

The env runs at roughly **400 ticks/s** with rendering on a single laptop CPU core.
