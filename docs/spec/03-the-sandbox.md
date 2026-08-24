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
| damping | 0.8 | 0.5 | 0.3 |
| armature (rotor inertia) | 0.12 | 0.02 | 0.002 |
| passive decay τ (ms) | 184 | 80 | 10 |

The armature column is the arm's **timescale ladder** and is doing deliberate work — see
*A timescale ladder in the body*, below.

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
| μ | 0.20 | 0.30 | 0.45 |
| centre of mass | central | **0.018 m off-centre** | central |
| colour | red | green | blue |

Pucks differ in size, mass, and friction, so *which* puck is being pushed changes what happens —
the dynamics are not one model with a colour attached. **μ differs per puck, not just its product
with mass**, which is what makes the difference survive the coasting phase: a free puck decelerates
at `μg`, so a first draft in which all three shared `μ = 0.3` had all three decelerating identically
and differing only at contact. Measured coast from 1.0 m/s is now **0.225 / 0.162 / 0.111 m**, a
2.0× spread ([#21](https://github.com/NGL321/patchworks/issues/21)).

**Puck 1's centre of mass is deliberately off-centre**, 0.018 m from the geometric centre of a
0.045 m disc, placed by burying mass inside the cylinder so the top-down render is unchanged. This
is what makes its orientation θ a real hidden variable rather than a decoupled integrator — see
*Known exposure*, below.

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
rotation is unreadable from the render. Unreadable is not the same as hidden: a variable that
appears in no term of the equations of motion is not hidden, it is *absent*, and a predictor that
ignores it loses nothing.

That was the first draft's mistake. Pucks are cylinders and the marker geom has no collision, so θ
was a pure integrator of ω, dynamically decoupled from everything the arm could do
([#21](https://github.com/NGL321/patchworks/issues/21)). **Puck 1's off-centre centre of mass is
the fix.** A contact through the rim now exerts a torque about a point that moves with θ, so θ
feeds back into where the puck goes. Measured: the same push, delivered at eight different starting
angles, spreads the puck's lateral deflection by **5.80 mm** on an 89 mm push; the two circular
pucks spread by 0.06 mm and 0.02 mm, which is the numerical floor.

The eccentric mass was chosen over a non-circular puck precisely because it does not touch the
render. A non-circular puck presents a rotating silhouette, which would partly *un*-hide the
variable it was introduced to hide. This route leaves the render, the observation contract, and the
sampler untouched.

One puck of three, not all three, so *which* puck is being pushed stays discriminating: the agent
must learn that θ matters for the green one and not for the others.

### Dynamics exploration, not spatial exploration

The whole arena is visible in every frame, so **spatial** exploration is not forced — and that is a
decision, not a leftover. The destination's *a sandbox it must explore to model* is honoured by
**dynamics exploration**: mass and contact outcome are nowhere in the render and are recoverable only
by acting. An agent that has looked at this arena and never touched it knows almost nothing that
predicts the next tick; the heaviest puck does not move at all below ~2 N at the tip (*Motor
surface*), so even *whether a push works* is knowledge only action buys.

The argument rests on three things the render does not carry, in descending order of confidence:
**mass and contact outcome**, **per-puck μ** (which now survives the coasting phase, above), and
**puck 1's orientation** (which now enters the equations of motion, above). The first alone would
carry the section; [#21](https://github.com/NGL321/patchworks/issues/21) restored the other two
after finding both weaker than the first draft claimed.

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

Torque limits are sufficient for all three pucks. Two numbers a first draft quoted here were wrong,
and [#21](https://github.com/NGL321/patchworks/issues/21) replaced both by measuring them.

**Break-away force at the tip: 0.21 / 0.40 / 0.98 N.** The draft said the heaviest puck needed more
than 2 N, which was out by a factor of three. At the joint the threshold is the frictionloss value
exactly — `μmg`, 0.098 / 0.294 / 0.883 N — and the tip needs a little more than that because the
contact and the damped arm absorb some of it.

**Push travel is not a constant, and quoting one was the error.** The draft's "a clean push moves a
puck 0.12–0.17 m" describes the top few percent, not the typical case: measured over the scripted
pusher, median travel per push is **4 mm**, p90 **46 mm**, max **233 mm**. What is stable is the
coast law — a free puck decelerates at `μg`, so travel is `v²/2μg` and the controller chooses `v`.
Quote that, not an anecdote.

**The world is not ballistic.** Peak puck speed under the scripted pusher is **151 mm/s median, 544
mm/s maximum**, which sits inside the range the planar-pushing literature actually samples (Yu et
al. top out at 500 mm/s) rather than above it. It is *not* quasi-static either — Bauza & Rodriguez
put that breakdown at 50–80 mm/s — so none of the quasi-static apparatus applies here, and nothing
in this spec reaches for it. Said plainly so nobody reaches for it later.

### Friction is anisotropic, and this is accepted

Table friction is joint frictionloss, and MuJoCo bounds each frictionloss row **element-wise**. With
two independent slide joints the admissible force set is therefore `‖f‖∞ ≤ μmg` — a **square, not a
disc** — so a puck sliding diagonally meets `√2` times the friction it meets along an axis, in a
frame fixed to the world rather than to the puck. Measured, at all three pucks: a 45° coast is
**26.9 / 28.5 / 29.1%** shorter than an axial one, matching `1/√2` to within a percent. With three
independent rows (`x`, `y`, `r`) the limit *surface* is a box in wrench space, which lets a puck
resist maximum translational and maximum torsional friction at once — physically impossible, and a
worse approximation than the ellipsoid the literature already treats as a compromise.

**Accepted, not fixed.** The alternative is abandoning joint frictionloss for contact with a
supporting surface, and joint frictionloss is exactly what makes this world *planar by construction*
— the commitment that keeps "dimensionality is a parameter" honest. The error is consistent, smooth,
and learnable: it is a fact about this world the agent can model, not noise it must average over.
Worth naming because it is real — Yu et al. measured 3/2 anisotropy in the worst of four real
materials and called it significant — and because it is largest in coupled translation-plus-rotation,
which is every off-centre push.

### Sub-threshold holds do not creep

A frictionloss constraint's position residual is identically zero, so it takes its impedance from
`solimp[0]`, which defaults to 0.9 — and about a tenth of the load leaks through. Held below
threshold, a puck therefore does not sit still: it creeps. Measured on the first draft, puck 2 held
at 90% of its threshold for 60 s **drifted 159 mm**, twice a zone radius, at a steady 2.7 mm/s. That
is enough to satisfy or unsatisfy a goal on its own, and enough to corrupt the acceptance demo's
onset measurement.

It is not a numerical artefact: it is invariant under a 4× smaller timestep, 200 solver iterations,
the CG solver, and the implicit integrator, and responds to `solimp[0]` alone. The puck joints
therefore set `solimpfriction` to 0.9999, which takes the same 60 s hold to **0.24 mm**. Static
equilibrium below threshold is now a claim this spec can make.

### The table is not uniform

Repeated identical pushes in a rigid-body simulator are bit-identical. Real ones are not: Yu et al.
measured 1.6–12.5% translation standard deviation over 2,000 repetitions, with at least three modes
and clearly non-Gaussian. A world whose disagreement can be *fully* cleared is a weak proving ground
for an architecture whose central claim is about disagreement that **never fully clears**
([`CONTEXT.md`](../../CONTEXT.md)).

So each puck's frictionloss is scaled by a **friction field**: a smooth, fixed function of the
puck's position, mean 1 and range 0.75–1.25, standing in for the spatial variation in coefficient of
friction that Yu et al. mapped across a real table. The same push at two places in the arena gives
two different outcomes, and the difference is a property of the world rather than of a random seed.

Two alternatives were rejected. **Resampling frictionloss at each `reset()`** would put a number
into the model that `mjSTATE_INTEGRATION` does not cover, so restore would diverge silently — the
exact failure mode [#22](https://github.com/NGL321/patchworks/issues/22) closed on. **Accepting the
determinism** leaves the proving ground weak. A field is a pure function of state, so snapshot and
restore stay bit-exact (verified: zero divergence over a replayed 100-tick tail), and it restores
some spatial epistemic pressure as a side effect: where a puck *is* now changes what a push does.

### Calibration note: paddle–puck friction

The paddle–puck coefficient is 0.6, against Yu et al.'s measured ~0.25 for a steel pusher on their
objects. A rubber-faced paddle is not steel, so 0.6 is not indefensible, but it biases contact
toward sticking rather than sliding — and sticking is the regime that produced the clean pushes
every number above was measured in. Recorded as a calibration choice, not a measurement.

### A timescale ladder in the body: rotor inertia, not gearing

Joint damping is `0.8 / 0.5 / 0.3` and torque limits `3 / 2 / 1 N·m`, so the arm is heterogeneous in
*strength*. It was not deliberately heterogeneous in *timescale*, and the measurement this section
used to owe has now been taken ([#60](https://github.com/NGL321/patchworks/issues/60),
`prototypes/sandbox/precedence_probe.py`).

**The incidental spread was not wide.** Impulsing each joint and fitting the decay of `|qvel|` gave
time constants of **85 / 79 / 44 ms** — a **1.95x** ladder, and the spread *within* a single joint
across arm configurations was larger than the spread between joints (j1 varies 5.4x from folded to
extended). Driven response was flatter still: time to sweep 0.5 rad from rest at full control was
**182 / 158 / 188 ms**, a 1.19x spread, because `ctrlrange 3 / 2 / 1` was already cancelling the
inertia difference. The ladder was not free, and the arm was closer to deliberately *homogeneous*
than anyone had assumed.

**`gear` is the wrong knob, and this is the substantive correction.** `gear` multiplies control into
joint torque; it does not appear in the passive `M / b` decay at all — measured directly, τ(j0) is
57.4 ms at `gear="1"` and 57.4 ms at `gear="4"`. On the driven side it moves speed and strength
*together*, so a shoulder geared down to be slow is also weak: at `gear (0.4, 1, 3)` the
worst-direction tip force falls to **2.61 N**, against the >2 N a heavy puck needs to move at all.
"Shoulder slow and strong" is not reachable through gearing, and the version of this section that
proposed it was proposing something the actuator model cannot do.

**`armature` reaches it, and reaches it alone.** Rotor inertia enters the decay and not the statics,
so it buys timescale at no cost in force. `armature="0.12" / "0.02" / "0.002"` on `j0 / j1 / j2`,
with `gear` and `ctrlrange` untouched:

|                        | j0     | j1     | j2     | ladder    |
| ---------------------- | ------ | ------ | ------ | --------- |
| passive decay τ        | 184 ms | 80 ms  | 10 ms  | **17.9x** |
| driven sweep, 0.5 rad  | 278 ms | 178 ms | 160 ms | 1.7x      |

Torque limits stay `3 / 2 / 1 N·m`, worst-direction tip force stays **6.53 N**, and the shoulder is
now both the slowest joint and the strongest — the ladder the sandbox wanted. Gear compensation
turned out to be unnecessary: armature is absent from the statics, so nothing needed restoring. The
new ladder also dominates configuration dependence rather than hiding under it (j0 now varies 1.14x
across poses, against a 17.9x ladder). The scripted controller solves **14 of 72** sampled tasks
after the change against **16 of 72** before it on the same seeds — two tasks, inside the noise of a
controller this weak.

Which quantity moved matters. It is the **passive decay** that spreads 17.9x; the driven sweep
spreads only 1.7x, because full torque against a heavier rotor is still full torque. That is the
right one to have moved — [ADR-0005](../adr/0005-timescale-is-persistence-not-a-schedule.md) makes
timescale *persistence*, and passive decay is persistence measured in the world — but it is a
narrower thing than "the arm is slow at the shoulder", and the spec should not be read as claiming
the wider one.

This is realism the sandbox lacked, not scaffolding: every real limb has shoulder inertia dwarfing
wrist inertia. One constraint on it, and it is the one a sceptical reader will reach for first —
**the ladder must not be aligned with the graph's levels by construction.** Three joint timescales
must never be built to correspond to three levels of the core, or "recovered at the appropriate
level" degenerates into a lookup. Two things keep that honest here: the rungs are deliberately
uneven (2.3x from j0 to j1, 7.8x from j1 to j2, against a core whose depth is eight), and a cell's
timescale is a distribution rather than a constant
([#41](https://github.com/NGL321/patchworks/issues/41)), so there is no constant for a joint to be
matched against. ADR-0005 has the graph's timescales emerging from persistence; the world's job is
to give that something to find, not to tell it what to find. The acceptance demo respects the same
constraint from the other side, by measuring onset rather than decay
([`08-the-acceptance-demo.md`](./08-the-acceptance-demo.md)).

Known exposure, unchanged by the measurement: this buys a **timescale ladder**, which is not the
same object as precedence depth — and precedence depth is now the thing this sandbox does not have
at all (next section). The claim that traversing the ladder induces ordered sub-goals is plausible
and unproven.

## The sampler

A task is a **(layout, target puck, target zone)** triple. Layout is 3 puck poses in the spawn
annulus; the goal is one of 3×3 puck-zone pairs.

### Route-blocking layouts: proposed, measured, dropped

Left to itself the sampler makes one puck matter and leaves the other two as scenery that may or may
not be in the way. That gives the task set a **precedence depth of 1**: every push is locally
correctable, and nothing must happen before anything else. The proposal
([#30](https://github.com/NGL321/patchworks/issues/30)) was to buy depth 2 with a layout constraint —
a fixed fraction of layouts placing a non-target puck across the target's route, so that clearing
the blocker first is forced. No new bodies, no physics change, no change to the action space.

**It was built, measured, and it does not work.** The test is paired: the *same* layout run twice,
once with the blocker in the corridor and once with that same blocker displaced out of it, so the
only difference is the blocker's position ([#60](https://github.com/NGL321/patchworks/issues/60),
`prototypes/sandbox/precedence_probe.py`). Five constructions, 136 pairs, 60 s of sim each:

| construction | blocked | blocker moved aside |
| --- | --- | --- |
| straight-route clip, overlap depth > 0.5 | 6/32 | 6/32 |
| clip, light blocker (puck 0) / heavy blocker (puck 2) | 6/20 / 4/20 | 6/20 / 3/20 |
| blocker on the paddle's **standoff point** | 5/20 / 5/20 | 5/20 / 4/20 |
| **pocket**: target jammed on the pedestal, blocker sealing the short way round | 2/12 / 1/12 | 0/12 / 1/12 |
| pooled | **29/136** | **25/136** |

Null in every construction, and if anything tilted the wrong way. It is not that the blocker gets
swept aside — the heavy puck moves a median of 39 mm across a whole run in the corridor case.

**The reason is structural, and it is the useful part of the finding: nothing in this arena is
concave, so no puck can be pinned.** The pedestal is a convex cylinder and a puck jammed against it
slides around it (median displacement 169 mm in the pocket construction, *more* than in open
space); the ring wall lies outside the spawn annulus and off every puck→zone route. Every blocker is
therefore displaceable, and a straight-line route that a blocker clips was never the route anything
actually followed — the corridor is ~0.09 m wide in an arena with room to go around. Making blocking
real needs a **concave feature** — a recess in the wall, an L-shaped obstacle — which is new geometry,
which is enrichment, which this map has ruled out of scope.

Two measurements are kept because a later effort will want them: **16.9%** of `train` layouts already
clip the target's route by accident, and forcing the condition by rejection sampling costs a mean of
**6 draws**. The mechanism is cheap. It is just inert.

Precedence depth therefore stays at **1**, and *What this sandbox does not exercise*, below, says so.

The **held-out slice** is defined along two axes at once, so generalisation is tested
combinatorially and spatially:

- **Pairs**: `(puck 0 → zone 2)` and `(puck 2 → zone 0)` are never sampled in training.
- **Sector**: training never places the *target* puck in the wedge between 30° and 75° at radius
  above 0.22.

`split="train"` samples the complement of both. `split="heldout_pair"` and `split="heldout_sector"`
sample one axis each; `split="any"` ignores the distinction. **There is deliberately no value that
returns the union.** The two axes are not the same kind of holdout: withholding two puck-zone pairs
leaves every puck, zone and region seen and withholds only compounds, while withholding a sector
removes a target-puck position outright. A single value spanning both would produce a number
attributable to neither, and the value would get reached for precisely because it exists. Naming
them separately costs a line in the sampler and makes the confound unconstructible.

The sector axis has no reader in this proof of concept — nothing samples it, and the demo's
compositional claim rides on the pair axis alone ([`08-the-acceptance-demo.md`](./08-the-acceptance-demo.md),
*The repeated runs*). It is kept rather than deleted because it is a clean handle for the second
proof of concept, and kept *separate* so that effort inherits an axis rather than a union.

**No zone lies inside the wedge, but not by much.** Zone centres sit at 90°, 210° and 330°; a zone's
0.075 m radius at 0.30 m subtends about 14.5°, so `zone_0` spans roughly 75.5° to 104.5° — tangent
to the wedge's upper edge with about half a degree to spare. The sector holdout therefore does not
silently restrict approach geometry for pair-holdout tasks. Recorded because the margin is thin
enough that moving a zone or widening the wedge would couple the two axes without anything
announcing it.

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
  is satisfied. It is for logging and for the acceptance demo's instrumentation only. Feeding it to
  the agent defeats the sandbox. *Evaluation* here means the pre-registered readouts of
  [`08-the-acceptance-demo.md`](./08-the-acceptance-demo.md) and nothing else, which fixes `info`'s
  three consumers: goal satisfaction as a **gate** on whether a trial is valid, puck poses to place
  `perturb()` and to compute onset ground truth, and goal identity for `retarget()`. **Goal
  satisfaction is a gate, never a score** — no rate is aggregated over trials, which is why the
  absence of a scalar on which two runs could be compared is a property of this sandbox rather than
  a gap in it.

Because there is no episode boundary to restart from, reproducibility comes from
**snapshot/restore** of the full state, consistent with
[ADR-0001](../adr/0001-continual-learning-applies-to-the-adapting-surface.md). The state is
**`mjSTATE_INTEGRATION`**, plus the task and the sampler's RNG, which MuJoCo does not know about.
The friction field (*The table is not uniform*, above) writes to the model every tick and is
deliberately **not** part of this state: it is a pure function of puck position, so restoring the
state restores the field with it.

**A restore is not a reset**, and the two must not be read as the same operation wearing different
names. `reset()` is **in-band**: it is part of normal operation, the agent lives through it, and it
finds out by being wrong. A **restore** rewinds the whole universe, the agent's own adapting surface
included, and is therefore invisible from inside — there is no tick at which a cell could observe
one. It never appears in the env's contract; it is an experimenter's tool. That is what lets
evaluation have a defined start without bending the reset-free contract, and it is the distinction
[ADR-0001](../adr/0001-continual-learning-applies-to-the-adapting-surface.md) records.

Name the engine constant, never an enumeration of fields. MuJoCo defines `mjSTATE_INTEGRATION` as
the entire set of inputs to the forward dynamics, so it tracks the model: an enumeration drifts the
moment the arena gains a feature, and it drifts silently, as a trajectory that diverges rather than
an error. The field the obvious enumeration omits is **`qacc_warmstart`**, and MuJoCo's docs flag
exactly this case — warmstarts matter for reproducibility "when loading a non-initial state (since
the initial state is always cold-started)", with differences accumulating exponentially under
time-stepping. **Every restore here is a non-initial-state load**, because there is no episode
boundary; the load-bearing case is the only case.

Three fields the constant covers are inert in *this* arena and are the reason to name it anyway:
`act` is empty (the actuators are direct-drive `motor`), and `qfrc_applied` / `xfrc_applied` are
zero only because `perturb()` teleports a puck by writing `qpos`. Implement that hand as an applied
force instead and the force becomes state — a change to the human's hand would silently break
restore, in a file nobody would think to reread.

### What the deviations cost in the ecosystem

The three deviations are the contract, and they are honoured against Gymnasium rather than by it.
Each of the following is a place where stock tooling assumes episodes and gets none. They are named
here because every one of them fails *quietly*, and a run corrupted quietly is indistinguishable
from the thing the agent is supposed to experience when `reset()` fires.

**`check_env` fails exactly one check, and that is asserted, not tolerated.**
`check_reset_seed_determinism` calls `reset(seed=123)` twice and requires the observations to match;
here they do not, because `reset()` never resets the arm. The failure is correct — it is the first
deviation, detected. The registration flag that would silence it, `EnvSpec.nondeterministic`, is
documented as meaning the observation cannot be repeated from the same initial state, RNG state and
actions, which is **false** of this env and is precisely what snapshot/restore delivers; setting it
would assert something untrue about the physics in order to quiet a check about `reset()`. So the
conformance test runs `check_env` and asserts that `check_reset_seed_determinism` is the **only**
failure, by name. Every other check keeps its value, and a *second* failure — a space dtype
drifting, a malformed `info` — breaks the build instead of being lost in a known-fails suite.

**A step limit is refused, not merely undocumented.** `make()` wraps in `TimeLimit` whenever
`max_episode_steps` is passed or carried on the registered spec, and `TimeLimit` sets
`truncated=True` — which every standard loop **resets on**, and a reset here is not a restart but an
unannounced rearrangement of the world mid-trajectory. The registration therefore sets
`max_episode_steps=None`, and the env **raises** if it finds a spec-level limit present rather than
trusting that a caller read this paragraph. The reasoning is the snapshot list's: prefer the
constraint that cannot drift to the sentence nobody rereads.

**Stock episode-shaped wrappers go inert, and should not be reached for.**
`RecordEpisodeStatistics` fills its info key only under `if terminated or truncated:`, so across an
entire run it emits nothing at all. Under `VectorEnv`, with both flags pinned `False`, every
`AutoresetMode` is `DISABLED` in effect: a loop expecting sub-envs to hand back fresh initial states
periodically receives none, no error is raised, and each sub-env quietly runs one infinite
trajectory. Neither is a defect to fix — both are what "no episodes" means downstream.

**Known exposure: no metric shape is inherited.** Continual-RL benchmarks avoid the
boundary-agnostic regime partly because their metrics stop being *computable* without boundaries —
Continual World's forgetting measure and CORA's isolated forgetting both index by task boundary.
Patchworks keeps the boundaries in `info`, so its metrics stay computable; the exposure is that no
cited benchmark asks of its agents what this one asks, so there is no established metric shape to
inherit and one must be defined outright.

**The debt is now discharged, and smaller than it looked.** No metric shape is inherited because
none is needed: the map rules baselines, benchmarks and degree of success out of scope, so nothing
here aggregates. What had to be defined outright was not a metric but a **trial** — a defined start
for the repeated runs the acceptance demo's latency ordering is computed over. That definition lives
in [`08-the-acceptance-demo.md`](./08-the-acceptance-demo.md), *The repeated runs*.

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
which is precisely what *A timescale ladder in the body* above builds a deliberate spread in;
reading
recovery off that ladder is what the non-alignment constraint forbids. See
[`08-the-acceptance-demo.md`](./08-the-acceptance-demo.md) for the protocol.

## Achievability

A deliberately dumb scripted controller (Jacobian-transpose reaching, no learning, reading the
privileged `info`) solves **14 of 72** sampled tasks within 60 s of sim time each, across all three
splits and all three pucks. That is a **lower bound**: it establishes that the geometry, torque
limits, and friction admit the tasks the sampler generates. It is not a baseline, and no agent
should be compared against it.

The number has been re-measured twice as the world changed, and has not really moved:
[#21](https://github.com/NGL321/patchworks/issues/21)'s puck changes took it from 15/48 to 12/48
over the two splits that then existed, and [#60](https://github.com/NGL321/patchworks/issues/60)'s
armature ladder gives 14/72 against 16/72 on the same seeds. Each change is two or three tasks out
of the set, inside the noise of a controller this weak, and the world is as solvable as it was.

**The same number read the other way, which is the less flattering and equally valid reading:** a
controller with no model, no learning, no hierarchy and no planning takes about 20% of the task set.
So it also bounds how much of this task set requires anything the architecture provides. A build
session should meet that fact at the start rather than rediscover it. Route-blocking layouts were
meant to move it and were measured not to (above), so nothing in this sandbox moves it.

## What this sandbox does not exercise

Stated plainly, because the alternative is discovering it in the acceptance demo. This world is
**thin in precedence depth** — **1**, against a graph eight levels deep, and the one mechanism
proposed to raise it was built and measured inert (*Route-blocking layouts*, above). The
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
