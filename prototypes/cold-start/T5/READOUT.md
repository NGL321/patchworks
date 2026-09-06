# T5 (#530): does the arm move at all, under *any* command?

**Yes.** Precondition **P1** — a non-zero actuator → outcome response — **holds**, by four
decades of linear response and with no dead zone. T3's `0.000e+00` is a property of the
**command**, not of the body.

Mechanism claims only (#517 guard 2). Nothing here says anything about abstraction, depth,
retention, or ADR-0026's bar.

* `probe.py` — the probe. Bypasses the graph and writes the actuator's **commanded** block by
  hand; #506 established that block is what the world reads, and part 4 verifies the identity
  against a live dome rather than asserting it.
* `readout.py` → `530-readout.txt` — the tables. `530-probe.json` — the raw readings.
* Seeds 42/43/44, spread published as max − min over seeds. Cost: **67 s**, one env at a time,
  taken beside T3's three live 100k processes.

---

## 1. The initial pose is **not** at the joint limits. The parked pose is.

The ticket's alternative explanation for the exact zero — that the arm is built at a stop, so a
command pushing outward produces zero travel for reasons unrelated to learning — is **refuted at
the initial pose and confirmed at the pose the reading was taken**.

| | j0 | j1 | j2 |
|---|---|---|---|
| joint range (rad) | ±3.1416 | ±2.6 | ±2.6 |
| **pose as built** | 0.0 | 0.0 | 0.0 |
| margin to nearest limit | 3.1416 | 2.6 | 2.6 |
| … as a fraction of the half-range | **1.00** | **1.00** | **1.00** |
| **pose after 800 ticks of #120's constant** | −3.1418 | −2.6009 | 2.6024 |
| margin to nearest limit | **−2.1e−04** | **−9.0e−04** | **−2.42e−03** |

The arm is built at the **exact centre** of every joint's range — the furthest point from every
stop there is. It is *driven* to the stops by the untrained constant, arriving within ~600 ticks
(#120's own reading) and overshooting each limit by 2e−4 to 2e−3 rad, which is the soft
constraint absorbing the residual torque. Identical to four decimals across all three seeds: the
parked pose is a property of the command, not of the layout.

So the stops **are** the binding constraint on every travel figure taken to date — but they bind
at a pose the arm *reached*, not one it was placed in.

## 2. The zero, and what it is made of

#120's constant, held again from the parked pose (100 ticks, per-tick travel):

| seed | travel32 / tick | travel64 / tick | tip path |
|---|---|---|---|
| 42 | 0.000e+00 | 1.475e−13 | 8.40e−13 m |
| 43 | 0.000e+00 | 1.573e−13 | 8.98e−13 m |
| 44 | 0.000e+00 | 1.525e−13 | 8.64e−13 m |

`travel32` is the rig's own quantity — `progress.py` sums `|Δq|` over the float32 `qpos` the
observation carries. `travel64` is the same sum over MuJoCo's float64 register. **The exact zero
is not a float32 artefact**: the float64 reading is 1.5e−13, which is round-off on a quantity of
order 3, i.e. the arm is genuinely still. T3's `0.000e+00` is a true zero, reproduced here in
100 ticks without a graph.

## 3. Travel under a directly written command: P1, with magnitude and spread

**From the built pose.** Small-signal response is **exactly linear over four decades**, with zero
spread across seeds — the same number to every printed digit at 1e−6, 1e−5, 1e−4 and 1e−3:

| direction | joint travel, rad/tick per unit amplitude | tip displacement over 100 ticks, m per unit amplitude |
|---|---|---|
| +e0 (j0 alone) | 7.50e−02 | **3.08** |
| +e1 (j1 alone) | 8.00e−02 | **1.98** |
| +e2 (j2 alone) | 6.67e−02 | **0.651** |
| ±u120 (#120's own direction, unit-normalised) | 1.22e−01 | **2.37** |

At amplitude 0.1 the tip moves 0.064–0.30 m in 2 s of world; at 0.5 it saturates at ~0.9 m, which
is the workspace, not the actuator.

**From the parked pose** — the pose every travel figure to date was read at. Here direction is
everything, and the split is eleven orders of magnitude wide (tail travel, rad/tick, amplitude 1.0,
mean ± spread over seeds):

| command | tail travel / tick | tip net over 100 ticks |
|---|---|---|
| **−u120** (away from all three stops) | **7.72e−02 ± 2e−02** | **0.321 ± 5e−02 m** |
| +e0 (away from j0's stop) | 5.21e−02 ± 4e−03 | 1.78e−02 m |
| −e2 (away from j2's stop) | 4.39e−02 ± 3e−03 | 4.52e−02 m |
| +u120 (**into** all three stops — #120's own command) | 2.11e−13 ± 1e−16 | 1.22e−04 m |
| −e0 (into j0's stop) | 1.87e−13 | 3.50e−04 m |

**The falsifying end of P1 is zero response, and the reading is not zero.** The body answers, at
the parked pose included, the moment the command points away from the stop it is resting on. The
ratio between the two ends at that pose is ~3.7e11.

**Dead zone: none.** The sweep found non-zero tail travel at every amplitude down to 1e−6 (the
linear floor) and at 1e−7 as well, in every direction tested. The 1e−7 readings are the arm's own
settling off the soft-constraint overshoot rather than a response, so the honest statement is *no
dead zone above the settling floor, and the response is linear from 1e−6 up*.

**Saturation** is in the way, and it is not the actuator's. Above amplitude ~0.5 the *tail* travel
from the built pose collapses to ~1e−6 rad/tick — because the arm has arrived at a stop **within
the 100-tick hold**. Big commands do not push harder; they arrive sooner.

## 4. What actually produces the zero: constancy, not magnitude

#120's constant is `[-0.2341, -0.5523, 0.3187]` — comfortably **inside** the action space, so no
clip is involved. 5000-tick holds from the built pose (100 s of world; margin < 0 means past the
limit, i.e. resting on the stop):

| command | tail travel64 / tick | margin to nearest limit, per joint |
|---|---|---|
| +u120 @ 0.1 | **1.03e−12 ± 1e−13** | [−3e−05, −2.6e−04, −5.3e−04] — on all three stops |
| +e0 @ 0.1 | **3.86e−14 ± 8e−14** | [−9e−05, 1.570, 2.420] — on j0's stop |
| −u120 @ 0.1 | 1.86e−07 ± 6e−07 | [0.996, 0.287, −5.3e−04] |
| +u120 @ 0.01 | 7.88e−04 ± 7e−04 | [2.194, 0.115, 0.889] — still travelling |
| +e0 @ 0.001 | 5.00e−05 ± 7e−05 | [2.874, 2.600, 2.600] — still travelling |

The arm has no gravity and no restoring force, only joint damping, so a **constant** torque has one
terminal state: the joint runs to its stop (or into a contact) at terminal velocity and rests
there. Every 0.1-amplitude constant parks within 5000 ticks; the smaller ones have not arrived by
tick 5000 and are still creeping toward one at 5e−5 to 8e−4 rad/tick.

So the mechanism behind T3's zero is **not** that the body cannot answer, and **not** that the
command is too small. It is that the command does not **turn over**. A constant command of any
magnitude ends on a stop; #120's is simply one that got there by tick 600.

## 5. The shape of the map: gain, rank, conditioning

Command → tip displacement, by central differences, restored between columns.

| pose | σ₁ | σ₂ | effective rank (participation) |
|---|---|---|---|
| built (arm stretched straight) | 1.2005 ± 0.079 | 0.0090 ± 0.027 | 1.016 ± 0.047 |
| **generic** ([1.149, −0.944, 0.389]) | 0.1739 ± 0.009 | 0.0368 ± 0.001 | **1.405 ± 0.018** |
| parked | 8e−04 ± 0 | 1e−04 ± 0 | 1.312 ± 0.000 |

The built pose reads **exactly rank one** — σ₂ = 0 to machine precision at 2 and 10 ticks, at both
deltas (see the horizon grid in `530-readout.txt`). That is not a fact about the body: the pose the
arm is built in is the arm **stretched straight**, all three links collinear, which is a kinematic
singularity where every joint turns the tip the same way. Read at a generic pose reached in-band,
the map is **rank two — full, for a planar effector** — merely ill-conditioned at about 4.7:1.

This is worth stating precisely against [R-B](https://github.com/NGL321/patchworks/issues/529)'s
reading that *rank one is the non-exitable state in the literature's own diagnostic*: whatever is
rank one in this system, **the body's command → effector map is not**, once the arm is anywhere but
its own build pose or its stops.

## 6. Does the graph reach the commanded block at all?

**Yes, and it is not inert.** The commanded block is initialised to exactly zero and `Agent.write`
never touches it (#506); it is filled by **reconciliation**, from inside the graph. 400 ticks of an
untrained dome, seed 42:

| | shallow (358 cells, 507 edges) | full (414 cells, 682 edges) |
|---|---|---|
| commanded before any tick | [0, 0, 0] | [0, 0, 0] |
| commanded after one reconciliation | [0, 0, 0] | [0, 0, 0] |
| ever non-zero over the run | **yes** | **yes** |
| command, mean of ticks 300–400 | [5.913, −8.881, 1.322] | [−0.896, 0.280, −0.060] |
| command, sd of ticks 300–400 | [0.200, 0.077, 0.245] | [0.049, 0.164, 0.154] |
| travel64 / tick, last 100 | 3.69e−05 | 6.18e−03 |

Two things fall out. The full dome at 400 ticks is *not yet* locked — travel is 6e−3 rad/tick and
the command still has sd 0.05–0.16 — which is consistent with #120 reading the lock in by ~600.
And on the **shallow** dome the untrained command sits at **±9**, nine times outside the declared
action space, so it is clipped to a corner every tick and the arm is already parked. Recorded, not
pursued: it is a fact about the shallow diagnostic surface, retired at T3 (#517 guard 5).

The identity the rest of this probe rests on holds exactly: a hand-written commanded block
`[0.61, −0.37, 0.29]` comes back bit-identical from `Agent.command()`, is applied bit-identical to
the arm, and lands bit-identical in the efference block.

---

## What this does not settle, and what is left for the user

P1 holds, so P6, P7 and P14 are no longer decided against by the body. **This ticket does not
touch them** and makes no claim about whether a curiosity drive is worth building.

The reading suggests a fix and, per the ticket, does not build or propose one — stated and left:
the arm's stillness is downstream of the command being **constant**, not of it being small, not of
the body, and not of the drive's magnitude. Anything that makes the command's *direction* turn over
keeps travel non-zero. That is the same variable #496 found for the apex (direction, not
magnitude), arriving independently on the supply side. **The user's call.**
