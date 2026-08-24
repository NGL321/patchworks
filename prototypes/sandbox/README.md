# PROTOTYPE — the planar sandbox

**Throwaway.** Built to answer [issue #3](https://github.com/NGL321/patchworks/issues/3): what the
world looks like and what the agent's sensory and motor surfaces are. The validated decisions live
in [`docs/spec/03-the-sandbox.md`](../../docs/spec/03-the-sandbox.md) on the `action` branch. This
code is the primary source behind them, not something to build on.

## Run it

**Setup and every entry point live in the [repo README](../../README.md) and only there.** Two
copies drifted apart once already; this file does not keep a second one.

`precedence_probe.py` is the newest of them — [#60](https://github.com/NGL321/patchworks/issues/60)'s
timescale ladder and route-blocking constructions.

## What is a proposal and what is scaffolding

- `arena.xml` and `sandbox_env.py` — **the proposal.** Geometry, observation and action contracts,
  the sampler, and the continual `reset()`.
- `watch.py`'s `ScriptedPusher` — **scaffolding.** A weak hand-written controller that reads the
  privileged `info` no agent may see. It exists only to show the tasks are physically achievable.
  It is not a baseline.
- `probe.py`, `achievable.py`, `precedence_probe.py` — **evidence.** The runs that produced the
  numbers in the spec. `precedence_probe.py` takes a section argument (`a`, `b`, `b4`, `b6`, `b7`,
  `b8`) because the paired runs are minutes each.

## What broke on the way

Kept because each one is a fact about the world, and each was invisible until the thing ran:

1. **Puck bodies had non-zero frame origins**, so joint coordinates were not world coordinates. A
   puck was spawned inside the arena wall and launched to `y = −7.5`.
2. **A square arena's corners sat at radius 0.76**, far outside the arm's 0.46 reach — a puck pushed
   into one was unrecoverable. Hence the circular ring.
3. **The ring first went at 0.44, inside the arm's 0.49 extent**, so the arm jammed through the
   boundary. It has to sit outside the arm but close enough that a pinned puck stays reachable.
4. **The arm is singular at its own shoulder.** A puck nudged toward the centre reached a place
   where the push-line standoff point lay inside the base. Hence the pedestal obstacle.
5. **MuJoCo does not filter the base/link0 contact pair**, because the base is welded to the world
   and that disables the parent filter. Link 0 lived permanently inside the new pedestal, reading
   10⁶ N of contact force. Fixed with an explicit `<exclude>`.
6. **The sampler ignored the arm's current pose.** Since the arm is never reset, layouts spawned
   pucks inside the links. Now it places, checks for contact, and re-places.
7. **A fixed approach waypoint is a trap.** The controller parked on it, the geometry stopped
   changing, and the arm sat still commanding zero torque forever. The approach has to sweep a
   bearing around the puck.
8. **`gear` cannot make a joint slow.** It multiplies control into torque and is absent from the
   passive `M/b` decay entirely, so gearing a joint down to be slow only makes it weak. `armature`
   is the knob that buys timescale, and it costs nothing in force because rotor inertia is not in
   the statics (#60).
9. **Nothing in this arena is concave, so no puck can be pinned.** Every attempt to force a task
   ordering by putting a blocker in the way measured null — a puck jammed against the pedestal
   slides around it, and the straight puck→zone route was never the route anything followed (#60).
10. **Push too gently and nothing moves ever again.** The arm and puck settle into a static
   equilibrium, touching but not moving. The heaviest puck needs >2 N at the tip.
