# PROTOTYPE — the planar sandbox

**Throwaway.** Built to answer [issue #3](https://github.com/NGL321/patchworks/issues/3): what the
world looks like and what the agent's sensory and motor surfaces are. The validated decisions live
in [`docs/spec/03-the-sandbox.md`](../../docs/spec/03-the-sandbox.md) on the `action` branch. This
code is the primary source behind them, not something to build on.

## Run it

```bash
python3.12 -m venv .venv-proto
.venv-proto/bin/pip install 'mujoco==3.10.0' gymnasium numpy imageio
```

`mujoco` is pinned: newer releases ship no macOS **x86_64** wheels and try to build from source.

```bash
cd prototypes/sandbox

../../.venv-proto/bin/python watch.py              # scripted pusher, live viewer
../../.venv-proto/bin/python watch.py --babble     # motor babble instead
../../.venv-proto/bin/python probe.py              # headless: shapes, reset semantics, sampler
../../.venv-proto/bin/python achievable.py         # solve rate over sampled tasks
```

In the viewer, ctrl-drag a puck to perturb the world by hand, and press `r` to rearrange it without
resetting the arm.

## What is a proposal and what is scaffolding

- `arena.xml` and `sandbox_env.py` — **the proposal.** Geometry, observation and action contracts,
  the sampler, and the continual `reset()`.
- `watch.py`'s `ScriptedPusher` — **scaffolding.** A weak hand-written controller that reads the
  privileged `info` no agent may see. It exists only to show the tasks are physically achievable.
  It is not a baseline.
- `probe.py`, `achievable.py` — **evidence.** The runs that produced the numbers in the spec.

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
8. **Push too gently and nothing moves ever again.** The arm and puck settle into a static
   equilibrium, touching but not moving. The heaviest puck needs >2 N at the tip.
