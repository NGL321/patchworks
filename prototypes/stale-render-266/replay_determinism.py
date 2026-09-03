"""Does the added mj_forward make a snapshot/replay diverge?

`tests/test_sandbox_snapshot.py::test_a_replayed_hundred_tick_tail_shows_zero_divergence`
failed once in a full-suite run while #266's edit was in the tree, and did not
reproduce. The suspicion worth ruling out is real: `mj_forward` re-solves and
writes `qacc_warmstart`, so the edit changes the warm-start the *next* step
begins from. `mjSTATE_INTEGRATION` carries `qacc_warmstart`, so a restore should
put it back and both passes should agree -- but that is an argument, and this is
the measurement.

The test's own body, run over many mid-flight seeds. Run it on the tree with the
edit and on the tree without it and compare the divergence counts.
"""
import sys
import numpy as np, mujoco
from patchworks.sandbox import PlanarPushSandbox
from patchworks.sandbox.state import snapshot, restore

STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION
SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 40


def physics_state(env):
    v = np.empty(mujoco.mj_stateSize(env.model, STATE_SPEC))
    mujoco.mj_getState(env.model, env.data, v, STATE_SPEC)
    return v


diverged, contactless = [], 0
for seed in range(SEEDS):
    env = PlanarPushSandbox(split="train")
    env.reset(seed=7, options={"reset_arm": True})
    rng = np.random.default_rng(seed)
    for _ in range(40):  # mid_flight, at this seed
        env.step(rng.uniform(-1, 1, 3).astype(np.float32))
    if env.data.ncon == 0:
        contactless += 1

    actions = np.random.default_rng(0).uniform(-1, 1, (100, 3)).astype(np.float32)
    state = snapshot(env)
    first = [(env.step(a)[0], physics_state(env)) for a in actions]
    restore(env, state)
    second = [(env.step(a)[0], physics_state(env)) for a in actions]

    for tick, ((oa, sa), (ob, sb)) in enumerate(zip(first, second)):
        if not np.array_equal(sa, sb):
            diverged.append((seed, tick, "state"))
            break
        bad = [k for k in oa if not np.array_equal(oa[k], ob[k])]
        if bad:
            diverged.append((seed, tick, ",".join(bad)))
            break
    env.close()

print(f"{SEEDS} mid-flight seeds, 100-tick replays, {contactless} started contactless")
print(f"diverged: {len(diverged)}")
for seed, tick, what in diverged:
    print(f"  seed {seed}: {what} at tick {tick}")
