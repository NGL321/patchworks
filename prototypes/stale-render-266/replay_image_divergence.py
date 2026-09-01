"""Where does a replayed image differ while the integration state matches?

`replay_determinism.py` finds replays that agree on `mjSTATE_INTEGRATION` at
every tick and still draw a different frame. `render_repeatability.py` rules out
the renderer: it reproduces itself exactly for a state that never moved. So the
difference is in something the render reads that the *state* does not carry.

This dumps that tick: the integration state, the derived quantities the renderer
actually draws from (`xpos`/`xquat`), the model fields the env mutates
(`dof_frictionloss`, `site_rgba`), and the contact count.
"""
import numpy as np, mujoco
from patchworks.sandbox import PlanarPushSandbox
from patchworks.sandbox.state import snapshot, restore

STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION
SEED = 0


def physics_state(env):
    v = np.empty(mujoco.mj_stateSize(env.model, STATE_SPEC))
    mujoco.mj_getState(env.model, env.data, v, STATE_SPEC)
    return v


def derived(env):
    return {
        "xpos": env.data.xpos.copy(),
        "xquat": env.data.xquat.copy(),
        "ncon": int(env.data.ncon),
        "frictionloss": env.model.dof_frictionloss.copy(),
        "site_rgba": env.model.site_rgba.copy(),
        "qacc": env.data.qacc.copy(),
    }


def pass_over(env, actions):
    out = []
    for a in actions:
        obs = env.step(a)[0]
        out.append((obs["image"].copy(), physics_state(env), derived(env)))
    return out


env = PlanarPushSandbox(split="train")
env.reset(seed=7, options={"reset_arm": True})
rng = np.random.default_rng(SEED)
for _ in range(40):
    env.step(rng.uniform(-1, 1, 3).astype(np.float32))

actions = np.random.default_rng(0).uniform(-1, 1, (100, 3)).astype(np.float32)
state = snapshot(env)
first = pass_over(env, actions)
restore(env, state)
second = pass_over(env, actions)

for tick, ((ia, sa, da), (ib, sb, db)) in enumerate(zip(first, second)):
    if np.array_equal(ia, ib) and np.array_equal(sa, sb):
        continue
    d_img = np.abs(ia.astype(int) - ib.astype(int))
    print(f"tick {tick}: state identical = {np.array_equal(sa, sb)}")
    print(f"  image  max {d_img.max():3d}, {(d_img != 0).mean() * 100:.3f}% of pixels")
    for key in da:
        a, b = da[key], db[key]
        if isinstance(a, int):
            same = a == b
            print(f"  {key:14} identical = {same}   ({a} vs {b})")
        else:
            same = np.array_equal(a, b)
            extra = "" if same else f"   max |diff| {np.abs(a - b).max():.3e}"
            print(f"  {key:14} identical = {same}{extra}")
    break
else:
    print("no divergence at this seed")
env.close()
