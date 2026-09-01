"""Which recorded readings does the corrected image move?

#266 framed the split as *does the reading render at all*. It is narrower than
that. `reset()` already ends in `_rederive_from_state`, which forwards, so an
observation taken from a reset was never stale. Only an observation taken from
a *step* was -- and `step` is the one mutator that had no forward in it.

So the discriminator is **does the reading render after a step**, not **does it
render**. This checks both halves of that claim directly:

* a reset observation is forward-consistent -- an extra `mj_forward` moves
  nothing, so a reading that resets once and holds that observation still
  (`alignment_read.py`, `construction_grading.py`) reads what it always read;
* a stepped observation is the one the fix changes, which `stale_channels.py`
  measures.

Run against the tree *before* the fix and the reset rows still read zero; that
is the point of them.
"""
import numpy as np, mujoco
from patchworks.sandbox import PlanarPushSandbox

env = PlanarPushSandbox(split="any", render_obs=True)

print(f"{'trial':>5} {'image max':>10} {'touch max':>12}   after")
for i in range(5):
    # The shape `alignment_read.py` and `construction_grading.py` use: one
    # observation from a reset, then the sheaf is ticked against it. The world
    # is never stepped, so this is the only observation they ever see.
    obs, _ = env.reset(seed=i)
    mujoco.mj_forward(env.model, env.data)
    fresh = env._obs()
    d_img = int(np.abs(fresh["image"].astype(int) - obs["image"].astype(int)).max())
    d_touch = float(np.abs(fresh["touch"] - obs["touch"]).max())
    print(f"{i:5d} {d_img:10d} {d_touch:12.6g}   reset()")

# And the contrast: the same reading taken after a step, which is what
# `untrained_fixed_point.run` -> `agent.tick` -> `env.step` produces.
rng = np.random.default_rng(0)
env.reset(seed=0)
for i in range(5):
    obs, *_ = env.step(rng.uniform(-1, 1, env.action_space.shape).astype(np.float32))
    mujoco.mj_forward(env.model, env.data)
    fresh = env._obs()
    d_img = int(np.abs(fresh["image"].astype(int) - obs["image"].astype(int)).max())
    d_touch = float(np.abs(fresh["touch"] - obs["touch"]).max())
    print(f"{i:5d} {d_img:10d} {d_touch:12.6g}   step()")
env.close()
