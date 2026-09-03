"""Which observation channels lag the qpos/qvel beside them?

`mj_step` integrates qpos/qvel but leaves every *derived* quantity -- the body
positions the renderer draws from, and sensordata -- describing the state at
the *start* of the tick. This walks the sandbox and, at each tick, compares the
observation as `step` returns it against the same observation re-read after
`mj_forward` reconciles the derived quantities with the integrated state.
"""
import numpy as np, mujoco
from patchworks.sandbox import PlanarPushSandbox

env = PlanarPushSandbox(split="any", render_obs=True)
obs, _ = env.reset(seed=0)
rng = np.random.default_rng(0)

print(f"{'tick':>4} {'|qvel|max':>10} {'image max':>10} {'image !=':>9} "
      f"{'touch max':>12} {'qpos same':>10} {'qvel same':>10}")
for t in range(20):
    action = rng.uniform(-1.0, 1.0, env.action_space.shape).astype(np.float32)
    obs, *_ = env.step(action)
    stale_image, stale_touch = obs["image"], obs["touch"]
    stale_qpos, stale_qvel = obs["qpos"], obs["qvel"]
    speed = float(np.abs(env.data.qvel).max())

    # Reconcile derived quantities with the integrated state, then re-read.
    mujoco.mj_forward(env.model, env.data)
    fresh = env._obs()

    d_img = np.abs(fresh["image"].astype(int) - stale_image.astype(int))
    d_touch = float(np.abs(fresh["touch"] - stale_touch).max())
    print(f"{t:4d} {speed:10.4f} {d_img.max():10d} "
          f"{float((d_img != 0).mean()):9.4f} {d_touch:12.6g} "
          f"{str(np.array_equal(fresh['qpos'], stale_qpos)):>10} "
          f"{str(np.array_equal(fresh['qvel'], stale_qvel)):>10}")
env.close()
