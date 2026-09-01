"""Does the added mj_forward move the world? #254 said no over 20 ticks; this
re-checks over a long horizon and several seeds, as #266 asked.

Two envs are driven with an identical action stream. One takes `step` as it now
stands (mj_step, then mj_forward, then observe); the other replays `step` as it
was before the edit (mj_step, then observe). qpos, qvel and the privileged puck
poses are compared bit-exactly at every tick.
"""
import numpy as np, mujoco
from patchworks.sandbox import PlanarPushSandbox

TICKS, SEEDS = 2000, 5


def step_without_forward(env, action):
    """`PlanarPushSandbox.step` as it stood before the mj_forward was added."""
    env._require_task()
    env._apply_friction_field()
    torque = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0) * env._torque_limit
    env.data.ctrl[:] = torque
    mujoco.mj_step(env.model, env.data, nstep=env.frame_skip)
    return env._obs()


print(f"{'seed':>4} {'ticks':>6} {'qpos bit-identical':>19} {'qvel bit-identical':>19} "
      f"{'contacts seen':>14}")
for seed in range(SEEDS):
    # render_obs=False: the image is what the edit changes, and rendering 2000
    # ticks twice buys nothing here. The world is the question.
    fixed = PlanarPushSandbox(split="any", render_obs=False)
    plain = PlanarPushSandbox(split="any", render_obs=False)
    fixed.reset(seed=seed)
    plain.reset(seed=seed)
    rng = np.random.default_rng(seed)

    qpos_same = qvel_same = True
    first_divergence, contacts = None, 0
    for t in range(TICKS):
        action = rng.uniform(-1.0, 1.0, fixed.action_space.shape).astype(np.float32)
        fixed.step(action)
        step_without_forward(plain, action)
        contacts += int(fixed.data.ncon > 0)
        if not np.array_equal(fixed.data.qpos, plain.data.qpos):
            qpos_same = False
            first_divergence = first_divergence if first_divergence is not None else t
        if not np.array_equal(fixed.data.qvel, plain.data.qvel):
            qvel_same = False
    note = "" if first_divergence is None else f"  first differs at tick {first_divergence}"
    print(f"{seed:4d} {TICKS:6d} {str(qpos_same):>19} {str(qvel_same):>19} "
          f"{contacts:14d}{note}")
    fixed.close()
    plain.close()
