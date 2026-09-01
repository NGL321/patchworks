"""What the added mj_forward costs a tick.

Timed against the two rows `benchmarks/sandbox_throughput.py` publishes -- the
camera-drawn step and the camera-blanked one -- because the forward is a fixed
cost and what matters is which row it is visible in.
"""
import statistics, time
import numpy as np, mujoco
from patchworks.sandbox import PlanarPushSandbox

TICKS, WARMUP = 300, 20


def samples_ms(fn, n):
    out = []
    for i in range(WARMUP + n):
        start = time.perf_counter()
        fn()
        out.append((time.perf_counter() - start) * 1e3)
    return out[WARMUP:]


env = PlanarPushSandbox(split="any", render_obs=True)
env.reset(seed=0, options={"reset_arm": True})
action = np.zeros(3, np.float32)

forward = samples_ms(lambda: mujoco.mj_forward(env.model, env.data), TICKS)
step = samples_ms(lambda: env.step(action), TICKS)
env.close()

plain = PlanarPushSandbox(split="any", render_obs=False)
plain.reset(seed=0, options={"reset_arm": True})
plain_step = samples_ms(lambda: plain.step(action), TICKS)
plain.close()

f = statistics.median(forward)
for label, s in (("step, camera rendered", step), ("step, camera blanked", plain_step)):
    m = statistics.median(s)
    print(f"{label:<24} median {m:6.3f} ms   forward is {100 * f / m:5.1f}% of it")
print(f"{'mj_forward alone':<24} median {f:6.3f} ms")
