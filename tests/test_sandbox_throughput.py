"""How fast the world runs.

The spec records roughly **400 ticks/s with rendering on a single laptop CPU
core** (`docs/spec/03-the-sandbox.md`, last line); the reference machine
measures 471 ticks/s. That number belongs to a machine, so the assertion here
is an order-of-magnitude floor rather than the number itself -- it catches a
regression that makes the sandbox unusable (a render path falling back to
software, a substep count drifting) without failing on whatever CPU and GL
stack the run happens to have.
"""

import time

import numpy as np

from patchworks.sandbox import PlanarPushSandbox

TICKS = 300
FLOOR_HZ = 50.0


def test_throughput_with_rendering(capsys):
    env = PlanarPushSandbox(split="any")
    try:
        env.reset(seed=0, options={"reset_arm": True})
        action = np.zeros(3, np.float32)
        for _ in range(20):  # warm the renderer up
            env.step(action)

        start = time.perf_counter()
        for _ in range(TICKS):
            env.step(action)
        rate = TICKS / (time.perf_counter() - start)
    finally:
        env.close()

    with capsys.disabled():
        print(f"\n  sandbox throughput with rendering: {rate:.0f} ticks/s")
    assert rate > FLOOR_HZ
