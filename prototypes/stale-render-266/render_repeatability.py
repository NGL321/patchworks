"""Is the renderer bit-reproducible for one unchanged state?

`replay_determinism.py` finds a snapshot/replay that agrees on the whole
physics state and disagrees on the image -- on the tree with #266's edit and,
more often, on the tree without it. If the renderer does not reproduce itself
from a state that never moved, that is the source, and it is nothing to do with
the edit.

Nothing here steps. The state is held still and drawn repeatedly.
"""
import numpy as np
from patchworks.sandbox import PlanarPushSandbox

env = PlanarPushSandbox(split="train")
env.reset(seed=7, options={"reset_arm": True})
rng = np.random.default_rng(0)
for _ in range(40):
    env.step(rng.uniform(-1, 1, 3).astype(np.float32))

reference = env._camera_image().copy()
worst, differing = 0, 0
for _ in range(200):
    again = env._camera_image()
    d = np.abs(again.astype(int) - reference.astype(int))
    worst = max(worst, int(d.max()))
    differing += int(d.any())
env.close()

print(f"200 redraws of one unchanged state: {differing} differed, worst {worst} levels")
