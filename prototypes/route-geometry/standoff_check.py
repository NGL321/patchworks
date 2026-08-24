"""The pedestal's real bite: not the puck's path, but where the paddle must stand.

To push the goal puck straight at the goal zone, the paddle (radius 0.03) must sit
at  standoff = p - u * (puck_r + 0.03),  u = unit(zone - p).
If that point is inside the pedestal+paddle disk (0.08 + 0.03 = 0.11), the direct
push is IMPOSSIBLE and the puck must first be nudged sideways -- a discrete choice
of which side, i.e. two routes that are not blendable.

Also: the tip workspace is an ANNULUS (inner 0.11, outer 0.49), so repositioning
the paddle from bearing a to bearing b is always a choice of swing direction.
Measure how often a task demands a large bearing change.
"""
import sys, math
import numpy as np

sys.path.insert(0, "/Users/angl/Documents/patchworks/prototypes/sandbox")
from sandbox_env import PlanarPushSandbox, ZONE_XY  # noqa

PED, PADDLE = 0.08, 0.03
PUCK_R = [0.035, 0.045, 0.055]


def main(n=600):
    for split in ("train", "heldout"):
        env = PlanarPushSandbox(split=split, seed=7, render_obs=False)
        blocked = 0
        near_blocked = 0
        bearings = []
        tot = 0
        for _ in range(n):
            t = env.sample_task()
            p = np.asarray(t.puck_xy[t.goal_puck][:2], dtype=float)
            q = ZONE_XY[t.goal_zone].astype(float)
            u = (q - p) / np.linalg.norm(q - p)
            stand = p - u * (PUCK_R[t.goal_puck] + PADDLE)
            r = np.linalg.norm(stand)
            tot += 1
            if r < PED + PADDLE:
                blocked += 1
            elif r < PED + PADDLE + 0.04:
                near_blocked += 1
            # bearing change the paddle must make: from puck bearing to standoff bearing
            # proxy for "cross the arena": bearing between the two pucks not being pushed
            bearings.append(abs(math.atan2(*stand[::-1]) - math.atan2(*p[::-1])))
        env.close()
        print(f"[{split}] n={tot}")
        print(f"    direct push IMPOSSIBLE (standoff inside pedestal+paddle): "
              f"{blocked} = {100*blocked/tot:.1f}%")
        print(f"    marginal (standoff within 4 cm of it): "
              f"{near_blocked} = {100*near_blocked/tot:.1f}%")
        print(f"    combined blocked-or-marginal: {100*(blocked+near_blocked)/tot:.1f}%")


if __name__ == "__main__":
    main()
