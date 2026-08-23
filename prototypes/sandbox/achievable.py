"""PROTOTYPE — throwaway. Is the task actually achievable in this world?

    .venv-proto/bin/python prototypes/sandbox/achievable.py

Runs the scripted pusher over many sampled tasks with a time cap and reports the
solve rate, per puck and per split. The scripted policy is a weak hand-written
controller, so this is a LOWER bound on achievability — but a zero here would mean
the world is broken, and a healthy number means the torque limits, friction, and
geometry admit the tasks the sampler generates.
"""

import time
from collections import defaultdict

import numpy as np

from sandbox_env import ZONE_RADIUS, PlanarPushSandbox
from watch import ScriptedPusher

CAP = 3000          # ticks per task = 60 s of sim time
N_TASKS = 24


def main():
    for split in ("train", "heldout"):
        env = PlanarPushSandbox(split=split, seed=11, render_obs=False)
        policy = ScriptedPusher(env)
        obs, info = env.reset(seed=11)
        by_puck = defaultdict(lambda: [0, 0])
        solved = ticks_to_solve = 0
        t0 = time.time()

        for _ in range(N_TASKS):
            gp, hold, done = info["goal_puck"], 0, False
            d0 = info["goal_distance"]
            for k in range(CAP):
                obs, _, _, _, info = env.step(policy(info))
                hold = hold + 1 if info["goal_satisfied"] else 0
                if hold == 25:
                    done = True
                    break
            by_puck[gp][1] += 1
            if done:
                solved += 1
                by_puck[gp][0] += 1
                ticks_to_solve += k
            else:
                print(f"    unsolved: puck {gp} -> zone {info['goal_zone']}, "
                      f"d {d0:.3f} -> {info['goal_distance']:.3f}")
            obs, info = env.reset()

        print(f"[{split}] {solved}/{N_TASKS} solved within {CAP * 0.02:.0f} s each"
              f"   ({time.time() - t0:.0f} s wall)")
        for p in sorted(by_puck):
            ok, n = by_puck[p]
            print(f"    puck {p}: {ok}/{n}")
        if solved:
            print(f"    mean time to solve: {ticks_to_solve / solved * 0.02:.1f} s of sim")
        env.close()


if __name__ == "__main__":
    main()
