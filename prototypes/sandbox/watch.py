"""PROTOTYPE — throwaway. Watch the sandbox run.

    .venv-proto/bin/python prototypes/sandbox/watch.py            # scripted pusher, live viewer
    .venv-proto/bin/python prototypes/sandbox/watch.py --babble   # motor babble instead
    .venv-proto/bin/python prototypes/sandbox/watch.py --headless --ticks 3000

In the viewer, double-click a puck and ctrl-drag it: that is the acceptance demo's
perturbation, done by hand. Pressing `r` rearranges the world without resetting the arm.

The scripted policy is NOT a proposal. It exists to prove the arm can move each puck and
that the task is achievable at these torque limits — a hand-written controller, not the
agent, and it reads the privileged `info` no agent may see.
"""

import argparse
import time

import mujoco
import numpy as np

from sandbox_env import ZONE_RADIUS, ZONE_XY, PlanarPushSandbox


class ScriptedPusher:
    """Jacobian-transpose reaching: pick a standoff point behind the puck, push through it."""

    STANDOFF = 0.10          # how far behind the puck the tip lines up
    ARC = 0.17               # radius of the swing-around when the tip is on the wrong side
    PEDESTAL = 0.145         # pedestal radius plus a margin: no standoff point can live here

    def __init__(self, env):
        self.env = env
        self._jac = np.zeros((3, env.model.nv))

    def __call__(self, info) -> np.ndarray:
        puck = info["puck_pose"][info["goal_puck"], :2]
        zone = ZONE_XY[info["goal_zone"]]
        tip, tip_v = info["tip_xy"], self._tip_vel()

        heading = zone - puck
        heading /= np.linalg.norm(heading) + 1e-9

        # A puck pinned against the pedestal can have its standoff point *inside* the
        # pedestal — no straight push exists. Walk it around the obstacle instead.
        if np.linalg.norm(puck - heading * self.STANDOFF) < self.PEDESTAL:
            radial = puck / (np.linalg.norm(puck) + 1e-9)
            tangent = np.array([-radial[1], radial[0]])
            zone_dir = zone / (np.linalg.norm(zone) + 1e-9)
            turn = radial[0] * zone_dir[1] - radial[1] * zone_dir[0]
            heading = tangent * (np.sign(turn) or 1.0)

        standoff = puck - heading * self.STANDOFF

        rel = tip - puck
        bearing = np.arctan2(rel[1], rel[0])
        want = np.arctan2(-heading[1], -heading[0])     # bearing of the standoff point
        delta = (want - bearing + np.pi) % (2 * np.pi) - np.pi

        if abs(delta) > 0.35:
            # The tip is not behind the puck. Sweep it around on a circle of radius ARC,
            # advancing the bearing a little each tick. A *fixed* waypoint here is a trap:
            # the tip parks on it, the geometry stops changing, and the arm sits still
            # forever commanding zero torque.
            step = np.clip(delta, -0.25, 0.25)
            aim = bearing + step
            target = puck + self.ARC * np.array([np.cos(aim), np.sin(aim)])
        elif np.linalg.norm(tip - standoff) > 0.035:
            target = standoff                       # line up
        else:
            target = puck + heading * 0.12          # push through, hard

        # The gain matters: the heaviest puck needs >2 N at the tip to break its
        # friction at all. Push too gently and the whole system sits in a static
        # equilibrium, touching the puck forever without moving it.
        f = np.clip((target - tip) * 30.0 - tip_v * 1.5, -3.0, 3.0)
        return self._torques(f)

    def _tip_vel(self) -> np.ndarray:
        e = self.env
        mujoco.mj_jacSite(e.model, e.data, self._jac, None, e._tip_sid)
        return (self._jac[:2] @ e.data.qvel)

    def _torques(self, f_xy) -> np.ndarray:
        e = self.env
        mujoco.mj_jacSite(e.model, e.data, self._jac, None, e._tip_sid)
        dofs = [e.model.jnt_dofadr[j] for j in e._jid]
        tau = self._jac[:2, dofs].T @ np.asarray(f_xy)
        return np.clip(tau / e._ctrl_scale, -1.0, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--babble", action="store_true")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--ticks", type=int, default=100_000)
    ap.add_argument("--split", default="train")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    env = PlanarPushSandbox(split=args.split, seed=args.seed)
    obs, info = env.reset(seed=args.seed)
    policy = ScriptedPusher(env)
    rng = np.random.default_rng(args.seed)
    a = np.zeros(3)

    solved = 0
    hold = 0

    def tick():
        nonlocal a, solved, hold, obs, info
        if args.babble:
            a = 0.9 * a + 0.1 * rng.uniform(-1, 1, 3)
        else:
            a = policy(info)
        obs, _, _, _, info = env.step(a)
        hold = hold + 1 if info["goal_satisfied"] else 0
        if hold == 25:                       # half a second inside the zone
            solved += 1
            print(f"  solved #{solved}: puck {info['goal_puck']} -> zone {info['goal_zone']} "
                  f"at t={info['sim_time']:.1f}s; rearranging")
            obs, info = env.reset()
            hold = 0

    if args.headless:
        t0 = time.time()
        for i in range(args.ticks):
            tick()
            if i % 500 == 0:
                print(f"t={info['sim_time']:7.1f}s  goal puck {info['goal_puck']}->zone "
                      f"{info['goal_zone']}  d={info['goal_distance']:.3f}  "
                      f"touch={np.round(obs['touch'], 2)}  solved={solved}")
        print(f"\n{solved} goals reached in {args.ticks} ticks "
              f"({args.ticks / (time.time() - t0):.0f} env-ticks/s, render included)")
        env.close()
        return

    import mujoco.viewer

    def key_callback(k):
        if k == ord("R"):
            env.reset()
            print("  rearranged by hand")

    with mujoco.viewer.launch_passive(env.model, env.data,
                                      key_callback=key_callback) as viewer:
        viewer.cam.lookat[:] = [0, 0, 0]
        viewer.cam.distance = 1.6
        viewer.cam.elevation = -90
        viewer.cam.azimuth = 90
        last = time.time()
        for i in range(args.ticks):
            if not viewer.is_running():
                break
            tick()
            viewer.sync()
            dt = 0.002 * env.frame_skip - (time.time() - last)
            if dt > 0:
                time.sleep(dt)
            last = time.time()
            if i % 250 == 0:
                print(f"t={info['sim_time']:7.1f}s  d={info['goal_distance']:.3f}  "
                      f"touch={np.round(obs['touch'], 2)}  solved={solved}")
    env.close()


if __name__ == "__main__":
    main()
