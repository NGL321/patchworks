"""Achievability: how much of the sampler's task set a dumb hand takes (ticket #82).

`docs/spec/03-the-sandbox.md`, *Achievability*, records **14 of 72** sampled
tasks solved within 60 s of sim time each, across all three splits. This script
is that number's provenance: a deliberately dumb scripted controller
(Jacobian-transpose reaching, no learning, no model, no planning) reading the
privileged `info` no agent may see.

    python benchmarks/achievability.py

The count is a **lower bound** on the world, not a baseline for an agent. It
establishes that the geometry, torque limits and friction admit the tasks the
sampler generates -- a zero would mean the world is broken. Read the other way,
which the spec insists on: a controller with nothing the architecture provides
takes about 20% of this task set, which bounds how much of the set requires
anything the architecture provides. Nothing here is a score, and no agent is to
be compared against it.

The controller is not a proposal. It exists so that the number exists.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import mujoco
import numpy as np

from patchworks.sandbox import (
    CONTROL_HZ,
    ZONE_XY,
    BlockedAnnulusError,
    PlanarPushSandbox,
)

#: 3000 ticks at 50 Hz control is the spec's 60 s of sim time per task.
CAP_TICKS = 3000
#: Half a second inside the zone. `goal_satisfied` is instantaneous, so a puck
#: skidding across a zone would otherwise count as a solve.
HOLD_TICKS = 25
TASKS_PER_SPLIT = 24
#: All three splits, which is where 3 x 24 = 72 comes from. `any` is excluded:
#: it ignores the split distinction and draws from the whole space, so its
#: tasks would land in the count attributable to no split.
MEASURED_SPLITS = ("train", "heldout_pair", "heldout_sector")
SEED = 11


class ScriptedPusher:
    """Jacobian-transpose reaching: line up behind the puck, push through it.

    Reads `env.model`, `env.data` and the privileged `info`. Deliberately has
    no model of the puck, no memory between ticks, and no plan: what it takes
    off the task set is what geometry alone gives away.
    """

    STANDOFF = 0.10  # how far behind the puck the tip lines up
    ARC = 0.17  # radius of the swing-around when the tip is on the wrong side
    PEDESTAL = 0.145  # pedestal radius plus a margin: no standoff point lives here
    GAIN = 30.0  # tip-space P gain; the heaviest puck needs >2 N to break friction
    DAMPING = 1.5
    FORCE_LIMIT = 3.0

    def __init__(self, env: PlanarPushSandbox):
        self.model = env.model
        self.data = env.data
        self._tip_sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tip")
        self._arm_dofs = [
            self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)]
            for name in ("j0", "j1", "j2")
        ]
        self._torque_limit = self.model.actuator_ctrlrange[:, 1].copy()
        self._jac = np.zeros((3, self.model.nv))

    def __call__(self, info: dict) -> np.ndarray:
        puck = info["puck_pose"][info["goal_puck"], :2]
        zone = ZONE_XY[info["goal_zone"]]

        mujoco.mj_jacSite(self.model, self.data, self._jac, None, self._tip_sid)
        tip = np.array(self.data.site_xpos[self._tip_sid][:2])
        tip_v = self._jac[:2] @ self.data.qvel

        heading = zone - puck
        heading /= np.linalg.norm(heading) + 1e-9

        # A puck pinned against the pedestal can have its standoff point
        # *inside* the pedestal -- no straight push exists. Walk it around the
        # obstacle instead.
        if np.linalg.norm(puck - heading * self.STANDOFF) < self.PEDESTAL:
            radial = puck / (np.linalg.norm(puck) + 1e-9)
            tangent = np.array([-radial[1], radial[0]])
            zone_dir = zone / (np.linalg.norm(zone) + 1e-9)
            turn = radial[0] * zone_dir[1] - radial[1] * zone_dir[0]
            heading = tangent * (np.sign(turn) or 1.0)

        standoff = puck - heading * self.STANDOFF
        rel = tip - puck
        bearing = np.arctan2(rel[1], rel[0])
        want = np.arctan2(-heading[1], -heading[0])  # bearing of the standoff point
        delta = (want - bearing + np.pi) % (2 * np.pi) - np.pi

        if abs(delta) > 0.35:
            # The tip is not behind the puck. Sweep it around on a circle of
            # radius ARC, advancing the bearing a little each tick. A *fixed*
            # waypoint here is a trap: the tip parks on it, the geometry stops
            # changing, and the arm sits still commanding zero torque forever.
            aim = bearing + np.clip(delta, -0.25, 0.25)
            target = puck + self.ARC * np.array([np.cos(aim), np.sin(aim)])
        elif np.linalg.norm(tip - standoff) > 0.035:
            target = standoff  # line up
        else:
            target = puck + heading * 0.12  # push through, hard

        force = np.clip(
            (target - tip) * self.GAIN - tip_v * self.DAMPING,
            -self.FORCE_LIMIT,
            self.FORCE_LIMIT,
        )
        torque = self._jac[:2, self._arm_dofs].T @ force
        return np.clip(torque / self._torque_limit, -1.0, 1.0)


@dataclass
class SplitResult:
    """One split's share of the task set, and what the scripted hand did with it."""

    split: str
    #: The tick budget each task was given, carried with the result so that
    #: what gets reported cannot disagree with what was measured.
    cap: int = CAP_TICKS
    tasks: int = 0
    solved: int = 0
    #: `[solved, tasks]` per target puck: the spec's figure is claimed
    #: across all three pucks, so a split solved entirely on the light one
    #: would be a different fact wearing the same number.
    by_puck: dict[int, list[int]] = field(default_factory=dict)
    ticks_to_solve: list[int] = field(default_factory=list)
    #: Times the arm had to be moved out of the spawn annulus by hand (below).
    arm_in_the_way: int = 0


def run_task(env, policy, info: dict, cap: int, hold: int) -> tuple[bool, int]:
    """Push until the goal has held for `hold` ticks, or until `cap` ticks are spent.

    `env` is anything with the sandbox's `step()`: the tick loop is the whole
    of what this needs, and a stand-in is how the hold gets tested without
    spending physics on it.
    """
    held = 0
    for tick in range(1, cap + 1):
        _, _, _, _, info = env.step(policy(info))
        held = held + 1 if info["goal_satisfied"] else 0
        if held >= hold:
            return True, tick
    return False, cap


def measure(
    tasks_per_split: int = TASKS_PER_SPLIT,
    cap: int = CAP_TICKS,
    hold: int = HOLD_TICKS,
    seed: int = SEED,
    splits: tuple[str, ...] = MEASURED_SPLITS,
) -> dict[str, SplitResult]:
    """Run the scripted controller over `tasks_per_split` tasks from each split."""
    results = {}
    for split in splits:
        env = PlanarPushSandbox(split=split, render_obs=False)
        policy = ScriptedPusher(env)
        result = SplitResult(split=split, cap=cap)
        try:
            _, info = env.reset(seed=seed, options={"reset_arm": True})
            for task in range(tasks_per_split):
                # The rearrangement leads the task rather than trailing it, so
                # that nothing is drawn after the last task counted -- a draw
                # outside the 72 could otherwise report a blocked annulus the
                # count does not cover.
                if task:
                    # The arm is not reset between tasks -- physics time is
                    # monotonic and there are no episodes -- so it can end one
                    # standing where the next layout has to go, which reset()
                    # refuses to place one inside. Moving it is then the
                    # caller's job, and how often that was needed is reported
                    # rather than absorbed, because it starts a task from a
                    # pose the rest of the run does not draw from.
                    try:
                        _, info = env.reset()
                    except BlockedAnnulusError:
                        result.arm_in_the_way += 1
                        _, info = env.reset(options={"reset_arm": True})

                goal_puck = info["goal_puck"]
                solved, ticks = run_task(env, policy, info, cap, hold)

                result.tasks += 1
                tally = result.by_puck.setdefault(goal_puck, [0, 0])
                tally[1] += 1
                if solved:
                    result.solved += 1
                    tally[0] += 1
                    result.ticks_to_solve.append(ticks)
        finally:
            env.close()
        results[split] = result
    return results


def report(results: dict[str, SplitResult]) -> None:
    if not results:
        raise ValueError("nothing was measured: no splits were run")
    solved = sum(r.solved for r in results.values())
    tasks = sum(r.tasks for r in results.values())
    # One budget across the run, or the headline line would advertise a number
    # some of the tasks under it were never given -- which is the disagreement
    # carrying the cap on the result was meant to rule out.
    caps = {r.cap for r in results.values()}
    if len(caps) != 1:
        raise ValueError(f"one tick budget per run, got {sorted(caps)}")
    cap = caps.pop()
    for result in results.values():
        print(f"[{result.split}] {result.solved}/{result.tasks} solved")
        for puck in sorted(result.by_puck):
            ok, n = result.by_puck[puck]
            print(f"    puck {puck}: {ok}/{n}")
        if result.ticks_to_solve:
            mean = np.mean(result.ticks_to_solve) / CONTROL_HZ
            print(f"    mean time to solve: {mean:.1f} s of sim")
        if result.arm_in_the_way:
            print(f"    arm moved out of the annulus by hand: {result.arm_in_the_way}")
    print(
        f"\nachievability: {solved}/{tasks} solved within "
        f"{cap / CONTROL_HZ:.0f} s of sim each, a lower bound on the world"
    )


def main() -> None:
    start = time.perf_counter()
    report(measure())
    print(f"({time.perf_counter() - start:.0f} s wall)")


if __name__ == "__main__":
    main()
