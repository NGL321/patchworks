"""T5 (#530): does the arm move at all, under *any* command?

A read of the body, not of learning. The graph is bypassed: the actuator
boundary cell's **commanded** block is written by hand and the arm's answer is
measured. #506 established that the commanded block is the half the world reads
(`Agent.command()` is a slice of that node stalk; `Agent.act` clips it and hands
it straight to `env.step`), so writing a command *is* writing that block --
there is no decode path in between. Part 6 verifies that identity against a live
dome rather than asserting it.

What is read, in the ticket's order:

1. **Travel under a directly written command**, swept over amplitude decades and
   over the commanded block's basis, from two poses: the pose the arm is built
   in, and the pose #120's untrained constant parks it at.
2. **Whether the stops are the binding constraint** -- the initial pose against
   the joint limits, and the parked pose against them.
3. **The shape of the command -> displacement map**: gain, rank of the tip
   Jacobian, dead zone, saturation.
4. **The efference/commanded split's consequence** -- whether the graph reaches
   the commanded block at all.

Rendering is off for parts 1-5 (`render_obs=False`): the command is written by
hand, so nothing in these parts reads the image, and the physics is identical.

Cost: a few minutes, no learning, one env at a time. Three 100k T3 processes are
live on this box and this probe is sized to sit beside them.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass

import mujoco
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "src"))

from patchworks.sandbox.env import ARM_JOINTS, PlanarPushSandbox  # noqa: E402
from patchworks.sandbox.state import restore, snapshot  # noqa: E402

#: #120's measured untrained fixed point, on the full dome at seed 0. The
#: command the arm has been read under in every travel figure to date, T3's
#: 0.000e+00 included.
UNTRAINED_CONSTANT = np.array([-0.2341, -0.5523, 0.3187], dtype=np.float32)

#: How long the arm is driven by #120's constant before it is called parked.
#: #120 reads the lock in by tick ~600; 800 is that with room.
PARK_TICKS = 800

#: The horizon every swept command is held for. 2 s of world at 50 Hz -- long
#: enough for the arm to cross the workspace under a saturating torque, short
#: enough that 200-odd of them cost minutes.
HOLD_TICKS = 100

#: The tail of a hold that is called *sustained* travel, as against the
#: transient of leaving a pose. #120's own distinction: the first 300 ticks'
#: motion was the arm travelling once to the stops.
TAIL = 20

SEEDS = (42, 43, 44)

AMPLITUDES = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1.0)


def _tip_xy(env: PlanarPushSandbox) -> np.ndarray:
    sid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, "tip")
    return np.array(env.data.site_xpos[sid][:2])


def _pose(env: PlanarPushSandbox) -> np.ndarray:
    return np.array(env.data.qpos[env._arm_qadr])


def _limits(env: PlanarPushSandbox) -> np.ndarray:
    return np.array([env.model.jnt_range[j] for j in env._arm_jid])


def _hold(env: PlanarPushSandbox, command: np.ndarray, ticks: int) -> dict:
    """Hold one command for `ticks` and report what the body did.

    Travel is read two ways on purpose. `travel32` is the rig's own quantity --
    `progress.py` sums |dq| over the float32 `qpos` the observation carries --
    and `travel64` is the same sum over MuJoCo's float64 register. A zero in the
    first and a non-zero in the second is quantisation, not a still arm, and the
    ticket's exact 0.000e+00 is a claim that has to survive that distinction.
    """
    action = np.clip(np.asarray(command, dtype=np.float32), -1.0, 1.0)
    prev32 = np.asarray(env._obs()["qpos"], dtype=np.float32)
    prev64 = _pose(env)
    start_tip = _tip_xy(env)
    travel32 = np.zeros(3)
    travel64 = np.zeros(3)
    tail32 = np.zeros(3)
    tail64 = np.zeros(3)
    tip_path = 0.0
    tail_tip = 0.0
    prev_tip = start_tip
    for t in range(ticks):
        observation, _r, _t, _tr, _i = env.step(action.copy())
        now32 = np.asarray(observation["qpos"], dtype=np.float32)
        now64 = _pose(env)
        now_tip = _tip_xy(env)
        d32 = np.abs(now32.astype(np.float64) - prev32.astype(np.float64))
        d64 = np.abs(now64 - prev64)
        dtip = float(np.linalg.norm(now_tip - prev_tip))
        travel32 += d32
        travel64 += d64
        tip_path += dtip
        if t >= ticks - TAIL:
            tail32 += d32
            tail64 += d64
            tail_tip += dtip
        prev32, prev64, prev_tip = now32, now64, now_tip
    end_tip = _tip_xy(env)
    return {
        "travel32": float(travel32.sum()),
        "travel64": float(travel64.sum()),
        "travel32_per_tick": float(travel32.sum() / ticks),
        "travel64_per_tick": float(travel64.sum() / ticks),
        "tail_travel32_per_tick": float(tail32.sum() / TAIL),
        "tail_travel64_per_tick": float(tail64.sum() / TAIL),
        "tip_path": tip_path,
        "tail_tip_per_tick": tail_tip / TAIL,
        "tip_net": float(np.linalg.norm(end_tip - start_tip)),
        "pose_end": _pose(env).tolist(),
        "tip_end": end_tip.tolist(),
    }


@dataclass
class Row:
    seed: int
    pose: str
    direction: str
    amplitude: float
    travel64_per_tick: float
    travel32_per_tick: float
    tail_travel64_per_tick: float
    tail_travel32_per_tick: float
    tip_path: float
    tip_net: float
    tail_tip_per_tick: float


def _directions() -> dict[str, np.ndarray]:
    unit = np.eye(3, dtype=np.float32)
    out: dict[str, np.ndarray] = {}
    for i in range(3):
        out[f"+e{i}"] = unit[i]
        out[f"-e{i}"] = -unit[i]
    k = UNTRAINED_CONSTANT / np.linalg.norm(UNTRAINED_CONSTANT)
    out["+u120"] = k.astype(np.float32)
    out["-u120"] = (-k).astype(np.float32)
    return out


def _jacobian(env: PlanarPushSandbox, base: np.ndarray, parked, delta: float, ticks: int) -> dict:
    """The command -> tip-displacement map at one pose, by central differences.

    Each column is the net tip displacement produced by holding `base +- delta`
    along one commanded component for `ticks`, restored between columns so every
    column is taken at the same pose. Rank is read off the singular values: this
    is what "the body answers, and in how many directions" means for a two-
    dimensional effector under a three-component command.
    """
    cols = []
    for i in range(3):
        step = np.zeros(3, dtype=np.float32)
        step[i] = delta
        restore(env, parked)
        plus = _hold(env, base + step, ticks)
        restore(env, parked)
        minus = _hold(env, base - step, ticks)
        cols.append(
            (np.array(plus["tip_end"]) - np.array(minus["tip_end"])) / (2.0 * delta)
        )
    restore(env, parked)
    jac = np.stack(cols, axis=1)  # 2 x 3
    sv = np.linalg.svd(jac, compute_uv=False)
    pr = float(sv.sum() ** 2 / (sv**2).sum()) if sv.sum() > 0 else 0.0
    return {
        "delta": delta,
        "ticks": ticks,
        "jacobian": jac.tolist(),
        "singular_values": sv.tolist(),
        "effective_rank_participation": pr,
        "condition": float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf"),
    }


def run(seed: int) -> dict:
    env = PlanarPushSandbox(split="train", render_obs=False)
    env.reset(seed=seed)

    limits = _limits(env)
    built_pose = _pose(env)
    built = snapshot(env)

    # -- part 2, first half: the pose the arm is built in, against its limits --
    margin_built = np.minimum(built_pose - limits[:, 0], limits[:, 1] - built_pose)

    # -- park it under #120's constant ---------------------------------------
    park_trace = _hold(env, UNTRAINED_CONSTANT, PARK_TICKS)
    parked_pose = _pose(env)
    margin_parked = np.minimum(parked_pose - limits[:, 0], limits[:, 1] - parked_pose)
    parked = snapshot(env)
    # ... and confirm the lock: the constant, held again from the parked pose,
    # is what every travel figure to date was read under.
    locked = _hold(env, UNTRAINED_CONSTANT, HOLD_TICKS)
    restore(env, parked)

    rows: list[Row] = []
    for pose_name, state in (("built", built), ("parked", parked)):
        for dname, d in _directions().items():
            for amp in AMPLITUDES:
                restore(env, state)
                r = _hold(env, d * np.float32(amp), HOLD_TICKS)
                rows.append(
                    Row(
                        seed=seed,
                        pose=pose_name,
                        direction=dname,
                        amplitude=amp,
                        travel64_per_tick=r["travel64_per_tick"],
                        travel32_per_tick=r["travel32_per_tick"],
                        tail_travel64_per_tick=r["tail_travel64_per_tick"],
                        tail_travel32_per_tick=r["tail_travel32_per_tick"],
                        tip_path=r["tip_path"],
                        tip_net=r["tip_net"],
                        tail_tip_per_tick=r["tail_tip_per_tick"],
                    )
                )

    # -- part 3: the map's shape at each pose --------------------------------
    restore(env, parked)
    jac_parked = _jacobian(env, UNTRAINED_CONSTANT, parked, delta=0.2, ticks=40)
    restore(env, built)
    jac_built = _jacobian(env, np.zeros(3, dtype=np.float32), built, delta=0.2, ticks=40)
    # A rank read on a held command over 40 ticks is a *finite-time* map, and a
    # long enough horizon under a constant torque is dominated by whichever
    # joint has the most lever -- which would manufacture rank one out of the
    # horizon rather than out of the body. So the same read is taken at two
    # deltas and three horizons, and the rank is only a fact about the body if
    # it survives all six.
    jac_grid = []
    for delta in (0.02, 0.2):
        for ticks in (2, 10, 40):
            restore(env, built)
            entry = _jacobian(env, np.zeros(3, dtype=np.float32), built, delta, ticks)
            jac_grid.append(entry)

    # The pose the arm is built in is the arm *stretched straight* -- all three
    # links collinear, which is a kinematic singularity: every joint turns the
    # tip the same way, so a rank read taken there is a fact about that pose and
    # not about the body. A third pose is therefore reached in-band, by holding
    # an ordinary command from the built pose, and the map is read there too.
    restore(env, built)
    _hold(env, np.array([0.30, -0.20, 0.10], dtype=np.float32), 60)
    generic_pose = _pose(env)
    generic = snapshot(env)
    jac_generic = _jacobian(env, np.zeros(3, dtype=np.float32), generic, 0.02, 10)

    # -- dead zone: the smallest amplitude whose tail travel is non-zero ------
    dead = {}
    for dname in ("-e0", "+e0", "-u120", "+u120"):
        d = _directions()[dname]
        lo, hi = None, None
        for amp in (1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0):
            restore(env, parked)
            r = _hold(env, d * np.float32(amp), HOLD_TICKS)
            if r["tail_travel64_per_tick"] > 0.0:
                hi = amp
                break
            lo = amp
        dead[dname] = {"largest_still": lo, "smallest_moving": hi}

    # -- is arriving at a stop a property of *constancy*, or of amplitude? ----
    # The holds above are 100 ticks, and a small command is still travelling at
    # tick 100 only because it has not got there yet. #120's constant is inside
    # the action space and clips nothing, so if a small constant parks too, the
    # thing that parks the arm is that the command is *constant* -- a fixed
    # torque on a damped joint with no restoring force has one terminal state --
    # rather than that it is large. 5000 ticks is 100 s of world.
    long_hold = {}
    for dname in ("+e0", "+u120", "-u120"):
        d = _directions()[dname]
        for amp in (1e-3, 1e-2, 1e-1):
            restore(env, built)
            r = _hold(env, d * np.float32(amp), 5000)
            long_hold[f"{dname}@{amp:g}"] = {
                "tail_travel64_per_tick": r["tail_travel64_per_tick"],
                "tail_travel32_per_tick": r["tail_travel32_per_tick"],
                "pose_end": r["pose_end"],
                "margin_to_nearest_limit": np.minimum(
                    np.array(r["pose_end"]) - limits[:, 0],
                    limits[:, 1] - np.array(r["pose_end"]),
                ).tolist(),
            }

    env.close()
    return {
        "seed": seed,
        "long_hold_5000": long_hold,
        "joint_names": list(ARM_JOINTS),
        "joint_limits": limits.tolist(),
        "built_pose": built_pose.tolist(),
        "built_margin_to_nearest_limit": margin_built.tolist(),
        "parked_pose": parked_pose.tolist(),
        "parked_margin_to_nearest_limit": margin_parked.tolist(),
        "park_ticks": PARK_TICKS,
        "park_trace": park_trace,
        "locked_under_untrained_constant": locked,
        "jacobian_parked": jac_parked,
        "jacobian_built": jac_built,
        "jacobian_grid": jac_grid,
        "generic_pose": generic_pose.tolist(),
        "jacobian_generic": jac_generic,
        "dead_zone": dead,
        "rows": [asdict(r) for r in rows],
    }


#: How long the graph-side check runs an untrained dome for. #120 reads the
#: command locked constant by tick ~600 on the full dome; 400 is enough to see
#: whether the commanded block ever leaves zero, which is the question here.
GRAPH_TICKS = 400


def graph_side_check(name: str, spec, seed: int = 42, ticks: int = GRAPH_TICKS) -> dict:
    """Part 4: does anything the graph writes reach the commanded block?

    Read off a live dome rather than off the source:

    * whether the commanded slice ever leaves its initialised zero under
      reconciliation alone -- `Agent.write` never touches it (#506), so if it
      moves, reconciliation moved it;
    * whether the arm travels over the same run, on the rig's own float32
      travel and on MuJoCo's float64 register;
    * whether a hand-written commanded block is what the arm receives -- the
      identity every other part of this probe rests on.
    """
    import torch

    from patchworks.agent import Agent
    from patchworks.sandbox.env import PATCH_PX
    from patchworks.graph import build_graph

    env = PlanarPushSandbox(split="train", image_size=spec.patch_grid * PATCH_PX)
    agent = Agent(
        env, dome=build_graph(spec), generator=torch.Generator().manual_seed(seed)
    )
    observation, _info = env.reset(seed=seed)
    agent.observe(observation)

    before = agent.command().copy()
    agent.sheaf.tick()
    after_tick = agent.command().copy()

    commands = []
    poses = []
    for _ in range(ticks):
        outcome = agent.tick()
        commands.append(outcome.command.copy())
        poses.append(np.array(env.data.qpos[env._arm_qadr]))
    commands = np.array(commands)
    poses = np.array(poses)
    travel = np.abs(np.diff(poses, axis=0)).sum()
    tail = commands[-100:]

    # the identity: write the commanded block by hand, and see it come out as
    # the action the world is stepped with.
    probe = np.array([0.61, -0.37, 0.29], dtype=np.float32)
    with torch.no_grad():
        agent.sheaf.stalks[agent._commanded_slice] = torch.as_tensor(
            probe, dtype=agent.sheaf.stalks.dtype
        )
    read_back = agent.command().copy()
    out2 = agent.act(read_back)

    result = {
        "dome": name,
        "seed": seed,
        "ticks": ticks,
        "cells": len(agent.dome.cells),
        "edges": len(agent.dome.edges),
        "commanded_before_any_tick": before.tolist(),
        "commanded_after_one_reconciliation": after_tick.tolist(),
        "commanded_ever_nonzero": bool(np.any(commands != 0.0)),
        "commanded_max_abs_over_run": float(np.abs(commands).max()),
        "commanded_tail_mean": tail.mean(axis=0).tolist(),
        "commanded_tail_sd": tail.std(axis=0).tolist(),
        "pose_final": poses[-1].tolist(),
        "travel64_total": float(travel),
        "travel64_per_tick_last_100": float(
            np.abs(np.diff(poses[-101:], axis=0)).sum() / 100.0
        ),
        "hand_written_commanded": probe.tolist(),
        "read_back_as_command": read_back.tolist(),
        "applied_to_arm": out2.applied.tolist(),
        "efference_after_act": agent.sheaf.stalks[agent._efference_slice]
        .detach()
        .numpy()
        .tolist(),
    }
    env.close()
    return result


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    out = {"seeds": {}}
    for seed in SEEDS:
        print(f"seed {seed} ...", flush=True)
        out["seeds"][str(seed)] = run(seed)
    from patchworks.graph import DomeSpec

    # #517's guard 1: the shallow dome, `core_degree=7` because the spec at the
    # default 6 does not build (T1, ledger row 2). The full dome beside it,
    # because #120's constant and T3's 0.000e+00 are both read there, and guard
    # 3 says a shallow finding is replicated full before it moves anything.
    domes = {
        "shallow": DomeSpec(
            vision_sides=(8,), somatomotor_sizes=(6,), core_sizes=(16, 8), core_degree=7
        ),
        "full": DomeSpec(),
    }
    out["graph_side"] = {}
    for name, spec in domes.items():
        print(f"graph-side check, {name} dome ...", flush=True)
        try:
            out["graph_side"][name] = graph_side_check(name, spec)
        except Exception as exc:  # a full dome beside three live 100k runs
            out["graph_side"][name] = {"dome": name, "failed": repr(exc)}
            print(f"  {name}: {exc!r}", flush=True)
    path = os.path.join(here, "530-probe.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
