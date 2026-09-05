"""#496: does exogenous variation abate the collapse -- the arm, and the drive?

Three conditions on one rig, because they share an instrument, a ladder and a
baseline:

* ``baseline``   -- no intervention. Supplies every denominator and ceiling the
  two readouts need (``rho_soma_frozen``, ``rho_vision``, ``rho_apex_fixed``,
  ``rho_core``) *and* the ``travel`` trace the un-yoked trajectory half wants.
* ``arm``        -- torque babble in place of ``agent.command()``. One line, as
  `#481 <https://github.com/NGL321/patchworks/issues/481>`_ ruled: ``act()``
  still clips, so the efference copy stays honest.
* ``drive``      -- a piecewise-constant, task-blind schedule in place of the
  drive stalk's standing assertion. One line, as
  `#495 <https://github.com/NGL321/patchworks/issues/495>`_ ruled, and
  ``DRIVE_ASSERTION`` itself is **not** retuned.

The instrument is `#166 <https://github.com/NGL321/patchworks/issues/166>`_'s
``prototypes/chart-double-duty-166/read.py``, unmodified and imported, so a
checkpoint here is the same moment and the same statistic as #132's and #274's.

**Why the baseline is not optional, and is worth more than the ageing rule
claimed.** #496 asks for it because the map's *a rig's recorded data ages with
`main`* rule demands it of #120's numbers. It is stronger than that: #477's
figures -- every denominator and both horizon choices this ticket pre-registers
-- were read off #132's committed 100k JSON, which was taken at
``interior_m = 4, boundary_m = 8``. `#474
<https://github.com/NGL321/patchworks/issues/474>`_ has since moved the pair to
``(3, 4)``, changing private dimension at every level, the cut capacities and
ADR-0032's attainable-mask set. That commit says so itself: *"the first reading
on the new surface is owed and unmade."* So the baseline arm is not a control
being refreshed, it is the **first** reading of the collapse on the surface
`main` actually has.

``levels`` and ``columns`` are computed here rather than through ``read.py``'s
``cell_context``, which reads ``dome.cells[c].level`` and raises -- the fallback
at ``read.py:265`` then writes zeros silently, which is the defect #477's
``structure.py`` exists to work around. The right accessor is
``dome.cells[c].index.level``.

Usage::

    PYTHONPATH=src python prototypes/exogenous-variation-496/run.py \
        --condition baseline --seeds 42 43 44 --ticks 100000
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Same reason as the #166 rig states: 150 cells of 12x12 is far too small for
# intra-op parallelism to pay, and this box runs other sessions' benchmarks
# alongside. Pinned before torch is asked to do anything.
torch.set_num_threads(1)

_RIG = Path(__file__).resolve().parents[1] / "chart-double-duty-166"
if str(_RIG) not in sys.path:
    sys.path.insert(0, str(_RIG))
_BENCH = Path(__file__).resolve().parents[2] / "benchmarks"
if str(_BENCH) not in sys.path:
    sys.path.append(str(_BENCH))

import read as rig  # noqa: E402
from patchworks import agent as agent_module  # noqa: E402
from patchworks.graph import CellKind  # noqa: E402
from untrained_fixed_point import build, teaching  # noqa: E402

#: The #166/#274 ladder, plus whatever horizon the run is taken to. Both
#: pre-registered readouts land on it already: the arm reads 5000 and 100k, the
#: apex reads 20000 and 100k.
CHECKPOINTS = rig.CHECKPOINTS

#: The drive schedule's segment length, resampled uniformly from this closed
#: range and then **held**. #495's first constraint: a bare scalar has no
#: integrator, so per-tick noise on the assertion sits inside the cell's own
#: filter and averages back to a constant. Apex `tau` is 2.52 at 20k, so a
#: segment three orders above it is varied as far as the cell is concerned.
HOLD_TICKS = (1_000, 2_000)

#: The schedule's magnitude range, log-uniform, so its **geometric mean is
#: exactly `DRIVE_ASSERTION`** and the instrument does not quietly move the
#: architecture's operating point while varying it. #495's second constraint --
#: never visits zero -- is met by construction: a log-uniform cannot reach 0,
#: and 0.25 is the floor. #183 measured the apex deposit *linear* in the
#: assertion, so this is a 16x swing in what actually arrives.
DRIVE_RANGE = (0.25, 4.0)


def surface() -> dict:
    """What `main` was when this run was taken, since the readings age with it."""
    def git(*args: str) -> str:
        try:
            return subprocess.run(
                ("git", *args),
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                check=True,
            ).stdout.decode("utf-8", "replace").strip()
        except Exception:
            return "unknown"

    from patchworks.graph import DEFAULT_SPEC

    return {
        "commit": git("rev-parse", "HEAD"),
        "describe": git("log", "-1", "--format=%h %ad %s", "--date=short"),
        "dirty": bool(git("status", "--porcelain")),
        "interior_m": int(DEFAULT_SPEC.interior_m),
        "boundary_m": int(DEFAULT_SPEC.boundary_m),
    }


def cell_context(agent) -> dict:
    """Per predicting cell: level, column, degree, private width, drive adjacency.

    `read.py`'s own `cell_context` raises on `dome.cells[c].level` and its caller
    swallows that into all-zero arrays. This is the accessor that works, and it
    also carries the column, which is what names the somatomotor class.
    """
    dome = agent.dome
    predicting = list(dome.predicting)
    drive_cells = {
        cid for cid, cell in enumerate(dome.cells) if cell.kind == CellKind.DRIVE
    }
    boundary = set(dome.boundary)
    levels, columns, degrees, drive_adj, boundary_adj = [], [], [], [], []
    for c in predicting:
        cell = dome.cells[c]
        levels.append(int(cell.index.level))
        columns.append(str(cell.index.column))
        degrees.append(int(dome.degrees[c]))
        neighbours = set(dome.neighbours(c))
        drive_adj.append(int(bool(neighbours & drive_cells)))
        boundary_adj.append(int(len(neighbours & boundary)))
    try:
        p_v = [int(x) for x in dome.private_dimensions]
    except Exception:
        p_v = [0] * len(predicting)
    return {
        "cell_ids": [int(c) for c in predicting],
        "levels": levels,
        "columns": columns,
        "degrees": degrees,
        "p_v": p_v,
        "drive_adjacent": drive_adj,
        "boundary_adjacent": boundary_adj,
    }


class DriveSchedule:
    """Piecewise-constant, resampled every ~1000-2000 ticks, held, never zero.

    **Task-blind by construction**, which is the property that makes this an
    instrument rather than a second architecture: the whole schedule is drawn
    from a seed before the first tick and reads nothing at all -- not
    `info.goal_satisfied`, not prediction error, not `travel`. A schedule
    contingent on satisfaction is #5, not this.
    """

    def __init__(self, seed: int, ticks: int) -> None:
        rng = np.random.default_rng(seed * 1_000_003 + 496)
        low, high = np.log(DRIVE_RANGE[0]), np.log(DRIVE_RANGE[1])
        self.segments: list[dict] = []
        at = 0
        while at < ticks:
            hold = int(rng.integers(HOLD_TICKS[0], HOLD_TICKS[1] + 1))
            self.segments.append(
                {
                    "start": at,
                    "hold": hold,
                    "value": float(np.exp(rng.uniform(low, high))),
                }
            )
            at += hold
        self._edges = np.array([s["start"] for s in self.segments])
        self._values = np.array([s["value"] for s in self.segments])

    def at(self, tick: int) -> float:
        return float(self._values[np.searchsorted(self._edges, tick, "right") - 1])

    def summary(self) -> dict:
        return {
            "segments": len(self.segments),
            "range": list(DRIVE_RANGE),
            "hold_ticks": list(HOLD_TICKS),
            "geometric_mean": float(np.exp(np.log(self._values).mean())),
            "min": float(self._values.min()),
            "max": float(self._values.max()),
            "first_ten": [float(v) for v in self._values[:10]],
        }


def babble(agent, seed: int):
    """Torque babble, uniform in the declared action space, as `probe.py:78`.

    Not the scripted pusher: a competent policy confounds *varied* with *good*,
    and the hypothesis under test is about constancy alone. `act()` still clips
    and still writes the clipped value as the efference copy, so substituting
    here changes what is asked and leaves what is reported honest.
    """
    rng = np.random.default_rng(seed * 1_000_003 + 4961)
    low = np.asarray(agent.action_low, dtype=np.float64)
    high = np.asarray(agent.action_high, dtype=np.float64)

    def command() -> np.ndarray:
        return rng.uniform(low, high).astype(np.float32)

    return command


def stage(target: Path) -> Path:
    """Where a run in flight writes, so a retry cannot destroy a deeper attempt.

    Borrowed from `benchmarks/rim_stalk_scale.py:297`, which exists because
    checkpointing straight to the final name means a **re-run truncates the file
    from its first frame** -- a retry killed at tick 300 destroys a previous
    attempt that reached 10,000. It also makes "did this run finish?" answerable
    from the filesystem: only a completed run has the final name, which is the
    check a watcher wants and the one this rig got wrong on 2026-09-05.
    """
    inflight = target.with_suffix(".inflight.json")
    if inflight.exists():
        index = 0
        while (kept := target.with_suffix(f".killed-{index}.json")).exists():
            index += 1
        inflight.replace(kept)
    return inflight


def run_seed(condition: str, name: str, split: str, seed: int, ticks: int,
             out: Path) -> dict:
    started = time.time()
    inflight = stage(out)
    env, agent = build(name, split, seed)
    schedule = DriveSchedule(seed, ticks) if condition == "drive" else None
    baseline_assertion = agent_module.DRIVE_ASSERTION
    try:
        context = cell_context(agent)
        record = {
            "issue": 496,
            "condition": condition,
            "dome": name,
            "split": split,
            "seed": seed,
            "ticks": ticks,
            "surface": surface(),
            "cells": int(agent.sheaf.operators.cells),
            "k": int(agent.sheaf.operators.shape.k),
            "band": [1.0 / agent.sheaf.operators.rho_k, 1.0],
            "scale_at_construction": float(agent.sheaf.operators.scale),
            "drive_assertion": float(baseline_assertion),
            "context": context,
            "checkpoints": [],
        }
        if schedule is not None:
            record["drive_schedule"] = schedule.summary()

        if condition == "arm":
            agent.command = babble(agent, seed)

        levels, p_v = context["levels"], context["p_v"]
        record["at_construction"] = rig.read(
            rig.spectra(agent.sheaf.operators),
            levels,
            p_v,
            rig.moduli(agent.sheaf.operators),
            rig.nonnormality(agent.sheaf.operators),
        )

        ladder = [c for c in CHECKPOINTS if c <= ticks]
        if ticks not in ladder:
            ladder.append(ticks)

        seen = 0
        # The pose row that closes one window opens the next, so `travel` is a
        # partition of the run's travel rather than dropping the step between
        # two windows -- the same fix `untrained_fixed_point.learning` carries.
        carry: np.ndarray | None = None
        travel_total = 0.0
        for target in ladder:
            window = target - seen
            poses: list[np.ndarray] = [] if carry is None else [carry]
            for outcome in teaching(agent, window, seed=seed + seen):
                poses.append(outcome.observation["qpos"].copy())
                if schedule is not None:
                    # The next tick's write, set before it happens and reading
                    # nothing about this one.
                    agent_module.DRIVE_ASSERTION = schedule.at(agent.sheaf.ticks)
            pose = np.asarray(poses)
            window_travel = (
                float(np.abs(np.diff(pose, axis=0)).sum()) if len(pose) > 1 else 0.0
            )
            travel_total += window_travel
            carry = poses[-1]
            seen = target

            entry = rig.read(
                rig.spectra(agent.sheaf.operators),
                levels,
                p_v,
                rig.moduli(agent.sheaf.operators),
                rig.nonnormality(agent.sheaf.operators),
            )
            entry["ticks"] = target
            entry["travel_window"] = window_travel
            entry["travel_cumulative"] = travel_total
            entry["travel_per_tick"] = window_travel / max(1, window)
            entry["final_pose"] = [float(x) for x in pose[-1]]
            if schedule is not None:
                entry["drive_at_checkpoint"] = schedule.at(target)
            record["checkpoints"].append(entry)
            record["elapsed_minutes"] = (time.time() - started) / 60.0

            # Written as it lands, not at the end: nine 100k runs is many hours
            # and the map's standing rule is to checkpoint rather than lose one.
            # To the in-flight name, so a retry displaces rather than truncates.
            inflight.write_text(json.dumps(record, indent=1))

            print(
                f"  [{condition}] seed {seed} @ {target:>6}: "
                f"rho(K) med {entry['memory']['rho_K']['median']:.4f}  "
                f"retaining {entry['memory']['modes_retaining_per_cell']['median']:.1f}  "
                f"travel {window_travel:9.3f}  "
                f"({record['elapsed_minutes']:.1f} min)",
                flush=True,
            )
        # Only a run that reached its horizon earns the final name. A kill
        # leaves `.inflight.json` behind, which is how a reader -- or a watcher
        # -- tells a finished run from a partial one.
        inflight.replace(out)
        return record
    finally:
        agent_module.DRIVE_ASSERTION = baseline_assertion
        env.close()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--condition", choices=("baseline", "arm", "drive"), required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--ticks", type=int, default=100_000)
    p.add_argument("--dome", default="real")
    p.add_argument("--split", default="train")
    p.add_argument("--tag", default="")
    args = p.parse_args()

    here = Path(__file__).parent
    for seed in args.seeds:
        tag = f"-{args.tag}" if args.tag else ""
        out = here / (
            f"496-{args.condition}{tag}-{args.dome}-{args.split}"
            f"-seed{seed}-{args.ticks}.json"
        )
        print(f"[{args.condition}] seed {seed}, {args.ticks} ticks -> {out.name}",
              flush=True)
        run_seed(args.condition, args.dome, args.split, seed, args.ticks, out)
        print(f"  wrote {out.name}", flush=True)


if __name__ == "__main__":
    main()
