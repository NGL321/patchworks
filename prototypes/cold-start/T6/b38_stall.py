"""B38 (#599) item 3: the stall boundary, per live arm, stated once.

[B33](#592) traced `reserve_p12` seed 42 and found the world's own state stops
moving at **tick ~125**, not the 1,000–2,000 the map's standing note carried
from [#572](#572) and [#518](#518) — because that note was measured on
`reserve` and `shipped`. A per-arm number was never taken, so every reading on
this map has been inheriting a horizon measured on a different arm.

This takes it on all four arms the map currently reads on, at 10-tick
resolution, and states the boundary rather than leaving a table to be eyeballed.

**Two marks, because "stalled" is two different claims:**

* `t_moving` — the last tick at which the world is *visibly* moving:
  `std_max >= MOVING`. Past it, any dynamical reading is being taken on
  displacements below the scale the arm actually traverses.
* `t_dead` — the last tick at which *any* component moves at all,
  `moving_share > 0` at the `1e-6` floor `b33_motion.py` sets. Past it the
  reading is a ratio of numerical zeros, which is the shape #577 found and
  refused to report.

Both are reported at their window's **end** tick and are therefore accurate to
the window, which is stated in the record. The honest horizon for a reading is
`t_moving`, and it is the smaller of the two by a wide margin on every arm.

Measured on MuJoCo's own `qpos`/`qvel`, not the boundary stalks — #577's point
that a dead representation and a dead world have to be told apart.

Usage::

    PYTHONPATH=src python prototypes/cold-start/T6/b38_stall.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


motion = _load("t6_b33_motion", _HERE / "b33_motion.py")

#: The scale below which the world is not visibly moving. `qvel`/`qpos` on this
#: arena run O(1) while the arm traverses; #572's own `world_std_*` readings and
#: B33's tick-50 `std_max` of 1.287 are both O(1). 1e-2 is two decades below
#: that, so it is a generous floor rather than a tuned one.
MOVING = 1e-2


def boundary(windows: list[dict]) -> dict:
    """The two marks, as the last window that clears each bar and never returns."""
    t_moving = 0
    t_dead = 0
    for row in windows:
        if row["std_max"] >= MOVING:
            t_moving = row["tick"]
        if row["moving_share"] > 0:
            t_dead = row["tick"]
    peak = max(r["std_max"] for r in windows)
    tail = windows[-1]
    return {
        "t_moving": t_moving,
        "t_dead": t_dead,
        "std_max_peak": peak,
        "std_max_final": tail["std_max"],
        "fall_from_peak": peak / tail["std_max"] if tail["std_max"] else float("inf"),
        "moving_share_final": tail["moving_share"],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--arms",
        nargs="+",
        default=["shipped", "reserve", "reserve_p12", "reserve_p16"],
    )
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--ticks", type=int, default=2000)
    p.add_argument("--window", type=int, default=10)
    p.add_argument("--out", type=Path, default=_HERE / "599-stall.json")
    args = p.parse_args()

    record = {
        "issue": 599,
        "reading": "stall boundary per live arm, on the world's own qpos/qvel",
        "moving_threshold": MOVING,
        "window": args.window,
        "ticks": args.ticks,
        "rows": [],
    }
    print(f"{'arm':>12}{'seed':>6}{'t_moving':>10}{'t_dead':>9}{'peak':>11}{'final':>11}{'mins':>7}")
    for arm in args.arms:
        for seed in args.seeds:
            t0 = time.time()
            rec = motion.trace(arm, seed, args.ticks, args.window)
            b = boundary(rec["windows"])
            row = {"arm": arm, "seed": seed, **b, "windows": rec["windows"]}
            record["rows"].append(row)
            # Checkpoint as each arm lands: a long run that dies loses nothing.
            args.out.write_text(json.dumps(record, indent=1))
            print(
                f"{arm:>12}{seed:>6}{b['t_moving']:>10}{b['t_dead']:>9}"
                f"{b['std_max_peak']:>11.3e}{b['std_max_final']:>11.3e}"
                f"{(time.time() - t0) / 60:>7.1f}",
                flush=True,
            )
    print(f"\nwritten to {args.out.name}")


if __name__ == "__main__":
    main()
