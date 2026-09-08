"""When does the world actually stop moving on `reserve_p12`?

The record says the sandbox stalls **between 1,000 and 2,000 ticks** (#572's
`world_std_*` on `reserve`/`shipped` x seeds 42/43; #518's own `travel_window`).
B33's per-checkpoint stamp read `std_max` **1.59 at tick 100 and 1.0e-03 by tick
250** on `reserve_p12` seed 42, which is a different number on a different arm,
so this traces it directly rather than arguing from two checkpoints.

Successive disjoint windows from 0 to the horizon, on the world's own state
(`qpos`/`qvel` off MuJoCo, not the boundary stalks — a dead representation and a
dead world are exactly what has to be told apart).

Usage::

    PYTHONPATH=src python prototypes/cold-start/T6/b33_motion.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
_T0 = _HERE.parent / "T0"
for extra in (_ROOT / "prototypes" / "chart-double-duty-166", _ROOT / "benchmarks", _T0):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


t0 = _load("t0_run", _T0 / "run.py")
arms_mod = _load("t6_arms", _HERE / "arms.py")

from patchworks.learning import PredictionRule, TransportRule  # noqa: E402


def trace(arm: str, seed: int, ticks: int, window: int) -> dict:
    env, agent = arms_mod.build_arm(arm, seed)
    bias = PredictionRule(agent.sheaf)
    transport = TransportRule(agent.sheaf)
    rows, buf, tick = [], [], 0
    for _ in t0.run_ticks(agent, ticks, seed=seed):
        buf.append(
            np.concatenate(
                [
                    np.asarray(env.data.qvel, dtype=np.float64).ravel(),
                    np.asarray(env.data.qpos, dtype=np.float64).ravel(),
                ]
            )
        )
        bias.step()
        if agent.sheaf.ticks > 1:
            transport.step()
        tick += 1
        if len(buf) == window:
            std = np.stack(buf, axis=0).std(axis=0)
            rows.append(
                {
                    "tick": tick,
                    "std_max": float(std.max()),
                    "moving_share": float((std > 1e-6).mean()),
                }
            )
            buf = []
    return {
        "issue": 592,
        "reading": "when the world stops moving on this arm",
        "arm": arm,
        "seed": seed,
        "ticks": ticks,
        "window": window,
        "windows": rows,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", default="reserve_p12")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ticks", type=int, default=2000)
    p.add_argument("--window", type=int, default=50)
    args = p.parse_args()
    rec = trace(args.arm, args.seed, args.ticks, args.window)
    out = _HERE / f"592-motion-{args.arm}-seed{args.seed}-{args.ticks}.json"
    out.write_text(json.dumps(rec, indent=1))
    print(f"{'tick':>6}{'std_max':>12}{'moving':>9}")
    for row in rec["windows"]:
        print(f"{row['tick']:>6}{row['std_max']:>12.3e}{row['moving_share']:>9.3f}")
    print(f"\nwritten to {out.name}")


if __name__ == "__main__":
    main()
