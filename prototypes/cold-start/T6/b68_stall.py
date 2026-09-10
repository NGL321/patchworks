"""B68 (#645): the stall stamp, per rule mode, on this ticket's own arms.

`B38 (#599) <https://github.com/NGL321/patchworks/issues/599>`_'s stamp is
**per run**, so it is taken here rather than inherited -- and it has to be taken
per *mode*, because the two rules write different parameters, the agent
therefore issues different commands, and the tick at which the world stops
moving is not a property of the dome alone.

**Two stamping rules, both reported, neither substituted.**
`b38_stall.boundary` takes the **last** window clearing `MOVING = 1e-2`, which
over 2,000 ticks is the same as *when did the body stop* because the body stops
once and stays stopped. Over a 20,000-tick run the world sporadically
re-crosses the threshold long after it has died, and that rule latches onto the
re-crossing and marks the whole run live -- silently strengthening whatever
claim the reading supports, which is the exact failure a per-run stamp exists to
prevent. So this file reports:

* `t_last_above` -- B38's own rule, the last window above `MOVING`;
* `t_first_fall` -- the first window at which the world falls under `MOVING`
  and the honest horizon for a dynamical reading;
* every later `re_crossings`, with its tick and `std_max`, so the gap between
  the two rules is visible rather than argued.

Measured on MuJoCo's own `qpos`/`qvel` per #577 -- a dead representation and a
dead world have to be told apart -- and on `b57_agreement.build_arm` with
`b62_frozen.step`, which is the *same* construction the readings are taken on.

Usage::

    PYTHONPATH=src python prototypes/cold-start/T6/b68_stall.py --ticks 20000
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)

_HERE = Path(__file__).resolve().parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


b57 = _load("b68s_b57", _HERE / "b57_agreement.py")
b38 = _load("b68s_b38", _HERE / "b38_stall.py")

from patchworks.learning import PredictionRule, TransportRule  # noqa: E402

#: B38's own threshold, unchanged and not retuned.
MOVING = b38.MOVING


def stamp(windows: list[dict]) -> dict:
    """Both rules and the re-crossings between them."""
    above = [r for r in windows if r["std_max"] >= MOVING]
    t_last_above = above[-1]["tick"] if above else 0
    t_first_fall = 0
    for row in windows:
        if row["std_max"] < MOVING:
            t_first_fall = row["tick"]
            break
    re_crossings = [
        {"tick": r["tick"], "std_max": r["std_max"]}
        for r in windows
        if r["tick"] > t_first_fall and r["std_max"] >= MOVING
    ]
    dead = [r for r in windows if r["moving_share"] > 0]
    peak = max(r["std_max"] for r in windows)
    tail = windows[-1]
    return {
        "t_last_above": t_last_above,
        "t_first_fall": t_first_fall,
        "honest_horizon": t_first_fall,
        "re_crossings": re_crossings,
        "n_re_crossings": len(re_crossings),
        "t_dead": dead[-1]["tick"] if dead else 0,
        "std_max_peak": peak,
        "std_max_final": tail["std_max"],
        "moving_share_final": tail["moving_share"],
    }


def trace(mode: str, condition: str, seed: int, ticks: int, window: int) -> dict:
    env, agent, arm, _flat = b57.build_arm(condition, seed)
    try:
        bias = PredictionRule(agent.sheaf, operator_rate_ratio=arm["c"]) if mode in ("bias", "both") else None
        transport = TransportRule(agent.sheaf) if mode in ("transport", "both") else None
        rows, buf, tick = [], [], 0
        for _ in b57.t0.run_ticks(agent, ticks, seed=seed):
            buf.append(
                np.concatenate(
                    [
                        np.asarray(env.data.qvel, dtype=np.float64).ravel(),
                        np.asarray(env.data.qpos, dtype=np.float64).ravel(),
                    ]
                )
            )
            if bias is not None:
                bias.step()
            if transport is not None and agent.sheaf.ticks > 1:
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
    finally:
        env.close()
    return {"mode": mode, "condition": condition, "seed": seed, "ticks": ticks, "windows": rows}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--modes", nargs="+", default=["frozen", "bias", "transport", "both"])
    p.add_argument("--condition", default="baseline")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ticks", type=int, default=20_000)
    p.add_argument("--window", type=int, default=10)
    p.add_argument("--out", type=Path, default=_HERE / "645-stall.json")
    args = p.parse_args()

    record = {
        "issue": 645,
        "reading": "stall stamp per rule mode, both stamping rules, on the world's own qpos/qvel",
        "moving_threshold": MOVING,
        "window": args.window,
        "ticks": args.ticks,
        "surface": b57.t0.surface(),
        "rows": [],
    }
    print(
        f"{'mode':>11}{'last_above':>12}{'first_fall':>12}{'recross':>9}"
        f"{'t_dead':>9}{'peak':>11}{'final':>11}{'mins':>7}"
    )
    for mode in args.modes:
        started = time.time()
        rec = trace(mode, args.condition, args.seed, args.ticks, args.window)
        s = stamp(rec["windows"])
        record["rows"].append({**rec, **s})
        args.out.write_text(json.dumps(record, indent=1))
        print(
            f"{mode:>11}{s['t_last_above']:>12}{s['t_first_fall']:>12}"
            f"{s['n_re_crossings']:>9}{s['t_dead']:>9}"
            f"{s['std_max_peak']:>11.3e}{s['std_max_final']:>11.3e}"
            f"{(time.time() - started) / 60:>7.1f}",
            flush=True,
        )
    print(f"\nwritten to {args.out.name}")


if __name__ == "__main__":
    main()
