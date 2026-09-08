"""T6 ([B29](#585)): the `flat` null **split by the width of the loop it is drawn for**.

`b29_holonomy.py` pools the Haar null over the whole basis. That is the right null for
the wide basis, where every cycle carries 12 or 15 dimensions, and the wrong one for the
250 cycles a lateral caps at rank 1: a 1x1 holonomy's `identification` can only be 0 or
`sqrt(2)` and its `channel_return` is identically 1, so pooling those with the wide
cycles produces a median that belongs to neither population. A claim about the
one-dimensional loops has to be read against a one-dimensional null.

This module draws the null on its own and reports it **per base width**, so the
comparison is like for like. It needs no training and no checkpoint: `flat_maps` is a
fresh exactly-flat draw on the arm's own block structure, which is the same object
whatever the surface has learned -- so one run covers every arm and every horizon.

Usage::

    PYTHONPATH=src python prototypes/cold-start/T6/b29_nulls.py --draws 8
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)

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


arms_mod = _load("t6_arms", _HERE / "arms.py")
b29 = _load("t6_b29", _HERE / "b29_holonomy.py")

import holonomy_read as hr  # noqa: E402


def null_by_width(arm: str, seed: int, draws: int) -> dict:
    """The Haar null per base width, on both bases, for one arm."""
    env, agent = arms_mod.build_arm(arm, seed)
    try:
        dome = agent.dome
        bases = b29.cycles_of(dome)
        out = {"arm": arm, "reserve_p": arms_mod.ARMS[arm][1], "draws": draws, "bases": {}}
        for name, cycles in bases.items():
            pooled: dict[int, dict[str, list]] = {}
            for d in range(draws):
                g = torch.Generator().manual_seed(seed * 1000 + d)
                fm = hr.flat_maps(dome, g)
                rows = hr.read_surface(dome, fm, cycles)
                for row in rows:
                    bucket = pooled.setdefault(
                        int(row["m"]), {c: [] for c in b29.COLUMNS}
                    )
                    for c in b29.COLUMNS:
                        bucket[c].append(float(row.get(c, float("nan"))))
            out["bases"][name] = {
                "by_m": {
                    str(m): {
                        "readings": len(vals[b29.COLUMNS[0]]),
                        **{c: b29._q(vals[c]) for c in b29.COLUMNS},
                    }
                    for m, vals in sorted(pooled.items())
                }
            }
            # The sign question the 1x1 loops actually pose: what share come back
            # positive? `identification` is 0 for a positive return and sqrt(2) for a
            # negative one, so the share is the whole distribution.
            one = pooled.get(1)
            if one is not None:
                ident = np.array(one["identification"], dtype=np.float64)
                out["bases"][name]["one_dim_positive_share"] = float((ident < 0.5).mean())
        return out
    finally:
        env.close()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--arms", nargs="+", default=["shipped", "reserve_p8", "reserve_p16", "reserve_p24"]
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--draws", type=int, default=8)
    p.add_argument("--out", type=Path, default=_HERE / "585-nulls-by-width.json")
    args = p.parse_args()

    record = {"issue": 585, "reading": "the Haar null per base width", "arms": {}}
    for arm in args.arms:
        record["arms"][arm] = null_by_width(arm, args.seed, args.draws)
        for name, entry in record["arms"][arm]["bases"].items():
            share = entry.get("one_dim_positive_share")
            widths = ", ".join(
                f"m={m}: ident {v['identification']['median']:.4f}"
                f" ret {v['channel_return']['median']:.4f}"
                for m, v in entry["by_m"].items()
            )
            print(
                f"  {arm} [{name}] {widths}"
                + (f" | 1x1 positive share {share:.3f}" if share is not None else ""),
                flush=True,
            )
        args.out.write_text(json.dumps(record, indent=1))
    print(f"wrote {args.out.name}")


if __name__ == "__main__":
    main()
