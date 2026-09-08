"""T6 (#568 / B19): how much of the exposed block can a cell's neighbours read?

The state-vs-emitted reading turns on `Blocks.interior_rowspace` — the projector
onto the span of a cell's interior maps' rows. If that span is the *whole* exposed
block, then "the maps select a subspace" is vacuous: nothing is selected out, and
any narrowness the reading finds is in the state or in the mask, never in transport.

Cheap: builds each arm, runs a few hundred ticks so the maps are not at their
initialisation, and reports `rank(interior_rowspace)` against `k_v` per cell.

Usage::

    PYTHONPATH=src python prototypes/cold-start/T6/rowspace_568.py
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


t0 = _load("t0_run", _T0 / "run.py")
arms_mod = _load("t6_arms", _HERE / "arms.py")

from excitation import blocks  # noqa: E402
from patchworks.graph import EdgeKind  # noqa: E402
from patchworks.learning import PredictionRule, TransportRule  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", nargs="+", default=["reserve", "reserve_p12", "reserve_p16"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ticks", type=int, default=400)
    args = ap.parse_args()

    record = {"issue": 568, "reading": "rank(interior_rowspace) against k_v", "rows": []}
    for arm in args.arms:
        env, agent = arms_mod.build_arm(arm, args.seed)
        try:
            bias = PredictionRule(agent.sheaf)
            transport = TransportRule(agent.sheaf)
            for _ in t0.run_ticks(agent, args.ticks, seed=args.seed):
                bias.step()
                if agent.sheaf.ticks > 1:
                    transport.step()
            dome = agent.dome
            bl = blocks(agent)
            ranks = torch.linalg.matrix_rank(bl.interior_rowspace.double()).numpy()
            k_v = np.array([int(dome._permitted[c]) for c in dome.predicting])
            sum_int = np.array(
                [
                    sum(
                        dome.edges[e].m
                        for e in dome.incident[c]
                        if dome.edges[e].kind is EdgeKind.INTERIOR
                    )
                    for c in dome.predicting
                ]
            )
            generic = np.minimum(sum_int, k_v)
            row = {
                "arm": arm,
                "reserve_p": arms_mod.ARMS[arm][1],
                "ticks": args.ticks,
                "cells": int(len(ranks)),
                "k_v_median": float(np.median(k_v)),
                "sum_interior_m_median": float(np.median(sum_int)),
                "rank_median": float(np.median(ranks)),
                "rank_min": int(ranks.min()),
                "rank_max": int(ranks.max()),
                "cells_reading_whole_exposed_block": int((ranks >= k_v).sum()),
                "cells_at_generic_rank": int((ranks == generic).sum()),
                "rank_over_k_v_median": float(np.median(ranks / k_v)),
                "rank_over_k_v_min": float((ranks / k_v).min()),
            }
            record["rows"].append(row)
            print(
                f"  {arm:>12} (p={row['reserve_p']:>2}): k_v {row['k_v_median']:.0f}, "
                f"sum interior m {row['sum_interior_m_median']:.0f}, "
                f"rank(rowspace) med {row['rank_median']:.0f} "
                f"[{row['rank_min']}-{row['rank_max']}] | "
                f"whole exposed block at "
                f"{row['cells_reading_whole_exposed_block']}/{row['cells']} cells, "
                f"generic rank at {row['cells_at_generic_rank']}/{row['cells']}, "
                f"rank/k_v med {row['rank_over_k_v_median']:.3f} "
                f"min {row['rank_over_k_v_min']:.3f}",
                flush=True,
            )
        finally:
            env.close()
    out = _HERE / f"568-rowspace-seed{args.seed}-{args.ticks}.json"
    out.write_text(json.dumps(record, indent=1))
    print(f"wrote {out.name}")


if __name__ == "__main__":
    main()
