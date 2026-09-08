"""T5 (#546): composed rank against **rebuilt** interior lane width, at construction.

`#546 <https://github.com/NGL321/patchworks/issues/546>`_ asks whether training's
erosion of composed rank scales with lane width. Before spending ticks, this
reads the *construction* end of the curve on domes that are genuinely **rebuilt**
at each width -- not T4's ``sweep_m``, which redraws frames at a fixed mask width.

**Why the distinction decides the ticket's arm choice.** A cell's mask width is
``k_v = min(n, sum_e m_e)`` (:mod:`patchworks.graph`, ``_assemble``), so widening
the lanes widens the ambient too, until it saturates at ``n = 32``. The
genericity that drives the product of principal-angle cosines is ``m / k_v``
(T4's finding), so a real rebuild moves that ratio far less than ``sweep_m``
suggests. This reads how much, so the trained arms are run at widths whose
construction ER actually differs.

Usage::

    PYTHONPATH=src python prototypes/cold-start/T5/width.py --widths 3 4 6 8 10
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
_T0, _T2, _T4 = (_HERE.parent / n for n in ("T0", "T2", "T4"))
for extra in (_ROOT / "prototypes" / "chart-double-duty-166", _ROOT / "benchmarks", _T0):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


t2 = _load("t2_run", _T2 / "run.py")
angles = _load("t4_angles", _T4 / "angles.py")

from patchworks.agent import Agent  # noqa: E402
from patchworks.graph import build_graph  # noqa: E402
from patchworks.sandbox import PlanarPushSandbox  # noqa: E402
from untrained_fixed_point import IMAGE_SIZE, dome_named  # noqa: E402


def build_at(width: int, seed: int, split: str = "train"):
    """`build("real", ...)` with the interior lane width overridden on the spec."""
    base, _ = dome_named("real")
    spec = replace(base, interior_m=int(width))
    env = PlanarPushSandbox(split=split, image_size=IMAGE_SIZE[spec.patch_grid])
    agent = Agent(env, dome=build_graph(spec), generator=torch.Generator().manual_seed(seed))
    return env, agent


def mask_read(dome) -> dict:
    """`k_v` and the private-dimension cost, over the predicting cells."""
    k = np.array(
        [dome._permitted[c.id] for c in dome.cells if not c.is_boundary], dtype=np.float64
    )
    sums = np.array(
        [dome.stalk_sums[c.id] for c in dome.cells if not c.is_boundary], dtype=np.float64
    )
    n = dome.shape.n
    return {
        "k_v_median": float(np.median(k)),
        "k_v_max": float(k.max()),
        "k_v_saturated_cells": int((k >= n).sum()),
        "predicting_cells": int(len(k)),
        "stalk_sum_median": float(np.median(sums)),
        "private_dim_zero_cells": int((sums >= n).sum()),
        "private_dim_total": int(np.maximum(0.0, n - sums).sum()),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--widths", type=int, nargs="+", default=[3, 4, 6, 8, 10])
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--out", type=Path, default=_HERE / "546-construction.json")
    args = p.parse_args()

    record = {"issue": 546, "reading": "construction, rebuilt at each width", "rows": []}
    for width in args.widths:
        for seed in args.seeds:
            env, agent = build_at(width, seed)
            try:
                chains = t2.rim_chains(agent.dome)
                read = angles.read_surface(agent, chains, f"construction m={width} seed {seed}")
                row = {
                    "interior_m": width,
                    "seed": seed,
                    "chains": len(chains),
                    "mask": mask_read(agent.dome),
                    "composed_er": read["composed_er"],
                    "cos_all": read["cos_all"],
                    "cos_leading_per_hop": read["cos_leading_per_hop"],
                }
                record["rows"].append(row)
                e = read["composed_er"]
                print(
                    f"  m={width:>2} seed {seed}: ER med {e['median']:.4f} "
                    f"p90 {e['p90']:.4f} max {e['max']:.4f} | k_v med "
                    f"{row['mask']['k_v_median']:.0f} sat {row['mask']['k_v_saturated_cells']}"
                    f"/{row['mask']['predicting_cells']}",
                    flush=True,
                )
            finally:
                env.close()
            args.out.write_text(json.dumps(record, indent=1))
    print(f"wrote {args.out.name}", flush=True)


if __name__ == "__main__":
    main()
