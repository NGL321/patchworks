"""T6 (#560/B13) item 1: sweep the reserve size `p` at construction.

[B12/#556](#556) invented `p` and read it at one value; [B11](#555) built and
trained that one value. `p` is a whole curve, and it moves two quantities in the
same direction at once:

* **composed rank** — [B1](#537)'s law says composed ER is a function of
  `(m, k_v, hops)`, and the reserve mask sets `k_v = n - p`, so larger `p`
  narrows the ambient and *raises* rank at fixed `m`;
* **the `dim H^0` floor** — `p_v = p` flat at every predicting cell, so the floor
  is `p x |predicting|` and rises linearly.

They cannot rise forever, because the lanes must fit: `allocate_lane_widths` is
capped at `n - p`, so as `p` grows the cap binds on more edges, `m` falls, and
`m / k_v` must turn over. This script finds the turn.

Every figure here is a **construction** figure, and [B11](#555)'s central finding
is that construction is the wrong number to rule on — `trained_arms.py` carries
the peak and its neighbours to a horizon. This sweep only says *where to look*.

Usage::

    PYTHONPATH=src python prototypes/cold-start/T6/sweep_p.py --p 0 4 8 12 16 20
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

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
t4_trained = _load("t4_trained", _T4 / "trained.py")
arms_mod = _load("t6_arms", _HERE / "arms.py")
departure = _load("t6_departure", _HERE / "departure.py")

from patchworks.graph import NODE_STALK_DIM, EdgeKind, build_graph  # noqa: E402
from untrained_fixed_point import dome_named  # noqa: E402


def cap_binding(p: int) -> dict:
    """How hard the `n - p` lane ceiling bites, against the uncapped allocation.

    The comparison is against the *same* water-fill run at `cap = n`, so this
    isolates the ceiling from the budget. `binding` counts interior edges whose
    width the ceiling actually lowered.
    """
    base, _ = dome_named("real")
    spec = replace(base, privacy_budget=arms_mod.BUDGET)
    proto = build_graph(spec)
    cells = list(proto.cells)
    bare = [replace(e, m=0) if e.kind is EdgeKind.INTERIOR else e for e in proto.edges]
    free = arms_mod.allocate_capped(cells, bare, spec, cap=NODE_STALK_DIM)
    held = arms_mod.allocate_capped(cells, bare, spec, cap=NODE_STALK_DIM - p)
    interior = [
        (f.m, h.m) for f, h in zip(free, held) if f.kind is EdgeKind.INTERIOR
    ]
    lowered = [(f, h) for f, h in interior if h < f]
    at_ceiling = [h for _, h in interior if h == NODE_STALK_DIM - p]
    return {
        "interior_edges": len(interior),
        "binding": len(lowered),
        "at_ceiling": len(at_ceiling),
        "width_lost_total": int(sum(f - h for f, h in lowered)),
        "uncapped_max": int(max(f for f, _ in interior)),
        "capped_max": int(max(h for _, h in interior)),
    }


def read_p(p: int, seed: int) -> dict:
    started = time.time()
    arm = f"reserve_p{p}"
    env, agent = arms_mod.build_arm(arm, seed)
    try:
        dome = agent.dome
        chains = t2.rim_chains(dome)
        read = angles.read_surface(agent, chains, f"{arm} seed {seed}")
        row = {
            "p": p,
            "arm": arm,
            "seed": seed,
            "chains": len(chains),
            "privacy": arms_mod.privacy_read(dome, p),
            "widths": arms_mod.widths_read(dome, chains),
            "cap_binding": cap_binding(p),
            "generic": arms_mod.generic_at(dome, chains),
            "cap": t4_trained.cap_read(agent),
            "sigma": departure.sigma_read(agent),
            "composed_er": read["composed_er"],
            "cos_leading_per_hop": read["cos_leading_per_hop"],
            "cos_second_per_hop": read["cos_second_per_hop"],
            "s2_over_s1": read["s2_over_s1"],
            "sigma_min_over_max": read["sigma_min_over_max"],
            "elapsed_minutes": (time.time() - started) / 60.0,
        }
        e, g, pr, w, cb = (
            row["composed_er"],
            row["generic"],
            row["privacy"],
            row["widths"],
            row["cap_binding"],
        )
        print(
            f"  p={p:>2}: k_v {pr['k_v_median']:.0f} | built ER {e['median']:.4f} "
            f"p90 {e['p90']:.4f} max {e['max']:.4f} | generic {g['median']:.4f} "
            f"| floor {pr['private_dim_total']} | lanes {w['interior_m_min']}-"
            f"{w['interior_m_max']} modal {w['modal']} | cap binds {cb['binding']}"
            f"/{cb['interior_edges']} (lost {cb['width_lost_total']}) "
            f"| lead {row['cos_leading_per_hop']['median']:.4f} "
            f"2nd {row['cos_second_per_hop']['median']:.4f} "
            f"({row['elapsed_minutes']:.1f} min)",
            flush=True,
        )
        return row
    finally:
        env.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--p", type=int, nargs="+", default=[0, 4, 8, 12, 16, 20])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--out", type=Path, default=_HERE / "560-sweep-p-construction.json")
    args = ap.parse_args()

    record = {
        "issue": 560,
        "reading": "reserve size p swept at construction, budget 63",
        "budget": arms_mod.BUDGET,
        "n": NODE_STALK_DIM,
        "rows": [],
    }
    for seed in args.seeds:
        for p in args.p:
            record["rows"].append(read_p(p, seed))
            args.out.write_text(json.dumps(record, indent=1))
    print(f"wrote {args.out.name}", flush=True)


if __name__ == "__main__":
    main()
