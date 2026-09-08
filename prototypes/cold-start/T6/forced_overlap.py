"""T6 (#560/B13): how much of the rank `p` buys is *forced* rather than learned.

[B1](#537)'s law says composed rank is a function of `(m, k_v, hops)`, and the
reserve mask moves it by narrowing `k_v`. This asks what that narrowing does
geometrically, because the answer prices the whole lever.

At a relay cell two lanes of widths `m_in`, `m_out` are carved out of the same
readable block of dimension `k_v`. Two subspaces of a `k_v`-dimensional space
**must** intersect in at least

    max(0, m_in + m_out - k_v)

dimensions, whatever the maps learn. That is the *forced* overlap: rank the
composite gets from dimension-counting rather than from anything the surface
did. When it reaches `min(m_in, m_out)` the two lanes coincide by construction,
every principal angle is zero, and the hop is an isometry — the composite then
reads effective rank exactly `m`, which is what `p >= 18` does.

So this separates "transport learned to carry more than one direction" from
"the ambient got too small for the lanes to differ".

Usage::

    PYTHONPATH=src python prototypes/cold-start/T6/forced_overlap.py
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
_T0, _T2 = (_HERE.parent / n for n in ("T0", "T2"))
for extra in (_ROOT / "prototypes" / "chart-double-duty-166", _ROOT / "benchmarks", _T0):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


t2 = _load("t2_run", _T2 / "run.py")
arms_mod = _load("t6_arms", _HERE / "arms.py")

import construction_grading as cg  # noqa: E402


def read_p(p: int, seed: int = 42) -> dict:
    env, agent = arms_mod.build_arm(f"reserve_p{p}", seed)
    try:
        dome = agent.dome
        chains = t2.rim_chains(dome)
        forced, ratio, coincident, hops = [], [], 0, 0
        for chain in chains:
            for edge_in, cell, edge_out in cg.hops_of(dome, tuple(chain["edges"])):
                k = int(dome.restriction_mask(edge_in, cell).sum())
                mi = min(dome.edges[edge_in].m, k)
                mo = min(dome.edges[edge_out].m, k)
                f = max(0, mi + mo - k)
                forced.append(f)
                ratio.append(f / min(mi, mo))
                coincident += int(f >= min(mi, mo))
                hops += 1
        forced_a, ratio_a = np.array(forced), np.array(ratio)
        return {
            "p": p,
            "k_v": 32 - p,
            "hops": hops,
            "forced_dim_median": float(np.median(forced_a)),
            "forced_dim_max": int(forced_a.max()),
            "forced_share_median": float(np.median(ratio_a)),
            "forced_share_mean": float(ratio_a.mean()),
            "hops_with_forced_overlap": int((forced_a > 0).sum()),
            "hops_fully_coincident": coincident,
        }
    finally:
        env.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--p", type=int, nargs="+", default=[0, 4, 8, 12, 16, 20])
    ap.add_argument("--out", type=Path, default=_HERE / "560-forced-overlap.json")
    args = ap.parse_args()

    rows = []
    print(f"  {'p':>3} {'k_v':>4} {'forced dim':>11} {'forced share':>13} "
          f"{'hops w/ overlap':>16} {'coincident':>11}")
    for p in args.p:
        r = read_p(p)
        rows.append(r)
        print(
            f"  {r['p']:>3} {r['k_v']:>4} {r['forced_dim_median']:>11.1f} "
            f"{r['forced_share_median']:>12.0%} "
            f"{r['hops_with_forced_overlap']:>9}/{r['hops']:<6} "
            f"{r['hops_fully_coincident']:>11}",
            flush=True,
        )
    args.out.write_text(json.dumps({"issue": 560, "rows": rows}, indent=1))
    print(f"wrote {args.out.name}")


if __name__ == "__main__":
    main()
