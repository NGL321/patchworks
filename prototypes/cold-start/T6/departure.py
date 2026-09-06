"""T6 (#555) item 1b: why the built stack reads below its own generic prediction.

The rebuilt stack reads composed ER **1.5933** where [#540](#540)'s instrument,
run at the *same realised per-hop widths and the same `k_v`*, predicts **2.0637**.
#540 named that gap its own falsifier: *"if a surface is found whose composed ER
departs from the generic prediction at its own per-hop widths, §2's whole table is
wrong and the lever ranking must be re-derived."* Before ruling on that, the gap
has to be attributed, because the generic instrument makes **two** idealisations
at once and only one of them is the genericity claim:

1. the two carried subspaces at each relay cell are **Haar** `m`-frames in `k_v`;
2. each map is an **exact co-isometry** on its active block -- every singular
   value 1, which is ADR-0032's band taken to its limit.

Idealisation 2 is measurably false at construction and gets worse with width:
`sigma_min/sigma_max` over a map's active block reads 0.444 at today's widths and
**0.194** at the stack's. So this walks the ladder:

* ``generic``   -- Haar `V`, unit `sigma`   (#540's number)
* ``real_V``    -- the surface's own `V`, unit `sigma`
* ``real_all``  -- the surface's own `V` and `sigma` (the truth, and equal to
                   ``composed_er`` up to the `U`-rotations already in it)

Usage::

    PYTHONPATH=src python prototypes/cold-start/T6/departure.py
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
stack = _load("t6_stack", _HERE / "stack.py")

import construction_grading as cg  # noqa: E402


@torch.no_grad()
def ladder(agent, chains, rng_seed: int = 0) -> dict:
    """The three predictors above, chain by chain, on one surface."""
    dome, maps = agent.dome, agent.sheaf.maps
    rng = np.random.default_rng(rng_seed)
    out = {k: [] for k in ("generic", "real_V", "real_all", "truth")}
    for chain in chains:
        comp = {k: None for k in out}
        for edge_in, cell, edge_out in cg.hops_of(dome, tuple(chain["edges"])):
            k = int(dome.restriction_mask(edge_in, cell).sum())
            u_i, v_i, s_i = angles.carried(dome, maps, edge_in, cell)
            u_o, v_o, s_o = angles.carried(dome, maps, edge_out, cell)
            mi, mo = v_i.shape[1], v_o.shape[1]

            hv_i = np.linalg.qr(rng.standard_normal((k, mi)))[0]
            hv_o = np.linalg.qr(rng.standard_normal((k, mo)))[0]
            hu_i = np.linalg.qr(rng.standard_normal((mi, mi)))[0]
            hu_o = np.linalg.qr(rng.standard_normal((mo, mo)))[0]

            hop = {
                "generic": hu_o @ (hv_o.T @ hv_i) @ hu_i.T,
                # the surface's own carried subspaces and own U-rotations,
                # with every singular value set to 1
                "real_V": u_o @ (v_o.T @ v_i) @ u_i.T,
                # ... and now with the surface's own singular values
                "real_all": (u_o * s_o) @ (v_o.T @ v_i) @ (u_i * s_i).T,
            }
            for name, h in hop.items():
                comp[name] = h if comp[name] is None else h @ comp[name]

        for name in ("generic", "real_V", "real_all"):
            out[name].append(angles.effective_rank(np.linalg.svd(comp[name], compute_uv=False)))
        out["truth"].append(angles.chain_read(dome, maps, chain)["composed_er"])

    return {
        name: {
            "median": float(np.median(v)),
            "mean": float(np.mean(v)),
            "p90": float(np.quantile(v, 0.90)),
            "max": float(np.max(v)),
        }
        for name, v in out.items()
    }


def sigma_read(agent) -> dict:
    """The band on each map's **active block** -- what idealisation 2 assumes away."""
    dome, maps = agent.dome, agent.sheaf.maps
    ratios, spreads = [], []
    for e in dome.edges:
        for cell in (e.u, e.v):
            if dome.cells[cell].is_boundary:
                continue
            _, _, s = angles.carried(dome, maps, e.id, cell)
            if len(s) < 2:
                continue
            ratios.append(float(s.min() / max(s.max(), 1e-300)))
            spreads.append(float(np.std(s) / max(np.mean(s), 1e-300)))
    r, sp = np.array(ratios), np.array(spreads)
    return {
        "maps": int(len(r)),
        "sigma_min_over_max_median": float(np.median(r)),
        "sigma_min_over_max_p10": float(np.quantile(r, 0.10)),
        "sigma_cv_median": float(np.median(sp)),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rungs", nargs="+", default=["today", "b_invariant"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=_HERE / "555-departure.json")
    args = p.parse_args()

    record = {"issue": 555, "reading": "attribution of the generic-vs-built gap", "rows": []}
    for rung in args.rungs:
        cap, lateral_m, per_edge = stack.RUNGS[rung]
        env, agent = stack.build_stack(args.seed, cap=cap, lateral_m=lateral_m, per_edge=per_edge)
        try:
            chains = t2.rim_chains(agent.dome)
            row = {
                "rung": rung,
                "seed": args.seed,
                "ladder": ladder(agent, chains),
                "sigma": sigma_read(agent),
            }
            record["rows"].append(row)
            lad = row["ladder"]
            print(
                f"  {rung:>12}: generic {lad['generic']['median']:.4f} -> real_V "
                f"{lad['real_V']['median']:.4f} -> real_all {lad['real_all']['median']:.4f} "
                f"(truth {lad['truth']['median']:.4f}) | sigma min/max med "
                f"{row['sigma']['sigma_min_over_max_median']:.4f} cv "
                f"{row['sigma']['sigma_cv_median']:.4f}",
                flush=True,
            )
        finally:
            env.close()
        args.out.write_text(json.dumps(record, indent=1))
    print(f"wrote {args.out.name}", flush=True)


if __name__ == "__main__":
    main()
