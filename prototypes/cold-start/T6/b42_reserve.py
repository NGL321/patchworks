"""B42 (#605): does the privacy reserve actually have to give?

B40's `b40_exactness.py` installs an `n x n` frame per cell and takes the first
`edge.m` *rows* of it as each incident edge's map. The reserve mask then zeroes
every column at or beyond `k_v = n - p`, and exactness dies: `identification`
0.0000 -> 0.6123. B40 read that as *the privacy reserve `p` is the obstacle*.

This instrument asks whether the obstacle is `p` itself or B40's **placement of
the frame**. The reserve mask is flat (#556/#562: `k_v = n - p` everywhere) and
permits the *same leading block* on every incident edge, so the permitted window
is one fixed coordinate subspace shared by every edge at the cell. A frame built
*inside* that window is still one orthogonal frame per cell, still telescopes,
and has nothing outside `k_v` for the mask to zero.

Arms, all on `reserve_p12` (n = 32, k_v = 20, p = 12):

* ``ambient``      -- B40's construction: `n x n` frame, rows `[:m_e]`.
* ``ambient+mask`` -- the same, columns `>= k_v` zeroed. B40's 0.6123 row.
* ``reserved``     -- an orthogonal frame on the permitted `k_v` block only,
                      embedded at `[:k_v, :k_v]` with zeros elsewhere; edge rows
                      still `[:m_e]`, which now lie inside the window.
* ``reserved+mask``-- the same, mask applied. Should be a no-op if the claim holds.

If ``reserved+mask`` reads 0.0000 / 1.0000, the reserve does **not** have to give:
the flat bundle is compatible with `p` intact, and the price is not privacy but
forced lane overlap -- at most `k_v` distinct rows to serve `sum_e m_e` demand.

The overlap census is reported alongside for exactly that reason.

Usage::

    PYTHONPATH=src python prototypes/cold-start/T6/b42_reserve.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
for extra in (_ROOT / "prototypes" / "chart-double-duty-166", _ROOT / "benchmarks"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


b33 = _load("b42_b33", _HERE / "b33_coexist.py")
b29 = _load("b42_b29", _HERE / "b29_holonomy.py")
b40x = _load("b42_b40x", _HERE / "b40_exactness.py")

from patchworks.restriction import pair_index  # noqa: E402
import construction_grading as cg  # noqa: E402


def build_with_frames(seed: int = 42, *, reserved: bool = False):
    """Install one orthogonal frame per predicting cell.

    ``reserved`` builds the frame on the cell's permitted `k_v` block and embeds
    it there, leaving the private columns zero; otherwise it is B40's ambient
    `n x n` frame.
    """
    env, agent = b33.arms_mod.build_arm(b33.ARM, seed)
    dome = agent.dome
    maps = agent.sheaf.maps
    n = dome.shape.n
    gen = torch.Generator().manual_seed(seed + 9001)
    frames = {}
    for cell in dome.predicting:
        width = int(dome._permitted[cell]) if reserved else n
        a = torch.randn(width, width, generator=gen, dtype=torch.float64)
        q, r = torch.linalg.qr(a)
        q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)
        if reserved:
            full = torch.zeros(n, n, dtype=torch.float64)
            full[:width, :width] = q
            q = full
        frames[cell] = q
    with torch.no_grad():
        for edge_id, edge in enumerate(dome.edges):
            for cell in (edge.u, edge.v):
                if cell not in frames:
                    continue
                try:
                    idx = pair_index(edge_id, cg.side_of(dome, edge_id, cell))
                except Exception:
                    continue
                block = frames[cell][: edge.m]
                t = maps.maps[idx]
                rr = min(block.shape[0], t.shape[0])
                cc = min(block.shape[1], t.shape[1])
                t[:rr, :cc] = block[:rr, :cc].to(t.dtype)
    return dome, agent, maps


def overlap_census(dome) -> dict:
    """How hard the permitted window is being asked to work, per predicting cell.

    ``demand`` is `sum_e m_e`; ``supply`` is `k_v`. A cell with demand above
    supply cannot give its edges disjoint rows -- that is forced lane overlap,
    and under the reserved frame it is the whole price.
    """
    rows = []
    for cell in dome.predicting:
        demand = int(dome.stalk_sums[cell])
        supply = int(dome._permitted[cell])
        rows.append(
            {
                "cell": int(cell),
                "degree": int(dome.degrees[cell]),
                "demand": demand,
                "supply": supply,
                "forced_overlap": max(0, demand - supply),
            }
        )
    forced = [r for r in rows if r["forced_overlap"] > 0]
    ratios = sorted(r["demand"] / r["supply"] for r in rows)
    mid = len(ratios) // 2
    return {
        "cells": len(rows),
        "cells_forced": len(forced),
        "fraction_forced": len(forced) / len(rows) if rows else 0.0,
        "median_demand_over_supply": ratios[mid] if ratios else 0.0,
        "max_demand_over_supply": ratios[-1] if ratios else 0.0,
        "per_cell": rows,
    }


def main() -> None:
    seed = 42
    out: dict = {"ticket": 605, "surface": b33.ARM, "seed": seed, "configurations": {}}
    wide = None

    def read(label: str, dome, maps) -> dict:
        nonlocal wide
        if wide is None:
            wide = b29.cycles_of(dome)["wide"]
        s = b29.surface_read(dome, maps, wide, label)["subsets"]["wide"]
        row = {
            "identification": s["identification"]["median"],
            "channel_return": s["channel_return"]["median"],
            "sigma_max": s["sigma_max"]["median"],
            "cycles": s["cycles"],
        }
        print(
            f"{label:>34}: ident {row['identification']:.4f}  "
            f"chan {row['channel_return']:.4f}  sigma {row['sigma_max']:.3e}"
        )
        return row

    dome, _agent, maps = build_with_frames(seed, reserved=False)
    out["n"] = int(dome.shape.n)
    out["k_v"] = int(dome._permitted[dome.predicting[0]])
    out["p"] = out["n"] - out["k_v"]
    print(f"n = {out['n']}, k_v = {out['k_v']}, p = {out['p']}")
    out["configurations"]["ambient"] = read("ambient frame", dome, maps)

    dome, _agent, maps = build_with_frames(seed, reserved=False)
    b40x.apply_mask_only(dome, maps)
    out["configurations"]["ambient_masked"] = read("ambient frame + mask", dome, maps)

    dome, _agent, maps = build_with_frames(seed, reserved=True)
    out["configurations"]["reserved"] = read("reserved frame", dome, maps)

    dome, _agent, maps = build_with_frames(seed, reserved=True)
    b40x.apply_mask_only(dome, maps)
    out["configurations"]["reserved_masked"] = read("reserved frame + mask", dome, maps)

    dome, _agent, maps = build_with_frames(seed, reserved=True)
    maps.project()
    out["configurations"]["reserved_projected"] = read(
        "reserved frame + full project()", dome, maps
    )

    out["overlap"] = overlap_census(dome)
    ov = out["overlap"]
    print(
        f"\nforced lane overlap: {ov['cells_forced']}/{ov['cells']} cells "
        f"({ov['fraction_forced']:.0%}), demand/supply median "
        f"{ov['median_demand_over_supply']:.2f}, max {ov['max_demand_over_supply']:.2f}"
    )

    path = _HERE / f"605-reserve-seed{seed}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {path.name}")


if __name__ == "__main__":
    main()
