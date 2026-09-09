"""B40 (#603): the flat bundle is exactly flat, and which commitment breaks it.

`b40_routes.py` reads arm 4 *after* `maps.project()`, because that is the surface the
architecture actually has. This instrument reads it before, and separates the two
things the projection does, because the answer decides which commitment the finding
indicts.

`project()` enforces two constraints at once:

* the **dimension mask** -- columns beyond `k_v` are zeroed (#597's reserve mask,
  `k_v = n - p`, which is 20 against `n = 32` on this surface);
* **ADR-0032's band** on each surviving block.

The map's destination names ADR-0032 as *"unchanged and load-bearing"*, and the
reserve is a separate commitment, so which one costs the exactness matters.

What it finds, on `reserve_p12` seed 42:

| configuration | `identification` | `channel_return` | `sigma_max` |
|---|---|---|---|
| frames only | **0.0000** | **1.0000** | **1.000** |
| mask only | 0.6123 | 0.8645 | 2.91e-01 |
| mask + band | 0.0386 | 0.9998 | 7.72e-04 |

So the construction is **exactly** cycle-consistent at unit gain, at the rig's actual
`m_e < n`; the **mask** is what breaks it; and the band then repairs most of the
flatness at three orders of amplitude.

Why it is exact at `m_e < n`: every edge at a cell takes *rows of the same frame*, so
a hop is the top-left `m_out x m_in` block of `R_c R_c^T = I` -- a rectangular
identity -- and the cycle telescopes whatever the widths are.

Scope: this is path-independence in B27's retained **coherent-region** form, on local
cycles. It says nothing about path-independence *at distance*, which B27 struck as
unavailable.

Usage::

    PYTHONPATH=src python prototypes/cold-start/T6/b40_exactness.py
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


b33 = _load("b40x_b33", _HERE / "b33_coexist.py")
b29 = _load("b40x_b29", _HERE / "b29_holonomy.py")

from patchworks.restriction import pair_index  # noqa: E402
import construction_grading as cg  # noqa: E402


def build_with_frames(seed: int = 42):
    """The flat bundle installed and *not* projected."""
    env, agent = b33.arms_mod.build_arm(b33.ARM, seed)
    dome = agent.dome
    maps = agent.sheaf.maps
    n = dome.shape.n
    gen = torch.Generator().manual_seed(seed + 9001)
    frames = {}
    for cell in dome.predicting:
        a = torch.randn(n, n, generator=gen, dtype=torch.float64)
        q, r = torch.linalg.qr(a)
        frames[cell] = q * torch.sign(torch.diagonal(r)).unsqueeze(0)
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


def apply_mask_only(dome, maps) -> None:
    """Zero the columns the reserve mask forbids, without touching the band."""
    predicting = set(dome.predicting)
    with torch.no_grad():
        for edge_id, edge in enumerate(dome.edges):
            for cell in (edge.u, edge.v):
                if cell not in predicting:
                    continue
                try:
                    idx = pair_index(edge_id, cg.side_of(dome, edge_id, cell))
                except Exception:
                    continue
                maps.maps[idx][:, int(dome._permitted[cell]) :] = 0.0


def main() -> None:
    seed = 42
    out: dict = {"ticket": 603, "surface": b33.ARM, "seed": seed, "configurations": {}}
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

    dome, _agent, maps = build_with_frames(seed)
    out["n"] = int(dome.shape.n)
    out["k_v"] = int(dome._permitted[dome.predicting[0]])
    print(f"n = {out['n']}, k_v = {out['k_v']}")
    out["configurations"]["frames_only"] = read("frames only", dome, maps)

    dome, _agent, maps = build_with_frames(seed)
    apply_mask_only(dome, maps)
    out["configurations"]["mask_only"] = read(
        "mask only (columns >= k_v zeroed)", dome, maps
    )

    dome, _agent, maps = build_with_frames(seed)
    maps.project()
    out["configurations"]["mask_and_band"] = read(
        "full project() (mask + band)", dome, maps
    )

    path = _HERE / f"603-exactness-seed{seed}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {path.name}")
    print(
        "the construction is exactly flat at unit gain; the dimension mask breaks it; "
        "the band repairs most of the flatness at three orders of amplitude"
    )


if __name__ == "__main__":
    main()
