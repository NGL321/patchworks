"""B42 (#605): is the flat bundle's exactness the same fact as its lane overlap?

`b42_reserve.py` shows the reserve does not have to give: a frame built inside the
permitted `k_v` window telescopes exactly through both the mask and the band. But
B40's construction gives *every* edge at a cell the **same leading rows**,
`frames[cell][:edge.m]`. A degree-9 cell's nine edges therefore all read rows
`0..m-1` of one frame -- nested, not distributed.

That is the shape [B17](https://github.com/NGL321/patchworks/issues/565) indicted as
a *wide empty pipe*, and the map's own fog names it B42's central risk: *overlap
genuinely shared versus overlap bought by shrinking the ambient*.

This instrument asks whether the overlap is a **choice** or the **mechanism**. A hop
at a cell is `S_out R R^T S_in^T = S_out S_in^T`, where `S` selects rows. If the two
edges take the same rows that is a rectangular identity -- exact, unit gain. If they
take **disjoint** rows it is exactly **zero** -- nothing transports at all.

So it sweeps a per-edge row offset. At ``stagger = 0`` every edge takes rows
`0..m-1` (B40). At ``stagger = s`` the `i`-th incident edge takes rows
`(i*s) .. (i*s + m - 1)` mod `k_v`, spreading the edges across the window.

What comes back decides whether "one frame per cell" buys path-independence at the
price of every edge at a cell seeing the same thing.

Usage::

    PYTHONPATH=src python prototypes/cold-start/T6/b42_stagger.py
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


b33 = _load("b42s_b33", _HERE / "b33_coexist.py")
b29 = _load("b42s_b29", _HERE / "b29_holonomy.py")

from patchworks.restriction import pair_index  # noqa: E402
import construction_grading as cg  # noqa: E402


def build_staggered(seed: int, stagger: int):
    """Reserved frame per cell, with the `i`-th incident edge offset by `i*stagger`."""
    env, agent = b33.arms_mod.build_arm(b33.ARM, seed)
    dome = agent.dome
    maps = agent.sheaf.maps
    n = dome.shape.n
    gen = torch.Generator().manual_seed(seed + 9001)
    frames, widths = {}, {}
    for cell in dome.predicting:
        width = int(dome._permitted[cell])
        a = torch.randn(width, width, generator=gen, dtype=torch.float64)
        q, r = torch.linalg.qr(a)
        q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)
        full = torch.zeros(n, n, dtype=torch.float64)
        full[:width, :width] = q
        frames[cell] = full
        widths[cell] = width
    with torch.no_grad():
        for cell in dome.predicting:
            width = widths[cell]
            for slot, edge_id in enumerate(dome.incident[cell]):
                edge = dome.edges[edge_id]
                try:
                    idx = pair_index(edge_id, cg.side_of(dome, edge_id, cell))
                except Exception:
                    continue
                rows = [(slot * stagger + j) % width for j in range(edge.m)]
                block = frames[cell][rows]
                t = maps.maps[idx]
                rr = min(block.shape[0], t.shape[0])
                cc = min(block.shape[1], t.shape[1])
                t[:rr, :cc] = block[:rr, :cc].to(t.dtype)
    return dome, agent, maps


def distinctness(dome, maps) -> float:
    """Median over cells of the mean |cos| between incident edges' row spaces.

    1.0 means every edge at the cell reads the same subspace -- one pipe wearing
    nine hats. 0.0 means the edges read orthogonal subspaces.
    """
    vals = []
    n = dome.shape.n
    for cell in dome.predicting:
        blocks = []
        for edge_id in dome.incident[cell]:
            try:
                idx = pair_index(edge_id, cg.side_of(dome, edge_id, cell))
            except Exception:
                continue
            # The slot is a padded `[max_m, stalk]` buffer; only the leading
            # `m_e` rows and `n` columns are the edge's own map.
            m_e = dome.edges[edge_id].m
            b = maps.maps[idx].detach().to(torch.float64)[:m_e, :n]
            if b.shape[0] == 0 or b.norm() < 1e-12:
                continue
            q, _ = torch.linalg.qr(b.T)
            blocks.append(q)
        if len(blocks) < 2:
            continue
        pair_cos = []
        for i in range(len(blocks)):
            for j in range(i + 1, len(blocks)):
                s = torch.linalg.svdvals(blocks[i].T @ blocks[j])
                pair_cos.append(float(s.mean()))
        if pair_cos:
            vals.append(sum(pair_cos) / len(pair_cos))
    vals.sort()
    return vals[len(vals) // 2] if vals else float("nan")


def main() -> None:
    seed = 42
    out: dict = {"ticket": 605, "surface": b33.ARM, "seed": seed, "stagger": {}}
    wide = None
    print(f"{'stagger':>8}  {'ident':>8}  {'chan':>8}  {'sigma':>10}  {'overlap':>8}")
    for stagger in (0, 1, 2, 3, 4, 6, 8):
        dome, _agent, maps = build_staggered(seed, stagger)
        if wide is None:
            wide = b29.cycles_of(dome)["wide"]
        s = b29.surface_read(dome, maps, wide, f"stagger{stagger}")["subsets"]["wide"]
        row = {
            "identification": s["identification"]["median"],
            "channel_return": s["channel_return"]["median"],
            "sigma_max": s["sigma_max"]["median"],
            "edge_overlap": distinctness(dome, maps),
        }
        out["stagger"][str(stagger)] = row
        print(
            f"{stagger:>8}  {row['identification']:>8.4f}  {row['channel_return']:>8.4f}"
            f"  {row['sigma_max']:>10.3e}  {row['edge_overlap']:>8.4f}"
        )
    path = _HERE / f"605-stagger-seed{seed}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {path.name}")


if __name__ == "__main__":
    main()
