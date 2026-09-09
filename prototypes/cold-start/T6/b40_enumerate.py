"""B40 (#603): enumerate the short local cycles *through each edge*, not a basis.

`b40_census.py` read the per-edge count off B29's wide cycle **basis** and found 90
of 194 wide interior edges on no cycle at all. That number is not trustworthy as a
statement about B34's criterion, and this module exists to say why and to replace it.

A cycle basis is a *minimal spanning set* -- one cycle per non-tree edge of a BFS
forest. Which edges land on many basis cycles and which land on none is a property
of where BFS happened to root, not of the graph's cycle structure. B34's criterion
does not read a basis: it counts directions clearing threshold on **the short local
cycles through `e`**, which is every such cycle, however many there are.

So this enumerates them directly. For each wide interior edge `(u, v)`, every cycle
through it is a `u`-`v` path in the wide subgraph with `(u, v)` removed; the short
ones are the paths of length `<= max_len - 1`. Locality is B33's `RADIUS` test
unchanged -- the cycle must lie wholly inside some cell's closed 2-hop
neighbourhood -- so the admitted set is comparable with what the holonomy term
descends on.

Both numbers go in the readout. The basis count is what a session reusing B29's
instrument would have seen; the enumerated count is what the criterion actually has
to work with.

Usage::

    PYTHONPATH=src python prototypes/cold-start/T6/b40_enumerate.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

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


b29 = _load("b40e_b29", _HERE / "b29_holonomy.py")
b33 = _load("b40e_b33", _HERE / "b33_coexist.py")

#: Cycles up to this many edges. Radius 2 admits cycles of length up to 5-6 in
#: practice; 6 is taken as the ceiling so the enumeration terminates quickly and
#: nothing longer than the locality bound could admit is counted.
MAX_LEN = 6


def wide_adjacency(dome, min_m: int = 2):
    """The wide subgraph as `cell -> [(other, edge_id), ...]`, interior edges only."""
    interior, edge_ids = b29.hr.interior_graph(dome)
    kept = [e for e in edge_ids if dome.edges[e].m >= min_m]
    adj: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for edge_id in kept:
        edge = dome.edges[edge_id]
        adj[edge.u].append((edge.v, edge_id))
        adj[edge.v].append((edge.u, edge_id))
    return adj, kept


def cycles_through(adj, dome, edge_id: int, max_len: int = MAX_LEN):
    """Every cycle through `edge_id` with at most `max_len` edges, as edge tuples.

    Depth-first over simple `v`-`u` paths avoiding `edge_id`; each cycle is
    canonicalised by its frozenset of edges so the two traversal directions and the
    choice of starting point collapse to one cycle.
    """
    edge = dome.edges[edge_id]
    start, goal = edge.u, edge.v
    found: dict[frozenset, tuple[int, ...]] = {}

    def walk(cell: int, used_edges: list[int], seen_cells: set[int]) -> None:
        if len(used_edges) + 1 > max_len:
            return
        for other, eid in adj[cell]:
            if eid == edge_id or eid in used_edges:
                continue
            if other == goal:
                cycle = tuple(used_edges + [eid, edge_id])
                found.setdefault(frozenset(cycle), cycle)
                continue
            if other in seen_cells:
                continue
            walk(other, used_edges + [eid], seen_cells | {other})

    walk(start, [], {start})
    return list(found.values())


def main() -> None:
    env, agent = b33.arms_mod.build_arm(b33.ARM, 42)
    dome = agent.dome
    adj, wide_edges = wide_adjacency(dome)

    def is_local(cycle) -> bool:
        return bool(b33.local_cycles(dome, [cycle], b33.RADIUS))

    rows = []
    all_local: dict[frozenset, tuple[int, ...]] = {}
    for edge_id in wide_edges:
        cycles = cycles_through(adj, dome, edge_id)
        local = [c for c in cycles if is_local(c)]
        for c in local:
            all_local.setdefault(frozenset(c), c)
        rows.append(
            {
                "edge": int(edge_id),
                "m": int(dome.edges[edge_id].m),
                "cycles": len(cycles),
                "local": len(local),
                "shortest": min((len(c) for c in cycles), default=0),
            }
        )

    def q(vals):
        arr = np.asarray(vals, dtype=float)
        if arr.size == 0:
            return {"n": 0}
        return {
            "n": int(arr.size),
            "min": float(arr.min()),
            "q1": float(np.percentile(arr, 25)),
            "median": float(np.median(arr)),
            "q3": float(np.percentile(arr, 75)),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
        }

    counts = [r["cycles"] for r in rows]
    locals_ = [r["local"] for r in rows]
    report = {
        "ticket": 603,
        "reading": "short local cycles enumerated per edge, not read off a basis",
        "surface": b33.ARM,
        "seed": 42,
        "radius": b33.RADIUS,
        "max_len": MAX_LEN,
        "wide_interior_edges": len(wide_edges),
        "distinct_local_cycles": len(all_local),
        "per_edge_all": q(counts),
        "per_edge_local": q(locals_),
        "edges_with_no_cycle": sum(1 for r in rows if r["cycles"] == 0),
        "edges_with_no_local": sum(1 for r in rows if r["local"] == 0),
        "edges_local_ge2": sum(1 for r in rows if r["local"] >= 2),
        "edges_local_ge4": sum(1 for r in rows if r["local"] >= 4),
        "histogram_local": {
            str(k): v for k, v in sorted(Counter(locals_).items())
        },
        "rows": rows,
    }

    print(f"wide interior edges           {report['wide_interior_edges']}")
    print(f"distinct short local cycles   {report['distinct_local_cycles']}")
    print(f"edges on no cycle at all      {report['edges_with_no_cycle']}")
    print(f"edges on no *local* cycle     {report['edges_with_no_local']}")
    print(f"edges with >=2 local cycles   {report['edges_local_ge2']}")
    print(f"edges with >=4 local cycles   {report['edges_local_ge4']}")
    pa, pl = report["per_edge_all"], report["per_edge_local"]
    print(
        f"cycles per edge   (all)  min {pa['min']:.0f} median {pa['median']:.1f} "
        f"q3 {pa['q3']:.1f} max {pa['max']:.0f} mean {pa['mean']:.2f}"
    )
    print(
        f"cycles per edge (local)  min {pl['min']:.0f} median {pl['median']:.1f} "
        f"q3 {pl['q3']:.1f} max {pl['max']:.0f} mean {pl['mean']:.2f}"
    )
    print(f"histogram (local) {report['histogram_local']}")

    out = _HERE / "603-enumerate-seed42.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out.name}")


if __name__ == "__main__":
    main()
