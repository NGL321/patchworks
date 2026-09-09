"""B40 (#603): are the 90 cycle-less wide edges bridges, or stranded by the radius?

`b40_enumerate.py` found 90 of 194 wide interior edges on **no** cycle of length
`<= 6`, by exhaustive enumeration rather than off a basis. B34's criterion counts
directions clearing threshold on the short local cycles through `e`, so on those 90
it counts over an empty set and every one of them floors to `m_e = 1` whatever the
training route does. That is half the wide population, and it is worth knowing
which of two very different things it means:

* **Bridges.** The edge lies on no cycle *at any length* -- removing it disconnects
  the wide subgraph. Nothing about locality, nothing about training, and no route
  can rescue it. The criterion is simply undefined there, and B34 needs a second
  rule for acyclic edges.
* **Stranded by the bound.** The edge lies on cycles, but only long ones, so
  ADR-0011's radius-2 locality is what excludes it. Then the finding indicts the
  locality bound rather than B34's criterion, and widening the radius is a lever.

The two call for different repairs, so this separates them: Tarjan bridges on the
wide subgraph, then, for the non-bridges, the shortest cycle actually through each
edge at unbounded length.

Usage::

    PYTHONPATH=src python prototypes/cold-start/T6/b40_stranded.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter, deque
from pathlib import Path

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


b29 = _load("b40s_b29", _HERE / "b29_holonomy.py")
b33 = _load("b40s_b33", _HERE / "b33_coexist.py")
b40e = _load("b40s_enum", _HERE / "b40_enumerate.py")


def bridges(adj, edge_ids) -> set[int]:
    """Tarjan's bridge-finding, iterative so a deep graph cannot blow the stack."""
    disc: dict[int, int] = {}
    low: dict[int, int] = {}
    found: set[int] = set()
    timer = 0
    for root in list(adj):
        if root in disc:
            continue
        # (cell, parent_edge, iterator over incident edges)
        stack = [(root, -1, iter(adj[root]))]
        disc[root] = low[root] = timer
        timer += 1
        while stack:
            cell, parent_edge, it = stack[-1]
            advanced = False
            for other, eid in it:
                if eid == parent_edge:
                    continue
                if other in disc:
                    low[cell] = min(low[cell], disc[other])
                    continue
                disc[other] = low[other] = timer
                timer += 1
                stack.append((other, eid, iter(adj[other])))
                advanced = True
                break
            if not advanced:
                stack.pop()
                if stack:
                    up = stack[-1][0]
                    low[up] = min(low[up], low[cell])
                    if low[cell] > disc[up]:
                        found.add(parent_edge)
    return found


def shortest_cycle_len(adj, dome, edge_id: int, cap: int = 40) -> int:
    """Length of the shortest cycle through `edge_id`, or 0 if none within `cap`.

    BFS for the shortest `v`-`u` path in the wide subgraph with `edge_id` removed;
    the cycle is that path plus the edge itself.
    """
    edge = dome.edges[edge_id]
    start, goal = edge.u, edge.v
    seen = {start: 0}
    queue = deque([start])
    while queue:
        cell = queue.popleft()
        if seen[cell] >= cap:
            break
        for other, eid in adj[cell]:
            if eid == edge_id or other in seen:
                continue
            seen[other] = seen[cell] + 1
            if other == goal:
                return seen[other] + 1
            queue.append(other)
    return 0


def main() -> None:
    env, agent = b33.arms_mod.build_arm(b33.ARM, 42)
    dome = agent.dome
    adj, wide_edges = b40e.wide_adjacency(dome)
    prev = json.loads((_HERE / "603-enumerate-seed42.json").read_text(encoding="utf-8"))
    cycleless = [r["edge"] for r in prev["rows"] if r["cycles"] == 0]

    br = bridges(adj, wide_edges)
    br_wide = sorted(e for e in br if e in set(wide_edges))

    rows = []
    for edge_id in cycleless:
        is_bridge = edge_id in br
        girth = 0 if is_bridge else shortest_cycle_len(adj, dome, edge_id)
        rows.append(
            {
                "edge": int(edge_id),
                "m": int(dome.edges[edge_id].m),
                "bridge": bool(is_bridge),
                "shortest_cycle": int(girth),
            }
        )

    n_bridge = sum(1 for r in rows if r["bridge"])
    stranded = [r for r in rows if not r["bridge"]]
    report = {
        "ticket": 603,
        "reading": "are the cycle-less wide edges bridges, or stranded by the radius",
        "surface": b33.ARM,
        "seed": 42,
        "wide_interior_edges": len(wide_edges),
        "cycleless_at_len6": len(cycleless),
        "bridges_in_wide_subgraph": len(br_wide),
        "cycleless_that_are_bridges": n_bridge,
        "cycleless_stranded_by_length": len(stranded),
        "stranded_shortest_cycle_hist": {
            str(k): v
            for k, v in sorted(Counter(r["shortest_cycle"] for r in stranded).items())
        },
        "rows": rows,
    }

    print(f"wide interior edges                {report['wide_interior_edges']}")
    print(f"bridges in the wide subgraph       {report['bridges_in_wide_subgraph']}")
    print(f"cycle-less at length <= 6          {report['cycleless_at_len6']}")
    print(f"  of those, true bridges           {report['cycleless_that_are_bridges']}")
    print(f"  of those, stranded by length     {report['cycleless_stranded_by_length']}")
    print(f"  their shortest cycle: {report['stranded_shortest_cycle_hist']}")

    out = _HERE / "603-stranded-seed42.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out.name}")


if __name__ == "__main__":
    main()
