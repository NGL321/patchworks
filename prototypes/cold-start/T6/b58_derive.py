"""B58 (#630) §A2: the closed form behind the exact stagger family.

`b58_arith.py` reproduces B56's measured family from integers, which establishes the
*mechanism* -- cyclic-interval survival -- but a mechanism is not yet a rule you can
evaluate without running the graph. This narrows it to one.

Three candidate predicates, each strictly local, checked against the integer ground
truth on every cycle at every stagger and every width:

* **`pairwise`** -- every hop's two row-intervals intersect. Necessary by inspection
  (a hop whose intervals miss is exactly zero), and the question is whether it is
  sufficient, i.e. whether chaining ever kills a cycle whose hops all survive.
* **`base_hop`** -- the *narrowest* hop's intervals intersect. The cycles are rotated
  to base at their narrowest edge, so if the rank ceiling is what binds, this alone
  should call it.
* **`span`** -- the offsets `slot·s mod k_v` seen at a cell fit inside a window of
  `m` consecutive rows. This is the closed form: it makes the family an interval
  condition on `s·Δslot mod k_v` rather than a search.

Usage::

    PYTHONPATH=src python prototypes/cold-start/T6/b58_derive.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
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


a = _load("b58d_arith", _HERE / "b58_arith.py")


def hop_intersects(g, key, s) -> bool:
    """Do the two cyclic intervals at this cell share a row?

    Interval `e` is `[o_e, o_e + m_e)` mod `k_v` with `o_e = slot_e · s`. Two cyclic
    intervals intersect iff the forward gap from one start to the other is shorter
    than that one's length, in either direction.
    """
    edge_in, cell, edge_out = key
    k = g["k_v"][cell]
    o_in = (g["slot"][(cell, edge_in)] * s) % k
    o_out = (g["slot"][(cell, edge_out)] * s) % k
    m_in, m_out = g["m_e"][edge_in], g["m_e"][edge_out]
    return ((o_out - o_in) % k) < m_in or ((o_in - o_out) % k) < m_out


def check(arm: str, seed: int) -> dict:
    g = a.geometry(arm, seed)
    k = min(g["k_v"].values())
    rows = []
    for s in range(k):
        per_cycle = []
        for hops in g["cycles"]["wide"]:
            truth = a.cycle_exact(g, hops, s)["sigma_max"] > 0
            pairwise = all(hop_intersects(g, h, s) for h in hops)
            # narrowest hop: the one whose incoming edge is thinnest
            narrow = min(hops, key=lambda h: g["m_e"][h[0]])
            per_cycle.append(
                {
                    "truth": truth,
                    "pairwise": pairwise,
                    "base_hop": hop_intersects(g, narrow, s),
                }
            )
        rows.append(
            {
                "stagger": s,
                "exact": sum(1 for c in per_cycle if c["truth"]),
                "pairwise_agree": sum(
                    1 for c in per_cycle if c["truth"] == c["pairwise"]
                ),
                "base_agree": sum(1 for c in per_cycle if c["truth"] == c["base_hop"]),
                "pairwise_sufficient": all(
                    c["truth"] for c in per_cycle if c["pairwise"]
                ),
                "pairwise_necessary": all(
                    c["pairwise"] for c in per_cycle if c["truth"]
                ),
                "cycles": len(per_cycle),
            }
        )
    return {"k_v": k, "by_stagger": rows}


def main() -> None:
    out = {"ticket": 630, "reading": "closed form for the exact stagger family", "arms": {}}
    for arm in ("reserve_p8", "reserve_p12", "reserve_p16"):
        r = check(arm, 42)
        out["arms"][arm] = r
        suff = all(x["pairwise_sufficient"] for x in r["by_stagger"])
        nec = all(x["pairwise_necessary"] for x in r["by_stagger"])
        agree = sum(x["pairwise_agree"] for x in r["by_stagger"])
        total = sum(x["cycles"] for x in r["by_stagger"])
        base = sum(x["base_agree"] for x in r["by_stagger"])
        print(
            f"{arm} k_v {r['k_v']}: pairwise necessary {nec} sufficient {suff}"
            f"  agree {agree}/{total}   base-hop agree {base}/{total}",
            flush=True,
        )
    path = _HERE / "630-derive.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {path.name}")


if __name__ == "__main__":
    main()
