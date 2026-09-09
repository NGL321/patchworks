"""B40 (#603), the named risk: is there enough cycle *per edge* to split?

B29 read only **40 of 45** wide cycles local at radius 2. B34's criterion counts,
for each edge `e`, the directions whose principal-angle cosine clears threshold on
**the short local cycles through `e`** -- so the count is read per edge, not per
surface. Arm 1 (joint objective, held-out cycles) then wants to *split* those
cycles into a set the holonomy term descends on and a set it never touches.

The ticket names this as the one place B34's proposal can break on arithmetic
rather than on principle. So this runs before any arm is built, and reports the
per-edge cycle count whatever else happens.

What it prints, for the wide basis and for the radius-2 local subset:

* how many cycles each wide edge carries;
* the distribution of that count, and how many edges carry 0 or 1;
* what a 50/50 split would leave per edge, and on how many edges the held-out
  side would be **empty** -- an edge with no held-out cycle has no criterion at
  all under arm 1, and must fall back to its floor.

Usage::

    PYTHONPATH=src python prototypes/cold-start/T6/b40_census.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
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


b29 = _load("b40_b29", _HERE / "b29_holonomy.py")
b33 = _load("b40_b33", _HERE / "b33_coexist.py")


def _q(values) -> dict:
    arr = np.asarray([v for v in values if v == v], dtype=float)
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


def per_edge_counts(cycles) -> Counter:
    """How many of `cycles` pass through each edge id."""
    counts: Counter = Counter()
    for cycle in cycles:
        for edge_id in set(cycle):
            counts[edge_id] += 1
    return counts


def alternating_split(cycles):
    """The cheapest honest split: even-indexed cycles train, odd are held out.

    Deliberately not random -- a split this small wants to be reproducible and
    inspectable, and any imbalance it shows is a property of the basis, not of a
    seed. Arm 1 reads the criterion on `heldout` and descends on `train`.
    """
    train = [c for i, c in enumerate(cycles) if i % 2 == 0]
    heldout = [c for i, c in enumerate(cycles) if i % 2 == 1]
    return train, heldout


def census(dome, label: str, cycles) -> dict:
    counts = per_edge_counts(cycles)
    interior, interior_edge_ids = b29.hr.interior_graph(dome)
    wide_interior = [e for e in interior_edge_ids if dome.edges[e].m >= 2]
    covered = [e for e in wide_interior if counts.get(e, 0) > 0]

    train, heldout = alternating_split(cycles)
    tc, hc = per_edge_counts(train), per_edge_counts(heldout)

    rows = []
    for edge_id in wide_interior:
        rows.append(
            {
                "edge": int(edge_id),
                "m": int(dome.edges[edge_id].m),
                "cycles": int(counts.get(edge_id, 0)),
                "train": int(tc.get(edge_id, 0)),
                "heldout": int(hc.get(edge_id, 0)),
            }
        )

    all_counts = [r["cycles"] for r in rows]
    out = {
        "label": label,
        "cycles": len(cycles),
        "wide_interior_edges": len(wide_interior),
        "edges_on_some_cycle": len(covered),
        "edges_on_no_cycle": len(wide_interior) - len(covered),
        "per_edge_cycles": _q(all_counts),
        "histogram": {str(k): v for k, v in sorted(Counter(all_counts).items())},
        "split": {
            "train_cycles": len(train),
            "heldout_cycles": len(heldout),
            "per_edge_train": _q([r["train"] for r in rows]),
            "per_edge_heldout": _q([r["heldout"] for r in rows]),
            "edges_heldout_empty": sum(1 for r in rows if r["heldout"] == 0),
            "edges_heldout_empty_among_covered": sum(
                1 for r in rows if r["cycles"] > 0 and r["heldout"] == 0
            ),
            "edges_train_empty_among_covered": sum(
                1 for r in rows if r["cycles"] > 0 and r["train"] == 0
            ),
        },
        "rows": rows,
    }
    return out


def main() -> None:
    env, agent = b33.arms_mod.build_arm(b33.ARM, 42)
    dome = agent.dome
    cyc = b29.cycles_of(dome)
    wide = cyc["wide"]
    local = b33.local_cycles(dome, wide, b33.RADIUS)

    report = {
        "ticket": 603,
        "reading": "per-edge cycle census before any arm is built",
        "surface": b33.ARM,
        "radius": b33.RADIUS,
        "wide_cycles": len(wide),
        "local_cycles": len(local),
        "census": {
            "wide": census(dome, "wide basis", wide),
            "local": census(dome, f"local r={b33.RADIUS}", local),
        },
    }

    for name in ("wide", "local"):
        c = report["census"][name]
        print(f"\n== {c['label']} : {c['cycles']} cycles ==")
        print(
            f"  wide interior edges {c['wide_interior_edges']}, "
            f"on some cycle {c['edges_on_some_cycle']}, "
            f"on none {c['edges_on_no_cycle']}"
        )
        pe = c["per_edge_cycles"]
        print(
            f"  cycles per edge: min {pe['min']:.0f} q1 {pe['q1']:.1f} "
            f"median {pe['median']:.1f} q3 {pe['q3']:.1f} max {pe['max']:.0f} "
            f"mean {pe['mean']:.2f}"
        )
        print(f"  histogram {c['histogram']}")
        s = c["split"]
        print(
            f"  split {s['train_cycles']}/{s['heldout_cycles']}: "
            f"held-out per edge median {s['per_edge_heldout']['median']:.1f}, "
            f"max {s['per_edge_heldout']['max']:.0f}; "
            f"edges with an empty held-out side "
            f"{s['edges_heldout_empty_among_covered']} of {c['edges_on_some_cycle']}"
        )

    out = _HERE / "603-census-seed42.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out.name}")


if __name__ == "__main__":
    main()
