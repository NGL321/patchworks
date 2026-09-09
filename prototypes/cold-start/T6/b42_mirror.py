"""B42 (#605): on an exactly flat bundle, is B34's count a reading or a mirror?

`b42_falsifier.py` re-ran B40's arm-4 falsifier on a bundle that is actually flat
(`identification` 0.0000 through mask and band) and the criterion still showed a
spread: min 12, median 12, max 15, std 1.27, 4 distinct values. Read as B40 read it,
that says the falsifier did not fire.

But B40's falsifier was stated as *"if holonomy cannot fail then the count reads
**full width** everywhere and carves nothing"* -- and full width is `m_e`, which is
not a constant. A criterion that returns each edge's own `m_e` **produces exactly the
spread the edge widths already have**, and would be scored as discriminating by any
statistic taken over the counts alone.

So this compares the count to `m_e` **edge for edge**. If `count == m_e` everywhere,
the criterion is a mirror on this surface: the spread is the width distribution
echoing back, the carving pressure is nil, and B40's negative result was scored
against the wrong null.

Reported for all three arms, so the control shows what a real reading looks like.

Usage::

    PYTHONPATH=src python prototypes/cold-start/T6/b42_mirror.py
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


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


b33 = _load("b42m_b33", _HERE / "b33_coexist.py")
b29 = _load("b42m_b29", _HERE / "b29_holonomy.py")
routes = _load("b42m_routes", _HERE / "b40_routes.py")
crit = _load("b42m_crit", _HERE / "b40_criterion.py")
res = _load("b42m_res", _HERE / "b42_reserve.py")

import holonomy_read as hr  # noqa: E402  (b29's helper, already on the path)


def raw_counts(dome, maps, per_edge, threshold):
    counts = {}
    for edge_id, cycles in per_edge.items():
        if not cycles:
            continue
        holos = [
            hr.holonomy(dome, maps, b29._base_at(tuple(c), edge_id)) for c in cycles
        ]
        n, _cos = crit.warranted_count(holos, threshold)
        counts[edge_id] = n
    return counts


def main() -> None:
    seed, threshold = 42, 0.9
    out: dict = {"ticket": 605, "surface": b33.ARM, "seed": seed,
                 "threshold": threshold, "arms": {}}

    dome0, _a0, _m0 = res.build_with_frames(seed, reserved=False)
    per_edge = routes.cycle_sets(dome0)["per_edge"]

    def compare(label, dome, maps):
        counts = raw_counts(dome, maps, per_edge, threshold)
        same = sum(1 for e, c in counts.items() if c == dome.edges[e].m)
        deltas = [c - dome.edges[e].m for e, c in counts.items()]
        row = {
            "edges_with_evidence": len(counts),
            "count_equals_m_e": same,
            "share_equals_m_e": round(same / len(counts), 4) if counts else 0.0,
            "delta_min": min(deltas) if deltas else None,
            "delta_max": max(deltas) if deltas else None,
            "delta_mean": round(sum(deltas) / len(deltas), 4) if deltas else None,
        }
        out["arms"][label] = row
        print(
            f"{label:>10}: count == m_e on {same}/{len(counts)} edges "
            f"({row['share_equals_m_e']:.0%})   delta "
            f"[{row['delta_min']}, {row['delta_max']}] mean {row['delta_mean']}"
        )

    _env, agent = b33.arms_mod.build_arm(b33.ARM, seed)
    compare("control", agent.dome, agent.sheaf.maps)

    dome, _a, maps = res.build_with_frames(seed, reserved=False)
    maps.project()
    compare("ambient", dome, maps)

    dome, _a, maps = res.build_with_frames(seed, reserved=True)
    maps.project()
    compare("reserved", dome, maps)

    path = _HERE / f"605-mirror-seed{seed}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {path.name}")


if __name__ == "__main__":
    main()
