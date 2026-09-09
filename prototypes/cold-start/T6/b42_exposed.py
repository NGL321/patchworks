"""B42 (#605): what the flat bundle costs a cell in exposed dimension.

`b42_stagger.py` finds that exact path-independence survives only at edge-overlap
1.0000 -- every edge at a cell reading the *same* leading rows of the one frame.
Nested prefixes of one ordered basis, not nine distinct channels.

So the exposed dimension of a cell -- how many of its own node stalk directions
reach its neighbours at all -- collapses from what the edges can *jointly* carry to
what the *widest single* edge carries:

* today (independent per-edge blocks): `min(sum_e m_e, k_v)`
* flat bundle at stagger 0:            `max_e m_e`

This reports the distribution of both, which is the price of the architecture stated
in the map's own currency rather than in holonomy.

Usage::

    PYTHONPATH=src python prototypes/cold-start/T6/b42_exposed.py
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


b33 = _load("b42e_b33", _HERE / "b33_coexist.py")


def _stats(xs):
    xs = sorted(xs)
    return {
        "min": xs[0],
        "median": xs[len(xs) // 2],
        "mean": round(sum(xs) / len(xs), 3),
        "max": xs[-1],
    }


def main() -> None:
    seed = 42
    _env, agent = b33.arms_mod.build_arm(b33.ARM, seed)
    dome = agent.dome
    today, flat, rows = [], [], []
    for cell in dome.predicting:
        ms = [dome.edges[e].m for e in dome.incident[cell]]
        k_v = int(dome._permitted[cell])
        t = min(sum(ms), k_v)
        f = max(ms) if ms else 0
        today.append(t)
        flat.append(f)
        rows.append(
            {
                "cell": int(cell),
                "degree": len(ms),
                "k_v": k_v,
                "sum_m": sum(ms),
                "max_m": f,
                "exposed_today": t,
                "exposed_flat": f,
            }
        )
    out = {
        "ticket": 605,
        "surface": b33.ARM,
        "seed": seed,
        "n": int(dome.shape.n),
        "cells": len(rows),
        "exposed_today": _stats(today),
        "exposed_flat": _stats(flat),
        "ratio_median": round(_stats(flat)["median"] / _stats(today)["median"], 3),
        "per_cell": rows,
    }
    print(f"cells: {out['cells']}  n = {out['n']}")
    print(f"  exposed today (min(sum_e m_e, k_v)): {out['exposed_today']}")
    print(f"  exposed flat  (max_e m_e)          : {out['exposed_flat']}")
    print(f"  median ratio flat/today            : {out['ratio_median']}")
    path = _HERE / f"605-exposed-seed{seed}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {path.name}")


if __name__ == "__main__":
    main()
