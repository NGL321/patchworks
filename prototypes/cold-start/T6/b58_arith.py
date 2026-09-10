"""B58 (#630) §A: why `{0, 1, 2, 6, 7, 10, 13, 14, 18, 19}`, and what a stagger exposes.

[B56 (#628)](https://github.com/NGL321/patchworks/issues/628) §0 *measured* the exact
stagger family and did not derive it. This derives it, and the derivation is integer
arithmetic with no maps in it at all.

**The observation that makes it arithmetic.** `holonomy_read.hop_operator` is
`F_out · F_inᵀ`. Under `b42_stagger.build_staggered` both incident maps at a cell are
row-selections of the *same* orthonormal frame `Q_v`, so

    hop = S_out Q_v Q_vᵀ S_inᵀ = S_out S_inᵀ

and the frame cancels exactly. `S_out S_inᵀ` is a 0/1 matrix with entry `(a, b) = 1`
iff `rows_out[a] == rows_in[b]` -- a **partial permutation matrix**. Products of
partial permutation matrices are partial permutation matrices, so every cycle's
holonomy has singular values in `{0, 1}` and

    sigma_max = 1 exactly  <=>  the composed matrix is nonzero
    sigma_max = 0 exactly  <=>  it is not

which is precisely what B56 measured (1.000 against 4e-08 of float noise). **There is
no continuum here to explain** -- exactness is a survival question about row indices,
and it is decidable by counting.

**What survives.** At cell `v`, edge `e` in slot `i` takes the cyclic interval
`rows(v, e) = {(i·s + j) mod k_v : j < m_e}`. A lane position `j` in edge `e` carries
row `o_v(e) + j`, where `o_v(e) = slot_v(e)·s`. Passing `e -> f` at `v` needs a shared
row, which shifts the lane position by `o_v(e) - o_v(f)`. So a position survives the
whole cycle iff it stays inside every interval it passes through and the accumulated
shift closes.

This instrument computes that directly and checks it against B56's measured sweep.

It also reports the **exposure** each stagger costs, in
[B42 (#605)](https://github.com/NGL321/patchworks/issues/605)'s own currency
(`b42_exposed.py`): a cell's exposed dimension is how many distinct rows of its frame
reach a neighbour at all, `|union_e rows(v, e)|`. B42 priced two points of that curve
-- `min(sum_e m_e, k_v)` today, `max_e m_e` at stagger 0. A stagger interpolates them,
and the interpolation is exact and free.

Usage::

    PYTHONPATH=src python prototypes/cold-start/T6/b58_arith.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
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


b33 = _load("b58a_b33", _HERE / "b33_coexist.py")
b29 = _load("b58a_b29", _HERE / "b29_holonomy.py")

import construction_grading as cg  # noqa: E402


# -- the geometry, once per arm -------------------------------------------------


def geometry(arm: str, seed: int) -> dict:
    """Everything the arithmetic needs, and nothing that depends on a map.

    `slot[(cell, edge)]` is the edge's index in `dome.incident[cell]` -- the `i` that
    `build_staggered` multiplies by the stagger. `k_v` is `dome._permitted[cell]`, the
    reserve mask's window, which is what the offsets wrap in.
    """
    env, agent = b33.arms_mod.build_arm(arm, seed)
    try:
        dome = agent.dome
        slot = {}
        for cell in dome.predicting:
            for i, edge_id in enumerate(dome.incident[cell]):
                slot[(int(cell), int(edge_id))] = i
        k_v = {int(c): int(dome._permitted[c]) for c in dome.predicting}
        m_e = {int(e): int(dome.edges[e].m) for e in range(len(dome.edges))}
        bases = b29.cycles_of(dome)
        cycles = {
            name: [
                [
                    (int(a), int(v), int(b))
                    for (a, v, b) in b29.hr.cycle_hops(dome, tuple(c))
                ]
                for c in cyc
            ]
            for name, cyc in bases.items()
        }
        incident = {
            int(c): [int(e) for e in dome.incident[c]] for c in dome.predicting
        }
        return {
            "arm": arm,
            "seed": seed,
            "slot": slot,
            "k_v": k_v,
            "m_e": m_e,
            "cycles": cycles,
            "incident": incident,
            "n": int(dome.shape.n),
        }
    finally:
        env.close()


def rows(g: dict, cell: int, edge: int, stagger: int) -> list[int]:
    """`build_staggered`'s row list, verbatim: `(slot*s + j) mod k_v`."""
    width = g["k_v"][cell]
    off = g["slot"][(cell, edge)] * stagger
    return [(off + j) % width for j in range(g["m_e"][edge])]


# -- the composed holonomy, as integers ----------------------------------------


def hop_matrix(g: dict, key, stagger: int) -> np.ndarray:
    """`S_out S_inᵀ` as a 0/1 integer array -- the frame has cancelled."""
    edge_in, cell, edge_out = key
    r_in = rows(g, cell, edge_in, stagger)
    r_out = rows(g, cell, edge_out, stagger)
    index = {}
    for b, r in enumerate(r_in):
        index.setdefault(r, b)
    h = np.zeros((len(r_out), len(r_in)), dtype=np.int8)
    for a, r in enumerate(r_out):
        b = index.get(r)
        if b is not None:
            h[a, b] = 1
    return h


def cycle_exact(g: dict, hops, stagger: int) -> dict:
    """`sigma_max` and `channel_return` for one cycle, by integer composition.

    The composite is a partial permutation matrix, so `sigma_max` is 1 when it is
    nonzero and 0 when it is not, and `channel_return = |<u1, v1>|` is 1 exactly when
    some position returns to **itself** -- a fixed point of the induced partial map --
    and 0 otherwise. Both are read off the integer matrix; no SVD is needed and none
    is taken, which is what makes this a derivation rather than a re-measurement.
    """
    composed = hop_matrix(g, hops[0], stagger)
    for key in hops[1:]:
        composed = hop_matrix(g, key, stagger) @ composed
    nonzero = int(composed.sum()) > 0
    fixed = int(np.trace(composed)) if composed.shape[0] == composed.shape[1] else 0
    return {
        "sigma_max": 1.0 if nonzero else 0.0,
        "channel_return": 1.0 if fixed > 0 else 0.0,
        "carried": int(composed.sum()),
        "fixed": int(fixed),
        "m": int(composed.shape[0]),
    }


def exposure(g: dict, stagger: int) -> dict:
    """B42's currency: `|union_e rows(v, e)|` per cell, against its two named points.

    `exposed_today` is `min(sum_e m_e, k_v)` -- independent per-edge blocks -- and
    `exposed_flat` is `max_e m_e`, the stagger-0 collapse. The staggered value sits
    between them and is exact.
    """
    vals, today, flat = [], [], []
    for cell, edges in g["incident"].items():
        ms = [g["m_e"][e] for e in edges]
        if not ms:
            continue
        union = set()
        for e in edges:
            union.update(rows(g, cell, e, stagger))
        vals.append(len(union))
        today.append(min(sum(ms), g["k_v"][cell]))
        flat.append(max(ms))
    return {
        "exposed_stagger": _stats(vals),
        "exposed_today": _stats(today),
        "exposed_flat": _stats(flat),
        "recovered_share": round(
            (np.median(vals) - np.median(flat))
            / max(np.median(today) - np.median(flat), 1e-9),
            4,
        ),
    }


def _stats(xs) -> dict:
    a = np.asarray(sorted(xs), dtype=np.float64)
    return {
        "min": float(a[0]),
        "median": float(np.median(a)),
        "mean": round(float(a.mean()), 3),
        "max": float(a[-1]),
    }


def sweep(g: dict, basis: str = "wide") -> dict:
    """Every stagger `0..k_v-1`, predicted rather than measured."""
    k = min(g["k_v"].values())
    out = {}
    for s in range(k):
        rowsr = [cycle_exact(g, h, s) for h in g["cycles"][basis]]
        out[str(s)] = {
            "sigma_max_median": float(np.median([r["sigma_max"] for r in rowsr])),
            "channel_return_median": float(
                np.median([r["channel_return"] for r in rowsr])
            ),
            "cycles_exact": int(sum(1 for r in rowsr if r["sigma_max"] > 0)),
            "cycles": len(rowsr),
            **exposure(g, s),
        }
    return out


# -- the closed form, and its check --------------------------------------------


def predicate(g: dict, stagger: int, basis: str = "wide") -> dict:
    """The candidate rule, stated on the cycle's own integers.

    Around a cycle the lane position accumulates a shift of `s · D`, where

        D = sum over hops of (slot_v(edge_in) - slot_v(edge_out))

    is a **pure graph quantity** -- it does not mention `s`, the seed, or any map. A
    position can only return to itself if that shift closes modulo the widths it
    passes through, so `s · D ≡ 0` is necessary. It is not sufficient: the position
    must also stay inside every interval, which is what `cycle_exact` decides. This
    reports both so the gap between them is visible rather than assumed.
    """
    hits = []
    for hops in g["cycles"][basis]:
        d = 0
        for (edge_in, cell, edge_out) in hops:
            d += g["slot"][(cell, edge_in)] - g["slot"][(cell, edge_out)]
        width = g["k_v"][hops[0][1]]
        hits.append(
            {
                "D": d,
                "closes": (stagger * d) % width == 0,
                "exact": cycle_exact(g, hops, stagger)["sigma_max"] > 0,
            }
        )
    closes = sum(1 for h in hits if h["closes"])
    exact = sum(1 for h in hits if h["exact"])
    agree = sum(1 for h in hits if h["closes"] == h["exact"])
    return {
        "cycles": len(hits),
        "closes": closes,
        "exact": exact,
        "agree": agree,
        "necessary_holds": all(h["closes"] for h in hits if h["exact"]),
        "D_values": sorted({h["D"] for h in hits}),
    }


def main() -> None:
    arms = ["reserve_p8", "reserve_p12", "reserve_p16"]
    seeds = [42, 43, 44]
    out = {
        "ticket": 630,
        "reading": "why the exact stagger family, and what it exposes",
        "note": "construction only, and derived rather than measured -- integer arithmetic, no maps",
        "arms": {},
    }
    for arm in arms:
        per_seed = {}
        for seed in seeds:
            g = geometry(arm, seed)
            s_sweep = sweep(g)
            exact_set = sorted(
                int(s) for s, r in s_sweep.items() if r["sigma_max_median"] > 0
            )
            per_seed[str(seed)] = {
                "k_v": min(g["k_v"].values()),
                "exact_staggers": exact_set,
                "sweep": s_sweep,
                "predicate": {
                    str(s): predicate(g, s) for s in (0, 1, 3, 6, 9, 13, 19)
                    if s < min(g["k_v"].values())
                },
            }
            print(
                f"{arm} seed {seed}: k_v {min(g['k_v'].values())}"
                f"  exact {exact_set}",
                flush=True,
            )
        out["arms"][arm] = per_seed
    path = _HERE / "630-arith.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {path.name}")


if __name__ == "__main__":
    main()
