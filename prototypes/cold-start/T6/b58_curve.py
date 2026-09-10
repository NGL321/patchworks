"""B58 (#630) §A3: the `p`-versus-differentiation curve, predicted rather than swept.

`b58_arith.py` shows exactness is integer arithmetic. So is the other column.

Under `b42_stagger.build_staggered` every incident map at a cell is a row-selection of
one orthonormal frame, so two edges' row spaces are **coordinate subspaces**. The
singular values of `Q[R_i]ᵀ Q[R_j]` are then `|R_i ∩ R_j|` ones and the rest zeros, and
`b42_stagger.distinctness`'s mean over `min(m_i, m_j)` of them is exactly

    edge_overlap(i, j) = |R_i ∩ R_j| / min(m_i, m_j)

with `differentiation = 1 - edge_overlap`. **No frame is drawn and no SVD is taken** --
which is why B56 measured `edge_overlap` agreeing to four decimals across three seeds
while `identification` moved: one column is arithmetic and the other is not.

That makes B56 §0's second table computable at any `p` without building a surface, so
the ceiling #630 asks about is **predicted**, and the two prices on `p` -- this curve
and [B44 (#608)](https://github.com/NGL321/patchworks/issues/608)'s community band --
can be quoted on the same axis.

This validates the prediction against B56's own measured sweep before extending it.

Usage::

    PYTHONPATH=src python prototypes/cold-start/T6/b58_curve.py
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


a = _load("b58c_arith", _HERE / "b58_arith.py")


def edge_overlap(g: dict, stagger: int) -> float:
    """`b42_stagger.distinctness`, evaluated as set intersections.

    Mirrors that function's own skips: a cell with fewer than two readable incident
    blocks contributes nothing, and the statistic is the **median over cells** of the
    mean over incident pairs.
    """
    vals = []
    for cell, edges in g["incident"].items():
        blocks = [(e, set(a.rows(g, cell, e, stagger))) for e in edges if g["m_e"][e] > 0]
        if len(blocks) < 2:
            continue
        pair = []
        for i in range(len(blocks)):
            for j in range(i + 1, len(blocks)):
                e_i, r_i = blocks[i]
                e_j, r_j = blocks[j]
                denom = min(g["m_e"][e_i], g["m_e"][e_j])
                pair.append(len(r_i & r_j) / denom if denom else 0.0)
        if pair:
            vals.append(sum(pair) / len(pair))
    return float(np.median(vals)) if vals else float("nan")


def curve(arm: str, seed: int) -> dict:
    """Every stagger at this `p`: exactness, differentiation, exposure."""
    g = a.geometry(arm, seed)
    k = min(g["k_v"].values())
    rows = []
    for s in range(k):
        exact = [a.cycle_exact(g, h, s) for h in g["cycles"]["wide"]]
        sigma = float(np.median([r["sigma_max"] for r in exact]))
        ov = edge_overlap(g, s)
        exp = a.exposure(g, s)
        rows.append(
            {
                "stagger": s,
                "sigma_max_median": sigma,
                "channel_return_median": float(
                    np.median([r["channel_return"] for r in exact])
                ),
                "edge_overlap": round(ov, 4),
                "differentiation": round(1.0 - ov, 4),
                "exposed_median": exp["exposed_stagger"]["median"],
                "exposed_today": exp["exposed_today"]["median"],
                "exposed_flat": exp["exposed_flat"]["median"],
            }
        )
    exact_rows = [r for r in rows if r["sigma_max_median"] > 0]
    best = max(exact_rows, key=lambda r: r["differentiation"]) if exact_rows else None
    return {
        "arm": arm,
        "seed": seed,
        "k_v": k,
        "exact_staggers": [r["stagger"] for r in exact_rows],
        "exact_count": len(exact_rows),
        "ceiling": best["differentiation"] if best else None,
        "ceiling_at_stagger": best["stagger"] if best else None,
        "ceiling_exposed": best["exposed_median"] if best else None,
        "rows": rows,
    }


def validate(out: dict) -> dict:
    """Against B56's measured sweep, which is on this branch. Predicted vs measured."""
    path = _HERE / "628-stagger-sweep.json"
    if not path.exists():
        return {"checked": False, "reason": "628-stagger-sweep.json not present"}
    measured = json.loads(path.read_text(encoding="utf-8"))
    pred = {r["stagger"]: r for r in out["arms"]["reserve_p12"]["42"]["rows"]}
    diffs = []
    for s, row in measured["by_seed"]["42"]["stagger"].items():
        p = pred[int(s)]
        diffs.append(
            {
                "stagger": int(s),
                "d_overlap": round(abs(row["edge_overlap"] - p["edge_overlap"]), 6),
                "exact_measured": row["sigma_max"] > 0.5,
                "exact_predicted": p["sigma_max_median"] > 0,
            }
        )
    return {
        "checked": True,
        "staggers": len(diffs),
        "exactness_agrees": sum(
            1 for d in diffs if d["exact_measured"] == d["exact_predicted"]
        ),
        "max_overlap_error": max(d["d_overlap"] for d in diffs),
        "per_stagger": diffs,
    }


def main() -> None:
    out = {
        "ticket": 630,
        "reading": "p versus differentiation at channel_return 1.0000, predicted",
        "note": "construction only; integer arithmetic, no frame drawn and no SVD taken",
        "arms": {},
    }
    arms = [f"reserve_p{p}" for p in (4, 8, 12, 16, 20, 24)]
    for arm in arms:
        out["arms"][arm] = {}
        for seed in (42, 43, 44):
            c = curve(arm, seed)
            out["arms"][arm][str(seed)] = c
            if seed == 42:
                print(
                    f"{arm:>12}  k_v {c['k_v']:>3}  exact {c['exact_count']:>3}/{c['k_v']:<3}"
                    f"  ceiling {c['ceiling']}  at stagger {c['ceiling_at_stagger']}"
                    f"  exposed {c['ceiling_exposed']}"
                    f" (today {c['rows'][0]['exposed_today']}, flat {c['rows'][0]['exposed_flat']})",
                    flush=True,
                )
    out["validation_vs_B56"] = validate(out)
    v = out["validation_vs_B56"]
    if v.get("checked"):
        print(
            f"\nvalidation vs B56's measured sweep: exactness agrees on "
            f"{v['exactness_agrees']}/{v['staggers']} staggers, "
            f"max |edge_overlap| error {v['max_overlap_error']}",
            flush=True,
        )
    path = _HERE / "630-curve.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {path.name}")


if __name__ == "__main__":
    main()
