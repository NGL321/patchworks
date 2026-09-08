"""T6 (#567 / B18): read `b18_centred.py`'s output.

Answers the ticket's items 2-4: how much of the uncentred reading the mean was
responsible for, what the high-reading tail of cells has in common, and how the
split moves by level, degree and lane width.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
_T0 = _HERE.parent / "T0"
for extra in (_ROOT / "prototypes" / "chart-double-duty-166", _ROOT / "benchmarks", _T0):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


arms_mod = _load("t6_arms", _HERE / "arms.py")

ARMS = ["reserve", "reserve_p16"]
TICKS = 20_000
SEED = 42


def structure(arm: str) -> dict:
    """Level, degree and incident lane widths per cell, from construction only."""
    env, agent = arms_mod.build_arm(arm, SEED)
    try:
        dome = agent.dome
        deg: dict[int, int] = {}
        widths: dict[int, list[int]] = {}
        for e in dome.edges:
            for c in (e.u, e.v):
                deg[c] = deg.get(c, 0) + 1
                widths.setdefault(c, []).append(int(e.m))
        return {
            int(c.id): {
                "level": int(c.index.level),
                "degree": deg.get(int(c.id), 0),
                "m_sum": int(sum(widths.get(int(c.id), []))),
                "m_max": int(max(widths.get(int(c.id), [0]))),
            }
            for c in dome.cells
        }
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()


def _corr(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or a[ok].std() == 0 or b[ok].std() == 0:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def main() -> None:
    out = {"issue": 567, "arms": {}}
    for arm in ARMS:
        path = _HERE / f"567-centred-{arm}-seed{SEED}-{TICKS}.json"
        if not path.exists():
            print(f"missing {path.name}")
            continue
        rec = json.loads(path.read_text())
        struct = structure(arm)

        print(f"\n{'=' * 78}\n{arm}  (reserve_p={rec['reserve_p']}, "
              f"{rec['relay_cells']} relay cells)\n{'=' * 78}")
        print(f"{'window':>7} {'er_unc':>8} {'er_cen':>8} {'ratio':>7} "
              f"{'excess_unc':>11} {'excess_cen':>11} {'mean_share':>11} {'pooled_cen':>11}")
        rows_out = []
        for cp in rec["checkpoints"]:
            if cp["ticks"] == 0:
                continue
            u = cp["er_uncentred"]["median"]
            c = cp["er_centred"]["median"]
            rows_out.append({
                "ticks": cp["ticks"], "er_uncentred": u, "er_centred": c,
                "excess_uncentred": u - 1.0, "excess_centred": c - 1.0,
                "excess_ratio": (c - 1.0) / max(u - 1.0, 1e-12),
                "mean_share": cp["mean_share"]["median"],
                "pooled_centred": cp["pooled"]["er_centred"],
            })
            print(f"{cp['ticks']:>7} {u:>8.4f} {c:>8.4f} {c / u:>7.3f} "
                  f"{u - 1:>11.4f} {c - 1:>11.4f} "
                  f"{cp['mean_share']['median']:>11.4f} {cp['pooled']['er_centred']:>11.2f}")

        final = rec["checkpoints"][-1]
        cells = final["cells"]
        for r in cells:
            r.update(struct.get(r["cell"], {"level": -1, "degree": 0, "m_sum": 0, "m_max": 0}))

        unc = np.array([r["er_uncentred"] for r in cells])
        cen = np.array([r["er_centred"] for r in cells])
        print(f"\n  @{TICKS} per-cell distribution over {len(cells)} relay cells:")
        for name, a in (("uncentred", unc), ("centred", cen)):
            print(f"    {name:>10}: min {a.min():.4f}  p10 {np.quantile(a, .1):.4f}  "
                  f"median {np.median(a):.4f}  p90 {np.quantile(a, .9):.4f}  max {a.max():.4f}")

        # --- item 3: the tail --------------------------------------------------
        order = np.argsort(unc)[::-1][:8]
        print(f"\n  the uncentred tail (item 3) -- top 8 cells by er_uncentred:")
        print(f"    {'cell':>6} {'lvl':>4} {'deg':>4} {'k_v':>4} {'m_sum':>6} "
              f"{'er_unc':>8} {'er_cen':>8} {'mean_share':>11}")
        tail_ids = []
        for i in order:
            r = cells[i]
            tail_ids.append(r["cell"])
            print(f"    {r['cell']:>6} {r['level']:>4} {r['degree']:>4} {r['k_v']:>4} "
                  f"{r['m_sum']:>6} {r['er_uncentred']:>8.4f} {r['er_centred']:>8.4f} "
                  f"{r['mean_share']:>11.5f}")
        rest = [r for r in cells if r["cell"] not in tail_ids]
        print(f"    tail mean_share {np.mean([cells[i]['mean_share'] for i in order]):.5f} "
              f"vs rest {np.mean([r['mean_share'] for r in rest]):.5f}")
        print(f"    tail level {sorted({cells[i]['level'] for i in order})} "
              f"vs all levels {sorted({r['level'] for r in cells})}")

        # --- item 2 / 4: correlates -------------------------------------------
        print(f"\n  what predicts the reading (item 4):")
        corrs = {}
        for key in ("level", "degree", "k_v", "m_sum", "m_max", "mu_energy", "tr_cov"):
            vals = [r[key] for r in cells]
            corrs[f"unc~{key}"] = _corr(unc, vals)
            corrs[f"cen~{key}"] = _corr(cen, vals)
            print(f"    {key:>10}: corr(er_unc) {corrs[f'unc~{key}']:+.3f}   "
                  f"corr(er_cen) {corrs[f'cen~{key}']:+.3f}")
        print(f"    {'er_unc':>10}: corr(er_cen) {_corr(unc, cen):+.3f}   "
              f"<- do the two readings even rank cells the same way?")

        print(f"\n  by level:")
        print(f"    {'lvl':>4} {'cells':>6} {'er_unc':>8} {'er_cen':>8} {'mean_share':>11}")
        by_level = {}
        for lv in sorted({r["level"] for r in cells}):
            g = [r for r in cells if r["level"] == lv]
            by_level[lv] = {
                "cells": len(g),
                "er_uncentred": float(np.median([r["er_uncentred"] for r in g])),
                "er_centred": float(np.median([r["er_centred"] for r in g])),
                "mean_share": float(np.median([r["mean_share"] for r in g])),
            }
            print(f"    {lv:>4} {len(g):>6} {by_level[lv]['er_uncentred']:>8.4f} "
                  f"{by_level[lv]['er_centred']:>8.4f} {by_level[lv]['mean_share']:>11.5f}")

        out["arms"][arm] = {
            "reserve_p": rec["reserve_p"],
            "relay_cells": rec["relay_cells"],
            "windows": rows_out,
            "final_correlations": corrs,
            "final_by_level": by_level,
            "final_tail": [
                {k: cells[i][k] for k in
                 ("cell", "level", "degree", "k_v", "m_sum",
                  "er_uncentred", "er_centred", "mean_share")}
                for i in order
            ],
        }

    dst = _HERE / "567-centred-analysis.json"
    dst.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {dst.name}")


if __name__ == "__main__":
    main()
