"""#433's pre-registered read, both builds, from the per-seed checkpoints.

    python summarise.py <baseline dir> <normalised dir>

Reports `rim tau / apex tau` and median apex `lambda(K)` per seed on each build,
with the per-cell apex rows (#181: per cell, not per level, as #422 did), then
states the pre-registered verdict.
"""

from __future__ import annotations

import json
import math
import pathlib
import statistics
import sys

sys.path.insert(0, "benchmarks")
sys.path.insert(0, "src")

from patchworks.graph import DEFAULT_SPEC, build_graph  # noqa: E402

#: #422's figures, on the post-hoc-projection build, at 100k on three seeds.
#: Quoted with the surface they were taken on, per #437 and the Notes.
PRE_REGISTERED_APEX_LAMBDA = {0: 0.529, 1: 0.415, 2: 0.289}
PRE_REGISTERED_RIM_OVER_APEX_TAU = 12.87

DOME = build_graph(DEFAULT_SPEC)
LEVEL = {c.id: c.index.level for c in DOME.cells}


def tau(radius: float) -> float:
    if not 0.0 < radius < 1.0:
        return math.inf if radius >= 1.0 else 0.0
    return -1.0 / math.log(radius)


def load(directory: pathlib.Path) -> dict[int, dict]:
    out = {}
    for path in sorted(directory.glob("seed*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        out[int(data["seed"])] = data
    return out


def by_level(values: dict[str, float]) -> dict[int, list[float]]:
    grouped: dict[int, list[float]] = {}
    for cell, value in values.items():
        grouped.setdefault(LEVEL[int(cell)], []).append(value)
    return dict(sorted(grouped.items()))


def _ratio(numerator: float, denominator: float) -> float:
    if numerator == 0.0:
        return 0.0
    if denominator == 0.0:
        return math.inf
    return numerator / denominator


def seed_row(data: dict) -> dict:
    radius = {c: float(v) for c, v in data["radius"].items()}
    grouped = by_level(radius)
    rim, apex = min(grouped), max(grouped)
    taus = {level: [tau(r) for r in rows] for level, rows in grouped.items()}
    return {
        "apex_level": apex,
        "rim_level": rim,
        "apex_lambda": statistics.median(grouped[apex]),
        "rim_lambda": statistics.median(grouped[rim]),
        "apex_tau": statistics.median(taus[apex]),
        "rim_tau": statistics.median(taus[rim]),
        "apex_cells": sorted(
            (int(c), v) for c, v in radius.items() if LEVEL[int(c)] == apex
        ),
        "rim_tau_over_apex_tau": _ratio(
            statistics.median(taus[rim]), statistics.median(taus[apex])
        ),
        "raw_sigma": (
            statistics.median(float(v) for v in data["raw_norm"].values())
            if "raw_norm" in data
            else None
        ),
        "ticks": data.get("ticks"),
        "seconds": data.get("seconds"),
    }


def main() -> int:
    base = load(pathlib.Path(sys.argv[1]))
    new = load(pathlib.Path(sys.argv[2]))
    print("\n== #433's pre-registered read: both builds, 100k ticks, seeds 0/1/2 ==")

    rows = {}
    for label, data in (("post-hoc projection", base), ("forward normalisation", new)):
        print(f"\n-- {label} --")
        print(
            "   seed   apex lambda   rim lambda   apex tau   rim tau   "
            "rim tau / apex tau"
        )
        rows[label] = {}
        for seed in sorted(data):
            row = seed_row(data[seed])
            rows[label][seed] = row
            print(
                f"   {seed:>4}   {row['apex_lambda']:>11.6g}   "
                f"{row['rim_lambda']:>10.6g}   {row['apex_tau']:>8.4g}   "
                f"{row['rim_tau']:>7.4g}   {row['rim_tau_over_apex_tau']:>18.4g}"
            )
        if any(r["raw_sigma"] is not None for r in rows[label].values()):
            print(
                "   median raw sigma(K) per seed: "
                + ", ".join(
                    f"{s}: {rows[label][s]['raw_sigma']:.4g}" for s in sorted(data)
                )
                + "   (diagnostic; sigma_max(used) is in band regardless)"
            )

    print("\n-- apex cells, per cell rather than per level (#181, as #422 did) --")
    for seed in sorted(set(rows["post-hoc projection"]) & set(rows["forward normalisation"])):
        b = dict(rows["post-hoc projection"][seed]["apex_cells"])
        n = dict(rows["forward normalisation"][seed]["apex_cells"])
        print(f"\n   seed {seed}   cell   projection lambda   normalisation lambda")
        for cell in sorted(set(b) | set(n)):
            print(
                f"            {cell:>6}   {b.get(cell, float('nan')):>17.6g}"
                f"   {n.get(cell, float('nan')):>20.6g}"
            )

    print("\n-- the pre-registered verdict --")
    print(
        "   Falsified if apex lambda(K) does not rise materially above #422's\n"
        "   0.529 / 0.415 / 0.289 AND rim tau / apex tau does not fall materially\n"
        "   below 12.87. Both builds re-run; #422's stored figures are shown for\n"
        "   reference only, and the comparison that counts is the two columns."
    )
    for seed in sorted(rows["forward normalisation"]):
        if seed not in rows["post-hoc projection"]:
            continue
        b = rows["post-hoc projection"][seed]
        n = rows["forward normalisation"][seed]
        print(
            f"\n   seed {seed}: apex lambda {b['apex_lambda']:.6g} -> "
            f"{n['apex_lambda']:.6g}"
            f"   ({(n['apex_lambda'] - b['apex_lambda']):+.6g}, "
            f"#422 recorded {PRE_REGISTERED_APEX_LAMBDA.get(seed)})"
        )
        print(
            f"            rim tau / apex tau "
            f"{b['rim_tau_over_apex_tau']:.4g} -> {n['rim_tau_over_apex_tau']:.4g}"
            f"   (#422 recorded {PRE_REGISTERED_RIM_OVER_APEX_TAU} across seeds)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
