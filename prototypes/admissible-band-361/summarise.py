"""Roll every `361-*.json` into the cross-seed picture (#361).

The point of the rig is that dwell and `tau` come from the **same run**, so the
roll-up's job is to say what a verdict rests on **across** seeds rather than
inside one. Three things are separated, because they fail for different reasons
and #226 conflated the first two:

- **The empty band** -- `dwell_c <= |loop(c)|`. Two graph quantities, no `tau`
  in it, so it is untouched by the operator controversy that moved every `tau`
  in the record.
- **Conduction** -- `tau_c >= |loop(c)|`, ADR-0026's ratio.
- **Licence** -- `dwell_c > tau_c`, ADR-0005's validity condition as #226
  re-pointed it.

Everything is reported as a **range over seeds**, never a median from one run
(#274's binding caveat: `tau = -1/ln rho` diverges as `rho -> 1`).

Usage::

    PYTHONPATH=src python prototypes/admissible-band-361/summarise.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent


def load(directory: Path, ticks: int | None) -> list[dict]:
    """Every run in `directory`, optionally narrowed to one horizon.

    Narrowing matters: dwell is **cumulative**, so rolling a 30,000-tick run in
    beside a 100,000-tick one compares a short run's dwell against a long one's
    and reproduces #226's defect in a new place.
    """
    runs = [json.loads(path.read_text()) for path in sorted(directory.glob("361-*.json"))]
    if ticks is not None:
        runs = [run for run in runs if run["ticks"] == ticks]
    return sorted(runs, key=lambda run: run["seed"])


def at_horizon(run: dict) -> dict:
    return run["checkpoints"][-1]


def span(values: list[float]) -> str:
    return f"{min(values):.3f} to {max(values):.3f}"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", type=Path, default=_HERE)
    parser.add_argument(
        "--ticks",
        type=int,
        default=None,
        help="only roll up runs at this horizon; dwell is cumulative, so mixing "
        "horizons compares a short run's dwell against a long one's",
    )
    args = parser.parse_args(argv)

    runs = load(args.dir, args.ticks)
    if not runs:
        raise SystemExit(f"no 361-*.json under {args.dir}")

    horizons = {run["ticks"] for run in runs}
    cells = runs[0]["cells"]
    loop = np.array(runs[0]["loop_lengths"], dtype=float)
    levels = np.array(runs[0]["levels"])
    print(
        f"{len(runs)} seeds, {cells} predicting cells, horizon "
        + ("/".join(str(h) for h in sorted(horizons)))
        + (" ticks" if len(horizons) == 1 else " ticks -- MIXED, see the caveat below")
    )
    if len(horizons) > 1:
        print(
            "  Dwell is cumulative over the run, so a mixed horizon compares a long\n"
            "  run's dwell against a short one's. Read the fixed-horizon rows only."
        )
    print(
        f"|loop(c)| enumerated per cell from the mask: "
        f"{int(loop.min())} to {int(loop.max())} ticks\n"
    )

    print("THE EMPTY BAND: `dwell_c <= |loop(c)|`, the consequence with no `tau` in it")
    print("  seed  ticks  dwell med  dwell/|loop| med   empty band   worst-cell dwell/|loop|")
    empty_counts, dwell_over_loop = [], []
    per_cell_empty = []
    for run in runs:
        final = at_horizon(run)
        full = final["against_loop_full"]
        dwell = np.array(final["per_cell"]["dwell"])
        per_cell_empty.append(dwell <= loop)
        empty_counts.append(full["empty_band"])
        dwell_over_loop.append(full["dwell_over_loop"]["median"])
        print(
            f"  {run['seed']:>4} {run['ticks']:>6} {final['dwell']['median']:>10.2f} "
            f"{full['dwell_over_loop']['median']:>17.3f} "
            f"{full['empty_band']:>9}/{cells} "
            f"{full['dwell_over_loop']['min']:>24.3f}"
        )
    print(
        f"  Across {len(runs)} seeds the empty-band count runs "
        f"{min(empty_counts)} to {max(empty_counts)} of {cells}; "
        f"median dwell/|loop| runs {span(dwell_over_loop)}."
    )

    stacked = np.array(per_cell_empty)
    always = stacked.all(axis=0)
    ever = stacked.any(axis=0)
    print(
        f"  Per cell: {int(always.sum())}/{cells} have an empty band on EVERY seed, "
        f"{int(ever.sum())}/{cells} on at least one, "
        f"{cells - int(ever.sum())}/{cells} on none."
    )

    print("\nCONDUCTION: `tau_c >= |loop(c)|`, ADR-0026's ratio, off the full loop")
    print("  seed   tau_full med   tau/|loop| med   conducts   apex tau_full med")
    conducts, tau_over_loop = [], []
    for run in runs:
        final = at_horizon(run)
        full = final["against_loop_full"]
        tau = np.array(final["per_cell"]["tau_full"])
        apex = levels == levels.max()
        conducts.append(full["conducts"])
        tau_over_loop.append(full["tau_over_loop"]["median"])
        print(
            f"  {run['seed']:>4} {final['tau_full']['median']:>14.3f} "
            f"{full['tau_over_loop']['median']:>16.3f} "
            f"{full['conducts']:>7}/{cells} {np.median(tau[apex]):>18.3f}"
        )
    print(
        f"  Conduction clears at {min(conducts)} to {max(conducts)} of {cells} cells; "
        f"median tau/|loop| runs {span(tau_over_loop)}."
    )

    print("\nLICENCE: `dwell_c > tau_c`, ADR-0005's validity condition as #226 re-pointed it")
    print("  seed   dwell/tau med   dwell/tau p05-p95        licensed   both ends held")
    ratios, licensed = [], []
    for run in runs:
        final = at_horizon(run)
        full = final["against_loop_full"]
        ratios.append(full["dwell_over_tau"]["median"])
        licensed.append(full["licensed"])
        print(
            f"  {run['seed']:>4} {full['dwell_over_tau']['median']:>15.3f} "
            f"{full['dwell_over_tau']['p05']:>12.3f}-{full['dwell_over_tau']['p95']:<12.3f} "
            f"{full['licensed']:>7}/{cells} {full['band_open_and_occupied']:>15}/{cells}"
        )
    print(
        f"  Median dwell/tau runs {span(ratios)} across seeds; the licence clears at "
        f"{min(licensed)} to {max(licensed)} of {cells}."
    )

    print("\nTHE SAME READING OFF THE CHART-ONLY OPERATOR, for the record's sake")
    print("  The number the record published before #274; kept so the operator's")
    print("  contribution is visible rather than asserted.")
    print("  seed   dwell/tau med (chart)   licensed (chart)   conducts (chart)")
    for run in runs:
        chart = at_horizon(run)["against_loop_chart"]
        print(
            f"  {run['seed']:>4} {chart['dwell_over_tau']['median']:>22.3f} "
            f"{chart['licensed']:>17}/{cells} {chart['conducts']:>15}/{cells}"
        )

    print("\nDWELL OVER THE RUN: median dwell at each checkpoint")
    for run in runs:
        trace = " ".join(
            f"{c['ticks']}:{c['dwell']['median']:.1f}" for c in run["checkpoints"]
        )
        print(f"  seed {run['seed']:>4}  {trace}")

    print("\nEMPTY BAND OVER THE RUN: count at each checkpoint")
    for run in runs:
        trace = " ".join(
            f"{c['ticks']}:{c['against_loop_full']['empty_band']}"
            for c in run["checkpoints"]
        )
        print(f"  seed {run['seed']:>4}  {trace}")

    print("\nBY LEVEL at the horizon -- a diagnostic only, never an index (#181)")
    print("  level  cells  |loop|   empty band, across seeds   conducts, across seeds")
    for level in sorted(set(levels.tolist())):
        mask = levels == level
        lengths = sorted({int(v) for v in loop[mask]})
        empty_span, conduct_span = [], []
        for run in runs:
            final = at_horizon(run)
            dwell = np.array(final["per_cell"]["dwell"])
            tau = np.array(final["per_cell"]["tau_full"])
            empty_span.append(int((dwell[mask] <= loop[mask]).sum()))
            conduct_span.append(int((tau[mask] >= loop[mask]).sum()))
        print(
            f"  L{level:<5} {int(mask.sum()):>5} "
            f"{'/'.join(str(v) for v in lengths):>7} "
            f"{min(empty_span):>16}-{max(empty_span):<8}/{int(mask.sum()):<5} "
            f"{min(conduct_span):>13}-{max(conduct_span):<8}/{int(mask.sum())}"
        )

    print("\nCHECKS, per run")
    for run in runs:
        checks = run["checks"]
        tick1 = checks.get("relay_identity_tick_1", {})
        tick10 = checks.get("relay_identity_tick_10", {})
        print(
            f"  seed {run['seed']:>4}  relay identity tick 1 "
            f"{tick1.get('max_relative_error', float('nan')):.2e}, tick 10 "
            f"{tick10.get('max_relative_error', float('nan')):.2e}; chart-only tau vs "
            f"the #206 rig {checks.get('chart_only_vs_206_tau_max_abs', float('nan')):.2e}"
        )


if __name__ == "__main__":
    main()
