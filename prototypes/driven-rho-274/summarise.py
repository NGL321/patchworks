"""Roll every `274-*.json` this rig wrote into the cross-seed picture (#274).

Separates the two claims the ticket separates: the **comparative** one (how much
the omitted relay term adds), which #271 argued is robust, and the **absolute**
one (`rho` itself, and the `tau` read off it), which #271 refused to quote until
it was seen on a driven run and across more than one seed.

Usage::

    PYTHONPATH=src python prototypes/driven-rho-274/summarise.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

RHO_CEILING = 1.0 - 1e-6

#: ADR-0026's enumerated ladder for `DEFAULT_SPEC`: `|loop(c)| = 2 * d(c, rim)`,
#: which on this dome coincides with twice the construction level. Quoted per
#: cell via the cell's own level, never as a per-level claim (#181).
LOOP_LENGTH = {1: 2, 2: 4, 3: 6, 4: 8, 5: 10, 6: 12, 7: 14}


def tau_of(rho: np.ndarray) -> np.ndarray:
    return -1.0 / np.log(np.clip(rho, 1e-12, RHO_CEILING))


def load(directory: Path) -> list[dict]:
    return [
        json.loads(path.read_text()) for path in sorted(directory.glob("274-*.json"))
    ]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", type=Path, default=Path("prototypes/driven-rho-274"))
    args = parser.parse_args(argv)

    runs = load(args.dir)
    if not runs:
        raise SystemExit(f"no 274-*.json under {args.dir}")

    print("THE COMPARATIVE CLAIM: what the omitted relay term is worth")
    print("  seed  ticks   rho chart  rho full   ratio med  ratio p25-p75   relay/chart norm")
    ratios = []
    for run in runs:
        final = run["checkpoints"][-1]
        ratios.append(final["ratio"]["median"])
        print(
            f"  {run['seed']:>4} {final['ticks']:>6}   "
            f"{final['rho_chart']['median']:>9.3f} {final['rho_full']['median']:>9.3f}  "
            f"{final['ratio']['median']:>9.2f}x  "
            f"{final['ratio']['p25']:>5.2f}-{final['ratio']['p75']:<5.2f}  "
            f"{final['relay_norm_share']['median']:>14.2f}"
        )
    print(
        f"  Across {len(runs)} seeds the ratio's median runs "
        f"{min(ratios):.2f}x to {max(ratios):.2f}x."
    )

    print("\nTHE ABSOLUTE CLAIM: rho itself, driven, and what tau reads off it")
    print("  seed   undriven full  driven full   expansive   tau full med   tau full p25-p75")
    for run in runs:
        final = run["checkpoints"][-1]
        print(
            f"  {run['seed']:>4} {run['undriven_control']['rho_full']['median']:>14.3f} "
            f"{final['rho_full']['median']:>12.3f} "
            f"{final['cells_expansive_full']:>9}/{run['cells']:<4} "
            f"{final['tau_full']['median']:>12.2f}   "
            f"{final['tau_full']['p25']:>6.2f}-{final['tau_full']['p75']:<8.2f}"
        )

    print("\nSTABILITY OVER THE RUN: rho_full median at each checkpoint")
    for run in runs:
        trace = " ".join(
            f"{row['ticks']}:{row['rho_full']['median']:.3f}"
            for row in run["checkpoints"]
        )
        print(f"  seed {run['seed']:>4}  {trace}")

    print("\nDEPTH: is the lift flat? corr(p_v, delta rho) at the horizon")
    for run in runs:
        final = run["checkpoints"][-1]
        print(
            f"  seed {run['seed']:>4}  corr(p_v, delta rho) "
            f"{final['corr_p_v_delta_rho']:+.3f}   corr(degree, delta rho) "
            f"{final['corr_degree_delta_rho']:+.3f}"
        )

    print("\nADR-0026: the corrected loop tau against `|loop(c)|`, per cell")
    print("  A conduction ratio read off `tau = -1/ln rho` rather than off ADR-0026's")
    print("  own paired-counterfactual `tau_hat`; the two are different quantities and")
    print("  this does not replace that reading. Reported as a range because near")
    print("  `rho = 1` a small change in rho is a large change in tau.")
    print("  seed   apex tau chart  apex tau full   apex ratio vs |loop|=14   cells tau>=|loop|")
    for run in runs:
        final = run["checkpoints"][-1]
        levels = np.array(run["levels"])
        chart = np.array(final["per_cell"]["rho_chart"])
        full = np.array(final["per_cell"]["rho_full"])
        loop = np.array([LOOP_LENGTH[int(level)] for level in levels], dtype=float)
        apex = levels == levels.max()
        tau_full = tau_of(full)
        print(
            f"  {run['seed']:>4} {np.median(tau_of(chart)[apex]):>14.2f} "
            f"{np.median(tau_full[apex]):>14.2f} "
            f"{np.median(tau_full[apex]) / 14.0:>25.2f}x "
            f"{int((tau_full >= loop).sum()):>18}/{run['cells']}"
        )

    print("\nCHECKS, per run")
    for run in runs:
        checks = run["checks"]
        exact = checks.get("relay_identity_tick_1", {})
        stepped = checks.get("relay_identity_tick_10", {})
        print(
            f"  seed {run['seed']:>4}  relay identity, tick 1 (no transport step yet): "
            f"max rel err {exact.get('max_relative_error', float('nan')):.2e}; "
            f"tick 10: {stepped.get('max_relative_error', float('nan')):.2e}; "
            f"chart-only tau vs the #206 rig: "
            f"{checks.get('chart_only_vs_206_tau_max_abs', float('nan')):.2e}"
        )


if __name__ == "__main__":
    main()
