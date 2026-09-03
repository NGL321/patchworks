"""027's timescale spread, re-run against the converted body (#349).

`docs/research/027-regional-jacobian-spectra.md` measured the across-cell spread
of `tau` on a **stand-in** body -- iid Gaussian ReLU MLPs at `k = 12, n = 32` --
and said so in its own caveats: *"This establishes the shape of the answer and
the sensitivity of the rig, not the body's number."* Its first listed
measurement-to-run was a re-run against the real body. This is that re-run, and
what makes it possible is that the body now exists.

**What changed under the rig, and why the original cannot simply be re-pointed.**
027 read the spectral radius of a *regional* Jacobian of a piecewise-linear ReLU
`step`, and the Koopman conversion (#138) made `step` a linear `K`, which has
**one region globally**. `spread_pilot.py`'s object no longer exists in the
design. What replaced it is not one quantity but three, named separately by
[ADR-0028](../../docs/adr/0028-a-cell-holds-a-spectrum-of-retention-constants.md):

* `rho(K)` -- the **operator's retention**, per-cell, learned, region-independent.
  ADR-0028's reported quantity. Read here; #274 did not record it.
* `rho(K @ J_encode)` -- the **realised chart retention**, region-dependent and
  per-tick. #206's object.
* `rho(K @ (J_chart + relay))` -- the **full chart loop**, which #274 established
  is the reading that does not omit a real term of the recurrence.

The last two are already measured, per cell and per checkpoint, across nine
driven seeds in `prototypes/driven-rho-274/`. This rig reads `rho(K)` off a
driven run and then re-expresses **all three** in 027's own spread statistic, so
the comparison against 7.7x is like-for-like rather than a change of units.

**The statistic carries 027 section 5's protocol correction with it.** The ratio
is quantiles of `tau`, not moments; expansive cells are excluded and counted
rather than clipped, because a cell at `rho >= 1` has no finite retention
constant and folding one in at a ceiling invents the tail the ratio is most
sensitive to; and `sd(log10 rho)` is reported beside it, because that is the
statistic that stays finite when the ratio does not.

Usage::

    # The spread of `rho(K)` off a driven run, and the roll-up against #274.
    PYTHONPATH=src python prototypes/regional-spectra/converted_spread.py
    PYTHONPATH=src python prototypes/regional-spectra/converted_spread.py --ticks 2000

    # The roll-up alone, over #274's stored seeds -- no run, seconds not minutes.
    python prototypes/regional-spectra/converted_spread.py --stored-only
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from typing import Iterable

import numpy as np

__all__ = ["spread_statistics", "stored_seeds", "format_table"]

#: The stand-in's numbers, for the comparison the amendment is about. 027
#: section 6 and section 7: a `tau` p95/p05 ratio of 7.7 from biases alone at a
#: fixed operating point, and `sd(log10 rho)` per width configuration.
PILOT_TAU_RATIO = 7.7
PILOT_SD_LOG10_RHO = {
    "[12,12]": 0.32,
    "[32]": 0.12,
    "[64,64]": 0.13,
    "[128,128,128]": 0.13,
}
#: #42's re-measurement on the same stand-in, under draw-then-select.
SELECTION_TAU_RATIO_DRAWN = 4.5
SELECTION_TAU_RATIO_SELECTED = 16.0


def spread_statistics(rho: Iterable[float]) -> dict[str, float]:
    """027's spread statistic over one population of spectral radii.

    `tau = -1/ln rho` at p95 over p05, taken over the cells with `rho < 1`;
    `sd(log10 rho)` over every cell. Returns `nan` for the ratio when no cell
    admits a finite `tau`, rather than a number standing in for one.
    """
    radii = np.asarray(list(rho), dtype=float)
    if radii.size == 0:
        raise ValueError("no cells")
    contracting = radii[radii < 1.0]
    if contracting.size:
        tau = -1.0 / np.log(np.clip(contracting, 1e-12, None))
        p05, p50, p95 = (float(v) for v in np.quantile(tau, [0.05, 0.5, 0.95]))
        ratio = p95 / p05 if p05 > 0 else float("nan")
    else:
        p05 = p50 = p95 = ratio = float("nan")
    return {
        "cells": int(radii.size),
        "finite": int(contracting.size),
        "expansive": int((radii >= 1.0).sum()),
        "tau_p05": p05,
        "tau_median": p50,
        "tau_p95": p95,
        "tau_p95_over_p05": ratio,
        "rho_median": float(np.median(radii)),
        "sd_log10_rho": float(np.std(np.log10(np.clip(radii, 1e-12, None)))),
    }


def stored_seeds(directory: pathlib.Path) -> list[dict]:
    """#274's per-cell radii at each run's horizon, one entry per seed.

    Reads the files rather than re-running: the two realised readings are
    already on disk across nine seeds, and re-deriving them here would produce a
    second set of numbers to reconcile against the first.
    """
    out = []
    paths = sorted(
        directory.glob("274-*.json"),
        key=lambda p: int(p.stem.split("seed")[1].split("-")[0]),
    )
    for path in paths:
        run = json.loads(path.read_text(encoding="utf-8"))
        horizon = run["checkpoints"][-1]
        out.append(
            {
                "seed": run["seed"],
                "ticks": run["ticks"],
                "chart": spread_statistics(horizon["per_cell"]["rho_chart"]),
                "full": spread_statistics(horizon["per_cell"]["rho_full"]),
            }
        )
    return out


def radii_of_operators(agent) -> np.ndarray:
    """`rho(K)` per predicting cell, off the live operators."""
    import torch  # noqa: PLC0415

    with torch.no_grad():
        return agent.sheaf.operators.radii().double().numpy()


_HEADER = (
    "    seed   ticks   cells  expansive   tau p05   tau med   tau p95"
    "   p95/p05   sd(log10 rho)"
)


def _row(seed: int, ticks: int, s: dict[str, float]) -> str:
    return (
        f"    {seed:>4} {ticks:>7} {s['cells']:>7} {s['expansive']:>10}"
        f"   {s['tau_p05']:7.2f}   {s['tau_median']:7.2f}   {s['tau_p95']:7.2f}"
        f"   {s['tau_p95_over_p05']:7.1f}   {s['sd_log10_rho']:13.3f}"
    )


def format_table(rows: list[dict], key: str, label: str) -> str:
    lines = [f"  {label}", _HEADER]
    ratios = []
    for entry in rows:
        s = entry[key]
        ratios.append(s["tau_p95_over_p05"])
        lines.append(_row(entry["seed"], entry["ticks"], s))
    finite = [r for r in ratios if not math.isnan(r)]
    if finite:
        lines.append(
            f"    across {len(finite)} seeds the p95/p05 ratio runs "
            f"{min(finite):.1f}x to {max(finite):.1f}x "
            f"(median {float(np.median(finite)):.1f}x)"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dome", default="real", choices=("small", "real"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--ticks", type=int, default=2_000)
    parser.add_argument(
        "--stored-only",
        action="store_true",
        help="skip the driven run; roll up #274's stored seeds only",
    )
    parser.add_argument(
        "--stored",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[1] / "driven-rho-274",
    )
    args = parser.parse_args(argv)

    print("027's timescale spread, re-run against the converted body (#349)\n")
    print(
        f"  The stand-in 027 measured reported a tau p95/p05 ratio of "
        f"{PILOT_TAU_RATIO}x from biases alone, and sd(log10 rho) of "
        + ", ".join(f"{v} at {k}" for k, v in PILOT_SD_LOG10_RHO.items())
        + f".\n  #42 re-measured the same stand-in under draw-then-select: "
        f"{SELECTION_TAU_RATIO_DRAWN}x drawn, "
        f"{SELECTION_TAU_RATIO_SELECTED:.0f}x selected.\n"
    )

    rows = stored_seeds(args.stored)
    if not rows:
        print(f"  no #274 runs found under {args.stored}")
    else:
        print("REALISED RETENTION, driven, off #274's stored per-cell radii")
        print(
            format_table(
                rows,
                "chart",
                "rho(K @ J_chart) -- the realised chart retention (#206's object)",
            )
        )
        print()
        print(
            format_table(
                rows,
                "full",
                "rho(K @ (J_chart + relay)) -- the full chart loop (#274's correction)",
            )
        )
        print()

    if args.stored_only:
        return

    sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "benchmarks"))
    from untrained_fixed_point import build, teaching  # noqa: PLC0415

    print("OPERATOR RETENTION, rho(K) -- ADR-0028's reported quantity")
    print(_HEADER)
    for seed in args.seeds:
        _, agent = build(args.dome, args.split, seed)
        for _ in teaching(agent, args.ticks, seed):
            pass
        print(_row(seed, args.ticks, spread_statistics(radii_of_operators(agent))),
              flush=True)


if __name__ == "__main__":
    main()
