"""Dwell and `tau` from the **same run**, against `|loop(c)|` (#361).

[#226](https://github.com/NGL321/patchworks/issues/226) paired
[#206](https://github.com/NGL321/patchworks/issues/206)'s
`final_cumulative_dwell` against [#274](https://github.com/NGL321/patchworks/issues/274)'s
per-cell `tau_full` and found the median `dwell/tau` falling from the published
9.49 to 1.997. But dwell was held at seed 42's while `tau` varied across #274's
nine seeds, so the reading was a **sensitivity statement, not a measurement** --
the exact defect [#208](https://github.com/NGL321/patchworks/issues/208) SS3
exists to kill. This rig closes it: one run per seed, both quantities read off
that run, at a horizon held fixed across seeds.

**No new instrument.** `spectra`, `tau_of` and both checks are imported from
`prototypes/driven-rho-274/read.py`; dwell is `FoldRead.dwell`, the cumulative
estimator #208 fixed, read off the same live tick. The two rigs already agree to
~1e-5 on the chart-only half and `check_chart_only_matches_206` still runs here,
so what is new is the **join** and the **sweep**, not the measurement surface.

**The consequence with no `tau` in it.** #226's chain is
`|loop(c)| <= tau_c < dwell_c`, whose consequence is `dwell_c > |loop(c)|` --
both sides graph quantities, readable without settling the operator
controversy. A cell failing it has an **empty admissible band**: no value of
`tau` both conducts and is licensed. That count is this rig's headline, and
`|loop(c)|` comes per cell from `loops.py`'s enumeration of the mask, never from
a level (#181).

**Every number written into the record from this reading is a range over
seeds**, never a median from one run: `tau = -1/ln rho` is violently sensitive
near `rho = 1` (#274's binding caveat), so the seed spread is the honest object.

Usage::

    PYTHONPATH=src python prototypes/admissible-band-361/read.py --ticks 30000 --seed 42
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_BENCH = str(_HERE.parents[1] / "benchmarks")
if _BENCH not in sys.path:
    sys.path.append(_BENCH)
from untrained_fixed_point import build, teaching  # noqa: E402

from loops import loop_lengths  # noqa: E402


def _load_274():
    """#274's rig, loaded by path under its own name.

    Both rigs are called `read`, and #274's own `check_chart_only_matches_206`
    has the same problem with #206's -- a plain import would find the wrong one.
    """
    source = _HERE.parents[0] / "driven-rho-274" / "read.py"
    spec = importlib.util.spec_from_file_location("driven_rho_274", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_RHO = _load_274()

#: #206's checkpoint ladder, which #274 also uses, so a checkpoint here is
#: comparable to the same checkpoint in either rig.
CHECKPOINTS = (
    100, 200, 500, 1_000, 2_000, 5_000, 10_000, 20_000, 30_000, 50_000,
    75_000, 100_000,
)


def quantiles(values: np.ndarray) -> dict[str, float]:
    q = np.quantile(values, [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0])
    return {
        "min": float(q[0]), "p05": float(q[1]), "p25": float(q[2]),
        "median": float(q[3]), "p75": float(q[4]), "p95": float(q[5]),
        "max": float(q[6]),
    }


def band(dwell: np.ndarray, tau: np.ndarray, loop: np.ndarray) -> dict:
    """The admissible band `|loop(c)| <= tau_c < dwell_c`, per cell.

    Three readings, kept apart because they fail for different reasons:

    - `empty_band` -- `dwell <= |loop|`, the consequence with no `tau` in it.
      No `tau` can satisfy both ends, whatever the operator controversy does.
    - `conducts` -- `tau >= |loop|`, ADR-0026's conduction ratio, which is the
      reading that is short everywhere today.
    - `licensed` -- `dwell > tau`, ADR-0005's validity condition on the
      spectral instrument as #226 re-pointed it.
    """
    empty = dwell <= loop
    conducts = tau >= loop
    licensed = dwell > tau
    return {
        "cells": int(dwell.size),
        "empty_band": int(empty.sum()),
        "conducts": int(conducts.sum()),
        "licensed": int(licensed.sum()),
        "band_open_and_occupied": int((conducts & licensed).sum()),
        "dwell_over_loop": quantiles(dwell / loop),
        "tau_over_loop": quantiles(tau / loop),
        "dwell_over_tau": quantiles(dwell / np.maximum(tau, 1e-12)),
    }


def row(agent, reached: int, loop: np.ndarray) -> dict:
    """One checkpoint: dwell and both `tau` off the same live state."""
    read = agent.sheaf.fold_read
    dwell = read.dwell.numpy().astype(float)
    radii = _RHO.spectra(agent)
    tau_chart = _RHO.tau_of(radii["chart"])
    tau_full = _RHO.tau_of(radii["full"])
    return {
        "ticks": reached,
        "dwell": quantiles(dwell),
        "tau_chart": quantiles(tau_chart),
        "tau_full": quantiles(tau_full),
        "against_loop_full": band(dwell, tau_full, loop),
        "against_loop_chart": band(dwell, tau_chart, loop),
        "per_cell": {
            "dwell": dwell.tolist(),
            "tau_chart": tau_chart.tolist(),
            "tau_full": tau_full.tolist(),
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dome", default="real", choices=("small", "real"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ticks", type=int, default=30_000)
    parser.add_argument("--out", type=Path, default=_HERE)
    parser.add_argument("--tag", default=None)
    args = parser.parse_args(argv)

    _, agent = build(args.dome, args.split, args.seed)
    dome = agent.dome
    cells = [dome.cells[c] for c in dome.predicting]
    levels = np.array([cell.index.level for cell in cells])

    # `|loop(c)|` per cell, in the row order dwell and `tau` are indexed by.
    lengths = loop_lengths(dome)
    loop = np.array([lengths[cell.id] for cell in cells], dtype=float)

    checkpoints = [c for c in CHECKPOINTS if c <= args.ticks]
    print(f"dome={args.dome} split={args.split} seed={args.seed} ticks={args.ticks}")
    print(
        f"{len(cells)} predicting cells; |loop(c)| enumerated from the mask, "
        f"{int(loop.min())} to {int(loop.max())} ticks"
    )

    rows = []
    checked: dict[str, object] = {}
    start = time.time()
    for reached, _outcome in enumerate(teaching(agent, args.ticks, args.seed), start=1):
        if reached in _RHO.IDENTITY_TICKS:
            checked[f"relay_identity_tick_{reached}"] = _RHO.check_relay_identity(agent)
        if reached == 10:
            checked["chart_only_vs_206_tau_max_abs"] = (
                _RHO.check_chart_only_matches_206(agent, _RHO.spectra(agent)["chart"])
            )
            print(
                f"  chart-only tau against the #206 rig: "
                f"{checked['chart_only_vs_206_tau_max_abs']:.2e}"
            )
        if reached in checkpoints:
            entry = row(agent, reached, loop)
            rows.append(entry)
            full = entry["against_loop_full"]
            rate = reached / (time.time() - start)
            print(
                f"  {reached:>7} ticks: dwell median {entry['dwell']['median']:8.2f}  "
                f"tau_full median {entry['tau_full']['median']:7.3f}  "
                f"dwell/tau {full['dwell_over_tau']['median']:7.3f}  "
                f"empty band {full['empty_band']:>3}/{full['cells']}  "
                f"conducts {full['conducts']:>3}  licensed {full['licensed']:>3}  "
                f"({rate:.0f} tick/s)",
                flush=True,
            )

    final = rows[-1]
    dwell = np.array(final["per_cell"]["dwell"])
    tau_full = np.array(final["per_cell"]["tau_full"])
    print("\n  AT THE HORIZON, by level (a reporting axis only; nothing is concluded")
    print("  from it -- the bar is indexed per cell, #181)")
    print("    level  cells  |loop|  dwell  tau_full  dwell/tau  empty  conducts")
    by_level = []
    for level in sorted(set(levels.tolist())):
        mask = levels == level
        entry = {
            "level": int(level),
            "cells": int(mask.sum()),
            "loop": sorted({int(v) for v in loop[mask]}),
            "dwell": float(np.median(dwell[mask])),
            "tau_full": float(np.median(tau_full[mask])),
            "dwell_over_tau": float(np.median(dwell[mask] / np.maximum(tau_full[mask], 1e-12))),
            "empty_band": int((dwell[mask] <= loop[mask]).sum()),
            "conducts": int((tau_full[mask] >= loop[mask]).sum()),
        }
        by_level.append(entry)
        print(
            f"    L{entry['level']:<5} {entry['cells']:>5} "
            f"{'/'.join(str(v) for v in entry['loop']):>7} "
            f"{entry['dwell']:>6.1f} {entry['tau_full']:>9.2f} "
            f"{entry['dwell_over_tau']:>10.2f} {entry['empty_band']:>6} "
            f"{entry['conducts']:>9}"
        )

    tag = args.tag or f"{args.dome}-{args.split}-seed{args.seed}-{args.ticks}"
    args.out.mkdir(parents=True, exist_ok=True)
    payload = {
        "dome": args.dome, "split": args.split, "seed": args.seed,
        "ticks": args.ticks, "cells": len(cells),
        "elapsed_minutes": (time.time() - start) / 60,
        "checks": checked,
        "levels": levels.tolist(),
        "cell_ids": [cell.id for cell in cells],
        "loop_lengths": loop.astype(int).tolist(),
        "by_level_at_horizon": by_level,
        "checkpoints": rows,
    }
    (args.out / f"361-{tag}.json").write_text(json.dumps(payload, indent=2))
    print(f"\n  Written to {args.out}/361-{tag}.json")


if __name__ == "__main__":
    main()
