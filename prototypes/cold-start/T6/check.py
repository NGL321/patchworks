"""T6 (#555): re-derive every figure #555's readout asserts, from the records.

[T5](#546)'s `check.py` in the same spirit: the readout is written from these
files, so this reads them back and fails loudly if a number moved. Run it before
trusting a figure quoted anywhere off this rig.

Usage::

    python prototypes/cold-start/T6/check.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_T5 = _HERE.parent / "T5"

TOL = 5e-4
failures: list[str] = []


def load(path: Path):
    if not path.exists():
        failures.append(f"[missing] {path.name}")
        return None
    return json.loads(path.read_text())


def near(label: str, got, want, tol=TOL) -> None:
    if got is None:
        failures.append(f"{label}: no value")
    elif abs(got - want) > tol:
        failures.append(f"{label}: {got!r} != {want!r} (tol {tol})")
    else:
        print(f"  ok  {label} = {got:.6g}")


def exact(label: str, got, want) -> None:
    if got != want:
        failures.append(f"{label}: {got!r} != {want!r}")
    else:
        print(f"  ok  {label} = {got!r}")


#: Trained figures are checked loosely, and the reason is measured rather than
#: assumed: two runs of the **same arm at the same seed** -- the standalone 20k
#: run and the 100k run's own 20k checkpoint -- agree to 3e-6 at 2,000 ticks and
#: drift to 6e-3 by 20,000. `torch.set_num_threads(1)` is set, so this is
#: floating-point accumulation amplified by a chaotic trajectory, not a seeding
#: bug. It is far below every effect this rig reports (the arms differ by 0.35)
#: but it does mean a trained median is not a reproducible fourth decimal.
TRAINED_TOL = 0.02


def horizon(path: Path, label: str, med: float, p90: float, erosion: float) -> None:
    d = load(path)
    if not d:
        return
    base = d["at_construction"]["angles"]["composed_er"]["median"] - 1.0
    last = d["checkpoints"][-1]
    e = last["angles"]["composed_er"]
    near(f"{label} median @{last['ticks']}", e["median"], med, tol=TRAINED_TOL)
    near(f"{label} p90 @{last['ticks']}", e["p90"], p90, tol=TRAINED_TOL)
    near(f"{label} erosion @{last['ticks']}", base / max(e["median"] - 1.0, 1e-300), erosion, tol=0.2)


def main() -> None:
    con = load(_HERE / "555-arms-construction.json")
    if con:
        rows = {r["arm"]: r for r in con["rows"]}
        print("construction, on the shipped spec (555-arms-construction.json)")
        # The rig must reproduce #556's construction figures, or it is not
        # measuring the arms #556 ruled.
        near("shipped built ER", rows["shipped"]["composed_er"]["median"], 1.1489)
        near("shipped generic ER (#556 read 1.304)", rows["shipped"]["generic"]["median"], 1.3029)
        exact("shipped private dim total (#548 read 914)",
              rows["shipped"]["privacy"]["private_dim_total"], 914)
        exact("shipped cells at private dim 0", rows["shipped"]["privacy"]["private_dim_zero_cells"], 0)

        near("doubling built ER", rows["doubling"]["composed_er"]["median"], 1.5757)
        near("doubling generic ER (#556 read 2.073)", rows["doubling"]["generic"]["median"], 2.0665)
        exact("doubling private dim total (#556 read 54)",
              rows["doubling"]["privacy"]["private_dim_total"], 54)
        exact("doubling cells at private dim 0 (#556 read 104)",
              rows["doubling"]["privacy"]["private_dim_zero_cells"], 104)

        near("reserve built ER", rows["reserve"]["composed_er"]["median"], 1.3825)
        near("reserve generic ER (#556 read 2.231)", rows["reserve"]["generic"]["median"], 2.2652)
        exact("reserve private dim total (#556 read 1200)",
              rows["reserve"]["privacy"]["private_dim_total"], 1200)
        exact("reserve cells at private dim 0", rows["reserve"]["privacy"]["private_dim_zero_cells"], 0)
        exact("reserve private dim min = p", int(rows["reserve"]["privacy"]["private_dim_min"]), 8)

        print("\n#555 item 3: the Gram cap does not move across the arms at construction")
        for arm in ("shipped", "doubling", "reserve"):
            exact(f"{arm} cells at cap", rows[arm]["cap"]["cells_at_cap"], 4)
            near(f"{arm} cap ratio p90", rows[arm]["cap"]["ratio_p90"],
                 {"shipped": 0.3851, "doubling": 0.3900, "reserve": 0.3859}[arm], tol=2e-3)
        near("reserve leading cosine", rows["reserve"]["cos_leading_per_hop"]["median"], 0.9784, tol=2e-3)
        near("doubling leading cosine", rows["doubling"]["cos_leading_per_hop"]["median"], 0.9335, tol=2e-3)

        print("\n   zero invariant violations at every arm")
        for arm in ("shipped", "doubling", "reserve"):
            exact(f"{arm} violations", rows[arm]["privacy"]["violations"], 0)

    dep = load(_HERE / "555-arms-departure.json")
    if dep:
        rows = {r["rung"]: r for r in dep["rows"]}
        print("\nattribution of the generic-vs-built gap (555-arms-departure.json)")
        for arm, (g, rv, ra) in {
            "shipped": (1.3029, 1.2688, 1.1489),
            "doubling": (2.0665, 2.0420, 1.5757),
            "reserve": (2.2652, 2.2203, 1.3825),
        }.items():
            lad = rows[arm]["ladder"]
            near(f"{arm} generic (Haar V, unit sigma)", lad["generic"]["median"], g, tol=2e-3)
            near(f"{arm} real V, unit sigma", lad["real_V"]["median"], rv, tol=2e-3)
            near(f"{arm} real V and real sigma", lad["real_all"]["median"], ra, tol=2e-3)
            near(f"{arm} truth equals real_all", lad["truth"]["median"], ra, tol=2e-3)

    print("\nthe trained arms")
    horizon(_HERE / "555-reserve-baseline-seed42-100000.json", "reserve s42", 1.4237, 2.2791, 0.9)
    horizon(_HERE / "555-reserve-baseline-seed42-20000.json", "reserve s42 20k", 1.4386, 2.3504, 0.9)
    horizon(_HERE / "555-reserve-baseline-seed43-20000.json", "reserve s43 20k", 1.5307, 2.3244, 0.9)
    horizon(_HERE / "555-doubling-baseline-seed42-20000.json", "doubling s42 20k", 1.0936, 1.6720, 6.2)
    horizon(_HERE / "555-doubling-baseline-seed43-20000.json", "doubling s43 20k", 1.0725, 1.6336, 6.7)

    res = load(_HERE / "555-reserve-baseline-seed42-100000.json")
    if res:
        print("\n   the reserve arm does not erode: every checkpoint from 10k on")
        base = res["at_construction"]["angles"]["composed_er"]["median"]
        for cp in res["checkpoints"]:
            if cp["ticks"] >= 10_000:
                m = cp["angles"]["composed_er"]["median"]
                mark = "ok" if m > base else "!!"
                print(f"  {mark}  @{cp['ticks']:>6}: {m:.4f} against construction {base:.4f}")
                if m <= base:
                    failures.append(f"reserve @{cp['ticks']} fell below construction")
        near("sigma closes to 1 by 100 ticks",
             res["checkpoints"][0]["sigma"]["sigma_min_over_max_median"], 1.0, tol=1e-3)

    m14 = load(_T5 / "546-baseline-m14-seed42-100000.json")
    if m14:
        print("\nB6's m=14 comparator, the previous best trained on the record")
        near("m=14 at 100k", m14["checkpoints"][-1]["angles"]["composed_er"]["median"], 1.1379)

    print()
    if failures:
        print(f"FAILED ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("all figures re-derived.")


if __name__ == "__main__":
    main()
