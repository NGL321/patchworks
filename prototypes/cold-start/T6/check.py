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


def main() -> None:
    con = load(_HERE / "555-construction.json")
    dep = load(_HERE / "555-departure.json")

    if con:
        rows = {r["rung"]: r for r in con["rows"]}
        print("construction, rebuilt (555-construction.json)")
        # The instrument's calibration: the "today" control must reproduce the
        # ledger's construction reading, or nothing else here is worth reading.
        near("today built ER median (ledger: 1.0255)", rows["today"]["composed_er"]["median"], 1.0255)
        near("today generic ER median", rows["today"]["generic"]["median"], 1.0273)
        near("stack built ER median", rows["b_invariant"]["composed_er"]["median"], 1.5933)
        near("stack built ER p90", rows["b_invariant"]["composed_er"]["p90"], 1.5933 + 0.4588)
        near("stack generic ER median (#540 priced 2.193)",
             rows["b_invariant"]["generic"]["median"], 2.0637)
        near("(d) per-edge built ER median", rows["d_per_edge"]["composed_er"]["median"], 1.0991)
        near("(c1) laterals built ER median", rows["c1_laterals"]["composed_er"]["median"], 1.1669)

        print("\n#555 item 3: the Gram cap does not move across the rungs")
        for rung, want in (("today", 0.3288), ("d_per_edge", 0.3271),
                           ("c1_laterals", 0.3252), ("b_invariant", 0.3294)):
            near(f"{rung} cap ratio median", rows[rung]["cap"]["ratio_median"], want)
            exact(f"{rung} cells at cap", rows[rung]["cap"]["cells_at_cap"], 4)
        near("today leading cosine", rows["today"]["cos_leading_per_hop"]["median"], 0.5680)
        near("stack leading cosine", rows["b_invariant"]["cos_leading_per_hop"]["median"], 0.9372)

        print("\n#555 item 4: what the doubled invariant costs")
        exact("today private dim total", rows["today"]["invariant"]["private_dim_total"], 1278)
        exact("stack private dim total", rows["b_invariant"]["invariant"]["private_dim_total"], 35)
        exact("stack cells at private dim 0",
              rows["b_invariant"]["invariant"]["private_dim_zero_cells"], 122)
        exact("stack predicting cells", rows["b_invariant"]["invariant"]["predicting_cells"], 150)
        exact("stack invariant violations", rows["b_invariant"]["invariant"]["violations"], 0)
        exact("(d) invariant violations at n-1", rows["d_per_edge"]["invariant"]["violations"], 0)
        exact("(c1) invariant violations at n-1", rows["c1_laterals"]["invariant"]["violations"], 0)
        exact("stack modal chain widths",
              rows["b_invariant"]["widths"]["modal"], [4, 12, 12, 13, 12, 13, 12])

    if dep:
        rows = {r["rung"]: r for r in dep["rows"]}
        print("\nattribution of the generic-vs-built gap (555-departure.json)")
        s = rows["b_invariant"]["ladder"]
        near("stack generic (Haar V, unit sigma)", s["generic"]["median"], 2.0637)
        near("stack real V, unit sigma", s["real_V"]["median"], 2.0756)
        near("stack real V and real sigma", s["real_all"]["median"], 1.5933)
        near("stack truth (composed_er)", s["truth"]["median"], 1.5933)
        near("stack sigma min/max median",
             rows["b_invariant"]["sigma"]["sigma_min_over_max_median"], 0.3203)
        near("today sigma min/max median",
             rows["today"]["sigma"]["sigma_min_over_max_median"], 0.6275)

    trained = load(_HERE / "555-b_invariant-baseline-seed42-20000.json")
    if trained:
        print("\nthe trained arm (555-b_invariant-baseline-seed42-20000.json)")
        near("construction ER median",
             trained["at_construction"]["angles"]["composed_er"]["median"], 1.5933)
        first = trained["checkpoints"][0]
        exact("first checkpoint is 100 ticks", first["ticks"], 100)
        near("sigma closes to 1 by 100 ticks",
             first["sigma"]["sigma_min_over_max_median"], 1.0000, tol=1e-3)
        near("ER at 100 ticks", first["angles"]["composed_er"]["median"], 1.9537, tol=2e-3)
        last = trained["checkpoints"][-1]
        print(f"  ..  horizon {last['ticks']}: ER median "
              f"{last['angles']['composed_er']['median']:.6f}, "
              f"cap p90 {last['cap']['ratio_p90']:.4f}, "
              f"cells at cap {last['cap']['cells_at_cap']}")

    m14 = load(_T5 / "546-baseline-m14-seed42-100000.json")
    if m14:
        print("\nB6's m=14 comparator (T5/546-baseline-m14-seed42-100000.json)")
        near("m=14 construction ER",
             m14["at_construction"]["angles"]["composed_er"]["median"], 1.6158)
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
