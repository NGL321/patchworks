"""T6 (#560/B13): re-derive every figure B13's readout asserts, from the records.

Same spirit as `check.py`, which does the job for [#555](#555) and still passes
unchanged — this rig added arms and scripts, it moved none of #555's data.

Usage::

    python prototypes/cold-start/T6/check_560.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent

failures: list[str] = []

#: See `check.py`: two runs of the same arm at the same seed are not reproducible
#: in the fourth decimal. B13 measured this **larger** than #555 did — the `p=12`
#: 20k standalone run reads 2.1309 where the 100k run's own 20k checkpoint reads
#: 2.0713, a drift of 0.060 against #555's measured 6e-3. Every ranking this
#: readout makes is separated by far more than that (p=12 to p=16 is 0.79), but
#: no fourth decimal here is a reproducible number.
TRAINED_TOL = 0.08
TOL = 5e-4


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


print("\nitem 1: the construction sweep, and the floor's arithmetic")
sweep = load(_HERE / "560-sweep-p-construction.json")
if sweep:
    rows = {(r["p"], r["seed"]): r for r in sweep["rows"]}
    # The built curve peaks at p = 0 and falls; no interior peak at construction.
    s42 = {p: r["composed_er"]["median"] for (p, s), r in rows.items() if s == 42}
    exact("construction peak over p (seed 42)", max(s42, key=s42.get), 0)
    near("built ER at p=0 (seed 42)", s42[0], 1.5884, 1e-3)
    near("built ER at p=28 (seed 42)", s42[28], 1.0056, 1e-3)
    # The generic instrument saturates at exactly the modal first-hop width.
    for p in (20, 22, 24, 26, 28):
        near(f"generic ER at p={p} saturates", rows[(p, 42)]["generic"]["median"], 4.0, 1e-9)
    # dim H^0 floor is 150p: p_v = p flat over 150 predicting cells.
    for p in (2, 8, 12, 16, 20):
        exact(f"floor at p={p}", rows[(p, 42)]["privacy"]["private_dim_total"], 150 * p)
        exact(f"predicting cells at p={p}", rows[(p, 42)]["privacy"]["predicting_cells"], 150)
    # The n - p cap is cheap to p=12 and then bites.
    for p, want in ((8, 2), (12, 2), (16, 10), (20, 70), (24, 194)):
        exact(f"cap binds at p={p}", rows[(p, 42)]["cap_binding"]["binding"], want)

print("\nB11's p=8 arm is the same object under an older label")
b11 = load(_HERE / "555-reserve-baseline-seed42-20000.json")
if b11 and sweep:
    near(
        "reserve_p8 reproduces #555's construction ER",
        rows[(8, 42)]["composed_er"]["median"],
        b11["at_construction"]["angles"]["composed_er"]["median"],
        1e-9,
    )
    exact("reserve_p8 reproduces #555's floor", rows[(8, 42)]["privacy"]["private_dim_total"], 1200)


def trained(path: Path, label: str, med: float, erosion: float, tol=TRAINED_TOL) -> None:
    d = load(path)
    if not d:
        return
    last = d["checkpoints"][-1]
    got = last["angles"]["composed_er"]["median"]
    near(f"{label} median", got, med, tol)
    base = d["at_construction"]["angles"]["composed_er"]["median"] - 1.0
    near(f"{label} erosion", base / max(got - 1.0, 1e-300), erosion, 0.6)


print("\nitem 2: the trained ladder inverts the construction ordering")
trained(_HERE / "560-reserve_p0-baseline-seed42-20000.json", "p=0 @20k", 1.0845, 6.97)
trained(_HERE / "560-reserve_p4-baseline-seed42-20000.json", "p=4 @20k", 1.1824, 3.13)
trained(_HERE / "555-reserve-baseline-seed42-20000.json", "p=8 @20k", 1.4386, 0.87)
trained(_HERE / "560-reserve_p12-baseline-seed42-20000.json", "p=12 @20k", 2.1309, 0.36)
trained(_HERE / "560-reserve_p16-baseline-seed42-20000.json", "p=16 @20k", 2.9171, 0.18)
trained(_HERE / "560-reserve_p20-baseline-seed42-20000.json", "p=20 @20k", 4.0000, 0.08)

print("\n  both candidates hold at the 100k horizon, and both clear the bar of 2.0")
trained(_HERE / "560-reserve_p12-baseline-seed42-100000.json", "p=12 @100k", 2.2127, 0.34)
trained(_HERE / "560-reserve_p16-baseline-seed42-100000.json", "p=16 @100k", 2.9408, 0.18)

print("\nitem 3: the second seed, and where B11's falsifier fires")
trained(_HERE / "560-reserve_p12-baseline-seed43-20000.json", "p=12 @20k seed 43", 2.0328, 0.38)
trained(_HERE / "560-reserve_p16-baseline-seed43-20000.json", "p=16 @20k seed 43", 2.8922, 0.21)

for path, label in (
    (_HERE / "560-reserve_p0-baseline-seed42-20000.json", "p=0"),
    (_HERE / "560-reserve_p4-baseline-seed42-20000.json", "p=4"),
):
    d = load(path)
    if d:
        base = d["at_construction"]["angles"]["composed_er"]["median"]
        end = d["checkpoints"][-1]["angles"]["composed_er"]["median"]
        if end >= base:
            failures.append(f"{label} did not fall below construction; falsifier does not fire")
        else:
            print(f"  ok  {label} falls below construction ({end:.4f} < {base:.4f}) — fires")

print("\n  the floor itself is a construction guarantee at every p, training-independent")
for p, path in (
    (12, _HERE / "560-reserve_p12-baseline-seed42-100000.json"),
    (16, _HERE / "560-reserve_p16-baseline-seed42-100000.json"),
):
    d = load(path)
    if d:
        exact(f"floor at p={p} on the trained arm", d["privacy"]["private_dim_total"], 150 * p)

print("\nthe price: how much of each lane is forced to overlap")
fo = load(_HERE / "560-forced-overlap.json")
if fo:
    rows = {r["p"]: r for r in fo["rows"]}
    for p, share, coincident in ((8, 0.0, 4), (12, 1 / 3, 19), (16, 2 / 3, 181), (20, 1.0, 1450)):
        near(f"forced share at p={p}", rows[p]["forced_share_median"], share, 1e-6)
        exact(f"fully coincident hops at p={p}", rows[p]["hops_fully_coincident"], coincident)
    exact("hops read", rows[12]["hops"], 1578)

if failures:
    print("\nFAILURES")
    for f in failures:
        print(f"  {f}")
    sys.exit(1)
print("\nall figures re-derived.")
