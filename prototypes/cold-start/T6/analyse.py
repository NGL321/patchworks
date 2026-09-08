"""T6 (#555): the readout tables for #556's two arms, built on the shipped spec.

Prints, in the order [#555](https://github.com/NGL321/patchworks/issues/555) asks
its four questions:

1. the arms at construction, beside #540's generic instrument at each arm's
   **realised** widths, and beside what each costs in private dimension;
2. the attribution of the generic-vs-built gap (Haar `V` -> real `V` -> real
   `sigma`), which is #540's own stated falsifier;
3. the trained trajectories, both seeds, against the record's nearest arms;
4. whether either arm clears the bar.

Usage::

    python prototypes/cold-start/T6/analyse.py
"""

from __future__ import annotations

import json
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_T4 = _HERE.parent / "T4"
_T5 = _HERE.parent / "T5"

ARMS_CONSTRUCTION = _HERE / "555-arms-construction.json"
DEPARTURE = _HERE / "555-arms-departure.json"

#: The trained arms, and the record's nearest comparators.
TRAINED = [
    ("reserve s42 100k", _HERE / "555-reserve-baseline-seed42-100000.json"),
    ("reserve s42 20k", _HERE / "555-reserve-baseline-seed42-20000.json"),
    ("reserve s43 20k", _HERE / "555-reserve-baseline-seed43-20000.json"),
    ("doubling s42 20k", _HERE / "555-doubling-baseline-seed42-20000.json"),
    ("doubling s43 20k", _HERE / "555-doubling-baseline-seed43-20000.json"),
    ("B6 m=14 100k", _T5 / "546-baseline-m14-seed42-100000.json"),
    ("B1 m=3 100k", _T4 / "537-baseline-seed42-100000.json"),
]

BAR = 2.0


def _rows(path: Path):
    return json.loads(path.read_text())["rows"] if path.exists() else []


def construction() -> None:
    print("1. #556's arms at construction, on the shipped spec (seed 42)")
    print(
        f"  {'arm':>9} {'built ER':>9} {'p90':>7} {'max':>7} {'generic':>8} "
        f"{'k_v med':>8} {'priv min':>9} {'priv 0':>8} {'priv tot':>9} {'lanes':>8}"
    )
    for r in _rows(ARMS_CONSTRUCTION):
        e, g, p, w = r["composed_er"], r["generic"], r["privacy"], r["widths"]
        print(
            f"  {r['arm']:>9} {e['median']:>9.4f} {e['p90']:>7.4f} {e['max']:>7.4f} "
            f"{g['median']:>8.4f} {p['k_v_median']:>8.0f} {p['private_dim_min']:>9.0f} "
            f"{p['private_dim_zero_cells']:>3}/{p['predicting_cells']:<4} "
            f"{p['private_dim_total']:>9} "
            f"{w['interior_m_min']:>3}-{w['interior_m_max']:<4}"
        )
    print("\n   alignment and incoherence at construction (#555 item 3)")
    print(
        f"  {'arm':>9} {'cos lead':>9} {'cos 2nd':>9} {'gap':>8} {'cos all':>9} "
        f"{'s2/s1':>8} {'sigma':>8} {'cap p90':>8} {'at cap':>8}"
    )
    for r in _rows(ARMS_CONSTRUCTION):
        lead = r["cos_leading_per_hop"]["median"]
        sec = r["cos_second_per_hop"]["median"]
        print(
            f"  {r['arm']:>9} {lead:>9.4f} {sec:>9.4f} {lead - sec:>8.4f} "
            f"{r['cos_all']['median']:>9.4f} {r['s2_over_s1']['median']:>8.4f} "
            f"{r['sigma_min_over_max']['median']:>8.4f} {r['cap']['ratio_p90']:>8.4f} "
            f"{r['cap']['cells_at_cap']:>3}/{r['cap']['cells_measured']:<4}"
        )


def departure() -> None:
    rows = _rows(DEPARTURE)
    if not rows:
        print("\n2. [missing] 555-arms-departure.json")
        return
    print("\n2. where the built reading departs from #540's generic prediction")
    print(
        f"  {'arm':>9} {'generic':>9} {'real V':>9} {'real all':>9} {'truth':>9} "
        f"{'sigma m/M':>10} {'sigma cv':>9}"
    )
    for r in rows:
        lad, s = r["ladder"], r["sigma"]
        print(
            f"  {r['rung']:>9} {lad['generic']['median']:>9.4f} {lad['real_V']['median']:>9.4f} "
            f"{lad['real_all']['median']:>9.4f} {lad['truth']['median']:>9.4f} "
            f"{s['sigma_min_over_max_median']:>10.4f} {s['sigma_cv_median']:>9.4f}"
        )
    print("   real V ~ generic says the carried subspaces ARE generic: #540's law stands.")
    print("   real all << real V says the gap is singular-value spread, which #540's")
    print("   instrument set to 1 and which ADR-0032's band closes by 100 ticks.")


def trained() -> None:
    print("\n3. the trained trajectories")
    for label, path in TRAINED:
        if not path.exists():
            print(f"\n  {label}: [missing] {path.name}")
            continue
        d = json.loads(path.read_text())
        c = d["at_construction"]["angles"]
        base = c["composed_er"]["median"] - 1.0
        print(f"\n  {label}  construction ER {c['composed_er']['median']:.6f} (excess {base:.4e})")
        print(
            f"    {'ticks':>7} {'ER med':>10} {'excess':>11} {'erosion':>9} {'p90':>8} "
            f"{'max':>8} {'lead':>7} {'2nd':>7} {'cap p90':>8}"
        )
        for cp in d["checkpoints"]:
            a = cp["angles"]
            ex = a["composed_er"]["median"] - 1.0
            print(
                f"    {cp['ticks']:>7} {a['composed_er']['median']:>10.6f} {ex:>11.3e} "
                f"x{base / max(ex, 1e-300):>8.1f} {a['composed_er']['p90']:>8.4f} "
                f"{a['composed_er']['max']:>8.4f} "
                f"{a['cos_leading_per_hop']['median']:>7.4f} "
                f"{a['cos_second_per_hop']['median']:>7.4f} "
                f"{cp['cap']['ratio_p90']:>8.4f}"
            )


def verdict() -> None:
    print(f"\n4. against the bar ({BAR}, generic median)")
    print(f"  {'arm':>18} {'ticks':>7} {'median':>9} {'p90':>8} {'excess':>9} {'verdict':>9}")
    for label, path in TRAINED[:5]:
        if not path.exists():
            continue
        d = json.loads(path.read_text())
        last = d["checkpoints"][-1]
        e = last["angles"]["composed_er"]
        print(
            f"  {label:>18} {last['ticks']:>7} {e['median']:>9.4f} {e['p90']:>8.4f} "
            f"{e['median'] - 1:>9.4f} "
            f"{'CLEARS' if e['median'] >= BAR else 'FAILS':>9}"
        )
    print("\n   the bar's excess over one is 1.0. Reserve delivers ~0.42-0.53,")
    print("   doubling ~0.07-0.09 -- and the reserve arm's p90 sits above the bar.")


def main() -> None:
    construction()
    departure()
    trained()
    verdict()


if __name__ == "__main__":
    main()
