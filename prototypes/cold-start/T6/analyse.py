"""T6 (#555): the readout tables — the stack rebuilt, attributed, and trained.

Prints, in the order [#555](https://github.com/NGL321/patchworks/issues/555) asks
its four questions:

1. the rung ladder at construction, rebuilt, beside #540's generic instrument run
   at each rung's **realised** widths;
2. the attribution of the generic-vs-built gap (Haar `V` -> real `V` -> real
   `sigma`), which is #540's own stated falsifier;
3. the trained trajectory, beside [B6](#546)'s `m = 14` arm — the nearest thing on
   the record — with the leading-to-second cosine gap B6's headroom mechanism
   predicts from;
4. the private-dimension cost of the doubled invariant.

Usage::

    python prototypes/cold-start/T6/analyse.py
"""

from __future__ import annotations

import json
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_T4 = _HERE.parent / "T4"
_T5 = _HERE.parent / "T5"

CONSTRUCTION = _HERE / "555-construction.json"
DEPARTURE = _HERE / "555-departure.json"

#: The trained arms, and the record's nearest comparators.
TRAINED = [
    ("stack 20k", _HERE / "555-b_invariant-baseline-seed42-20000.json"),
    ("stack 100k", _HERE / "555-b_invariant-baseline-seed42-100000.json"),
    ("B6 m=14", _T5 / "546-baseline-m14-seed42-100000.json"),
    ("B6 m=10", _T5 / "546-baseline-m10-seed42-20000.json"),
    ("B1 m=3", _T4 / "537-baseline-seed42-100000.json"),
]

BAR = 2.0


def _rows(path: Path):
    return json.loads(path.read_text())["rows"] if path.exists() else []


def construction() -> None:
    print("1. #540's stack, rebuilt at each rung (seed 42)")
    print(
        f"  {'rung':>12} {'built ER':>9} {'p90':>7} {'max':>7} {'generic':>8} "
        f"{'modal widths':>28} {'k_v sat':>8} {'cap p90':>8} {'at cap':>7}"
    )
    for r in _rows(CONSTRUCTION):
        e, g, c = r["composed_er"], r["generic"], r["cap"]
        print(
            f"  {r['rung']:>12} {e['median']:>9.4f} {e['p90']:>7.4f} {e['max']:>7.4f} "
            f"{g['median']:>8.4f} {str(r['widths']['modal']):>28} "
            f"{r['mask']['k_v_saturated_cells']:>4}/{r['mask']['predicting_cells']:<3} "
            f"{c['ratio_p90']:>8.4f} {c['cells_at_cap']:>3}/{c['cells_measured']:<3}"
        )
    print("\n   incoherence and alignment across the rungs (#555 item 3)")
    print(f"  {'rung':>12} {'cos lead':>9} {'cos 2nd':>9} {'cos all':>9} {'s2/s1':>8} {'sigma':>9}")
    for r in _rows(CONSTRUCTION):
        print(
            f"  {r['rung']:>12} {r['cos_leading_per_hop']['median']:>9.4f} "
            f"{r['cos_second_per_hop']['median']:>9.4f} {r['cos_all']['median']:>9.4f} "
            f"{r['s2_over_s1']['median']:>8.4f} {r['sigma_min_over_max']['median']:>9.4f}"
        )


def departure() -> None:
    print("\n2. where the built reading departs from #540's generic prediction")
    print(
        f"  {'rung':>12} {'generic':>9} {'real V':>9} {'real all':>9} {'truth':>9} "
        f"{'sigma m/M':>10} {'sigma cv':>9}"
    )
    for r in _rows(DEPARTURE):
        lad, s = r["ladder"], r["sigma"]
        print(
            f"  {r['rung']:>12} {lad['generic']['median']:>9.4f} {lad['real_V']['median']:>9.4f} "
            f"{lad['real_all']['median']:>9.4f} {lad['truth']['median']:>9.4f} "
            f"{s['sigma_min_over_max_median']:>10.4f} {s['sigma_cv_median']:>9.4f}"
        )
    print("   real V ~ generic says the carried subspaces ARE generic: #540's law stands.")
    print("   real all << real V says the whole gap is singular-value spread, which")
    print("   #540's instrument set to 1 and which ADR-0032's band closes under training.")


def trained() -> None:
    print("\n3. the trained trajectory, against the record's nearest arms")
    for label, path in TRAINED:
        if not path.exists():
            print(f"\n  {label}: [missing] {path.name}")
            continue
        d = json.loads(path.read_text())
        c = d["at_construction"]["angles"]
        base = c["composed_er"]["median"] - 1.0
        print(
            f"\n  {label}  construction ER {c['composed_er']['median']:.6f} "
            f"(excess {base:.4e})"
        )
        print(
            f"    {'ticks':>7} {'ER med':>10} {'excess':>11} {'erosion':>9} {'max':>8} "
            f"{'lead':>7} {'2nd':>7} {'gap':>7} {'sigma':>7}"
        )
        for cp in d["checkpoints"]:
            a = cp["angles"]
            ex = a["composed_er"]["median"] - 1.0
            lead = a["cos_leading_per_hop"]["median"]
            sec = a["cos_second_per_hop"]["median"]
            sig = cp.get("sigma", {}).get("sigma_min_over_max_median")
            print(
                f"    {cp['ticks']:>7} {a['composed_er']['median']:>10.6f} {ex:>11.3e} "
                f"x{base / max(ex, 1e-300):>8.1f} {a['composed_er']['max']:>8.4f} "
                f"{lead:>7.4f} {sec:>7.4f} {lead - sec:>7.4f} "
                f"{'-' if sig is None else format(sig, '.4f'):>7}"
            )


def cost() -> None:
    print(f"\n4. what the stack costs, and whether it clears the bar ({BAR})")
    print(
        f"  {'rung':>12} {'sum_m med':>10} {'sum_m max':>10} {'priv dim 0':>11} "
        f"{'priv dim tot':>13} {'violations':>11}"
    )
    for r in _rows(CONSTRUCTION):
        inv = r["invariant"]
        print(
            f"  {r['rung']:>12} {inv['sum_m_median']:>10.0f} {inv['sum_m_max']:>10.0f} "
            f"{inv['private_dim_zero_cells']:>4}/{inv['predicting_cells']:<6} "
            f"{inv['private_dim_total']:>13} {inv['violations']:>11}"
        )
    print("\n   `dim H^0 >= sum_v max(0, n - sum_e m_e)` is the floor 06-graph-topology.md")
    print("   and 05-timescales.md read through. Private dim is 0 wherever sum_e m_e >= n.")

    for label, path in TRAINED[:2]:
        if not path.exists():
            continue
        d = json.loads(path.read_text())
        last = d["checkpoints"][-1]["angles"]["composed_er"]
        print(
            f"\n   {label} at {d['checkpoints'][-1]['ticks']}: median "
            f"{last['median']:.4f}, p90 {last['p90']:.4f}, max {last['max']:.4f} "
            f"-- bar {BAR}: {'CLEARS' if last['median'] >= BAR else 'FAILS'} "
            f"(median is {BAR / max(last['median'], 1e-12):.1f}x short)"
        )


def main() -> None:
    construction()
    departure()
    trained()
    cost()


if __name__ == "__main__":
    main()
