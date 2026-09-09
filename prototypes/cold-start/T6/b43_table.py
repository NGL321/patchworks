"""B43 (#607): the tables, printed from the records rather than retyped.

Every number in `READOUT-607.md` comes out of here, in the house rule B40 states:
*nothing is retyped by hand*.

Usage::

    python prototypes/cold-start/T6/b43_table.py layouts
    python prototypes/cold-start/T6/b43_table.py width
    python prototypes/cold-start/T6/b43_table.py detect
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def _read(name: str):
    path = _HERE / name
    if not path.exists():
        print(f"[missing] {name}")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def layouts() -> None:
    d = _read("607-layouts.json")
    if not d:
        return
    print(
        f"{'layout':<10}{'sites':>6}{'+e':>4} | {'wl_med':>7}{'wl_max':>7}"
        f"{'apex_med':>9}{'apex_max':>9} | {'no_local':>9}{'bridge':>7}"
        f" | {'over_B':>7}{'reuse':>6}{'load':>6}"
    )
    print("-" * 96)
    for r in d["layouts"]:
        w, b, bu, rc = r["clause1_world"], r["bridges"], r["budget"], r["relay_cycles"]
        lr = r["layout_record"]
        print(
            f"{r['layout']:<10}{r.get('sites', 0):>6}{r['edges_added']:>4} | "
            f"{w['world_loop_median']:>7.1f}{w['world_loop_max']:>7.1f}"
            f"{w['world_loop_deepest_median']:>9.1f}{w['world_loop_deepest_max']:>9.1f} | "
            f"{rc.get('relays_with_no_local_cycle', 0):>9}"
            f"{b['relay_edges_that_are_bridges']:>7} | "
            f"{bu['cells_over_budget']:>7}{lr.get('anchor_reuse_max', 0):>6}"
            f"{r['relay_load'].get('max_load', 0):>6.0f}"
        )


def width() -> None:
    d = _read("607-width-seed42.json")
    if not d:
        return
    print(
        f"{'arm':<13}{'tick':>5}{'th':>6} | {'memb_med':>9}{'max':>5}{'std':>7}"
        f"{'distinct':>9}{'=deg':>6} | {'flr_med':>8}{'max':>5}{'std':>7}"
        f"{'distinct':>9}{'=deg':>6}"
    )
    print("-" * 100)
    for a in d["arms"]:
        arm = f"{a['layout']}{a['sites']}"
        for ck in a["checkpoints"]:
            for th in ("0.3", "0.5", "0.9"):
                m = ck["membership"][th]["membership"]
                e = ck["membership"][th]["equals_degree_cells"]
                f = ck["membership_floored"][th]["membership"]
                fe = ck["membership_floored"][th]["equals_degree_cells"]
                print(
                    f"{arm:<13}{ck['ticks']:>5}{th:>6} | "
                    f"{m['median']:>9.1f}{m['max']:>5.0f}{m['std']:>7.2f}"
                    f"{m['distinct']:>9}{e:>6} | "
                    f"{f['median']:>8.1f}{f['max']:>5.0f}{f['std']:>7.2f}"
                    f"{f['distinct']:>9}{fe:>6}"
                )
        s = a["checkpoints"][-1]["strict"]
        sc = s.get("strict_criterion", {})
        print(
            f"  strict {arm}: mask median {s['strict_mask']['median']:.1f} "
            f"std {s['strict_mask']['std']:.2f} distinct {s['strict_mask']['distinct']} "
            f"| = degree at {s['strict_mask_equals_degree_cells']}/{s['cells']} "
            f"| degree median {s['degree']['median']:.1f} "
            f"| criterion sum median {sc.get('median', float('nan')):.1f} "
            f"= degree at {s.get('strict_criterion_equals_degree_cells')}"
            f"/{s.get('strict_criterion_cells')}"
        )


def detect() -> None:
    print(
        f"{'arm':<12}{'seed':>5}{'relays':>7}{'learn':>7} | "
        f"{'bottleneck_med':>15}{'max':>11}"
        f" | {'tau_med':>8}{'ratio_med':>10}{'ratio_max':>10}{'>=1':>5}"
        f" | {'wl_cohort':>10}{'motion':>10}"
    )
    print("-" * 106)
    for path in sorted(_HERE.glob("607-detect-*.json")):
        r = json.loads(path.read_text(encoding="utf-8"))
        t = r["trials_out"]
        if not t:
            continue
        pc = t[0]["per_cell"]
        import numpy as np

        tau = float(np.median([e["per_cell"]["tau_median"] for e in t]))
        rmed = float(np.median([e["per_cell"]["ratio_median"] for e in t]))
        rmax = float(max(e["per_cell"]["ratio_max"] for e in t))
        rge1 = int(np.median([e["per_cell"]["ratio_above_one"] for e in t]))
        print(
            f"{r['layout'] + str(r['sites']):<12}{r['seed']:>5}{r['relays']:>7}"
            f"{r['learn']:>7} | "
            f"{r.get('bottleneck_median', float('nan')):>15.3e}"
            f"{r.get('bottleneck_max', float('nan')):>11.3e} | "
            f"{tau:>8.2f}{rmed:>10.3f}{rmax:>10.3f}{rge1:>5} | "
            f"{r['world_loop_cohort_median']:>10.1f}"
            f"{r['motion'].get('std_max', float('nan')):>10.2e}"
        )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["layouts", "width", "detect", "all"])
    args = p.parse_args()
    if args.command in ("layouts", "all"):
        layouts()
        print()
    if args.command in ("width", "all"):
        width()
        print()
    if args.command in ("detect", "all"):
        detect()


if __name__ == "__main__":
    main()
