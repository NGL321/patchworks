"""B57 (#629): the tables. Reads the records `b57_agreement.py` writes.

Nothing is computed here that is not already in the records -- this arranges
them. Every table carries B48's three columns together, because the standing
constraint is that agreement never travels alone.

Usage::

    python prototypes/cold-start/T6/b57_analyse.py 629-accept-baseline-seed42.json
    python prototypes/cold-start/T6/b57_analyse.py 629-baseline-seed42-20000.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

THETAS = ("0.05", "0.1", "0.25")


def _n(row: dict, theta: str) -> float:
    return row["profile"][theta]["N"]


def _a(row: dict, theta: str) -> float:
    return row["profile"][theta]["A"]


def profile_line(tag: str, a: dict) -> str:
    cells = " ".join(
        f"{_a(a, t):.4f}/{_n(a, t):.3f}" for t in THETAS
    )
    return (
        f"{tag:<26} ER {a['traffic_effective_rank']:8.4f} | A/N {cells} "
        f"| q_w {a['q_weighted_mean']:.4f} | q_med {a['q_quantiles']['median']:.4f}"
    )


def full_profile(tag: str, a: dict) -> str:
    out = [f"  {tag}: theta -> A (N)"]
    for theta, v in a["profile"].items():
        out.append(f"    {theta:>6}  A {v['A']:.6f}  N {v['N']:.4f}  dirs {v['directions']}")
    return "\n".join(out)


def joint_line(tag: str, row: dict) -> str:
    a, ad, ex = row["agreement"], row["audience_differentiation"], row["exposure"]
    cells = " ".join(f"{_n(a, t):.3f}" for t in THETAS)
    return (
        f"{tag:<22} N {cells} | q_w {a['q_weighted_mean']:.4f} "
        f"| aud-diff med {ad['median']:.4f} | exposure {ex['effective_median']:.2f}"
        f"/{ex['rank_median']:.0f} of {ex['n']}"
    )


def show_accept(r: dict) -> None:
    print(f"== B57 acceptance check, {r['condition']} seed {r['seed']}, window {r['window']} ==")
    print(f"   surface {r['surface']['describe']}")
    print(f"   delta check: {r['delta_check']}")
    held = r["held"]["agreement"]
    print()
    print(profile_line("held configuration", held))
    for name, a in r["scrambles"].items():
        print(profile_line(f"scrambled/{name}", a))
    print()
    print(full_profile("held", held))
    for name, a in r["scrambles"].items():
        print(full_profile(f"scrambled/{name}", a))
    print()
    print("-- window stability (same held configuration, frozen maps) --")
    for w, a in r["window_stability"].items():
        print(profile_line(f"T = {w}", a))
    print()
    print("-- matched-generic null on the same traffic --")
    print(f"   generic q_weighted mean {r['generic']['q_weighted_mean']:.4f} "
          f"over {r['generic']['draws']} draws")
    for theta in THETAS:
        print(f"   theta {theta:>5}: N {_n(held, theta):.4f}  N_gen "
              f"{r['generic']['N_mean'][theta]:.4f}  excess "
              f"{_n(held, theta) - r['generic']['N_mean'][theta]:+.4f}")


def show_trained(r: dict) -> None:
    print(f"== B57 trained, {r['condition']} seed {r['seed']}, window {r['window']} ==")
    print(f"   surface {r['surface']['describe']}")
    print(f"   delta check: {r['delta_check']}")
    print()
    rows = [("construction", r["at_construction"])] + [
        (str(c["ticks"]), c) for c in r["checkpoints"]
    ]
    for tag, row in rows:
        print(joint_line(tag, row))
    print()
    print("-- excess over matched generic, same traffic --")
    for tag, row in rows:
        a = row["agreement"]
        g = row["generic"]["N_mean"]
        cells = " ".join(
            f"{t}: {_n(a, t):.3f}-{g[t]:.3f}={_n(a, t) - g[t]:+.3f}" for t in THETAS
        )
        print(f"{tag:<14} {cells}")
    print()
    print("-- the acceptance check on every row: held vs traffic scrambled at frozen maps --")
    for tag, row in rows:
        a = row["agreement"]
        for name, sc in row.get("scrambles", {}).items():
            print(
                f"{tag:<14} {name:<10} ER {a['traffic_effective_rank']:7.3f}->"
                f"{sc['traffic_effective_rank']:8.3f} | q_w {a['q_weighted_mean']:.4f}->"
                f"{sc['q_weighted_mean']:.4f} | "
                + " ".join(
                    f"N@{t} {_n(a, t):.3f}->{_n(sc, t):.3f}" for t in THETAS
                )
                + f" | A@1 {_a(a, '1'):.4f}->{_a(sc, '1'):.4f}"
                + f" N@1 {_n(a, '1'):.3f}->{_n(sc, '1'):.3f}"
            )
    print()
    print("-- the profile at the horizon --")
    print(full_profile(rows[-1][0], rows[-1][1]["agreement"]))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("records", type=Path, nargs="+")
    args = p.parse_args()
    for path in args.records:
        r = json.loads(path.read_text())
        if r["stage"] == "accept":
            show_accept(r)
        else:
            show_trained(r)
        print()


if __name__ == "__main__":
    main()
