"""T6 ([B22](#571)): read `b22_regions.py`'s JSON into the tables the readout quotes.

Kept separate from the instrument for the reason the rig keeps every other
analyser separate: the runs are long and the tables get re-cut several times
while the readout is written, and re-running a 20k arm to change a column is not
a thing anyone should have to do.

Usage::

    python prototypes/cold-start/T6/b22_analyse.py 571-regions-construction.json
    python prototypes/cold-start/T6/b22_analyse.py 571-regions-*.json --trained
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# The tables carry `H⁰`, `Σ` and `−`, and this box's console is cp1252. Without
# this the analyser dies on its own first heading.
sys.stdout.reconfigure(encoding="utf-8")

_HERE = Path(__file__).resolve().parent


def _split_row(arm: str, p, w: dict) -> str:
    return (
        f"| {arm} | {'-' if p is None else p} | {w['rows']} | {w['dim_h0']} | "
        f"{w['trivial']} | {w['earned']} | {w['earned_generic']} | "
        f"{w['earned_above_generic']:+d} | {w['dim_h1']} |"
    )


SPLIT_HEAD = (
    "| arm | p | rows (Σm_e) | dim H⁰ | trivial | earned | generic | earned−generic | dim H¹ |\n"
    "|---|---|---|---|---|---|---|---|---|"
)


def _region_row(arm: str, p, read: dict) -> str:
    b, lat, lad = read["balls"], read["lateral"], read["ladder"]
    m = read["matched_size"].get("ladder_minus_lateral", {})

    def s(pool_, key):
        return pool_.get("size", {}).get(key, float("nan"))

    return (
        f"| {arm} | {'-' if p is None else p} | {s(b, 'median'):.0f} | {s(b, 'max'):.0f} | "
        f"{b['levels_spanned']['median']:.0f} | {100 * b['fraction_censored']:.0f}% | "
        f"{s(lat, 'median'):.0f} | {s(lad, 'median'):.0f} | "
        f"{m.get('median', float('nan')):+.2f} | "
        f"{100 * m.get('fraction_zero', float('nan')):.0f}% |"
    )


REGION_HEAD = (
    "| arm | p | ball cells med | ball max | levels med | censored | lateral med | "
    "ladder med | ladder−lateral med | at-par |\n"
    "|---|---|---|---|---|---|---|---|---|---|"
)


def construction(path: Path) -> None:
    d = json.loads(path.read_text())
    print(f"# {path.name} -- {d['reading']}\n")
    print("## Item 1: dim H⁰ split, and the counting control\n")
    print(SPLIT_HEAD)
    for row in d["rows"]:
        w = row["reads"]["sealed"]["whole_graph"]
        print(_split_row(row["arm"], row["reserve_p"], w))
    for convention in ("internal", "sealed"):
        print(f"\n## Items 2-4: region sizes, `{convention}` rows\n")
        print(REGION_HEAD)
        for row in d["rows"]:
            print(_region_row(row["arm"], row["reserve_p"], row["reads"][convention]))
    print("\n## Ball size histograms (`sealed`)\n")
    for row in d["rows"]:
        h = row["reads"]["sealed"]["balls"]["size"]["histogram"]
        top = sorted(((int(k), v) for k, v in h.items()), key=lambda kv: -kv[1])[:6]
        print(f"- **{row['arm']}**: " + ", ".join(f"{v} dirs @ {k} cells" for k, v in top))


def trained(paths: list[Path]) -> None:
    for path in paths:
        d = json.loads(path.read_text())
        print(f"\n# {path.name} -- {d['arm']} p={d['reserve_p']} {d['condition']} "
              f"seed {d['seed']}, {d['ticks']} ticks\n")
        print("## The split over training\n")
        print("| ticks | dim H⁰ | trivial | earned | generic | earned−generic | dim H¹ |")
        print("|---|---|---|---|---|---|---|")
        w0 = d["at_construction"]["sealed"]["whole_graph"]
        print(
            f"| 0 (construction) | {w0['dim_h0']} | {w0['trivial']} | {w0['earned']} | "
            f"{w0['earned_generic']} | {w0['earned_above_generic']:+d} | {w0['dim_h1']} |"
        )
        for cp in d["checkpoints"]:
            w = cp["whole_graph"]
            print(
                f"| {cp['ticks']} | {w['dim_h0']} | {w['trivial']} | {w['earned']} | "
                f"{w['earned_generic']} | {w['earned_above_generic']:+d} | {w['dim_h1']} |"
            )
        for convention in ("internal", "sealed"):
            print(f"\n## Region sizes over training, `{convention}` rows\n")
            print(REGION_HEAD.replace("| arm | p |", "| ticks | - |"))
            print(_region_row("0 (construction)", None, d["at_construction"][convention]))
            for cp in d["checkpoints"]:
                if convention in cp["reads"]:
                    print(_region_row(str(cp["ticks"]), None, cp["reads"][convention]))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("paths", nargs="+", type=Path)
    p.add_argument("--trained", action="store_true")
    args = p.parse_args()
    paths = [q if q.exists() else _HERE / q.name for q in args.paths]
    if args.trained:
        trained(paths)
    else:
        for q in paths:
            construction(q)


if __name__ == "__main__":
    main()
