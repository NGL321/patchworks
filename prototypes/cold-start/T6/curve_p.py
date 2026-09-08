"""T6 (#560/B13) item 2: assemble the trained `p` curve beside the construction one.

The whole point of this ticket is that these two curves are **different curves**.
[B11](#555) found that a construction reading is the wrong number to rule on —
ADR-0032's band is a training-time projection, all three of its arms departed
from their construction values within 100 ticks, and two of them *inverted*. So
`sweep_p.py` says where to look and this says what is actually there.

Reads:

* `560-sweep-p-construction.json` — the construction curve, both seeds.
* `560-reserve_p<N>-baseline-seed42-20000.json` — this ticket's trained ladder.
* `555-reserve-baseline-seed42-{20000,100000}.json` — [B11](#555)'s `p = 8` arm,
  reused rather than re-run: its construction figures match this rig's
  `reserve_p8` exactly (ER 1.3824568, generic 2.2652, floor 1200), so it is the
  same object under an older label.

Usage::

    python prototypes/cold-start/T6/curve_p.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def trained_rows(ticks: int) -> dict[int, dict]:
    """`p` -> the trained record at `ticks`, from either ticket's files."""
    rows: dict[int, dict] = {}
    for path in sorted(_HERE.glob(f"560-reserve_p*-baseline-seed42-{ticks}.json")):
        d = load(path)
        if d:
            rows[int(d["reserve_p"])] = d
    b11 = load(_HERE / f"555-reserve-baseline-seed42-{ticks}.json")
    if b11:
        rows.setdefault(int(b11["reserve_p"]), b11)
    return rows


def summarise(d: dict) -> dict:
    base = d["at_construction"]["angles"]["composed_er"]["median"] - 1.0
    last = d["checkpoints"][-1]
    er = last["angles"]["composed_er"]
    excess = er["median"] - 1.0
    return {
        "p": d["reserve_p"],
        "ticks": last["ticks"],
        "construction_er": base + 1.0,
        "trained_er": er["median"],
        "trained_p90": er["p90"],
        "trained_max": er["max"],
        "erosion": base / max(excess, 1e-300),
        "floor": d["privacy"]["private_dim_total"],
        "k_v": d["privacy"]["k_v_median"],
        "generic": d["at_construction"]["generic"]["median"],
        "cos_lead_construction": (
            d["at_construction"]["angles"]["cos_leading_per_hop"]["median"]
        ),
        "cos_lead_trained": last["angles"]["cos_leading_per_hop"]["median"],
        "gap_construction": (
            d["at_construction"]["angles"]["cos_leading_per_hop"]["median"]
            - d["at_construction"]["angles"]["cos_second_per_hop"]["median"]
        ),
        "cap_p90": last["cap"]["ratio_p90"],
        "cells_at_cap": last["cap"]["cells_at_cap"],
        "sigma": last["sigma"]["sigma_min_over_max_median"],
        "minutes": last.get("elapsed_minutes"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ticks", type=int, default=20_000)
    ap.add_argument("--out", type=Path, default=_HERE / "560-curve-p.json")
    args = ap.parse_args()

    construction = load(_HERE / "560-sweep-p-construction.json")
    by_p: dict[int, list[float]] = {}
    if construction:
        for row in construction["rows"]:
            by_p.setdefault(row["p"], []).append(row["composed_er"]["median"])

    trained = {p: summarise(d) for p, d in trained_rows(args.ticks).items()}
    long = {p: summarise(d) for p, d in trained_rows(100_000).items()}

    print(f"  {'p':>3} {'k_v':>4} {'floor':>6} {'gap':>7} {'constr':>8} "
          f"{'seeds':>15} {'trained':>8} {'erosion':>8} {'p90':>7} {'lead':>6}")
    for p in sorted(set(by_p) | set(trained)):
        seeds = by_p.get(p, [])
        t = trained.get(p)
        seed_txt = " ".join(f"{v:.3f}" for v in seeds)
        if t:
            print(
                f"  {p:>3} {t['k_v']:>4.0f} {t['floor']:>6} "
                f"{t['gap_construction']:>7.4f} {t['construction_er']:>8.4f} "
                f"{seed_txt:>15} {t['trained_er']:>8.4f} "
                f"x{t['erosion']:>7.2f} {t['trained_p90']:>7.4f} "
                f"{t['cos_lead_trained']:>6.3f}"
            )
        else:
            mean = sum(seeds) / len(seeds)
            print(f"  {p:>3} {'':>4} {150 * p:>6} {'':>7} {mean:>8.4f} {seed_txt:>15}")

    record = {
        "issue": 560,
        "reading": "construction vs trained composed rank across the reserve size p",
        "construction_by_p": by_p,
        "trained": trained,
        "trained_100k": long,
    }
    args.out.write_text(json.dumps(record, indent=1))
    print(f"wrote {args.out.name}")


if __name__ == "__main__":
    main()
