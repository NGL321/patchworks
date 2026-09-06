"""T5 (#546): the erosion table -- fixed fraction of the excess, or fixed amount?

Reads the trained ladders at each lane width and prints, per checkpoint, the
excess of median composed rim-to-apex ER over one and the **erosion factor**
`excess(construction) / excess(t)`.

The two hypotheses #546 pre-registers separate cleanly on this table:

* **Multiplicative** -- training removes a fixed *fraction* of the excess. The
  erosion factor is then the same at every width, and the endpoint excess scales
  with the construction excess.
* **Saturating** -- training spends a fixed *amount*, or lands on a floor. The
  *difference* `excess(construction) - excess(t)` is then the same at every
  width, or the endpoint excess is, and the factor grows with width.

A third column reports both differences so the reader is not asked to take the
ratio's word for it.

Usage::

    python prototypes/cold-start/T5/analyse.py
"""

from __future__ import annotations

import json
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_T4 = _HERE.parent / "T4"

ARMS = [
    (3, _T4 / "537-baseline-seed42-20000.json"),
    (3, _T4 / "537-baseline-seed42-100000.json"),
    (6, _HERE / "546-baseline-m6-seed42-20000.json"),
    (10, _HERE / "546-baseline-m10-seed42-20000.json"),
    (14, _HERE / "546-baseline-m14-seed42-20000.json"),
    (14, _HERE / "546-baseline-m14-seed42-100000.json"),
]

#: The replication arms, kept apart so a seed is never silently averaged into a
#: width's ladder. #546's verdict rests on the erosion factor moving with width,
#: so the second seed is reported beside the first rather than folded into it.
REPLICATIONS = [
    (10, 43, _HERE / "546-baseline-m10-seed43-20000.json"),
    (14, 43, _HERE / "546-baseline-m14-seed43-20000.json"),
]


def ladder(path: Path) -> tuple[float, list[tuple[int, float]], list[tuple[int, float]]]:
    d = json.loads(path.read_text())
    base = d["at_construction"]["angles"]["composed_er"]["median"] - 1.0
    rows = [(cp["ticks"], cp["angles"]["composed_er"]["median"] - 1.0) for cp in d["checkpoints"]]
    c1 = [
        (cp["ticks"], next(r for r in cp["sweep_c"] if r["c"] == 1)["er_median"] - 1.0)
        for cp in d["checkpoints"]
        if "sweep_c" in cp
    ]
    return base, rows, c1


def main() -> None:
    seen: dict[int, tuple[float, dict[int, float], dict[int, float]]] = {}
    for m, path in ARMS:
        if not path.exists():
            print(f"[missing] m={m}: {path.name}")
            continue
        base, rows, c1 = ladder(path)
        cur = seen.setdefault(m, (base, {}, {}))
        cur[1].update(dict(rows))
        cur[2].update(dict(c1))

    widths = sorted(seen)
    print("construction excess over one, by rebuilt interior lane width")
    for m in widths:
        print(f"  m={m:>2}: {seen[m][0]:.4e}")

    ticks = sorted({t for m in widths for t in seen[m][1]})
    print("\nexcess(t), and erosion factor excess(0)/excess(t)")
    head = "  ticks  " + "".join(f"{'m=' + str(m):>26}" for m in widths)
    print(head)
    for t in ticks:
        cells = []
        for m in widths:
            base, rows, _ = seen[m]
            if t in rows:
                e = rows[t]
                cells.append(f"{e:>12.3e} x{base / max(e, 1e-300):>10.1f}")
            else:
                cells.append(f"{'-':>26}")
        print(f"  {t:>6}  " + "".join(cells))

    print("\namount removed, excess(0) - excess(t)")
    for t in ticks:
        cells = []
        for m in widths:
            base, rows, _ = seen[m]
            cells.append(f"{base - rows[t]:>16.4e}" if t in rows else f"{'-':>16}")
        print(f"  {t:>6}  " + "".join(cells))

    print("\nc = 1 excess (ADR-0010's GAUGE_C tightened), by width")
    for t in ticks:
        cells = []
        for m in widths:
            _, _, c1 = seen[m]
            cells.append(f"{c1[t]:>16.3e}" if t in c1 else f"{'-':>16}")
        print(f"  {t:>6}  " + "".join(cells))

    print("\nreplication seeds, at the horizon each reached")
    for m, seed, path in REPLICATIONS:
        if not path.exists():
            print(f"  m={m:>2} seed {seed}: [missing] {path.name}")
            continue
        base, rows, _ = ladder(path)
        t, e = rows[-1]
        print(
            f"  m={m:>2} seed {seed}: construction {base:.4e} -> @{t} {e:.4e} "
            f"erosion x{base / max(e, 1e-300):.1f} (retains {100 * e / base:.2f}%)"
        )


if __name__ == "__main__":
    main()
