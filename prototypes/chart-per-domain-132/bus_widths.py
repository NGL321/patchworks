"""The third width: a cell's communication bus, both domains, side by side.

[#132](https://github.com/NGL321/patchworks/issues/132) §4 requires the three widths
to be held apart before anything is compared -- the chart `k`, the node stalk `n`,
and the **communication bus** `Sum_e m_e` ([#411](https://github.com/NGL321/patchworks/issues/411)'s
term, swept by [#414](https://github.com/NGL321/patchworks/issues/414)) -- and says
the wedge's own table already points at the bus as what is eating the language rim.

The wedge's side of that comparison is arithmetic and is already computed by
`prototypes/language-graph/wedge_counts.py`. The dome's side is not written down
anywhere per cell, so this reads it off the built graph, so that the two are
counted the same way rather than one being quoted from a spec and the other from a
build.

`dim H^0 >= max(0, n - Sum_e m_e)` per cell, `01-cell-and-sheaf.md`.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from patchworks.body import NODE_STALK_DIM
from patchworks.graph import DEFAULT_SPEC, build_graph

#: `11-the-language-graph.md`'s table, so the two domains print together.
WEDGE = [
    ("L1 (interior)", 7, 44),
    ("L1 (buffer end)", 6, 40),
    ("L2 (interior)", 7, 28),
    ("L2 (buffer end)", 6, 24),
    ("L3-L6 core", 6, 24),
    ("L7 apex (with drive)", 5, 17),
]


def dome_buses() -> dict:
    dome = build_graph(DEFAULT_SPEC)
    bus: dict[int, int] = defaultdict(int)
    degree: dict[int, int] = defaultdict(int)
    for edge in dome.edges:
        for cell_id in (edge.u, edge.v):
            bus[cell_id] += edge.m
            degree[cell_id] += 1
    return dome, bus, degree


def main() -> None:
    dome, bus, degree = dome_buses()
    n = NODE_STALK_DIM

    groups: dict[tuple, list[int]] = defaultdict(list)
    for cell in dome.cells:
        groups[(cell.index.level, cell.index.column, cell.kind.name)].append(cell.id)

    # **Ranges, not modes.** The vision lattice's edge cells carry fewer lateral
    # lanes than its interior ones, so a group has a spread of buses and a modal
    # value silently drops the narrow end -- which is exactly the end the
    # comparison with the wedge turns on.
    rows = []
    print(f"dome, n = {n}")
    print(
        f"{'group':34s} {'cells':>5s} {'deg':>7s} {'sum m_e':>9s} "
        f"{'dim H0 >=':>11s} {'zero':>5s}"
    )
    for key in sorted(groups):
        ids = groups[key]
        degs = sorted({degree[i] for i in ids})
        buses = sorted({bus[i] for i in ids})
        label = f"L{key[0]} {key[1]} ({key[2].lower()})"
        floors = sorted({max(0, n - bus[i]) for i in ids})
        zero = sum(1 for i in ids if n - bus[i] <= 0)
        rows.append(
            {
                "group": label,
                "cells": len(ids),
                "degree": [degs[0], degs[-1]],
                "bus": [buses[0], buses[-1]],
                "private_floor": [floors[0], floors[-1]],
                "zero_floor_cells": zero,
                "bus_over_n": [buses[0] / n, buses[-1] / n],
            }
        )
        span = lambda a: f"{a[0]}" if a[0] == a[-1] else f"{a[0]}-{a[-1]}"  # noqa: E731
        print(
            f"{label:34s} {len(ids):5d} {span(degs):>7s} {span(buses):>9s} "
            f"{span(floors):>11s} {zero:5d}"
        )

    # Corroboration against a number the map already owns: #385's structural
    # zero is *82 of 150* predicting cells, and this count must reproduce it or
    # the counting is wrong before either domain is compared.
    predicting_cells = [c for c in dome.cells if c.kind.name == "PREDICTING"]
    zero_total = sum(1 for c in predicting_cells if n - bus[c.id] <= 0)
    print()
    print(
        f"structural zero over predicting cells: {zero_total} of "
        f"{len(predicting_cells)}  (#385 records 82 of 150)"
    )

    print()
    print(f"language wedge, n = {n}  (11-the-language-graph.md)")
    print(f"{'group':34s} {'cells':>5s} {'deg':>5s} {'sum m_e':>9s} {'dim H0 >=':>10s}")
    wedge_rows = []
    for label, deg, sums in WEDGE:
        floor = max(0, n - sums)
        wedge_rows.append(
            {"group": label, "degree": deg, "bus": sums, "private_floor": floor,
             "bus_over_n": sums / n}
        )
        print(f"{label:34s} {'-':>5s} {deg:5d} {sums:9d} {floor:10d}")

    # The comparison #132 §4 actually asks for: which domain's *rim* carries the
    # wider bus. Taken over L1 predicting cells in both, since that is where the
    # wedge's table puts the pressure.
    dome_l1 = [r for r in rows if r["group"].startswith("L1")]
    lang_l1 = [r for r in wedge_rows if r["group"].startswith("L1")]
    dome_span = [min(r["bus"][0] for r in dome_l1), max(r["bus"][1] for r in dome_l1)]
    lang_span = [min(r["bus"] for r in lang_l1), max(r["bus"] for r in lang_l1)]

    print()
    print("the bus an L1 predicting cell carries")
    print(
        f"  dome     {dome_span[0]}-{dome_span[1]} "
        f"= {dome_span[0] / n:.2f}-{dome_span[1] / n:.2f} x n, "
        f"overflow {dome_span[0] - n}-{dome_span[1] - n}"
    )
    print(
        f"  language {lang_span[0]}-{lang_span[1]} "
        f"= {lang_span[0] / n:.2f}-{lang_span[1] / n:.2f} x n, "
        f"overflow {lang_span[0] - n}-{lang_span[1] - n}"
    )
    print(
        "  -> "
        + (
            "the dome's rim carries the wider bus"
            if dome_span[1] > lang_span[1]
            else "the language rim carries the wider bus"
        )
    )

    out = Path(__file__).parent / "132-bus-widths.json"
    out.write_text(
        json.dumps(
            {
                "n": n,
                "dome": rows,
                "language": wedge_rows,
                "structural_zero": [zero_total, len(predicting_cells)],
                "l1_bus": {"dome": dome_span, "language": lang_span},
            },
            indent=1,
        )
    )
    print(f"wrote {out.name}")


if __name__ == "__main__":
    main()
