"""B62 (#635): the four arms side by side, `A(theta)` on every row.

`frozen` / `bias` / `transport` are this ticket's; `both` is B57's own baseline
arm on the same surface and the same seed, which is the control item 2 is read
against rather than re-run.

`A` is printed beside `N` everywhere per this readout's section 2a: `N` is a
participation ratio *inside* the counted band and reads 1.0000 on a direction
carrying 0.4% of the traffic, so the count without the weight is not a reading.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent

ARMS = (
    ("629-baseline-seed42-20000.json", "both (B57)"),
    ("635-frozen-baseline-seed42-5000.json", "frozen"),
    ("635-bias-baseline-seed42-5000.json", "bias"),
    ("635-transport-baseline-seed42-5000.json", "transport"),
)
RUNGS = (0, 100, 200, 500, 1000, 2000, 5000)


def load(path: Path) -> dict:
    d = json.load(io.open(path, encoding="utf-8"))
    rows = {0: d["at_construction"]}
    for c in d["checkpoints"]:
        rows[c["ticks"]] = c
    return rows


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    data = {label: load(_HERE / name) for name, label in ARMS}

    print("uncentered traffic ER | A(0.25) | N(0.25) | q of leading direction")
    print()
    head = f"{'rung':>6} | " + " | ".join(f"{l:^33}" for _, l in ARMS)
    print(head)
    print(f"{'':>6} | " + " | ".join(f"{'ER     A(.25)  N(.25)  q_top':^33}" for _ in ARMS))
    print("-" * len(head))
    for t in RUNGS:
        cells = []
        for _, label in ARMS:
            r = data[label].get(t)
            if r is None:
                cells.append(f"{'--':^33}")
                continue
            a = r["agreement"]
            p = a["profile"]["0.25"]
            cells.append(
                f"{a['traffic_effective_rank']:7.4f} {p['A']:7.4f} {p['N']:7.4f} {a['q_of_leading']:7.4f}"
            )
        print(f"{t:>6} | " + " | ".join(cells))

    print()
    print("exposure (effective median, of 32) / audience differentiation (median)")
    print(f"{'rung':>6} | " + " | ".join(f"{l:^16}" for _, l in ARMS))
    for t in RUNGS:
        cells = []
        for _, label in ARMS:
            r = data[label].get(t)
            cells.append(
                f"{'--':^16}"
                if r is None
                else f"{r['exposure']['effective_median']:5.2f} / {r['audience_differentiation']['median']:.4f}"
            )
        print(f"{t:>6} | " + " | ".join(cells))


if __name__ == "__main__":
    main()
