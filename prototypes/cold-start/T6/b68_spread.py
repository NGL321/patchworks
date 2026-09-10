"""B68 (#645): the replicate spread, so a difference can be told from noise.

Stage 0 found that the arms running `TransportRule` do not reproduce -- two
*identical* invocations inside one process part by |dER| 2.5e-01 by tick 200.
Item 3's verdict is a claim about a gap of 0.0536 in audience differentiation,
and item 1's is a claim about a separation of ~0.2 in effective rank, so
neither may be asserted until the spread on the arm that carries it is
measured rather than borrowed.

This prints, per rung, the absolute difference between an arm and its
replicate -- same seed, same command, separate process -- beside the effect the
readout wants to quote from that arm. Rungs where the replicate is bit-identical
are the strongest ground there is: the effect at those rungs is not a sample.

Usage::

    python prototypes/cold-start/T6/b68_spread.py
"""

from __future__ import annotations

import io
import json
from pathlib import Path

_HERE = Path(__file__).resolve().parent

#: (label, run, replicate). Both halves are `--no-generic` runs of the same
#: command at the same seed, in separate processes.
PAIRS = (
    (
        "both",
        "645-both-baseline-seed42-20000.json",
        "645-both-rep2-baseline-seed42-20000.json",
    ),
    (
        "transport",
        "645-transport-baseline-seed42-20000.json",
        "645-transport-rep2-baseline-seed42-20000.json",
    ),
)

COLUMNS = (
    ("er", "uncentered ER"),
    ("q_top", "q_top"),
    ("auddiff", "aud-diff"),
    ("exposure", "exposure"),
)


def rows(path: Path) -> dict[int, dict]:
    record = json.load(io.open(path, encoding="utf-8"))
    out = {}
    for e in [record["at_construction"], *record["checkpoints"]]:
        a = e["agreement"]
        out[int(e["ticks"])] = {
            "er": a["traffic_effective_rank"],
            "q_top": a["q_of_leading"],
            "auddiff": e["audience_differentiation"]["median"],
            "exposure": e["exposure"]["effective_median"],
        }
    return out


def main() -> None:
    for label, first, second in PAIRS:
        p1, p2 = _HERE / first, _HERE / second
        if not (p1.exists() and p2.exists()):
            print(f"[B68] {label}: replicate pair not complete yet")
            continue
        a, b = rows(p1), rows(p2)
        print(f"\n=== {label}: |run - replicate|, same seed, separate processes ===")
        print(f"{'ticks':>7}" + "".join(f"{name:>16}" for _, name in COLUMNS))
        for t in sorted(set(a) & set(b)):
            cells = []
            for key, _ in COLUMNS:
                d = abs(a[t][key] - b[t][key])
                cells.append("        exact" if d == 0.0 else f"{d:>16.2e}")
            print(f"{t:>7}" + "".join(f"{c:>16}" for c in cells))


if __name__ == "__main__":
    main()
