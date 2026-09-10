"""B68 (#645): what the stall stamp actually looks like per rule mode.

The two stamping rules disagree by four orders on the same run -- B38's
`last window above MOVING` puts the `transport` arm live to 15,790 while its
first fall under the threshold is at 80 -- and neither number describes the
run. The frozen arm dies once at 160 and stays dead, 0 re-crossings. The
rule-carrying arms **re-excite the body**: 70 re-crossings under
`PredictionRule`, 131 under `TransportRule`. So past the first fall the world
is not frozen and it is not moving; it is intermittent, and a single scalar
stamp cannot say which rungs a dynamical reading may be quoted at.

This reports the shape instead: the share of windows above `MOVING`, and the
same share within each interval between the checkpoint rungs the readings are
taken at, so a rung can be judged on the motion in the stretch that produced it
rather than on a stamp taken from the whole run.

Usage::

    python prototypes/cold-start/T6/b68_stall_table.py
"""

from __future__ import annotations

import io
import json
from pathlib import Path

_HERE = Path(__file__).resolve().parent

#: The rungs every reading on this ticket is taken at.
RUNGS = (0, 100, 200, 500, 1_000, 2_000, 5_000, 10_000, 20_000)


def main() -> None:
    path = _HERE / "645-stall.json"
    if not path.exists():
        print("[B68] 645-stall.json not present yet")
        return
    record = json.load(io.open(path, encoding="utf-8"))
    moving = record["moving_threshold"]
    print(f"threshold MOVING = {moving:g}, window {record['window']} ticks")
    print(f"surface {record['surface'].get('describe', '?')}")

    print(
        f"\n=== the two stamping rules, and what sits between them ===\n"
        f"{'mode':>11}{'last_above':>12}{'first_fall':>12}{'recross':>9}"
        f"{'share above':>13}{'peak':>11}{'final':>11}"
    )
    for row in record["rows"]:
        w = row["windows"]
        share = sum(1 for r in w if r["std_max"] >= moving) / len(w)
        print(
            f"{row['mode']:>11}{row['t_last_above']:>12}{row['t_first_fall']:>12}"
            f"{row['n_re_crossings']:>9}{share:>13.4f}"
            f"{row['std_max_peak']:>11.3e}{row['std_max_final']:>11.3e}"
        )

    print("\n=== share of windows above MOVING, per interval between rungs ===")
    print("    a rung is judged on the stretch that produced it, not on one scalar")
    header = "".join(f"{f'{lo}-{hi}':>12}" for lo, hi in zip(RUNGS, RUNGS[1:]))
    print(f"{'mode':>11}{header}")
    for row in record["rows"]:
        w = row["windows"]
        cells = []
        for lo, hi in zip(RUNGS, RUNGS[1:]):
            seg = [r for r in w if lo < r["tick"] <= hi]
            cells.append(
                f"{sum(1 for r in seg if r['std_max'] >= moving) / len(seg):>12.4f}"
                if seg
                else f"{'-':>12}"
            )
        print(f"{row['mode']:>11}" + "".join(cells))


if __name__ == "__main__":
    main()
