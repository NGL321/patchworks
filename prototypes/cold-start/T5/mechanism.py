"""T5 (#546) item 3: do the per-hop angle statistics move differently at wider lanes?

[B1](https://github.com/NGL321/patchworks/issues/537) found training *raises* the
leading principal-angle cosine (0.568 -> 0.79/0.81) while composed rank falls,
and [B3](https://github.com/NGL321/patchworks/issues/538) found training *removes*
cross-edge alignment where ADR-0022 says it makes it. #546 asks whether those
same statistics move differently once the lanes are wider -- which is where the
mechanism would show itself.

Prints, per width, the leading and second per-hop cosine and the whole cosine
distribution's median, at construction and at the horizon each arm reached.

Usage::

    python prototypes/cold-start/T5/mechanism.py
"""

from __future__ import annotations

import json
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_T4 = _HERE.parent / "T4"

ARMS = [
    (3, _T4 / "537-baseline-seed42-100000.json"),
    (6, _HERE / "546-baseline-m6-seed42-20000.json"),
    (10, _HERE / "546-baseline-m10-seed42-20000.json"),
    (14, _HERE / "546-baseline-m14-seed42-100000.json"),
]

FIELDS = ("cos_leading_per_hop", "cos_second_per_hop", "cos_all")


def main() -> None:
    print("per-hop principal-angle cosines: construction -> horizon, by rebuilt width")
    print(f"  {'m':>3} {'ticks':>7} {'lead':>18} {'second':>18} {'all':>18} {'ER med':>10}")
    for m, path in ARMS:
        if not path.exists():
            print(f"  m={m:>2}: [missing] {path.name}")
            continue
        d = json.loads(path.read_text())
        stages = [("construction", d["at_construction"]["angles"])]
        stages.append((str(d["checkpoints"][-1]["ticks"]), d["checkpoints"][-1]["angles"]))
        for label, a in stages:
            cells = "".join(f"{a[f]['median']:>18.4f}" for f in FIELDS)
            print(f"  {m:>3} {label:>7}{cells}{a['composed_er']['median']:>10.4f}")

    print("\nthe same, as the change training makes")
    print(f"  {'m':>3} {'d lead':>10} {'d second':>10} {'d all':>10}")
    for m, path in ARMS:
        if not path.exists():
            continue
        d = json.loads(path.read_text())
        c = d["at_construction"]["angles"]
        t = d["checkpoints"][-1]["angles"]
        cells = "".join(f"{t[f]['median'] - c[f]['median']:>+10.4f}" for f in FIELDS)
        print(f"  {m:>3}{cells}")

    # The gap between the leading cosine and the second is what a *rank* reading
    # is sensitive to: #537 found rank 1.000 is domination rather than
    # annihilation, so what training has to do to collapse the composite is pull
    # the leading direction away from the rest. How much room it has to do that
    # in is set at construction, and that is what moves with width.
    print("\nthe leading-to-second gap, and the room training has to open it")
    print(f"  {'m':>3} {'gap(0)':>10} {'gap(T)':>10} {'opened':>10} {'ER(0)':>9} {'ER(T)':>9}")
    for m, path in ARMS:
        if not path.exists():
            continue
        d = json.loads(path.read_text())
        c = d["at_construction"]["angles"]
        t = d["checkpoints"][-1]["angles"]
        g0 = c["cos_leading_per_hop"]["median"] - c["cos_second_per_hop"]["median"]
        gt = t["cos_leading_per_hop"]["median"] - t["cos_second_per_hop"]["median"]
        print(
            f"  {m:>3} {g0:>10.4f} {gt:>10.4f} {gt - g0:>+10.4f} "
            f"{c['composed_er']['median']:>9.4f} {t['composed_er']['median']:>9.4f}"
        )


if __name__ == "__main__":
    main()
