"""B43 (#607): is the bottleneck's rise anything but a shorter path?

Clause 3 is the conjunction's **anti-Goodhart** clause: ADR-0026's bar carries
loop length and ADR-0021's does not, so an intervention that moves conduction
while the bottleneck sits still has shortened the measuring stick and nothing
else. That reasoning assumes the two are independent. They are not, for a
*topological* intervention:

* ADR-0026's divisor is a **path length**, and a relay shortens it.
* ADR-0021's bottleneck is a `max` over paths of a `min` over that path's edges,
  and a shorter path takes the `min` over fewer edges. [B21](#570) read 20–50x of
  gain per hop, so dropping hops raises the `min` by orders **whether or not
  anything conducts better**.

So both clauses are decreasing functions of the same quantity and the conjunction
cannot, by construction, tell *conducts better* from *has a shorter path*. This
prints the evidence: the hop count of the widest path each trial actually found,
beside the amplitude it carried. A length-matched null — the amplitude B21's
per-hop gain predicts from the hop count alone — is what a future clause would
have to beat.

Usage::

    python prototypes/cold-start/T6/b43_hops.py
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

#: B21's per-hop gain, the low and high end of the range it read. Used only to
#: state what a hop count alone predicts; nothing is fitted.
PER_HOP = (20.0, 50.0)


def main() -> None:
    rows = []
    for path in sorted(_HERE.glob("607-detect-*.json")):
        r = json.loads(path.read_text(encoding="utf-8"))
        for t in r["trials_out"]:
            rows.append(
                (
                    f"{r['layout']}{r['sites']}",
                    r["seed"],
                    t["kind"],
                    t["path_hops"],
                    t["bottleneck"],
                    r["world_loop_cohort_median"],
                )
            )
    if not rows:
        print("[missing] no 607-detect-*.json records")
        return
    base = [r for r in rows if r[0] == "none0"]
    base_hops = float(np.median([r[3] for r in base])) if base else float("nan")
    base_bn = float(np.median([r[4] for r in base])) if base else float("nan")

    print(
        f"{'arm':<12}{'seed':>5}{'kind':>16}{'hops':>6}{'bottleneck':>13}"
        f"{'wl_cohort':>10} | {'hops_saved':>10}{'predicted_x':>22}{'actual_x':>11}"
    )
    print("-" * 108)
    for arm, seed, kind, hops, bn, wl in rows:
        saved = base_hops - hops
        lo, hi = PER_HOP[0] ** saved, PER_HOP[1] ** saved
        actual = bn / base_bn if base_bn else float("nan")
        print(
            f"{arm:<12}{seed:>5}{kind:>16}{hops:>6}{bn:>13.3e}{wl:>10.1f} | "
            f"{saved:>10.1f}{lo:>10.2e} .. {hi:>8.2e}{actual:>11.2e}"
        )
    print(
        f"\nbaseline: median {base_hops:.1f} hops, median bottleneck {base_bn:.3e}"
        f"  (per-hop gain range from B21: {PER_HOP[0]:.0f}-{PER_HOP[1]:.0f}x)"
    )
    print(
        "`predicted_x` is what the hop count alone buys at B21's per-hop gain; a "
        "relay earns its clause 3 only where `actual_x` clears it."
    )


if __name__ == "__main__":
    main()
