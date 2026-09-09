"""B40 (#603): the arm comparison as one table, scored on the ticket's four scores.

Reads whatever `b40_routes.py` has written and prints the canonical table, so the
readout quotes a printed artifact rather than numbers retyped by hand.

The four scores, in the ticket's order:

1. **Does cycle closure happen** -- `identification` on the wide subset. 0 is
   perfect closure, 1 is chance. B29 read the shipped surface at chance on every
   cycle with room, so anything near 1 is the status quo, not a result.
2. **Is it honest** -- `sigma_max` and `channel_return` unchanged or better.
3. **Does the criterion discriminate** -- the spread of the per-edge warranted
   count. **A route that closes every cycle equally has no carving pressure and
   fails this while passing (1).** Reported on the edges that have evidence,
   because on the 90 that have none the count is floored by arithmetic and
   mixing the two hides whether the criterion carves anything.
4. **Do probe lanes leak** -- transmission across the floored edges against the
   kept ones. Above 1 means the reversibility floor is a live channel.

`world` is each checkpoint's own `motion.read()`, per B38's standing requirement.
A row whose world is dead is not a reading about learning, and the column is
printed rather than a flag so that judgement stays with the reader.

Usage::

    PYTHONPATH=src python prototypes/cold-start/T6/b40_table.py
"""

from __future__ import annotations

import json
from pathlib import Path

_HERE = Path(__file__).resolve().parent

THRESHOLD = "0.9"


def rows_for(path: Path):
    d = json.loads(path.read_text(encoding="utf-8"))
    for cp in d["checkpoints"]:
        sub = cp["holonomy"]["subsets"]["wide"]
        cr = cp["criterion"][THRESHOLD]
        sp = cr["allocated_with_evidence"]
        pl = cp.get("probe_leak", {})
        mo = cp.get("motion", {})
        yield {
            "arm": d["arm"],
            "seed": d["seed"],
            "ticks": cp["ticks"],
            "ident": sub["identification"]["median"],
            "chan": sub["channel_return"]["median"],
            "sigma": sub["sigma_max"]["median"],
            "m_med": sp.get("median", float("nan")),
            "m_max": sp.get("max", float("nan")),
            "m_std": sp.get("std", float("nan")),
            "m_distinct": sp.get("distinct", 0),
            "m_total": sp.get("total", 0),
            "leak": pl.get("ratio_median", float("nan")),
            "leak_share": pl.get("floored_above_kept_median", float("nan")),
            "world": mo.get("std_max", float("nan")) if mo.get("available") else float("nan"),
            "phase1_over": cp.get("phase1_over"),
        }


def main() -> None:
    paths = sorted(_HERE.glob("603-routes-*-seed*-*.json"))
    paths = [p for p in paths if ".inflight" not in p.name]
    if not paths:
        print("no B40 runs written yet")
        return
    rows = []
    for p in paths:
        rows.extend(rows_for(p))
    order = {"flat": 0, "control": 1, "split": 2, "phased": 3}
    rows.sort(key=lambda r: (order.get(r["arm"], 9), r["seed"], r["ticks"]))

    head = (
        f"{'arm':<9}{'seed':>5}{'ticks':>7}{'ident':>9}{'chan':>8}{'sigma':>11}"
        f"{'m_med':>7}{'m_max':>7}{'m_std':>7}{'m_dist':>7}{'leak':>7}{'lk>':>6}"
        f"{'world':>10}"
    )
    print(head)
    print("-" * len(head))
    last = None
    for r in rows:
        if last is not None and r["arm"] != last:
            print("-" * len(head))
        last = r["arm"]
        print(
            f"{r['arm']:<9}{r['seed']:>5}{r['ticks']:>7}"
            f"{r['ident']:>9.4f}{r['chan']:>8.4f}{r['sigma']:>11.2e}"
            f"{r['m_med']:>7.1f}{r['m_max']:>7.0f}{r['m_std']:>7.2f}"
            f"{r['m_distinct']:>7}{r['leak']:>7.3f}{r['leak_share']:>6.2f}"
            f"{r['world']:>10.2e}"
        )
    print(
        "\nident: 0 = closure, 1 = chance (B29's shipped surface).  "
        "chan/sigma: score 2.\n"
        "m_*: warranted count on the edges that have evidence -- score 3, where a\n"
        "     std of 0 is the flat bundle's predicted signature (no carving pressure).\n"
        "leak: floored-lane gain / kept-lane gain; lk> is the share of floored edges\n"
        "      above the kept median. Score 4 -- above 1 means the floor transmits.\n"
        "world: this checkpoint's own motion.read(); nothing about it is inherited."
    )


if __name__ == "__main__":
    main()
