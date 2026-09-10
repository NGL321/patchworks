"""B58 (#630) §C: the staggered arms against B56's own, on one table.

Reads this ticket's staggered arms and
[B56 (#628)](https://github.com/NGL321/patchworks/issues/628)'s random-init arms out of
the same directory and puts them in the same columns, because the comparison #630 needs
is *initialisation against objective*: B56 bought `channel_return` by training a term,
and this ticket asks what the construction buys before any term runs.

Every column is B56's, so nothing here re-defines a statistic:

* `chan` -- `channel_return` median over the 45 `wide` cycles.
* `diff` -- audience differentiation, [B42 (#605)](https://github.com/NGL321/patchworks/issues/605)'s
  `1 - edge_overlap`, which [B48 (#615)](https://github.com/NGL321/patchworks/issues/615)'s
  joint rule requires beside any agreement figure.
* `k_v rank` / `k_v part` -- the exposure pair, `rank` being the clamped window
  [B22 (#571)](https://github.com/NGL321/patchworks/issues/571) found invariant and
  `part` its participation ratio, which is the one that moves.

Rungs past this run's own motion stamp are marked `*`, per
[B38 (#599)](https://github.com/NGL321/patchworks/issues/599); the stamp is read from
each file's own `motion` block and inherited from nothing.

Usage::

    PYTHONPATH=src python prototypes/cold-start/T6/b58_analyse.py
"""

from __future__ import annotations

import json
from pathlib import Path

_HERE = Path(__file__).resolve().parent

#: label -> (file, what it is). B56's two arms are the random-init controls.
FILES = {
    "B56 baseline (random init)": "628-channel-baseline-seed42-2000.json",
    "B56 channel (random init + clause 1)": "628-channel-channel-seed42-2000.json",
    "s0 baseline (flat bundle)": "630-stagger-s0_baseline-seed42-2000.json",
    "s19 baseline (staggered, no term)": "630-stagger-s19_baseline-seed42-2000.json",
    "s19 channel (staggered + clause 1)": "630-stagger-s19_channel-seed42-2000.json",
    "p8 s22 baseline (staggered, p=8)": "630-stagger-p8_s22_baseline-seed42-2000.json",
}


def horizon(record: dict) -> int | None:
    """The last rung at which this run's own body was still moving.

    B38's stamp, per run and never per arm: `std_max` collapsing by orders is the
    body going still, and rungs past it are drift under a frozen stimulus.
    """
    last = None
    for c in record["checkpoints"]:
        m = c.get("motion") or {}
        std = m.get("std_max")
        if std is None or std != std:
            continue
        if std > 1e-2:
            last = c["ticks"]
    return last


def rows(record: dict) -> list[dict]:
    out = []
    for c in record["checkpoints"]:
        w = c["holonomy_wide"]
        e = c["exposure"]
        out.append(
            {
                "ticks": c["ticks"],
                "chan": w["channel_return"]["median"],
                "ident": w["identification"]["median"],
                "diff": c["differentiation"]["audience_differentiation"],
                "kv_rank": e["k_v_rank_median"],
                "kv_part": e["k_v_participation_median"],
            }
        )
    return out


def main() -> None:
    table = {}
    for label, name in FILES.items():
        path = _HERE / name
        if not path.exists():
            print(f"  (missing: {name})")
            continue
        rec = json.loads(path.read_text(encoding="utf-8"))
        table[label] = {"horizon": horizon(rec), "rows": rows(rec), "file": name}

    print(
        f"{'arm':<38}{'tick':>7}{'chan':>9}{'ident':>8}{'diff':>8}"
        f"{'kv rank':>9}{'kv part':>9}"
    )
    for label, t in table.items():
        h = t["horizon"]
        for r in t["rows"]:
            if r["ticks"] not in (0, 100, 2000):
                continue
            star = "*" if h is not None and r["ticks"] > h else " "
            print(
                f"{label:<38}{str(r['ticks']) + star:>7}{r['chan']:>9.4f}"
                f"{r['ident']:>8.4f}{r['diff']:>8.4f}"
                f"{r['kv_rank']:>9.1f}{r['kv_part']:>9.1f}"
            )
        print()

    for label, t in table.items():
        print(f"motion stamp  {label:<38} last live rung {t['horizon']}")
    print()

    # B48's joint rule, stated against the arm each candidate must actually beat.
    #
    # Reported at **both** rungs deliberately. B56's new standing rule exempts a term
    # whose gradient never reads a stalk from the stall stamp -- but these arms are the
    # *shipped transport rule*, which does read stalks, so the exemption does not cover
    # them and the **live rung is the verdict**, per B33's practice. The 2,000 row is
    # carried because B50 showed the null is not static and the drift is the thing a
    # candidate has to beat.
    base = table.get("B56 baseline (random init)")
    if not base:
        return
    for rung in (100, 2000):
        b = next((r for r in base["rows"] if r["ticks"] == rung), None)
        if b is None:
            continue
        live = base["horizon"] is not None and rung <= base["horizon"]
        print(
            f"B48 joint rule at {rung}"
            f"{' (live)' if live else ' (past the stall stamp)'}"
            " -- agreement up, differentiation nonzero, exposure reported:"
        )
        for label, t in table.items():
            if label.startswith("B56 baseline"):
                continue
            r = next((x for x in t["rows"] if x["ticks"] == rung), None)
            if r is None:
                continue
            passes = r["chan"] > b["chan"] and r["diff"] > b["diff"]
            print(
                f"  {label:<38} d_chan {r['chan'] - b['chan']:+.4f}"
                f"   d_diff {r['diff'] - b['diff']:+.4f}"
                f"   d_kv_part {r['kv_part'] - b['kv_part']:+.2f}"
                f"   {'PASSES' if passes else 'fails'}"
            )
        print()
    out = _HERE / "630-analysis.json"
    out.write_text(
        json.dumps({"ticket": 630, "arms": table}, indent=1), encoding="utf-8"
    )
    print(f"\nwrote {out.name}")


if __name__ == "__main__":
    main()
