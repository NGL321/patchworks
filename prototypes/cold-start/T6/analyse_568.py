"""T6 (#568 / B19): the tables the readout quotes, off the raw JSON.

Nothing is computed here that the run did not already record — this only selects,
orders and prints, so the readout and the raw files cannot drift.

Usage::

    python prototypes/cold-start/T6/analyse_568.py
"""

from __future__ import annotations

import json
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def load(pattern: str) -> list[dict]:
    """Completed runs only: an `.inflight.json` is a partial checkpoint ladder."""
    return [
        json.loads(p.read_text())
        for p in sorted(_HERE.glob(pattern))
        if not p.name.endswith(".inflight.json")
    ]


def med(entry: dict, key: str) -> float:
    q = entry["quantiles"].get(key)
    return float("nan") if q is None else q["median"]


def p90(entry: dict, key: str) -> float:
    q = entry["quantiles"].get(key)
    return float("nan") if q is None else q.get("p90", float("nan"))


def trajectory(record: dict) -> None:
    print(
        f"\n=== {record['arm']}  (p = {record['reserve_p']}, seed {record['seed']}, "
        f"{record['predicting_cells']} predicting cells, window {record['window']})"
    )
    print(
        "  ticks |  win |  state  priv  expo  read | emit  emitI | "
        "e/s   r/s   e/r  | meanshare"
    )
    for e in record["checkpoints"]:
        print(
            f"  {e['ticks']:>6} | {e['window_ticks']:>4} | "
            f"{med(e, 'pr_total_centred'):5.3f} "
            f"{med(e, 'pr_private_centred'):5.3f} "
            f"{med(e, 'pr_exposed_centred'):5.3f} "
            f"{med(e, 'pr_readable_centred'):5.3f} | "
            f"{med(e, 'pr_emitted_centred'):5.3f} "
            f"{med(e, 'pr_emitted_interior_centred'):5.3f} | "
            f"{med(e, 'ratio_emitted_over_state_centred'):5.3f} "
            f"{med(e, 'ratio_readable_over_state_centred'):5.3f} "
            f"{med(e, 'ratio_emitted_over_readable_centred'):5.3f} | "
            f"{med(e, 'mean_share_state'):5.3f}"
        )


def uncentred(record: dict) -> None:
    print(f"\n--- {record['arm']}: uncentred beside centred (B18's correction)")
    print("  ticks |  state_unc  state_cen |  emit_unc  emit_cen | mean_share")
    for e in record["checkpoints"]:
        print(
            f"  {e['ticks']:>6} |    {med(e, 'pr_total'):6.3f}    {med(e, 'pr_total_centred'):6.3f} "
            f"|   {med(e, 'pr_emitted'):6.3f}   {med(e, 'pr_emitted_centred'):6.3f} "
            f"|   {med(e, 'mean_share_state'):.4f}"
        )


def nulls(record: dict) -> None:
    if not record.get("null"):
        return
    print(f"\n--- {record['arm']}: learned against shape/mask/band-matched random maps")
    print("  ticks | learned |  haar  haar_scaled | learned/haar  learned/scaled")
    for e in record["checkpoints"]:
        print(
            f"  {e['ticks']:>6} |  {med(e, 'pr_emitted_centred'):6.3f} | "
            f"{med(e, 'pr_emitted_haar_centred'):6.3f} "
            f"{med(e, 'pr_emitted_haar_scaled_centred'):11.3f} | "
            f"{med(e, 'ratio_learned_over_haar_centred'):11.3f} "
            f"{med(e, 'ratio_learned_over_haar_scaled_centred'):15.3f}"
        )


def blocks_table(record: dict) -> None:
    print(f"\n--- {record['arm']}: where the state lives (energy / variance shares)")
    print("  ticks | E priv  E int | V priv  V int")
    for e in record["checkpoints"]:
        print(
            f"  {e['ticks']:>6} | {med(e, 'energy_share_private'):6.3f} "
            f"{med(e, 'energy_share_interior'):6.3f} | "
            f"{med(e, 'variance_share_private'):6.3f} "
            f"{med(e, 'variance_share_interior'):6.3f}"
        )


def grouped(record: dict, axis: str) -> None:
    last = record["checkpoints"][-1]
    print(f"\n--- {record['arm']} @ {last['ticks']}: by {axis}")
    print(f"  {axis:>14} | cells |  state   emit  |  e/s    e/r")
    for key, row in last["by"][axis].items():
        print(
            f"  {key:>14} | {row['cells']:>5} | "
            f"{row['pr_total_centred']:6.3f} {row['pr_emitted_centred']:6.3f} | "
            f"{row['ratio_emitted_over_state_centred']:6.3f} "
            f"{row['ratio_emitted_over_readable_centred']:6.3f}"
        )


def spread(record: dict) -> None:
    last = record["checkpoints"][-1]
    print(f"\n--- {record['arm']} @ {last['ticks']}: the distribution, not the median")
    for key in (
        "pr_total_centred",
        "pr_readable_centred",
        "pr_emitted_centred",
        "ratio_emitted_over_state_centred",
    ):
        q = last["quantiles"][key]
        print(
            f"  {key:>34}: min {q['min']:6.3f}  p10 {q['p10']:6.3f}  "
            f"med {q['median']:6.3f}  p90 {q['p90']:6.3f}  max {q['max']:6.3f}"
        )


def correlations(record: dict) -> None:
    """Is emitted rank tracking state rank, or is it flat against it?

    The median hides the shape of the answer: if emission were faithful, a cell
    with twice the state rank would emit twice as much. Pearson `r` across cells,
    plus what emitted rank does over the state-rank quartiles.
    """
    import statistics

    last = record["checkpoints"][-1]
    rows = last["cells"]
    st = [r["pr_total_centred"] for r in rows]
    em = [r["pr_emitted_centred"] for r in rows]

    def pearson(a, b):
        ma, mb = statistics.fmean(a), statistics.fmean(b)
        num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
        da = sum((x - ma) ** 2 for x in a) ** 0.5
        db = sum((y - mb) ** 2 for y in b) ** 0.5
        return num / (da * db) if da and db else float("nan")

    order = sorted(range(len(rows)), key=lambda i: st[i])
    q = len(order) // 4
    has_null = "pr_emitted_haar_centred" in rows[0]
    print(f"\n--- {record['arm']} @ {last['ticks']}: does emission track the state?")
    print(f"  corr(state, emitted)          r = {pearson(st, em):+.3f}")
    print(
        f"  corr(state, emitted/state)    r = "
        f"{pearson(st, [r['ratio_emitted_over_state_centred'] for r in rows]):+.3f}"
    )
    if has_null:
        # The control that says whether flatness is learned or is the shape.
        for name in ("haar", "haar_scaled"):
            col = [r[f"pr_emitted_{name}_centred"] for r in rows]
            print(f"  corr(state, {name:<11}) r = {pearson(st, col):+.3f}")
    header = "  state-rank quartile |  state   emit"
    if has_null:
        header += "    haar  scaled"
    print(header)
    for qi in range(4):
        idx = order[qi * q : (qi + 1) * q] if qi < 3 else order[3 * q :]
        line = (
            f"                   Q{qi + 1} | "
            f"{statistics.median([st[i] for i in idx]):6.3f} "
            f"{statistics.median([em[i] for i in idx]):6.3f}"
        )
        if has_null:
            for name in ("haar", "haar_scaled"):
                col = [rows[i][f"pr_emitted_{name}_centred"] for i in idx]
                line += f"  {statistics.median(col):6.3f}"
        print(line)


def main() -> None:
    records = load("568-state-emitted-*.json") + load("568-null-*.json")
    if not records:
        print("no 568-*.json yet")
        return
    for r in records:
        trajectory(r)
    for r in records:
        uncentred(r)
        nulls(r)
        blocks_table(r)
        for axis in ("level", "degree", "sum_interior_m"):
            grouped(r, axis)
        spread(r)
        correlations(r)


if __name__ == "__main__":
    main()
