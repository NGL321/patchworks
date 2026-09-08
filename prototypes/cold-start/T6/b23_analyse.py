"""T6 (#572 / B23): read `572-dome-*.json` — does level predict anything?

Rank correlation (Spearman) of each structural variable against each measured
one, plus per-level medians, at every checkpoint of every arm.

Spearman is computed here rather than imported: the rig has no scipy dependency
and the statistic is a Pearson correlation of ranks. Ties get average ranks. The
two-sided p-value uses the `t = r*sqrt((N-2)/(1-r^2))` approximation, which is
adequate at `N = 150` and is reported to two figures only.

Usage::

    python prototypes/cold-start/T6/b23_analyse.py
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent

STRUCTURAL = ("level", "hops_to_sensory_rim", "fan_in", "p_v", "k_v", "sum_m", "degree")
MEASURED = (
    ("tau_e_v", ("autocorr_v", "tau_e")),
    ("tau_int_v", ("autocorr_v", "tau_int")),
    ("lag1_v", ("autocorr_v", "lag1")),
    ("tau_e_h", ("autocorr_h", "tau_e")),
    ("tau_e_v_private", ("autocorr_v_private", "tau_e")),
    ("tau_e_v_exposed", ("autocorr_v_exposed", "tau_e")),
    ("rho_used", ("rho_used",)),
    ("rho_raw", ("rho_raw",)),
    ("tau_spectral", ("tau_spectral",)),
    ("pr_total_centred", ("pr_total_centred",)),
    ("pr_total", ("pr_total",)),
    ("pr_private_centred", ("pr_private_centred",)),
    ("pr_exposed_centred", ("pr_exposed_centred",)),
    ("h_pr_centred", ("h_pr_centred",)),
    ("mean_share", ("mean_share",)),
    ("v_rms", ("v_rms",)),
    ("e_rms", ("e_rms",)),
    ("emission_gain", ("emission_gain",)),
    ("decode_proprio", ("decode_proprio_h_lag0",)),
    ("decode_proprio_null", ("decode_proprio_h_null",)),
    ("decode_touch", ("decode_touch_h_lag0",)),
    ("decode_touch_null", ("decode_touch_h_null",)),
    ("decode_puck", ("decode_puck_h_lag0",)),
    ("decode_puck_null", ("decode_puck_h_null",)),
    ("decode_proprio_lag50", ("decode_proprio_h_lag50",)),
)


def _world_motion(entry: dict) -> str:
    """One line saying whether the world moved at all over this window.

    Every decode figure is read against this: at zero target variance an `R^2`
    is a fit to nothing, and the readout may not quote one.
    """
    parts = []
    for name in ("proprio", "touch", "puck"):
        key = f"world_std_{name}"
        if key in entry:
            parts.append(f"{name} {max(entry[key]):.4g}")
    return "world motion (max std over window): " + ", ".join(parts) if parts else ""


def _dig(entry: dict, path: tuple[str, ...]):
    cur = entry
    for key in path:
        cur = cur[key]
    return cur


def _rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    # average ties
    sx = x[order]
    i = 0
    while i < len(sx):
        j = i
        while j + 1 < len(sx) and sx[j + 1] == sx[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = np.mean(np.arange(i, j + 1))
        i = j + 1
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> tuple[float, float, int]:
    live = ~(np.isnan(a) | np.isnan(b))
    a, b = a[live], b[live]
    n = len(a)
    if n < 4 or np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan"), float("nan"), n
    ra, rb = _rank(a), _rank(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = math.sqrt(float((ra * ra).sum()) * float((rb * rb).sum()))
    r = float((ra * rb).sum() / denom) if denom > 0 else float("nan")
    if not math.isfinite(r) or abs(r) >= 1.0:
        return r, 0.0, n
    t = r * math.sqrt((n - 2) / (1 - r * r))
    # two-sided normal approximation to the t tail; n = 150 makes it adequate
    p = math.erfc(abs(t) / math.sqrt(2))
    return r, p, n


def _column(record: dict, name: str) -> np.ndarray:
    return np.array([row[name] for row in record["structure"]["cells"]], dtype=float)


def _measured(entry: dict, path: tuple[str, ...]) -> np.ndarray:
    vals = _dig(entry, path)
    return np.array([np.nan if v is None else float(v) for v in vals], dtype=float)


def report(path: Path, checkpoints: tuple[int, ...] | None) -> None:
    record = json.loads(path.read_text())
    # Join the corrected depth axis from the structure-only pass, for runs taken
    # before `hops_to_sensory_rim` existed. It is a construction quantity, so a
    # later pass on the same arm and seed reads the same graph.
    if "hops_to_sensory_rim" not in record["structure"]["cells"][0]:
        side = _HERE / f"572-structure-{record['arm']}.json"
        if side.exists():
            by_id = {r["cell_id"]: r for r in json.loads(side.read_text())["cells"]}
            for row in record["structure"]["cells"]:
                row["hops_to_sensory_rim"] = by_id[row["cell_id"]]["hops_to_sensory_rim"]
    levels = _column(record, "level")
    print(f"\n{'=' * 78}\n{path.name}   arm={record['arm']} seed={record['seed']}")
    print(f"  {len(levels):.0f} predicting cells, levels {int(levels.min())}-{int(levels.max())}")

    struct = {name: _column(record, name) for name in STRUCTURAL}
    print("\n  Structure by level (median):")
    header = "    level  n  " + "  ".join(f"{s:>11}" for s in STRUCTURAL[1:])
    print(header)
    for lv in sorted(set(levels.tolist())):
        sel = levels == lv
        row = "  ".join(f"{np.median(struct[s][sel]):>11.1f}" for s in STRUCTURAL[1:])
        print(f"    L{int(lv)}  {int(sel.sum()):>3}  {row}")

    for entry in record["checkpoints"]:
        if checkpoints and entry["ticks"] not in checkpoints:
            continue
        print(f"\n  --- {entry['ticks']} ticks (window {entry['window_ticks']}) ---")
        motion = _world_motion(entry)
        if motion:
            print(f"    {motion}")
        print(f"    {'measured':>20} | " + " | ".join(f"{s:>14}" for s in STRUCTURAL))
        for name, path_ in MEASURED:
            try:
                vals = _measured(entry, path_)
            except KeyError:
                continue
            cells = []
            for s in STRUCTURAL:
                r, p, n = spearman(struct[s], vals)
                mark = "*" if (p == p and p < 0.01) else " "
                cells.append(f"{r:>+8.3f}{mark}p{p:>4.2f}" if r == r else f"{'nan':>14}")
            print(f"    {name:>20} | " + " | ".join(cells))

        print(f"\n    {'measured':>20} | " + " | ".join(f"{'L' + str(int(lv)):>8}" for lv in sorted(set(levels.tolist()))))
        for name, path_ in MEASURED:
            try:
                vals = _measured(entry, path_)
            except KeyError:
                continue
            cells = []
            for lv in sorted(set(levels.tolist())):
                sel = vals[levels == lv]
                sel = sel[~np.isnan(sel)]
                cells.append(f"{np.median(sel):>8.3g}" if sel.size else f"{'-':>8}")
            print(f"    {name:>20} | " + " | ".join(cells))


HEADLINE = (
    ("tau_e_v", ("autocorr_v", "tau_e")),
    ("tau_int_v", ("autocorr_v", "tau_int")),
    ("lag1_v", ("autocorr_v", "lag1")),
    ("tau_spectral", ("tau_spectral",)),
    ("pr_total_centred", ("pr_total_centred",)),
    ("h_pr_centred", ("h_pr_centred",)),
    ("mean_share", ("mean_share",)),
    ("emission_gain", ("emission_gain",)),
    ("decode_proprio", ("decode_proprio_h_lag0",)),
    ("decode_touch", ("decode_touch_h_lag0",)),
)


def summary(files: list[Path], ticks: int) -> None:
    """One table: `Spearman(level, X)` at `ticks`, every arm and seed side by side.

    The ticket's first question in one place. A column is one run; a row is one
    thing a cell might have that the dome's wager says depth should predict.
    """
    runs = []
    for path in files:
        record = json.loads(path.read_text())
        entry = next((c for c in record["checkpoints"] if c["ticks"] == ticks), None)
        if entry is None:
            continue
        runs.append((f"{record['arm'][:4]}/s{record['seed']}", record, entry))
    if not runs:
        raise SystemExit(f"no run has a {ticks}-tick checkpoint")

    print(f"\nSpearman(level, X) at {ticks} ticks — 150 predicting cells, levels 1-7")
    print("  a dome that forces abstraction predicts a strong positive on the first six rows\n")
    print(f"  {'measured':>18} | " + " | ".join(f"{name:>12}" for name, _, _ in runs))
    print(f"  {'-' * 18}-+-" + "-+-".join("-" * 12 for _ in runs))
    for name, path_ in HEADLINE:
        cells = []
        for _, record, entry in runs:
            levels = _column(record, "level")
            try:
                vals = _measured(entry, path_)
            except KeyError:
                cells.append(f"{'-':>12}")
                continue
            r, p, n = spearman(levels, vals)
            if r != r:
                cells.append(f"{'nan':>12}")
            else:
                cells.append(f"{r:>+9.3f}{'*' if p < 0.01 else ' '}  ")
        print(f"  {name:>18} | " + " | ".join(cells))
    print("\n  * two-sided p < 0.01. World motion over each window:")
    for label, _, entry in runs:
        print(f"    {label:>12}  {_world_motion(entry)}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", type=int, default=None, metavar="TICKS")
    p.add_argument("--glob", default="572-dome-*.json")
    p.add_argument("--checkpoints", type=int, nargs="*", default=None)
    args = p.parse_args()
    files = [
        f
        for f in sorted(_HERE.glob(args.glob))
        if ".inflight" not in f.name and ".killed" not in f.name
    ]
    if not files:
        raise SystemExit(f"no files matching {args.glob}")
    if args.summary:
        summary(files, args.summary)
        return
    for f in files:
        if ".inflight" in f.name or ".killed" in f.name:
            continue
        report(f, tuple(args.checkpoints) if args.checkpoints else None)


if __name__ == "__main__":
    main()
