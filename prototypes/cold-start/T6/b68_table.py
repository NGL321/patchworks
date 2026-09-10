"""B68 (#645): the four arms side by side, rung for rung.

Reads whatever records are present and prints one table per arm plus a joined
headline table at the shared rungs. Every row carries the joint columns B48's
standing constraint asks for -- agreement `N(theta)` beside `A(theta)`, audience
differentiation, and the exposure it cost -- and both forms of the traffic's
effective rank, per B62's window rule (uncentered for magnitude, centred for
direction at a stated window).

Usage::

    python prototypes/cold-start/T6/b68_table.py
"""

from __future__ import annotations

import io
import json
from pathlib import Path

_HERE = Path(__file__).resolve().parent

#: label -> (path, what ran). The both-on arm is B57's own baseline record,
#: which is what B62 quoted; it is not re-run here.
ARMS = {
    "frozen": ("635-frozen-baseline-seed42-20000.json", "neither rule"),
    "bias": ("645-bias-baseline-seed42-20000.json", "PredictionRule only"),
    "transport": ("645-transport-baseline-seed42-20000.json", "TransportRule only"),
    "both": ("629-baseline-seed42-20000.json", "both rules (B57's baseline)"),
    "bias@5k": ("635-bias-baseline-seed42-5000.json", "PredictionRule only, B62's run"),
    "transport@5k": ("635-transport-baseline-seed42-5000.json", "TransportRule only, B62's run"),
}


def load(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.load(io.open(path, encoding="utf-8"))


def ticks_of(entry: dict) -> int:
    """`ticks` where the rig wrote one, else parsed off the label."""
    if entry.get("ticks") is not None:
        return int(entry["ticks"])
    label = entry.get("label", "")
    if "@construction" in label:
        return 0
    return int(label.rsplit("@", 1)[1])


def profile_at(agreement: dict, theta: str) -> tuple[float, float]:
    """`(N, A)` at `theta`. Never `N` alone -- B62's ledger row 11."""
    node = agreement.get("profile", {}).get(theta)
    if node is None:
        return float("nan"), float("nan")
    return float(node["N"]), float(node["A"])


def row(entry: dict) -> dict:
    a, c = entry["agreement"], entry["agreement_centred"]
    n25, a25 = profile_at(a, "0.25")
    return {
        "ticks": ticks_of(entry),
        "er": a["traffic_effective_rank"],
        "er_centred": c["traffic_effective_rank"],
        "q_top": a["q_of_leading"],
        "q_wm": a["q_weighted_mean"],
        "n25": n25,
        "a25": a25,
        "auddiff": entry["audience_differentiation"]["median"],
        "exposure": entry["exposure"]["effective_median"],
        "window": entry.get("window_ticks"),
    }


def rows(record: dict) -> list[dict]:
    entries = [record["at_construction"], *record["checkpoints"]]
    return [row(e) for e in entries]


HEAD = (
    f"{'ticks':>7} {'ER unc':>8} {'ER cen':>8} {'q_top':>7} {'q_wm':>7} "
    f"{'N(.25)':>7} {'A(.25)':>8} {'aud-diff':>9} {'exposure':>9}"
)


def line(r: dict) -> str:
    return (
        f"{r['ticks']:>7} {r['er']:>8.4f} {r['er_centred']:>8.4f} {r['q_top']:>7.4f} "
        f"{r['q_wm']:>7.4f} {r['n25']:>7.4f} {r['a25']:>8.4f} {r['auddiff']:>9.4f} "
        f"{r['exposure']:>9.2f}"
    )


def main() -> None:
    loaded = {}
    for name, (fname, what) in ARMS.items():
        record = load(_HERE / fname)
        if record is None:
            print(f"[B68] {name}: {fname} not present yet")
            continue
        loaded[name] = record
        surface = record.get("surface", {})
        print(f"\n=== {name} -- {what} ===")
        print(f"    surface {surface.get('describe', '?')}")
        print(f"    window {record.get('window')}  buffer {record.get('buffer')}")
        print(HEAD)
        for r in rows(record):
            print(line(r))

    shared = {}
    for name in ("frozen", "bias", "transport", "both"):
        if name in loaded:
            shared[name] = {r["ticks"]: r for r in rows(loaded[name])}
    if len(shared) < 2:
        return
    rungs = sorted(set.intersection(*(set(v) for v in shared.values())))
    print("\n=== joined at shared rungs: uncentered ER / q_top / aud-diff ===")
    print(f"{'ticks':>7} " + " ".join(f"{n:>26}" for n in shared))
    for t in rungs:
        cells = []
        for n in shared:
            r = shared[n][t]
            cells.append(f"{r['er']:8.4f} {r['q_top']:7.4f} {r['auddiff']:8.4f}")
        print(f"{t:>7} " + " ".join(f"{c:>26}" for c in cells))


if __name__ == "__main__":
    main()
