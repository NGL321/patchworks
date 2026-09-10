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
    "both": ("645-both-baseline-seed42-20000.json", "both rules, this surface"),
    "both-rep2": ("645-both-rep2-baseline-seed42-20000.json", "both rules, replicate"),
    "transport-rep2": (
        "645-transport-rep2-baseline-seed42-20000.json",
        "TransportRule only, replicate",
    ),
    "both@B57": ("629-baseline-seed42-20000.json", "both rules, B57's run on c925866"),
    "bias@5k": ("635-bias-baseline-seed42-5000.json", "PredictionRule only, B62's run"),
    "transport@5k": ("635-transport-baseline-seed42-5000.json", "TransportRule only, B62's run"),
}

#: Arms that enter the joined table -- one surface, one run each. The replicates
#: and the inherited records are printed above it but never joined into it: a
#: hull argument built out of two surfaces is exactly what #455's rule forbids.
JOINED = ("frozen", "bias", "transport", "both")


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
    for name in JOINED:
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

    if not {"bias", "transport", "both"} <= set(shared):
        return
    # The hull needs the three rule arms and not the frozen one, so its rungs are
    # theirs. B62's committed frozen record stops at 10,000 -- its 20,000 rung was
    # printed to `635-frozen-20k.log` (ER 2.7825, aud-diff 0.5307, exposure 9.05)
    # but the file was committed before the run's last write landed -- and taking
    # the intersection with it would silently drop the horizon rung from a
    # comparison the frozen arm does not enter.
    rungs = sorted(
        set(shared["bias"]) & set(shared["transport"]) & set(shared["both"])
    )
    print("\n=== item 3: is the pair inside the interval its solos span? ===")
    print(
        "    `lo`/`hi` are the two solo arms at that rung; `both` is the pair. "
        "`out` is how far\n    outside the interval the pair lies, 0 when it is "
        "inside. Reported per column,\n    never fused -- B49's struck move.\n"
    )
    for column, label in (("auddiff", "audience differentiation"), ("exposure", "exposure"), ("er", "uncentered ER")):
        print(f"  -- {label} --")
        print(f"{'ticks':>7}{'bias':>10}{'transport':>11}{'both':>10}{'out':>10}")
        for t in rungs:
            lo = min(shared["bias"][t][column], shared["transport"][t][column])
            hi = max(shared["bias"][t][column], shared["transport"][t][column])
            v = shared["both"][t][column]
            out = 0.0 if lo <= v <= hi else (v - hi if v > hi else v - lo)
            print(
                f"{t:>7}{shared['bias'][t][column]:>10.4f}"
                f"{shared['transport'][t][column]:>11.4f}{v:>10.4f}{out:>+10.4f}"
            )
        print()


if __name__ == "__main__":
    main()
