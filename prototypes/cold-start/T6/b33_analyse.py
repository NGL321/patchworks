"""B33 (#592): read the five arms against the ticket's own question.

The question is **not** *does flatness rise*. It is *does flatness rise on
cycles carrying more than one direction, with the floor holding* — so every
table here is taken on `surface_read`'s `wide` subset (base width >= 2) and the
floor is printed beside it, never after it.

Three columns decide it, and the third is the one B32 added:

* `identification` on wide cycles — 0 is flat, 1 is chance. Does the term move it?
* `pr_emitted_centred` against `pr_emitted_haar_centred` — B19's pairing. Does
  the floor hold, or does the arm slide back to the learned maps' degeneracy
  while the holonomy term is being satisfied?
* `sigma_max` on wide cycles — **the steganography check**. Chu et al. (2017)
  is the published way to satisfy a cycle-consistency term: hide the signal in a
  channel the loop preserves and the task never reads. A term satisfied while
  `sigma_max` collapses is a loop that closes on nothing, and it must not be
  scored as a win.

Usage::

    python prototypes/cold-start/T6/b33_analyse.py
"""

from __future__ import annotations

import json
from pathlib import Path

_HERE = Path(__file__).resolve().parent

ARM_ORDER = ["baseline", "holo", "floor", "floor_dec", "floor_si", "both", "both_si"]

WHAT = {
    "baseline": "the transport rule as shipped",
    "holo": "local holonomy term alone",
    "floor": "absolute variance floor alone (unreachable under ADR-0032's band)",
    "floor_dec": "absolute floor + VICReg decorrelation",
    "floor_si": "scale-free floor: participation ratio of the emitted covariance",
    "both": "holonomy + absolute floor + decorrelation (the ticket, literally)",
    "both_si": "holonomy + scale-free floor — #592's question, corrected",
}


def load(seed: int, ticks: int) -> dict:
    out = {}
    for arm in ARM_ORDER:
        path = _HERE / f"592-coexist-{arm}-seed{seed}-{ticks}.json"
        if not path.exists():
            path = path.with_suffix(".inflight.json")
        if path.exists():
            out[arm] = json.loads(path.read_text())
    return out


def _med(entry: dict, subset: str, column: str):
    try:
        return entry["holonomy"]["subsets"][subset][column]["median"]
    except (KeyError, TypeError):
        return float("nan")


def _floor(entry: dict, key: str):
    try:
        return entry["floor"]["quantiles"][key]["median"]
    except (KeyError, TypeError):
        return float("nan")


def _track(entry: dict, key: str):
    try:
        return entry["floor"]["tracking"][key]
    except (KeyError, TypeError):
        return float("nan")


def table(records: dict) -> None:
    print()
    print("=" * 104)
    print("WIDE CYCLES ONLY (base width >= 2) — the population where flatness is not free")
    print("=" * 104)
    head = (
        f"{'arm':<11}{'ticks':>6}{'ident':>8}{'chan':>7}"
        f"{'sigma_max':>11}{'corr':>8}{'corrHaar':>9}{'emit':>7}{'world':>10}"
    )
    print(head)
    print("-" * 104)
    for arm in ARM_ORDER:
        rec = records.get(arm)
        if not rec:
            continue
        for cp in rec["checkpoints"]:
            world = cp.get("motion", {}).get("std_max", float("nan"))
            star = "*" if cp.get("past_stall") else " "
            print(
                f"{arm:<11}{cp['ticks']:>5}{star}"
                f"{_med(cp, 'wide', 'identification'):>8.4f}"
                f"{_med(cp, 'wide', 'channel_return'):>7.3f}"
                f"{_med(cp, 'wide', 'sigma_max'):>11.2e}"
                f"{_track(cp, 'corr_learned'):>8.3f}"
                f"{_track(cp, 'corr_haar'):>9.3f}"
                f"{_floor(cp, 'pr_emitted_centred'):>7.2f}"
                f"{world:>10.2e}"
            )
        print("-" * 104)
    print("* past this arm's stall (~tick 125) — the world column is the check, not the flag")
    print("corr = across-cell corr(state, emitted); corrHaar is the same on the Haar control")


def verdict(records: dict) -> None:
    """The two clauses, each answered against its own arm, with no averaging."""
    print()
    print("=" * 104)
    print("THE TWO CLAUSES")
    print("=" * 104)
    base = records.get("baseline")
    if not base:
        print("no baseline arm — nothing to difference against")
        return
    # Read at the last LIVE rung. Reading the verdict at 2,000 ticks would take it
    # against a motionless world, which is the thing this rig has to stop doing.
    def live_cp(rec):
        live = [c for c in rec["checkpoints"] if not c.get("past_stall") and c["ticks"]]
        return live[-1] if live else rec["checkpoints"][-1]

    last = live_cp(base)
    b_ident = _med(last, "wide", "identification")
    b_corr = _track(last, "corr_learned")
    b_sig = _med(last, "wide", "sigma_max")
    print(
        f"baseline at {last['ticks']} ticks (last live rung): "
        f"ident {b_ident:.4f}  corr {b_corr:+.3f}  sigma_max {b_sig:.3e}"
    )
    print()
    for arm in ARM_ORDER[1:]:
        rec = records.get(arm)
        if not rec:
            continue
        cp = live_cp(rec)
        ident = _med(cp, "wide", "identification")
        corr = _track(cp, "corr_learned")
        haar = _track(cp, "corr_haar")
        sig = _med(cp, "wide", "sigma_max")
        d_ident = ident - b_ident
        d_corr = corr - b_corr
        # A term is only satisfied if it moved its target *and* the loop still
        # carries something. Chu et al.'s failure is exactly the second half.
        flat_moved = d_ident < -0.01
        floor_held = corr >= b_corr - 0.02
        carried = sig >= b_sig * 0.5
        print(f"{arm:<11} — {WHAT[arm]}  @{cp['ticks']} ticks")
        print(
            f"            flatness {ident:.4f} ({d_ident:+.4f})   "
            f"corr {corr:+.3f} ({d_corr:+.3f}, haar {haar:+.3f})   "
            f"sigma_max {sig:.3e}"
        )
        print(
            f"            moved flatness: {str(flat_moved):<5}   "
            f"floor held: {str(floor_held):<5}   "
            f"loop still carries: {str(carried):<5}"
        )
        print()


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ticks", type=int, default=2000)
    args = p.parse_args()
    records = load(args.seed, args.ticks)
    if not records:
        print("no records found")
        return
    for arm, rec in records.items():
        print(
            f"{arm:<11} local cycles {rec['local_cycles']:>3} of "
            f"{rec['cycles']['wide']} wide / {rec['cycles']['full']} full"
        )
    table(records)
    verdict(records)


if __name__ == "__main__":
    main()
