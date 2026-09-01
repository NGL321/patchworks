"""Run the halting rules over the curve bank and print what fired.

**Throwaway.** Built for [issue #156](https://github.com/NGL321/patchworks/issues/156).

    python report.py

Two tables. The first is entry 1 and is the point of the whole prototype: the
same six regimes, judged by a window/slope rule and then by the hold-confirmed
rule, with the verdict each regime deserves alongside. The rest exercises
entries 2, 4 and 5 on the readings they are actually taken from.
"""

from __future__ import annotations

import numpy as np

from curves import BANK
from rules import (
    detectable,
    hold_confirmed,
    honoured_decline,
    horizon_sufficient,
    naive_slope,
    projection_binding,
    regime_split,
    rollout_horizon,
)

TICKS = 3_000
SEED = 20260829

# The budget, and the only free choice in entry 1's rule. TAU_MAX says: this run
# will wait for learning with a time constant up to this many ticks, and past it
# "does not fall" is a claim about *this budget*, which is the only kind of claim
# available. Everything else about the nomination gate follows.
BUDGET = TICKS
TAU_MAX = BUDGET / 3.0
WINDOW = 400
MIN_DECLINE = honoured_decline(window=WINDOW, tau_max=TAU_MAX)
HOLD = 300
SURVIVES = 0.60

# ADR-0004's first comparison: 06-graph-topology.md's topology-only baseline.
# A measured construction-time quantity, stood in for here.
BASELINE = 0.05

# The invented level the naive rule cannot do without, chosen as favourably as
# possible: it sits between the learning regime's tail and the floors, which is
# the best a level can ever do here and still not be enough.
NAIVE_FLOOR = 0.10


def mark(ok: bool) -> str:
    return "ok  " if ok else "WRONG"


def entry_one() -> None:
    rng = np.random.default_rng(SEED)
    print("\n=== entry 1 — irreducible mid-depth error (#151) ===")
    print(f"    budget={BUDGET}  tau_max={TAU_MAX:.0f}  window={WINDOW}")
    print(f"    -> min_decline={MIN_DECLINE:.3f}, derived not chosen; "
          f"detectable at 6% noise: "
          f"{detectable(window=WINDOW, min_decline=MIN_DECLINE, noise=0.06)}")
    print(f"    hold={HOLD}  survives={SURVIVES}  baseline={BASELINE}  "
          f"naive floor={NAIVE_FLOOR}\n")
    header = f"{'regime':<18} {'should':<7} {'naive':<24} {'hold-confirmed':<24}"
    print(header)
    print("-" * len(header))

    naive_wrong = confirmed_wrong = 0
    for regime in BANK:
        error = regime.curve(TICKS, rng)
        n = naive_slope(
            error, window=WINDOW, min_decline=MIN_DECLINE, floor=NAIVE_FLOOR
        )
        c = hold_confirmed(
            error, regime, rng, window=WINDOW, min_decline=MIN_DECLINE,
            hold=HOLD, survives=SURVIVES, baseline=BASELINE,
        )
        naive_wrong += n.fired != regime.should_halt
        confirmed_wrong += c.fired != regime.should_halt
        want = "halt" if regime.should_halt else "run on"
        print(
            f"{regime.name:<18} {want:<7} "
            f"{mark(n.fired == regime.should_halt)} "
            f"{('halt @' + str(n.tick)) if n.fired else 'run on':<17} "
            f"{mark(c.fired == regime.should_halt)} "
            f"{('halt @' + str(c.tick)) if c.fired else 'run on':<17}"
        )

    print(f"\n  wrong verdicts: naive {naive_wrong}/{len(BANK)}, "
          f"hold-confirmed {confirmed_wrong}/{len(BANK)}")
    print("\n  why each regime deserves its verdict:")
    for regime in BANK:
        print(f"    {regime.name:<18} {regime.because}")


def entry_one_sweep() -> None:
    """Is the naive rule rescued by a better window? No, and here is the shape."""
    rng = np.random.default_rng(SEED)
    print("\n=== entry 1 — can a window save the naive rule? ===")
    print("    a window long enough to clear slow-learning is a window that "
          "has already\n    spent the run; and no window separates lag from "
          "curvature at all.\n")
    curves = {r.name: r.curve(TICKS, rng) for r in BANK}
    print(f"{'window':<9} " + " ".join(f"{r.name[:11]:<12}" for r in BANK))
    print("-" * (9 + 13 * len(BANK)))
    for window in (200, 400, 800, 1600, 2400):
        cells = []
        for regime in BANK:
            v = naive_slope(
                curves[regime.name], window=window,
                min_decline=MIN_DECLINE, floor=NAIVE_FLOOR,
            )
            cells.append(("halt" if v.fired else "run on")
                         + ("" if v.fired == regime.should_halt else " !"))
        print(f"{window:<9} " + " ".join(f"{c:<12}" for c in cells))
    print("\n    '!' marks a verdict that disagrees with the regime's truth.")


def robustness(seeds: int = 200) -> None:
    """One table is one seed. This is the same judgement over many."""
    print(f"\n=== entry 1 — the same judgement over {seeds} seeds ===\n")
    naive_wrong = confirmed_wrong = 0
    per = {r.name: 0 for r in BANK}
    for s in range(seeds):
        rng = np.random.default_rng(s)
        for regime in BANK:
            error = regime.curve(TICKS, rng)
            n = naive_slope(
                error, window=WINDOW, min_decline=MIN_DECLINE,
                floor=NAIVE_FLOOR,
            )
            c = hold_confirmed(
                error, regime, rng, window=WINDOW, min_decline=MIN_DECLINE,
                hold=HOLD, survives=SURVIVES, baseline=BASELINE,
            )
            naive_wrong += n.fired != regime.should_halt
            confirmed_wrong += c.fired != regime.should_halt
            per[regime.name] += c.fired != regime.should_halt

    total = seeds * len(BANK)
    print(f"  {total} judgements")
    print(f"    naive wrong:          {naive_wrong:5d}  "
          f"({naive_wrong / total:.1%})")
    print(f"    hold-confirmed wrong: {confirmed_wrong:5d}  "
          f"({confirmed_wrong / total:.1%})")
    print("\n  hold-confirmed errors by regime:")
    for name, wrong in per.items():
        note = ""
        if name == "slow-affordable" and wrong:
            note = "  <- tau 800 against tau_max 1000; the boundary case"
        print(f"    {name:<20} {wrong:3d}/{seeds}{note}")


def entry_two() -> None:
    print("\n=== entry 2 — rollout horizon from the singular-value gap (#144) ===")
    print("    a construction check, not a halt: read once from K.")
    print("    required horizons from 06-graph-topology.md via 08: reflex 3 "
          "ticks, visual 8.\n")
    tol = 0.1
    print(f"{'s2/s1':<8} {'horizon (ticks)':<18} {'reflex (3)':<14} {'visual (8)':<14}")
    print("-" * 54)
    for gap in (0.30, 0.50, 0.70, 0.85, 0.95, 0.99):
        s = np.array([1.0, gap] + [gap * 0.5] * 10)
        h = rollout_horizon(s, tol=tol)
        r3 = horizon_sufficient(s, required=3, tol=tol)
        r8 = horizon_sufficient(s, required=8, tol=tol)
        print(f"{gap:<8.2f} {h:<18.1f} "
              f"{('FAILS' if r3.fired else 'ok'):<14} "
              f"{('FAILS' if r8.fired else 'ok'):<14}")


def entry_four() -> None:
    rng = np.random.default_rng(SEED)
    print("\n=== entry 4 — the band bound, not the band exceeded (#140) ===")
    print("    post-projection sigma_max is never out of band, so the live "
          "reading is\n    the PRE-projection value: how often learning has to "
          "be pulled back.\n")
    ticks, cells = 1_200, 64
    dwell, fraction, burn_in = 100, 0.25, 200
    print(f"    dwell={dwell}  fraction={fraction}  burn_in={burn_in} "
          f"(K = aI at init, so the first excursion is the rig settling)\n")

    settling = rng.normal(0.85, 0.10, (ticks, cells))
    settling[:150] += np.linspace(0.35, 0.0, 150)[:, None]

    amplifying = rng.normal(0.85, 0.10, (ticks, cells))
    amplifying[400:, : cells // 2] += 0.40

    for name, series, want in (
        ("settling", settling, False),
        ("amplifying", amplifying, True),
    ):
        v = projection_binding(
            series, dwell=dwell, fraction=fraction, burn_in=burn_in
        )
        print(f"  {name:<12} want {'halt' if want else 'run on':<7} "
              f"{mark(v.fired == want)}  {v.reading}")


def entry_five() -> None:
    rng = np.random.default_rng(SEED)
    print("\n=== entry 5 — one band, two regimes (#146 on #149's ground) ===")
    print("    08's instrument, reused: non-overlapping IQRs plus a widening "
          "ordering.\n")
    n = 4_000
    windows, min_samples = 4, 200

    free = rng.lognormal(-2.0, 0.35, n)
    harder = rng.lognormal(-1.6, 0.35, n)              # contact is just harder
    straining = rng.lognormal(-1.6, 0.35, n) * np.linspace(1.0, 2.6, n)

    for name, contact, want in (
        ("comfortable", free * 1.02, False),
        ("harder-but-stable", harder, False),
        ("straining", straining, True),
    ):
        v = regime_split(
            contact, free, windows=windows, min_samples=min_samples
        )
        print(f"  {name:<20} want {'halt' if want else 'run on':<7} "
              f"{mark(v.fired == want)}  {v.reading}")


if __name__ == "__main__":
    entry_one()
    entry_one_sweep()
    robustness()
    entry_two()
    entry_four()
    entry_five()
    print()
