"""B61 (#634): read the long-horizon arms -- does the decay decay, or does it slide?

Four readings, in the order [#634](https://github.com/NGL321/patchworks/issues/634) asks
for them, and one refusal.

1. **Where differentiation goes**, per arm, on the extended ladder, with each run's
   **own** stall stamp marked. Reused verbatim from `b56_analyse.horizon` and `row` --
   [B38 (#599)](https://github.com/NGL321/patchworks/issues/599)'s rule is that the
   horizon is per-run and inherited from nothing.

2. **Segment-wise rate.** The model-free discriminator. A quantity sliding to zero at a
   constant rate and a quantity relaxing to a nonzero asymptote are the same curve over
   one decade and different curves over two, and the difference is visible in the
   **rate on adjacent rung pairs** without fitting anything. This is
   [B59 (#631)](https://github.com/NGL321/patchworks/issues/631)'s own form -- *at
   adjacent rung pairs and not end-to-end* -- applied to the drift rather than to the
   exchange rate.

3. **Against the null's own drift**, never against 0.0000.
   [B50 (#618)](https://github.com/NGL321/patchworks/issues/618): the null is not static,
   and `s0_baseline` is the flat bundle run to the same horizon on the same seed. The
   quantity that answers the ticket is the **gap at a rung**, not the level.

4. **`p = 8` against `p = 12`**, rung by rung, because #630 measured the ordering at
   2,000 alone and an ordering that flips is a fact about the first two thousand ticks.

**The refusal.** #634 says *fit nothing the checkpoints do not support*. Two candidate
shapes are reported -- a straight line through the last decade, and an exponential
relaxation to an asymptote -- **as arithmetic on the measured rungs and not as a claim
about the mechanism**. The straight line yields a falsifiable number (the tick at which
it would reach the null) and the ticket's own extrapolation was of that kind; the
exponential yields an asymptote. Where the two disagree the segment rates in §2 decide,
because they are the only model-free thing here.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


b56a = _load("b61_b56a", _HERE / "b56_analyse.py")

TICKS = 30000
ARMS = ("s19_baseline", "s0_baseline", "p8_s22_baseline")


def load(arm: str, seed: int, ticks: int = TICKS) -> dict | None:
    """The finished record if there is one, else the **deepest** attempt on disk.

    The box is contended and the low-memory guard kills long arms, so an arm may
    exist only as a set of partial attempts of differing depth. Taking the newest
    would be wrong -- a retry that died at rung 500 is not better evidence than one
    that reached 18,000 -- so this ranks by the last rung actually reached and says
    which file it took and how deep it got. A partial null read against a complete
    candidate is a reportable asymmetry, not a broken comparison, provided the gap
    is only ever read at **shared** rungs, which `gap_table` enforces by construction.
    """
    # The box's low-memory guard forced different arms to different horizons, so the
    # target horizon is **discovered** rather than assumed: every file for this arm and
    # seed competes, complete or partial, and the deepest wins. A complete run outranks
    # a partial that reached the same rung, because only the complete one is a run that
    # finished rather than one that was cut off there.
    best, best_depth, best_name, best_kind = None, -1, "", ""
    for p in sorted(_HERE.glob(f"634-horizon-{arm}-seed{seed}-*.json")):
        name = p.name
        if name.endswith(".inflight.json"):
            kind, rank = "partial", 0
        elif ".killed-" in name:
            kind, rank = "killed", 0
        else:
            kind, rank = "complete", 1
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
            depth = rec["checkpoints"][-1]["ticks"]
        except Exception:
            continue
        if (depth, rank) > (best_depth, 0 if best_kind != "complete" else 1):
            best, best_depth, best_name, best_kind = rec, depth, name, kind
    if best is not None:
        best["_source"] = (
            "complete to {} ticks".format(best_depth)
            if best_kind == "complete"
            else f"{best_kind}, deepest of {best_name} at {best_depth} ticks"
        )
    return best


def stall_diagnostic(record: dict) -> dict:
    """B56's stamp rule, and the thing it gets wrong on a 30,000-tick run.

    `b56_analyse.horizon` returns the **last** rung whose `std_max` clears `MOVING`.
    Over 2,000 ticks that is the same as *when did the body stop*, because the body
    stops once and stays stopped. Over 30,000 it is not: a single late rung that
    twitches back over the threshold drags the reported horizon to the end of the run
    and marks every intervening rung live, when the trajectory plainly shows the body
    frozen throughout. So this reports **both** -- B56's rule unchanged, and the
    **first fall**, the earliest rung after which motion goes under and the run spends
    the overwhelming majority of its rungs under.

    Reported, never silently substituted: B38's rule is that the horizon is stamped
    per-run, and the honest stamp here is the first fall with the late blip named.
    """
    stamps = [
        (c["ticks"], ((c.get("motion") or {}).get("std_max", float("nan"))))
        for c in record["checkpoints"]
        if (c.get("motion") or {}).get("available")
    ]
    last_above = 0
    first_fall = None
    for t, s in stamps:
        if s > b56a.MOVING:
            last_above = t
        elif first_fall is None:
            first_fall = t
    blips = [(t, s) for t, s in stamps if s > b56a.MOVING and first_fall is not None and t > first_fall]
    return {
        "b56_last_above": last_above,
        "first_fall": first_fall,
        "late_blips": blips,
        "stamps": stamps,
    }


def rows(record: dict) -> list[dict]:
    h = b56a.horizon(record)
    out = []
    for c in record["checkpoints"]:
        r = b56a.row(c)
        r["past_stall"] = r["ticks"] > h
        out.append(r)
    return out


def table(label: str, record: dict) -> list[dict]:
    rs = rows(record)
    h = b56a.horizon(record)
    sd = stall_diagnostic(record)
    print(
        f"\n=== {label}  ({record['_source']}, this run's live horizon: {h} ticks, "
        f"stamped not inherited) ==="
    )
    print(
        f"  stall: B56's last-above rule says {sd['b56_last_above']}; "
        f"the body first falls under after {sd['first_fall']}"
        + (f"; late blips over the threshold at {[t for t, _ in sd['late_blips']]}"
           if sd["late_blips"] else "")
    )
    print(
        f"{'ticks':>7} {'chan(w)':>8} {'diff':>7} {'kv rank':>8} {'kv part':>8} "
        f"{'rho(u)':>7} {'tau':>8} {'world':>9}"
    )
    for r in rs:
        print(
            f"{r['ticks']:>6}{'*' if r['past_stall'] else ' '} "
            f"{r['chan_wide']:>8.4f} {r['differentiation']:>7.4f} "
            f"{r['kv_rank']:>8.1f} {r['kv_participation']:>8.1f} "
            f"{r['rho_used']:>7.4f} {r['tau']:>8.4g} {r['std_max']:>9.2e}"
        )
    print("  * past this run's own stall stamp -- drift under a frozen stimulus")
    return rs


def segments(rs: list[dict], key: str = "differentiation") -> list[dict]:
    """Rate per 1,000 ticks on each adjacent rung pair. Model-free."""
    out = []
    for a, b in zip(rs, rs[1:]):
        span = b["ticks"] - a["ticks"]
        if span <= 0:
            continue
        out.append(
            {
                "from": a["ticks"],
                "to": b["ticks"],
                "span": span,
                "delta": b[key] - a[key],
                "per_1k": (b[key] - a[key]) / span * 1000.0,
                "level_to": b[key],
            }
        )
    return out


def show_segments(label: str, rs: list[dict]) -> list[dict]:
    segs = segments(rs)
    print(f"\n--- {label}: differentiation rate on adjacent rung pairs ---")
    print(f"{'from':>7} {'to':>7} {'delta':>9} {'per 1k':>9} {'level':>8}")
    for s in segs:
        print(
            f"{s['from']:>7} {s['to']:>7} {s['delta']:>9.4f} "
            f"{s['per_1k']:>9.5f} {s['level_to']:>8.4f}"
        )
    return segs


def gap_table(s19: list[dict], s0: list[dict], label: str = "s19 - s0") -> list[dict]:
    """B50's rule: the staggered arm read against the null's **own** drift at the rung."""
    by0 = {r["ticks"]: r for r in s0}
    out = []
    print(f"\n--- {label}: gap to the flat bundle at the same rung (B50) ---")
    print(f"{'ticks':>7} {'s19 diff':>9} {'s0 diff':>9} {'gap':>8} {'gap/1k':>9}")
    prev = None
    for r in s19:
        n = by0.get(r["ticks"])
        if n is None:
            continue
        gap = r["differentiation"] - n["differentiation"]
        rate = float("nan")
        if prev is not None and r["ticks"] > prev[0]:
            rate = (gap - prev[1]) / (r["ticks"] - prev[0]) * 1000.0
        print(
            f"{r['ticks']:>6}{'*' if r['past_stall'] else ' '} "
            f"{r['differentiation']:>9.4f} {n['differentiation']:>9.4f} "
            f"{gap:>8.4f} {rate:>9.5f}"
        )
        out.append({"ticks": r["ticks"], "gap": gap, "rate_per_1k": rate})
        prev = (r["ticks"], gap)
    return out


def _linfit(xs: list[float], ys: list[float]) -> tuple[float, float]:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx if sxx else float("nan")
    return slope, my - slope * mx


def shapes(label: str, rs: list[dict], null_level: float) -> dict:
    """The two candidate shapes, as arithmetic on the rungs and not as a mechanism.

    The straight line is fitted on the **last decade only** -- a line through the whole
    run is dominated by the fast transient before tick 250, which nobody claims is the
    decay in question. The exponential is fitted by a three-point ratio on
    geometrically-placed rungs, which is the standard trick and needs no optimiser: if
    `y = A + B r^t` on equally spaced `t`, successive differences fall by a constant
    factor `r`, and the asymptote follows.
    """
    tail = [r for r in rs if r["ticks"] >= 2000]
    print(f"\n--- {label}: two candidate shapes on the rungs at/after 2,000 ---")
    if len(tail) < 3:
        print("  too few rungs past 2,000 to say anything")
        return {}
    xs = [float(r["ticks"]) for r in tail]
    ys = [r["differentiation"] for r in tail]
    slope, intercept = _linfit(xs, ys)
    print(f"  straight line over {int(xs[0])}-{int(xs[-1])}: "
          f"{slope * 1000:.5f} per 1k, level at {int(xs[-1])} = {ys[-1]:.4f}")
    reach = float("nan")
    if slope < 0:
        reach = (null_level - intercept) / slope
        print(f"    -> would reach the null's level ({null_level:.4f}) at "
              f"tick {reach:,.0f}")
    else:
        print(f"    -> slope is not negative; the line does not reach the null")

    # Successive-difference ratio on the last three rungs, span-normalised.
    a, b, c = tail[-3], tail[-2], tail[-1]
    d1 = (b["differentiation"] - a["differentiation"]) / (b["ticks"] - a["ticks"])
    d2 = (c["differentiation"] - b["differentiation"]) / (c["ticks"] - b["ticks"])
    ratio = d2 / d1 if d1 else float("nan")
    print(f"  per-tick rate {a['ticks']}->{b['ticks']}: {d1 * 1000:.5f} per 1k")
    print(f"  per-tick rate {b['ticks']}->{c['ticks']}: {d2 * 1000:.5f} per 1k")
    # **The ratio is only meaningful while the rates are above the noise.** Once both
    # are at the floor the quotient of two near-zeros is arithmetic noise and reads as
    # a large number, which the legend below would misreport as *a slide* -- the exact
    # opposite of what a pair of vanished rates means. Two runs at a fixed seed differ
    # by ~5% on this quantity, so a rate that moves the level by less than 0.001 over
    # a rung pair is not a rate.
    NOISE = 0.001 / 1000.0
    if abs(d1) < NOISE and abs(d2) < NOISE:
        print(f"  both rates are at the noise floor (<{NOISE * 1000:.4f} per 1k): "
              f"the level has stopped moving, and their ratio ({ratio:.3f}) is "
              f"arithmetic on two near-zeros and carries no shape information")
    else:
        print(f"  ratio of the two: {ratio:.3f}"
              f"   (=1 constant rate -> a slide; <1 falling rate -> a settle)")
    asym = float("nan")
    if 0 < ratio < 1:
        # y_inf = c - (last step) * ratio/(1-ratio), stepping the geometric tail out.
        step = c["differentiation"] - b["differentiation"]
        asym = c["differentiation"] + step * ratio / (1.0 - ratio)
        print(f"  implied asymptote if the rate keeps falling at that ratio: {asym:.4f}"
              f"   (null at {null_level:.4f})")
    return {
        "line_per_1k": slope * 1000,
        "line_reaches_null_at": reach,
        "rate_ratio": ratio,
        "implied_asymptote": asym,
        "last_level": ys[-1],
    }


def main() -> None:
    seeds = [int(a) for a in sys.argv[1:]] or [42]
    report: dict = {"issue": 634, "ticks": TICKS, "seeds": seeds, "arms": {}}
    for seed in seeds:
        loaded = {}
        for arm in ARMS:
            rec = load(arm, seed)
            if rec is None:
                print(f"\n[B61] no record yet for {arm} seed {seed}")
                continue
            loaded[arm] = table(f"{arm} seed {seed}", rec)
        for arm, rs in loaded.items():
            segs = show_segments(f"{arm} seed {seed}", rs)
            # #630's *other* headline was exposure: +4.9 to +5.5 participation
            # dimensions over random init. B59's void clause reads participation,
            # not rank, so it gets the same horizon treatment as differentiation.
            psegs = segments(rs, "kv_participation")
            print(f"\n--- {arm} seed {seed}: participation rate on adjacent rung pairs ---")
            print(f"{'from':>7} {'to':>7} {'delta':>9} {'per 1k':>9} {'level':>8}")
            for s in psegs:
                print(
                    f"{s['from']:>7} {s['to']:>7} {s['delta']:>9.3f} "
                    f"{s['per_1k']:>9.5f} {s['level_to']:>8.2f}"
                )
            report["arms"][f"{arm}-s{seed}"] = {
                "rows": rs,
                "segments": segs,
                "participation_segments": psegs,
            }
        null_last = 0.0
        if "s0_baseline" in loaded:
            null_last = loaded["s0_baseline"][-1]["differentiation"]
        for arm in ("s19_baseline", "p8_s22_baseline"):
            if arm in loaded:
                report["arms"][f"{arm}-s{seed}"]["shapes"] = shapes(
                    f"{arm} seed {seed}", loaded[arm], null_last
                )
        if "s19_baseline" in loaded and "s0_baseline" in loaded:
            report["arms"][f"gap-s19-s0-s{seed}"] = gap_table(
                loaded["s19_baseline"], loaded["s0_baseline"], f"s19 vs s0 seed {seed}"
            )
        if "p8_s22_baseline" in loaded and "s0_baseline" in loaded:
            report["arms"][f"gap-p8-s0-s{seed}"] = gap_table(
                loaded["p8_s22_baseline"], loaded["s0_baseline"], f"p8 vs s0 seed {seed}"
            )
        if "p8_s22_baseline" in loaded and "s19_baseline" in loaded:
            by12 = {r["ticks"]: r for r in loaded["s19_baseline"]}
            print(f"\n--- p=8 against p=12 at every shared rung, seed {seed} ---")
            print(f"{'ticks':>7} {'p8 diff':>8} {'p12 diff':>9} {'p8-p12':>8} "
                  f"{'p8 chan':>8} {'p12 chan':>9} {'p8 part':>8} {'p12 part':>9}")
            order = []
            for r in loaded["p8_s22_baseline"]:
                o = by12.get(r["ticks"])
                if o is None:
                    continue
                print(
                    f"{r['ticks']:>7} {r['differentiation']:>8.4f} "
                    f"{o['differentiation']:>9.4f} "
                    f"{r['differentiation'] - o['differentiation']:>8.4f} "
                    f"{r['chan_wide']:>8.4f} {o['chan_wide']:>9.4f} "
                    f"{r['kv_participation']:>8.1f} {o['kv_participation']:>9.1f}"
                )
                order.append(
                    {"ticks": r["ticks"], "p8_minus_p12": r["differentiation"] - o["differentiation"]}
                )
            report["arms"][f"p8-vs-p12-s{seed}"] = order

    out = _HERE / "634-analysis.json"
    out.write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    print(f"\n[B61] -> {out.name}")


if __name__ == "__main__":
    main()
