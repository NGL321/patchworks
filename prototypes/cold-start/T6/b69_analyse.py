"""B69 (#646): what the flat bundle is at horizon, and whether the pre-registered falsifier fires.

Reads the arms `b61_horizon.py` wrote and answers this ticket's items in order. It
**reuses [B61 (#634)](https://github.com/NGL321/patchworks/issues/634)'s reader wholesale**
-- `b61_analyse.load` (deepest attempt on disk), `stall_diagnostic` (both stamping rules
plus the blips), `table` and `segments` -- so nothing about how a rung is read is new here
and #634's published rows are reproduced rather than re-derived.

What it adds is the three readings #646 asks for that #634's reader does not express:

* **the falsifier**, evaluated rather than described: *the null's climb settles below the
  staggered frame's 0.3912, on both seeds, and the gap therefore has a positive floor.*
  Settling is decided by the **same** noise floor `b61_analyse.shapes` uses to call the
  staggered arm stopped -- 0.001 per 1k -- because a settle criterion that is looser for
  the null than for the candidate would decide the comparison by its own asymmetry.
* **the seed spread** on level and on rate, separately: #634 found the flat bundle's
  *level* carries a 19% spread against the staggered arm's 2% while its *rate* agrees to
  5%, so a single-seed level statement is not a finding.
* **`earned` against differentiation** on the same trajectory, from `b69_earned.py`'s arm.

Usage::

    PYTHONPATH=src python prototypes/cold-start/T6/b69_analyse.py 42 43
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
for extra in (_ROOT / "prototypes" / "chart-double-duty-166", _ROOT / "benchmarks"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


b61 = _load("b69_b61", _HERE / "b61_analyse.py")

#: #634 §1's measured settle for `s19_baseline`, seed 42, to 30,000. The level the
#: falsifier is written against. Quoted, not re-measured -- it is that run's number.
STAGGERED_SETTLE = 0.3912

#: `b61_analyse.shapes`'s own floor: a rate that moves the level by less than 0.001
#: over a rung pair is not a rate, because two runs at a fixed seed differ by ~5%.
NOISE_PER_1K = 0.001


def climb(rs: list[dict]) -> list[dict]:
    """Adjacent-rung climb rate, #634 §2's discriminator. No fit."""
    return b61.segments(rs, "differentiation")


def settled(segs: list[dict], pairs: int = 2) -> tuple[bool, list[float]]:
    """Has the level stopped, by the criterion applied to the staggered arm?

    The last `pairs` adjacent-rung rates must both sit under the noise floor. This is
    deliberately the **same** test `shapes` applies to `s19_baseline`; a null held to a
    looser standard than the candidate would settle by definition.
    """
    tail = [s["per_1k"] for s in segs[-pairs:]]
    return (len(tail) == pairs and all(abs(r) < NOISE_PER_1K for r in tail)), tail


def implied_asymptote(segs: list[dict], rs: list[dict]) -> tuple[float, float]:
    """Where the climb lands if its rate keeps falling at the ratio it currently is.

    `b61_analyse.shapes`'s three-point trick, pointed upward: if `y = A - B r^t` on
    rungs whose successive differences fall by a constant factor `r`, the asymptote is
    `y_last + step * r / (1 - r)`. Reported as **arithmetic on the last three rungs and
    not as a mechanism** -- #634's standing phrasing -- because nothing here derives
    that the rate keeps falling at that ratio; it is what the rungs currently do.
    """
    if len(segs) < 2 or len(rs) < 3:
        return float("nan"), float("nan")
    d1, d2 = segs[-2]["per_1k"], segs[-1]["per_1k"]
    ratio = d2 / d1 if d1 else float("nan")
    if not (0 < ratio < 1):
        return ratio, float("nan")
    step = rs[-1]["differentiation"] - rs[-2]["differentiation"]
    return ratio, rs[-1]["differentiation"] + step * ratio / (1.0 - ratio)


def falsifier(per_seed: dict) -> dict:
    """The pre-registered proposition, evaluated on every seed that reached the horizon.

    Stated on #646: *the null's climb settles below the staggered frame's 0.391, on both
    seeds, and the gap therefore has a positive floor.* Two ends are named -- it settles
    below, or it reaches 0.391 -- and this reports which fires, **including neither**,
    which is a real outcome and the one a 30,000-tick horizon can produce.
    """
    verdict = {"level_to_beat": STAGGERED_SETTLE, "seeds": {}}
    for seed, rs in per_seed.items():
        segs = climb(rs)
        is_settled, tail = settled(segs)
        level = rs[-1]["differentiation"]
        gap = STAGGERED_SETTLE - level
        rate = segs[-1]["per_1k"] if segs else float("nan")
        ratio, asym = implied_asymptote(segs, rs)
        verdict["seeds"][seed] = {
            "rate_ratio": ratio,
            "implied_asymptote": asym,
            "asymptote_below_staggered": (asym < STAGGERED_SETTLE) if asym == asym else None,
            "horizon": rs[-1]["ticks"],
            "level": level,
            "gap": gap,
            "last_rate_per_1k": rate,
            "tail_rates_per_1k": tail,
            "settled": is_settled,
            "reached_staggered_level": level >= STAGGERED_SETTLE,
            # Linear only, and named as unsafe: the rate is decaying over the run as a
            # whole, so this is an *upper* bound on how fast a crossing could arrive if
            # the current rate persisted, not a prediction that it will.
            "ticks_to_close_at_current_rate": (
                gap / rate * 1000.0 if rate and rate > 0 else float("nan")
            ),
        }
    ends = verdict["seeds"].values()
    verdict["settles_below_on_every_seed"] = bool(ends) and all(
        e["settled"] and not e["reached_staggered_level"] for e in ends
    )
    verdict["reaches_on_any_seed"] = any(e["reached_staggered_level"] for e in ends)
    if verdict["settles_below_on_every_seed"]:
        verdict["fires"] = "settles below -- the gap has a positive floor"
    elif verdict["reaches_on_any_seed"]:
        verdict["fires"] = "reaches 0.391 -- the staggered frame's guarantee is a transient"
    else:
        verdict["fires"] = (
            "neither end fires at this horizon: the climb has not settled and has not "
            "reached the staggered level"
        )
    return verdict


def earned_table(seed: int, ticks: int = 9000) -> list[dict] | None:
    """`earned` beside the differentiation it is supposed to be paired with.

    B48's null is *88 earned at differentiation 0.0000*. Both halves are construction
    readings; this puts them on one trajectory so the pairing can be read rather than
    assumed.
    """
    path = _HERE / f"646-earned-s0_baseline-seed{seed}-{ticks}.json"
    if not path.exists():
        cand = sorted(_HERE.glob(f"646-earned-s0_baseline-seed{seed}-*.json"))
        cand = [c for c in cand if ".killed-" not in c.name]
        if not cand:
            return None
        path = cand[-1]
    rec = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for c in rec["checkpoints"]:
        e = (c.get("exposure") or {}).get("earned") or {}
        if "error" in e:
            out.append({"ticks": c["ticks"], "error": e["error"]})
            continue
        out.append(
            {
                "ticks": c["ticks"],
                "differentiation": c["differentiation"]["audience_differentiation"],
                "dim_h0": e.get("dim_h0"),
                "trivial": e.get("trivial"),
                "earned": e.get("earned"),
                "generic": e.get("earned_generic"),
                "above_generic": e.get("earned_above_generic"),
                "kv_participation": (c.get("exposure") or {}).get("k_v_participation_median"),
            }
        )
    return out


def main() -> None:
    seeds = [int(a) for a in sys.argv[1:]] or [42, 43]
    report: dict = {"issue": 646, "staggered_settle": STAGGERED_SETTLE, "seeds": {}}
    per_seed: dict = {}

    for seed in seeds:
        rec = b61.load("s0_baseline", seed, 30000)
        if rec is None:
            print(f"\n[B69] no s0_baseline record for seed {seed}")
            continue
        rs = b61.table(f"s0_baseline (the flat bundle) seed {seed}", rec)
        per_seed[seed] = rs
        stall = b61.stall_diagnostic(rec)
        print(
            f"  stall, both rules: B56's last-above says {stall['b56_last_above']}; "
            f"first fall after {stall['first_fall']}; late blips {[t for t, _ in stall['late_blips']]}"
        )
        segs = climb(rs)
        print(f"\n--- s0_baseline seed {seed}: climb rate on adjacent rung pairs ---")
        print(f"{'from':>7} {'to':>7} {'per 1k':>10} {'level':>9} {'gap':>9}")
        for s in segs:
            print(
                f"{s['from']:>7} {s['to']:>7} {s['per_1k']:>+10.5f} "
                f"{s['level_to']:>9.4f} {STAGGERED_SETTLE - s['level_to']:>9.4f}"
            )
        report["seeds"][seed] = {"source": rec.get("_source"), "stall": stall, "rows": rs}

    print("\n=== the pre-registered falsifier ===")
    verdict = falsifier(per_seed)
    for seed, e in verdict["seeds"].items():
        print(
            f"  seed {seed}: horizon {e['horizon']}, level {e['level']:.4f}, "
            f"gap {e['gap']:.4f}, last rate {e['last_rate_per_1k']:+.5f} per 1k, "
            f"settled={e['settled']}, reached={e['reached_staggered_level']}"
        )
        if e["ticks_to_close_at_current_rate"] == e["ticks_to_close_at_current_rate"]:
            print(
                f"      at that rate the remaining gap closes at tick "
                f"~{e['horizon'] + e['ticks_to_close_at_current_rate']:,.0f} "
                f"(linear, and the rate is decaying -- a bound, not a prediction)"
            )
        if e["implied_asymptote"] == e["implied_asymptote"]:
            print(
                f"      rate ratio on the last three rungs {e['rate_ratio']:.3f} "
                f"(<1 -> a falling rate); implied asymptote {e['implied_asymptote']:.4f}, "
                f"{'below' if e['asymptote_below_staggered'] else 'at or above'} the "
                f"staggered frame's {STAGGERED_SETTLE:.4f}"
            )
    print(f"  -> {verdict['fires']}")
    report["falsifier"] = verdict

    if len(per_seed) > 1:
        print("\n=== seed spread, level against rate (#634 §3's asymmetry) ===")
        shared = sorted(
            set.intersection(*({r["ticks"] for r in rs} for rs in per_seed.values()))
        )
        levels = {s: {r["ticks"]: r["differentiation"] for r in rs} for s, rs in per_seed.items()}
        print(f"{'ticks':>7} " + " ".join(f"{'seed ' + str(s):>10}" for s in per_seed) + f" {'spread':>9}")
        for t in shared:
            vals = [levels[s][t] for s in per_seed]
            lo, hi = min(vals), max(vals)
            spread = (hi - lo) / hi if hi else 0.0
            print(f"{t:>7} " + " ".join(f"{v:>10.4f}" for v in vals) + f" {spread:>8.1%}")
        report["shared_rungs"] = shared

    print("\n=== item 3: `earned` against the differentiation it is paired with ===")
    for seed in seeds:
        rows = earned_table(seed)
        if rows is None:
            print(f"  seed {seed}: no 646-earned record yet")
            continue
        print(f"\n  s0_baseline seed {seed}")
        print(
            f"{'ticks':>7} {'diff':>8} {'dim H0':>8} {'trivial':>8} {'earned':>8} "
            f"{'generic':>8} {'above':>7} {'kv part':>8}"
        )
        for r in rows:
            if "error" in r:
                print(f"{r['ticks']:>7}  split failed: {r['error']}")
                continue
            print(
                f"{r['ticks']:>7} {r['differentiation']:>8.4f} {r['dim_h0']:>8} "
                f"{r['trivial']:>8} {r['earned']:>8} {r['generic']:>8} "
                f"{r['above_generic']:>7} {r['kv_participation']:>8.2f}"
            )
        report.setdefault("earned", {})[seed] = rows

    out = _HERE / "646-analysis.json"
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\n[B69] wrote {out.name}")


if __name__ == "__main__":
    main()
