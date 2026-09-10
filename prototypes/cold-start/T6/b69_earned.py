"""B69 (#646) item 3: does `earned` move as the trained flat bundle's differentiation does?

[B48 (#615)](https://github.com/NGL321/patchworks/issues/615)'s null is stated as *the
flat bundle's 88 earned dimensions at audience differentiation **0.0000**, bought with
5.2 dimensions of exposure per cell*. Both halves of that are **construction** readings.
[B61 (#634)](https://github.com/NGL321/patchworks/issues/634) showed the differentiation
half does not stay put -- `s0_baseline` climbs 0.0000 -> 0.1269 by 9,000 ticks on seed 42
-- so the paired quantity has to be re-read on the same trajectory before the null can be
called a point.

**This stands up no new rig, and it does not touch the arms.** It is `b61_horizon.run_one`
with exactly one addition: `b56_channel.exposure` -- which the checkpoint already calls at
every rung with the live `(dome, maps)` -- is wrapped so the returned dict also carries
[B22 (#571)](https://github.com/NGL321/patchworks/issues/571)'s `whole_graph_split`, which
is B48's own instrument (`b42_earned.py` was three calls to it, unmodified). The arm, the
term, the initialisation, the ladder, the seed and the null are B61's byte for byte, so
every column beside `earned` is a regression test against #634's published table.

**Why 9,000 and not 30,000.** The box is at 94% commit and #634 lost five arms to the
low-memory guard; the 30,000-tick runs that answer items 1-2 carry the pre-registered
falsifier and must not be put at risk by an SVD added at every rung. 9,000 is where #634
published `s0` on both seeds, and it spans differentiation 0.0000 -> 0.1269, which is the
whole range item 3 asks about.

Usage::

    PYTHONPATH=src python prototypes/cold-start/T6/b69_earned.py run --seeds 42 --ticks 9000
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
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


b58 = _load("b69_b58", _HERE / "b58_train.py")
b56 = b58.b56
b22 = _load("b69_b22", _HERE / "b22_regions.py")

#: B61's ladder, truncated at the horizon this runs to. Same rungs, so every row is
#: directly comparable to #634's table rather than re-indexed.
LADDER = (0, 50, 100, 150, 250, 500, 1000, 2000, 4000, 6000, 9000)

ARM = "s0_baseline"
TICKS = 9000


def run_one(name: str, seed: int, ticks: int) -> None:
    stagger, base_arm, dome_arm = b58.ARMS[name]
    out = _HERE / f"646-earned-{name}-seed{seed}-{ticks}.json"
    if out.exists():
        print(f"[B69] {out.name} already at the horizon, skipping", flush=True)
        return

    # #634's staging rule: keep a killed attempt, numbered, and never overwrite it.
    inflight = out.with_suffix(".inflight.json")
    if inflight.exists():
        n = 1
        while (kept := out.with_suffix(f".killed-{n}.json")).exists():
            n += 1
        inflight.rename(kept)
        try:
            depth = json.loads(kept.read_text(encoding="utf-8"))["checkpoints"][-1]["ticks"]
            print(f"[B69] kept a killed attempt that reached {depth} ticks -> {kept.name}",
                  flush=True)
        except Exception:
            print(f"[B69] kept a killed attempt -> {kept.name}", flush=True)

    original_build = b56.b33.arms_mod.build_arm

    def staggered_build_arm(arm, s, *args, **kwargs):
        env, agent = original_build(arm, s, *args, **kwargs)
        b58.apply_stagger(agent, s, stagger)
        return env, agent

    # The one addition. `exposure` is looked up in `b56_channel`'s globals at every
    # checkpoint call, so wrapping it here reaches the closure without touching
    # `run_arm`. The surface's *shape* is a construction fact, so it is built once
    # and `refresh`ed -- B22's own note.
    original_exposure = b56.exposure
    surface: dict = {}

    def exposure_with_earned(dome, maps):
        entry = original_exposure(dome, maps)
        try:
            started_split = time.time()
            if "s" not in surface:
                surface["s"] = b22.Surface(dome, maps)
            surface["s"].refresh(maps)
            entry["earned"] = b22.whole_graph_split(surface["s"])
            entry["earned"]["split_seconds"] = round(time.time() - started_split, 1)
        except Exception as exc:  # never take the arm down for the secondary reading
            entry["earned"] = {"error": type(exc).__name__, "detail": str(exc)[:300]}
        return entry

    started = time.time()
    b56.b33.arms_mod.build_arm = staggered_build_arm
    b56.exposure = exposure_with_earned
    prior_arm, prior_ladder = b56.ARM, b56.CHECKPOINTS
    b56.ARM = dome_arm
    b56.CHECKPOINTS = LADDER
    try:
        record = b56.run_arm(base_arm, seed, ticks, out)
    finally:
        b56.b33.arms_mod.build_arm = original_build
        b56.exposure = original_exposure
        b56.ARM = prior_arm
        b56.CHECKPOINTS = prior_ladder
    record["issue"] = 646
    record["reading"] = "does `earned` move as the trained flat bundle's differentiation does"
    record["stagger"] = stagger
    record["b56_arm"] = base_arm
    record["dome_arm"] = dome_arm
    record["arm"] = name
    record["ladder"] = list(LADDER)
    out.write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(
        f"[B69] {name} (stagger {stagger}, seed {seed}) done in "
        f"{(time.time() - started) / 60:.1f} min -> {out.name}",
        flush=True,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["run"])
    p.add_argument("--arms", nargs="+", default=[ARM])
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--ticks", type=int, default=TICKS)
    args = p.parse_args()
    # Sequential, never parallel: this map's standing note and #555's.
    for seed in args.seeds:
        for name in args.arms:
            run_one(name, seed, args.ticks)


if __name__ == "__main__":
    main()
