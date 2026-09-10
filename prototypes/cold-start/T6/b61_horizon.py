"""B61 (#634): does the staggered frame's differentiation settle, or slide to the flat bundle?

[B58 (#630)](https://github.com/NGL321/patchworks/issues/630) measured a staggered
initialisation surviving 2,000 ticks of the *shipped* transport rule at
`channel_return` 0.9984 and audience differentiation 0.5317 -- the first arm on
[#532](https://github.com/NGL321/patchworks/issues/532) to pass
[B48 (#615)](https://github.com/NGL321/patchworks/issues/615)'s joint rule at a live
rung, with no objective term at all. But between rungs 1,000 and 2,000 the
differentiation is still falling at ~0.035 per 1,000 ticks, and #630 says in terms
that it *"does not show it settles rather than continuing to the flat bundle over a
much longer run"*.

**This stands up no new rig either.** It re-runs `b58_train`'s own arms, byte-identical
in every column, term, null and initialisation, and patches exactly one thing: the
**checkpoint ladder**, which B56 capped at 2,000. Nothing about the arms changes, so
every rung at or below 2,000 is directly comparable to #630's published table and acts
as its own regression test.

Two arms carry the question and a third and fourth carry the controls:

* **`s19_baseline`** -- #630's headline arm, unchanged, run out. The decaying quantity.
* **`s0_baseline`** -- stagger 0 **is** B42's flat bundle, the null the decay is
  supposedly heading for. [B50 (#618)](https://github.com/NGL321/patchworks/issues/618)'s
  *the null is not static* means the gap must be read against the null's **own** drift
  at the same rung, never against 0.0000.
* **`p8_s22_baseline`** -- #630 measured `p = 8` at *higher* differentiation than
  `p = 12` at 2,000 (0.5697 against 0.5317). Whether that ordering survives the horizon
  is what makes it a fact about `p` rather than about the first two thousand ticks.
* A **second seed**, because [B38 (#599)](https://github.com/NGL321/patchworks/issues/599)
  found the stall horizon varying **13x** between seeds of one arm. Every run stamps its
  own horizon through B56's `motion` reader and inherits nothing.

**What a reading past the stamp is a reading of.** B56's standing exemption covers *a
term whose gradient never reads a stalk*. These arms are the **shipped transport rule**,
which does read stalks, so #630 did not claim the exemption and neither does this. Every
rung past each run's own stamp is drift under a frozen stimulus, and a longer horizon
makes that worse rather than better. The ladder below is therefore built to answer the
*shape* question -- is the decay decaying -- which the stamp does not touch, and the
readout must say what the 30,000-tick number is a number about.

Usage::

    PYTHONPATH=src python prototypes/cold-start/T6/b61_horizon.py run \
        --arms s19_baseline --seed 42 --ticks 30000
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


b58 = _load("b61_b58", _HERE / "b58_train.py")
b56 = b58.b56

#: B56's ladder, extended. Everything at or below 2,000 is B56's own tuple unchanged,
#: so #630's rungs are reproduced rather than re-indexed; above it the spacing is
#: geometric-ish, because the question -- *is the rate falling* -- is read off adjacent
#: rung **pairs** and needs pairs at several scales, not a fine grid at one.
LADDER = (50, 100, 150, 250, 500, 1000, 2000, 4000, 6000, 9000, 13000, 18000, 24000, 30000)

TICKS = 30000


def run_one(name: str, seed: int, ticks: int) -> None:
    """One of B58's arms, on the extended ladder, written under this ticket's name."""
    stagger, base_arm, dome_arm = b58.ARMS[name]
    out = _HERE / f"634-horizon-{name}-seed{seed}-{ticks}.json"
    if out.exists():
        print(f"[B61] {out.name} already at the horizon, skipping", flush=True)
        return

    # **Stage any leftover in-flight file aside before starting.** B56's `run_arm`
    # writes each rung to `<name>.inflight.json` and renames only at the horizon,
    # which bounds a kill -- but it writes that file from the *first* frame, so a
    # retry that dies early **destroys a deeper attempt that came before it**. That
    # is not hypothetical: on this ticket a retry at rung 500 overwrote a partial
    # that had reached 18,000. `benchmarks/rim_stalk_scale.py` is the worked example
    # and this is its rule -- keep the corpse, numbered, and never overwrite it.
    inflight = out.with_suffix(".inflight.json")
    if inflight.exists():
        n = 1
        while (kept := out.with_suffix(f".killed-{n}.json")).exists():
            n += 1
        inflight.rename(kept)
        try:
            depth = json.loads(kept.read_text(encoding="utf-8"))["checkpoints"][-1]["ticks"]
            print(f"[B61] kept a killed attempt that reached {depth} ticks -> {kept.name}",
                  flush=True)
        except Exception:
            print(f"[B61] kept a killed attempt -> {kept.name}", flush=True)

    original = b56.b33.arms_mod.build_arm

    def staggered_build_arm(arm, s, *args, **kwargs):
        env, agent = original(arm, s, *args, **kwargs)
        b58.apply_stagger(agent, s, stagger)
        return env, agent

    started = time.time()
    b56.b33.arms_mod.build_arm = staggered_build_arm
    prior_arm, prior_ladder = b56.ARM, b56.CHECKPOINTS
    b56.ARM = dome_arm
    b56.CHECKPOINTS = LADDER
    try:
        record = b56.run_arm(base_arm, seed, ticks, out)
    finally:
        b56.b33.arms_mod.build_arm = original
        b56.ARM = prior_arm
        b56.CHECKPOINTS = prior_ladder
    record["issue"] = 634
    record["reading"] = "does the staggered frame's differentiation settle above the flat bundle"
    record["stagger"] = stagger
    record["b56_arm"] = base_arm
    record["dome_arm"] = dome_arm
    record["arm"] = name
    record["ladder"] = list(LADDER)
    out.write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(
        f"[B61] {name} (stagger {stagger}, term {base_arm}, seed {seed}) done in "
        f"{(time.time() - started) / 60:.1f} min -> {out.name}",
        flush=True,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["run"])
    p.add_argument("--arms", nargs="+", default=["s19_baseline", "s0_baseline"])
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--ticks", type=int, default=TICKS)
    args = p.parse_args()
    # Sequential, never parallel: this map's standing note, and #555's -- parallel long
    # runs trip the low-memory guard and the record is written as each rung lands.
    for seed in args.seeds:
        for name in args.arms:
            run_one(name, seed, args.ticks)


if __name__ == "__main__":
    main()
