"""B69 (#646) item 2: the flat bundle past 30,000, where a crossing would have to live.

At 30,000 both seeds sit at ~0.19 and are **still climbing** (+0.0032 and +0.0019 per 1k
against the 0.001-per-1k floor by which `b61_analyse.shapes` calls the staggered arm
stopped), so neither end of #646's pre-registered falsifier fires. #646 item 2 says what
to do about that in terms: *if the null's climb has no floor, a crossing becomes the live
risk and its horizon is what matters.* This runs the rungs that discriminate.

**It changes one thing about `b61_horizon.py`: the ladder.** `b56_channel.run_arm` runs
only to the largest checkpoint at or below `--ticks`, so a longer `--ticks` alone buys
nothing -- it stops at 30,000 and names the file 60,000, which is worse than not running
it. So the ladder is extended and `b61_horizon.run_one` is called unchanged, which keeps
the arm, the staggering, the staging rule and the output filename exactly B61's; every
rung at or below 30,000 reproduces the run this extends.

Rungs are added at 40,000, 50,000 and 60,000 -- three further adjacent pairs at the
spacing #634 §2's discriminator needs, doubling the deepest horizon on this map.

Usage::

    PYTHONPATH=src python prototypes/cold-start/T6/b69_deep.py run --seeds 42 --ticks 60000
"""

from __future__ import annotations

import argparse
import importlib.util
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


b61h = _load("b69_b61h", _HERE / "b61_horizon.py")

#: B61's ladder with three deeper rungs. Everything at or below 30,000 is B61's tuple
#: unchanged, so the extended run is its own regression test against the 30,000 run.
DEEP_LADDER = b61h.LADDER + (40000, 50000, 60000)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["run"])
    p.add_argument("--arms", nargs="+", default=["s0_baseline"])
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--ticks", type=int, default=60000)
    args = p.parse_args()
    b61h.LADDER = DEEP_LADDER
    # Sequential, never parallel: this map's standing note and #555's.
    for seed in args.seeds:
        for name in args.arms:
            b61h.run_one(name, seed, args.ticks)


if __name__ == "__main__":
    main()
