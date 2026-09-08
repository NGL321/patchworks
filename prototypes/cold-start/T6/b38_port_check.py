"""B38 item 1: the port of `arms.py` to the surface #597 shipped, checked.

Three assertions, each of which would have caught a silent drift:

1. **Every arm still builds.** `shipped`, `doubling` and the reserve sweep.
2. **`reserve_p12` IS `main`.** The arm's dome is identical, edge for edge and
   mask for mask, to `build_graph(DomeSpec())` with nothing overridden. This is
   the assertion the old `check_allocator_matches_shipped` was reaching for and
   could not make, because the rig then had no shipped reserve to compare to.
3. **The union arms still reproduce the pre-#556 mask.** `k_v = min(n, Σ_e m_e)`
   at every predicting cell, and the private width is the residual — including
   the 104-of-150 zero-privacy reading that made #548 hold the doubling.

Usage::

    PYTHONPATH=src python prototypes/cold-start/T6/b38_port_check.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
for extra in (_ROOT / "prototypes" / "chart-double-duty-166", _ROOT / "benchmarks"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


arms = _load("t6_arms", _HERE / "arms.py")

from patchworks.graph import NODE_STALK_DIM, DomeSpec, build_graph  # noqa: E402


def main() -> None:
    n = NODE_STALK_DIM

    # 1. every arm builds
    labels = ["shipped", "doubling", "reserve"] + [
        f"reserve_p{p}" for p in (0, 4, 8, 12, 16, 20)
    ]
    for arm in labels:
        dome = build_graph(arms.spec_for(arm))
        _, p, policy = arms.ARMS[arm]
        if policy == "union":
            dome = arms.apply_union(dome)
        perm = np.array([dome._permitted[c] for c in dome.predicting])
        sums = np.array([dome.stalk_sums[c] for c in dome.predicting])
        priv = n - perm
        print(
            f"  {arm:>12} [{policy:>7}] budget {dome.spec.capacity_budget:>2} "
            f"k_v med {np.median(perm):>5.1f} priv min {priv.min():>3} "
            f"zero {(priv <= 0).sum():>3}/{len(perm)} total {priv.clip(min=0).sum():>4}"
        )

    # 2. reserve_p12 is main, exactly
    mine = build_graph(arms.spec_for("reserve_p12"))
    ship = build_graph(DomeSpec())
    assert mine.spec == ship.spec, "reserve_p12's spec is not DomeSpec()"
    assert tuple(e.m for e in mine.edges) == tuple(e.m for e in ship.edges), "lane widths differ"
    assert mine._permitted == ship._permitted, "permitted blocks differ"
    assert torch.equal(mine._private_mask, ship._private_mask), "masks differ"
    print("\n  reserve_p12 == build_graph(DomeSpec()): spec, lanes, permitted, mask")

    # 3. the union arms reproduce the retired mask
    for arm in ("shipped", "doubling"):
        dome = arms.apply_union(build_graph(arms.spec_for(arm)))
        perm = np.array([dome._permitted[c] for c in dome.predicting])
        sums = np.array([dome.stalk_sums[c] for c in dome.predicting])
        assert np.array_equal(perm, np.minimum(n, sums)), f"{arm}: not the union mask"
        zero = int((n - perm <= 0).sum())
        print(f"  {arm:>12}: k_v == min(n, sum_m) at all {len(perm)} cells, "
              f"zero-privacy at {zero}")

    print("\nport check passed")


if __name__ == "__main__":
    main()
