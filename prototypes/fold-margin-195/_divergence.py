"""Does a checkpoint's hold perturb the trajectory it measures?

#178 asserts it does not, on the evidence that its 30k column reproduced #158's
independent 30k run to five decimals. This asks the question directly: the same
budget read with and without an earlier checkpoint in front of it.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2] / "benchmarks"))
from untrained_fixed_point import build, hold, restore, snapshot, teaching  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from margin import per_cell_floor, surface_margins  # noqa: E402

TARGET = 600
EARLY = 300
HOLD = 400


def run(budgets):
    _, agent = build("real", "train", 42)
    out = {}
    reached = 0
    for outcome in teaching(agent, TARGET, 42):
        reached += 1
        if reached not in budgets:
            continue
        state = snapshot(agent.sheaf)
        hold(agent, outcome.observation, outcome.applied, None, HOLD)
        floors = per_cell_floor(agent)
        surface_margins(agent)
        restore(agent.sheaf, state)
        out[reached] = floors
    return out


alone = run({TARGET})
after = run({EARLY, TARGET})
a, b = alone[TARGET], after[TARGET]
print(f"median floor at {TARGET}: alone {np.median(a):.6f}  "
      f"after a checkpoint at {EARLY} {np.median(b):.6f}")
print(f"max abs difference per cell: {np.abs(a - b).max():.3e}")
print("identical" if np.array_equal(a, b) else "DIVERGED")
