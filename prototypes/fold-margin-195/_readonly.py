"""Does reading the margin every tick of the hold disturb the floor?

The margin read goes through `sheaf.evidence()` and `body.encode_parts`, both of
which are pure by inspection. This checks it rather than asserting it: the same
hold, run with and without the per-tick read, and the floor compared.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2] / "benchmarks"))
from untrained_fixed_point import build, hold, restore, snapshot, teaching  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from margin import margins_over_window, per_cell_floor  # noqa: E402

TARGET = 600
HOLD = 400


def go(with_margins: bool):
    _, agent = build("real", "train", 42)
    outcome = None
    for outcome in teaching(agent, TARGET, 42):
        pass
    snapshot(agent.sheaf)
    if with_margins:
        margins_over_window(agent, outcome.observation, outcome.applied, HOLD)
    else:
        hold(agent, outcome.observation, outcome.applied, None, HOLD)
    return per_cell_floor(agent)


a1 = go(False)
a2 = go(False)
b = go(True)
print(f"plain vs plain      : maxdiff {np.abs(a1 - a2).max():.3e}  "
      f"{'identical' if np.array_equal(a1, a2) else 'DIVERGED'}")
print(f"plain vs with-read  : maxdiff {np.abs(a1 - b).max():.3e}  "
      f"{'identical' if np.array_equal(a1, b) else 'DIVERGED'}")
