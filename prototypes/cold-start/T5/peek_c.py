"""Print the `sweep_c` ladder from a T4 trained-arm record."""

import json
import sys
from pathlib import Path

for path in sys.argv[1:]:
    d = json.loads(Path(path).read_text())
    print(Path(path).name)
    for cp in d["checkpoints"]:
        row = " ".join(f"c{r['c']}={r['er_median'] - 1:.3e}" for r in cp["sweep_c"])
        print(f"  @{cp['ticks']:>7}: {row}")
