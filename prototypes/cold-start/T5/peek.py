"""Print the composed-ER ladder from a T4 trained-arm record."""

import json
import sys
from pathlib import Path

for path in sys.argv[1:]:
    d = json.loads(Path(path).read_text())
    c = d["at_construction"]["angles"]["composed_er"]
    print(f"{Path(path).name}: construction med {c['median']:.6f} p90 {c['p90']:.4f} max {c['max']:.4f}")
    for cp in d["checkpoints"]:
        e = cp["angles"]["composed_er"]
        print(
            f"  @{cp['ticks']:>7}: med {e['median']:.8f} p90 {e['p90']:.6f} max {e['max']:.4f} "
            f"| excess {e['median'] - 1:.3e}"
        )
