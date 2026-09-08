"""Print the boundary table from a `b38_stall.py` checkpoint, mid-run."""

from __future__ import annotations

import json
import sys
from pathlib import Path

for path in sys.argv[1:]:
    rec = json.loads(Path(path).read_text())
    print(f"\n{path}  window={rec['window']}  threshold={rec['moving_threshold']}")
    print(f"{'arm':>12}{'seed':>6}{'t_moving':>10}{'t_dead':>9}{'peak':>11}{'final':>11}")
    for r in rec["rows"]:
        print(
            f"{r['arm']:>12}{r['seed']:>6}{r['t_moving']:>10}{r['t_dead']:>9}"
            f"{r['std_max_peak']:>11.3e}{r['std_max_final']:>11.3e}"
        )
