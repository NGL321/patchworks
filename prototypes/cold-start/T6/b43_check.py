"""B43 (#607): the claims in `READOUT-607.md` that a table does not already print.

A few sentences in the readout quantify over thresholds or arms the tables leave
out. Rather than trust the sentence, this prints exactly those numbers.

Usage::

    python prototypes/cold-start/T6/b43_check.py
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def main() -> None:
    d = json.loads((_HERE / "607-width-seed42.json").read_text(encoding="utf-8"))
    print("floored membership == interior degree, per arm/checkpoint/threshold:")
    for a in d["arms"]:
        for ck in a["checkpoints"]:
            row = " ".join(
                f"{th}:{ck['membership_floored'][th]['equals_degree_cells']:>3}"
                for th in sorted(ck["membership_floored"], key=float)
            )
            print(f"  {a['layout']}{a['sites']:<3} @{ck['ticks']:>4}  {row}")
    print("\nstrict Σ_e m_e off B34's allocation == degree, per arm (last checkpoint):")
    for a in d["arms"]:
        s = a["checkpoints"][-1]["strict"]
        print(
            f"  {a['layout']}{a['sites']:<3} "
            f"{s.get('strict_criterion_equals_degree_cells')}"
            f"/{s.get('strict_criterion_cells')}"
        )


if __name__ == "__main__":
    main()
