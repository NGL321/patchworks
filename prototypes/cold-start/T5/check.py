"""Re-derive every figure the READOUT asserts, straight from the arm records."""

import json
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_T4 = _HERE.parent / "T4"


def arm(path):
    d = json.loads(Path(path).read_text())
    base = d["at_construction"]["angles"]["composed_er"]["median"] - 1.0
    last = d["checkpoints"][-1]
    return d, base, last


for label, path in [
    ("m=3 100k", _T4 / "537-baseline-seed42-100000.json"),
    ("m=3 20k", _T4 / "537-baseline-seed42-20000.json"),
    ("m=6 20k", _HERE / "546-baseline-m6-seed42-20000.json"),
    ("m=10 20k", _HERE / "546-baseline-m10-seed42-20000.json"),
    ("m=14 20k", _HERE / "546-baseline-m14-seed42-20000.json"),
    ("m=14 100k", _HERE / "546-baseline-m14-seed42-100000.json"),
]:
    d, base, last = arm(path)
    e = last["angles"]["composed_er"]["median"] - 1.0
    c1 = next(r for r in last["sweep_c"] if r["c"] == 1)["er_median"] - 1.0
    print(
        f"{label:>10}: base {base:.4e} -> @{last['ticks']} {e:.4e} "
        f"x{base / e:.1f} retains {100 * e / base:.3f}% removed {base - e:.4e} "
        f"| c1 {c1:.4e} adv x{c1 / e:.2f}"
    )
