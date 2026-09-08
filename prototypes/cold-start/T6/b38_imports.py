"""B38 (#599) ask 1: every T6 instrument built on `arms.py` imports again.

The ticket's words: *"every T6 instrument built on it is stranded on `e75c86c`"*.
This walks them and reports, rather than leaving the claim to the next session.

Usage::

    PYTHONPATH=src python prototypes/cold-start/T6/b38_imports.py
"""

from __future__ import annotations

import importlib.util
import sys
import traceback
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
for extra in (_ROOT / "prototypes" / "chart-double-duty-166", _ROOT / "benchmarks"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

SKIP = {"b38_imports.py", "b38_show.py"}


def main() -> None:
    ok, bad = [], []
    for path in sorted(_HERE.glob("*.py")):
        if path.name in SKIP:
            continue
        spec = importlib.util.spec_from_file_location(f"t6_{path.stem}", path)
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
            ok.append(path.name)
        except Exception:
            bad.append((path.name, traceback.format_exc().strip().splitlines()[-1]))
    for name in ok:
        print(f"  ok    {name}")
    for name, why in bad:
        print(f"  FAIL  {name}: {why}")
    print(f"\n{len(ok)} import, {len(bad)} do not")


if __name__ == "__main__":
    main()
