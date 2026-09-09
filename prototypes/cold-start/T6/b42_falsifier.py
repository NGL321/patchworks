"""B42 (#605): B40's falsifier, re-run on a flat bundle that is actually flat.

B40 carried the flat bundle as **arm 4, a falsifier** of B34's criterion: if
holonomy cannot fail, the count reads full width everywhere and carves nothing.
It did not fire -- the criterion still discriminated (std 1.2-1.5, 6-7 distinct
values). But `b42_reserve.py` shows *why* it did not: B40 placed the frame in the
ambient `n`, the reserve mask then zeroed columns past `k_v`, and the surface B40
scored was **not** cycle-consistent (`identification` 0.6123, and 0.0386 after the
band). The falsifier was read on a broken bundle.

With the frame built inside the permitted window, `identification` is 0.0000 through
both the mask and the band. So the antecedent B40 could not establish now holds, and
the falsifier gets its real run.

Arms, all `reserve_p12` seed 42, criterion read exactly as `b40_routes.read_criterion`:

* ``control``  -- the arm's own trained maps, as B40 scored them.
* ``ambient``  -- B40's flat bundle: ambient frame, projected. Not actually flat.
* ``reserved`` -- the frame inside the window, projected. `identification` 0.0000.

If ``reserved`` collapses the spread that ``ambient`` retained, B34's criterion is
falsified on a genuinely flat bundle and B40's negative result was an artefact of
the placement.

Usage::

    PYTHONPATH=src python prototypes/cold-start/T6/b42_falsifier.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
for extra in (_ROOT / "prototypes" / "chart-double-duty-166", _ROOT / "benchmarks"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


b33 = _load("b42f_b33", _HERE / "b33_coexist.py")
b29 = _load("b42f_b29", _HERE / "b29_holonomy.py")
routes = _load("b42f_routes", _HERE / "b40_routes.py")
res = _load("b42f_res", _HERE / "b42_reserve.py")


def main() -> None:
    seed, threshold = 42, 0.9
    out: dict = {"ticket": 605, "surface": b33.ARM, "seed": seed,
                 "threshold": threshold, "arms": {}}

    dome0, _agent0, _maps0 = res.build_with_frames(seed, reserved=False)
    per_edge = routes.cycle_sets(dome0)["per_edge"]

    def score(label, dome, maps):
        s = b29.surface_read(dome, maps, b29.cycles_of(dome)["wide"], label)
        ident = s["subsets"]["wide"]["identification"]["median"]
        c = routes.read_criterion(dome, maps, per_edge, threshold)
        ev = c["allocated_with_evidence"]
        row = {
            "identification": ident,
            "spread": ev,
            "edges_with_no_cycle": c["edges_with_no_cycle"],
        }
        print(f"{label:>10}: ident {ident:.4f}   spread {ev}")
        return row

    _env, agent = b33.arms_mod.build_arm(b33.ARM, seed)
    out["arms"]["control"] = score("control", agent.dome, agent.sheaf.maps)

    dome, _a, maps = res.build_with_frames(seed, reserved=False)
    maps.project()
    out["arms"]["ambient"] = score("ambient", dome, maps)

    dome, _a, maps = res.build_with_frames(seed, reserved=True)
    maps.project()
    out["arms"]["reserved"] = score("reserved", dome, maps)

    path = _HERE / f"605-falsifier-seed{seed}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {path.name}")


if __name__ == "__main__":
    main()
