"""B56 (#628): does the exact-stagger set move with the window it wraps in?

`b56_stagger6.py` settles #628's cheap question — stagger 6 is **not** an
artifact: the staggers reading `sigma_max` 1.000 at `channel_return` 1.0000 are
`{0, 1, 2, 6, 7, 10, 13, 14, 18, 19}` on **all three** seeds, and `edge_overlap`
agrees to four decimals across seeds. So the effect is a property of the row
arithmetic and not of the random frame.

That leaves the mechanism. Every stagger wraps in the permitted window
`k_v = n - p`, which is **20** at the shipped `p = 12`. If the exact set is
arithmetic then changing `p` must move it, and move it in a way keyed to the new
`k_v` rather than to anything about training.

This sweeps `p` in `{8, 12, 16}` — `k_v` in `{24, 20, 16}` — at one seed, and
reports the exact set at each. Nothing here is trained. Cheap; construction only.

**Why this is worth the two minutes.** B54's argument was carried by staggers 1
and 2 at differentiation 0.3000 and 0.3333. If the exact set is arithmetic in
`k_v`, then *where the differentiation ceiling sits at perfect channel return* is
something the architecture **chooses** when it chooses `p` — and `p` is an
initialisation, not a constant ([B42](https://github.com/NGL321/patchworks/issues/605)).

Usage::

    PYTHONPATH=src python prototypes/cold-start/T6/b56_stagger_width.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
for extra in (_ROOT / "prototypes" / "chart-double-duty-166", _ROOT / "benchmarks"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


b42 = _load("b56w_b42", _HERE / "b42_stagger.py")
b29 = _load("b56w_b29", _HERE / "b29_holonomy.py")
b33 = b42.b33

SEED = 42
ARMS = ("reserve_p8", "reserve_p12", "reserve_p16")


def main() -> None:
    started = time.time()
    out: dict = {
        "ticket": 628,
        "reading": "does the exact-stagger set move with k_v = n - p",
        "seed": SEED,
        "note": "construction only -- nothing here is trained",
        "by_arm": {},
    }
    # `b42.build_staggered` reads the arm off `b33.ARM`; swap it per sweep rather
    # than copying the builder, so the construction stays B42's unmodified one.
    original = b33.ARM
    try:
        for arm in ARMS:
            b33.ARM = arm
            dome0, _a, _m = b42.build_staggered(SEED, 0)
            width = int(min(int(dome0._permitted[c]) for c in dome0.predicting))
            rows: dict[str, dict] = {}
            print(f"\n=== {arm}  (k_v {width}) ===")
            print(f"{'stagger':>8}  {'ident':>8}  {'chan':>8}  {'sigma':>10}  {'overlap':>8}  {'diff':>7}")
            for stagger in range(width):
                dome, _agent, maps = b42.build_staggered(SEED, stagger)
                wide = b29.cycles_of(dome)["wide"]
                s = b29.surface_read(dome, maps, wide, f"{arm} st{stagger}")["subsets"]["wide"]
                overlap = b42.distinctness(dome, maps)
                rows[str(stagger)] = {
                    "identification": s["identification"]["median"],
                    "channel_return": s["channel_return"]["median"],
                    "sigma_max": s["sigma_max"]["median"],
                    "edge_overlap": overlap,
                    "differentiation": 1.0 - overlap,
                }
                r = rows[str(stagger)]
                print(
                    f"{stagger:>8}  {r['identification']:>8.4f}  {r['channel_return']:>8.4f}"
                    f"  {r['sigma_max']:>10.3e}  {overlap:>8.4f}  {r['differentiation']:>7.4f}"
                )
            exact = sorted(int(k) for k, r in rows.items() if r["sigma_max"] > 0.5)
            best = max(
                (r for r in rows.values() if r["sigma_max"] > 0.5),
                key=lambda r: r["differentiation"],
            )
            out["by_arm"][arm] = {
                "k_v": width,
                "cycles": {k: len(v) for k, v in b29.cycles_of(dome0).items()},
                "stagger": rows,
                "exact_staggers": exact,
                "exact_count": len(exact),
                # The ceiling this `p` puts on differentiation at perfect return.
                "best_exact": best,
            }
            print(f"  exact: {exact}")
            print(
                f"  best differentiation at chan 1.0000: {best['differentiation']:.4f} "
                f"(ident {best['identification']:.4f})"
            )
    finally:
        b33.ARM = original

    out["minutes"] = (time.time() - started) / 60.0
    path = _HERE / "628-stagger-width.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nwrote {path.name} in {out['minutes']:.1f} min")


if __name__ == "__main__":
    main()
