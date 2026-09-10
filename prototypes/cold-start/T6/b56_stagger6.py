"""B56 (#628): is stagger 6 a finding or an artifact?

`605-stagger-seed42.json` carries an unexplained row. Across B42's sweep the
staggers split cleanly into two groups by `sigma_max`:

* `sigma_max` **1.000**, `channel_return` **1.0000** -- staggers 0, 1, 2 **and 6**
* `sigma_max` **~4e-08**, `channel_return` 0.17-0.65 -- staggers 3, 4, 8

Staggers 0, 1 and 2 are the ones B54's argument was built on and their mechanics
are legible: consecutive edges take overlapping row windows, so a hop is a
rectangular identity on the shared rows. Stagger 3, 4 and 8 push the windows
apart, the shared rows run out, and the hop is (numerically) zero.

**Stagger 6 sits in the collapsed group by offset and in the exact group by
reading**, at audience differentiation **0.5444** rather than stagger 1's
**0.3000**. #628 says in as many words that settling this is worth more than
another training arm: it is the difference between ~0.30 and ~0.54 of
differentiation being available at perfect channel return.

Two hypotheses, and they are distinguishable without training anything:

1. **Structural.** The offset lands on a wrap of the `k_v = n - p = 20` window
   that re-aligns the incident edges of the cycles the `wide` basis actually
   uses -- in which case the effect is a *property of the arithmetic* and must
   reproduce on every seed, and other staggers sharing that arithmetic must
   behave the same way.
2. **Artifact.** The particular row assignment happens to align on this dome's
   particular degree sequence and cycle basis, in which case the good set moves
   with the seed.

So: sweep **every** stagger `0..k_v-1` on **three** seeds, and record the row
arithmetic alongside the reading. `distinctness` and the holonomy columns are
B42's own, unmodified; nothing here is trained and nothing here is a term.

Usage::

    PYTHONPATH=src python prototypes/cold-start/T6/b56_stagger6.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from collections import Counter
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


b42 = _load("b56s_b42", _HERE / "b42_stagger.py")
b29 = _load("b56s_b29", _HERE / "b29_holonomy.py")
b33 = b42.b33

SEEDS = (42, 43, 44)


def geometry(dome) -> dict:
    """The row arithmetic, so a pattern in the readings has something to match.

    `k_v` is the permitted window every stagger wraps in, `m_e` the width each
    edge takes out of it, and the degree is how many slots the stagger has to
    walk. If stagger 6 is structural the explanation lives in these three
    numbers and not in the seed.
    """
    kvs = Counter(int(dome._permitted[c]) for c in dome.predicting)
    degs = Counter(len(dome.incident[c]) for c in dome.predicting)
    ms = Counter(int(e.m) for e in dome.edges)
    return {
        "k_v": {str(k): v for k, v in sorted(kvs.items())},
        "degree": {str(k): v for k, v in sorted(degs.items())},
        "edge_m": {str(k): v for k, v in sorted(ms.items())},
    }


def main() -> None:
    started = time.time()
    out: dict = {
        "ticket": 628,
        "reading": "is stagger 6 a finding or an artifact",
        "surface": b33.ARM,
        "seeds": list(SEEDS),
        "note": "construction only -- nothing here is trained",
        "by_seed": {},
    }
    for seed in SEEDS:
        dome0, _agent0, _maps0 = b42.build_staggered(seed, 0)
        width = int(min(int(dome0._permitted[c]) for c in dome0.predicting))
        bases = b29.cycles_of(dome0)
        rows: dict[str, dict] = {}
        out["by_seed"][str(seed)] = {
            "geometry": geometry(dome0),
            "min_k_v": width,
            "cycles": {k: len(v) for k, v in bases.items()},
            "stagger": rows,
        }
        print(f"\n=== seed {seed}  (min k_v {width}, wide {len(bases['wide'])}) ===")
        print(f"{'stagger':>8}  {'ident':>8}  {'chan':>8}  {'sigma':>10}  {'overlap':>8}  {'diff':>7}")
        for stagger in range(width):
            dome, _agent, maps = b42.build_staggered(seed, stagger)
            wide = b29.cycles_of(dome)["wide"]
            s = b29.surface_read(dome, maps, wide, f"s{seed} st{stagger}")["subsets"]["wide"]
            overlap = b42.distinctness(dome, maps)
            row = {
                "identification": s["identification"]["median"],
                "channel_return": s["channel_return"]["median"],
                "sigma_max": s["sigma_max"]["median"],
                "edge_overlap": overlap,
                "differentiation": 1.0 - overlap,
                # The stagger's own arithmetic: how far the last slot walks and
                # whether it wraps. This is what hypothesis 1 would key on.
                "wraps_at_degree": {
                    str(d): int((d - 1) * stagger // width)
                    for d in sorted({len(dome.incident[c]) for c in dome.predicting})
                },
            }
            rows[str(stagger)] = row
            print(
                f"{stagger:>8}  {row['identification']:>8.4f}  {row['channel_return']:>8.4f}"
                f"  {row['sigma_max']:>10.3e}  {overlap:>8.4f}  {row['differentiation']:>7.4f}"
            )
        path = _HERE / "628-stagger-sweep.json"
        out["minutes"] = (time.time() - started) / 60.0
        path.write_text(json.dumps(out, indent=1), encoding="utf-8")

    # The verdict this instrument can give on its own: is the exact set stable?
    exact = {
        seed: sorted(
            int(k)
            for k, r in out["by_seed"][str(seed)]["stagger"].items()
            if r["sigma_max"] > 0.5
        )
        for seed in SEEDS
    }
    out["exact_staggers_by_seed"] = {str(k): v for k, v in exact.items()}
    sets = [set(v) for v in exact.values()]
    out["exact_set_is_seed_stable"] = all(s == sets[0] for s in sets)
    print("\n=== staggers with sigma_max > 0.5 (exact channel, unit gain) ===")
    for seed in SEEDS:
        print(f"  seed {seed}: {exact[seed]}")
    print(f"  seed-stable: {out['exact_set_is_seed_stable']}")
    path = _HERE / "628-stagger-sweep.json"
    out["minutes"] = (time.time() - started) / 60.0
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nwrote {path.name} in {out['minutes']:.1f} min")


if __name__ == "__main__":
    main()
