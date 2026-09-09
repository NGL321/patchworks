"""B43 (#607): clauses 2 and 3 of the adoption conjunction, read on both masks.

The ticket pre-registers a **conjunction** on B35, all three clauses or nothing:

1. `world_loop(c)` falls on the relayed mask — free, no run, and `b43_relay.py`
   has it.
2. **`τ̂_c` holds.** The measured `τ` is read off `ρ(K · (J_chart + J_stalk · A_v
   · D))` and `A_v` is the adjacency, which a relay moves. A check, not a threat.
3. **ADR-0021's bottleneck amplitude rises.** This is the **anti-Goodhart
   clause**: ADR-0026's bar carries loop length and ADR-0021's does not, so a
   relay that shortens the measuring stick while the bottleneck sits still has
   bought nothing and has bought it dishonestly ([B32](#591)'s steganographic
   failure in topological form).

Both are read here with `benchmarks/detectability.py`'s own instruments — `trial`
unchanged, which is what makes the two masks commensurable.

**The baseline is taken in this session, not quoted.** B38's rim→apex bottleneck
of **8.35e-10** is at 30,000 ticks; running the relayed mask at a shorter horizon
and comparing it to that number would be comparing two horizons and calling the
difference a relay. So `none` runs beside `pair` at the *same* horizon, the same
seed and the same trial rotation, and the contrast is internal. B38's figure is
quoted only as the scale the reading should land near.

**The horizon is stamped, never inherited.** [B38](#599) found the motion horizon
is per-run and varies 13x between seeds of one arm, so every run here reads
`b33.Motion` over its own last window and records it beside its numbers.

**What is *not* read: a max-min-over-paths conduction verdict.** [B39](#601)
struck ADR-0026's path quantifiers and this map forbids quoting one, B38's §4
absolutes included. `Trial.conduction` is computed by the shipped instrument and
is recorded in the JSON for provenance, but the reading taken is the **per-cell**
`τ̂_c / world_loop(c)` distribution, which B39 left untouched.

Usage::

    PYTHONPATH=src python prototypes/cold-start/T6/b43_detect.py run --layouts none pair
    PYTHONPATH=src python prototypes/cold-start/T6/b43_detect.py run --learn 2000 --trials 3
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
for extra in (_ROOT / "prototypes" / "chart-double-duty-166", _ROOT / "benchmarks"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))


def _load(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


b33 = _load("b43d_b33", _HERE / "b33_coexist.py")
b43r = _load("b43d_relay", _HERE / "b43_relay.py")

import detectability as det  # noqa: E402
import untrained_fixed_point as ufp  # noqa: E402
import loop_length as ll  # noqa: E402
from patchworks.agent import Agent  # noqa: E402

#: Training horizon. **Not 30,000.** B38 measured the world's own motion stopping
#: by tick ~125 on this arm and found the horizon per-run rather than per-arm, so
#: a long run buys a longer stall rather than a better-trained surface. 2,000 is
#: B33's horizon, it is where this map's contrast was read, and both masks get it.
LEARN = 2000

#: Trials per direction. Small by design: the contrast is between two masks run
#: at one seed with one rotation, and the per-trial spread is reported rather
#: than averaged away.
TRIALS = 3


def build_agent(layout: str, sites: int, seed: int, depth: int, relay_m: int):
    """The arm's env and an agent on the relayed mask, or the shipped one."""
    env, agent = b33.arms_mod.build_arm(b33.ARM, seed)
    base = agent.dome
    if layout == "none":
        return env, agent, base, [], {"layout": "none", "relays": 0}
    relays, record = b43r.lay_relays(base, layout, depth, relay_m, sites)
    dome = b43r.relayed(base, relays)
    relay_ids = list(range(len(base.edges), len(dome.edges)))
    agent = Agent(env, dome=dome, generator=torch.Generator().manual_seed(seed))
    return env, agent, base, relay_ids, record


def per_cell_ratio(dome, tau: np.ndarray) -> dict:
    """Clause 2: `τ̂_c / world_loop(c)` per cell, which B39 left standing.

    The path quantifier is struck and not reported. What ADR-0026 still says
    per cell is this ratio, and its distribution over the outbound population is
    what a relay is allowed to move: `world_loop(c)` is its divisor and the relay
    shortens it, so the ratio should rise even at unchanged `τ̂` — which is
    exactly the shortening-the-stick that clause 3 exists to catch.
    """
    loops = ll.world_loops(dome).lengths
    cells = det.population(dome)
    vals, taus = [], []
    for row, cell in enumerate(cells):
        if cell not in loops:
            continue
        vals.append(tau[row] / loops[cell])
        taus.append(tau[row])
    v = np.array(vals, dtype=float)
    t = np.array(taus, dtype=float)
    finite = v[np.isfinite(v)]
    return {
        "cells": int(v.size),
        "ratio_median": float(np.median(finite)) if finite.size else float("nan"),
        "ratio_max": float(finite.max()) if finite.size else float("nan"),
        "ratio_mean": float(finite.mean()) if finite.size else float("nan"),
        "ratio_above_one": int((finite >= 1.0).sum()),
        "tau_median": float(np.median(t)) if t.size else float("nan"),
        "tau_max": float(t.max()) if t.size else float("nan"),
        "tau_zero_cells": int((t == 0).sum()),
    }


def run_one(
    layout: str,
    sites: int,
    seed: int,
    learn: int,
    trials: int,
    window: int,
    hold: int,
    depth: int,
    relay_m: int,
    out: Path,
) -> dict:
    started = time.time()
    env, agent, base, relay_ids, layout_record = build_agent(
        layout, sites, seed, depth, relay_m
    )
    dome = agent.dome
    motion = b33.Motion(env)

    print(
        f"[B43] {layout} sites={sites}: {len(dome.cells)} cells, "
        f"{len(dome.edges)} edges (+{len(relay_ids)} relays); "
        f"training {learn} ticks",
        flush=True,
    )
    # Train with both rules, exactly as `detectability.prepared` does, and stamp
    # the motion over the last window rather than inheriting a horizon.
    for i, _ in enumerate(ufp.teaching(agent, learn, seed)):
        if i >= learn - b33.READ_WINDOW:
            motion.observe()
    stamped = motion.read()
    cast = det.double_precision(agent.sheaf)

    # The cohort: the cells that were deepest **before** any relay was laid, by
    # `d(c, rim)` off the baseline mask. `detectability.apex` reads
    # `CellIndex.level`, which this map forbids appealing to; the two sets are
    # compared and the depth-defined one is used.
    base_depth = b43r.depth_from_rim(base)
    dmax = max(base_depth[c] for c in base.predicting)
    cohort = tuple(sorted(c for c in base.predicting if base_depth[c] >= dmax))
    by_level = tuple(sorted(det.apex(base)))

    picker = np.random.default_rng(seed)
    picks = det.sources(dome, "rim-to-apex", picker, trials)
    generator = torch.Generator().manual_seed(seed + 1)

    record = {
        "issue": 607,
        "reading": "clauses 2 and 3 on the relayed mask",
        "layout": layout,
        "sites": sites,
        "seed": seed,
        "learn": learn,
        "trials": trials,
        "window": window,
        "hold": hold,
        "relay_m": relay_m,
        "depth_floor": depth,
        "layout_record": layout_record,
        "relays": len(relay_ids),
        "edges": len(dome.edges),
        "motion": stamped,
        "read_window": len(motion.buffer),
        "double_precision_tensors": cast,
        "cohort": [int(c) for c in cohort],
        "cohort_matches_level_apex": sorted(cohort) == list(by_level),
        "world_loop_cohort_median": float(
            np.median([ll.world_loops(dome).lengths[c] for c in cohort])
        ),
        "b38_baseline_bottleneck_at_30k": 8.35e-10,
        "trials_out": [],
    }

    for i in range(trials):
        observation, _info = env.reset(seed=seed * 1000 + i)
        agent.observe(observation)
        applied = np.zeros(env.action_space.shape, dtype=np.float64)
        det.hold_still(agent, observation, applied, hold)
        t = det.trial(
            agent, observation, applied, picks[i], cohort, generator, window, det.PROBE
        )
        entry = {
            "trial": i,
            "source": [int(c) for c in t.source],
            "kind": t.kind,
            # Clause 3: ADR-0021's amplitude, which carries no loop length.
            "bottleneck": float(t.bottleneck),
            "target": int(t.target),
            "edge": det.name_edge(dome, int(t.edge)),
            "path_hops": len(t.path),
            # Clause 2, per cell. The path verdict is recorded for provenance and
            # is **not** the reading: B39 struck its quantifier.
            "per_cell": per_cell_ratio(dome, t.tau),
            "conduction_path_value_provenance_only": float(t.conduction),
            "censored_cells": int(np.sum(t.censored)),
            "readable_cells": int(np.sum(t.resolved)),
            "floor": float(t.floor),
            "elapsed_minutes": (time.time() - started) / 60.0,
        }
        record["trials_out"].append(entry)
        pc = entry["per_cell"]
        print(
            f"  trial {i + 1}/{trials} [{t.kind}] bottleneck {t.bottleneck:.3e} "
            f"| tau med {pc['tau_median']:.3e} "
            f"| ratio med {pc['ratio_median']:.3e} "
            f"max {pc['ratio_max']:.3e} >=1 {pc['ratio_above_one']}",
            flush=True,
        )
        out.write_text(json.dumps(record, indent=1), encoding="utf-8")

    bn = np.array([e["bottleneck"] for e in record["trials_out"]], dtype=float)
    record["bottleneck_median"] = float(np.median(bn))
    record["bottleneck_max"] = float(bn.max())
    record["minutes"] = (time.time() - started) / 60.0
    out.write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(
        f"[B43] {layout}: bottleneck median {record['bottleneck_median']:.3e} "
        f"in {record['minutes']:.1f} min -> {out.name}",
        flush=True,
    )
    return record


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["run"])
    p.add_argument("--layouts", nargs="+", default=["none", "motor", "pair"])
    p.add_argument("--sites", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--learn", type=int, default=LEARN)
    p.add_argument("--trials", type=int, default=TRIALS)
    p.add_argument("--window", type=int, default=det.WINDOW)
    p.add_argument("--hold", type=int, default=det.HOLD)
    p.add_argument("--depth", type=int, default=b43r.DEPTH)
    p.add_argument("--relay-m", type=int, default=b43r.RELAY_M)
    args = p.parse_args()

    for layout in args.layouts:
        sites = 0 if layout == "none" else args.sites
        out = _HERE / f"607-detect-{layout}{sites}-seed{args.seed}-{args.learn}.json"
        if out.exists():
            print(f"[B43] {out.name} already taken, skipping", flush=True)
            continue
        run_one(
            layout,
            sites,
            args.seed,
            args.learn,
            args.trials,
            args.window,
            args.hold,
            args.depth,
            args.relay_m,
            out,
        )


if __name__ == "__main__":
    main()
