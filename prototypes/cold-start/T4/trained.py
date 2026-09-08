"""T4 (#537): the angle read on the **trained** surface, and whether the `c` cap bites there.

`#537 <https://github.com/NGL321/patchworks/issues/537>`_ says "nothing here
needs building or running -- the surfaces are already on disk in
`prototypes/cold-start/T3/`". **That is not true of the maps.** T3's npz holds
the chart operators `K`, the per-cell reads and the per-chain composed effective
rank; it does not hold `agent.sheaf.maps`, and the angle read is a function of
exactly those. So the trained arm of items 1-3, and the whole of item 5 on a
learned surface, need the surface rebuilt.

**Why it is worth the ticks rather than being waved off.** At construction the
incoherence cap is slack -- it binds at 4 of 414 measured cells -- so `c` cannot
move anything there whatever value it takes, and `angles.py` reads its sweep
flat. But `#439 <https://github.com/NGL321/patchworks/issues/439>`_ found the
actuator's maps drifting **into** coherence with training, reaching 99.6% of the
`g_v^2 * deg` ceiling by 100k taught ticks. If the cap comes to bite once the
transport rule has run, `c` is a live constraint on the trained surface even
though it is a dead one at construction, and item 5's answer would differ
between the two. That is the one reading that could overturn "`c` alone buys
nothing", so it is read rather than argued.

Same loop as T3 (#524) verbatim -- `t0.teaching_read`, T3's two arms, the frozen
world, T0's checkpoint ladder -- with the angle read and the cap read added at
each checkpoint and nothing else changed. Written at every checkpoint, so a kill
costs the checkpoint in flight and nothing before it.

Usage::

    PYTHONPATH=src python prototypes/cold-start/T4/trained.py --condition baseline --seeds 42
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
_T0, _T1, _T2, _T3 = (_HERE.parent / n for n in ("T0", "T1", "T2", "T3"))
for extra in (_ROOT / "prototypes" / "chart-double-duty-166", _ROOT / "benchmarks", _T0):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


t0 = _load("t0_run", _T0 / "run.py")
t1 = _load("t1_run", _T1 / "run.py")
t2 = _load("t2_run", _T2 / "run.py")
t3 = _load("t3_run", _T3 / "run.py")
angles = _load("t4_angles", _HERE / "angles.py")

from patchworks.learning import PredictionRule, TransportRule  # noqa: E402
from untrained_fixed_point import build  # noqa: E402


def cap_read(agent) -> dict:
    """`lambda_max(sum_e F^T F) / (g_v^2 c_v)` over the cells the cap can reach.

    The whole of item 5 on a trained surface: a cap that never approaches 1 is a
    constraint `c` cannot be relaxed *off*, because nothing is resting on it.
    """
    maps = agent.sheaf.maps
    peaks = maps.gram_peaks().numpy()
    target = maps.overlap_target.numpy()
    live = target > 0
    ratio = peaks[live] / target[live]
    return {
        "cells_measured": int(live.sum()),
        "ratio_median": float(np.median(ratio)),
        "ratio_p90": float(np.quantile(ratio, 0.90)),
        "ratio_max": float(ratio.max()),
        "cells_at_cap": int((ratio > 1 - 1e-6).sum()),
        "cells_over_half": int((ratio > 0.5).sum()),
    }


def run_seed(condition: str, seed: int, ticks: int, out: Path, c_values, m_values) -> dict:
    started = time.time()
    inflight = out.with_suffix(".inflight.json")
    arm = t3.CONDITIONS[condition]
    pin, c = arm["pin"], arm["c"]
    env, agent = build("real", "train", seed)
    try:
        if pin:
            # T1's `rho = 1` rig flag, applied to a built agent for its side
            # effect on the drive-edge maps; #504 lands the change properly on
            # its own schedule.
            t1.pin_drive_edges(agent)
        chains = t2.rim_chains(agent.dome)
        record = {
            "issue": 537,
            "condition": condition,
            "arm": {"rho1_drive_edges": bool(pin), "c_learning_rate": float(c)},
            "note": (
                "`c` in `arm` is T3's learning-rate ratio `eta_K = c * eta`, NOT "
                "ADR-0010's incoherence constant `GAUGE_C`. Two different `c`s; "
                "#537's lever is the second one, swept in `sweep_c`."
            ),
            "seed": seed,
            "ticks": ticks,
            "surface": t0.surface(),
            "k": int(agent.sheaf.operators.shape.k),
            "n": int(agent.dome.shape.n),
            "chains": chains,
            "checkpoints": [],
        }
        recorder = t2.EdgeRecorder(agent)
        bias = PredictionRule(agent.sheaf, operator_rate_ratio=c)
        transport = TransportRule(agent.sheaf)

        record["at_construction"] = {
            "angles": angles.read_surface(agent, chains, f"construction {condition} seed {seed}"),
            "cap": cap_read(agent),
        }

        ladder = [cp for cp in t3.CHECKPOINTS if cp <= ticks]
        if ticks not in ladder:
            ladder.append(ticks)
        seen = 0
        for target in ladder:
            window = target - seen
            for _ in t0.teaching_read(agent, window, seed + seen, recorder, bias, transport):
                pass
            seen = target

            entry = {"ticks": target}
            entry["angles"] = angles.read_surface(agent, chains, f"{condition} seed {seed} @{target}")
            entry["cap"] = cap_read(agent)
            entry["sweep_c"] = angles.sweep_c(agent, chains, c_values)
            entry["sweep_m"] = angles.sweep_m(agent.dome, chains, m_values)
            entry["elapsed_minutes"] = (time.time() - started) / 60.0
            record["checkpoints"].append(entry)
            inflight.write_text(json.dumps(record, indent=1))

            a = entry["angles"]
            print(
                f"  {condition} s{seed} @{target:>6}: ER med {a['composed_er']['median']:.4f} "
                f"max {a['composed_er']['max']:.4f} | cos med {a['cos_all']['median']:.3f} "
                f"| sig_min/max med {a['sigma_min_over_max']['median']:.6f} "
                f"| cap max {entry['cap']['ratio_max']:.3f} at-cap {entry['cap']['cells_at_cap']} "
                f"| c-sweep {'flat' if len({round(r['er_median'], 6) for r in entry['sweep_c']}) == 1 else 'MOVES'} "
                f"({entry['elapsed_minutes']:.1f} min)",
                flush=True,
            )
        inflight.replace(out)
        return record
    finally:
        env.close()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--condition", choices=sorted(t3.CONDITIONS), nargs="+", default=["baseline", "winner"])
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--ticks", type=int, default=100_000)
    p.add_argument("--c-values", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8, 12])
    p.add_argument("--m-values", type=int, nargs="+", default=[3, 4, 6, 8, 12])
    args = p.parse_args()
    for condition in args.condition:
        for seed in args.seeds:
            out = _HERE / f"537-{condition}-seed{seed}-{args.ticks}.json"
            if out.exists():
                print(f"[T4] {out.name} already at the horizon, skipping", flush=True)
                continue
            print(f"[T4] {condition} seed {seed}, {args.ticks} ticks -> {out.name}", flush=True)
            run_seed(condition, seed, args.ticks, out, args.c_values, args.m_values)
            print(f"  wrote {out.name}", flush=True)


if __name__ == "__main__":
    main()
