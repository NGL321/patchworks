"""T3 (#524): full-dome replication of T1's winner, and the canonical table.

Does the apex ladder's winning cell -- `rho = 1` at the drive edges with
`c = 0.1` -- still close the core/apex retention gap on the **full** dome at
100k, where the shallow surface is retired and every claim must live?
`#524 <https://github.com/NGL321/patchworks/issues/524>`_.

**Surface.** Full dome (`DEFAULT_SPEC`, forward normalisation, `interior_m = 3`,
`boundary_m = 4`), `map/cold-start` with `main` merged at the wave-3 boundary,
100k ticks with 30k printed beside it, seeds 42/43/44, frozen world (the arm
under the untrained command, no induced activity at any point -- T0's and T1's
loop verbatim). Two conditions, six runs:

===========  ========================================================
`baseline`   `pin = 0`, `c = 1.0` -- the frozen baseline, re-run here
`winner`     `pin = 1`, `c = 0.1` -- T1's winning cell
===========  ========================================================

**No induced-activity arm is run.** `#522
<https://github.com/NGL321/patchworks/issues/522>`_ found no winning induced
condition, its branch row skipped wave 3, and `#523
<https://github.com/NGL321/patchworks/issues/523>`_ is closed unrun. The
nine-run variant is withdrawn with the structure axis.

**Both arms run to the same horizon on the same seeds**, which is T2's first
cost-of-learning: a comparator borrowed from a shorter run confounds the
intervention with the horizon (its zero-supply control read -0.005 on vision
against T1's 30k baseline from the extra ticks alone). Nothing here is
differenced against stored JSON from another surface; the baseline is a run.

**Done-when (2) is read, not assumed.** T2 observed composed rim-to-apex rank
at 1.078-1.116 against the bar of 1.5 and warned it may be pre-decided -- but
every one of those reads was three hops on the shallow dome, on the *baseline*
build, under an induced supply. Here it is seven hops on the full dome under
T1's winner, and the instrument is a checkpoint-time read that costs the run
nothing. Reading it is cheaper than arguing about it.

**Logged**, the map's canonical table in full, every class published and not
only the ones a guard names (T2's second cost-of-learning: a guard chosen for
one intervention does not transfer to another):

* T0's checkpoint read verbatim -- per-cell `rho` of the raw and used operator,
  `modes_retaining`, `h_bar`, `e_bar`, the P1/P3/P4 alignment and persistence
  reads, the per-cell excitation-rank split -- for soma L1, vision L1, core
  L3-L6 and the apex, named by column and adjacency (#181), never by level;
* T1's drive-edge read -- the standing relative disagreement and the apex-side
  map norm on each drive edge -- so the `rho = 1` arm is seen to null #488's
  gauge artifact on *this* surface rather than assumed to;
* T2's composed rim-to-apex effective rank (#436/#497's construction, Done-when
  (2)'s own quantity) and per-edge excitation rank by stratum, centred and
  uncentred, neither of which T0 or T1 recorded;
* arm travel per window, cumulative and per tick, which is Done-when (2)'s
  other half.

Usage::

    PYTHONPATH=src python prototypes/cold-start/T3/run.py --condition winner --seeds 42
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
_T0 = _HERE.parent / "T0"
_T1 = _HERE.parent / "T1"
_T2 = _HERE.parent / "T2"
for extra in (_ROOT / "prototypes" / "chart-double-duty-166", _ROOT / "benchmarks", _T0):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))


def _load(name: str, path: Path):
    """A sibling rig by path: every rig here is `run.py`, so a bare import is ambiguous."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


t0 = _load("t0_run", _T0 / "run.py")
t1 = _load("t1_run", _T1 / "run.py")
t2 = _load("t2_run", _T2 / "run.py")

from patchworks import agent as agent_module  # noqa: E402
from patchworks.learning import PredictionRule, TransportRule  # noqa: E402
from untrained_fixed_point import build  # noqa: E402

#: T0's ladder with the full-dome horizon on the end, so 30k is printed beside
#: 100k on the same run rather than read off a second, shorter one.
CHECKPOINTS = tuple(t0.CHECKPOINTS) + (100_000,)

#: The two arms. `pin` is T1's `rho = 1` rig flag at the drive edges (applied to
#: a built agent, because #504 lands the change on its own schedule); `c` is
#: `eta_K = c * eta`.
CONDITIONS = {
    "baseline": {"pin": False, "c": 1.0},
    "winner": {"pin": True, "c": 0.1},
}


def run_seed(condition: str, seed: int, ticks: int, out: Path) -> dict:
    started = time.time()
    inflight = t0.stage(out)
    npz_path = out.with_suffix(".npz")
    arm = CONDITIONS[condition]
    pin, c = arm["pin"], arm["c"]
    env, agent = build("real", "train", seed)
    try:
        pairs = t1.drive_pairs(agent)
        pinning = t1.pin_drive_edges(agent) if pin else None
        context = t0.cell_context(agent)
        strata = t2.edge_strata(agent.dome)
        chains = t2.rim_chains(agent.dome)
        record = {
            "issue": 524,
            "condition": condition,
            "arm": {"rho1_drive_edges": bool(pin), "c": float(c)},
            "dome": "real",
            "split": "train",
            "seed": seed,
            "ticks": ticks,
            "surface": t0.surface(),
            "cells": int(agent.sheaf.operators.cells),
            "k": int(agent.sheaf.operators.shape.k),
            "n": int(agent.dome.shape.n),
            "band": [1.0 / agent.sheaf.operators.rho_k, 1.0],
            "drive_assertion": float(agent_module.DRIVE_ASSERTION),
            "pinning": pinning,
            "window": t0.WINDOW,
            "block": t0.BLOCK,
            "context": context,
            "groups": t0.groups(context),
            "edge_strata": strata,
            "chains": chains,
            "checkpoints": [],
        }
        recorder = t2.EdgeRecorder(agent)
        bias = PredictionRule(agent.sheaf, operator_rate_ratio=c)
        transport = TransportRule(agent.sheaf)
        record["operator_rate"] = float(bias.operator_rate)
        record["learning_rate"] = float(bias.learning_rate)
        # The untrained surface's composed rank: the floor Done-when (2)'s
        # reading is above or not, on this dome and before a tick is run.
        record["composed_at_construction"] = t2.composed_reads(agent, chains)["by_kind"]
        arrays: dict[str, np.ndarray] = {}

        ladder = [cp for cp in CHECKPOINTS if cp <= ticks]
        if ticks not in ladder:
            ladder.append(ticks)
        seen = 0
        carry = None
        travel_total = 0.0
        for target in ladder:
            window = target - seen
            poses = [] if carry is None else [carry]
            for outcome in t0.teaching_read(agent, window, seed + seen, recorder, bias, transport):
                poses.append(outcome.observation["qpos"].copy())
            pose = np.asarray(poses)
            window_travel = float(np.abs(np.diff(pose, axis=0)).sum()) if len(pose) > 1 else 0.0
            travel_total += window_travel
            carry = poses[-1]
            seen = target

            entry, arr = t0.checkpoint_read(agent, recorder, bias, context)
            entry["ticks"] = target
            entry["travel_window"] = window_travel
            entry["travel_cumulative"] = travel_total
            entry["travel_per_tick"] = window_travel / max(1, window)
            entry["drive_edges"] = t1.drive_edge_read(agent, pairs)
            entry["composed"] = t2.composed_reads(agent, chains)
            entry["edges"] = t2.edge_reads(agent, recorder, strata)
            if pin:
                # The flag's own invariant, re-read at every checkpoint.
                norms = agent.sheaf.maps.norms().detach()[torch.tensor(pinning["pairs"])]
                entry["pinned_norms"] = [float(x) for x in norms]
            record["checkpoints"].append(entry)
            record["elapsed_minutes"] = (time.time() - started) / 60.0
            for key, val in arr.items():
                arrays[f"t{target}_{key}"] = val
            arrays[f"t{target}_composed_er"] = np.array(entry["composed"]["effective_rank"])
            record["blocks"] = [
                {
                    "end_tick": b["end_tick"],
                    **{
                        key: [float(x) for x in b[key]]
                        for key in b
                        if key not in ("end_tick", "mean_h", "mean_e")
                    },
                }
                for b in recorder.blocks
            ]
            if recorder.blocks:
                arrays["block_end_ticks"] = np.array([b["end_tick"] for b in recorder.blocks])
                arrays["block_mean_e"] = np.stack([b["mean_e"].numpy() for b in recorder.blocks])
                arrays["block_mean_h"] = np.stack([b["mean_h"].numpy() for b in recorder.blocks])

            # Written the moment it lands, to the staged name: a kill costs the
            # run in flight and nothing that came before it.
            inflight.write_text(json.dumps(record, indent=1))
            np.savez_compressed(npz_path, **arrays)

            g = record["groups"]
            pc = entry["per_cell"]
            rho = np.array(pc["rho_used"])
            rel = np.array(entry["drive_edges"]["relative"])
            comp = entry["composed"]["by_kind"]
            comp_er = np.array(entry["composed"]["effective_rank"])
            print(
                f"  {condition} seed {seed} @ {target:>6}: rho_used apex {np.median(rho[g['apex']]):.3f} "
                f"core {np.median(rho[g['core']]):.3f} vis {np.median(rho[g['vision']]):.3f} "
                f"soma {np.median(rho[g['soma']]):.3f} | gap {np.median(rho[g['core']]) - np.median(rho[g['apex']]):+.3f} | "
                f"drive rel-dis {np.median(rel):.3f} | composed ER {np.median(comp_er):.3f} "
                f"({len(comp)} kinds) | travel/tick {entry['travel_per_tick']:.3e} | "
                f"dead {int((np.asarray(entry['used']['per_cell']['modes_retaining']) == 0).sum())} "
                f"({record['elapsed_minutes']:.1f} min)",
                flush=True,
            )
        inflight.replace(out)
        return record
    finally:
        env.close()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--condition", choices=sorted(CONDITIONS), nargs="+", default=["baseline", "winner"])
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--ticks", type=int, default=100_000)
    args = p.parse_args()
    for condition in args.condition:
        for seed in args.seeds:
            out = _HERE / f"524-{condition}-seed{seed}-{args.ticks}.json"
            if out.exists():
                print(f"[T3] {out.name} already at the horizon, skipping", flush=True)
                continue
            print(f"[T3] {condition} seed {seed}, {args.ticks} ticks -> {out.name}", flush=True)
            run_seed(condition, seed, args.ticks, out)
            print(f"  wrote {out.name}", flush=True)


if __name__ == "__main__":
    main()
