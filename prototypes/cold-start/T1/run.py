"""T1 (#521): the apex ladder, a 2x2 on the frozen world.

Do the two already-licensed rungs of the apex ladder move apex `rho(K)` toward
core's, and by how much? `#521 <https://github.com/NGL321/patchworks/issues/521>`_.

**Surface.** The shallow dome (three levels: an 8x8 rim, one core level of 16,
an apex of 8 -- `SHALLOW` below, and see its note), `map/cold-start` at HEAD
with `main` merged at the wave-1 boundary, frozen world (the arm under the
untrained command, no induced activity -- T0's loop verbatim), 30k ticks,
seeds 42/43/44. Conditions: `{rho = 1 at the drive edges: off, on}` x
`{c in {1.0, 0.1}}`, twelve runs.

**The two rungs, as rig flags.**

* ``--pin 1`` pins the **apex-side** map of each drive edge at the exact
  gauge, so both ends of a drive edge carry `rho = 1` (#488's ruling, #504's
  build), implemented as `#502
  <https://github.com/NGL321/patchworks/issues/502>`_ ruled and not as a
  shortcut: `pinned` is widened, so the gauge bounds, `hold_pairs` and what
  `_push_apart` corrects move together, and the projection's `overlap_target`
  at the eight apex cells becomes `gain_denominators - |P_v|` -- the reduced
  target on the held subset, derived from the one denominator the gain divides
  by, which does not move (`Sheaf.gain` is built from `gain_denominators` at
  construction and is not touched). A rig flag applied to a built agent rather
  than a change to `restriction.py`, because #504 lands the change on its own
  schedule and this ticket reads the mechanism, not the build.
* ``--c`` is `eta_K = c * eta` (`DEFAULT_OPERATOR_RATE_RATIO`, ADR-0008's
  construction constant with a stated retune duty), passed straight to
  `PredictionRule`.

**What is logged** is T0's checkpoint read verbatim (`rho` of the used and raw
operator, `modes_retaining`, `h_bar`, `e_bar`, the P1/P3/P4 alignment,
persistence and decomposition reads, the excitation-rank split), plus what the
`rho = 1` arm is for: the standing disagreement on each drive edge and the
apex-side drive map's norm, so the flag is seen to null #488's 1/3 rather than
assumed to.

Usage::

    PYTHONPATH=src python prototypes/cold-start/T1/run.py --pin 0 --c 1.0 --seeds 42 43 44
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
for extra in (_ROOT / "prototypes" / "chart-double-duty-166", _ROOT / "benchmarks", _T0):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))


def _load_t0():
    """T0's rig, by path: both files are `run.py`, so a bare import would find this one."""
    spec = importlib.util.spec_from_file_location("t0_run", _T0 / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


t0 = _load_t0()

from patchworks import agent as agent_module  # noqa: E402
from patchworks.agent import Agent  # noqa: E402
from patchworks.graph import CellKind, DomeSpec, build_graph  # noqa: E402
from patchworks.learning import PredictionRule, TransportRule  # noqa: E402
from patchworks.restriction import gain_denominators, pair_index  # noqa: E402
from patchworks.tick import reconciliation_gain  # noqa: E402
from patchworks.sandbox import PlanarPushSandbox  # noqa: E402
from untrained_fixed_point import IMAGE_SIZE  # noqa: E402

CHECKPOINTS = t0.CHECKPOINTS

#: The shallow diagnostic dome, #517's guard 1: three levels, never two. An
#: 8x8 rim over the 16x16 tiling, one core level of 16, an apex of 8 -- the
#: counts the map names.
#:
#: **`core_degree` is 7 here and not the default 6, because the map's spec
#: does not build.** With one vision lattice the 64 L1 cells land four to a
#: core cell, the six somatomotor cells add a fifth to six of them, and every
#: core cell carries two up-edges to the apex: 6 or 7 vertical edges against a
#: target of 6, and `build_graph` refuses an overshoot ("no edge is ever
#: removed"). 7 is the smallest target that admits the vertical edges the taper
#: forces. The **apex is untouched**: `apex_degree` stays 4, so each apex cell
#: has four down-edges of `m = 3` and one drive edge of `m = 1`, a 19-private /
#: 13-exposed stalk split identical to the full dome's -- the cell under study
#: is the same object. The core comparator's guaranteed private dimension is
#: 11 (`32 - 7 * 3`) against the full dome's 14. Ledger row on #520.
SHALLOW = DomeSpec(
    vision_sides=(8,), somatomotor_sizes=(6,), core_sizes=(16, 8), core_degree=7
)


def build_shallow(seed: int, split: str = "train") -> tuple[PlanarPushSandbox, Agent]:
    env = PlanarPushSandbox(split=split, image_size=IMAGE_SIZE[SHALLOW.patch_grid])
    agent = Agent(
        env, dome=build_graph(SHALLOW), generator=torch.Generator().manual_seed(seed)
    )
    return env, agent


def drive_pairs(agent) -> list[tuple[int, int, int]]:
    """`(edge_id, apex-side pair index, apex cell id)` for every drive edge."""
    dome = agent.dome
    drive_cells = {c.id for c in dome.cells if c.kind == CellKind.DRIVE}
    out = []
    for edge in dome.edges:
        for side, cell_id in enumerate((edge.u, edge.v)):
            if cell_id not in drive_cells and edge.other(cell_id) in drive_cells:
                out.append((edge.id, pair_index(edge.id, side), cell_id))
    return out


@torch.no_grad()
def pin_drive_edges(agent) -> dict:
    """The `rho = 1` rig flag: the apex-side drive maps carry the exact gauge.

    Per #502's ruling, `pinned` is widened and the target re-derived in one
    place: `overlap_target = gain_denominators - pinned incidence`, so the
    held subset at a partly-pinned cell spends what is left of the same budget
    and Weyl returns the cap to `g_v^2 c_v`. The gain's own denominator is not
    touched. Returns what was pinned, for the record.
    """
    maps = agent.sheaf.maps
    dome = agent.dome
    pairs = drive_pairs(agent)
    idx = torch.tensor([p for _e, p, _c in pairs], dtype=torch.long)
    maps.pinned[idx] = True
    maps.hold_pairs = (~maps.pinned).nonzero().squeeze(-1)
    pinned_count = torch.zeros(len(dome.cells), dtype=torch.float32)
    pinned_count.index_add_(
        0, maps.owner[maps.pinned], torch.ones(int(maps.pinned.sum()), dtype=torch.float32)
    )
    denominators = gain_denominators(dome, rho=maps.rho)
    maps.overlap_target = denominators - pinned_count * maps.holding.to(torch.float32)
    maps.project()
    norms = maps.norms()[idx]
    if not torch.allclose(norms, torch.ones_like(norms), atol=1e-6):
        raise RuntimeError(f"pinned drive maps are not at the exact gauge: {norms}")
    recomputed = reconciliation_gain(dome, gamma=agent.sheaf.gamma, rho=maps.rho)
    return {
        "pairs": [int(p) for _e, p, _c in pairs],
        "edges": [int(e) for e, _p, _c in pairs],
        "cells": [int(c) for _e, _p, c in pairs],
        "overlap_target_apex": [float(maps.overlap_target[c]) for _e, _p, c in pairs],
        "gain_denominator_apex": [float(denominators[c]) for _e, _p, c in pairs],
        "hold_pairs": int(maps.hold_pairs.numel()),
        "pinned": int(maps.pinned.sum()),
        "gain_unchanged": bool(torch.equal(recomputed, agent.sheaf.gain)),
    }


def groups(context: dict) -> dict[str, list[int]]:
    """The shallow dome's classes, by column and adjacency (#181), as row indices.

    On three levels the core comparator is the one core level, so it is named
    by column and *not* drive-adjacent rather than by a level range.
    """
    cols, lv, drv, bnd = (
        context["columns"], context["levels"], context["drive_adjacent"], context["boundary_adjacent"]
    )
    out = {"soma": [], "vision": [], "apex": [], "core": []}
    for i, (c, l, d, b) in enumerate(zip(cols, lv, drv, bnd)):
        if c == "somatomotor" and l == 1 and b > 0:
            out["soma"].append(i)
        elif c == "vision" and l == 1:
            out["vision"].append(i)
        elif c == "core" and d:
            out["apex"].append(i)
        elif c == "core":
            out["core"].append(i)
    return out


@torch.no_grad()
def drive_edge_read(agent, pairs: list[tuple[int, int, int]]) -> dict:
    """The standing disagreement on each drive edge, and the apex-side map's norm.

    `Sheaf.disagreement()` is `u`'s restriction minus `v`'s on the edge's one
    lane; the relative form is #488's, `|d| / (|a| + |b|)` over the two ends'
    broadcasts, so a nulled edge reads 0 and a fixed 2x scale mismatch reads 1/3.
    """
    sheaf = agent.sheaf
    dis = sheaf.disagreement()
    ends = sheaf.broadcast.reshape(-1, 2, sheaf.maps.edge_width)
    norms = sheaf.maps.norms()
    out = {"edge": [], "cell": [], "disagreement": [], "relative": [], "apex_broadcast": [], "drive_broadcast": [], "apex_map_norm": []}
    for edge_id, pair, cell in pairs:
        edge = agent.dome.edges[edge_id]
        side = 0 if edge.u == cell else 1
        a = float(ends[edge_id, side, 0])
        b = float(ends[edge_id, 1 - side, 0])
        d = float(dis[edge_id, 0])
        out["edge"].append(int(edge_id))
        out["cell"].append(int(cell))
        out["disagreement"].append(d)
        out["relative"].append(abs(d) / max(abs(a) + abs(b), 1e-12))
        out["apex_broadcast"].append(a)
        out["drive_broadcast"].append(b)
        out["apex_map_norm"].append(float(norms[pair]))
    return out


def run_seed(pin: bool, c: float, seed: int, ticks: int, out: Path) -> dict:
    started = time.time()
    inflight = t0.stage(out)
    npz_path = out.with_suffix(".npz")
    env, agent = build_shallow(seed)
    try:
        pairs = drive_pairs(agent)
        pinning = pin_drive_edges(agent) if pin else None
        context = t0.cell_context(agent)
        record = {
            "issue": 521,
            "condition": {"rho1_drive_edges": bool(pin), "c": float(c)},
            "dome": "shallow",
            "dome_spec": {
                "vision_sides": list(SHALLOW.vision_sides),
                "somatomotor_sizes": list(SHALLOW.somatomotor_sizes),
                "core_sizes": list(SHALLOW.core_sizes),
                "core_degree": SHALLOW.core_degree,
                "apex_degree": SHALLOW.apex_degree,
                "interior_m": SHALLOW.interior_m,
                "boundary_m": SHALLOW.boundary_m,
                "cells": len(agent.dome.cells),
                "edges": len(agent.dome.edges),
            },
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
            "groups": groups(context),
            "checkpoints": [],
        }
        recorder = t0.Recorder(agent)
        bias = PredictionRule(agent.sheaf, operator_rate_ratio=c)
        transport = TransportRule(agent.sheaf)
        record["operator_rate"] = float(bias.operator_rate)
        record["learning_rate"] = float(bias.learning_rate)
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
            entry["drive_edges"] = drive_edge_read(agent, pairs)
            if pin:
                # The flag's own invariant, re-read at every checkpoint.
                norms = agent.sheaf.maps.norms().detach()[torch.tensor(pinning["pairs"])]
                entry["pinned_norms"] = [float(x) for x in norms]
            record["checkpoints"].append(entry)
            record["elapsed_minutes"] = (time.time() - started) / 60.0
            for key, val in arr.items():
                arrays[f"t{target}_{key}"] = val
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

            inflight.write_text(json.dumps(record, indent=1))
            np.savez_compressed(npz_path, **arrays)

            g = record["groups"]
            pc = entry["per_cell"]
            rho = np.array(pc["rho_used"])
            rel = np.array(entry["drive_edges"]["relative"])
            print(
                f"  seed {seed} @ {target:>6}: rho_used apex {np.median(rho[g['apex']]):.3f} "
                f"core {np.median(rho[g['core']]):.3f} vis {np.median(rho[g['vision']]):.3f} "
                f"soma {np.median(rho[g['soma']]):.3f} | "
                f"|e| apex {np.median(np.array(pc['p3_ebar_norm'])[g['apex']]):.2e} "
                f"stab {np.nanmedian(np.array(pc['p3_direction_stability'])[g['apex']]):.2f} | "
                f"drive rel-dis {np.median(rel):.3f} | "
                f"dead {int((np.asarray(entry['used']['per_cell']['modes_retaining']) == 0).sum())} "
                f"({record['elapsed_minutes']:.1f} min)",
                flush=True,
            )
        inflight.replace(out)
        return record
    finally:
        env.close()


def condition_name(pin: bool, c: float) -> str:
    return f"rho1{'on' if pin else 'off'}-c{c:g}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pin", type=int, choices=(0, 1), required=True, help="rho = 1 at the drive edges")
    p.add_argument("--c", type=float, required=True, help="eta_K = c * eta")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--ticks", type=int, default=30_000)
    args = p.parse_args()
    for seed in args.seeds:
        out = _HERE / f"521-{condition_name(bool(args.pin), args.c)}-seed{seed}-{args.ticks}.json"
        print(f"[T1] {condition_name(bool(args.pin), args.c)} seed {seed}, {args.ticks} ticks -> {out.name}", flush=True)
        run_seed(bool(args.pin), args.c, seed, args.ticks, out)
        print(f"  wrote {out.name}", flush=True)


if __name__ == "__main__":
    main()
