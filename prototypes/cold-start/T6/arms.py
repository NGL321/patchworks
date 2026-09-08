"""T6 (#555): the two arms [B12/#556](#556) ruled, built on the **shipped** spec.

#555 was opened to build [#540](#540)'s stack. While it was being worked,
[B8/#548](#548) shipped levers (d) and (c1) into `src/patchworks/graph.py` —
`allocate_lane_widths`, `DomeSpec.lateral_m`, `DomeSpec.privacy_budget` — and
**held** lever (b), the doubled invariant, because at budget 63 the guaranteed
private dimension `max(0, n − Σ_e m_e)` reads zero at most cells. #556 then ruled
that collision, and its ruling turns this ticket into **two arms**:

* **doubling** — `privacy_budget = 63`, mask unchanged (`k_v = min(n, Σ_e m_e)`).
  #540's stack exactly as written. Buys lane width by spending privacy.
* **reserve** — `privacy_budget = 63`, mask `k_v = n − p` with `p = 8`. #556's
  ruling. `Σ_e m_e ≤ n − 1` never enforced *distinct* lanes — `_assemble` permits
  the same leading block on every incident edge — so privacy and lane capacity
  were welded by one line. Unwelded, both rise together: reserving `p` narrows
  `k_v` at fixed `m`, and composed rank is driven by `m / k_v`.

Both arms keep per-edge `m_e` and laterals at `m = 1`, and both sit at budget 63
— one change at a time, and 63 is the number #540 already ruled.

**Nothing here edits `src/`.** #555's notes say *build the stack on the rig, not
on the live surface*, and #556's ruling is a decision the map has not yet written.
The reserve mask is applied to a built `Dome` by `dataclasses.replace`, and the
lane cap by :func:`allocate_capped`, which is `graph.allocate_lane_widths` with
its ceiling parameterised — :func:`check_allocator_matches_shipped` asserts the
two agree exactly at `cap = n`, so the rig cannot drift from what shipped.

Usage::

    PYTHONPATH=src python prototypes/cold-start/T6/arms.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
_T0, _T2, _T4, _T5 = (_HERE.parent / n for n in ("T0", "T2", "T4", "T5"))
for extra in (_ROOT / "prototypes" / "chart-double-duty-166", _ROOT / "benchmarks", _T0):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


t2 = _load("t2_run", _T2 / "run.py")
angles = _load("t4_angles", _T4 / "angles.py")
t4_trained = _load("t4_trained", _T4 / "trained.py")

import construction_grading as cg  # noqa: E402
from patchworks import graph as G  # noqa: E402
from patchworks.agent import Agent  # noqa: E402
from patchworks.graph import (  # noqa: E402
    NODE_STALK_DIM,
    Dome,
    EdgeKind,
    allocate_lane_widths,
    build_graph,
)
from patchworks.sandbox import PlanarPushSandbox  # noqa: E402
from untrained_fixed_point import IMAGE_SIZE, dome_named  # noqa: E402

#: #556's reserve size. Its construction figures are read at `p = 8`.
RESERVE_P = 8
#: #540's doubled invariant, which #548 held and #556 ruled on.
BUDGET = 2 * NODE_STALK_DIM - 1


# -- the allocation, with the ceiling made a parameter -------------------------


def allocate_capped(cells, edges, spec, cap: int):
    """`graph.allocate_lane_widths` with its lane ceiling parameterised.

    The shipped rule caps a lane at `NODE_STALK_DIM`, because a lane can carry no
    more than the node stalk it reads from. Under #556's reserve the readable
    block is `n − p`, not `n`, so the same reasoning caps a lane at `n − p`:
    #556 measured **2 of 682** edges affected at budget 63, `p = 8`, and its
    construction figures already have this cap applied.

    Everything else is `allocate_lane_widths` verbatim, and
    :func:`check_allocator_matches_shipped` asserts it.
    """
    incident: list[list[int]] = [[] for _ in cells]
    for e in edges:
        incident[e.u].append(e.id)
        incident[e.v].append(e.id)

    budgeted = {c.id for c in cells if not c.is_boundary}
    level = {c.id: c.index.level for c in cells}

    width: dict[int, int] = {}
    for e in edges:
        if e.kind is not EdgeKind.INTERIOR:
            width[e.id] = e.m
        elif level[e.u] == level[e.v]:
            width[e.id] = spec.lateral_m

    remaining = {v: spec.privacy_budget for v in budgeted}
    for v in budgeted:
        for eid in incident[v]:
            if eid in width:
                remaining[v] -= width[eid]
    unsized = {e.id for e in edges if e.id not in width}

    while unsized:
        offer = {}
        for v in budgeted:
            free = [i for i in incident[v] if i in unsized]
            if free:
                offer[v] = remaining[v] // len(free)
        ceiling = {}
        for eid in unsized:
            e = edges[eid]
            bids = [offer[x] for x in (e.u, e.v) if x in offer]
            ceiling[eid] = min(bids + [cap])
        lowest = min(ceiling.values())
        for eid in [i for i in unsized if ceiling[i] == lowest]:
            width[eid] = lowest
            for v in (edges[eid].u, edges[eid].v):
                if v in remaining:
                    remaining[v] -= lowest
            unsized.discard(eid)

    return tuple(
        e if e.kind is not EdgeKind.INTERIOR else replace(e, m=width[e.id]) for e in edges
    )


def check_allocator_matches_shipped(spec) -> None:
    """The rig's allocator must be the shipped one at `cap = n`, or nothing holds."""
    b = G._Builder(spec)
    G._lay_out(b) if hasattr(G, "_lay_out") else None
    dome = build_graph(spec)
    cells, edges = list(dome.cells), list(dome.edges)
    bare = [replace(e, m=0) if e.kind is EdgeKind.INTERIOR else e for e in edges]
    shipped = allocate_lane_widths(cells, bare, spec)
    mine = allocate_capped(cells, bare, spec, cap=NODE_STALK_DIM)
    if tuple(e.m for e in shipped) != tuple(e.m for e in mine):
        raise AssertionError("rig allocator has drifted from graph.allocate_lane_widths")


# -- the reserve mask ---------------------------------------------------------


def apply_reserve(dome: Dome, p: int) -> Dome:
    """#556's one-line change, applied to a built dome rather than to `src/`.

    `_assemble` line 813: `min(c.stalk, stalk_sums[c.id])` -> `c.stalk - p`. A
    predicting cell exposes its leading `n − p` directions on **every** incident
    edge and keeps the trailing `p` off all of them, so `p_v = p` at every cell
    whatever the lanes sum to. Lane *dimensions* stop summing to distinct
    *directions*, which is exactly what makes the budget and the floor
    independent.

    A boundary cell is untouched, as in `_assemble`.
    """
    permitted = [c.stalk if c.is_boundary else c.stalk - p for c in dome.cells]
    mask = torch.ones((len(dome.predicting), NODE_STALK_DIM), dtype=torch.bool)
    for row, cell_id in enumerate(dome.predicting):
        mask[row, : permitted[cell_id]] = False
    return replace(dome, _permitted=tuple(permitted), _private_mask=mask)


# -- the arms -----------------------------------------------------------------

ARMS = {
    # label: (privacy_budget, reserve p or None)
    "shipped": (NODE_STALK_DIM - 1, None),
    "doubling": (BUDGET, None),
    "reserve": (BUDGET, RESERVE_P),
}

#: [B13](#560) sweeps `p`. `reserve` above is `p = 8` and stays as #555 wrote it;
#: `reserve_p<N>` is the same arm at any other reserve size, so both the
#: construction sweep and `trained_arms.py` take an arbitrary `p` unchanged.
#: `p = 0` is the reserve *policy* at zero reserve — `k_v = n`, no private block —
#: which is not the same object as the `union` policy the doubling arm runs.
for _p in range(0, NODE_STALK_DIM - 3):
    ARMS.setdefault(f"reserve_p{_p}", (BUDGET, _p))


def build_arm(arm: str, seed: int, split: str = "train"):
    """Build one arm on the shipped spec."""
    budget, p = ARMS[arm]
    base, _ = dome_named("real")
    spec = replace(base, privacy_budget=budget)
    check_allocator_matches_shipped(spec)

    if p is None:
        dome = build_graph(spec)
    else:
        # Rebuild with the lane cap at `n - p`, then reserve the block.
        proto = build_graph(spec)
        bare = [
            replace(e, m=0) if e.kind is EdgeKind.INTERIOR else e for e in proto.edges
        ]
        edges = allocate_capped(list(proto.cells), bare, spec, cap=NODE_STALK_DIM - p)
        dome = apply_reserve(Dome._assemble(spec, proto.cells, tuple(edges)), p)

    env = PlanarPushSandbox(split=split, image_size=IMAGE_SIZE[spec.patch_grid])
    agent = Agent(env, dome=dome, generator=torch.Generator().manual_seed(seed))
    return env, agent


# -- the reads ----------------------------------------------------------------


def privacy_read(dome, p: int | None) -> dict:
    """`p_v` under whichever mask policy this arm runs, and the budget it spends."""
    n = dome.shape.n
    sums = np.array([dome.stalk_sums[c] for c in dome.predicting], dtype=np.float64)
    perm = np.array([dome._permitted[c] for c in dome.predicting], dtype=np.float64)
    private = n - perm
    return {
        "policy": "reserve" if p is not None else "union",
        "reserve_p": p,
        "budget": int(dome.spec.privacy_budget),
        "sum_m_median": float(np.median(sums)),
        "sum_m_max": float(sums.max()),
        "violations": int((sums > dome.spec.privacy_budget).sum()),
        "k_v_median": float(np.median(perm)),
        "k_v_max": float(perm.max()),
        "private_dim_min": float(private.min()),
        "private_dim_zero_cells": int((private <= 0).sum()),
        "private_dim_total": int(private.clip(min=0).sum()),
        "predicting_cells": int(len(sums)),
    }


def widths_read(dome, chains) -> dict:
    rows = [tuple(dome.edges[eid].m for eid in ch["edges"]) for ch in chains]
    uniq: dict[tuple, int] = {}
    for r in rows:
        uniq[r] = uniq.get(r, 0) + 1
    top = sorted(uniq.items(), key=lambda kv: -kv[1])
    lanes = [e.m for e in dome.edges if e.kind is EdgeKind.INTERIOR]
    return {
        "modal": list(top[0][0]),
        "modal_share": top[0][1] / len(rows),
        "distinct": len(uniq),
        "interior_m_min": int(min(lanes)),
        "interior_m_max": int(max(lanes)),
        "lanes_over_k_v": int(
            sum(
                1
                for e in dome.edges
                if e.kind is EdgeKind.INTERIOR
                and e.m > min(dome._permitted[e.u], dome._permitted[e.v])
            )
        ),
    }


def generic_at(dome, chains, rng_seed: int = 0) -> dict:
    """#540's instrument at **this** arm's realised per-hop widths and `k_v`."""
    rng = np.random.default_rng(rng_seed)
    er = []
    for chain in chains:
        composed = None
        for edge_in, cell, edge_out in cg.hops_of(dome, tuple(chain["edges"])):
            k = int(dome.restriction_mask(edge_in, cell).sum())
            mi = min(dome.edges[edge_in].m, k)
            mo = min(dome.edges[edge_out].m, k)
            v_in = np.linalg.qr(rng.standard_normal((k, mi)))[0]
            v_out = np.linalg.qr(rng.standard_normal((k, mo)))[0]
            u_in = np.linalg.qr(rng.standard_normal((mi, mi)))[0]
            u_out = np.linalg.qr(rng.standard_normal((mo, mo)))[0]
            hop = u_out @ (v_out.T @ v_in) @ u_in.T
            composed = hop if composed is None else hop @ composed
        er.append(angles.effective_rank(np.linalg.svd(composed, compute_uv=False)))
    er = np.array(er)
    return {
        "median": float(np.median(er)),
        "mean": float(er.mean()),
        "p90": float(np.quantile(er, 0.90)),
        "max": float(er.max()),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arms", nargs="+", default=list(ARMS))
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--out", type=Path, default=_HERE / "555-arms-construction.json")
    args = p.parse_args()

    record = {"issue": 555, "reading": "#556's two arms, on the shipped spec", "rows": []}
    for arm in args.arms:
        _, reserve_p = ARMS[arm]
        for seed in args.seeds:
            env, agent = build_arm(arm, seed)
            try:
                dome = agent.dome
                chains = t2.rim_chains(dome)
                read = angles.read_surface(agent, chains, f"{arm} seed {seed}")
                row = {
                    "arm": arm,
                    "seed": seed,
                    "chains": len(chains),
                    "privacy": privacy_read(dome, reserve_p),
                    "widths": widths_read(dome, chains),
                    "generic": generic_at(dome, chains),
                    "cap": t4_trained.cap_read(agent),
                    "composed_er": read["composed_er"],
                    "cos_leading_per_hop": read["cos_leading_per_hop"],
                    "cos_second_per_hop": read["cos_second_per_hop"],
                    "cos_all": read["cos_all"],
                    "s2_over_s1": read["s2_over_s1"],
                    "sigma_min_over_max": read["sigma_min_over_max"],
                }
                record["rows"].append(row)
                e, g, pr, w = row["composed_er"], row["generic"], row["privacy"], row["widths"]
                print(
                    f"  {arm:>9} s{seed}: built ER {e['median']:.4f} p90 {e['p90']:.4f} "
                    f"max {e['max']:.4f} | generic {g['median']:.4f} | k_v med "
                    f"{pr['k_v_median']:.0f} | priv min {pr['private_dim_min']:.0f} "
                    f"zero {pr['private_dim_zero_cells']}/{pr['predicting_cells']} "
                    f"tot {pr['private_dim_total']}\n"
                    f"            lanes {w['interior_m_min']}-{w['interior_m_max']} "
                    f"modal {w['modal']} | cos lead "
                    f"{row['cos_leading_per_hop']['median']:.4f} second "
                    f"{row['cos_second_per_hop']['median']:.4f} | cap p90 "
                    f"{row['cap']['ratio_p90']:.4f} at-cap {row['cap']['cells_at_cap']}",
                    flush=True,
                )
            finally:
                env.close()
            args.out.write_text(json.dumps(record, indent=1))
    print(f"wrote {args.out.name}", flush=True)


if __name__ == "__main__":
    main()
