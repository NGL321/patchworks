"""T6 (#555): build [B2](#540)'s per-edge stack on a real dome, and read it.

`#555 <https://github.com/NGL321/patchworks/issues/555>`_ item 1. #540 ruled a
three-part stack -- ``m_e`` per edge, lateral lanes at ``m = 1``, and the privacy
invariant doubled to ``sum_e m_e <= 2n - 1`` -- and priced it at composed ER
**2.193** on *generic chains at the widths the stack would realise*. No dome has
ever been built that way. [B6](#546) could only vary ``interior_m``, a single
global constant, and that **saturates** (rebuilt construction ER 1.4115 at m=10,
1.6158 at 14, 1.7772 at 20) because ``k_v = min(n, sum_e m_e)`` caps at ``n = 32``
while the chain stays seven hops.

This module builds the stack for real and reads it at construction.

**The allocation rule.** #540 and [B8](#548) both state it as *"``m_e`` set per
edge, largest both endpoints can afford"*. Taken literally that is
under-determined -- two edges at one cell compete for the same budget, so
"largest" depends on which is served first. The only order-independent reading is
**max-min fairness**, which is what :func:`allocate` implements: every free edge
starts at 1 and is raised a unit at a time, round-robin, while both its endpoints
have slack. No edge is raised while a competitor is narrower, so the allocation
is a function of the graph and the cap alone.

What is **not** free, and why:

* **boundary lanes** keep ``boundary_m`` -- #540's ruled stack does not move
  them (its ``boundary_m`` 4 -> 6 row is *outside* the stack, at 1.481).
* **drive lanes** keep ``drive_m = 1``. ADR-0009's width axis has a standing
  unread re-read ([T7](#535)'s advisory on B8) *and* a ceiling of ``m_e ~ 4``
  from the apex-side map's construction effective rank 3.66 ([#356](#356)).
  Widening the drive edge is not this ticket's to do.
* **lateral lanes** are pinned at ``lateral_m`` -- rung (c1) of the stack.

Usage::

    PYTHONPATH=src python prototypes/cold-start/T6/stack.py
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
width_mod = _load("t5_width", _T5 / "width.py")

import construction_grading as cg  # noqa: E402
from patchworks.agent import Agent  # noqa: E402
from patchworks.graph import Dome, EdgeKind, build_graph  # noqa: E402
from patchworks.sandbox import PlanarPushSandbox  # noqa: E402
from untrained_fixed_point import IMAGE_SIZE, dome_named  # noqa: E402


# -- the allocation -----------------------------------------------------------


def is_lateral(dome, edge) -> bool:
    """Both endpoints on the same construction level.

    #540 measured **0 of 405** chain edges lateral, so a lateral lane is width
    the composed object never traverses -- which is what makes rung (c1) cheap.
    """
    return dome.cells[edge.u].index.level == dome.cells[edge.v].index.level


def allocate(dome, cap: int, lateral_m: int = 1, per_edge: bool = True) -> dict[int, int]:
    """Max-min fair `m_e` over the free (non-lateral, non-boundary, non-drive) edges.

    `cap` is the privacy invariant's right-hand side: `n - 1` today, `2n - 1`
    under #540's rung (b). It binds at **predicting** cells only -- a boundary
    cell is not masked (`graph.py::_assemble`), so it imposes nothing.

    With `per_edge=False` the free edges are all held at the dome's own
    `interior_m`, which is the "today" control.
    """
    n = dome.shape.n
    fixed_kinds = {EdgeKind.SENSORY, EdgeKind.MOTOR, EdgeKind.DRIVE}
    m = {}
    free = []
    for e in dome.edges:
        if e.kind in fixed_kinds:
            m[e.id] = e.m
        elif is_lateral(dome, e):
            m[e.id] = int(lateral_m)
        elif per_edge:
            m[e.id] = 1
            free.append(e.id)
        else:
            m[e.id] = e.m

    predicting = {c.id for c in dome.cells if not c.is_boundary}

    def load(cell_id: int) -> int:
        return sum(m[eid] for eid in dome.incident[cell_id])

    used = {c: load(c) for c in predicting}
    if not per_edge:
        return m

    # Round-robin unit raises. Max-min fair: no edge is raised past a competitor
    # that is still narrower, and the fixed point is independent of edge order
    # up to the single unit a round can differ by.
    while True:
        raised = False
        for eid in free:
            if m[eid] >= n:
                continue
            ends = [c for c in (dome.edges[eid].u, dome.edges[eid].v) if c in predicting]
            if all(used[c] < cap for c in ends):
                m[eid] += 1
                for c in ends:
                    used[c] += 1
                raised = True
        if not raised:
            return m


def build_stack(
    seed: int,
    cap: int,
    lateral_m: int = 1,
    per_edge: bool = True,
    split: str = "train",
):
    """`build("real", ...)` with the per-edge allocation written onto the edges."""
    spec, _ = dome_named("real")
    base = build_graph(spec)
    m = allocate(base, cap=cap, lateral_m=lateral_m, per_edge=per_edge)
    edges = tuple(replace(e, m=m[e.id]) for e in base.edges)
    dome = Dome._assemble(spec, base.cells, edges)
    env = PlanarPushSandbox(split=split, image_size=IMAGE_SIZE[spec.patch_grid])
    agent = Agent(env, dome=dome, generator=torch.Generator().manual_seed(seed))
    return env, agent


# -- what the stack costs, and what it realises -------------------------------


def invariant_read(dome, cap: int) -> dict:
    """Does the built dome honour `cap`, and what does it spend to?"""
    sums = np.array([dome.stalk_sums[c] for c in dome.predicting], dtype=np.float64)
    n = dome.shape.n
    return {
        "cap": int(cap),
        "violations": int((sums > cap).sum()),
        "at_cap_cells": int((sums == cap).sum()),
        "sum_m_median": float(np.median(sums)),
        "sum_m_max": float(sums.max()),
        "private_dim_zero_cells": int((sums >= n).sum()),
        "private_dim_total": int(np.maximum(0.0, n - sums).sum()),
        "predicting_cells": int(len(sums)),
    }


def chain_widths(dome, chains) -> dict:
    """The realised rim -> apex lane widths, which is what #540's table priced."""
    rows = [tuple(dome.edges[eid].m for eid in ch["edges"]) for ch in chains]
    uniq = {}
    for r in rows:
        uniq[r] = uniq.get(r, 0) + 1
    top = sorted(uniq.items(), key=lambda kv: -kv[1])[:6]
    return {
        "modal": list(top[0][0]),
        "modal_share": top[0][1] / len(rows),
        "distinct": len(uniq),
        "top": [{"widths": list(w), "chains": c} for w, c in top],
    }


def generic_at(dome, chains, rng_seed: int = 0) -> dict:
    """#540's own instrument, run at **this** dome's realised per-hop widths.

    Every structural fact is the built dome's -- which cell, its `k_v`, the two
    lane widths at each hop, the chain length. Only the frames are redrawn Haar.
    This is the number #540's table reports, so the gap between it and the built
    surface's own reading is exactly #540's stated falsifier: *"if a surface is
    found whose composed ER departs from the generic prediction at its own
    per-hop widths, §2's whole table is wrong."*
    """
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
        "chains": int(len(er)),
    }


# -- the rungs ----------------------------------------------------------------

N = 32
RUNGS = {
    # label: (cap, lateral_m, per_edge)   -- #540 §2's ruled stack, rebuilt
    "today": (N - 1, 3, False),
    "d_per_edge": (N - 1, 3, True),
    "c1_laterals": (N - 1, 1, True),
    "b_invariant": (2 * N - 1, 1, True),
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rungs", nargs="+", default=list(RUNGS))
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--out", type=Path, default=_HERE / "555-construction.json")
    args = p.parse_args()

    record = {"issue": 555, "reading": "construction, #540's stack rebuilt", "rows": []}
    for rung in args.rungs:
        cap, lateral_m, per_edge = RUNGS[rung]
        for seed in args.seeds:
            env, agent = build_stack(seed, cap=cap, lateral_m=lateral_m, per_edge=per_edge)
            try:
                dome = agent.dome
                chains = t2.rim_chains(dome)
                read = angles.read_surface(agent, chains, f"{rung} seed {seed}")
                row = {
                    "rung": rung,
                    "cap": cap,
                    "lateral_m": lateral_m,
                    "per_edge": per_edge,
                    "seed": seed,
                    "chains": len(chains),
                    "widths": chain_widths(dome, chains),
                    "invariant": invariant_read(dome, cap),
                    "mask": width_mod.mask_read(dome),
                    "generic": generic_at(dome, chains),
                    # #555 item 3: does the stack buy its rank by raising
                    # incoherence? Read beside the ER, not after the fact.
                    "cap": t4_trained.cap_read(agent),
                    "composed_er": read["composed_er"],
                    "cos_product_er": read["cos_product_er"],
                    "cos_all": read["cos_all"],
                    "cos_leading_per_hop": read["cos_leading_per_hop"],
                    "cos_second_per_hop": read["cos_second_per_hop"],
                    "sigma_min_over_max": read["sigma_min_over_max"],
                    "s2_over_s1": read["s2_over_s1"],
                    "spectrum_normalised_median": read["spectrum_normalised_median"],
                }
                record["rows"].append(row)
                e, g, cap_r = row["composed_er"], row["generic"], row["cap"]
                print(
                    f"  {rung:>12} seed {seed}: built ER med {e['median']:.4f} "
                    f"p90 {e['p90']:.4f} max {e['max']:.4f} | generic med {g['median']:.4f} "
                    f"| widths {row['widths']['modal']} "
                    f"({row['widths']['modal_share']:.0%}) | k_v med "
                    f"{row['mask']['k_v_median']:.0f} sat "
                    f"{row['mask']['k_v_saturated_cells']}/{row['mask']['predicting_cells']} "
                    f"| private {row['invariant']['private_dim_total']}\n"
                    f"               cos lead {row['cos_leading_per_hop']['median']:.4f} "
                    f"second {row['cos_second_per_hop']['median']:.4f} "
                    f"all {row['cos_all']['median']:.4f} | s2/s1 {row['s2_over_s1']['median']:.4f} "
                    f"| sigma min/max {row['sigma_min_over_max']['median']:.7f} "
                    f"| cap ratio med {cap_r['ratio_median']:.4f} p90 {cap_r['ratio_p90']:.4f} "
                    f"max {cap_r['ratio_max']:.4f} at-cap {cap_r['cells_at_cap']}"
                    f"/{cap_r['cells_measured']}",
                    flush=True,
                )
            finally:
                env.close()
            args.out.write_text(json.dumps(record, indent=1))
    print(f"wrote {args.out.name}", flush=True)


if __name__ == "__main__":
    main()
