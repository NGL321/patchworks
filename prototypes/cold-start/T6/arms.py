"""T6 (#555): the arms [B12/#556](#556) ruled — now **on** the shipped spec, not beside it.

**Ported on [B38](#599) to the surface #597 shipped.** #555 wrote this module
while #556's ruling was still a decision the map had not written, so the rig
carried its own copy of the ruling: a parameterised allocator and a
`dataclasses.replace` that stamped the reserve mask onto a built `Dome`.
[B15/#562](#562), landed by [#597](#597), wrote that ruling into
`src/patchworks/graph.py`, and the fake and the real have swapped places:

* `DomeSpec.privacy_budget` is gone. The capacity bound is
  `DomeSpec.capacity_budget`, defaulting to `2n − 1 = 63` — #540's doubling,
  which #548 held and #556 unblocked.
* `DomeSpec.private_reserve` is new and defaults to `CHART_DIM = 12`. It is
  #556's `p`, and `Dome._assemble` now applies the reserve mask itself:
  `k_v = n − p` at every predicting cell, boundary cells untouched.
* `allocate_lane_widths` now caps an interior lane at `n − spec.private_reserve`
  natively, which is exactly what this module's `allocate_capped` existed to do.

So **the reserve arms are no longer a rig construction at all** — `reserve_p<N>`
is `replace(spec, private_reserve=N)` and a plain `build_graph`. The rig's own
allocator and its `apply_reserve` are deleted rather than repaired: calling the
shipped allocator directly is strictly stronger than asserting a copy of it
agrees, which is all `check_allocator_matches_shipped` ever bought.

**What is left of the rig-side masking is the inverse of what was there.** The
`shipped` and `doubling` arms run the **union** mask, `k_v = min(n, Σ_e m_e)`,
which #556 retired and #597 removed from `src/`. They are kept, and only kept,
so this map's pre-#556 readings stay reproducible; :func:`apply_union` is the
one `dataclasses.replace` left in this file and it reconstructs a **retired**
surface rather than standing in for an unwritten ruling.

The arms, per #455's name-the-surface rule:

* **shipped** — *historical*. `capacity_budget = n − 1 = 31`, union mask. The
  surface before #548. Not what `main` builds today.
* **doubling** — *historical*. `capacity_budget = 63`, union mask. #540's stack
  as written, which #548 declined to ship because the union mask read
  `p_v = 0` at 104 of 150 predicting cells.
* **reserve_p<N>** — *live*. `capacity_budget = 63`, `private_reserve = N`.
  Built by `src/` unmodified. `reserve` is the #555 alias at `p = 8`.

**`reserve_p12` is what `main` builds.** `DomeSpec()` with nothing overridden is
`reserve_p12`, so that arm is no longer a variant — it is the shipped dome, and
every other arm here is a deliberate departure from it.

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
from patchworks.agent import Agent  # noqa: E402
from patchworks.body import CHART_DIM  # noqa: E402
from patchworks.graph import (  # noqa: E402
    NODE_STALK_DIM,
    Dome,
    EdgeKind,
    build_graph,
)
from patchworks.sandbox import PlanarPushSandbox  # noqa: E402
from untrained_fixed_point import IMAGE_SIZE, dome_named  # noqa: E402

#: #556's reserve size as #555 first read it. **Not the shipped value** — since
#: #597 that is `DomeSpec.private_reserve`, `= CHART_DIM = 12`, on #560's
#: derivation. Kept so the `reserve` alias still names the arm #555 built.
RESERVE_P = 8
#: #540's doubled invariant. Since #597 this is `DomeSpec.capacity_budget`'s
#: own default, so the reserve arms no longer have to set it.
BUDGET = 2 * NODE_STALK_DIM - 1
#: The pre-#548 capacity bound, which only the `shipped` arm still runs.
LEGACY_BUDGET = NODE_STALK_DIM - 1
#: What `main` builds with nothing overridden.
SHIPPED_P = CHART_DIM


# -- the retired union mask ---------------------------------------------------


def apply_union(dome: Dome) -> Dome:
    """Put a built dome back on the **union** mask `_assemble` used before #556.

    The inverse of what this file used to hold. Pre-#556 `_assemble` read
    `permitted = min(c.stalk, stalk_sums[c.id])` at a predicting cell, so the
    readable block was whatever the incident lanes happened to sum to and the
    private width was the residual `max(0, n − Σ_e m_e)`. #597 replaced that
    with the flat reserve, so reconstructing it is now the rig's job.

    Used **only** by the `shipped` and `doubling` arms, which exist to keep this
    map's pre-#556 readings reproducible. Nothing live runs through here.

    A boundary cell is untouched, as in `_assemble` then and now.
    """
    permitted = [
        c.stalk if c.is_boundary else min(c.stalk, dome.stalk_sums[c.id])
        for c in dome.cells
    ]
    mask = torch.ones((len(dome.predicting), NODE_STALK_DIM), dtype=torch.bool)
    for row, cell_id in enumerate(dome.predicting):
        mask[row, : permitted[cell_id]] = False
    return replace(dome, _permitted=tuple(permitted), _private_mask=mask)


# -- the arms -----------------------------------------------------------------

ARMS = {
    # label: (capacity_budget, reserve p, mask policy)
    "shipped": (LEGACY_BUDGET, None, "union"),
    "doubling": (BUDGET, None, "union"),
    "reserve": (BUDGET, RESERVE_P, "reserve"),
}

#: [B13](#560) sweeps `p`. `reserve` above is `p = 8` and stays as #555 wrote it;
#: `reserve_p<N>` is the same arm at any other reserve size, so both the
#: construction sweep and `trained_arms.py` take an arbitrary `p` unchanged.
#: `p = 0` is the reserve *policy* at zero reserve — `k_v = n`, no private block —
#: which is not the same object as the `union` policy the doubling arm runs.
for _p in range(0, NODE_STALK_DIM - 3):
    ARMS.setdefault(f"reserve_p{_p}", (BUDGET, _p, "reserve"))


def spec_for(arm: str):
    """The `DomeSpec` this arm is, on shipped fields.

    For a reserve arm this is the whole of the arm: `src/` builds it unmodified.
    For a union arm it is the build, and :func:`apply_union` then puts the mask
    back. `private_reserve = 0` on a union arm is not that arm's privacy policy —
    it is how the shipped allocator is told to cap a lane at `n`, which is what
    the pre-#556 allocator did.
    """
    budget, p, policy = ARMS[arm]
    base, _ = dome_named("real")
    return replace(base, capacity_budget=budget, private_reserve=0 if p is None else p)


def build_arm(arm: str, seed: int, split: str = "train"):
    """Build one arm. Reserve arms are `src/` verbatim; union arms are remasked."""
    _, p, policy = ARMS[arm]
    dome = build_graph(spec_for(arm))
    if policy == "union":
        dome = apply_union(dome)

    env = PlanarPushSandbox(split=split, image_size=IMAGE_SIZE[dome.spec.patch_grid])
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
        "budget": int(dome.spec.capacity_budget),
        "sum_m_median": float(np.median(sums)),
        "sum_m_max": float(sums.max()),
        "violations": int((sums > dome.spec.capacity_budget).sum()),
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
        _, reserve_p, _policy = ARMS[arm]
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
