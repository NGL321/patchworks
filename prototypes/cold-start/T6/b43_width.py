"""B43 (#607) §2: both abstraction readings, scored side by side on every arm.

[B35](#594) redefined abstraction to **community membership** — the number of
distinct communities a cell belongs to, a community being a *connected* set of
cells over which one direction stays consistent. The user's instruction was that
the **strict** reading still be investigated rather than dropped, so both are
scored on every arm:

* **strict** — `Σ_e m_e` at a cell. B35 §7 argued it fails because a saturating
  budget makes it uniform. [B40](#603) measured the same outcome by a second
  mechanism: the criterion's width sits on the **floor**, `m_e = 1` on all 194
  wide edges at every checkpoint, so `Σ_e m_e` is **exactly `deg(v)`**. *The
  strict reading should read as plain degree and nothing else* — that is the
  null, it is falsifiable, and it is tested here on both masks.
* **community membership** — counted cell-locally, per B35: how many
  near-orthogonal direction-clusters a cell's incident edges carry. At `m_e = 1`
  this is *better* posed, not worse: each edge carries exactly one direction and
  membership is how many of a degree-`d` cell's `d` directions are mutually
  near-orthogonal.

**Two strict readings, because the mask and the criterion disagree.** `Σ_e m_e`
off the shipped allocation is **not** the same number as `Σ_e m_e` off B34's
allocation: the shipped dome allocates a lane per edge (#548), while B34's
criterion floors every wide edge at 1. Both are reported — `strict_mask` and
`strict_criterion` — because B40's null is about the second and the architecture
ships the first.

**Named risk, carried from the ticket.** [B22](#571) found the community size
distribution bimodal — *the whole graph or a handful of cells, little between* —
median 7 cells with 30% at size 1 on `reserve_p12`. If membership is near-binary
on this surface the reading cannot discriminate, and **that is a result about the
surface rather than about the definition**. It is reported as such.

Usage::

    PYTHONPATH=src python prototypes/cold-start/T6/b43_width.py read
    PYTHONPATH=src python prototypes/cold-start/T6/b43_width.py read --ticks 500
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


b33 = _load("b43w_b33", _HERE / "b33_coexist.py")
b40e = _load("b43w_enum", _HERE / "b40_enumerate.py")
crit = _load("b43w_crit", _HERE / "b40_criterion.py")
b43r = _load("b43w_relay", _HERE / "b43_relay.py")

import construction_grading as cg  # noqa: E402
import holonomy_read as hr  # noqa: E402
from patchworks.agent import Agent  # noqa: E402
from patchworks.learning import PredictionRule, TransportRule  # noqa: E402
from patchworks.restriction import pair_index  # noqa: E402

#: Alignment thresholds. B40's three (0.8, 0.9, 0.95) are kept so the two
#: tickets' numbers are commensurable, and **two lower ones are added because
#: the high ones are inert here**: B40's thresholds cut a *holonomy* spectrum,
#: where 0.9 is a demanding bar, while these cut pairwise alignment between
#: directions in a 20-dimensional exposed block, where two generic directions sit
#: near 0.2 and nothing ever clusters. A reading that returns the direction count
#: at every threshold has not measured communities; the sweep is what shows that.
THRESHOLDS = (0.3, 0.5, 0.8, 0.9, 0.95)

#: Checkpoints. B22 found 20k ticks move every community number by exactly zero
#: on the shipped mask, and B40 found the criterion's own input settles by tick
#: 50 with drift exactly 0.0 — so a long run is not what would move these.
CHECKPOINTS = (0, 500)


def directions(dome, maps, cell: int, cap: int | None = None) -> np.ndarray:
    """The unit directions a cell puts on its incident edges, as `[k, stalk]`.

    Each incident edge contributes the orthonormal basis of its own map's row
    space — `m_e` directions from an `m_e × stalk` map, taken as the right
    singular vectors so a rank-deficient map contributes only what it carries.
    That is the cell-local object B35's membership counts over: *a community is a
    set over which one direction stays consistent*, so the directions a cell has
    are the ones its lanes actually carry.
    """
    rows = []
    for edge_id in dome.incident[cell]:
        edge = dome.edges[edge_id]
        if dome.cells[edge.u].is_boundary or dome.cells[edge.v].is_boundary:
            continue
        try:
            idx = pair_index(edge_id, cg.side_of(dome, edge_id, cell))
        except Exception:
            continue
        with torch.no_grad():
            width = edge.m if cap is None else min(edge.m, cap)
            f = maps.maps[idx][:width].double()
            if f.numel() == 0:
                continue
            u, s, vh = torch.linalg.svd(f, full_matrices=False)
        keep = (s > s.max().clamp_min(1e-300) * 1e-8) if s.numel() else s
        for i in range(vh.shape[0]):
            if bool(keep[i]):
                rows.append(vh[i].numpy())
    if not rows:
        return np.zeros((0, dome.shape.n))
    return np.stack(rows)


def membership(dirs: np.ndarray, threshold: float) -> int:
    """B35's community membership at one cell: near-orthogonal direction clusters.

    Two directions belong to the **same** community when they are aligned —
    `|cos| >= threshold` — because that is what *one direction staying
    consistent* means across two edges. Clusters are the connected components of
    that relation, and the membership count is how many of them the cell has.
    A cell all of whose lanes carry one direction reads `1`; a cell whose lanes
    carry `d` mutually near-orthogonal directions reads `d`.
    """
    k = dirs.shape[0]
    if k == 0:
        return 0
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = dirs / norms
    cos = np.abs(unit @ unit.T)
    parent = list(range(k))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(k):
        for j in range(i + 1, k):
            if cos[i, j] >= threshold:
                ra, rb = find(i), find(j)
                if ra != rb:
                    parent[ra] = rb
    return len({find(i) for i in range(k)})


def spread(values: dict[int, float]) -> dict:
    v = np.array(list(values.values()), dtype=float)
    if v.size == 0:
        return {"n": 0}
    return {
        "n": int(v.size),
        "median": float(np.median(v)),
        "mean": float(v.mean()),
        "min": float(v.min()),
        "max": float(v.max()),
        "std": float(v.std()),
        "distinct": int(len(set(np.round(v, 6).tolist()))),
    }


def strict_readings(dome, alloc: dict[int, int] | None) -> dict:
    """`Σ_e m_e` per cell, off the mask and off B34's allocation, against degree.

    The null B40 hands this ticket: **the strict reading is plain degree**. It is
    tested by equality, cell by cell, not by a correlation that could be high and
    still not be the same number.
    """
    degree = {c: float(dome.degrees[c]) for c in dome.predicting}
    mask_sum = {c: float(dome.stalk_sums[c]) for c in dome.predicting}
    out = {
        "degree": spread(degree),
        "strict_mask": spread(mask_sum),
        "strict_mask_equals_degree_cells": int(
            sum(1 for c in dome.predicting if mask_sum[c] == degree[c])
        ),
        "cells": len(dome.predicting),
    }
    if alloc is not None:
        crit_sum: dict[int, float] = {c: 0.0 for c in dome.predicting}
        for edge_id, m in alloc.items():
            edge = dome.edges[edge_id]
            for end in (edge.u, edge.v):
                if end in crit_sum:
                    crit_sum[end] += float(m)
        # Only the edges the criterion allocates over are in `alloc`; the rest of
        # a cell's incidence keeps its mask width, so the comparison against
        # degree is taken on the criterion's own population.
        touched = {
            c: crit_sum[c]
            for c in dome.predicting
            if crit_sum[c] > 0
        }
        deg_touched = {c: float(len([
            e for e in dome.incident[c] if e in alloc
        ])) for c in touched}
        out["strict_criterion"] = spread(touched)
        out["strict_criterion_equals_degree_cells"] = int(
            sum(1 for c in touched if touched[c] == deg_touched[c])
        )
        out["strict_criterion_cells"] = len(touched)
    return out


def membership_reading(dome, maps, threshold: float, cap: int | None = None) -> dict:
    """Membership per predicting cell, its spread, and the two nulls it can fail.

    Two ways this reading can be empty and they are different failures:

    * **degree** — if membership equals `deg(v)` at every cell it is plain degree
      wearing a new name, which is exactly the null B40 handed the strict
      reading. Reported as `equals_degree_cells`.
    * **binary** — B22's bimodality risk: if membership takes two values it
      cannot grade anything, whatever its spread. Reported as `distinct`.
    """
    counts: dict[int, float] = {}
    dirs_count: dict[int, float] = {}
    for cell in dome.predicting:
        dirs = directions(dome, maps, cell, cap)
        dirs_count[cell] = float(dirs.shape[0])
        counts[cell] = float(membership(dirs, threshold))
    interior_degree = {
        c: float(
            len(
                [
                    e
                    for e in dome.incident[c]
                    if not dome.cells[dome.edges[e].u].is_boundary
                    and not dome.cells[dome.edges[e].v].is_boundary
                ]
            )
        )
        for c in dome.predicting
    }
    return {
        "threshold": threshold,
        "cap": cap,
        "membership": spread(counts),
        "directions_per_cell": spread(dirs_count),
        "equals_degree_cells": int(
            sum(1 for c in counts if counts[c] == interior_degree[c])
        ),
        "equals_one_cells": int(sum(1 for c in counts if counts[c] == 1.0)),
        "cells": len(counts),
        "histogram": {
            str(int(v)): int(sum(1 for x in counts.values() if x == v))
            for v in sorted(set(counts.values()))
        },
    }


def criterion_alloc(dome, maps, threshold: float) -> dict[int, int]:
    """B34's per-edge count and allocation on this mask, via B40's instruments."""
    adj, wide_edges = b40e.wide_adjacency(dome)
    counts: dict[int, int] = {}
    for edge_id in wide_edges:
        cycles = b40e.cycles_through(adj, dome, edge_id)
        local = [c for c in cycles if b33.local_cycles(dome, [c], b33.RADIUS)]
        if not local:
            counts[edge_id] = 0
            continue
        b29 = _load("b43w_b29", _HERE / "b29_holonomy.py")
        holos = [
            hr.holonomy(dome, maps, b29._base_at(tuple(c), edge_id)) for c in local
        ]
        n, _cos = crit.warranted_count(holos, threshold)
        counts[edge_id] = n
    warrant = {c: int(dome._permitted[c]) for c in dome.predicting}
    return crit.allocate(
        dome, counts, budget=int(dome.spec.capacity_budget), endpoint_warrant=warrant
    )


def read_layout(
    layout: str, sites: int, seed: int, ticks: int, depth: int, relay_m: int
) -> dict:
    started = time.time()
    env, agent = b33.arms_mod.build_arm(b33.ARM, seed)
    base = agent.dome
    relay_ids: list[int] = []
    layout_record = {"layout": "none", "relays": 0}
    if layout != "none":
        relays, layout_record = b43r.lay_relays(base, layout, depth, relay_m, sites)
        dome = b43r.relayed(base, relays)
        relay_ids = list(range(len(base.edges), len(dome.edges)))
        agent = Agent(env, dome=dome, generator=torch.Generator().manual_seed(seed))
    dome = agent.dome

    bias = PredictionRule(agent.sheaf)
    transport = TransportRule(agent.sheaf)
    record = {
        "issue": 607,
        "reading": "both abstraction readings, per arm",
        "layout": layout,
        "sites": sites,
        "seed": seed,
        "relays": len(relay_ids),
        "relay_m": relay_m,
        "edges": len(dome.edges),
        "layout_record": layout_record,
        "checkpoints": [],
    }

    seen = 0
    for target in [c for c in CHECKPOINTS if c <= ticks]:
        span = target - seen
        for _ in b33.t0.run_ticks(agent, span, seed=seed + seen):
            bias.step()
            if agent.sheaf.ticks > 1:
                transport.step()
        seen = target
        maps = agent.sheaf.maps
        alloc = criterion_alloc(dome, maps, crit.THRESHOLD)
        entry = {
            "ticks": target,
            "elapsed_minutes": (time.time() - started) / 60.0,
            "strict": strict_readings(dome, alloc),
            "criterion_alloc": spread({k: float(v) for k, v in alloc.items()}),
            "membership": {
                str(th): membership_reading(dome, maps, th) for th in THRESHOLDS
            },
            # **The floored variant is B35's own case.** B40 read `m_e = 1` on
            # every wide edge at every checkpoint, and B35 §7's argument that
            # membership is *better* posed there is a claim about this reading:
            # one direction per edge, so membership counts how many of a
            # degree-`d` cell's `d` directions are mutually near-orthogonal.
            "membership_floored": {
                str(th): membership_reading(dome, maps, th, cap=1)
                for th in THRESHOLDS
            },
        }
        record["checkpoints"].append(entry)
        m = entry["membership"][str(0.9)]["membership"]
        s = entry["strict"]
        print(
            f"  [{layout}{sites} @{target:>4}] "
            f"strict_mask med {s['strict_mask']['median']:.1f} "
            f"(= degree at {s['strict_mask_equals_degree_cells']}/{s['cells']}) "
            f"| crit m_e med {entry['criterion_alloc']['median']:.1f} "
            f"| membership@0.9 med {m['median']:.1f} max {m['max']:.0f} "
            f"std {m['std']:.2f} distinct {m['distinct']} "
            f"(= degree at {entry['membership'][str(0.9)]['equals_degree_cells']})",
            flush=True,
        )
    record["minutes"] = (time.time() - started) / 60.0
    return record


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["read"])
    p.add_argument("--layouts", nargs="+", default=["none", "motor", "pair", "pair_wide"])
    p.add_argument("--sites", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ticks", type=int, default=max(CHECKPOINTS))
    p.add_argument("--depth", type=int, default=b43r.DEPTH)
    p.add_argument("--relay-m", type=int, default=b43r.RELAY_M)
    p.add_argument("--json", type=str, default="607-width-seed42.json")
    args = p.parse_args()

    out = []
    for layout in args.layouts:
        sites = 0 if layout == "none" else args.sites
        out.append(
            read_layout(layout, sites, args.seed, args.ticks, args.depth, args.relay_m)
        )
        Path(_HERE / args.json).write_text(
            json.dumps({"issue": 607, "arms": out}, indent=1), encoding="utf-8"
        )
    print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
