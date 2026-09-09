"""B43 (#607): the relay layer as a mask, and clause 1 of the adoption conjunction.

[B35](#594) decided the justification and the adoption rule; this ticket reads
whether the numbers arrive. Building the relay layer is therefore a **rig**
question, and this module is the rig: it takes the shipped dome, adds a relay
layer to it, and re-assembles a `Dome` from the widened edge list. Nothing in
`src/` is touched and no architecture is changed — a relayed mask is a
counterfactual graph in exactly the sense `loop_length.split` builds one.

**Clause 1 is free and takes no run.** The ticket says so:

> *`world_loop(c)` falls on the relayed mask. Exact integers off the mask,
> `benchmarks/loop_length.py world`. This clause is free and takes no run — it
> can be read against a candidate relay layout before anything trains.*

So it is read here, at construction, on every layout, before a tick.

**[B40](#603)'s layout constraint does not fire as written, and the reason is
worth more than the constraint.** B40 requires *at least two disjoint relay paths
between any pair of regions joined*, because *a relay edge is a bridge by
construction* and a bridge has no cycle for B34's criterion to count over. On
this surface the antecedent cannot be met: **the dome is already connected**, so a
relay is a **chord**, never a bridge, at every layout and every count —
:func:`bridge_census` reads 0 relay bridges throughout, tree-shaped layouts
included. What survives is the *practical* half, and it is sharper:
:func:`local_cycle_census` counts the **short local** cycles B34's criterion
actually reads, and a long chord has none unless two relays land close enough to
close on each other. That, not bridgehood, is what pins a relay at the probe
floor.

**The layouts, and why each exists.**

* `none` — the shipped mask. Arm A's surface.
* `sensory` / `motor` — one relay per site, to a rim-adjacent cell touching the
  sensory rim or the actuator respectively. **Both are negative controls for the
  leg they omit**, and they are how the two-leg constraint below was found.
* `pair` — the layout under test: one relay to each pool per site, which both
  shortens both legs of `world_loop` and gives the site two anchor-disjoint
  paths out.
* `pair_wide` — the same layout at the interior allocation cap, so the capacity
  budget's teeth are visible.
* `random` — **the unaimed control**, the same relay count and width laid between
  predicting cells drawn at random. `pair` raises the rim-to-apex bottleneck by
  nine orders and a chord across a seven-hop graph raises it whether or not it
  was aimed; what separates `pair` from `random` is the aim alone.

**What the layout is aimed at.** ADR-0026's enumeration gives the apex
`world_loop = 15–16` against L1's 3–9, floor `world_loop(c) >= 2·d(c, rim) + w`.
The ticket pre-registers the target: *a sparse relay network putting the apex 2–3
hops from the rim predicts roughly 5–8*. That is arm B's number and it is read
below without training.

**No appeal to `level`.** [B27](#576) retired the dome as architecture and this
map forbids `level`-indexed readings. Depth here is `d(c, rim)` — hop distance on
the mask, which is `world_loop`'s own ingredient (`loop_length.world_loops`
computes it by breadth-first sweep) and not a construction stratum. The deepest
cells are found by that sweep, not by reading `CellIndex.level`.

Usage::

    PYTHONPATH=src python prototypes/cold-start/T6/b43_relay.py layouts
    PYTHONPATH=src python prototypes/cold-start/T6/b43_relay.py layouts --json 607-layouts.json
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
for extra in (_ROOT / "prototypes" / "chart-double-duty-166", _ROOT / "benchmarks"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

import loop_length as ll  # noqa: E402
from patchworks.graph import (  # noqa: E402
    CellKind,
    Dome,
    DomeSpec,
    Edge,
    EdgeKind,
    build_graph,
)

#: Layouts built. `none` is the shipped mask, unwidened, and is arm A's surface.
LAYOUTS = ("none", "sensory", "motor", "pair", "pair_wide")

#: A cell is a **relay target** when it is this far from the sensorimotor rim or
#: further. Not a level: `d(c, rim)` off `loop_length.adjacency`. Chosen so the
#: targets are the cells whose `world_loop` the relay exists to shorten, and
#: swept by `--depth` so nothing rests on the value.
DEPTH = 4

#: How many deep cells get a relay. The sparsity knob, swept by `--sites`: the
#: deep set on this dome is one connected core, so *how many of its cells are
#: served* is what "sparse" can mean here, and a region count is not available.
SITES = 8

#: Lane width on a relay edge. `1` is the cheapest thing that carries anything
#: and is the default, because a relay is **budget-bearing and prediction-free**
#: (B35 §6) and the honest first reading is the one that prices it lowest.
#: `pair_wide` runs the same layout at the interior allocation cap so the
#: budget's teeth are visible.
RELAY_M = 1


def depth_from_rim(dome: Dome) -> dict[int, int]:
    """`d(c, rim)` for every cell, by the same sweep `loop_length` uses."""
    neighbours = ll.adjacency(dome)
    rim = ll.rim_of(dome)
    depth: dict[int, int] = {}
    for source in rim:
        for cell, d in ll.distances_from(neighbours, source).items():
            if d < depth.get(cell, 1 << 30):
                depth[cell] = d
    return depth


def regions(dome: Dome, depth: dict[int, int], floor: int) -> list[list[int]]:
    """Connected components of the cells at `d(c, rim) >= floor`.

    A **region** here is a connected set of deep cells — the thing a relay
    joins. It is not B22's community (which is a set over which one *direction*
    stays consistent) and does not pretend to be: this is a rig partition for
    laying cable, and the community reading is `b43_width.py`'s.
    """
    deep = {c for c, d in depth.items() if d >= floor and not dome.cells[c].is_boundary}
    neighbours = ll.adjacency(dome)
    seen: set[int] = set()
    out: list[list[int]] = []
    for cell in sorted(deep):
        if cell in seen:
            continue
        stack, comp = [cell], []
        seen.add(cell)
        while stack:
            here = stack.pop()
            comp.append(here)
            for other in neighbours[here]:
                if other in deep and other not in seen:
                    seen.add(other)
                    stack.append(other)
        out.append(sorted(comp))
    return out


def anchors(dome: Dome, depth: dict[int, int]) -> dict[str, list[int]]:
    """Where a relay lands, in **two pools split by which rim they touch**.

    A first version pooled every predicting cell at `d(c, rim) == 1` and split it
    in half by id. That layout moved `world_loop` almost not at all — 16 to 11 at
    32 relays — and the reason is structural rather than a tuning failure:

        `world_loop(c) = min over (a, p) of d(c, a) + w + d(p, c)`

    has **two legs**, and `a` ranges over the actuators alone. Every anchor in
    that pool was adjacent to a *sensory* cell, so the sensory leg shortened and
    the actuator leg did not, and a min of the two is bounded by the leg nobody
    touched. **A relay network that anchors on one side of ADR-0016's ban buys
    half a loop and the bar reads the other half.** That is a layout constraint
    of the same species as B40's, found here rather than inherited, and it is why
    the pools below are `motor` and `sensory` rather than two halves of one list.

    Anchoring **on** a boundary cell is not done: the ban is what makes
    `world_loop` the loop it is, and a relay into a sensory cell would be a
    second write on a cell the world writes.
    """
    neighbours = ll.adjacency(dome)
    actuators = {c.id for c in dome.cells if c.kind is CellKind.ACTUATOR}
    sensory = {c.id for c in dome.cells if c.kind in ll.SENSORY}
    motor_pool, sensory_pool = [], []
    for cell in sorted(depth):
        if dome.cells[cell].is_boundary:
            continue
        near = neighbours[cell]
        if near & actuators:
            motor_pool.append(cell)
        elif near & sensory:
            sensory_pool.append(cell)
    return {"motor": motor_pool, "sensory": sensory_pool}


def _spread(items: list[int], k: int) -> list[int]:
    """`k` anchors drawn from `items`, spread evenly, cycling when the pool runs out.

    Cycling is not a convenience. The **motor** pool on this dome is tiny — the
    cells adjacent to the one actuator — so past a handful of sites a relay
    network *must* reuse a motor anchor, and #311's anti-hub objection stops
    being self-enforcing and starts being enforced by the graph. The reuse is
    reported (`anchor_reuse_max`) rather than hidden, because it is the shape
    B20's joint-span collapse takes here.
    """
    if not items:
        return []
    if k <= len(items):
        step = len(items) / k
        return [items[int(i * step)] for i in range(k)]
    return [items[i % len(items)] for i in range(k)]


def lay_relays(
    dome: Dome,
    layout: str,
    depth_floor: int = DEPTH,
    relay_m: int = RELAY_M,
    sites: int = SITES,
) -> tuple[list[tuple[int, int, int]], dict]:
    """The relay edge list for a layout, as `(u, v, m)` triples, and its record.

    **A relay site is a deep cell, not a region.** The first version of this
    partitioned the deep cells into connected regions and gave each region a
    relay; on this dome every cell at `d(c, rim) >= 4` is *one* connected
    component, so that layout is a single relay and not a network. The deep set
    is a core, so the sparsity knob is **how many of its cells get a relay** —
    `sites`, swept — and the deepest cells are served first.

    `tree` gives each site **one** relay; `pair` gives each site **two**, to
    anchors drawn from two disjoint pools, so the two relay paths out of a site
    share no anchor. Which cells are picked is fixed by `(-depth, id)` — a rig
    choice, stated, and not swept: the question is whether the layout *shape*
    buys the clauses, not which of 150 cells is luckiest.
    """
    depth = depth_from_rim(dome)
    if layout == "none":
        return [], {"layout": "none", "sites": 0, "relays": 0}
    if layout == "random":
        # **The unaimed control, and it is not optional.** `pair` raises the
        # rim-to-apex bottleneck by nine orders, and a chord across a seven-hop
        # graph raises it whether or not it was aimed at the loop: B21 read
        # 20–50x of gain per hop, so *any* short circuit buys orders. This lays
        # the same count of relays at the same width between predicting cells
        # drawn at random, aimed at nothing. What separates `pair` from this is
        # the aim alone — the same role `haar` played for B40's `flat`.
        rng = np.random.default_rng(20607)
        pool = [c for c in dome.predicting]
        count = max(1, int(sites)) * 2
        relays, detail = [], []
        for _ in range(count):
            u, v = rng.choice(len(pool), size=2, replace=False)
            relays.append((int(pool[u]), int(pool[v]), int(relay_m)))
            detail.append(
                {
                    "deep_cell": int(pool[u]),
                    "deep_depth": int(depth[pool[u]]),
                    "anchor": int(pool[v]),
                    "leg": "random",
                }
            )
        return relays, {
            "layout": "random",
            "relay_m": int(relay_m),
            "sites": int(sites),
            "relays": len(relays),
            "legs": ["random"],
            "anchor_reuse_max": max(
                collections.Counter(d["anchor"] for d in detail).values(), default=0
            ),
            "edges": detail,
        }
    regs = regions(dome, depth, depth_floor)
    deep = sorted(
        (c for c, d in depth.items() if d >= depth_floor and not dome.cells[c].is_boundary),
        key=lambda c: (-depth[c], c),
    )
    picked = deep[: max(1, int(sites))]
    anch = anchors(dome, depth)
    if not anch["motor"] or not anch["sensory"]:
        raise RuntimeError("a rim-adjacent pool is empty; cannot anchor a relay")
    m = int(relay_m)
    if layout == "pair_wide":
        m = int(dome.shape.n - dome.spec.private_reserve)
    #: `sensory` and `motor` are the one-leg layouts and are the negative
    #: controls: each shortens one side of ADR-0016's ban and the `min` over
    #: `(a, p)` is then bounded by the side it did not touch. `pair` is the
    #: layout under test — one relay to each pool per site, which is also what
    #: makes the two relay paths out of a site disjoint.
    legs = {"sensory": ("sensory",), "motor": ("motor",)}.get(
        layout, ("motor", "sensory")
    )
    picks_by_leg = {
        leg: _spread(anch[leg], len(picked)) for leg in legs
    }

    relays: list[tuple[int, int, int]] = []
    detail: list[dict] = []
    for i, cell in enumerate(picked):
        for leg in legs:
            anchor = picks_by_leg[leg][i]
            relays.append((cell, anchor, m))
            detail.append(
                {
                    "deep_cell": int(cell),
                    "deep_depth": int(depth[cell]),
                    "anchor": int(anchor),
                    "leg": leg,
                }
            )
    return relays, {
        "layout": layout,
        "depth_floor": int(depth_floor),
        "relay_m": m,
        "sites": len(picked),
        "deep_cells_available": len(deep),
        "deep_components": len(regs),
        "component_sizes": [len(c) for c in regs],
        "anchors_motor": len(anch["motor"]),
        "anchors_sensory": len(anch["sensory"]),
        "legs": list(legs),
        "anchor_reuse_max": max(
            collections.Counter(d["anchor"] for d in detail).values(), default=0
        ),
        "relays": len(relays),
        "edges": detail,
    }


def relayed(dome: Dome, relays: list[tuple[int, int, int]]) -> Dome:
    """The mask with the relay layer on it, re-assembled from the widened list.

    `Dome._assemble` is the shipped constructor's own final step: it recomputes
    incidence, degrees, `Σ_e m_e` and the reserve mask from `(spec, cells,
    edges)`. Adding edges and calling it is therefore the *same* mask machinery
    the shipped dome gets, not a rig copy of it — the lesson #555 wrote up when
    the rig's parameterised allocator was deleted in favour of `src/`.

    A relay is an `INTERIOR` edge because that is what it is by ADR-0003's test:
    its disagreement is cleared cell-to-cell, immediately, like every other
    interior lane. It is not a new edge kind and this ticket does not propose one.
    """
    edges = list(dome.edges)
    next_id = len(edges)
    for u, v, m in relays:
        edges.append(Edge(id=next_id, u=int(u), v=int(v), m=int(m), kind=EdgeKind.INTERIOR))
        next_id += 1
    return Dome._assemble(dome.spec, dome.cells, tuple(edges))


def budget_read(dome: Dome, base: Dome) -> dict:
    """What the relay layer costs against `Σ_e m_e <= capacity_budget`.

    B35 §6 prices the relay count with this bound and calls #311's anti-hub
    objection self-enforcing. Whether it *is* self-enforcing on this dome is a
    number, so here it is: how many cells the layer pushes over `B`, and by how
    much. `p` is untouched — since #556 the reserve is a constant, not
    `n - Σ_e m_e` — so a relay costs capacity and never privacy.
    """
    B = int(dome.spec.capacity_budget)
    sums = np.array([dome.stalk_sums[c] for c in dome.predicting], dtype=float)
    before = np.array([base.stalk_sums[c] for c in base.predicting], dtype=float)
    over = sums - B
    return {
        "capacity_budget": B,
        "sum_m_median": float(np.median(sums)),
        "sum_m_max": float(sums.max()),
        "sum_m_median_before": float(np.median(before)),
        "sum_m_max_before": float(before.max()),
        "cells_over_budget": int((sums > B).sum()),
        "worst_overshoot": float(max(0.0, over.max())),
        "k_v_median": float(
            np.median([dome._permitted[c] for c in dome.predicting])
        ),
        "degree_median": float(np.median([dome.degrees[c] for c in dome.predicting])),
    }


def bridges(dome: Dome, edge_ids: list[int] | None = None) -> set[int]:
    """Tarjan on the cell graph: which edges lie on no cycle at any length.

    `b40_stranded.py` runs this on the **wide interior** subgraph, which is the
    population B34's criterion carves. A relay at `m = 1` is not in that
    population, so the test that matters for the layout is the one on the graph
    the relay actually joins: run it on the whole mask and report the relay
    edges' verdict. Both are reported by :func:`bridge_census`.
    """
    ids = set(range(len(dome.edges))) if edge_ids is None else set(edge_ids)
    adj: dict[int, list[tuple[int, int]]] = collections.defaultdict(list)
    for e in dome.edges:
        if e.id in ids:
            adj[e.u].append((e.v, e.id))
            adj[e.v].append((e.u, e.id))
    disc: dict[int, int] = {}
    low: dict[int, int] = {}
    found: set[int] = set()
    timer = 0
    for root in list(adj):
        if root in disc:
            continue
        # Iterative DFS: the dome is small but recursion here is gratuitous.
        stack = [(root, -1, iter(adj[root]))]
        disc[root] = low[root] = timer
        timer += 1
        while stack:
            node, via, it = stack[-1]
            advanced = False
            for nxt, eid in it:
                if eid == via:
                    continue
                if nxt in disc:
                    low[node] = min(low[node], disc[nxt])
                    continue
                disc[nxt] = low[nxt] = timer
                timer += 1
                stack.append((nxt, eid, iter(adj[nxt])))
                advanced = True
                break
            if advanced:
                continue
            stack.pop()
            if stack:
                parent, pvia, _ = stack[-1]
                low[parent] = min(low[parent], low[node])
                if low[node] > disc[parent]:
                    found.add(via)
    return found


def bridge_census(dome: Dome, relay_ids: list[int], base_edges: int) -> dict:
    """The layout's own falsifier: is a relay edge a bridge?

    B40's finding is that a bridge is permanently at the probe floor whatever the
    route trains, because there is no cycle for the criterion to count over. A
    `tree` layout must read every relay a bridge and a `pair` layout none; if
    `pair` reads any, its second path is not disjoint and the layout is wrong.
    """
    all_bridges = bridges(dome)
    wide_ids = [
        e.id
        for e in dome.edges
        if e.kind is EdgeKind.INTERIOR and e.m > 1 and not dome.cells[e.u].is_boundary
        and not dome.cells[e.v].is_boundary
    ]
    wide_bridges = bridges(dome, wide_ids)
    relay_set = set(relay_ids)
    return {
        "relay_edges": len(relay_ids),
        "relay_edges_that_are_bridges": len(relay_set & all_bridges),
        "bridges_total": len(all_bridges),
        "bridges_among_original": len(
            {b for b in all_bridges if b < base_edges}
        ),
        "wide_interior_edges": len(wide_ids),
        "wide_bridges": len(wide_bridges),
    }


def local_cycle_census(dome: Dome, relay_ids: list[int]) -> dict:
    """**The sharp form of B40's constraint, and it is not the bridge test.**

    B40's words are that *a relay edge is a bridge by construction*, so a
    tree-shaped relay network leaves every relay edge counting over an empty set.
    On this surface that antecedent cannot be met: the dome is already connected,
    so a relay is a **chord**, never a bridge, whatever the layout — the bridge
    test below reads 0/N on `tree` as well as `pair`.

    The constraint survives in a stronger form. B34's criterion does not count
    cycles; it counts **short local** cycles — length `<= MAX_LEN` and inside the
    radius-2 neighbourhood (`b33.local_cycles`). A relay is long by definition:
    the shortest cycle through a relay joining cells `d` hops apart has `d + 1`
    edges, so unless a relay is laid across a *short* gap, or two relays land
    close enough to close on each other, the criterion counts over an empty set
    **exactly as on a bridge, and for the same practical reason**.

    So this is the census that matters: cycles through each relay edge at
    `MAX_LEN`, and how many survive the locality filter.
    """
    b40e, b33 = _cycle_modules()
    adj: dict[int, list[tuple[int, int]]] = collections.defaultdict(list)
    interior = [
        e
        for e in dome.edges
        if e.kind is EdgeKind.INTERIOR
        and not dome.cells[e.u].is_boundary
        and not dome.cells[e.v].is_boundary
    ]
    for e in interior:
        adj[e.u].append((e.v, e.id))
        adj[e.v].append((e.u, e.id))
    per_edge, per_edge_local = {}, {}
    for eid in relay_ids:
        cycles = b40e.cycles_through(adj, dome, eid)
        local = [c for c in cycles if b33.local_cycles(dome, [c], b33.RADIUS)]
        per_edge[eid] = len(cycles)
        per_edge_local[eid] = len(local)
    if not relay_ids:
        return {"relays": 0}
    short = np.array(list(per_edge.values()), dtype=float)
    loc = np.array(list(per_edge_local.values()), dtype=float)
    return {
        "relays": len(relay_ids),
        "max_len": int(b40e.MAX_LEN),
        "radius": int(b33.RADIUS),
        "short_cycles_median": float(np.median(short)),
        "short_cycles_max": float(short.max()),
        "relays_with_no_short_cycle": int((short == 0).sum()),
        "local_cycles_median": float(np.median(loc)),
        "local_cycles_max": float(loc.max()),
        "relays_with_no_local_cycle": int((loc == 0).sum()),
    }


def _cycle_modules():
    """`b40_enumerate` and `b33_coexist`, loaded the way every T6 script loads them."""
    import importlib.util

    def load(name: str, path: Path):
        if name in sys.modules:
            return sys.modules[name]
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    b33 = load("b43_b33", _HERE / "b33_coexist.py")
    b40e = load("b43_enum", _HERE / "b40_enumerate.py")
    return b40e, b33


def world_read(dome: Dome, depth: dict[int, int], cohort: list[int] | None = None) -> dict:
    """Clause 1: `world_loop(c)` off the mask, and `|loop(c)|` beside it.

    Both are exact integers and neither takes a run. The *deepest* cells are
    reported separately because they are what the relay is for and what ADR-0026's
    enumeration puts at 15–16: a median that moves while the tail does not would
    be a relay buying nothing where the bar actually binds.
    """
    wl = ll.world_loops(dome)
    lp = ll.loops(dome)
    vals = np.array(sorted(wl.lengths.values()), dtype=float)
    dmax = max(depth[c] for c in wl.lengths if c in depth)
    # **The cohort is pinned to the baseline, not recomputed.** An earlier version
    # took the deepest cells *of the relayed graph*, which is a moving set: adding
    # relays makes the apex shallow, so "deepest" silently became the mid-depth
    # cells the relay never served and the reading understated the buy by five
    # ticks. The cohort a relay is judged on is the cells that were deepest
    # **before** it was laid — the apex ADR-0026's enumeration puts at 15–16.
    deep = (
        list(cohort)
        if cohort is not None
        else [c for c in wl.lengths if depth.get(c, -1) >= dmax - 1]
    )
    deep = [c for c in deep if c in wl.lengths]
    deep_vals = np.array([wl.lengths[c] for c in deep], dtype=float)
    excess = [wl.lengths[c] - lp.lengths[c] for c in lp.lengths if c in wl.lengths]
    return {
        "world_tick": wl.world_tick,
        "population": int(len(wl.lengths)),
        "world_loop_median": float(np.median(vals)),
        "world_loop_max": float(vals.max()),
        "world_loop_min": float(vals.min()),
        "world_loop_mean": float(vals.mean()),
        "deepest_depth": int(dmax),
        "deepest_cells": len(deep),
        "world_loop_deepest_median": float(np.median(deep_vals)),
        "world_loop_deepest_max": float(deep_vals.max()),
        "loop_median": float(np.median(list(lp.lengths.values()))),
        "loop_max": float(max(lp.lengths.values())),
        "world_loop_excess_max": float(max(excess)) if excess else float("nan"),
        "depth_max": int(max(depth[c] for c in dome.predicting)),
        "depth_median": float(np.median([depth[c] for c in dome.predicting])),
        "histogram": {
            str(int(v)): int((vals == v).sum()) for v in sorted(set(vals.tolist()))
        },
    }


def relay_load(dome: Dome, base_depth: dict[int, int], relay_ids: list[int]) -> dict:
    """B20's guarded end, measured: how much of the fleet leans on one relay.

    B35 §6 names [B20](#569)'s joint-span collapse as the failure mode at the
    sparse end — six of eight apexes taking all sixteen chains through one shared
    three-hop tail. The same shape here is a fleet whose shortest world loops all
    run through one relay edge. Counted by removing each relay edge in turn and
    reading how many cells' `world_loop` gets longer.
    """
    full = ll.world_loops(dome).lengths
    loads: dict[int, int] = {}
    for eid in relay_ids:
        kept = tuple(e for e in dome.edges if e.id != eid)
        # Re-id so `Dome._assemble`'s incidence stays consistent.
        renum = tuple(
            Edge(id=i, u=e.u, v=e.v, m=e.m, kind=e.kind) for i, e in enumerate(kept)
        )
        without = ll.world_loops(Dome._assemble(dome.spec, dome.cells, renum)).lengths
        loads[eid] = sum(
            1 for c in full if c in without and without[c] > full[c]
        )
    if not loads:
        return {"relays": 0}
    vals = np.array(list(loads.values()), dtype=float)
    return {
        "relays": len(loads),
        "max_load": float(vals.max()),
        "median_load": float(np.median(vals)),
        "carrying_relays": int((vals > 0).sum()),
        "share_on_worst": float(vals.max() / max(1, len(full))),
    }


def build(
    layout: str, depth_floor: int, relay_m: int, spec: DomeSpec, sites: int = SITES
) -> dict:
    base = build_graph(spec)
    base_depth = depth_from_rim(base)
    relays, record = lay_relays(base, layout, depth_floor, relay_m, sites)
    dome = relayed(base, relays) if relays else base
    relay_ids = list(range(len(base.edges), len(dome.edges)))
    depth = depth_from_rim(dome)
    base_dmax = max(base_depth[c] for c in base.predicting)
    cohort = sorted(c for c in base.predicting if base_depth[c] >= base_dmax - 1)
    out = {
        "layout": layout,
        "spec": {
            "capacity_budget": int(spec.capacity_budget),
            "private_reserve": int(spec.private_reserve),
        },
        "layout_record": record,
        "cells": len(dome.cells),
        "edges": len(dome.edges),
        "edges_added": len(relay_ids),
        "budget": budget_read(dome, base),
        "bridges": bridge_census(dome, relay_ids, len(base.edges)),
        "relay_cycles": local_cycle_census(dome, relay_ids),
        "cohort": {"cells": len(cohort), "baseline_depth_min": int(base_dmax - 1)},
        "clause1_world": world_read(dome, depth, cohort),
        "relay_load": relay_load(dome, base_depth, relay_ids),
    }
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["layouts"])
    p.add_argument("--layouts", nargs="+", default=list(LAYOUTS))
    p.add_argument("--depth", type=int, default=DEPTH)
    p.add_argument("--relay-m", type=int, default=RELAY_M)
    p.add_argument("--sites", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    p.add_argument("--json", type=str, default="607-layouts.json")
    args = p.parse_args()

    spec = DomeSpec()
    results = []
    print(
        f"{'layout':<10} {'sites':>5} {'+e':>4} | {'wl_med':>6} {'wl_max':>6} "
        f"{'deep_med':>8} {'dmax':>4} | {'no-cyc':>6} {'no-loc':>6} {'brdg':>5} "
        f"| {'over_B':>6} {'load':>5}"
    )
    for layout in args.layouts:
        site_list = [0] if layout == "none" else args.sites
        for sites in site_list:
            rec = build(layout, args.depth, args.relay_m, spec, sites)
            rec["sites"] = sites
            results.append(rec)
            w = rec["clause1_world"]
            b = rec["bridges"]
            bu = rec["budget"]
            rc = rec["relay_cycles"]
            print(
                f"{layout:<10} {sites:>5} {rec['edges_added']:>4} | "
                f"{w['world_loop_median']:>6.1f} {w['world_loop_max']:>6.1f} "
                f"{w['world_loop_deepest_median']:>8.1f} {w['depth_max']:>4} | "
                f"{rc.get('relays_with_no_short_cycle', 0):>6} "
                f"{rc.get('relays_with_no_local_cycle', 0):>6} "
                f"{b['relay_edges_that_are_bridges']:>5} | "
                f"{bu['cells_over_budget']:>6} "
                f"{rec['relay_load'].get('max_load', 0):>5.0f}",
                flush=True,
            )
    Path(_HERE / args.json).write_text(
        json.dumps({"issue": 607, "reading": "relay layouts, clause 1", "layouts": results}, indent=1),
        encoding="utf-8",
    )
    print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
