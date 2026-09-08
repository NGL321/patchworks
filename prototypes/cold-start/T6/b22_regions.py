"""T6 ([B22](#571)): coherent structures -- how large a region a direction survives.

[B17](#565) found that every composed-rank reading on [#532](#532) is a property of
the transport *operator*, and separately that `diagnostics.py:762-832` computes
`dim H⁰` over the **whole** predicting subcomplex only -- no subset, no
component-wise variant -- while the construction bound it is quoted against
(`diagnostics.py:967`) is **exactly** the private-dimension count. So every
`dim H⁰` this map has published is dominated by directions *no edge constrains*,
globally consistent for free and carrying no coordination at all.

This module builds the two objects that were missing.

## 1. The earned part and the trivial part

A cell `v`'s incident restriction maps, stacked, span a subspace `U_v ⊆ R^n` of
dimension `k_v`. Its orthogonal complement `U_v^⊥` is the cell's **private**
block: every vector there is killed by every incident map, so any assignment
supported in `⊕_v U_v^⊥` is a global section for free. Write

    trivial = Σ_v dim U_v^⊥ = Σ_v (n − k_v)          (over predicting cells)
    earned  = dim H⁰ − trivial

`trivial ⊆ ker δ` always, so `earned ≥ 0` and the split is exact -- not an
estimate. Note `trivial` is measured here as a **rank**, `n − rank(stack of v's
incident maps)`; the figure the record quotes, `Σ_v private_dimensions`, is the
*combinatorial* lower bound `max(0, n − Σ_e m_e)`, and the two are reported side
by side because their gap is itself a finding.

## 2. Regions, and what "stays consistent" means

A **region** `R` is a set of predicting cells. A section over `R` is one vector
`x_v ∈ R^n` per cell of `R` satisfying the agreement condition on the edges `R`
carries; `H⁰(R) = ker δ_R`. Two row conventions are computed, because they
bracket the question rather than settling it:

* **`internal`** -- rows are the edges with **both** ends in `R`. This is the
  sheaf-correct reading of "a region holds this direction consistently": what
  lies outside `R` is not `R`'s business.
* **`sealed`** -- also includes the edges from `R` to the dome's **boundary**
  cells, homogeneously. This is `diagnostics.py`'s own convention (its `δ_P` has
  a row for every edge, boundary-incident ones included), so `sealed` at
  `R = all predicting cells` reproduces the map's `dim H⁰` exactly, and the
  instrument is checked against `Diagnostics.whole_graph` on that identity.

For a seed cell `c`, the **trace** is what `R` still permits at `c`:

    P_c(R) = { s(c) : s ∈ H⁰(R) } ⊆ R^n

`P_c(R)` shrinks monotonically as `R` grows, and it always contains `U_c^⊥`
(a vector private at `c`, zero elsewhere, is a section of any `R`). Hence

    P_c(R) = U_c^⊥ ⊕ (P_c(R) ∩ U_c)

exactly, and the **earned trace** `E_c(R) = P_c(R) ∩ U_c` -- the coordinated
directions still available -- has

    dim E_c(R) = dim P_c(R) − (n − k_c),      dim E_c({c}) = k_c.

Its dimension is computed without ever forming a null space, from two ranks:

    dim P_c(R) = n − rank δ_R + rank δ_R^{(c deleted)}

## 3. Why a nested family answers "the distribution of region sizes"

Grow `R` by BFS balls around `c`. The traces are **nested**,
`E_c(R_0) ⊇ E_c(R_1) ⊇ …`, so `dim E_c(R_r)` *is* the count of directions at `c`
that survive to radius `r`. The number whose maximal region is exactly `R_r` is
`dim E_c(R_r) − dim E_c(R_{r+1})`, and the region-size distribution follows from
the dimensions alone -- no direction is ever named or tracked.

A direction surviving to radius `r` is held consistently across **every** cell of
`R_r`, so the level span of `R_r` is the level span of that direction. That is
item 3 of the ticket, and it is exact rather than inferred.

**Balls are one family of regions and a maximal region need not be a ball.**
`greedy` grows an irregular region instead -- accepting any neighbour that keeps
`dim E_c ≥ target` -- and is run on a subsample as a check on the ball reading.
Both are *maximal* (no single cell can be added), neither is *maximum*.

## What this module does not measure

It reads the **restriction maps** only, exactly as `composed_reads` does. It does
not touch `agent.sheaf.stalks`, so nothing here says whether traffic travels the
directions a region retains -- B17's finding applies to this instrument too, and
is not repaired by it. See [B19](#568) for state against emitted rank.

Usage::

    PYTHONPATH=src python prototypes/cold-start/T6/b22_regions.py check
    PYTHONPATH=src python prototypes/cold-start/T6/b22_regions.py construction --arms shipped reserve_p8
    PYTHONPATH=src python prototypes/cold-start/T6/b22_regions.py trained --arms reserve --ticks 20000
"""

from __future__ import annotations

import argparse
import collections
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
_T0, _T1, _T2, _T3, _T4 = (_HERE.parent / n for n in ("T0", "T1", "T2", "T3", "T4"))
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
t4_trained = _load("t4_trained", _T4 / "trained.py")
arms_mod = _load("t6_arms", _HERE / "arms.py")

from patchworks.diagnostics import Diagnostics  # noqa: E402
from patchworks.restriction import pair_index  # noqa: E402

#: `δ`'s two ends carry opposite signs, as `diagnostics._SIGN` does. Agreement is
#: `W_u x_u − W_v x_v = 0`, so the sign is not cosmetic.
_SIGN = (1.0, -1.0)


def _rank(matrix: np.ndarray) -> int:
    """`rank M`, through `M Mᵀ` and in float64 -- `diagnostics._rank`'s convention.

    Deliberately the same routine and the same tolerance as the reading this
    ticket is re-cutting, so the split in :func:`whole_graph_split` is a split of
    the map's own number and not of a differently-rounded one.
    """
    if matrix.size == 0 or matrix.shape[0] == 0:
        return 0
    gram = matrix @ matrix.T
    eigenvalues = np.linalg.eigvalsh(gram)
    top = float(eigenvalues.max())
    if top <= 0.0:
        return 0
    tol = top * max(matrix.shape) * np.finfo(np.float64).eps
    return int((eigenvalues > tol).sum())


# -- the graph objects --------------------------------------------------------


class Surface:
    """One arm's coboundary, its column layout, and the graph it sits on.

    Built once per surface and re-read at every checkpoint: the *shape* is a
    construction fact and only the numbers in `delta` move, so
    :meth:`refresh` rewrites the blocks in place rather than rebuilding.
    """

    def __init__(self, dome, maps):
        self.dome = dome
        self.n = int(dome.shape.n)
        self.predicting = list(dome.predicting)
        self.column = {cid: i * self.n for i, cid in enumerate(self.predicting)}
        self.rows = sum(int(e.m) for e in dome.edges)
        self.cols = len(self.predicting) * self.n

        # Row span of each edge, and where each end's block lands.
        self.edge_rows: dict[int, tuple[int, int]] = {}
        self._blocks: list[tuple[int, int, int, int, float]] = []
        #: edge id -> the predicting ends it has (0, 1 or 2 of them)
        self.edge_ends: dict[int, tuple[int, ...]] = {}
        at = 0
        for edge in dome.edges:
            self.edge_rows[edge.id] = (at, int(edge.m))
            ends = []
            for side, cid in enumerate((edge.u, edge.v)):
                if cid in self.column:
                    ends.append(cid)
                    self._blocks.append(
                        (pair_index(edge.id, side), at, int(edge.m), self.column[cid], _SIGN[side])
                    )
            self.edge_ends[edge.id] = tuple(ends)
            at += int(edge.m)

        # The predicting-cell adjacency the balls are grown on.
        self.neighbours: dict[int, set[int]] = {c: set() for c in self.predicting}
        for eid, ends in self.edge_ends.items():
            if len(ends) == 2:
                self.neighbours[ends[0]].add(ends[1])
                self.neighbours[ends[1]].add(ends[0])

        self.level = {c: int(dome.cells[c].index.level) for c in self.predicting}
        self.incident = {c: sorted(dome.incident[c]) for c in self.predicting}
        self.delta = np.zeros((self.rows, self.cols), dtype=np.float64)
        #: `rank δ_R` by region, valid only for the maps currently in `delta`.
        self.rank_cache: dict[tuple[frozenset[int], bool], int] = {}
        self.refresh(maps)

    def refresh(self, maps) -> None:
        """Rewrite `delta` from the current restriction maps. Shape never moves."""
        self.rank_cache.clear()
        with torch.no_grad():
            weights = maps.maps.detach().to(torch.float64).numpy()
        self.delta[:] = 0.0
        for pair, at, m, column, sign in self._blocks:
            self.delta[at : at + m, column : column + self.n] += sign * weights[pair, :m, : self.n]
        # `k_v` per cell: the rank of the stack of v's incident maps, which is the
        # dimension of the block anything at all can read.
        self.k = {}
        for cid in self.predicting:
            blocks = []
            for eid in self.incident[cid]:
                m = int(self.dome.edges[eid].m)
                if m == 0:
                    continue
                side = 0 if self.dome.edges[eid].u == cid else 1
                blocks.append(weights[pair_index(eid, side)][:m, : self.n])
            self.k[cid] = _rank(np.concatenate(blocks, axis=0)) if blocks else 0

    # -- row selection --------------------------------------------------------

    def region_rows(self, region: set[int], *, sealed: bool) -> np.ndarray:
        """Row indices of the edges a region carries, under one of the two conventions."""
        picked: list[np.ndarray] = []
        for eid, ends in self.edge_ends.items():
            at, m = self.edge_rows[eid]
            if m == 0:
                continue
            inside = [c for c in ends if c in region]
            if not inside:
                continue
            if len(ends) == 2 and len(inside) == 2:
                keep = True
            elif len(ends) == 1 and sealed:
                # An edge from this region to a boundary cell: `δ_P` carries it.
                keep = True
            else:
                keep = False
            if keep:
                picked.append(np.arange(at, at + m))
        return np.concatenate(picked) if picked else np.empty(0, dtype=int)

    def region_cols(self, cells: list[int]) -> np.ndarray:
        return np.concatenate([np.arange(self.column[c], self.column[c] + self.n) for c in cells])


# -- item 1: the earned part and the trivial part -----------------------------


def whole_graph_split(surface: Surface) -> dict:
    """`dim H⁰` cut into the part no edge constrains and the part that is earned."""
    delta_p = surface.delta
    rank = _rank(delta_p)
    dim_h0 = surface.cols - rank
    dim_h1 = surface.rows - rank
    trivial = sum(surface.n - surface.k[c] for c in surface.predicting)
    combinatorial = int(surface.dome.private_dimensions.sum())
    permitted = [int(surface.dome._permitted[c]) for c in surface.predicting]
    # The counting control, and the one number that decides whether `earned` means
    # anything. Delete the private columns -- they are null by construction -- and a
    # surface whose remaining rows are in general position has
    #   earned = max(0, (columns − trivial) − rows).
    # Any *coordination* in the maps would have to show up as `earned` **above**
    # this: sections that exist because the maps agree, not because the system is
    # underdetermined. Equality means the earned part is slack, not structure.
    generic = max(0, (surface.cols - trivial) - surface.rows)
    return {
        "rows": surface.rows,
        "columns": surface.cols,
        "rank_delta": rank,
        "dim_h0": dim_h0,
        "dim_h1": dim_h1,
        "trivial": int(trivial),
        "earned": int(dim_h0 - trivial),
        "earned_fraction": float((dim_h0 - trivial) / max(dim_h0, 1)),
        "earned_generic": int(generic),
        "earned_above_generic": int(dim_h0 - trivial - generic),
        "quoted_construction_bound": combinatorial,
        "k_v_median": float(np.median(surface_k := np.array([surface.k[c] for c in surface.predicting]))),
        "k_v_min": int(surface_k.min()),
        "k_v_max": int(surface_k.max()),
        "permitted_median": float(np.median(permitted)),
    }


# -- item 2/3: the trace filtration -------------------------------------------


def trace_dim(surface: Surface, region: set[int], seed: int, *, sealed: bool) -> int:
    """`dim P_c(R)` -- how many directions at `c` some section over `R` still allows.

    `rank δ_R` does not depend on the seed, and balls grown from different seeds run
    through the same regions near the top (every one of them ends at the whole
    graph), so it is cached on the surface. The seed-dependent half is not.
    """
    rows = surface.region_rows(region, sealed=sealed)
    if rows.size == 0:
        return surface.n
    cells = sorted(region)
    key = (frozenset(region), sealed)
    rank_all = surface.rank_cache.get(key)
    if rank_all is None:
        rank_all = _rank(surface.delta[np.ix_(rows, surface.region_cols(cells))])
        surface.rank_cache[key] = rank_all
    without = [c for c in cells if c != seed]
    if not without:
        # `c` alone with sealed rows: `W x = 0` is the only condition.
        return surface.n - rank_all
    block_wo = surface.delta[np.ix_(rows, surface.region_cols(without))]
    return surface.n - rank_all + _rank(block_wo)


def _span(surface: Surface, region: set[int], rows: int) -> dict:
    """Size, level span, and **how many constraints the region actually carries**.

    `rows` is the third of these and it is not decoration: the dome's lateral
    edges run at `m = 1` while its interior edges are several lanes wide, so two
    regions of the same *cell* count can differ several-fold in constraint count.
    A `ladder` and a `lateral` region matched on cells are therefore **not**
    matched on constraints, and the readout has to say which way the handicap
    runs rather than quietly compare them as equals.
    """
    levels = sorted({surface.level[c] for c in region})
    return {
        "cells": len(region),
        "rows": int(rows),
        "levels": len(levels),
        "level_min": levels[0],
        "level_max": levels[-1],
    }


def _grow(surface: Surface, region: set[int], seed: int, how: str) -> set[int] | None:
    """One step of a growth family, or `None` when the family is exhausted.

    Three families, because **the ball confounds the ticket's two questions.** On
    this dome a radius-2 ball already spans three or four levels, so "large" and
    "cuts across levels" are not independent properties of a ball and item 3
    cannot be read off item 2. The other two families separate them at matched
    size:

    * `ball` -- ordinary BFS. Size grows fast, level span comes along for free.
    * `lateral` -- BFS confined to the seed's **own level**. Same-level cells
      only, so level span is pinned at 1 however large it gets.
    * `ladder` -- a one-cell-at-a-time path that always steps to the
      highest-level neighbour available. Maximum level span per cell spent.

    Comparing `lateral` and `ladder` at the same number of cells is the direct
    test of the user's claim that degrees of abstraction are not a ladder: if a
    direction survives as far across levels as it does within one, the claim
    holds on this surface; if `ladder` dies first, abstraction is behaving like a
    ladder after all.
    """
    if how == "ball":
        grown = set(region)
        for c in region:
            grown |= surface.neighbours[c]
        return None if grown == region else grown
    if how == "lateral":
        level = surface.level[seed]
        grown = set(region)
        for c in region:
            grown |= {u for u in surface.neighbours[c] if surface.level[u] == level}
        return None if grown == region else grown
    if how == "ladder":
        candidates = {u for c in region for u in surface.neighbours[c]} - region
        if not candidates:
            return None
        best = max(candidates, key=lambda u: (surface.level[u], -u))
        return region | {best}
    raise ValueError(how)


def ball_filtration(
    surface: Surface, seed: int, *, sealed: bool, how: str = "ball", max_steps: int | None = None
) -> dict:
    """Grow a region around `seed` and record the earned trace at each step.

    Stops when the earned trace is empty (nothing survives further) or the family
    is exhausted. The `steps` list is the whole reading: nested traces mean
    `earned` at step `r` **is** the count of directions whose region reaches at
    least that far.
    """
    # `ladder` adds one cell per step, so its step count *is* its cell count and an
    # arm where nothing ever dies would walk it to 150 -- 150 rank pairs on a matrix
    # growing to `2477 x 4800`, hours per seed. It is capped instead, and the cap is
    # recorded as the stop reason so a censored region is never read as a dead one.
    # The cap costs nothing the matched-size test needs: `lateral` is bounded by its
    # level's size, well under 40 on every level of this dome.
    if max_steps is None:
        max_steps = 40 if how in ("ladder", "lateral") else 160
    k_c = surface.k[seed]
    region = {seed}
    steps = []
    radius = 0
    while True:
        dim_p = trace_dim(surface, region, seed, sealed=sealed)
        earned = dim_p - (surface.n - k_c)
        rows = surface.region_rows(region, sealed=sealed).size
        steps.append(
            {"radius": radius, **_span(surface, region, rows), "dim_p": dim_p, "earned": earned}
        )
        if earned <= 0:
            stopped = "earned_zero"
            break
        if radius >= max_steps:
            stopped = "cap"
            break
        grown = _grow(surface, region, seed, how)
        if grown is None:
            # The family ran out of graph, not out of consistency. A region that
            # ends this way is a *lower* bound on where the direction would have
            # died, and the two must never be pooled as if they were the same
            # event -- `lateral` almost always ends here, because a level is
            # finite and small.
            stopped = "exhausted"
            break
        region = grown
        radius += 1
    return {
        "seed_cell": seed,
        "level": surface.level[seed],
        "k_v": k_c,
        "how": how,
        "stopped": stopped,
        "available": _family_size(surface, seed, how),
        "steps": steps,
    }


def _family_size(surface: Surface, seed: int, how: str) -> int:
    """How many cells this growth family could ever reach from `seed`."""
    if how == "lateral":
        return sum(1 for c in surface.predicting if surface.level[c] == surface.level[seed])
    return len(surface.predicting)


def greedy_region(surface: Surface, seed: int, target: int, *, sealed: bool, cap: int = 60) -> dict:
    """A maximal *irregular* region holding at least `target` earned directions.

    Balls are one family and a maximal region need not be one; this accepts any
    neighbour that does not push the earned trace below `target`, in BFS order,
    repeating until a whole pass adds nothing. Maximal, not maximum, and
    order-dependent -- both stated rather than hidden.
    """
    k_c = surface.k[seed]
    if k_c < target:
        return {"seed_cell": seed, "target": target, "reachable": False}
    region = {seed}
    tried: set[int] = set()
    added = True
    tests = 0
    while added and len(region) < cap:
        added = False
        frontier = sorted({u for c in region for u in surface.neighbours[c]} - region - tried)
        for u in frontier:
            tried.add(u)
            trial = region | {u}
            tests += 1
            if trace_dim(surface, trial, seed, sealed=sealed) - (surface.n - k_c) >= target:
                region = trial
                added = True
        if added:
            tried &= region  # newly reachable cells deserve a fresh look
    return {
        "seed_cell": seed,
        "target": target,
        "reachable": True,
        "k_v": k_c,
        "tests": tests,
        **_span(surface, region, surface.region_rows(region, sealed=sealed).size),
    }


# -- pooling ------------------------------------------------------------------


def pool(filtrations: list[dict]) -> dict:
    """The region-size distribution, pooled over seeds, one entry per direction.

    A seed with `k_v` earned directions contributes `k_v` entries: for `d` in
    `1..k_v`, the size of the largest ball whose earned trace still has dimension
    `≥ d`. Directions that do not survive even one hop get the seed's own cell
    count, 1 -- they are consistent nowhere but at home.
    """
    sizes: list[int] = []
    levels: list[int] = []
    radii: list[int] = []
    censored: list[bool] = []
    per_seed = []
    for f in filtrations:
        steps = f["steps"]
        k_c = f["k_v"]
        # A direction still alive when growth stopped for a reason other than the
        # trace emptying is **right-censored**: its region is at least this big and
        # we do not know how much bigger. Kept as a flag rather than dropped,
        # because on `lateral` almost every direction is censored and a median that
        # hid that would be a lie about where directions die.
        last_step = steps[-1]
        seed_sizes = []
        for d in range(1, k_c + 1):
            surviving = [s for s in steps if s["earned"] >= d]
            last = surviving[-1] if surviving else steps[0]
            sizes.append(last["cells"])
            levels.append(last["levels"])
            radii.append(last["radius"])
            censored.append(f["stopped"] != "earned_zero" and last_step["earned"] >= d)
            seed_sizes.append(last["cells"])
        per_seed.append(
            {
                "seed_cell": f["seed_cell"],
                "level": f["level"],
                "k_v": k_c,
                "stopped": f["stopped"],
                "available": f["available"],
                "size_max": max(seed_sizes) if seed_sizes else 0,
                "size_median": float(np.median(seed_sizes)) if seed_sizes else 0.0,
                "earned_by_radius": [s["earned"] for s in steps],
                "cells_by_radius": [s["cells"] for s in steps],
                "levels_by_radius": [s["levels"] for s in steps],
            }
        )
    if not sizes:
        return {"directions": 0, "per_seed": per_seed}
    a = np.array(sizes, dtype=float)
    lv = np.array(levels, dtype=float)
    return {
        "directions": len(sizes),
        "seeds": len(filtrations),
        "size": {
            "min": int(a.min()),
            "median": float(np.median(a)),
            "p90": float(np.quantile(a, 0.90)),
            "max": int(a.max()),
            "mean": float(a.mean()),
            "histogram": dict(collections.Counter(int(x) for x in a)),
        },
        "levels_spanned": {
            "median": float(np.median(lv)),
            "max": int(lv.max()),
            "histogram": dict(collections.Counter(int(x) for x in lv)),
        },
        "radius": {"median": float(np.median(radii)), "max": int(max(radii))},
        "fraction_size_one": float((a <= 1).mean()),
        "fraction_censored": float(np.mean(censored)),
        "stopped": dict(collections.Counter(f["stopped"] for f in filtrations)),
        "per_seed": per_seed,
    }


def choose_seeds(surface: Surface, count: int, rng_seed: int = 0) -> list[int]:
    """A sample stratified by level, so the reading is not all rim or all apex."""
    rng = np.random.default_rng(rng_seed)
    by_level: dict[int, list[int]] = collections.defaultdict(list)
    for c in surface.predicting:
        by_level[surface.level[c]].append(c)
    levels = sorted(by_level)
    per = max(1, count // len(levels))
    picked: list[int] = []
    for lv in levels:
        pool_ = sorted(by_level[lv])
        take = min(per, len(pool_))
        picked.extend(int(x) for x in rng.choice(pool_, size=take, replace=False))
    return sorted(set(picked))


def matched_size(families: dict[str, list[dict]]) -> dict:
    """`lateral` against `ladder` at the same number of cells -- item 3's actual test.

    For each seed and each region size both families reach, the earned trace under
    each. `ladder_minus_lateral` pooled over all such matches is the statistic: `0`
    means spanning levels costs a direction nothing, negative means it costs.
    """
    lateral = {f["seed_cell"]: f for f in families.get("lateral", [])}
    ladder = {f["seed_cell"]: f for f in families.get("ladder", [])}
    deltas: list[int] = []
    row_ratio: list[float] = []
    pairs = []
    for cell, lat in lateral.items():
        lad = ladder.get(cell)
        if lad is None:
            continue
        by_size_lat = {s["cells"]: s for s in lat["steps"]}
        by_size_lad = {s["cells"]: s for s in lad["steps"]}
        for size in sorted(set(by_size_lat) & set(by_size_lad)):
            if size <= 1:
                continue
            sl, sd = by_size_lat[size], by_size_lad[size]
            d = sd["earned"] - sl["earned"]
            deltas.append(d)
            row_ratio.append(sd["rows"] / max(sl["rows"], 1))
            pairs.append(
                {
                    "seed_cell": cell,
                    "cells": size,
                    "lateral_earned": sl["earned"],
                    "ladder_earned": sd["earned"],
                    "lateral_rows": sl["rows"],
                    "ladder_rows": sd["rows"],
                    "ladder_levels": sd["levels"],
                }
            )
    if not deltas:
        return {"matches": 0}
    a = np.array(deltas, dtype=float)
    return {
        "matches": len(deltas),
        # >1 means the cross-level family is carrying MORE constraints per cell than
        # the within-level one -- the handicap runs against `ladder`, so a
        # `ladder_minus_lateral` at or above zero is a conservative reading of
        # "spanning levels costs nothing".
        "ladder_over_lateral_rows": float(np.median(row_ratio)),
        "ladder_minus_lateral": {
            "median": float(np.median(a)),
            "mean": float(a.mean()),
            "min": int(a.min()),
            "max": int(a.max()),
            "fraction_zero": float((a == 0).mean()),
            "fraction_negative": float((a < 0).mean()),
        },
        "pairs": pairs,
    }


def region_read(surface: Surface, label: str, seeds: list[int], *, sealed: bool, greedy: list[int]) -> dict:
    started = time.time()
    families = {
        how: [ball_filtration(surface, c, sealed=sealed, how=how) for c in seeds]
        for how in ("ball", "lateral", "ladder")
    }
    read = {
        "label": label,
        "convention": "sealed" if sealed else "internal",
        "whole_graph": whole_graph_split(surface),
        "balls": pool(families["ball"]),
        "lateral": pool(families["lateral"]),
        "ladder": pool(families["ladder"]),
        "matched_size": matched_size(families),
        "greedy": [
            greedy_region(surface, c, target, sealed=sealed)
            for c in greedy
            for target in (1, 2)
        ],
        "seconds": round(time.time() - started, 1),
    }
    return read


def _line(read: dict, prefix: str) -> str:
    w, b = read["whole_graph"], read["balls"]
    size = b.get("size", {})
    m = read.get("matched_size", {}).get("ladder_minus_lateral", {})
    lat = read.get("lateral", {}).get("size", {})
    lad = read.get("ladder", {}).get("size", {})
    return (
        f"{prefix} dim H0 {w['dim_h0']} = trivial {w['trivial']} + earned {w['earned']} "
        f"(generic {w['earned_generic']}, {w['earned_above_generic']:+d}) | ball cells "
        f"med {size.get('median', 0):.1f} max {size.get('max', 0)} lvl "
        f"{b.get('levels_spanned', {}).get('median', 0):.1f} | lateral med "
        f"{lat.get('median', 0):.1f} ladder med {lad.get('median', 0):.1f} | "
        f"ladder-lateral med {m.get('median', float('nan')):.2f}"
    )


# -- drivers ------------------------------------------------------------------


def run_check(args) -> None:
    """Does this instrument reproduce `Diagnostics.whole_graph` exactly?

    Nothing here is worth reading if `delta` is not the same matrix
    `diagnostics.py` builds. The `sealed` convention is that matrix by
    construction, so `dim H⁰` and `dim H¹` must agree to the integer, and the
    trace machinery must return `n` for a lone cell with no rows.
    """
    for arm in args.arms:
        env, agent = arms_mod.build_arm(arm, args.seed)
        try:
            surface = Surface(agent.dome, agent.sheaf.maps)
            mine = whole_graph_split(surface)
            theirs = Diagnostics(agent.sheaf).whole_graph()
            ok_h0 = mine["dim_h0"] == theirs.dim_h0
            ok_h1 = mine["dim_h1"] == theirs.dim_h1
            print(
                f"  {arm:>13}: dim H0 mine {mine['dim_h0']} theirs {theirs.dim_h0} "
                f"{'OK' if ok_h0 else 'MISMATCH'} | dim H1 mine {mine['dim_h1']} "
                f"theirs {theirs.dim_h1} {'OK' if ok_h1 else 'MISMATCH'}",
                flush=True,
            )
            print(
                f"                 rows {mine['rows']} | trivial (rank) {mine['trivial']} vs "
                f"quoted construction bound {mine['quoted_construction_bound']} | earned "
                f"{mine['earned']} vs generic {mine['earned_generic']} "
                f"({mine['earned_above_generic']:+d}) | k_v median {mine['k_v_median']:.1f}",
                flush=True,
            )
            c = surface.predicting[0]
            lone = trace_dim(surface, {c}, c, sealed=False)
            print(
                f"                 lone-cell trace (internal) {lone} == n "
                f"{surface.n} {'OK' if lone == surface.n else 'MISMATCH'}",
                flush=True,
            )
            if not (ok_h0 and ok_h1):
                raise SystemExit("instrument does not reproduce diagnostics.whole_graph")
        finally:
            env.close()


def run_construction(args) -> None:
    record = {
        "issue": 571,
        "reading": "earned vs trivial dim H0, and the region-size distribution, at construction",
        "seeds_sampled": args.seed_count,
        "rows": [],
    }
    for arm in args.arms:
        env, agent = arms_mod.build_arm(arm, args.seed)
        try:
            surface = Surface(agent.dome, agent.sheaf.maps)
            _, reserve_p = arms_mod.ARMS[arm]
            cells = choose_seeds(surface, args.seed_count)
            greedy = cells[:: max(1, len(cells) // 4)][:4] if args.greedy else []
            row = {"arm": arm, "reserve_p": reserve_p, "seed": args.seed, "reads": {}}
            for sealed in (False, True):
                key = "sealed" if sealed else "internal"
                read = region_read(surface, f"{arm} {key}", cells, sealed=sealed, greedy=greedy)
                row["reads"][key] = read
                print(_line(read, f"  {arm:>13} {key:>8}:"), flush=True)
            record["rows"].append(row)
        finally:
            env.close()
        args.out.write_text(json.dumps(record, indent=1))
    print(f"wrote {args.out.name}", flush=True)


def train_one(arm: str, args) -> None:
    """One arm, trained, read at every checkpoint. Sequential, and written as it lands."""
    started = time.time()
    out = _HERE / f"571-regions-{arm}-{args.condition}-seed{args.seed}-{args.ticks}.json"
    if out.exists():
        print(f"[B22] {out.name} already at the horizon, skipping", flush=True)
        return
    inflight = out.with_suffix(".inflight.json")
    cond = t3.CONDITIONS[args.condition]
    env, agent = arms_mod.build_arm(arm, args.seed)
    try:
        if cond["pin"]:
            t1.pin_drive_edges(agent)
        surface = Surface(agent.dome, agent.sheaf.maps)
        cells = choose_seeds(surface, args.seed_count)
        _, reserve_p = arms_mod.ARMS[arm]
        record = {
            "issue": 571,
            "reading": "earned vs trivial dim H0, and region sizes, under training",
            "arm": arm,
            "reserve_p": reserve_p,
            "condition": args.condition,
            "seed": args.seed,
            "ticks": args.ticks,
            "seeds_sampled": len(cells),
            "seed_cells": cells,
            "checkpoints": [],
        }
        recorder = t2.EdgeRecorder(agent)
        from patchworks.learning import PredictionRule, TransportRule

        bias = PredictionRule(agent.sheaf, operator_rate_ratio=cond["c"])
        transport = TransportRule(agent.sheaf)

        record["at_construction"] = {
            k: region_read(surface, f"construction {arm} {k}", cells, sealed=(k == "sealed"), greedy=[])
            for k in ("internal", "sealed")
        }
        for k, read in record["at_construction"].items():
            print(_line(read, f"  {arm} s{args.seed} construction {k:>8}:"), flush=True)

        ladder = [cp for cp in t3.CHECKPOINTS if cp <= args.ticks]
        if args.ticks not in ladder:
            ladder.append(args.ticks)
        # Two cadences, because the two halves of this reading cost two orders apart.
        # The split (item 1) is one rank of `δ` -- seconds -- so it is taken at every
        # checkpoint and the whole training curve is visible. The region filtration
        # (items 2-4) is a rank pair per seed per radius, minutes, so it is taken at
        # `--region-at` only. Nothing is inferred for the checkpoints in between.
        region_at = set(args.region_at) | {ladder[-1]}
        record["region_at"] = sorted(region_at)
        seen = 0
        for target in ladder:
            for _ in t0.teaching_read(agent, target - seen, args.seed + seen, recorder, bias, transport):
                pass
            seen = target
            surface.refresh(agent.sheaf.maps)
            entry = {"ticks": target, "whole_graph": whole_graph_split(surface), "reads": {}}
            w = entry["whole_graph"]
            print(
                f"  {arm} s{args.seed} @{target:>6}: dim H0 {w['dim_h0']} = trivial "
                f"{w['trivial']} + earned {w['earned']} (generic {w['earned_generic']}, "
                f"{w['earned_above_generic']:+d})",
                flush=True,
            )
            if target in region_at:
                for k in ("internal", "sealed"):
                    entry["reads"][k] = region_read(
                        surface, f"{arm} @{target} {k}", cells, sealed=(k == "sealed"), greedy=[]
                    )
                    print(_line(entry["reads"][k], f"  {arm} s{args.seed} @{target:>6} {k:>8}:"), flush=True)
            entry["elapsed_minutes"] = (time.time() - started) / 60.0
            record["checkpoints"].append(entry)
            inflight.write_text(json.dumps(record, indent=1))
        inflight.replace(out)
        print(f"wrote {out.name}", flush=True)
    finally:
        env.close()


def run_trained(args) -> None:
    """The arms **one at a time**: parallel long runs trip this box's memory guard."""
    for arm in args.arms:
        print(f"[B22] {arm} {args.condition} seed {args.seed}, {args.ticks} ticks", flush=True)
        train_one(arm, args)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)

    k = sub.add_parser("check", help="does this reproduce Diagnostics.whole_graph?")
    k.add_argument("--arms", nargs="+", default=["shipped", "reserve"])
    k.add_argument("--seed", type=int, default=42)
    k.set_defaults(func=run_check)

    c = sub.add_parser("construction", help="the split and the region sizes, no training")
    c.add_argument("--arms", nargs="+", default=["shipped", "doubling", "reserve"])
    c.add_argument("--seed", type=int, default=42)
    c.add_argument("--seed-count", type=int, default=24, dest="seed_count")
    c.add_argument("--greedy", action="store_true", help="also grow irregular maximal regions")
    c.add_argument("--out", type=Path, default=_HERE / "571-regions-construction.json")
    c.set_defaults(func=run_construction)

    t = sub.add_parser("trained", help="arms trained one at a time, read at every checkpoint")
    t.add_argument("--arms", nargs="+", default=["reserve"])
    t.add_argument("--condition", choices=sorted(t3.CONDITIONS), default="baseline")
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--ticks", type=int, default=20_000)
    t.add_argument("--seed-count", type=int, default=12, dest="seed_count")
    t.add_argument(
        "--region-at",
        type=int,
        nargs="+",
        default=[1000, 5000],
        dest="region_at",
        help="checkpoints at which to take the (expensive) region filtration; "
        "the horizon is always included",
    )
    t.set_defaults(func=run_trained)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
