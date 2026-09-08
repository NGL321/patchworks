"""T6 ([B29](#585)): **is the sheaf flat?** Holonomy, path-independence, and whether
composition is order-invariant.

`benchmarks/holonomy_read.py` has enumerated the interior cycle basis and read the
departure of each cycle's holonomy from the identity since #315. **Nothing has ever
related those readings to composed rank, to the joint span, or to `p`.** That is this
module: the same instrument, carried onto [B13](#560)'s reserve arms, read beside
[B20](#569)'s joint span and `composed_reads`' per-chain ER **on the same surface in
the same process**, so the correlation is not a correlation across runs.

## What is measured, in the ticket's five parts

1. **How far from flat.** :func:`surface_read` -- per cycle, never graph-averaged:
   `sigma_max`, `flatness = sigma_min/sigma_max`, `identification = ||UV^T - I||_F /
   sqrt(2m)` and `channel_return = |<u_1, v_1>|`, all four `holonomy_read.departures`
   verbatim, all four invariant under the edge-stalk gauge. Against two nulls of the
   *same shapes*: `flat` (`holonomy_read.flat_maps` -- exactly-flat Haar maps drawn
   independently on this arm's own block structure, whose identification null is 1
   analytically) and `rewired` (the same **trained** maps permuted among endpoints of
   equal shape, which keeps every learned spectrum and destroys only which map sits
   next to which).

2. **Is what remains abelian?** :func:`abelian_read`. Holonomies around different
   cycles do not live in the same space, so they cannot be commuted as they stand.
   For each interior edge `e` carried by two or more basis cycles, each such cycle is
   **rotated to base at `e`** -- a cyclic rotation of the edge tuple, the same closed
   walk read from a different starting point -- giving a set of generators that are
   all `m_e x m_e`. Then, per pair:

   * `comm_op` -- `||H_i H_j - H_j H_i||_F / (||H_i H_j||_F + ||H_j H_i||_F)`, 0 when
     they commute.
   * `comm_polar` -- the same on the orthogonal polar factors `Q = UV^T`, normalised
     `/ sqrt(2m)` so it shares `identification`'s scale and its null at 1. The polar
     factor is what survives the decay, and the abelian rung of the ladder is a claim
     about rotations, not about gains.

   and, per edge, whether the ladder's bottom rung is there: `shared` is the top
   eigenvector of `sum_i u_i u_i^T` over the generators' dominant output directions,
   and `invariance` is `|<w, H_i w>| / ||H_i w||` for that `w` -- 1 when the loop
   returns `w` to itself, which is what *one common invariant direction* means.

   **The ladder itself (trivial -> full capacity, abelian -> partial, non-abelian ->
   one direction) is an agent inference on #585 and is not cited.** This module does
   not inherit it: it measures the three rungs separately and lets them disagree.

3. **Does holonomy predict composed rank?** :func:`link_read`. Per chain, the cycles
   sharing at least one edge with it give a `cycle_load` (median identification, median
   flatness); Spearman against that chain's `composed_er`. Per apex, the same against
   [B20](#569)'s `d_eff`. Per edge, non-commutativity against the median ER of the
   chains that use it. `n = 256` chains, so this is the reading with power in it.

4. **Does `p` move it?** Every arm carries its `p`, and `shipped` sits beside
   `reserve_p8/16/24` -- with the construction sweep free over eight seeds. `p`
   narrows `k_v` and forces lanes to coincide, and coincident lanes plausibly reduce
   holonomy for a trivial reason; `flat`, drawn at each arm's own widths, is what says
   whether a fall in departure is a fall relative to that arm's own chance.

5. **The tension.** Strict path-independence looks incompatible with lanes that
   select: a product of projections contracts, so a round trip does not return. That
   prediction is read directly -- `sigma_max` and `flatness` against cycle length, on
   the real surface and on `flat`, whose hops are the same projections with nothing
   learned. If the real arm tracks `flat`, the contraction is the geometry and not the
   training. **`channel_return` is the refinement**: path-independence asked only of
   the content that actually travels.

Usage::

    PYTHONPATH=src python prototypes/cold-start/T6/b29_holonomy.py construction --arms shipped reserve_p8 reserve_p16 reserve_p24
    PYTHONPATH=src python prototypes/cold-start/T6/b29_holonomy.py trained --arms reserve_p16 --ticks 20000

This module **asserts nothing and rules on no remedy**. It builds an instrument, takes
a reading, and states what it did not measure as plainly as what it did.
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import sys
import time
from collections import deque
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
angles = _load("t4_angles", _T4 / "angles.py")
t4_trained = _load("t4_trained", _T4 / "trained.py")
arms_mod = _load("t6_arms", _HERE / "arms.py")
joint = _load("t6_b20_joint", _HERE / "b20_joint.py")

import construction_grading as cg  # noqa: E402,F401
import holonomy_read as hr  # noqa: E402
from patchworks.learning import PredictionRule, TransportRule  # noqa: E402

#: Generators per edge for the abelian read. Every basis cycle through an edge is a
#: generator; the cap keeps the pair count linear and is stated rather than tuned.
MAX_GENERATORS = 6

#: Draws for the free `flat` null, `holonomy_read.DRAWS` unchanged.
DRAWS = 8

#: Below this, relative to the cycle's own top singular value, a holonomy has carried
#: nothing in its smallest direction. `holonomy_read.RELATIVE_ZERO`, reported not dropped.
RELATIVE_ZERO = hr.RELATIVE_ZERO

COLUMNS = ("sigma_max", "flatness", "identification", "channel_return")


# -- summaries ----------------------------------------------------------------


def _q(values) -> dict:
    a = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if a.size == 0:
        return {"n": 0}
    return {
        "n": int(a.size),
        "median": float(np.median(a)),
        "mean": float(a.mean()),
        "p10": float(np.quantile(a, 0.10)),
        "p90": float(np.quantile(a, 0.90)),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def _spearman(x, y) -> dict:
    """Rank correlation with ties averaged, and Pearson beside it. No scipy here."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if x.size < 3:
        return {"n": int(x.size), "rho": None, "pearson": None}

    def rank(v):
        order = np.argsort(v, kind="mergesort")
        r = np.empty(len(v), dtype=np.float64)
        r[order] = np.arange(len(v), dtype=np.float64)
        _, inv, counts = np.unique(v, return_inverse=True, return_counts=True)
        sums = np.zeros(len(counts))
        np.add.at(sums, inv, r)
        return (sums / counts)[inv]

    def corr(a, b):
        sa, sb = a.std(), b.std()
        if sa <= 0 or sb <= 0:
            return None
        return float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb))

    return {"n": int(x.size), "rho": corr(rank(x), rank(y)), "pearson": corr(x, y)}


# -- 0: what the graph permits before anything is learned ----------------------


def lane_census(dome, bases) -> dict:
    """The rank ceiling on every loop, which is a **construction** fact.

    A cycle's holonomy has rank at most the narrowest lane it passes through, whatever
    the maps are. This is the census of that ceiling: how the interior edges split by
    width, whether the narrow ones are lateral, and how many basis cycles are capped at
    one dimension before a map is drawn. Nothing here depends on training, on the seed,
    or on `p` -- and that is the point of printing it beside readings that do.
    """
    interior, edge_ids = hr.interior_graph(dome)

    def level(cell_id: int) -> int:
        return dome.cells[cell_id].index.level

    widths = collections.Counter(dome.edges[e].m for e in edge_ids)
    narrow = [e for e in edge_ids if dome.edges[e].m < 2]
    wide = [e for e in edge_ids if dome.edges[e].m >= 2]
    out = {
        "interior_cells": len(interior),
        "interior_edges": len(edge_ids),
        "edge_width_hist": {str(k): int(v) for k, v in sorted(widths.items())},
        "narrow_edges": len(narrow),
        "wide_edges": len(wide),
        "narrow_all_lateral": all(
            level(dome.edges[e].u) == level(dome.edges[e].v) for e in narrow
        ),
        "wide_all_cross_level": all(
            level(dome.edges[e].u) != level(dome.edges[e].v) for e in wide
        ),
        "bases": {},
    }
    for name, cycles in bases.items():
        caps = [min(dome.edges[e].m for e in c) for c in cycles]
        out["bases"][name] = {
            "cycles": len(cycles),
            "rank_cap_hist": {
                str(k): int(v) for k, v in sorted(collections.Counter(caps).items())
            },
            "capped_at_one": int(sum(1 for x in caps if x < 2)),
            "length": _q([len(c) for c in cycles]),
        }
    return out


# -- 1: how far from flat -----------------------------------------------------


def surface_read(dome, maps, cycles, label: str) -> dict:
    """Every cycle's four gauge-invariant columns off one surface. Never averaged."""
    rows = hr.read_surface(dome, maps, cycles)
    lengths = np.array([r["length"] for r in rows])
    out = {
        "label": label,
        "cycles": len(rows),
        "degenerate": int(sum(1 for r in rows if r.get("degenerate"))),
        "per_cycle": {c: [float(r.get(c, float("nan"))) for r in rows] for c in COLUMNS},
        "length": [int(x) for x in lengths],
        "m_base": [int(r["m"]) for r in rows],
    }
    for c in COLUMNS:
        out[c] = _q(out["per_cycle"][c])
    # By cycle length: the shortest cycles are where a per-map surface has the best
    # chance of composing back, so the trend across length is the part-5 reading.
    by_len = {}
    for length in sorted(set(int(x) for x in lengths.tolist())):
        idx = [i for i, r in enumerate(rows) if r["length"] == length]
        by_len[str(length)] = {
            "cycles": len(idx),
            **{c: _q([rows[i].get(c, float("nan")) for i in idx]) for c in COLUMNS},
        }
    out["by_length"] = by_len

    # Three populations, because two of them answer a different question.
    #
    # `wide`   -- base width >= 2. A 1x1 holonomy has `flatness` and
    #             `channel_return` identically 1 and `identification` in {0, sqrt(2)}
    #             whatever the surface does, so those cycles are arithmetic about a
    #             scalar. Kept in `all`, separated here.
    # `live`   -- `wide` and not degenerate. A polar factor taken off a holonomy whose
    #             smallest singular value is `RELATIVE_ZERO` of its largest is
    #             arithmetic about the decay, and the decay is part 5's finding rather
    #             than part 1's. Reported, never dropped.
    wide = [i for i, r in enumerate(rows) if r["m"] >= 2]
    live = [i for i in wide if not rows[i].get("degenerate")]
    out["subsets"] = {}
    for name, idx in (("wide", wide), ("live", live)):
        out["subsets"][name] = {
            "cycles": len(idx),
            "share": len(idx) / max(len(rows), 1),
            **{c: _q([rows[i].get(c, float("nan")) for i in idx]) for c in COLUMNS},
        }
    out["subsets"]["by_m"] = {
        str(m): {
            "cycles": sum(1 for r in rows if r["m"] == m),
            "degenerate": sum(1 for r in rows if r["m"] == m and r.get("degenerate")),
            **{
                c: _q([r.get(c, float("nan")) for r in rows if r["m"] == m])
                for c in COLUMNS
            },
        }
        for m in sorted({r["m"] for r in rows})
    }
    return out


# -- 2: is what remains abelian? ----------------------------------------------


def _base_at(cycle: tuple[int, ...], edge_id: int) -> tuple[int, ...]:
    """The same closed walk, read from `edge_id`. A cyclic rotation, nothing else."""
    i = cycle.index(edge_id)
    return cycle[i:] + cycle[:i]


def _polar(h: np.ndarray):
    u, s, vh = np.linalg.svd(h)
    return u @ vh, u, s, vh


def abelian_read(dome, maps, cycles, label: str, max_generators: int = MAX_GENERATORS) -> dict:
    """Do the loops through one edge commute, and is there one direction they all fix?"""
    through: dict[int, list[int]] = collections.defaultdict(list)
    for i, cycle in enumerate(cycles):
        for edge_id in cycle:
            through[edge_id].append(i)

    comm_op, comm_polar, invariance, spread_d_eff, edges_used = [], [], [], [], []
    # `m = 1` generators are scalars and commute for free, so they are held apart for
    # the same reason `surface_read` holds their cycles apart.
    wide_op, wide_polar, wide_inv, wide_edges = [], [], [], []
    per_edge = {}
    for edge_id in sorted(through):
        idx = through[edge_id][:max_generators]
        if len(idx) < 2:
            continue
        gens, tops = [], []
        for i in idx:
            h = hr.holonomy(dome, maps, _base_at(tuple(cycles[i]), edge_id)).numpy()
            if not np.all(np.isfinite(h)) or np.linalg.norm(h) <= 0:
                continue
            q, u, s, vh = _polar(h)
            gens.append((h, q, u[:, 0]))
            tops.append(u[:, 0])
        if len(gens) < 2:
            continue
        m = int(gens[0][0].shape[0])
        co, cp = [], []
        for a in range(len(gens)):
            for b in range(a + 1, len(gens)):
                ha, hb = gens[a][0], gens[b][0]
                ab, ba = ha @ hb, hb @ ha
                denom = np.linalg.norm(ab) + np.linalg.norm(ba)
                if denom > 0:
                    co.append(float(np.linalg.norm(ab - ba) / denom))
                qa, qb = gens[a][1], gens[b][1]
                cp.append(float(np.linalg.norm(qa @ qb - qb @ qa) / np.sqrt(2 * m)))
        # One common invariant direction? `w` is the consensus output direction.
        u_stack = np.stack(tops, axis=1)
        w = np.linalg.svd(u_stack, full_matrices=False)[0][:, 0]
        inv = []
        for h, _, _ in gens:
            hw = h @ w
            nrm = np.linalg.norm(hw)
            if nrm > 0:
                inv.append(float(abs(float(w @ hw)) / nrm))
        norms = np.linalg.norm(u_stack, axis=0, keepdims=True)
        sp = joint._spread(u_stack / np.maximum(norms, 1e-300))
        per_edge[str(edge_id)] = {
            "m": m,
            "generators": len(gens),
            "comm_op_median": float(np.median(co)) if co else None,
            "comm_polar_median": float(np.median(cp)) if cp else None,
            "invariance_median": float(np.median(inv)) if inv else None,
            "d_eff": sp["d_eff"],
        }
        comm_op.extend(co)
        comm_polar.extend(cp)
        invariance.extend(inv)
        spread_d_eff.append(sp["d_eff"])
        edges_used.append(edge_id)
        if m >= 2:
            wide_op.extend(co)
            wide_polar.extend(cp)
            wide_inv.extend(inv)
            wide_edges.append(edge_id)

    return {
        "label": label,
        "edges": len(edges_used),
        "pairs": len(comm_polar),
        "comm_op": _q(comm_op),
        "comm_polar": _q(comm_polar),
        "invariance": _q(invariance),
        "direction_d_eff": _q(spread_d_eff),
        "wide": {
            "edges": len(wide_edges),
            "pairs": len(wide_polar),
            "comm_op": _q(wide_op),
            "comm_polar": _q(wide_polar),
            "invariance": _q(wide_inv),
        },
        "per_edge": per_edge,
    }


# -- 3: does holonomy predict composed rank? ----------------------------------


def link_read(dome, maps, cycles, holo: dict, abel: dict, jread: dict, chain_rows) -> dict:
    """Departure from flatness against composed rank -- per chain, per apex, per edge."""
    ident = np.array(holo["per_cycle"]["identification"], dtype=np.float64)
    flat = np.array(holo["per_cycle"]["flatness"], dtype=np.float64)
    ret = np.array(holo["per_cycle"]["channel_return"], dtype=np.float64)

    # Only the informative cycles carry the load: a 1x1 holonomy's columns are
    # constants, and a decayed one's polar factor is arithmetic about the decay.
    # Including either would put a constant on one side of the correlation.
    m_base = np.array(holo["m_base"], dtype=int)
    degenerate = np.array(
        [holo["per_cycle"]["flatness"][i] < RELATIVE_ZERO for i in range(len(m_base))]
    )
    informative = {i for i in range(len(m_base)) if m_base[i] >= 2 and not degenerate[i]}

    cycles_of_edge: dict[int, list[int]] = collections.defaultdict(list)
    for i, cycle in enumerate(cycles):
        if i not in informative:
            continue
        for edge_id in cycle:
            cycles_of_edge[edge_id].append(i)

    chain_er, load_ident, load_flat, load_ret, touched = [], [], [], [], []
    for row in chain_rows:
        idx = sorted({i for e in row["edges"] for i in cycles_of_edge.get(e, ())})
        if not idx:
            continue
        chain_er.append(row["composed_er"])
        load_ident.append(float(np.nanmedian(ident[idx])))
        load_flat.append(float(np.nanmedian(flat[idx])))
        load_ret.append(float(np.nanmedian(ret[idx])))
        touched.append(len(idx))

    # Per apex: B20's d_eff against the load over the cycles the apex's chains touch.
    by_apex: dict[int, list] = collections.defaultdict(list)
    for row in chain_rows:
        by_apex[row["apex"]].append(row)
    apex_d_eff, apex_ident = [], []
    for apex, group in sorted(by_apex.items()):
        entry = jread["apexes"].get(str(apex))
        if entry is None:
            continue
        idx = sorted({i for r in group for e in r["edges"] for i in cycles_of_edge.get(e, ())})
        if not idx:
            continue
        apex_d_eff.append(entry["directions"]["d_eff"])
        apex_ident.append(float(np.nanmedian(ident[idx])))

    # Per edge: non-commutativity against the ER of the chains that run through it.
    er_of_edge: dict[int, list[float]] = collections.defaultdict(list)
    for row in chain_rows:
        for e in row["edges"]:
            er_of_edge[e].append(row["composed_er"])
    edge_comm, edge_er, edge_inv = [], [], []
    for key, entry in abel["per_edge"].items():
        e = int(key)
        if e not in er_of_edge or entry["comm_polar_median"] is None or entry["m"] < 2:
            continue
        edge_comm.append(entry["comm_polar_median"])
        edge_inv.append(entry["invariance_median"])
        edge_er.append(float(np.median(er_of_edge[e])))

    return {
        "chains": len(chain_er),
        "cycles_per_chain": _q(touched),
        "chain_er": _q(chain_er),
        "chain_er_vs_identification": _spearman(load_ident, chain_er),
        "chain_er_vs_flatness": _spearman(load_flat, chain_er),
        "chain_er_vs_channel_return": _spearman(load_ret, chain_er),
        "apexes": len(apex_d_eff),
        "apex_d_eff_vs_identification": _spearman(apex_ident, apex_d_eff),
        "edges": len(edge_er),
        "edge_er_vs_comm_polar": _spearman(edge_comm, edge_er),
        "edge_er_vs_invariance": _spearman(edge_inv, edge_er),
    }


# -- one surface, whole -------------------------------------------------------


def full_read(agent, bases, chains, label: str, *, with_joint: bool = True) -> dict:
    """Parts 1-3 and 5 on one surface, in one pass: nothing is differenced across runs.

    Both bases are read. `wide` is where the abelian ladder has room and where the
    correlation with composed rank is taken; `full` is `holonomy_read`'s own basis,
    kept so this reading is comparable with every earlier one.
    """
    dome, maps = agent.dome, agent.sheaf.maps
    out = {"bases": {}}
    for name, cycles in bases.items():
        holo = surface_read(dome, maps, cycles, f"{label} [{name}]")
        abel = abelian_read(dome, maps, cycles, f"{label} [{name}]")
        out["bases"][name] = {"holonomy": holo, "abelian": abel}
    if with_joint:
        jread = joint.joint_read(agent, chains, label)
        chain_rows = joint.delivered(agent, chains)
        out["joint"] = {k: v for k, v in jread.items() if k != "apexes"}
        for name, cycles in bases.items():
            entry = out["bases"][name]
            entry["link"] = link_read(
                dome, maps, cycles, entry["holonomy"], entry["abelian"], jread, chain_rows
            )
    return out


def null_reads(agent, bases, seed: int, draws: int = DRAWS) -> dict:
    """The two nulls of the same shapes: `flat` (Haar, independent) and `rewired`.

    `flat` is `holonomy_read.flat_maps` -- exactly-flat maps drawn independently on
    **this arm's own block structure**, so a fall in departure is read against this
    arm's chance rather than against another arm's. `rewired` keeps every learned
    spectrum and destroys only which map sits next to which.
    """
    dome, maps = agent.dome, agent.sheaf.maps
    out = {}
    for name, cycles in bases.items():
        flat_cols = {c: [] for c in COLUMNS}
        flat_comm, flat_inv = {}, {}
        for d in range(draws):
            g = torch.Generator().manual_seed(seed * 1000 + d)
            fm = hr.flat_maps(dome, g)
            r = surface_read(dome, fm, cycles, f"flat draw {d}")
            for c in COLUMNS:
                flat_cols[c].extend(r["per_cycle"][c])
            if d == 0:
                a = abelian_read(dome, fm, cycles, "flat draw 0")
                flat_comm, flat_inv = a["wide"]["comm_polar"], a["wide"]["invariance"]
        rew = hr.rewired(dome, maps, torch.Generator().manual_seed(seed))
        rewired_read = surface_read(dome, rew, cycles, "rewired")
        rewired_abel = abelian_read(dome, rew, cycles, "rewired")
        out[name] = {
            "flat": {
                "draws": draws,
                **{c: _q(flat_cols[c]) for c in COLUMNS},
                "comm_polar": flat_comm,
                "invariance": flat_inv,
            },
            "rewired": {
                **{c: rewired_read[c] for c in COLUMNS},
                "subsets": rewired_read["subsets"],
                "comm_polar": rewired_abel["wide"]["comm_polar"],
                "invariance": rewired_abel["wide"]["invariance"],
            },
        }
    return out


def _line(read: dict, prefix: str) -> str:
    w = read["bases"]["wide"]
    h, a = w["holonomy"], w["abelian"]["wide"]
    tail = ""
    if "joint" in read:
        tail = (
            f" | ER {read['joint']['per_chain_er']['median']:.4f}"
            f" d_eff {read['joint']['joint']['d_eff_median']:.3f}"
        )
    link = w.get("link", {}).get("chain_er_vs_identification", {})
    rho = link.get("rho")
    return (
        f"{prefix} wide ident {h['identification']['median']:.4f}"
        f" flat {h['flatness']['median']:.3e}"
        f" ret {h['channel_return']['median']:.4f}"
        f" | comm {a['comm_polar']['median']:.4f}"
        f" inv {a['invariance']['median']:.4f}"
        + (f" | rho {rho:+.3f}" if rho is not None else "")
        + tail
    )


# -- drivers ------------------------------------------------------------------


def _narrowest(dome, cycle: tuple[int, ...]) -> tuple[int, ...]:
    """The same loop, read from its narrowest edge -- the space it actually carries.

    A cycle's holonomy is `m_e x m_e` for whichever edge `e` the walk is read from,
    and which edge that is is arbitrary: the loop is the same loop. On `DEFAULT_SPEC`'s
    uniform `m = 4` the choice never mattered and `holonomy_read` took `cycle[0]`.
    It decides the reading here, because [B8](#548) allocates lanes **per edge**.

    Read from a wide edge, the operator is rank-deficient the moment the loop passes
    through anything narrower -- `flatness` reads 0 and the polar factor is arithmetic
    about a null space that transport never had. Read from the narrowest edge, it is
    generically full rank and every column is a statement about the content that
    actually goes round. That is the object, and it is also part 5's refinement
    arrived at by force rather than by choice. Ties go to the lowest edge id.
    """
    narrow = min(cycle, key=lambda e: (dome.edges[e].m, e))
    return _base_at(cycle, narrow)


def wide_cycle_basis(dome, min_m: int = 2) -> list[tuple[int, ...]]:
    """`holonomy_read.cycle_basis` restricted to edges of width `>= min_m`.

    **Why this basis exists at all.** On this surface every `m = 1` edge is a
    *lateral* -- same level at both ends -- and every `m >= 2` edge is cross-level;
    the split is exact, 215 against 194 of the 409 interior edges. A lateral therefore
    carries one dimension, so **250 of the 260 interior basis cycles have a rank-1
    holonomy before a single map is drawn**, and on those "is the sheaf flat" has no
    whole-operator content: a 1x1 holonomy is a sign.

    Dropping the width-1 edges leaves a subgraph that is still connected and still
    spans all 150 predicting cells, on 194 edges with **45 independent cycles**. That
    is the sheaf's cross-level transport, and it is the only place on this surface
    where *trivial*, *abelian* and *non-abelian* are three different answers.

    Same construction as `holonomy_read.cycle_basis` -- BFS spanning forest, one cycle
    per non-tree edge, shortest cycles available this way -- on the restricted edge
    set, and the count is checked against the rank rather than assumed.
    """
    interior, edge_ids = hr.interior_graph(dome)
    edge_ids = [e for e in edge_ids if dome.edges[e].m >= min_m]
    kept = set(edge_ids)
    incident: dict[int, list[int]] = {cell: [] for cell in interior}
    for edge_id in edge_ids:
        edge = dome.edges[edge_id]
        incident[edge.u].append(edge_id)
        incident[edge.v].append(edge_id)
    nodes = [cell for cell in interior if incident[cell]]

    def far(edge_id: int, cell: int) -> int:
        edge = dome.edges[edge_id]
        return edge.v if edge.u == cell else edge.u

    parent: dict[int, tuple[int, int]] = {}
    depth: dict[int, int] = {}
    tree_edges: set[int] = set()
    components = 0
    for root in nodes:
        if root in depth:
            continue
        components += 1
        depth[root] = 0
        queue = deque([root])
        while queue:
            cell = queue.popleft()
            for edge_id in incident[cell]:
                other = far(edge_id, cell)
                if other in depth:
                    continue
                depth[other] = depth[cell] + 1
                parent[other] = (cell, edge_id)
                tree_edges.add(edge_id)
                queue.append(other)

    def climb(cell: int, target: int):
        route: list[int] = []
        while depth[cell] > target:
            up, edge_id = parent[cell]
            route.append(edge_id)
            cell = up
        return cell, route

    cycles: list[tuple[int, ...]] = []
    for edge_id in sorted(kept - tree_edges):
        edge = dome.edges[edge_id]
        shallow = min(depth[edge.u], depth[edge.v])
        left, up_left = climb(edge.u, shallow)
        right, up_right = climb(edge.v, shallow)
        while left != right:
            parent_left, edge_left = parent[left]
            parent_right, edge_right = parent[right]
            up_left.append(edge_left)
            up_right.append(edge_right)
            left, right = parent_left, parent_right
        cycles.append(tuple(up_left + list(reversed(up_right)) + [edge_id]))

    rank = len(edge_ids) - len(nodes) + components
    if len(cycles) != rank:
        raise ValueError(f"{len(cycles)} wide cycles against a cycle rank of {rank}")
    return cycles


def cycles_of(dome) -> dict[str, list[tuple[int, ...]]]:
    """The two bases this reading needs, each rotated to base at its narrowest edge.

    * `full` -- `holonomy_read`'s 260 interior cycles, unchanged as a set.
    * `wide` -- :func:`wide_cycle_basis`' 45 cross-level cycles, where the holonomy
      has more than one dimension to be non-flat in.
    """
    return {
        "full": [_narrowest(dome, tuple(c)) for c in hr.cycle_basis(dome)],
        "wide": [_narrowest(dome, tuple(c)) for c in wide_cycle_basis(dome)],
    }


def run_construction(args) -> None:
    """Free: every arm at construction, over seeds, plus both nulls. No training."""
    record = {"issue": 585, "reading": "holonomy at construction", "arms": {}}
    for arm in args.arms:
        per_seed = []
        for seed in args.seeds:
            env, agent = arms_mod.build_arm(arm, seed)
            try:
                dome = agent.dome
                bases = cycles_of(dome)
                chains = t2.rim_chains(dome)
                read = full_read(agent, bases, chains, f"{arm} s{seed} construction")
                read["cycles_count"] = {k: len(v) for k, v in bases.items()}
                read["lane_census"] = lane_census(dome, bases)
                read["privacy"] = arms_mod.privacy_read(dome, arms_mod.ARMS[arm][1])
                read["widths"] = arms_mod.widths_read(dome, chains)
                if seed == args.seeds[0]:
                    read["nulls"] = null_reads(agent, bases, seed, draws=args.draws)
                per_seed.append(read)
                print(_line(read, f"  {arm} s{seed} construction:"), flush=True)
            finally:
                env.close()
        record["arms"][arm] = {"reserve_p": arms_mod.ARMS[arm][1], "seeds": per_seed}
        args.out.write_text(json.dumps(record, indent=1))
    print(f"wrote {args.out.name}", flush=True)


def train_one(arm: str, args) -> None:
    """One arm, trained, with the holonomy reading taken at every checkpoint.

    Sequential and checkpointed, as #555's notes require: long runs get killed when
    parallelised, and the record is written as each checkpoint lands.
    """
    started = time.time()
    out = _HERE / f"585-holonomy-{arm}-{args.condition}-seed{args.seed}-{args.ticks}.json"
    if out.exists():
        print(f"[B29] {out.name} already at the horizon, skipping", flush=True)
        return
    inflight = out.with_suffix(".inflight.json")
    cond = t3.CONDITIONS[args.condition]
    pin, c = cond["pin"], cond["c"]
    env, agent = arms_mod.build_arm(arm, args.seed)
    try:
        if pin:
            t1.pin_drive_edges(agent)
        dome = agent.dome
        bases = cycles_of(dome)
        chains = t2.rim_chains(dome)
        _, reserve_p = arms_mod.ARMS[arm]
        record = {
            "issue": 585,
            "reading": "holonomy, path-independence and composed rank under training",
            "arm": arm,
            "reserve_p": reserve_p,
            "condition": args.condition,
            "seed": args.seed,
            "ticks": args.ticks,
            "cycles": {k: len(v) for k, v in bases.items()},
            "cycle_lengths": {k: _q([len(c) for c in v]) for k, v in bases.items()},
            "lane_census": lane_census(dome, bases),
            "max_generators": MAX_GENERATORS,
            "relative_zero": RELATIVE_ZERO,
            "privacy": arms_mod.privacy_read(dome, reserve_p),
            "widths": arms_mod.widths_read(dome, chains),
            "checkpoints": [],
        }
        recorder = t2.EdgeRecorder(agent)
        bias = PredictionRule(agent.sheaf, operator_rate_ratio=c)
        transport = TransportRule(agent.sheaf)

        record["at_construction"] = full_read(
            agent, bases, chains, f"construction {arm} s{args.seed}"
        )
        record["nulls_at_construction"] = null_reads(agent, bases, args.seed, draws=args.draws)
        print(_line(record["at_construction"], f"  {arm} s{args.seed} construction:"), flush=True)

        ladder = [cp for cp in t3.CHECKPOINTS if cp <= args.ticks]
        if args.ticks not in ladder:
            ladder.append(args.ticks)
        seen = 0
        for target in ladder:
            for _ in t0.teaching_read(
                agent, target - seen, args.seed + seen, recorder, bias, transport
            ):
                pass
            seen = target
            entry = {"ticks": target}
            entry.update(full_read(agent, bases, chains, f"{arm} s{args.seed} @{target}"))
            entry["elapsed_minutes"] = (time.time() - started) / 60.0
            record["checkpoints"].append(entry)
            inflight.write_text(json.dumps(record, indent=1))
            print(
                _line(entry, f"  {arm} s{args.seed} @{target:>6}:")
                + f" ({entry['elapsed_minutes']:.1f} min)",
                flush=True,
            )
        # The nulls again at the horizon: `rewired` needs the *trained* maps.
        record["nulls_at_horizon"] = null_reads(agent, bases, args.seed, draws=args.draws)
        inflight.write_text(json.dumps(record, indent=1))
        inflight.replace(out)
        print(f"wrote {out.name}", flush=True)
    finally:
        env.close()


def run_trained(args) -> None:
    """The arms, **one at a time**: parallel long runs trip this box's memory guard."""
    for arm in args.arms:
        print(f"[B29] {arm} {args.condition} seed {args.seed}, {args.ticks} ticks", flush=True)
        train_one(arm, args)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)

    c = sub.add_parser("construction", help="holonomy at construction, no training")
    c.add_argument(
        "--arms", nargs="+", default=["shipped", "reserve_p8", "reserve_p16", "reserve_p24"]
    )
    c.add_argument("--seeds", type=int, nargs="+", default=[42])
    c.add_argument("--draws", type=int, default=DRAWS)
    c.add_argument("--out", type=Path, default=_HERE / "585-holonomy-construction.json")
    c.set_defaults(func=run_construction)

    t = sub.add_parser("trained", help="arms trained one at a time, read at every checkpoint")
    t.add_argument("--arms", nargs="+", default=["reserve_p16"])
    t.add_argument("--condition", choices=sorted(t3.CONDITIONS), default="baseline")
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--ticks", type=int, default=20_000)
    t.add_argument("--draws", type=int, default=DRAWS)
    t.set_defaults(func=run_trained)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
