"""T6 ([B20](#569)): the **joint** span all rim-to-apex chains deliver, not the per-chain median.

Every composed-rank reading on [#532](#532) to date is `composed_reads`
(`T2/run.py:496-520`): one scalar per chain, reduced by an order statistic.
`er.append(...)` at line 506 is the whole aggregation, and `T4/angles.py:191-195,
268-287`, `T6/b16_coherence.py:123-127`, `T6/b16_junction.py:160-163` are all
order statistics over that list. Nothing stacks the composed operators.

That leaves one reading of the collapse untested, and [B17](#565) is what makes it
matter: if a cell is a narrow specialist broadcasting *how much of its own thing is
happening*, then **per-chain composed rank of exactly 1 is correct**, and the
capacity is in 256 chains carrying 256 *different* directions. This module builds
the object that can tell the two apart.

## What is stacked, and in whose coordinates

`composed_reads` returns, per chain, an operator `C` from the **rim** edge's lane
coordinates to the **last** edge's lane coordinates. Chains that reach the same
apex cell do not in general arrive on the same edge, so their `C` live in
different spaces and cannot be stacked as they stand.

They are pulled back into the apex cell's own stalk, using the same adjoint
`hop_operator` already uses (`f_out @ f_in.T` -- pull into the cell, push onto the
next edge). If `F` is the map the apex holds into the chain's last edge, a lane
direction `y` is the stalk direction `F.T @ y`, so

    A = F.T @ C            # (n x m_first), in the apex cell's stalk basis

is the chain's delivered operator *at the apex*, and every chain terminating at
that apex now lives in one `R^n`. This is the same construction as
`T0/excitation.py:158-162`, which concatenates one cell's incident maps -- the
nearest joint object the rig already had.

## The three readings, and why three

Stacking is only half of it: a stack of 84 operators with wildly different gains
has an effective rank set by the loudest one, which answers a different question
from "how many directions are there".

* **`joint_energy`** -- effective rank of `[A_1 | ... | A_N]` as it stands.
  Energy-weighted, directly comparable to the per-chain median the map quotes.
  Says how many directions carry *signal*, gains included.
* **`joint_unit`** -- the same, each `A_i` scaled to unit Frobenius norm first.
  Removes the gain heterogeneity and asks about direction diversity alone.
* **`directions`** -- each chain reduced to the one unit vector it actually
  delivers (top left singular vector of `A_i`, in apex-stalk coordinates), stacked
  as `D` (`n x N`). This is the reading the specialist hypothesis is about: `N`
  chains, `N` unit vectors, how many distinct? Reported as effective rank, as
  numerical rank, and -- the scale-free one -- as

      d_eff = 1 / mean_{i != j} <u_i, u_j>^2

  which is `d` exactly when the `u_i` are spread isotropically over a `d`-dimensional
  subspace, and 1 when they all coincide. It needs no tolerance and no conditioning
  assumption, so it is the number to quote when the stack is ill-conditioned.

## The ceilings, which are the point

The joint span cannot exceed what the geometry allows, and those ceilings are
**construction** facts no amount of training can move:

* `k_v` at the apex -- under the reserve policy every map lives in the leading
  `n - p` block, so **raising `p` lowers this ceiling** while raising per-chain rank.
  That is [B13](#560)'s crux stated as a bound.
* `rank of the union of the apex's incident maps` (`ceiling_union`) -- the most
  directions *any* transport into that cell could deliver, whatever the chains do.
* `rank of the union over the distinct last edges the chains actually use`
  (`ceiling_lanes`) -- tighter, and the real bound on this set of chains.

A generic control (`generic`) redraws each chain's delivered direction as a Haar
unit vector in the apex's `k_v` block and takes the same statistics, so "a
population code working" has a number beside it rather than an intuition.

Usage::

    PYTHONPATH=src python prototypes/cold-start/T6/b20_joint.py construction --arms shipped reserve_p12
    PYTHONPATH=src python prototypes/cold-start/T6/b20_joint.py trained --arm reserve_p16 --ticks 20000
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
angles = _load("t4_angles", _T4 / "angles.py")
t4_trained = _load("t4_trained", _T4 / "trained.py")
arms_mod = _load("t6_arms", _HERE / "arms.py")

import construction_grading as cg  # noqa: E402
from patchworks.graph import NODE_STALK_DIM  # noqa: E402
from patchworks.learning import PredictionRule, TransportRule  # noqa: E402
from patchworks.restriction import pair_index  # noqa: E402

#: Relative tolerance for a numerical rank, on the leading singular value.
#: `np.linalg.matrix_rank`'s own default (`max(M,N) * eps`) is far below the
#: float64 noise a seven-hop product carries, so it is stated here instead.
RANK_RTOL = 1e-8


# -- the pullback -------------------------------------------------------------


@torch.no_grad()
def apex_map(dome, maps, edge_id: int, cell_id: int) -> np.ndarray:
    """The map `cell_id` holds into `edge_id`, on its live rows: `(m_e x n)`."""
    m = dome.edges[edge_id].m
    f = maps.maps[pair_index(edge_id, cg.side_of(dome, edge_id, cell_id))][:m]
    return f.double().numpy()


@torch.no_grad()
def delivered(agent, chains: list[dict]) -> list[dict]:
    """Per chain: the composed operator, and the same operator in apex-stalk coordinates.

    `composed` is `composed_reads`' object verbatim -- the identical product of
    `hop_operator`s -- so a per-chain ER read here reproduces the map's number
    exactly. `apex` is `F.T @ composed`, the pullback described in the module docstring.
    """
    dome, maps = agent.dome, agent.sheaf.maps
    rows = []
    for chain in chains:
        hops = cg.hops_of(dome, tuple(chain["edges"]))
        composed = t2.hop_operator(dome, maps, hops[0])
        for key in hops[1:]:
            composed = t2.hop_operator(dome, maps, key) @ composed
        c = composed.numpy()
        last = chain["edges"][-1]
        f_apex = apex_map(dome, maps, last, chain["apex"])
        a = f_apex.T @ c
        s = np.linalg.svd(c, compute_uv=False)
        # The chain's delivered direction: the leading left singular vector of `A`.
        u, sa, _ = np.linalg.svd(a, full_matrices=False)
        rows.append(
            {
                "rim": chain["rim"],
                "kind": chain["kind"],
                "apex": chain["apex"],
                "last_edge": last,
                "edges": tuple(chain["edges"]),
                "composed_er": angles.effective_rank(s),
                "sigma_max": float(s[0]),
                "apex_op": a,
                "apex_er": angles.effective_rank(sa),
                "direction": u[:, 0],
                "apex_frob": float(np.linalg.norm(a)),
            }
        )
    return rows


# -- the joint statistics -----------------------------------------------------


def _rank(m: np.ndarray) -> int:
    if m.size == 0:
        return 0
    s = np.linalg.svd(m, compute_uv=False)
    if s[0] <= 0:
        return 0
    return int((s > s[0] * RANK_RTOL).sum())


def _spread(directions: np.ndarray) -> dict:
    """`d_eff` and the pairwise-cosine distribution of a set of unit column vectors.

    `d_eff = 1 / mean_{i<j} cos^2` is `d` for vectors spread isotropically over a
    `d`-dimensional subspace and 1 when they coincide, and needs no tolerance.
    """
    n = directions.shape[1]
    if n < 2:
        return {"pairs": 0, "d_eff": 1.0, "cos_median": None, "cos_p90": None, "cos_max": None}
    g = directions.T @ directions
    iu = np.triu_indices(n, k=1)
    cos = np.abs(g[iu])
    mean_cos2 = float((cos**2).mean())
    return {
        "pairs": int(len(cos)),
        "d_eff": float(1.0 / max(mean_cos2, 1e-300)),
        "cos_median": float(np.median(cos)),
        "cos_p90": float(np.quantile(cos, 0.90)),
        "cos_max": float(cos.max()),
    }


def cross_kind(rows: list[dict]) -> dict:
    """Do different modalities deliver different directions into the same apex?

    #569 item 3 asks for a grouping by rim kind, and the honest answer needs this
    rather than a per-kind span: the rim is 256 `patch` cells against 1 actuator, 3
    proprioceptive and 3 touch, so a *within*-kind subspace dimension is undefined
    for three of the four kinds. What is well defined at any size is the angle
    *between* the kinds' spans, which is the claim itself -- a healthy population
    code puts different modalities on different directions.

    `cos_principal` is the cosine of the smallest principal angle between the two
    kinds' direction-spans (1.0 = one shared direction); `cos_pairwise_median` is
    the median `|<u_i, u_j>|` over chains across the two kinds.
    """
    kinds: dict[str, list[np.ndarray]] = collections.defaultdict(list)
    for r in rows:
        kinds[r["kind"]].append(r["direction"])
    out = {}
    names = sorted(kinds)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            da = np.stack(kinds[a], axis=1)
            db = np.stack(kinds[b], axis=1)
            qa = np.linalg.svd(da, full_matrices=False)[0][:, : _rank(da)]
            qb = np.linalg.svd(db, full_matrices=False)[0][:, : _rank(db)]
            cos = np.linalg.svd(qa.T @ qb, compute_uv=False)
            pair = np.abs(da.T @ db).ravel()
            out[f"{a}|{b}"] = {
                "chains": [len(kinds[a]), len(kinds[b])],
                "cos_principal": float(cos[0]) if cos.size else None,
                "cos_pairwise_median": float(np.median(pair)),
                "cos_pairwise_max": float(pair.max()),
            }
    return out


def joint_of(rows: list[dict], n: int) -> dict:
    """The three joint readings over one group of chains sharing an apex."""
    if not rows:
        return {}
    energy = np.concatenate([r["apex_op"] for r in rows], axis=1)
    unit = np.concatenate(
        [r["apex_op"] / max(r["apex_frob"], 1e-300) for r in rows], axis=1
    )
    dirs = np.stack([r["direction"] for r in rows], axis=1)
    se = np.linalg.svd(energy, compute_uv=False)
    su = np.linalg.svd(unit, compute_uv=False)
    sd = np.linalg.svd(dirs, compute_uv=False)
    per_chain = np.array([r["composed_er"] for r in rows])
    return {
        "chains": len(rows),
        "stalk_n": int(n),
        "per_chain_er_median": float(np.median(per_chain)),
        "per_chain_er_max": float(per_chain.max()),
        "joint_energy": {"er": angles.effective_rank(se), "rank": _rank(energy)},
        "joint_unit": {"er": angles.effective_rank(su), "rank": _rank(unit)},
        "directions": {
            "er": angles.effective_rank(sd),
            "rank": _rank(dirs),
            **_spread(dirs),
        },
        "sigma_max_ratio": float(
            max(r["sigma_max"] for r in rows) / max(min(r["sigma_max"] for r in rows), 1e-300)
        ),
    }


def ceilings(dome, maps, apex: int, rows: list[dict]) -> dict:
    """What the geometry permits, before any question of what training did."""
    lanes = sorted({r["last_edge"] for r in rows})
    lane_block = np.concatenate([apex_map(dome, maps, e, apex) for e in lanes], axis=0)
    incident = sorted(dome.incident[apex])
    union_block = np.concatenate([apex_map(dome, maps, e, apex) for e in incident], axis=0)
    return {
        "k_v": int(dome._permitted[apex]),
        "distinct_last_edges": len(lanes),
        "lane_m": [int(dome.edges[e].m) for e in lanes],
        "ceiling_lanes": _rank(lane_block),
        "ceiling_lanes_dim": int(sum(dome.edges[e].m for e in lanes)),
        "incident_edges": len(incident),
        "ceiling_union": _rank(union_block),
        # The tree bound nobody has stated: chains that share their final edges share
        # their final hops, so if a chain delivers one direction, chains with the same
        # suffix deliver the *same* one. This counts how many suffixes there are to be
        # different about, at depths 1-3.
        "distinct_suffix": [
            len({tuple(r["edges"][-d:]) for r in rows}) for d in (1, 2, 3)
        ],
    }


def _haar_in(basis: np.ndarray, count: int, rng: np.random.Generator) -> dict:
    """`count` Haar unit directions inside the span of `basis`' columns, read the same way."""
    g = rng.standard_normal((basis.shape[1], count))
    d = basis @ g
    d /= np.linalg.norm(d, axis=0, keepdims=True)
    s = np.linalg.svd(d, compute_uv=False)
    return {"er": angles.effective_rank(s), "rank": _rank(d), **_spread(d)}


def generic_spread(dome, maps, apex: int, rows: list[dict], rng: np.random.Generator) -> dict:
    """A population code working, with two references rather than one.

    `lanes` is the honest control: Haar directions inside the span the chains'
    own last edges actually reach, so it is what *these* chains could deliver if
    they used their lanes fully. `k_v` is the looser one -- the whole readable
    block -- and the gap between them is what the lane bottleneck costs before
    transport is asked to do anything.
    """
    n = int(dome.shape.n)
    count = len(rows)
    lane_ids = sorted({r["last_edge"] for r in rows})
    lane_block = np.concatenate([apex_map(dome, maps, e, apex) for e in lane_ids], axis=0)
    lane_basis = np.linalg.svd(lane_block.T, full_matrices=False)[0]
    lane_basis = lane_basis[:, : _rank(lane_block)]
    k = int(dome._permitted[apex])
    kv_basis = np.zeros((n, k))
    kv_basis[np.arange(k), np.arange(k)] = 1.0
    return {**_haar_in(lane_basis, count, rng), "k_v": _haar_in(kv_basis, count, rng)}


def joint_read(agent, chains: list[dict], label: str, rng_seed: int = 0) -> dict:
    """#569's whole reading on one surface: per apex, per apex-and-kind, and pooled."""
    dome, maps = agent.dome, agent.sheaf.maps
    rows = delivered(agent, chains)
    n = int(dome.shape.n)
    rng = np.random.default_rng(rng_seed)

    by_apex: dict[int, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by_apex[r["apex"]].append(r)

    apexes = {}
    for apex in sorted(by_apex):
        group = by_apex[apex]
        entry = {
            **joint_of(group, n),
            "ceilings": ceilings(dome, maps, apex, group),
            "generic": generic_spread(dome, maps, apex, group, rng),
            "by_kind": {},
            "cross_kind": cross_kind(group),
        }
        kinds: dict[str, list[dict]] = collections.defaultdict(list)
        for r in group:
            kinds[r["kind"]].append(r)
        for kind in sorted(kinds):
            entry["by_kind"][kind] = joint_of(kinds[kind], n)
        apexes[str(apex)] = entry

    # Pooled across apexes. A span is only defined within one apex's stalk, so what
    # pools is the *distribution* of per-apex spans, weighted by nothing -- plus the
    # per-chain median, which is the number the map has been quoting all along.
    per_chain = np.array([r["composed_er"] for r in rows])
    d_eff = np.array([apexes[a]["directions"]["d_eff"] for a in apexes])
    gen_eff = np.array([apexes[a]["generic"]["d_eff"] for a in apexes])
    energy_er = np.array([apexes[a]["joint_energy"]["er"] for a in apexes])
    unit_er = np.array([apexes[a]["joint_unit"]["er"] for a in apexes])
    caps = np.array([apexes[a]["ceilings"]["ceiling_lanes"] for a in apexes])

    return {
        "label": label,
        "chains": len(rows),
        "apex_cells": len(apexes),
        "stalk_n": n,
        "hops": len(chains[0]["edges"]) if chains else 0,
        "per_chain_er": {
            "median": float(np.median(per_chain)),
            "p90": float(np.quantile(per_chain, 0.90)),
            "max": float(per_chain.max()),
        },
        "joint": {
            "d_eff_median": float(np.median(d_eff)),
            "d_eff_min": float(d_eff.min()),
            "d_eff_max": float(d_eff.max()),
            "generic_d_eff_median": float(np.median(gen_eff)),
            "energy_er_median": float(np.median(energy_er)),
            "unit_er_median": float(np.median(unit_er)),
            "ceiling_lanes_median": float(np.median(caps)),
            "d_eff_over_ceiling_median": float(np.median(d_eff / np.maximum(caps, 1))),
        },
        "apexes": apexes,
    }


def _line(read: dict, prefix: str) -> str:
    j, pc = read["joint"], read["per_chain_er"]
    return (
        f"{prefix} per-chain ER {pc['median']:.4f} | joint d_eff {j['d_eff_median']:.3f} "
        f"(min {j['d_eff_min']:.3f} max {j['d_eff_max']:.3f}) vs generic "
        f"{j['generic_d_eff_median']:.3f} | ceiling {j['ceiling_lanes_median']:.1f} "
        f"| energy ER {j['energy_er_median']:.3f} unit ER {j['unit_er_median']:.3f}"
    )


# -- drivers ------------------------------------------------------------------


def run_check(args) -> None:
    """Is the pullback a faithful change of coordinates, or is it doing the work?

    The whole reading rests on `A = F.T @ C` putting chains that arrive on different
    edges into one comparable space. Two things could make that a lie, and both are
    checked here rather than asserted:

    1. **Conditioning.** `F.T` is an isometry only up to the conditioning of `F` on
       its live rows. `lane_cond` is `sigma_max / sigma_min` of the block; a large
       value means the pullback stretches some directions more than others and the
       joint numbers inherit that stretch.
    2. **Agreement.** At an apex where every chain arrives on the *same* edge, no
       pullback is needed -- the composed operators can be stacked in edge
       coordinates directly. The two readings must agree there, or the pullback is
       adding structure. `d_eff` is compared in both coordinate systems.
    """
    for arm in args.arms:
        env, agent = arms_mod.build_arm(arm, args.seed)
        try:
            dome, maps = agent.dome, agent.sheaf.maps
            chains = t2.rim_chains(dome)
            rows = delivered(agent, chains)
            by_apex: dict[int, list[dict]] = collections.defaultdict(list)
            for r in rows:
                by_apex[r["apex"]].append(r)
            print(f"  {arm} seed {args.seed}", flush=True)
            for apex in sorted(by_apex):
                group = by_apex[apex]
                lanes = sorted({r["last_edge"] for r in group})
                block = np.concatenate(
                    [apex_map(dome, maps, e, apex) for e in lanes], axis=0
                )
                s = np.linalg.svd(block, compute_uv=False)
                cond = float(s[0] / max(s[_rank(block) - 1], 1e-300))
                pulled = _spread(np.stack([r["direction"] for r in group], axis=1))["d_eff"]
                native = None
                if len(lanes) == 1:
                    # No pullback: the leading left singular vector of `C` itself.
                    d = []
                    for r in group:
                        f = apex_map(dome, maps, r["last_edge"], apex)
                        c = np.linalg.pinv(f.T) @ r["apex_op"]
                        d.append(np.linalg.svd(c, full_matrices=False)[0][:, 0])
                    native = _spread(np.stack(d, axis=1))["d_eff"]
                print(
                    f"    apex {apex}: lanes {len(lanes)} cond {cond:.3g} "
                    f"d_eff pulled {pulled:.4f}"
                    + (f" native {native:.4f}" if native is not None else " native n/a (2 lanes)"),
                    flush=True,
                )
        finally:
            env.close()


def run_construction(args) -> None:
    record = {
        "issue": 569,
        "reading": "joint rim-to-apex span at construction, across #560's p sweep",
        "rank_rtol": RANK_RTOL,
        "rows": [],
    }
    for arm in args.arms:
        for seed in args.seeds:
            env, agent = arms_mod.build_arm(arm, seed)
            try:
                chains = t2.rim_chains(agent.dome)
                _, reserve_p = arms_mod.ARMS[arm]
                read = joint_read(agent, chains, f"{arm} seed {seed}")
                row = {
                    "arm": arm,
                    "reserve_p": reserve_p,
                    "seed": seed,
                    "privacy": arms_mod.privacy_read(agent.dome, reserve_p),
                    "widths": arms_mod.widths_read(agent.dome, chains),
                    "joint": read,
                }
                record["rows"].append(row)
                print(_line(read, f"  {arm:>13} s{seed}:"), flush=True)
            finally:
                env.close()
            args.out.write_text(json.dumps(record, indent=1))
    print(f"wrote {args.out.name}", flush=True)


def train_one(arm: str, args) -> None:
    """One arm, trained, with the joint reading taken at every checkpoint.

    Sequential and checkpointed, as #555's notes require: long runs get killed when
    parallelised, and the record is written as each checkpoint lands.
    """
    started = time.time()
    out = _HERE / f"569-joint-{arm}-{args.condition}-seed{args.seed}-{args.ticks}.json"
    if out.exists():
        print(f"[B20] {out.name} already at the horizon, skipping", flush=True)
        return
    inflight = out.with_suffix(".inflight.json")
    cond = t3.CONDITIONS[args.condition]
    pin, c = cond["pin"], cond["c"]
    env, agent = arms_mod.build_arm(arm, args.seed)
    try:
        if pin:
            t1.pin_drive_edges(agent)
        dome = agent.dome
        chains = t2.rim_chains(dome)
        _, reserve_p = arms_mod.ARMS[arm]
        record = {
            "issue": 569,
            "reading": "joint rim-to-apex span under training",
            "arm": arm,
            "reserve_p": reserve_p,
            "condition": args.condition,
            "seed": args.seed,
            "ticks": args.ticks,
            "rank_rtol": RANK_RTOL,
            "privacy": arms_mod.privacy_read(dome, reserve_p),
            "widths": arms_mod.widths_read(dome, chains),
            "checkpoints": [],
        }
        recorder = t2.EdgeRecorder(agent)
        bias = PredictionRule(agent.sheaf, operator_rate_ratio=c)
        transport = TransportRule(agent.sheaf)

        record["at_construction"] = joint_read(
            agent, chains, f"construction {arm} s{args.seed}"
        )
        print(_line(record["at_construction"], f"  {arm} s{args.seed} construction:"), flush=True)

        ladder = [cp for cp in t3.CHECKPOINTS if cp <= args.ticks]
        if args.ticks not in ladder:
            ladder.append(args.ticks)
        seen = 0
        for target in ladder:
            for _ in t0.teaching_read(agent, target - seen, args.seed + seen, recorder, bias, transport):
                pass
            seen = target
            entry = {"ticks": target}
            entry["joint"] = joint_read(agent, chains, f"{arm} s{args.seed} @{target}")
            entry["cap"] = t4_trained.cap_read(agent)
            entry["elapsed_minutes"] = (time.time() - started) / 60.0
            record["checkpoints"].append(entry)
            inflight.write_text(json.dumps(record, indent=1))
            print(
                _line(entry["joint"], f"  {arm} s{args.seed} @{target:>6}:")
                + f" ({entry['elapsed_minutes']:.1f} min)",
                flush=True,
            )
        inflight.replace(out)
        print(f"wrote {out.name}", flush=True)
    finally:
        env.close()


def run_trained(args) -> None:
    """The arms, **one at a time**: parallel long runs trip this box's memory guard."""
    for arm in args.arms:
        print(f"[B20] {arm} {args.condition} seed {args.seed}, {args.ticks} ticks", flush=True)
        train_one(arm, args)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)

    c = sub.add_parser("construction", help="the joint span at construction, no training")
    c.add_argument("--arms", nargs="+", default=["shipped", "doubling", "reserve"])
    c.add_argument("--seeds", type=int, nargs="+", default=[42])
    c.add_argument("--out", type=Path, default=_HERE / "569-joint-construction.json")
    c.set_defaults(func=run_construction)

    k = sub.add_parser("check", help="is the apex-stalk pullback faithful?")
    k.add_argument("--arms", nargs="+", default=["shipped", "reserve_p16"])
    k.add_argument("--seed", type=int, default=42)
    k.set_defaults(func=run_check)

    t = sub.add_parser("trained", help="arms trained one at a time, read at every checkpoint")
    t.add_argument("--arms", nargs="+", default=["reserve"])
    t.add_argument("--condition", choices=sorted(t3.CONDITIONS), default="baseline")
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--ticks", type=int, default=20_000)
    t.set_defaults(func=run_trained)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
