"""B50 (#618): the *shape* of the per-hop cosine spectra -- corner-pinned, or interior?

`#618 <https://github.com/NGL321/patchworks/issues/618>`_, under
`the map <https://github.com/NGL321/patchworks/issues/532>`_.

**What is new here is the statistic, not the surface.** `angles.py` (#537)
already computes the per-hop principal-angle cosines between a relay cell's two
carried subspaces. Every read the map has taken over them has been a **scalar** --
effective rank, a mean, a leading value -- and a scalar cannot separate
*everything attenuates* from *`r` directions go through at gain 1 and the rest do
not*. This reads the spectra as spectra.

`r` is named on #618 as **the number of directions in a lane pair transporting at
gain ~1**. The flat bundle (`B42 <https://github.com/NGL321/patchworks/issues/605>`_)
is `r = m`; the trained surface is *argued* to be `r ~ 1`. Whether `r` is a real
design variable or a re-description turns on the shape, which is measurable here.

**Scope, as ruled on #618.** Existing surface only -- no constructed arm, no
imposed blocks, no candidate. Nothing is scored and nothing is built.

**The carried graph.** Per `B41 <https://github.com/NGL321/patchworks/issues/604>`_'s
reading rule, hops are enumerated over edges with `m_e > 0`, not over
`dome.edges` wholesale.

**Two hop populations, deliberately.** `chain` is the unique
`(edge_in, cell, edge_out)` set the rim->apex chains induce, which is the
population every #537 figure was taken on and therefore the comparable one.
`carried` is every unordered pair of carried edges incident on an interior cell,
which is the population #618's *"all interior hops"* names. Both are reported;
where they agree nothing rests on the choice.

**Audience differentiation falls out of the same cosines.** A cell's two incident
lanes *are* two audiences, so B42's quantity is `1 - overlap` with overlap the
normalised `sum cos^2 theta` over that pair -- 1.0000, and differentiation
0.0000, exactly when the two lanes carry the same subspace, which is B42's
stagger-0 anchor.

Usage::

    PYTHONPATH=src python prototypes/cold-start/T4/b50_spectra.py --stage construction
    PYTHONPATH=src python prototypes/cold-start/T4/b50_spectra.py --stage trained --condition baseline
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
_T0, _T1, _T2, _T3 = (_HERE.parent / n for n in ("T0", "T1", "T2", "T3"))
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

import construction_grading as cg  # noqa: E402
from untrained_fixed_point import build  # noqa: E402

CORNER_HI = 0.95
CORNER_LO = 0.05
R_THRESHOLDS = (0.90, 0.95, 0.99)


# -- the hop populations ------------------------------------------------------


def carried_edges_of(dome, cell_id: int) -> list[int]:
    """Incident edges of `cell_id` that carry anything -- B41's reading rule."""
    return [
        e
        for e, edge in enumerate(dome.edges)
        if edge.m > 0 and cell_id in (edge.u, edge.v)
    ]


def chain_hops(dome, chains) -> list[tuple[int, int, int]]:
    """The unique interior hops the rim->apex chains induce. #537's population."""
    seen, out = set(), []
    for chain in chains:
        for edge_in, cell, edge_out in cg.hops_of(dome, tuple(chain["edges"])):
            if dome.edges[edge_in].m <= 0 or dome.edges[edge_out].m <= 0:
                continue
            key = (min(edge_in, edge_out), cell, max(edge_in, edge_out))
            if key in seen:
                continue
            seen.add(key)
            out.append((edge_in, cell, edge_out))
    return out


def carried_hops(dome, chains) -> list[tuple[int, int, int]]:
    """Every unordered carried-edge pair at a cell that is interior to some chain.

    *Interior* is taken from the chains rather than from a level index -- the map
    forbids `level` (B27) -- so a cell qualifies if any chain passes *through* it.
    """
    interior = {cell for chain in chains for _, cell, _ in cg.hops_of(dome, tuple(chain["edges"]))}
    out = []
    for cell in sorted(interior):
        edges = carried_edges_of(dome, cell)
        for a, b in itertools.combinations(edges, 2):
            out.append((a, cell, b))
    return out


# -- the spectra --------------------------------------------------------------


@torch.no_grad()
def hop_cos(dome, maps, key) -> np.ndarray:
    """`cos theta` between the relay cell's two carried subspaces. #537's object."""
    from patchworks.restriction import pair_index

    edge_in, cell, edge_out = key
    frames = []
    for edge_id in (edge_in, edge_out):
        m = dome.edges[edge_id].m
        k = int(dome.restriction_mask(edge_id, cell).sum())
        f = maps.maps[pair_index(edge_id, cg.side_of(dome, edge_id, cell))][:m, :k]
        _, _, vh = torch.linalg.svd(f.double(), full_matrices=False)
        frames.append(vh.numpy().T)
    v_in, v_out = frames
    return np.linalg.svd(v_out.T @ v_in, compute_uv=False)


def _q(a: np.ndarray) -> dict:
    a = np.asarray(a, dtype=np.float64)
    if a.size == 0:
        return {}
    return {
        "min": float(a.min()),
        "p10": float(np.quantile(a, 0.10)),
        "median": float(np.median(a)),
        "mean": float(a.mean()),
        "p90": float(np.quantile(a, 0.90)),
        "max": float(a.max()),
    }


def read_spectra(dome, maps, hops, label: str) -> dict:
    """#618's items 1-4 over one hop population on one surface."""
    spectra = [np.sort(hop_cos(dome, maps, key))[::-1] for key in hops]
    flat = np.concatenate(spectra)

    # (1) corner mass -- where the cosines actually sit
    corner = {
        "n_cosines": int(flat.size),
        "frac_ge_0.99": float((flat >= 0.99).mean()),
        "frac_ge_0.95": float((flat >= CORNER_HI).mean()),
        "frac_ge_0.90": float((flat >= 0.90).mean()),
        "frac_le_0.05": float((flat <= CORNER_LO).mean()),
        "frac_le_0.10": float((flat <= 0.10).mean()),
        "frac_interior_05_95": float(((flat > CORNER_LO) & (flat < CORNER_HI)).mean()),
        "max": float(flat.max()),
        "min": float(flat.min()),
    }

    # (2) per-hop *shape*: each spectrum divided by its own leading value.
    #     Grouped by width, because a 3-wide and a 2-wide shape are not the same
    #     object and averaging them would invent a value at the missing slot.
    by_width: dict[int, list[np.ndarray]] = {}
    for s in spectra:
        by_width.setdefault(len(s), []).append(s / max(s[0], 1e-300))
    shape = {}
    for width, group in sorted(by_width.items()):
        stack = np.stack(group)
        shape[str(width)] = {
            "hops": int(stack.shape[0]),
            "median": [float(x) for x in np.median(stack, axis=0)],
            "p10": [float(x) for x in np.quantile(stack, 0.10, axis=0)],
            "p90": [float(x) for x in np.quantile(stack, 0.90, axis=0)],
            "iqr": [
                float(x)
                for x in np.quantile(stack, 0.75, axis=0) - np.quantile(stack, 0.25, axis=0)
            ],
        }

    # (4) `r` as measured -- per hop, at three thresholds so nothing rests on one
    r_counts = {}
    for thr in R_THRESHOLDS:
        counts = np.array([int((s >= thr).sum()) for s in spectra])
        r_counts[f"{thr:.2f}"] = {
            "mean": float(counts.mean()),
            "median": float(np.median(counts)),
            "max": int(counts.max()),
            "hist": {str(v): int(c) for v, c in zip(*np.unique(counts, return_counts=True))},
        }

    leading = np.array([s[0] for s in spectra])
    second = np.array([s[1] for s in spectra if len(s) > 1])
    trailing = np.array([s[-1] for s in spectra])
    ratio = np.array([s[0] / max(s[1], 1e-300) for s in spectra if len(s) > 1])

    return {
        "label": label,
        "hops": len(hops),
        "corner_mass": corner,
        "shape_by_width": shape,
        "r_at_threshold": r_counts,
        "cos_all": _q(flat),
        "cos_leading": _q(leading),
        "cos_second": _q(second),
        "cos_trailing": _q(trailing),
        "leading_over_second": _q(ratio),
        "width_hist": {str(w): len(g) for w, g in sorted(by_width.items())},
    }


# -- (5) audience differentiation, per #532's standing constraint from B42 -----


def audience_differentiation(dome, maps, chains) -> dict:
    """`1 - overlap` per cell, where overlap is normalised `sum cos^2 theta`.

    B42's anchor: two lanes carrying the *same* subspace read overlap 1.0000 and
    differentiation 0.0000, which is the flat bundle's stagger-0 row. Averaged
    over a cell's unordered carried-edge pairs, then reported over the interior
    cells the chains pass through.
    """
    interior = sorted(
        {cell for chain in chains for _, cell, _ in cg.hops_of(dome, tuple(chain["edges"]))}
    )
    per_cell = []
    for cell in interior:
        edges = carried_edges_of(dome, cell)
        if len(edges) < 2:
            continue
        vals = []
        for a, b in itertools.combinations(edges, 2):
            cos = hop_cos(dome, maps, (a, cell, b))
            width = min(dome.edges[a].m, dome.edges[b].m)
            vals.append(1.0 - float((cos**2).sum()) / max(width, 1))
        per_cell.append(float(np.mean(vals)))
    arr = np.array(per_cell)
    return {"cells": int(arr.size), **_q(arr)}


def read_all(agent, chains, label: str) -> dict:
    dome, maps = agent.dome, agent.sheaf.maps
    return {
        "label": label,
        "chain_hops": read_spectra(dome, maps, chain_hops(dome, chains), f"{label} / chain"),
        "carried_hops": read_spectra(dome, maps, carried_hops(dome, chains), f"{label} / carried"),
        "audience_differentiation": audience_differentiation(dome, maps, chains),
    }


def _line(tag: str, r: dict) -> str:
    c = r["chain_hops"]["corner_mass"]
    rr = r["chain_hops"]["r_at_threshold"]["0.95"]
    lead = r["chain_hops"]["cos_leading"]
    ad = r["audience_differentiation"]
    return (
        f"  {tag:<26} cos max {c['max']:.4f} | >=.95 {c['frac_ge_0.95']:.4f} "
        f"<=.05 {c['frac_le_0.05']:.4f} interior {c['frac_interior_05_95']:.4f} "
        f"| lead med {lead['median']:.4f} | r@.95 mean {rr['mean']:.3f} max {rr['max']} "
        f"| aud-diff med {ad['median']:.4f}"
    )


# -- the run ------------------------------------------------------------------


def stage_construction(seeds, out: Path) -> None:
    record = {"issue": 618, "stage": "construction", "surface": None, "seeds": {}}
    for seed in seeds:
        env, agent = build("real", "train", seed)
        try:
            record["surface"] = t0.surface()
            chains = t2.rim_chains(agent.dome)
            entry = read_all(agent, chains, f"construction seed {seed}")
            entry["structure"] = {
                "chains": len(chains),
                "n": int(agent.dome.shape.n),
                "carried_edges": int(sum(1 for e in agent.dome.edges if e.m > 0)),
                "dome_edges": len(agent.dome.edges),
            }
            record["seeds"][str(seed)] = entry
            print(_line(f"construction s{seed}", entry), flush=True)
        finally:
            env.close()
        out.write_text(json.dumps(record, indent=1))
    print(f"[B50] wrote {out.name}", flush=True)


def stage_trained(condition: str, seed: int, ticks: int, out: Path) -> None:
    from patchworks.learning import PredictionRule, TransportRule

    started = time.time()
    arm = t3.CONDITIONS[condition]
    env, agent = build("real", "train", seed)
    try:
        if arm["pin"]:
            t1.pin_drive_edges(agent)
        chains = t2.rim_chains(agent.dome)
        record = {
            "issue": 618,
            "stage": "trained",
            "condition": condition,
            "arm": {"rho1_drive_edges": bool(arm["pin"]), "c_learning_rate": float(arm["c"])},
            "seed": seed,
            "ticks": ticks,
            "surface": t0.surface(),
            "checkpoints": [],
        }
        recorder = t2.EdgeRecorder(agent)
        bias = PredictionRule(agent.sheaf, operator_rate_ratio=arm["c"])
        transport = TransportRule(agent.sheaf)

        record["at_construction"] = read_all(agent, chains, f"construction {condition} s{seed}")
        print(_line(f"{condition} s{seed} @0", record["at_construction"]), flush=True)
        out.write_text(json.dumps(record, indent=1))

        ladder = [cp for cp in t3.CHECKPOINTS if cp <= ticks]
        if ticks not in ladder:
            ladder.append(ticks)
        seen = 0
        for target in ladder:
            for _ in t0.teaching_read(agent, target - seen, seed + seen, recorder, bias, transport):
                pass
            seen = target
            entry = read_all(agent, chains, f"{condition} s{seed} @{target}")
            entry["ticks"] = target
            entry["elapsed_minutes"] = (time.time() - started) / 60.0
            record["checkpoints"].append(entry)
            # written at every checkpoint: a kill costs the one in flight, no more
            out.write_text(json.dumps(record, indent=1))
            print(
                _line(f"{condition} s{seed} @{target}", entry)
                + f" ({entry['elapsed_minutes']:.1f} min)",
                flush=True,
            )
    finally:
        env.close()
    print(f"[B50] wrote {out.name}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("construction", "trained"), required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--condition", default="baseline", choices=sorted(t3.CONDITIONS))
    p.add_argument("--ticks", type=int, default=20_000)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    if args.stage == "construction":
        out = args.out or _HERE / "618-construction.json"
        stage_construction(args.seeds, out)
    else:
        out = args.out or _HERE / f"618-{args.condition}-seed{args.seeds[0]}-{args.ticks}.json"
        stage_trained(args.condition, args.seeds[0], args.ticks, out)


if __name__ == "__main__":
    main()
