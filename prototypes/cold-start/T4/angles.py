"""T4 (#537): principal angles per relay cell, and whether `c` alone moves composed rank.

`#537 <https://github.com/NGL321/patchworks/issues/537>`_, under
`the map <https://github.com/NGL321/patchworks/issues/532>`_.

**The object.** `#533 <https://github.com/NGL321/patchworks/issues/533>`_ found
each interior hop is `F_out · F_inᵀ` with **both maps held by the same relay
cell**. Under ADR-0032's spectral floor each is a scaled co-isometry
`F = σ·U Vᵀ`, so

    `hop = σ² · U_out (V_outᵀ V_in) U_inᵀ`

and the hop's spectrum is `σ²·cos θ`, the cosines of the **principal angles**
between that cell's two carried subspaces (the row spaces `V_in`, `V_out`).
Seven hops multiply seven cosine matrices, up to the `U`-rotations between them.

**The ambient is the cell's open mask, not the stalk.** All of a cell's incident
maps share one structural mask, a prefix of its node stalk of width `k_v`
(`Dome.restriction_mask`; it is also why `_push_apart` can use one shared
right-transform per cell). So `V_in` and `V_out` are `m`-dimensional subspaces of
the same `k_v`-dimensional space, and the genericity that matters is `m / k_v`,
not `m / n`.

Reads, per #537's list: the angle distribution (1), whether the product of the
per-hop cosine spectra reproduces the observed composed ER (2), whether the
top-ER chains are the best-aligned chains (3), domination versus rank (4), and
the two synthetic sensitivities -- `c` alone (5) and `m/n` alone (6).

Usage::

    PYTHONPATH=src python prototypes/cold-start/T4/angles.py --seeds 42 43 44
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
_T0, _T1, _T2 = _HERE.parent / "T0", _HERE.parent / "T1", _HERE.parent / "T2"
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

import construction_grading as cg  # noqa: E402
from patchworks.restriction import pair_index  # noqa: E402
from untrained_fixed_point import build  # noqa: E402


def effective_rank(s: np.ndarray) -> float:
    """#436's quantity, verbatim from `composed_reads`: `(Σσ²)² / Σσ⁴`."""
    s2 = s.astype(np.float64) ** 2
    return float(s2.sum() ** 2 / max(float((s2**2).sum()), 1e-300))


@torch.no_grad()
def carried(dome, maps, edge_id: int, cell_id: int) -> tuple[np.ndarray, np.ndarray, float]:
    """`(U, V, σ)` of the map `cell_id` holds into `edge_id`, on its active block.

    The active block is `[:m_e, :k_v]` -- `m_e` live rows, and the mask prefix of
    width `k_v` shared by every map this cell holds. `V` is the **carried
    subspace**: the `m`-frame in the cell's own `k_v`-dimensional mask that this
    map reads.
    """
    m = dome.edges[edge_id].m
    k = int(dome.restriction_mask(edge_id, cell_id).sum())
    f = maps.maps[pair_index(edge_id, cg.side_of(dome, edge_id, cell_id))][:m, :k]
    u, s, vh = torch.linalg.svd(f.double(), full_matrices=False)
    return u.numpy(), vh.numpy().T, s.numpy()


@torch.no_grad()
def hop_angles(dome, maps, key) -> dict:
    """`cos θ` of the principal angles between one relay cell's two carried subspaces."""
    edge_in, cell, edge_out = key
    _, v_in, s_in = carried(dome, maps, edge_in, cell)
    _, v_out, s_out = carried(dome, maps, edge_out, cell)
    cos = np.linalg.svd(v_out.T @ v_in, compute_uv=False)
    return {
        "cell": int(cell),
        "k_v": int(dome.restriction_mask(edge_in, cell).sum()),
        "m_in": int(dome.edges[edge_in].m),
        "m_out": int(dome.edges[edge_out].m),
        "cos": cos,
        "sigma_ratio_in": float(s_in.min() / max(s_in.max(), 1e-300)),
        "sigma_ratio_out": float(s_out.min() / max(s_out.max(), 1e-300)),
    }


@torch.no_grad()
def chain_read(dome, maps, chain: dict) -> dict:
    """One chain: its per-hop angle spectra, the true composed spectrum, and both ERs.

    `cos_product_er` multiplies the per-hop **cosine spectra** as bare numbers --
    #537's item 2, the prediction the mechanism makes if the `U`-rotations
    between hops were absent. `composed_er` is the truth, from the same
    `hop_operator` chain `composed_reads` uses. The gap between them is exactly
    what the `U`-rotations carry.
    """
    hops = cg.hops_of(dome, tuple(chain["edges"]))
    angles = [hop_angles(dome, maps, key) for key in hops]

    composed = t2.hop_operator(dome, maps, hops[0])
    for key in hops[1:]:
        composed = t2.hop_operator(dome, maps, key) @ composed
    spectrum = torch.linalg.svdvals(composed).numpy()

    width = min(len(a["cos"]) for a in angles)
    product = np.ones(width, dtype=np.float64)
    for a in angles:
        product = product * np.sort(a["cos"])[::-1][:width]

    flat = np.concatenate([a["cos"] for a in angles])
    ratios = [a["sigma_ratio_in"] for a in angles] + [a["sigma_ratio_out"] for a in angles]
    return {
        "sigma_ratio": float(np.min(ratios)),
        "rim": chain["rim"],
        "kind": chain["kind"],
        "apex": chain["apex"],
        "hops": len(hops),
        "cos_all": flat,
        "cos_mean": float(flat.mean()),
        "cos_min": float(flat.min()),
        "cos_top_mean": float(np.mean([np.sort(a["cos"])[::-1][0] for a in angles])),
        "cos_second_mean": float(
            np.mean([np.sort(a["cos"])[::-1][1] for a in angles if len(a["cos"]) > 1])
        ),
        "spectrum": spectrum,
        "composed_er": effective_rank(spectrum),
        "cos_product": product,
        "cos_product_er": effective_rank(product),
        "sigma_max": float(spectrum[0]),
        "s2_over_s1": float(spectrum[1] / max(spectrum[0], 1e-300)) if len(spectrum) > 1 else 0.0,
    }


# -- the two synthetic sensitivities (#537 items 5 and 6) ----------------------


@torch.no_grad()
def sweep_c(agent, chains: list[dict], values) -> list[dict]:
    """#537 item 5: composed ER against ADR-0010's `c`, synthetically on these maps.

    `c` reaches the surface only through :meth:`RestrictionMaps._push_apart`,
    which caps each holding cell's summed Gram at `g_v^2 * c_v` and water-fills
    the excess back. So the honest synthetic is to **re-run the projection with a
    different `c`** and re-read the composed spectrum: nothing else in the build
    is a function of `c`. `overlap_counts`' two clamps are kept -- the pigeonhole
    floor and `c_v = deg(v)` on a wholly-pinned incidence (#228) -- because
    relaxing `c` past them is a different ADR's question.
    """
    from patchworks.restriction import cell_gauges, overlap_counts

    maps = agent.sheaf.maps
    keep = maps.maps.detach().clone()
    target_keep = maps.overlap_target.clone()
    out = []
    try:
        for c in values:
            maps.maps.copy_(keep)
            maps.overlap_target.copy_(
                cell_gauges(agent.dome, rho=maps.rho) ** 2
                * overlap_counts(agent.dome, c=int(c))
            )
            maps.project()
            er = np.array([chain_read(agent.dome, maps, ch)["composed_er"] for ch in chains])
            out.append(
                {
                    "c": int(c),
                    "target_median": float(np.median(maps.overlap_target.numpy())),
                    "er_median": float(np.median(er)),
                    "er_mean": float(er.mean()),
                    "er_p90": float(np.quantile(er, 0.90)),
                    "er_max": float(er.max()),
                }
            )
    finally:
        maps.maps.copy_(keep)
        maps.overlap_target.copy_(target_keep)
    return out


@torch.no_grad()
def cap_bite(agent, values) -> list[dict]:
    """Does the `c` cap fire at all? The peak Gram eigenvalue against its target.

    `_push_apart` is a no-op at a cell already inside the bound, so `c` can only
    move the surface at cells where `lambda_max(sum_e F^T F)` reaches
    `g_v^2 * c_v`. This reads the ratio before the cap runs, which is what says
    whether item 5's sweep is measuring a live constraint or a slack one.
    """
    maps = agent.sheaf.maps
    peaks = maps.gram_peaks().numpy()
    target = maps.overlap_target.numpy()
    live = target > 0
    ratio = np.where(live, peaks / np.maximum(target, 1e-300), np.nan)
    return [
        {
            "cells_measured": int(live.sum()),
            "ratio_median": float(np.nanmedian(ratio)),
            "ratio_p90": float(np.nanquantile(ratio, 0.90)),
            "ratio_max": float(np.nanmax(ratio)),
            "cells_at_cap": int(np.nansum(ratio > 1 - 1e-6)),
        }
    ]


@torch.no_grad()
def sweep_m(dome, chains: list[dict], widths) -> list[dict]:
    """#537 item 6: composed ER against interior lane width `m`, at fixed `c`.

    The maps are not rebuilt -- a retrain is not what item 6 asks for. Each hop's
    two carried subspaces are **redrawn as random `m`-frames in the relay cell's
    own `k_v`-dimensional mask**, keeping every other structural fact (chain
    length, which cell is which, each cell's `k_v`, which lanes are boundary)
    exactly as the surface has it. That isolates the genericity claim: what does
    composed ER look like when the only thing that changes is how much of the
    mask each lane spans?

    Boundary lanes keep their built `m`; only interior lanes move, because
    `boundary_m` is a different knob with a different ADR (#474).
    """
    rng = np.random.default_rng(0)
    interior_m = min(e.m for e in dome.edges if e.m > 1)
    out = []
    for width in widths:
        er = []
        for chain in chains:
            composed = None
            for edge_in, cell, edge_out in cg.hops_of(dome, tuple(chain["edges"])):
                k = int(dome.restriction_mask(edge_in, cell).sum())
                m_in, m_out = dome.edges[edge_in].m, dome.edges[edge_out].m
                mi = min(width, k) if m_in == interior_m else min(m_in, k)
                mo = min(width, k) if m_out == interior_m else min(m_out, k)
                v_in = np.linalg.qr(rng.standard_normal((k, mi)))[0]
                v_out = np.linalg.qr(rng.standard_normal((k, mo)))[0]
                # A scaled co-isometry with a random left frame, per ADR-0032.
                u_in = np.linalg.qr(rng.standard_normal((mi, mi)))[0]
                u_out = np.linalg.qr(rng.standard_normal((mo, mo)))[0]
                hop = u_out @ (v_out.T @ v_in) @ u_in.T
                composed = hop if composed is None else hop @ composed
            er.append(effective_rank(np.linalg.svd(composed, compute_uv=False)))
        er = np.array(er)
        out.append(
            {
                "m_interior": int(width),
                "er_median": float(np.median(er)),
                "er_mean": float(er.mean()),
                "er_p90": float(np.quantile(er, 0.90)),
                "er_max": float(er.max()),
            }
        )
    return out


# -- the read -----------------------------------------------------------------


def _q(a: np.ndarray) -> dict:
    return {
        "min": float(a.min()),
        "p10": float(np.quantile(a, 0.10)),
        "median": float(np.median(a)),
        "p90": float(np.quantile(a, 0.90)),
        "max": float(a.max()),
        "mean": float(a.mean()),
    }


def read_surface(agent, chains: list[dict], label: str) -> dict:
    """#537 items 1-4 on one surface."""
    dome, maps = agent.dome, agent.sheaf.maps
    rows = [chain_read(dome, maps, ch) for ch in chains]
    er = np.array([r["composed_er"] for r in rows])
    pred = np.array([r["cos_product_er"] for r in rows])
    cos_mean = np.array([r["cos_mean"] for r in rows])
    cos_min = np.array([r["cos_min"] for r in rows])
    cos_top = np.array([r["cos_top_mean"] for r in rows])
    cos_second = np.array([r["cos_second_mean"] for r in rows])
    all_cos = np.concatenate([r["cos_all"] for r in rows])
    ratio = np.array([r["s2_over_s1"] for r in rows])
    sigma_ratio = np.array([r["sigma_ratio"] for r in rows])

    order = np.argsort(er)[::-1]
    cut = max(1, len(order) // 20)
    top, rest = order[:cut], order[cut:]

    width = min(len(r["spectrum"]) for r in rows)
    spectra = np.stack([r["spectrum"][:width] / max(r["spectrum"][0], 1e-300) for r in rows])

    return {
        "label": label,
        "chains": len(rows),
        "hops": int(rows[0]["hops"]),
        # (1) the angle distribution itself
        "cos_all": _q(all_cos),
        "cos_leading_per_hop": _q(np.array([r["cos_top_mean"] for r in rows])),
        "cos_second_per_hop": _q(cos_second),
        "cos_min_per_chain": _q(cos_min),
        "sigma_min_over_max": _q(sigma_ratio),
        # (2) does the product of cosine spectra reproduce the composed ER?
        "composed_er": _q(er),
        "cos_product_er": _q(pred),
        "er_vs_product_corr": float(np.corrcoef(er, pred)[0, 1]),
        "er_minus_product_median": float(np.median(er - pred)),
        # (3) are the top-ER chains the best-aligned chains?
        "alignment_corr": {
            "cos_mean": float(np.corrcoef(er, cos_mean)[0, 1]),
            "cos_top_mean": float(np.corrcoef(er, cos_top)[0, 1]),
            "cos_second_mean": float(np.corrcoef(er, cos_second)[0, 1]),
            "cos_min": float(np.corrcoef(er, cos_min)[0, 1]),
        },
        "top5pct": {
            "n": int(len(top)),
            "er_mean": float(er[top].mean()),
            "cos_second_mean": float(cos_second[top].mean()),
            "cos_mean": float(cos_mean[top].mean()),
        },
        "rest": {
            "n": int(len(rest)),
            "er_mean": float(er[rest].mean()),
            "cos_second_mean": float(cos_second[rest].mean()),
            "cos_mean": float(cos_mean[rest].mean()),
        },
        # (4) domination or rank -- the full composed spectrum, normalised
        "spectrum_normalised_median": [float(x) for x in np.median(spectra, axis=0)],
        "spectrum_normalised_p90": [float(x) for x in np.quantile(spectra, 0.90, axis=0)],
        "s2_over_s1": _q(ratio),
        "rank_numeric_median": float(np.median((spectra > 1e-12).sum(axis=1))),
        "per_chain": {
            "rim": [r["rim"] for r in rows],
            "kind": [r["kind"] for r in rows],
            "effective_rank": [float(x) for x in er],
            "cos_product_er": [float(x) for x in pred],
            "cos_mean": [float(x) for x in cos_mean],
            "cos_second_mean": [float(x) for x in cos_second],
        },
    }


def main() -> None:
    import collections

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--c-values", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8, 12])
    p.add_argument("--m-values", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 8, 10, 12])
    p.add_argument("--out", type=Path, default=_HERE / "537-construction.json")
    args = p.parse_args()

    record = {"issue": 537, "surface": None, "seeds": {}}
    for seed in args.seeds:
        env, agent = build("real", "train", seed)
        try:
            record["surface"] = t0.surface()
            chains = t2.rim_chains(agent.dome)
            entry = {"construction": read_surface(agent, chains, f"construction seed {seed}")}

            kv, mm = collections.Counter(), collections.Counter()
            for ch in chains:
                for edge_in, cell, edge_out in cg.hops_of(agent.dome, tuple(ch["edges"])):
                    kv[int(agent.dome.restriction_mask(edge_in, cell).sum())] += 1
                    mm[f"{agent.dome.edges[edge_in].m}->{agent.dome.edges[edge_out].m}"] += 1
            entry["structure"] = {
                "chains": len(chains),
                "k": int(agent.sheaf.operators.shape.k),
                "n": int(agent.dome.shape.n),
                "k_v_hist": {str(k): v for k, v in sorted(kv.items())},
                "m_hist": dict(sorted(mm.items())),
            }
            print(
                f"[T4] seed {seed} construction ER median "
                f"{entry['construction']['composed_er']['median']:.4f} "
                f"max {entry['construction']['composed_er']['max']:.4f}",
                flush=True,
            )
            entry["cap_bite"] = cap_bite(agent, args.c_values)
            print(f"[T4] seed {seed} cap bite: {entry['cap_bite'][0]}", flush=True)
            entry["sweep_c"] = sweep_c(agent, chains, args.c_values)
            print(
                f"[T4] seed {seed} c sweep: "
                + ", ".join(f"c={r['c']}:{r['er_median']:.4f}" for r in entry["sweep_c"]),
                flush=True,
            )
            entry["sweep_m"] = sweep_m(agent.dome, chains, args.m_values)
            print(
                f"[T4] seed {seed} m sweep: "
                + ", ".join(f"m={r['m_interior']}:{r['er_median']:.4f}" for r in entry["sweep_m"]),
                flush=True,
            )
            record["seeds"][str(seed)] = entry
        finally:
            env.close()
        args.out.write_text(json.dumps(record, indent=1))
    print(f"[T4] wrote {args.out.name}", flush=True)


if __name__ == "__main__":
    main()
