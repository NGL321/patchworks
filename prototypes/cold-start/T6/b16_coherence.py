"""T6 (#564 / B16): how many coherent junction directions reproduce the measured ratio?

`b16_junction.py` established **where** the erosion lives: scrambling the
junctions with random orthogonals, every hop left exactly as trained, restores
`p = 8` from 1.478 to **2.251** against a generic 2.265. The whole collapse is
carried by the junction rotations between consecutive hops.

This asks **how much** coherence it takes, and the answer is a prediction rather
than a fit. `b16_drive.py` reads the cell's node-stalk traffic at effective rank
**1.002** — the transport rule descends `‖F_v x_v − y_e‖ / (...)` on a stalk that
occupies essentially *one* direction, so the objective has evidence about one
direction per junction and no evidence at all about the rest. The prediction is
therefore

    **r = 1: exactly one junction direction goes coherent, the others stay generic.**

Note the ordering that makes this non-trivial. Perfectly sorted junctions — every
direction fed to its opposite number — give `cos_product_er` = **3.07**, *above*
the random baseline 2.25. Training lands at **1.48**, *below* it. Aligning
everything spreads the spectrum; aligning exactly one direction while the rest
stay incoherent is what produces domination. So `r = 1` is not "a bit of
alignment", it is a different configuration from both extremes, and it is the one
the rank-one traffic picks out.

`r` is swept over 0, 1, 2 so the reading falsifies as well as confirms: `r = 0`
must reproduce `scrambled_er` and `r = 2` must undershoot.

Usage::

    PYTHONPATH=src python prototypes/cold-start/T6/b16_coherence.py --p 8 12 16 20
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
_T0, _T2, _T4 = (_HERE.parent / n for n in ("T0", "T2", "T4"))
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
arms_mod = _load("t6_arms", _HERE / "arms.py")

import construction_grading as cg  # noqa: E402

#: B13's measured trained-to-generic ratios at the 100k horizon.
B13_RATIO = {8: 0.629, 12: 0.910, 16: 1.004, 20: 1.000}


def _complete(basis: np.ndarray, r: int, dim: int, rng) -> np.ndarray:
    """A full orthonormal `dim x dim` frame whose first `r` columns are `basis`."""
    cols = [basis[:, j] for j in range(min(r, basis.shape[1]))]
    out = np.asarray(cols).T if cols else np.zeros((dim, 0))
    filler = rng.standard_normal((dim, dim))
    for j in range(filler.shape[1]):
        v = filler[:, j]
        if out.shape[1]:
            v = v - out @ (out.T @ v)
        nrm = np.linalg.norm(v)
        if nrm < 1e-9:
            continue
        out = np.hstack([out, (v / nrm)[:, None]])
        if out.shape[1] == dim:
            break
    return out


def composed_er(dome, chains, rng_seed: int, r: int) -> dict:
    """#540's generic instrument, with `r` coherent directions at every junction.

    `r = 0` inserts a uniformly random orthogonal — the `scrambled_er`
    counterfactual. `r >= 1` builds a rotation that carries hop `k-1`'s top `r`
    output singular directions **exactly onto** hop `k`'s top `r` input
    directions and is random on the orthogonal complement.
    """
    rng = np.random.default_rng(rng_seed)
    er = []
    for chain in chains:
        hops = []
        for edge_in, cell, edge_out in cg.hops_of(dome, tuple(chain["edges"])):
            k = int(dome.restriction_mask(edge_in, cell).sum())
            mi = min(dome.edges[edge_in].m, k)
            mo = min(dome.edges[edge_out].m, k)
            v_in = np.linalg.qr(rng.standard_normal((k, mi)))[0]
            v_out = np.linalg.qr(rng.standard_normal((k, mo)))[0]
            u_in = np.linalg.qr(rng.standard_normal((mi, mi)))[0]
            u_out = np.linalg.qr(rng.standard_normal((mo, mo)))[0]
            hops.append(u_out @ (v_out.T @ v_in) @ u_in.T)

        svd = [np.linalg.svd(h, full_matrices=False) for h in hops]
        composed = hops[0]
        for i in range(1, len(hops)):
            dim = hops[i].shape[1]
            p_prev = svd[i - 1][0]  # hop i-1's output directions, in lane e_i
            q_here = svd[i][2].T  # hop i's input directions, in lane e_i
            a = _complete(p_prev, r, dim, rng)
            b = _complete(q_here, r, dim, rng)
            rot = b @ a.T
            composed = hops[i] @ rot @ composed
        er.append(angles.effective_rank(np.linalg.svd(composed, compute_uv=False)))
    er = np.asarray(er)
    return {
        "median": float(np.median(er)),
        "mean": float(er.mean()),
        "p90": float(np.quantile(er, 0.90)),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--p", type=int, nargs="+", default=[8, 12, 16, 20])
    p.add_argument("--r", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rng-seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--out", type=Path, default=_HERE / "564-coherence.json")
    args = p.parse_args()

    record = {
        "issue": 564,
        "reading": (
            "composed ER with r coherent junction directions, against B13's "
            "measured trained/generic ratios. r=1 is what rank-one traffic predicts."
        ),
        "b13_ratio": B13_RATIO,
        "rows": [],
    }
    for pv in args.p:
        arm = "reserve" if pv == 8 else f"reserve_p{pv}"
        env, agent = arms_mod.build_arm(arm, args.seed)
        try:
            dome = agent.dome
            chains = t2.rim_chains(dome)
            gen = arms_mod.generic_at(dome, chains)["median"]
            row = {"p": pv, "arm": arm, "generic_median": gen, "measured_ratio": B13_RATIO.get(pv)}
            for r in args.r:
                vals = [composed_er(dome, chains, s, r)["median"] for s in args.rng_seeds]
                m = float(np.mean(vals))
                row[f"r{r}_median"] = m
                row[f"r{r}_ratio"] = m / max(gen, 1e-300)
            record["rows"].append(row)
            meas = row["measured_ratio"]
            cols = " | ".join(f"r={r} {row[f'r{r}_ratio']:.3f}" for r in args.r)
            print(
                f"  p={pv:>2} generic {gen:.4f} | {cols}"
                + (f" || measured {meas:.3f}" if meas else ""),
                flush=True,
            )
        finally:
            env.close()
    args.out.write_text(json.dumps(record, indent=1))
    print(f"  wrote {args.out.name}", flush=True)


if __name__ == "__main__":
    main()
