"""T4 (#537): what the composed spectrum is actually a function of.

#537 item 2 asks whether "the product of the per-hop cosine spectra reproduces
the observed composed ER". Taken literally that test is **unsound**: singular
values are not multiplicative along a product, so an elementwise product of
seven sorted cosine spectra is neither an upper nor a lower bound on the
composed spectrum, and its failing would falsify nothing. `angles.py` reports it
for completeness and it does indeed fail (correlation 0.16).

The sound version of the same question is an **ablation ladder**. Each rung
keeps less of the real surface and rebuilds the composed operator from what is
left; the rung at which the composed ER distribution stops matching is the rung
that carries the information:

===  ==================================================================
(a)  exact -- `hop_operator` verbatim, the truth `composed_reads` reads
(b)  angles only -- `hop = V_out^T V_in`, the real frames, the `U`
     rotations between hops dropped
(c)  real frames, random `U` -- `hop = U~_out (V_out^T V_in) U~_in^T`
(d)  random frames -- `V` redrawn as a Haar `m`-frame in the same cell's
     own `k_v`-dimensional mask; nothing of the learned surface kept but
     the dimension counts
===  ==================================================================

If (d) reproduces (a), the composed effective rank is a function of the
`(m_e, k_v)` counts alone, and no amount of rearranging the surface within those
counts can move it. That is what decides #537's lever question, and it decides
it much more sharply than a correlation.

Usage::

    PYTHONPATH=src python prototypes/cold-start/T4/ablate.py --seeds 42 43 44
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
_T0, _T2 = _HERE.parent / "T0", _HERE.parent / "T2"
for extra in (_ROOT / "prototypes" / "chart-double-duty-166", _ROOT / "benchmarks", _T0):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


t0 = _load("t0_run", _T0 / "run.py")
t2 = _load("t2_run", _T2 / "run.py")
angles = _load("t4_angles", _HERE / "angles.py")

import construction_grading as cg  # noqa: E402
from untrained_fixed_point import build  # noqa: E402


def haar(rng, rows: int, cols: int) -> np.ndarray:
    """A Haar-random `cols`-frame in `R^rows`."""
    q, r = np.linalg.qr(rng.standard_normal((rows, cols)))
    return q * np.sign(np.diag(r))


def ladder(dome, maps, chains, rng) -> dict:
    """The four rungs, per chain."""
    out = {k: [] for k in "abcd"}
    for chain in chains:
        hops = cg.hops_of(dome, tuple(chain["edges"]))
        comp = {k: None for k in "abcd"}
        for key in hops:
            edge_in, cell, edge_out = key
            u_in, v_in, _ = angles.carried(dome, maps, edge_in, cell)
            u_out, v_out, _ = angles.carried(dome, maps, edge_out, cell)
            k_v = v_in.shape[0]
            c = v_out.T @ v_in

            built = {
                "a": t2.hop_operator(dome, maps, key).numpy(),
                "b": c,
                "c": haar(rng, c.shape[0], c.shape[0]) @ c @ haar(rng, c.shape[1], c.shape[1]).T,
            }
            rv_in, rv_out = haar(rng, k_v, v_in.shape[1]), haar(rng, k_v, v_out.shape[1])
            rc = rv_out.T @ rv_in
            built["d"] = (
                haar(rng, rc.shape[0], rc.shape[0]) @ rc @ haar(rng, rc.shape[1], rc.shape[1]).T
            )
            for rung, hop in built.items():
                comp[rung] = hop if comp[rung] is None else hop @ comp[rung]
        for rung in "abcd":
            out[rung].append(angles.effective_rank(np.linalg.svd(comp[rung], compute_uv=False)))
    return {k: np.array(v) for k, v in out.items()}


def summarise(rungs: dict) -> dict:
    exact = rungs["a"]
    names = {
        "a": "exact",
        "b": "angles only (U dropped)",
        "c": "real frames, random U",
        "d": "random frames, same (m, k_v)",
    }
    rows = []
    for key in "abcd":
        v = rungs[key]
        rows.append(
            {
                "rung": key,
                "what": names[key],
                "er_median": float(np.median(v)),
                "er_mean": float(v.mean()),
                "er_p90": float(np.quantile(v, 0.90)),
                "er_max": float(v.max()),
                "frac_above_1p1": float((v > 1.1).mean()),
                "corr_with_exact": float(np.corrcoef(exact, v)[0, 1]) if key != "a" else 1.0,
            }
        )
    return {"rows": rows, "per_chain": {k: [float(x) for x in v] for k, v in rungs.items()}}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--out", type=Path, default=_HERE / "537-ablation.json")
    args = p.parse_args()

    record = {"issue": 537, "seeds": {}}
    for seed in args.seeds:
        env, agent = build("real", "train", seed)
        try:
            record["surface"] = t0.surface()
            chains = t2.rim_chains(agent.dome)
            rng = np.random.default_rng(seed)
            summary = summarise(ladder(agent.dome, agent.sheaf.maps, chains, rng))
            record["seeds"][str(seed)] = summary
            print(f"[T4] seed {seed}", flush=True)
            for r in summary["rows"]:
                print(
                    f"   ({r['rung']}) {r['what']:<32} median {r['er_median']:.4f} "
                    f"mean {r['er_mean']:.4f} p90 {r['er_p90']:.4f} max {r['er_max']:.4f} "
                    f"| >1.1 {r['frac_above_1p1']:.3f} | corr(exact) {r['corr_with_exact']:+.3f}",
                    flush=True,
                )
        finally:
            env.close()
        args.out.write_text(json.dumps(record, indent=1))
    print(f"[T4] wrote {args.out.name}", flush=True)


if __name__ == "__main__":
    main()
