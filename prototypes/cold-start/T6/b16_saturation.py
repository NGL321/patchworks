"""T6 (#564 / B16): does "the top cosine saturates" reproduce B13's whole ratio curve?

The discriminator [B16](#564) item 2 asks for. [B6](#546)'s headroom law and the
rank-one-update mechanism both predict erosion falls to zero as `p` rises; they
disagree about **why**, and therefore about what else could move it.

This is the mechanism's *quantitative* form, and it costs no training. If the
transport gradient is rank one with the cell's own node stalk on the input side,
and that stalk's traffic is near rank one (`b16_drive.py` reads its effective
rank at **1.08**), then training has exactly one direction of alignment pressure
per relay cell. Its fixed point is therefore:

    **the leading principal cosine of each hop goes to 1; the rest stay generic.**

Nothing else in the spectrum has anything dragging it. So take #540's generic
instrument at each arm's realised `(m_in, m_out, k_v, hops)`, saturate **only the
top cosine of every hop**, and compose. That prediction is arm-free and has no
free parameter — the saturation value is 1 because the traffic is one direction,
not because anything was fitted.

The test: `saturated_er / generic_er` against B13's measured trained-to-generic
ratios (0.629 at `p = 8`, 0.910 at 12, 1.004 at 16, 1.000 at 20). B6's law
predicts the ordering; only the mechanism predicts the **values**.

Usage::

    PYTHONPATH=src python prototypes/cold-start/T6/b16_saturation.py --p 0 4 8 12 16 20
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

#: B13's measured trained-to-generic ratios, from `560-curve-p.json` / the
#: [B14](#561) table. The bar this predictor is scored against.
B13_RATIO = {8: 0.629, 12: 0.910, 16: 1.004, 20: 1.000}


def composed_er(dome, chains, rng_seed: int, n_sat: int) -> dict:
    """#540's generic instrument, with the top `n_sat` cosines of each hop set to 1.

    `n_sat = 0` **is** `arms.generic_at` — the same construction, so the two
    columns are directly comparable and the ratio is not comparing instruments.
    """
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
            gram = v_out.T @ v_in
            if n_sat:
                a, s, bt = np.linalg.svd(gram, full_matrices=False)
                s = s.copy()
                s[: min(n_sat, len(s))] = 1.0
                gram = a @ np.diag(s) @ bt
            hop = u_out @ gram @ u_in.T
            composed = hop if composed is None else hop @ composed
        er.append(angles.effective_rank(np.linalg.svd(composed, compute_uv=False)))
    er = np.asarray(er)
    return {
        "median": float(np.median(er)),
        "mean": float(er.mean()),
        "p90": float(np.quantile(er, 0.90)),
        "max": float(er.max()),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--p", type=int, nargs="+", default=[0, 4, 8, 12, 16, 20])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rng-seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--out", type=Path, default=_HERE / "564-saturation.json")
    args = p.parse_args()

    record = {
        "issue": 564,
        "reading": (
            "the mechanism's parameter-free prediction: the top principal cosine "
            "of every hop saturates to 1 (one traffic direction), the rest stay "
            "generic. Scored against B13's measured trained/generic ratios."
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
            gen = [composed_er(dome, chains, s, 0) for s in args.rng_seeds]
            sat1 = [composed_er(dome, chains, s, 1) for s in args.rng_seeds]
            sat2 = [composed_er(dome, chains, s, 2) for s in args.rng_seeds]
            g = float(np.mean([r["median"] for r in gen]))
            s1 = float(np.mean([r["median"] for r in sat1]))
            s2 = float(np.mean([r["median"] for r in sat2]))
            row = {
                "p": pv,
                "arm": arm,
                "k_v": int(dome.restriction_mask(*_first_pair(dome, chains)).sum()),
                "generic_median": g,
                "sat1_median": s1,
                "sat2_median": s2,
                "predicted_ratio_sat1": s1 / max(g, 1e-300),
                "predicted_ratio_sat2": s2 / max(g, 1e-300),
                "measured_ratio": B13_RATIO.get(pv),
                "generic_seeds": gen,
                "sat1_seeds": sat1,
            }
            record["rows"].append(row)
            meas = row["measured_ratio"]
            print(
                f"  p={pv:>2} k_v={row['k_v']:>2} generic {g:.4f} | sat-1 {s1:.4f} "
                f"-> predicted ratio {row['predicted_ratio_sat1']:.3f}"
                + (f" | measured {meas:.3f} | err {row['predicted_ratio_sat1'] - meas:+.3f}" if meas else "")
                + f" | sat-2 ratio {row['predicted_ratio_sat2']:.3f}",
                flush=True,
            )
        finally:
            env.close()
    args.out.write_text(json.dumps(record, indent=1))
    print(f"  wrote {args.out.name}", flush=True)


def _first_pair(dome, chains):
    edge_in, cell, _ = cg.hops_of(dome, tuple(chains[0]["edges"]))[0]
    return edge_in, cell


if __name__ == "__main__":
    main()
