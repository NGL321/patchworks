"""B50 (#618): how much of the corner mass is forced by dimension counting alone.

`#618 <https://github.com/NGL321/patchworks/issues/618>`_ names `r` as *the
number of directions in a lane pair transporting at gain ~1* and asks whether it
is a real design variable. This is the control that decides it.

**The floor.** Two subspaces of dimensions `m_in` and `m_out` inside one
`k_v`-dimensional mask must intersect in at least

    `f = max(0, m_in + m_out - k_v)`

dimensions, and every direction of that intersection is a principal angle of
**exactly** zero -- `cos theta = 1`. So `f` cosines sit at the corner before any
map is drawn, let alone trained. `f` is a function of the *widths and the mask*
and of nothing else: no seed, no objective, no tick can move it.

`r` is therefore only a design variable in whatever it has **above** `f`. This
reports `f` from the dome alone, and -- given a built surface -- checks the
identity `#{cos >= 1 - eps} == f` hop by hop, which is the claim that the corner
is *entirely* the floor.

Usage::

    PYTHONPATH=src python prototypes/cold-start/T4/b50_forced.py --seeds 42 43 44
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
b50 = _load("b50_spectra", _HERE / "b50_spectra.py")

from untrained_fixed_point import build  # noqa: E402

EPS = 1e-6


def forced_dim(dome, key) -> int:
    """`max(0, m_in + m_out - k_v)`: the corner mass no surface can avoid."""
    edge_in, cell, edge_out = key
    k = int(dome.restriction_mask(edge_in, cell).sum())
    return max(0, dome.edges[edge_in].m + dome.edges[edge_out].m - k)


def _hist(a: np.ndarray) -> dict:
    return {str(int(v)): int(c) for v, c in zip(*np.unique(a, return_counts=True))}


def read(agent, chains, label: str) -> dict:
    dome, maps = agent.dome, agent.sheaf.maps
    hops = b50.chain_hops(dome, chains)
    forced = np.array([forced_dim(dome, key) for key in hops])
    at_one, r95, widths = [], [], []
    for key in hops:
        cos = b50.hop_cos(dome, maps, key)
        at_one.append(int((cos >= 1.0 - EPS).sum()))
        r95.append(int((cos >= 0.95).sum()))
        widths.append(len(cos))
    at_one, r95, widths = np.array(at_one), np.array(r95), np.array(widths)
    excess = r95 - forced
    return {
        "label": label,
        "hops": int(len(hops)),
        "forced_mean": float(forced.mean()),
        "forced_hist": _hist(forced),
        "at_one_hist": _hist(at_one),
        "identity_holds_on": int((forced == at_one).sum()),
        "identity_of": int(len(hops)),
        "r95_mean": float(r95.mean()),
        "excess_mean": float(excess.mean()),
        "excess_hist": _hist(excess),
        "hops_all_forced": int((widths == forced).sum()),
        "width_mean": float(widths.mean()),
        "forced_share_of_width": float(forced.sum() / widths.sum()),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--out", type=Path, default=_HERE / "618-forced.json")
    args = p.parse_args()

    record = {"issue": 618, "read": "pigeonhole floor on the corner mass", "seeds": {}}
    for seed in args.seeds:
        env, agent = build("real", "train", seed)
        try:
            record["surface"] = t0.surface()
            chains = t2.rim_chains(agent.dome)
            entry = read(agent, chains, f"construction seed {seed}")
            record["seeds"][str(seed)] = entry
            print(
                f"  s{seed}: forced mean {entry['forced_mean']:.3f} | "
                f"r@.95 mean {entry['r95_mean']:.3f} | excess mean {entry['excess_mean']:.3f} | "
                f"identity {entry['identity_holds_on']}/{entry['identity_of']} | "
                f"forced share of width {entry['forced_share_of_width']:.4f}",
                flush=True,
            )
        finally:
            env.close()
        args.out.write_text(json.dumps(record, indent=1))
    print(f"[B50] wrote {args.out.name}", flush=True)


if __name__ == "__main__":
    main()
