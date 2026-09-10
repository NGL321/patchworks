"""B44 (#610): why `ρ(used)` falls while `ρ(K)` rises — the two are one identity.

`CellOperators.used` rescales radially by `clamp(σ(K), 1/ρ_K, 1) / σ(K)`, so above
the band's upper face `ρ(used) = ρ(K) / σ(K)` exactly. The retention constant the
map reads is therefore **not** a fact about how big `K` got; it is the ratio of
`K`'s spectral radius to its spectral norm — its departure from normality — and
that is the quantity one-step prediction error is actually moving.

Read per cell off the trained checkpoint rather than argued, and reported against
tick 0, where `K = a·I` makes the ratio exactly 1 by construction.

Usage::

    PYTHONPATH=src python prototypes/cold-start/T6/b44_normality.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
for extra in (_ROOT / "prototypes" / "chart-double-duty-166", _ROOT / "benchmarks"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


b33 = _load("b44n_b33", _HERE / "b33_coexist.py")

SEED, LEARN = 42, 2000


def read(operators) -> dict:
    with torch.no_grad():
        rho = operators.raw_radii().to(torch.float64).numpy()
        sig = operators.raw_norms.to(torch.float64).numpy()
        used = operators.radii().to(torch.float64).numpy()
    ratio = rho / np.maximum(sig, 1e-300)
    return {
        "rho_raw_median": float(np.median(rho)),
        "sigma_raw_median": float(np.median(sig)),
        "normality_ratio_median": float(np.median(ratio)),
        "normality_ratio_min": float(ratio.min()),
        "rho_used_median": float(np.median(used)),
        "cells_sigma_above_band": int((sig > 1.0).sum()),
        "cells": int(rho.size),
        # Above the band's upper face `used = K / sigma(K)`, and the rescale is
        # radial, so `rho(used)` should be `rho(K)/sigma(K)` on every such cell.
        # Reported as the worst relative disagreement rather than a bool: the
        # two sides are separate `eigvals` calls and an exact-equality claim
        # would be a claim about LAPACK.
        "identity_max_relative_error": float(
            np.max(np.abs(used[sig > 1.0] - ratio[sig > 1.0]) / ratio[sig > 1.0])
        )
        if int((sig > 1.0).sum())
        else None,
    }


def main() -> None:
    _env, agent = b33.arms_mod.build_arm(b33.ARM, SEED)
    out = {"issue": 610, "seed": SEED, "learn": LEARN, "at_construction": read(agent.sheaf.operators)}
    blob = torch.load(_HERE / f"610-trained-seed{SEED}-{LEARN}.pt", weights_only=False)
    agent.sheaf.operators.load_state_dict(blob["operators"])
    out["trained"] = read(agent.sheaf.operators)

    for label in ("at_construction", "trained"):
        r = out[label]
        print(
            f"{label:>16}: rho(K) {r['rho_raw_median']:.6f}  "
            f"sigma(K) {r['sigma_raw_median']:.6f}  "
            f"rho/sigma {r['normality_ratio_median']:.6f}  "
            f"rho(used) {r['rho_used_median']:.6f}  "
            f"cells over band {r['cells_sigma_above_band']}/{r['cells']}"
        )
    print(
        "\nrho(used) = rho(K)/sigma(K) above the band's upper face, worst "
        f"relative error {out['trained']['identity_max_relative_error']:.2e}"
    )
    path = _HERE / f"610-normality-seed{SEED}-{LEARN}.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {path.name}")


if __name__ == "__main__":
    main()
