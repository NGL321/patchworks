"""B68 (#645) item 2: what `PredictionRule` does to the traffic, per cell.

`#645 <https://github.com/NGL321/patchworks/issues/645>`_, under
`the map <https://github.com/NGL321/patchworks/issues/532>`_.

`B62 (#635) <https://github.com/NGL321/patchworks/issues/635>`_ established the
*location* of the antagonism -- one rule writes `K`, the other writes the maps --
and said in as many words that a location is not a mechanism. This file takes
the two readings B62 could not.

**Reading 1: is the collapse per cell or across cells?**  The traffic is
`sheaf.evidence()`, `[predicting cells, n]`, flattened in `dome.predicting`
order, so column block `c·n : (c+1)·n` *is* cell `c`'s own node stalk and the
blocks are aligned with `CellOperators`' own `[cells, ·]` leading dimension --
no second convention to keep in step. A global effective rank of 1 over the
assembled 4,800-column configuration can be reached two quite different ways:

* every cell's own block collapses to one direction (a **per-cell** collapse), or
* the blocks keep their own dimensionality but their *temporal profiles* lock
  together, so the assembly moves as one (a **synchrony** collapse).

They are separable and the file separates them: per-cell block ER on one side,
and on the other the effective rank of the `[T, cells]` matrix of each block's
leading temporal coefficient. The distinction is the whole of what "the world
loop then samples one direction of it" would have to mean.

**Reading 2: does the per-cell collapse track `rho(used)` or `tau`?**  #645's
candidate account, off
`B44 (#610) <https://github.com/NGL321/patchworks/issues/610>`_: the rule drives
`K` toward something low-rank or strongly non-normal and the loop samples one
direction of it. B44 already put `rho(K)` moving on 149 of 150 cells while the
band pins `sigma(used)` at exactly 1.000 -- so above the band's face
`rho(used) = rho(K)/sigma(K)` exactly and retention **is** `K`'s departure from
normality. Whether the traffic's rank tracks that quantity **per cell** is a
correlation nobody has taken. Taken here as Spearman across cells, at every
checkpoint, on the arm that produces the collapse and on the two that do not.

**Three arms, and the controls are not optional.** `bias` is the arm under test;
`frozen` is the null the correlation must not also hold on (a correlation
present with nothing learning is a fact about construction); `transport` is the
arm that moves the traffic the *other* way, and if the same correlation appears
there with the rank rising, the account is not about `K` at all. B44's rig reads
`K` and B62's reads the traffic; this file is both at once, on one arm at a time.

**Horizon.** 2,000 ticks. The rules-on collapse is complete by 1,000 and settled
by 2,000 (B57: 2.9825 at 500, 1.0967 at 1,000), and item 1 carries the long
horizon separately. Per `B38 (#599) <https://github.com/NGL321/patchworks/issues/599>`_
the stall stamp is per-run, so it is stamped here rather than inherited.

Usage::

    PYTHONPATH=src python prototypes/cold-start/T6/b68_mechanism.py --mode bias
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)

_HERE = Path(__file__).resolve().parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


b57 = _load("b68m_b57", _HERE / "b57_agreement.py")
b62 = _load("b68m_b62", _HERE / "b62_frozen.py")

from patchworks.learning import PredictionRule, TransportRule  # noqa: E402

#: The buffer, and the window every reading below is taken at. B57's own.
BUFFER = 2_000
WINDOW = b57.WINDOW
#: The ladder. Dense before 1,000 because that is where the collapse happens.
CHECKPOINTS = (100, 200, 500, 1_000, 2_000)
#: Cell pairs sampled for the synchrony reading. 150 cells is 11,175 pairs,
#: which is cheap; the cap is here so a wider dome does not change the file.
MAX_PAIRS = 20_000


def effective_rank(weights: np.ndarray) -> float:
    """B57's own :func:`b57_agreement.soft_count` and deliberately not a second
    definition.

    ADR-0010's **participation ratio**, `(sum w)^2 / sum w^2`. Not `exp(H)`:
    every effective-rank figure this ticket compares against -- B57's 2.766,
    B62's 2.7798 / 1.0089 / 3.0024 -- is the participation form, and two
    statistics both called *effective rank* would put a spurious difference into
    the per-cell / global comparison this file exists to make. `B58 (#630)
    <https://github.com/NGL321/patchworks/issues/630>`_'s void clause names
    participation as the quantity as well.
    """
    return b57.soft_count(np.asarray(weights, dtype=float))


def block_readings(x: np.ndarray, cells: int, n: int) -> dict:
    """Per-cell block ER and each block's leading temporal coefficient.

    `x` is `[T, cells*n]` and block `c` is `x[:, c*n:(c+1)*n]`. Uncentered, per
    B62's window rule -- the centred form is bounded by the window rather than
    by the surface and doubles with `T` on an untrained arm, so magnitude is
    argued uncentered. The centred form is reported beside it and never alone.
    """
    er, er_centred, leading, weight = [], [], [], []
    for c in range(cells):
        block = x[:, c * n : (c + 1) * n]
        u, s, _ = np.linalg.svd(block, full_matrices=False)
        lam = s.astype(np.float64) ** 2
        er.append(effective_rank(lam))
        leading.append(u[:, 0] * (1.0 if u[0, 0] >= 0 else -1.0))
        weight.append(float(lam.sum()))
        centred = block - block.mean(axis=0, keepdims=True)
        s_c = np.linalg.svd(centred, compute_uv=False)
        er_centred.append(effective_rank(s_c.astype(np.float64) ** 2))
    return {
        "er": np.asarray(er),
        "er_centred": np.asarray(er_centred),
        "leading": np.stack(leading, axis=1),  # [T, cells]
        "weight": np.asarray(weight),
    }


def synchrony(leading: np.ndarray, rng: np.random.Generator) -> dict:
    """How locked the blocks' leading temporal profiles are to one another.

    Two forms, because neither alone says it. The **pairwise** form is the
    median `|cos|` between leading profiles over sampled cell pairs -- a
    magnitude, comparable across arms. The **joint** form is the effective rank
    of the `[T, cells]` leading-profile matrix itself: `1.0` means every cell is
    riding one temporal signal, `cells` means they are independent. A global
    collapse to ER 1 requires *both* per-cell blocks at rank ~1 and this at ~1;
    reporting them apart is what makes the two accounts separable.
    """
    t, cells = leading.shape
    unit = leading / np.maximum(np.linalg.norm(leading, axis=0, keepdims=True), 1e-300)
    pairs = cells * (cells - 1) // 2
    if pairs <= MAX_PAIRS:
        i, j = np.triu_indices(cells, k=1)
    else:
        i = rng.integers(0, cells, MAX_PAIRS)
        j = rng.integers(0, cells, MAX_PAIRS)
        keep = i != j
        i, j = i[keep], j[keep]
    cos = np.abs((unit[:, i] * unit[:, j]).sum(axis=0))
    s = np.linalg.svd(unit, compute_uv=False)
    return {
        "pairs": int(cos.size),
        "abs_cos_median": float(np.median(cos)),
        "abs_cos_q90": float(np.percentile(cos, 90)),
        "abs_cos_mean": float(cos.mean()),
        "joint_effective_rank": effective_rank(s.astype(np.float64) ** 2),
        "cells": int(cells),
    }


def operator_readings(operators) -> dict:
    """`rho(K)`, `sigma(K)`, `rho(used)` and `tau`, per cell, off the body's own
    accessors -- B44's :func:`b44_return.spectra` in per-cell form.

    `tau` is ADR-0026's currency, `-1/log(rho(used))`, reported as `inf` at or
    above one rather than clamped, which is the honesty B44 chose at that face.
    """
    with torch.no_grad():
        rho_raw = operators.raw_radii().to(torch.float64).cpu().numpy()
        sigma_raw = operators.raw_norms.to(torch.float64).cpu().numpy()
        rho_used = operators.radii().to(torch.float64).cpu().numpy()
        sigma_used = operators.norms.to(torch.float64).cpu().numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        tau = np.where(
            (rho_used > 0.0) & (rho_used < 1.0),
            -1.0 / np.log(np.clip(rho_used, 1e-300, 1.0)),
            np.inf,
        )
    return {
        "rho_raw": rho_raw,
        "sigma_raw": sigma_raw,
        "rho_used": rho_used,
        "sigma_used": sigma_used,
        "normality": rho_raw / np.maximum(sigma_raw, 1e-300),
        "tau": tau,
    }


def spearman(a: np.ndarray, b: np.ndarray) -> dict:
    """Rank correlation, with the finite mask and `n` reported beside it.

    Spearman rather than Pearson because `tau` is `inf` on cells at or above the
    band's face and because neither quantity has any reason to be linear in the
    other; the claim under test is *tracks*, which is monotone. `n` is quoted
    with every coefficient -- a correlation on the handful of cells where `tau`
    is finite is not the same statement as one on all 150.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    mask = np.isfinite(a) & np.isfinite(b)
    n = int(mask.sum())
    if n < 3:
        return {"rho": None, "n": n}
    x, y = a[mask], b[mask]
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return {"rho": None, "n": n, "note": "constant on one side"}
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = float(np.linalg.norm(rx) * np.linalg.norm(ry))
    return {"rho": float((rx * ry).sum() / denom) if denom else None, "n": n}


def quantiles(a: np.ndarray) -> dict:
    a = np.asarray(a, float)
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return {"n_finite": 0}
    return {
        "n_finite": int(finite.size),
        "min": float(finite.min()),
        "p10": float(np.percentile(finite, 10)),
        "median": float(np.median(finite)),
        "mean": float(finite.mean()),
        "p90": float(np.percentile(finite, 90)),
        "max": float(finite.max()),
    }


def checkpoint(agent, traffic, rng, cells: int, n: int) -> dict:
    x = traffic.matrix(WINDOW)
    blocks = block_readings(x, cells, n)
    ops = operator_readings(agent.sheaf.operators)
    w_unc, _ = b57.directions(x)
    w_cen, _ = b57.directions(x, centred=True)
    entry = {
        "window_ticks": int(x.shape[0]),
        "global": {
            "traffic_effective_rank": effective_rank(w_unc),
            "traffic_effective_rank_centred": effective_rank(w_cen),
        },
        "per_cell_block_er": quantiles(blocks["er"]),
        "per_cell_block_er_centred": quantiles(blocks["er_centred"]),
        "synchrony": synchrony(blocks["leading"], rng),
        "operators": {
            "rho_raw": quantiles(ops["rho_raw"]),
            "sigma_raw": quantiles(ops["sigma_raw"]),
            "rho_used": quantiles(ops["rho_used"]),
            "sigma_used": quantiles(ops["sigma_used"]),
            "normality": quantiles(ops["normality"]),
            "tau": quantiles(ops["tau"]),
            "cells_tau_infinite": int((~np.isfinite(ops["tau"])).sum()),
            "cells_sigma_used_above_band": int((ops["sigma_used"] > 1.0).sum()),
        },
        "correlations": {
            "block_er_vs_rho_used": spearman(blocks["er"], ops["rho_used"]),
            "block_er_vs_tau": spearman(blocks["er"], ops["tau"]),
            "block_er_vs_normality": spearman(blocks["er"], ops["normality"]),
            "block_er_vs_rho_raw": spearman(blocks["er"], ops["rho_raw"]),
            "block_er_vs_block_weight": spearman(blocks["er"], blocks["weight"]),
        },
        "per_cell": {
            "block_er": [float(v) for v in blocks["er"]],
            "rho_used": [float(v) for v in ops["rho_used"]],
            "rho_raw": [float(v) for v in ops["rho_raw"]],
            "normality": [float(v) for v in ops["normality"]],
        },
    }
    return entry


def _line(tag: str, e: dict) -> str:
    c = e["correlations"]
    s = e["synchrony"]

    def r(node):
        return "   n/a" if node["rho"] is None else f"{node['rho']:6.3f}"

    return (
        f"  {tag:<20} global-ER {e['global']['traffic_effective_rank']:7.4f} | "
        f"block-ER med {e['per_cell_block_er']['median']:6.3f} | "
        f"sync |cos| {s['abs_cos_median']:6.4f} joint-ER {s['joint_effective_rank']:7.3f} | "
        f"rho(used) med {e['operators']['rho_used']['median']:6.4f} | "
        f"r(ER,rho) {r(c['block_er_vs_rho_used'])} r(ER,tau) {r(c['block_er_vs_tau'])} "
        f"r(ER,nn) {r(c['block_er_vs_normality'])}"
    )


def run(mode: str, condition: str, seed: int, ticks: int, out: Path) -> None:
    started = time.time()
    env, agent, arm, flat_info = b57.build_arm(condition, seed)
    try:
        cells = len(agent.dome.predicting)
        n = int(agent.dome.shape.n)
        rng = np.random.default_rng(seed)
        record = {
            "issue": 645,
            "item": 2,
            "reading": "per-cell traffic collapse against per-cell K, on one rule at a time",
            "mode": mode,
            "condition": condition,
            "arm": {"rho1_drive_edges": bool(arm["pin"]), "c_learning_rate": float(arm["c"])},
            "flat_bundle": flat_info,
            "seed": seed,
            "ticks": ticks,
            "window": WINDOW,
            "buffer": BUFFER,
            "cells": cells,
            "n": n,
            "surface": b57.t0.surface(),
            "checkpoints": [],
        }
        traffic = b57.Traffic(agent, BUFFER)
        recorder = b57.t2.EdgeRecorder(agent)
        bias = PredictionRule(agent.sheaf, operator_rate_ratio=arm["c"]) if mode in ("bias", "both") else None
        transport = TransportRule(agent.sheaf) if mode in ("transport", "both") else None

        # The buffer is filled with no rule running in any mode, so every arm
        # starts from the same reading and B49's construction / trained line is
        # paid rather than inherited. B62's own opening move.
        b62.step(agent, BUFFER, seed, recorder, None, None, traffic)
        entry = checkpoint(agent, traffic, rng, cells, n)
        entry["ticks"] = 0
        record["at_construction"] = entry
        print(_line(f"{mode} @0", entry), flush=True)
        out.write_text(json.dumps(record, indent=1))

        ladder = [cp for cp in CHECKPOINTS if cp <= ticks]
        if ticks not in ladder:
            ladder.append(ticks)
        seen = 0
        for target in ladder:
            b62.step(agent, target - seen, seed + seen, recorder, bias, transport, traffic)
            seen = target
            entry = checkpoint(agent, traffic, rng, cells, n)
            entry["ticks"] = target
            entry["elapsed_minutes"] = (time.time() - started) / 60.0
            record["checkpoints"].append(entry)
            out.write_text(json.dumps(record, indent=1))
            print(_line(f"{mode} @{target}", entry) + f" ({entry['elapsed_minutes']:.1f} min)", flush=True)
    finally:
        env.close()
    print(f"[B68] wrote {out.name}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("frozen", "bias", "transport", "both"), required=True)
    p.add_argument("--condition", default="baseline", choices=("baseline", "winner", "flat"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ticks", type=int, default=2_000)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    out = args.out or _HERE / f"645-mech-{args.mode}-{args.condition}-seed{args.seed}-{args.ticks}.json"
    run(args.mode, args.condition, args.seed, args.ticks, out)


if __name__ == "__main__":
    main()
