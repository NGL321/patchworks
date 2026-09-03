"""Tier 1 of #166: is the chart's twelve-dimensional budget spent twice?

[#166](https://github.com/NGL321/patchworks/issues/166) pre-registered a cheap
early read -- *"numerical rank of each learned `K` across the population. If
they saturate at 12, the width is binding."* -- as the first of two tiers.

**The pre-registered statistic cannot fire.** `CellOperators.__init__`
(`body.py:638-642`) builds `K = a.I`, which is rank 12 at construction, and
:meth:`~patchworks.body.CellOperators.project` restores the band by
`K.mul_(target / norms)` -- a *rescaling* of the whole operator, which moves
every singular value by one common factor and can send none of them to zero.
So numerical rank 12 is the **null** that construction hands out for free, not
a signal that the width is binding; the pre-registration reads the statistic
backwards. Rank *deficiency* is the only thing a rank read could ever report,
and the body's own docstring calls that "the failure on the body".

This rig therefore reports the numerical rank -- so that the pre-registered
number is on the record and the claim that it is vacuous is *measured* rather
than argued -- alongside the statistic that carries the question:

* **Stable rank**, `||K||_F^2 / sigma_max(K)^2`, in `[1, k]`. How many
  directions `K` actually uses, continuously. Unlike numerical rank it is not
  a threshold on a spectrum that never reaches zero.
* **The singular spectrum itself**, per cell, normalised by `sigma_max` -- the
  shape the rank read throws away.
* **Effective rank**, `exp(H)` of the normalised spectrum's Shannon entropy
  (Roy & Vetterli), a second continuous count that weights the tail differently
  from stable rank, so a conclusion resting on both is not resting on one
  arbitrary summary.

`a.I` has stable rank exactly `k` and a flat spectrum, so **construction is the
maximum**, and learning can only spend the budget down. That orientation is the
whole reason the continuous statistics are informative where the rank is not:
the read is *how far below 12 does driving push it*, and a population that
stays at 12 is one where learning has not concentrated the operator at all.

Driven exactly as [#274](https://github.com/NGL321/patchworks/issues/274) drove
it -- the `real` dome, `split=train`, both rules, through the same
`untrained_fixed_point.build`/`teaching` harness at the same checkpoint ladder
-- because #166's own gate ("needs charts from a graph that transmits", #155)
is void and #274's driven rig is what replaced it.

Usage::

    PYTHONPATH=src python prototypes/chart-double-duty-166/read.py --ticks 2000 --seeds 42 43 44
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_BENCH = str(Path(__file__).resolve().parents[2] / "benchmarks")
if _BENCH not in sys.path:
    sys.path.append(_BENCH)
from untrained_fixed_point import build, teaching  # noqa: E402

#: #274's ladder, so a checkpoint here is the same moment as a checkpoint there.
CHECKPOINTS = (100, 200, 500, 1_000, 2_000, 5_000, 10_000, 20_000, 30_000)

#: `torch.linalg.matrix_rank`'s own default tolerance, spelled out so the
#: vacuity claim is not resting on an undeclared threshold.
RANK_RTOL = None


def spectra(operators) -> np.ndarray:
    """`[cells, k]`: the singular values of every cell's `K`, descending."""
    with torch.no_grad():
        return torch.linalg.svdvals(operators.K.double()).cpu().numpy()


def moduli(operators) -> np.ndarray:
    """`[cells, k]`: `|lambda_i(K)|`, descending. The memory allocation itself.

    Singular values say how many directions `K` uses; eigenvalue moduli say how
    long it holds each one. A mode at `|lambda| -> 1` retains, a mode at
    `|lambda| -> 0` forgets within a tick, and `tau_i = -1 / log|lambda_i|` is
    that mode's retention constant -- the per-mode form of the spectrum
    [ADR-0028](../../docs/adr/0028-a-cell-holds-a-spectrum-of-retention-constants.md)
    grants each cell. This is the quantity the double-duty question is actually
    about: the twelve modes are shared between holding history and resolving
    fresh evidence, and where each mode sits on `[0, 1]` *is* the split.
    """
    with torch.no_grad():
        lam = torch.linalg.eigvals(operators.K.double()).abs().cpu().numpy()
    return -np.sort(-lam, axis=1)


def nonnormality(operators) -> np.ndarray:
    """`[cells]`: `||K^T K - K K^T||_F / ||K||_F^2`, the departure from normal.

    The **tier-0 instrument #167 prescribed** and this ticket did not plan:
    *"measure the non-normality of learned `K` ... alongside the numerical rank
    the ticket already plans to take. Rank saturation and non-normality answer
    different halves of the same question."*

    It is the half that decides whether the memory ceiling is *reachable* at
    all. Ganguli, Huh & Sompolinsky (PNAS 2008) put the Fisher memory capacity
    of a **normal** connectivity matrix at *exactly 1*, whatever its dimension,
    and reaching `N` requires a non-normal operator with *"a hidden feedforward
    architecture"* whose input connectivity *"must optimally match"* it. `K` is
    `a.I` at construction, which is normal -- so the cell begins with one tap,
    not twelve, and any memory it acquires it must acquire by becoming
    non-normal. Normalised by `||K||_F^2` so the band's rescaling
    (:meth:`~patchworks.body.CellOperators.project` multiplies `K` by a scalar,
    which scales the commutator by that scalar squared) cannot move it: the
    statistic reads shape, never size.
    """
    with torch.no_grad():
        K = operators.K.double()
        Kt = K.transpose(-1, -2)
        commutator = torch.bmm(Kt, K) - torch.bmm(K, Kt)
        num = torch.linalg.matrix_norm(commutator, ord="fro")
        den = torch.linalg.matrix_norm(K, ord="fro") ** 2
        return (num / den.clamp(min=1e-300)).cpu().numpy()


def memory_read(lam: np.ndarray) -> dict:
    """How the twelve modes are spent, from one `[cells, k]` modulus block."""
    tau = np.where(lam > 1e-12, -1.0 / np.log(np.clip(lam, 1e-300, 1 - 1e-12)), 0.0)
    tau = np.where(lam >= 1 - 1e-12, np.inf, tau)
    finite = np.where(np.isfinite(tau), tau, 0.0)
    return {
        "rho_K": summarise(lam[:, 0]),
        "modulus_min": summarise(lam[:, -1]),
        "mean_modulus_spectrum": [float(x) for x in lam.mean(axis=0)],
        # A mode is "retaining" if it holds a direction longer than the one-tick
        # relaxation step ADR-0002 says a tick is: tau_i >= 1, i.e. |lambda| >= e^-1.
        "modes_retaining_per_cell": summarise((lam >= np.exp(-1.0)).sum(axis=1)),
        # Total held time across the twelve modes: the memory budget, spent.
        "summed_tau_per_cell": summarise(finite.sum(axis=1)),
        "max_tau_per_cell": summarise(finite.max(axis=1)),
        "cells_with_a_marginal_mode": int((lam >= 1 - 1e-9).sum(axis=1).astype(bool).sum()),
    }


def statistics(sigma: np.ndarray) -> dict:
    """The three counts, per cell, from one `[cells, k]` spectrum block."""
    k = sigma.shape[1]
    smax = sigma[:, 0]
    stable = (sigma**2).sum(axis=1) / np.maximum(smax**2, 1e-300)

    # Roy & Vetterli: exp of the entropy of the spectrum read as a distribution.
    total = np.maximum(sigma.sum(axis=1), 1e-300)
    p = sigma / total[:, None]
    entropy = -(p * np.log(np.maximum(p, 1e-300))).sum(axis=1)
    effective = np.exp(entropy)

    # The pre-registered read, at torch's own default tolerance.
    tol = smax * max(sigma.shape[-2:]) * np.finfo(np.float64).eps
    numerical = (sigma > tol[:, None]).sum(axis=1)

    return {
        "stable_rank": stable,
        "effective_rank": effective,
        "numerical_rank": numerical,
        "sigma_max": smax,
        "sigma_min": sigma[:, -1],
        "condition": smax / np.maximum(sigma[:, -1], 1e-300),
        "k": k,
    }


def summarise(values: np.ndarray) -> dict:
    v = np.asarray(values, dtype=float)
    return {
        "min": float(v.min()),
        "p25": float(np.percentile(v, 25)),
        "median": float(np.median(v)),
        "p75": float(np.percentile(v, 75)),
        "max": float(v.max()),
        "mean": float(v.mean()),
    }


def read(
    sigma: np.ndarray,
    levels: list[int],
    p_v: list[int],
    lam: np.ndarray,
    nn: np.ndarray,
) -> dict:
    stats = statistics(sigma)
    k = stats["k"]
    out = {
        name: summarise(stats[name])
        for name in (
            "stable_rank",
            "effective_rank",
            "numerical_rank",
            "sigma_max",
            "sigma_min",
            "condition",
        )
    }
    out["cells_full_numerical_rank"] = int((stats["numerical_rank"] == k).sum())
    out["cells"] = int(sigma.shape[0])
    out["mean_normalised_spectrum"] = [
        float(x) for x in (sigma / np.maximum(sigma[:, :1], 1e-300)).mean(axis=0)
    ]

    lv = np.asarray(levels)
    out["by_level"] = []
    for level in sorted(set(levels)):
        m = lv == level
        out["by_level"].append(
            {
                "level": int(level),
                "cells": int(m.sum()),
                "stable_rank": float(np.median(stats["stable_rank"][m])),
                "effective_rank": float(np.median(stats["effective_rank"][m])),
                "sigma_min": float(np.median(stats["sigma_min"][m])),
            }
        )

    pv = np.asarray(p_v, dtype=float)
    if pv.std() > 0:
        out["corr_p_v_stable_rank"] = float(
            np.corrcoef(pv, stats["stable_rank"])[0, 1]
        )
    out["memory"] = memory_read(lam)
    out["nonnormality"] = summarise(nn)
    out["per_cell"] = {
        "stable_rank": [float(x) for x in stats["stable_rank"]],
        "effective_rank": [float(x) for x in stats["effective_rank"]],
        "sigma_min": [float(x) for x in stats["sigma_min"]],
        "rho_K": [float(x) for x in lam[:, 0]],
        "modes_retaining": [int(x) for x in (lam >= np.exp(-1.0)).sum(axis=1)],
        "nonnormality": [float(x) for x in nn],
    }
    return out


def cell_context(agent) -> tuple[list[int], list[int]]:
    """Per predicting cell: its dome level, and its private width `p_v`."""
    dome = agent.dome
    predicting = list(dome.predicting)
    levels = [int(dome.cells[c].level) for c in predicting]
    maps = agent.sheaf.maps
    n = maps.stalk_width
    widths = []
    for c in predicting:
        used = 0
        for e in dome.cells[c].edges:
            used += int(maps.edge_width(e)) if hasattr(maps, "edge_width") else 0
        widths.append(max(0, n - used))
    return levels, widths


def run_seed(name: str, split: str, seed: int, ticks: int) -> dict:
    started = time.time()
    _env, agent = build(name, split, seed)
    try:
        levels, p_v = cell_context(agent)
    except Exception:  # the context is a nicety, never the measurement
        levels = [0] * agent.sheaf.operators.cells
        p_v = [0] * agent.sheaf.operators.cells

    record = {
        "dome": name,
        "split": split,
        "seed": seed,
        "ticks": ticks,
        "cells": int(agent.sheaf.operators.cells),
        "k": int(agent.sheaf.operators.shape.k),
        "band": [1.0 / agent.sheaf.operators.rho_k, 1.0],
        "scale_at_construction": float(agent.sheaf.operators.scale),
        "levels": levels,
        "p_v": p_v,
        "checkpoints": [],
    }

    record["at_construction"] = read(
        spectra(agent.sheaf.operators),
        levels,
        p_v,
        moduli(agent.sheaf.operators),
        nonnormality(agent.sheaf.operators),
    )

    ladder = [c for c in CHECKPOINTS if c <= ticks]
    if ticks not in ladder:
        ladder.append(ticks)
    seen = 0
    for target in ladder:
        for _ in teaching(agent, target - seen, seed=seed + seen):
            pass
        seen = target
        entry = read(
            spectra(agent.sheaf.operators),
            levels,
            p_v,
            moduli(agent.sheaf.operators),
            nonnormality(agent.sheaf.operators),
        )
        entry["ticks"] = target
        record["checkpoints"].append(entry)
        print(
            f"  seed {seed} @ {target:>6} ticks: "
            f"stable {entry['stable_rank']['median']:.3f}  "
            f"effective {entry['effective_rank']['median']:.3f}  "
            f"numerical {entry['numerical_rank']['median']:.1f}  "
            f"rho(K) {entry['memory']['rho_K']['median']:.3f}  "
            f"retaining {entry['memory']['modes_retaining_per_cell']['median']:.1f}/12  "
            f"sum_tau {entry['memory']['summed_tau_per_cell']['median']:.2f}  "
            f"nonnormal {entry['nonnormality']['median']:.4f}",
            flush=True,
        )

    record["elapsed_minutes"] = (time.time() - started) / 60.0
    return record


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dome", default="real")
    parser.add_argument("--split", default="train")
    parser.add_argument("--ticks", type=int, default=2_000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    for seed in args.seeds:
        print(f"seed {seed}, {args.ticks} ticks, dome {args.dome}", flush=True)
        record = run_seed(args.dome, args.split, seed, args.ticks)
        out = args.out or str(
            Path(__file__).parent
            / f"166-{args.dome}-{args.split}-seed{seed}-{args.ticks}.json"
        )
        Path(out).write_text(json.dumps(record, indent=1))
        print(f"  wrote {out}", flush=True)


if __name__ == "__main__":
    main()
