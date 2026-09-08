"""T6 (#567 / B18): is a cell's own state one-dimensional, or is that an uncentred statistic?

[B16](#564)'s `TrafficRecorder` accumulates `moment[c] += outer(h, h)` over ticks
and never subtracts the mean, then reduces the eigenvalues by `(Σλ)²/Σλ²`. For a
trajectory `h_t = μ + δ_t` that statistic is `μμᵀ + Cov(δ)`, so wherever `|μ|²`
dominates `tr Cov` the reading tends to **1 whatever the rank of the variation**.
A cell holding a settled belief that varies richly around itself reads 1.00.

The distinction is guarded elsewhere in this rig — `T2/run.py`'s `edge_reads`
reports `pr_uncentred` *and* `pr_centred` — and was not guarded here.

This instrument records **both moments over the same window**, so the two
readings are directly comparable per cell, plus the diagnostic that says how much
of the uncentred number the mean was responsible for:

    mean_share = |μ|² / (|μ|² + tr Cov)

`mean_share → 1` means the uncentred reading was measuring the baseline.

Deliberately *cheap*: no principal-angle surface read, no composed-rank read.
B16 already has those; this ticket asks one question and pays for one answer.

Usage::

    PYTHONPATH=src python prototypes/cold-start/T6/b18_centred.py --arms reserve reserve_p16 --ticks 20000
"""

from __future__ import annotations

import argparse
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
_T0, _T2 = (_HERE.parent / n for n in ("T0", "T2"))
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
arms_mod = _load("t6_arms", _HERE / "arms.py")

import construction_grading as cg  # noqa: E402
from patchworks.learning import PredictionRule, TransportRule  # noqa: E402

#: B16's own ladder, so the windows line up with `564-drive-*.json` exactly.
CHECKPOINTS = [100, 300, 1_000, 3_000, 10_000, 20_000, 50_000, 100_000]


def effective_rank_eigs(w: np.ndarray) -> float:
    """`(Σλ)²/Σλ²` on eigenvalues of a covariance — B16's quantity, unchanged."""
    w = np.clip(w.astype(np.float64), 0.0, None)
    return float(w.sum() ** 2 / max(float((w**2).sum()), 1e-300))


def _incident(dome, edge_id: int, cell_id: int) -> bool:
    try:
        cg.side_of(dome, edge_id, cell_id)
    except Exception:
        return False
    return True


class CentredTrafficRecorder:
    """Per relay cell, first *and* second moments of its node stalk over a window.

    Same tensor, same mask, same window as B16's `TrafficRecorder` — the only
    difference is that the first moment is kept, so the centred covariance
    `Cov = M/N - μμᵀ` is available beside the uncentred `M/N`.
    """

    def __init__(self, agent, cells: list[int]) -> None:
        self.agent = agent
        self.cells = cells
        dome = agent.dome
        self.masks = {}
        for c in cells:
            eid = next(e for e in range(len(dome.edges)) if _incident(dome, e, c))
            self.masks[c] = np.asarray(dome.restriction_mask(eid, c)).astype(bool)
        self.reset()

    def reset(self) -> None:
        self.moment = {c: np.zeros((int(self.masks[c].sum()),) * 2) for c in self.cells}
        self.first = {c: np.zeros(int(self.masks[c].sum())) for c in self.cells}
        self.count = 0

    @torch.no_grad()
    def observe(self) -> None:
        sh = self.agent.sheaf
        for c in self.cells:
            h = np.asarray(sh.stalks[sh.layout.slice(c)].double())[self.masks[c]]
            self.moment[c] += np.outer(h, h)
            self.first[c] += h
        self.count += 1

    def read(self) -> dict:
        out = {}
        n = max(self.count, 1)
        for c in self.cells:
            m_unc = self.moment[c] / n
            mu = self.first[c] / n
            cov = m_unc - np.outer(mu, mu)
            # Symmetrise against float drift before eigh.
            cov = 0.5 * (cov + cov.T)
            w_unc = np.linalg.eigvalsh(m_unc)
            w_cen = np.linalg.eigvalsh(cov)
            mu_energy = float(mu @ mu)
            tr_cov = float(np.clip(w_cen, 0.0, None).sum())
            out[c] = {
                "cell": int(c),
                "k_v": int(self.masks[c].sum()),
                "er_uncentred": effective_rank_eigs(w_unc),
                "er_centred": effective_rank_eigs(w_cen),
                "mu_energy": mu_energy,
                "tr_cov": tr_cov,
                "mean_share": mu_energy / max(mu_energy + tr_cov, 1e-300),
            }
        return out


def _pooled(recorder: CentredTrafficRecorder) -> dict:
    """Rank of the whole population's traffic, which B16 never reported.

    Cells have different `k_v`, so pooling is done on the *normalised* covariance
    spectra rather than by concatenating stalks: the pooled reading is the
    effective rank of the direct sum, which is what `(Σλ)²/Σλ²` over all cells'
    eigenvalues computes.
    """
    n = max(recorder.count, 1)
    all_unc: list[float] = []
    all_cen: list[float] = []
    for c in recorder.cells:
        m_unc = recorder.moment[c] / n
        mu = recorder.first[c] / n
        cov = m_unc - np.outer(mu, mu)
        cov = 0.5 * (cov + cov.T)
        all_unc.extend(np.linalg.eigvalsh(m_unc).tolist())
        all_cen.extend(np.linalg.eigvalsh(cov).tolist())
    return {
        "er_uncentred": effective_rank_eigs(np.asarray(all_unc)),
        "er_centred": effective_rank_eigs(np.asarray(all_cen)),
    }


def _q(values) -> dict:
    a = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if a.size == 0:
        return {"median": float("nan")}
    return {
        "min": float(a.min()),
        "p10": float(np.quantile(a, 0.10)),
        "median": float(np.median(a)),
        "mean": float(a.mean()),
        "p90": float(np.quantile(a, 0.90)),
        "max": float(a.max()),
    }


def run_seed(arm: str, seed: int, ticks: int, out: Path) -> dict:
    started = time.time()
    inflight = out.with_suffix(".inflight.json")
    env, agent = arms_mod.build_arm(arm, seed)
    try:
        dome = agent.dome
        chains = t2.rim_chains(dome)
        triples: dict[int, tuple[int, int]] = {}
        for ch in chains:
            for edge_in, cell, edge_out in cg.hops_of(dome, tuple(ch["edges"])):
                triples.setdefault(int(cell), (int(edge_in), int(edge_out)))
        cells = sorted(triples)
        _, reserve_p = arms_mod.ARMS[arm]

        record = {
            "issue": 567,
            "arm": arm,
            "reserve_p": reserve_p,
            "seed": seed,
            "ticks": ticks,
            "n": int(dome.shape.n),
            "relay_cells": len(cells),
            "question": (
                "B16's traffic rank is an uncentred second moment; a settled "
                "baseline reads rank one whatever the rank of the variation. "
                "This records both moments over the same window."
            ),
            "checkpoints": [],
        }

        bias = PredictionRule(agent.sheaf)
        transport = TransportRule(agent.sheaf)
        rec = CentredTrafficRecorder(agent, cells)

        def snapshot(tick: int) -> dict:
            rows = list(rec.read().values())
            entry = {
                "ticks": tick,
                "frames": rec.count,
                "er_uncentred": _q([r["er_uncentred"] for r in rows]),
                "er_centred": _q([r["er_centred"] for r in rows]),
                "mean_share": _q([r["mean_share"] for r in rows]),
                "mu_energy": _q([r["mu_energy"] for r in rows]),
                "tr_cov": _q([r["tr_cov"] for r in rows]),
                "pooled": _pooled(rec),
                "cells": rows,
            }
            entry["elapsed_minutes"] = (time.time() - started) / 60.0
            return entry

        rec.observe()
        record["checkpoints"].append(snapshot(0))
        c0 = record["checkpoints"][0]
        print(
            f"  {arm} s{seed} construction: er_unc {c0['er_uncentred']['median']:.4f} "
            f"er_cen {c0['er_centred']['median']:.4f} (1 frame, degenerate by construction)",
            flush=True,
        )

        ladder = [c for c in CHECKPOINTS if c <= ticks]
        if ticks not in ladder:
            ladder.append(ticks)
        seen = 0
        for target in ladder:
            rec.reset()
            for _ in t0.run_ticks(agent, target - seen, seed=seed + seen):
                rec.observe()
                bias.step()
                if agent.sheaf.ticks > 1:
                    transport.step()
            seen = target
            entry = snapshot(target)
            record["checkpoints"].append(entry)
            inflight.write_text(json.dumps(record, indent=1))
            print(
                f"  {arm} s{seed} @{target:>6}: "
                f"er_unc {entry['er_uncentred']['median']:.4f} "
                f"| er_cen {entry['er_centred']['median']:.4f} "
                f"| mean_share {entry['mean_share']['median']:.4f} "
                f"| pooled_cen {entry['pooled']['er_centred']:.2f} "
                f"| {entry['elapsed_minutes']:.1f}m",
                flush=True,
            )

        record["minutes"] = (time.time() - started) / 60.0
        out.write_text(json.dumps(record, indent=1))
        if inflight.exists():
            inflight.unlink()
        return record
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", nargs="+", default=["reserve", "reserve_p16"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ticks", type=int, default=20_000)
    args = ap.parse_args()

    for arm in args.arms:
        out = _HERE / f"567-centred-{arm}-seed{args.seed}-{args.ticks}.json"
        if out.exists():
            print(f"skip {arm}: {out.name} exists", flush=True)
            continue
        print(f"== {arm} seed {args.seed} to {args.ticks} ==", flush=True)
        run_seed(arm, args.seed, args.ticks, out)


if __name__ == "__main__":
    main()
