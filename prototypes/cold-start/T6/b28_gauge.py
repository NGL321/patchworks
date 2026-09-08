"""T6 (#577 / B28): is "the learned subspaces look random" gauge, overfitting, or real?

[B24](#573)'s literature pass flagged the one caveat that could overturn its own
central finding: **a sheaf has an obvious gauge freedom**, and *"the learned maps
look random"* and *"the learned maps are determined only up to a gauge nobody
fixed"* are different claims. [B1](#537) rung (d) and [B11](#555) are the two
readings at issue, and [B17](#565) rests on them.

**The gauge group of this construction.** A restriction map `F_{v<e}` is stored on
the block `[:m_e, :k_v]` — `m_e` live rows in the edge stalk, and the leading
`k_v` columns of the cell's node stalk, the mask prefix every one of a cell's
incident maps shares (`RestrictionMaps.support`, `column_mask`). The admissible
transformations are therefore

    per predicting cell `v`:  `O_v` in `O(k_v)` on the exposed block
                              (the private trailing block is a separate summand)
    per edge `e`:             `Q_e` in `O(m_e)` on the edge stalk

acting as `F_{v<e} -> Q_e F_{v<e} O_v^T`, with the cell states carried along as
`h_v -> O_v h_v`. Every constraint the build imposes is equivariant under it:

* the **support** is the rectangular block `[:m_e, :k_v]`, which `O_v` and `Q_e`
  preserve exactly;
* the **band** (`gauge_bounds`, `norms`) is Frobenius, orthogonally invariant;
* **ADR-0032's floor** (`_flatten`) is a function of the singular values alone;
* **ADR-0010's cap** (`_push_apart`, `gram_peaks`) reads
  `lambda_max(sum_e F^T F) -> lambda_max(O (sum_e F^T F) O^T)`, unchanged;
* the **transport objective** is `||F_u h_u - F_v h_v||^2`, and `Q_e` is shared by
  the edge's two ends, so it cancels.

`O_v` is **not** free at a cell whose whole incidence is pinned (`pinned_incidence`
— the exact gauge fixes those maps outright) nor at a boundary cell (the world
writes its stalk in fixed coordinates). Neither is a relay cell on a rim-to-apex
chain, so the group above is the one the composed operator sees.

**The consequence, which is why item 3 is not a search.** A chain's hop is
`F_out F_in^T` with both maps on the *same* relay cell, so

    `Q_out F_out O_v^T · O_v F_in^T Q_in^T = Q_out (F_out F_in^T) Q_in^T`

— `O_v` cancels inside every hop, and along a chain each interior `Q_e` meets its
own transpose. The composed operator transforms by `Q_last (·) Q_first^T`, an
orthogonal equivalence, so **its singular values, and therefore composed effective
rank, are exact gauge invariants**. So are the principal angles between a relay
cell's two carried subspaces. No change of basis can move either, which is what
:func:`orbit_read` measures rather than assumes.

**What is left to test, then**, is structure against a reference frame that is
*physical* rather than coordinate — one that transforms covariantly with the cell.
:func:`traffic_read` uses the cell's own centred state covariance ([B18](#567)'s
correction applies: uncentred is the wrong moment), and asks whether a lane carries
more of its cell's traffic variance than a Haar frame of the same shape would.
:func:`sibling_read` asks the gauge-invariant version of "shared leading directions
across a cell's edges".

Usage::

    PYTHONPATH=src python prototypes/cold-start/T6/b28_gauge.py --arm reserve_p16 --ticks 20000
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
_T0, _T2, _T4 = (_HERE.parent / n for n in ("T0", "T2", "T4"))
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
angles = _load("t4_angles", _T4 / "angles.py")
arms_mod = _load("t6_arms", _HERE / "arms.py")

from patchworks.graph import EdgeKind  # noqa: E402
from patchworks.learning import PredictionRule, TransportRule  # noqa: E402
from patchworks.restriction import (  # noqa: E402
    map_is_pinned,
    pair_index,
    pinned_incidence,
)

#: [B19](#568)/[B18](#567)'s window, so the traffic moment is the one those read.
WINDOW = 1_000

#: [B16](#564)/[B18](#567)'s ladder.
CHECKPOINTS = [100, 300, 1_000, 3_000, 10_000, 20_000, 50_000, 100_000]

#: Haar draws per (m, k) shape for the null in :func:`traffic_read`.
NULL_DRAWS = 400


def _q(a: np.ndarray) -> dict:
    a = np.asarray(a, dtype=np.float64)
    return {
        "n": int(a.size),
        "median": float(np.median(a)),
        "mean": float(a.mean()),
        "p05": float(np.quantile(a, 0.05)),
        "p95": float(np.quantile(a, 0.95)),
        "min": float(a.min()),
        "max": float(a.max()),
    }


# -- the gauge itself ---------------------------------------------------------


def cell_widths(dome) -> dict[int, int]:
    """`k_v` per cell, asserting every incident map shares one mask prefix.

    `angles.py` states this and `RestrictionMaps` builds on it (`column_mask`,
    `hold_width`); the gauge action below is only well defined if it holds, so
    it is checked rather than trusted.
    """
    widths: dict[int, int] = {}
    for cell in dome.cells:
        ks = {
            int(dome.restriction_mask(eid, cell.id).sum())
            for eid in dome.incident[cell.id]
        }
        if len(ks) != 1:
            raise AssertionError(
                f"cell {cell.id} does not share one mask across its incidence: {sorted(ks)}"
            )
        widths[cell.id] = ks.pop()
    return widths


def haar(rng: np.random.Generator, d: int) -> np.ndarray:
    """A Haar-distributed element of `O(d)` (QR with the sign fix)."""
    if d == 0:
        return np.zeros((0, 0))
    q, r = np.linalg.qr(rng.standard_normal((d, d)))
    return q * np.sign(np.diag(r))


def draw_gauge(dome, rng: np.random.Generator, *, free_only: bool = True) -> dict:
    """One element of the gauge group: `O_v` per cell and `Q_e` per edge.

    With ``free_only`` the identity is used at cells whose whole incidence is
    pinned and at boundary cells — the two places the gauge is genuinely fixed
    (see this module's docstring). The composed-operator claim does not depend on
    the choice, and ``free_only=False`` is available to show that.
    """
    widths = cell_widths(dome)
    pinned_cells = pinned_incidence(dome)
    o = {}
    for cell in dome.cells:
        fixed = free_only and (cell.is_boundary or pinned_cells[cell.id])
        k = widths[cell.id]
        o[cell.id] = np.eye(k) if fixed else haar(rng, k)
    q = {}
    for edge in dome.edges:
        fixed = free_only and all(
            map_is_pinned(dome, edge.id, c) for c in (edge.u, edge.v)
        )
        q[edge.id] = np.eye(edge.m) if fixed else haar(rng, edge.m)
    return {"O": o, "Q": q, "widths": widths}


@torch.no_grad()
def apply_gauge(agent, gauge: dict) -> None:
    """`F_{v<e} <- Q_e F_{v<e} O_v^T`, in place on the live block of every map."""
    dome, maps = agent.dome, agent.sheaf.maps
    for edge in dome.edges:
        qe = torch.from_numpy(gauge["Q"][edge.id]).to(maps.maps.dtype)
        for cell_id in (edge.u, edge.v):
            k = gauge["widths"][cell_id]
            ov = torch.from_numpy(gauge["O"][cell_id]).to(maps.maps.dtype)
            row = pair_index(edge.id, 0 if edge.u == cell_id else 1)
            block = maps.maps[row, : edge.m, :k]
            maps.maps[row, : edge.m, :k] = qe @ block @ ov.T


def _surface_stats(agent, chains: list[dict]) -> dict:
    """The statistics every ruling on #532 is stated in, from one surface."""
    read = angles.read_surface(agent, chains, "orbit")
    maps = agent.sheaf.maps
    return {
        "composed_er": read["composed_er"],
        "cos_all": read["cos_all"],
        "cos_leading_per_hop": read["cos_leading_per_hop"],
        "cos_second_per_hop": read["cos_second_per_hop"],
        "s2_over_s1": read["s2_over_s1"],
        "sigma_min_over_max": read["sigma_min_over_max"],
        "gram_peak_max": float(maps.gram_peaks().max()),
        "norm_median": float(maps.norms().detach().median()),
    }


def _worst(a: dict, b: dict) -> float:
    """Largest absolute difference between two nested stat dicts."""
    worst = 0.0
    for key, va in a.items():
        vb = b[key]
        if isinstance(va, dict):
            worst = max(worst, _worst(va, vb))
        elif isinstance(va, (int, float)):
            worst = max(worst, abs(float(va) - float(vb)))
    return worst


@torch.no_grad()
def objective_read(agent, gauge: dict | None, probes: np.ndarray) -> dict:
    """Edge Dirichlet energy on fixed probe states, carried through the gauge.

    The transport objective is `||F_u h_u - F_v h_v||^2`. Under the gauge the
    states go with the maps (`h_v -> O_v h_v`), so this evaluates the *same*
    physical configuration in the new basis. It is the check that the group above
    really is a symmetry of the objective and not merely of the maps.
    """
    dome, maps = agent.dome, agent.sheaf.maps
    widths = cell_widths(dome)
    energies = []
    for edge in dome.edges:
        ends = []
        for cell_id in (edge.u, edge.v):
            k = widths[cell_id]
            h = probes[cell_id, :k]
            if gauge is not None:
                h = gauge["O"][cell_id] @ h
            row = pair_index(edge.id, 0 if edge.u == cell_id else 1)
            f = maps.maps[row, : edge.m, :k].double().numpy()
            ends.append(f @ h)
        energies.append(float(np.sum((ends[0] - ends[1]) ** 2)))
    e = np.array(energies)
    return {"total": float(e.sum()), "median": float(np.median(e)), "max": float(e.max())}


@torch.no_grad()
def orbit_read(agent, chains: list[dict], seed: int = 0) -> dict:
    """Items 1-2: move along the gauge orbit and re-read everything the map rests on.

    Restores the surface exactly on the way out, so this is safe to call at any
    checkpoint of a live run.
    """
    dome = agent.dome
    maps = agent.sheaf.maps
    rng = np.random.default_rng(seed)
    keep = maps.maps.detach().clone()
    n_cells = len(dome.cells)
    probes = np.random.default_rng(seed + 991).standard_normal(
        (n_cells, int(maps.stalk_width))
    )

    try:
        before = _surface_stats(agent, chains)
        e_before = objective_read(agent, None, probes)
        rows = []
        for trial in range(3):
            maps.maps.copy_(keep)
            gauge = draw_gauge(dome, rng, free_only=trial < 2)
            apply_gauge(agent, gauge)
            after = _surface_stats(agent, chains)
            e_after = objective_read(agent, gauge, probes)
            rows.append(
                {
                    "trial": trial,
                    "free_only": trial < 2,
                    "worst_abs_delta": _worst(before, after),
                    "composed_er_median_after": after["composed_er"]["median"],
                    "composed_er_max_after": after["composed_er"]["max"],
                    "cos_all_median_after": after["cos_all"]["median"],
                    "gram_peak_max_after": after["gram_peak_max"],
                    "energy_total_after": e_after["total"],
                    "energy_rel_delta": abs(e_after["total"] - e_before["total"])
                    / max(e_before["total"], 1e-300),
                }
            )
    finally:
        maps.maps.copy_(keep)

    return {
        "before": before,
        "energy_before": e_before,
        "trials": rows,
        "worst_abs_delta": max(r["worst_abs_delta"] for r in rows),
        "worst_energy_rel_delta": max(r["energy_rel_delta"] for r in rows),
    }


# -- item 3: structure against a physical frame -------------------------------


class Motion:
    """Std of the boundary stalks over a window — was the world moving?

    Filed against the standing advisory on [#517](#517) that `PlanarPushSandbox`
    motion collapses between 1,000 and 2,000 ticks on the shipped spec and stays
    collapsed (#572's `world_std_*`, and #518's own `travel_window` records the
    same shape). Every checkpoint past ~2,000 ticks on this map is therefore
    taken against a world with no exogenous variation, so a reading that leans on
    a *live* world has to say whether the world was live rather than assume it.

    The boundary cells are the ones the world writes each tick, so their spread
    over a window is the most direct statement of it this rig can make.
    """

    def __init__(self, dome, layout) -> None:
        rows = []
        for cell in dome.cells:
            if cell.is_boundary:
                sl = layout.slice(cell.id)
                rows.append(np.arange(sl.start, sl.stop))
        self.index = torch.from_numpy(np.concatenate(rows)) if rows else None
        self.reset()

    def reset(self) -> None:
        self.n = 0
        self.sum = None
        self.sq = None

    @torch.no_grad()
    def observe(self, sheaf) -> None:
        if self.index is None:
            return
        h = sheaf.stalks[self.index].double().numpy()
        if self.sum is None:
            self.sum = np.zeros_like(h)
            self.sq = np.zeros_like(h)
        self.n += 1
        self.sum += h
        self.sq += h * h

    def read(self) -> dict:
        if self.index is None or self.n < 2:
            return {"ticks": self.n}
        mu = self.sum / self.n
        var = np.maximum(self.sq / self.n - mu * mu, 0.0)
        sd = np.sqrt(var)
        return {
            "ticks": int(self.n),
            "components": int(sd.size),
            "std_median": float(np.median(sd)),
            "std_p90": float(np.quantile(sd, 0.90)),
            "std_max": float(sd.max()),
            "frac_components_moving": float((sd > 1e-3).mean()),
        }


class Traffic:
    """Per-cell centred second moment of the node stalk over a window.

    Accumulated on the leading `k_v` block, which is the space the carried
    subspaces live in. [B18](#567) is why the moment is centred: the uncentred
    participation ratio of a settled belief reads ~1 whatever it is doing.
    """

    def __init__(self, dome, layout) -> None:
        self.cells = list(dome.predicting)
        self.n = int(dome.shape.n)
        self.count = 0
        self.sum = np.zeros((len(self.cells), self.n))
        self.outer = np.zeros((len(self.cells), self.n, self.n))
        # `sheaf.stalks` is one ragged flat vector -- a boundary patch cell holds
        # 48 components where a predicting cell holds `n` -- so the cells are
        # gathered by `StalkLayout.slice`, the same route `Sheaf.stalk` takes.
        rows = []
        for cell_id in self.cells:
            sl = layout.slice(cell_id)
            rows.append(np.arange(sl.start, sl.start + self.n))
        self.index = torch.from_numpy(np.stack(rows))

    @torch.no_grad()
    def observe(self, sheaf) -> None:
        h = sheaf.stalks[self.index].double().numpy()
        self.count += 1
        self.sum += h
        self.outer += h[:, :, None] * h[:, None, :]

    def reset(self) -> None:
        self.count = 0
        self.sum[:] = 0.0
        self.outer[:] = 0.0

    def cov(self, row: int) -> np.ndarray:
        mu = self.sum[row] / max(self.count, 1)
        return self.outer[row] / max(self.count, 1) - mu[:, None] * mu[None, :]

    def mean_share(self, row: int) -> float:
        mu = self.sum[row] / max(self.count, 1)
        c = self.cov(row)
        return float(mu @ mu / max(mu @ mu + np.trace(c), 1e-300))


def _null_capture(rng, cov_k: np.ndarray, m: int, draws: int) -> np.ndarray:
    """Capture fraction of `draws` Haar `m`-frames in the same `k`-space."""
    k = cov_k.shape[0]
    total = max(float(np.trace(cov_k)), 1e-300)
    out = np.empty(draws)
    for i in range(draws):
        v = np.linalg.qr(rng.standard_normal((k, m)))[0]
        out[i] = float(np.trace(v.T @ cov_k @ v)) / total
    return out


@torch.no_grad()
def traffic_read(agent, traffic: Traffic, seed: int = 0, draws: int = NULL_DRAWS) -> dict:
    """Item 3: does a lane carry more of its cell's traffic than a Haar frame would?

    Per (cell, interior edge): `capture = tr(V^T C V) / tr(C)` with `V` the
    carried subspace and `C` the cell's centred covariance on its mask block. Both
    `V` and `C` transform as `O_v (·) O_v^T`, so ``capture`` is **gauge invariant**
    — which is exactly what makes it a test and not a coordinate reading.

    Reported as a percentile against a Haar null drawn in the same `k`-space
    against the same `C`, so the null carries `C`'s own anisotropy. ``ceiling`` is
    the top-`m` eigenspace's capture, the most any `m`-frame could take.
    """
    dome, maps = agent.dome, agent.sheaf.maps
    rng = np.random.default_rng(seed)
    widths = cell_widths(dome)
    rows = []
    for row, cell_id in enumerate(traffic.cells):
        k = widths[cell_id]
        cov = traffic.cov(row)[:k, :k]
        total = float(np.trace(cov))
        if total <= 1e-18 or k < 2:
            continue
        evals = np.linalg.eigvalsh(cov)[::-1]
        null_cache: dict[int, np.ndarray] = {}
        for eid in dome.incident[cell_id]:
            edge = dome.edges[eid]
            if edge.kind is not EdgeKind.INTERIOR:
                continue
            m = min(edge.m, k)
            f = maps.maps[
                pair_index(eid, 0 if edge.u == cell_id else 1), : edge.m, :k
            ].double().numpy()
            v = np.linalg.svd(f, full_matrices=False)[2][:m].T  # k x m, orthonormal
            capture = float(np.trace(v.T @ cov @ v)) / total
            if m not in null_cache:
                null_cache[m] = _null_capture(rng, cov, m, draws)
            null = null_cache[m]
            rows.append(
                {
                    "cell": int(cell_id),
                    "edge": int(eid),
                    "k_v": int(k),
                    "m": int(m),
                    "degenerate": bool(m >= k),
                    "capture": capture,
                    "null_mean": float(null.mean()),
                    "null_median": float(np.median(null)),
                    "null_sd": float(null.std()),
                    "percentile": float((null < capture).mean()),
                    "z": float((capture - null.mean()) / max(null.std(), 1e-12)),
                    "ceiling": float(evals[:m].sum() / total),
                    "excess_over_null": capture - float(null.mean()),
                }
            )
    if not rows:
        return {"lanes": 0}

    def agg(subset: list[dict]) -> dict:
        if not subset:
            return {"lanes": 0}
        pct = np.array([r["percentile"] for r in subset])
        return {
            "lanes": len(subset),
            "cells": len({r["cell"] for r in subset}),
            "m": _q(np.array([r["m"] for r in subset])),
            "k_v": _q(np.array([r["k_v"] for r in subset])),
            "capture": _q(np.array([r["capture"] for r in subset])),
            "null_mean": _q(np.array([r["null_mean"] for r in subset])),
            "null_median": _q(np.array([r["null_median"] for r in subset])),
            "ceiling": _q(np.array([r["ceiling"] for r in subset])),
            "percentile": _q(pct),
            "z": _q(np.array([r["z"] for r in subset])),
            "frac_above_p95": float((pct > 0.95).mean()),
            "frac_below_p05": float((pct < 0.05).mean()),
        }

    # A lane with `m >= k_v` spans the whole exposed block: its capture is 1 and
    # so is every draw of the null, so it can carry no evidence either way.
    # [B13](#560) is why they exist at all on the `p = 16` arm -- the floor
    # climbs to a degenerate ceiling "where lanes stop being different lanes".
    live = [r for r in rows if not r["degenerate"]]
    return {
        "all": agg(rows),
        "non_degenerate": agg(live),
        "degenerate_lanes": len(rows) - len(live),
        "null_draws": draws,
        "mean_share": _q(np.array([traffic.mean_share(r) for r in range(len(traffic.cells))])),
        "traffic_window_ticks": traffic.count,
    }


@torch.no_grad()
def sibling_read(agent, seed: int = 0, draws: int = NULL_DRAWS) -> dict:
    """Item 3's other half, gauge-invariantly: do a cell's lanes share directions?

    All ordered pairs of interior lanes at a cell, their principal-angle cosines
    against a Haar null at the same `(m_a, m_b, k)`. This is the object
    [B1](#537) §1 reads per *hop*; here it is read per *cell over its whole
    incidence*, which is the "shared leading directions across a cell's edges"
    item 3 names.
    """
    dome, maps = agent.dome, agent.sheaf.maps
    rng = np.random.default_rng(seed + 7)
    widths = cell_widths(dome)
    lead, mean_cos, degenerate = [], [], []
    null_lead: dict[tuple, np.ndarray] = {}
    z_lead = []
    for cell_id in dome.predicting:
        k = widths[cell_id]
        frames = []
        for eid in dome.incident[cell_id]:
            edge = dome.edges[eid]
            if edge.kind is not EdgeKind.INTERIOR:
                continue
            m = min(edge.m, k)
            f = maps.maps[
                pair_index(eid, 0 if edge.u == cell_id else 1), : edge.m, :k
            ].double().numpy()
            frames.append((m, np.linalg.svd(f, full_matrices=False)[2][:m].T))
        for i in range(len(frames)):
            for j in range(i + 1, len(frames)):
                (ma, a), (mb, b) = frames[i], frames[j]
                cos = np.linalg.svd(a.T @ b, compute_uv=False)
                lead.append(float(cos[0]))
                mean_cos.append(float(cos.mean()))
                degenerate.append(bool(ma >= k or mb >= k))
                key = (k, ma, mb)
                if key not in null_lead:
                    vals = np.empty(draws)
                    for t in range(draws):
                        qa = np.linalg.qr(rng.standard_normal((k, ma)))[0]
                        qb = np.linalg.qr(rng.standard_normal((k, mb)))[0]
                        vals[t] = np.linalg.svd(qa.T @ qb, compute_uv=False)[0]
                    null_lead[key] = vals
                nl = null_lead[key]
                z_lead.append(float((cos[0] - nl.mean()) / max(nl.std(), 1e-12)))
    if not lead:
        return {"pairs": 0}
    lead = np.array(lead)
    mean_cos = np.array(mean_cos)
    z_lead = np.array(z_lead)
    live = ~np.array(degenerate)
    out = {
        "pairs": int(lead.size),
        "leading_cos": _q(lead),
        "mean_cos": _q(mean_cos),
        "leading_cos_z_vs_haar": _q(z_lead),
        "degenerate_pairs": int((~live).sum()),
        "null_draws": draws,
    }
    if live.any():
        out["non_degenerate"] = {
            "pairs": int(live.sum()),
            "leading_cos": _q(lead[live]),
            "mean_cos": _q(mean_cos[live]),
            "leading_cos_z_vs_haar": _q(z_lead[live]),
        }
    return out



# -- item 4: does the agreement generalise? -----------------------------------

#: The evaluation ladder, in order of increasing distance from what was trained on.
#:
#: ``train_seen`` replays the **exact arrangements the run trained on** — the
#: ladder's legs reset the world once each at `seed + ticks_seen`, so those seeds
#: are recoverable and are what "in sample" means here. ``train`` is the same
#: slice at arrangements never seen. The two held-out slices are
#: `PlanarPushSandbox`'s own; there is deliberately no union split
#: (`sandbox/env.py:118`), so the two axes are read separately.
#:
#: Separating the first two rungs is what distinguishes *memorising these worlds*
#: from *fitting this slice*, and only the second is what [B25](#574) predicts.
EVAL_SPLITS = ("train_seen", "train", "heldout_pair", "heldout_sector")

#: Which sandbox split each rung is evaluated on.
SPLIT_OF = {
    "train_seen": "train",
    "train": "train",
    "heldout_pair": "heldout_pair",
    "heldout_sector": "heldout_sector",
}


@torch.no_grad()
def edge_moments(agent) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """This tick's two edge-end vectors and the three squared norms built from them.

    The numerator is :meth:`~patchworks.tick.Sheaf.disagreement` verbatim
    (`ends[:, 0] - ends[:, 1]`, in its own sign convention).
    """
    sheaf = agent.sheaf
    ends = sheaf.broadcast.reshape(-1, 2, sheaf.maps.edge_width).double()
    a, b = ends[:, 0].numpy(), ends[:, 1].numpy()
    d = a - b
    return a, b, (a * a).sum(-1), (b * b).sum(-1), (d * d).sum(-1)


class EdgeAgreement:
    """Per-edge relative disagreement over a window, uncentred **and centred**.

        `rel = ||F_u h_u - F_v h_v||^2 / (||F_u h_u||^2 + ||F_v h_v||^2)`

    Scale-free, so it cannot be moved by the band, the gain, or one slice of the
    world being livelier than another; and gauge-invariant, since `Q_e` is shared
    by an edge's two ends and cancels in every term. Two independent ends of equal
    norm read **1**; perfect agreement reads **0**.

    **The centred form is the one to read, and [B18](#567) is why.** The mean
    carries 99.6-99.9% of a cell's stalk energy past the first window, so an
    uncentred reading mostly reports that both ends of an edge carry the same
    large standing component — agreement on the DC term, which says nothing about
    whether the lane transports what varies. Both are reported so the size of that
    artifact is visible rather than assumed, as [B18](#567)'s ledger row asks.
    """

    def __init__(self, edges: int, width: int) -> None:
        self.n = 0
        self.sum_a = np.zeros((edges, width))
        self.sum_b = np.zeros((edges, width))
        self.sq_a = np.zeros(edges)
        self.sq_b = np.zeros(edges)
        self.sq_d = np.zeros(edges)

    def observe(self, agent) -> None:
        a, b, sa, sb, sd = edge_moments(agent)
        self.n += 1
        self.sum_a += a
        self.sum_b += b
        self.sq_a += sa
        self.sq_b += sb
        self.sq_d += sd

    def read(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """`(uncentred, centred, variance)` per edge; `variance` is the denominator."""
        n = max(self.n, 1)
        mu_a, mu_b = self.sum_a / n, self.sum_b / n
        ea, eb, ed = self.sq_a / n, self.sq_b / n, self.sq_d / n
        var_a = ea - (mu_a * mu_a).sum(-1)
        var_b = eb - (mu_b * mu_b).sum(-1)
        mu_d = mu_a - mu_b
        var_d = ed - (mu_d * mu_d).sum(-1)
        denom = ea + eb
        cdenom = var_a + var_b
        return (
            np.where(denom > 1e-24, ed / np.maximum(denom, 1e-24), np.nan),
            np.where(cdenom > 1e-18, var_d / np.maximum(cdenom, 1e-18), np.nan),
            cdenom,
        )


def _interior_edges(dome) -> np.ndarray:
    return np.array(
        [e.kind is EdgeKind.INTERIOR for e in dome.edges], dtype=bool
    )


#: Where an evaluation run is split into its two windows. The world's exogenous
#: variation collapses within a few hundred ticks of a reset (see :class:`Motion`
#: and #572's advisory), so a single late window would measure agreement against
#: a world that had stopped moving. ``early`` is ticks ``SETTLE..EVAL_SPLIT_AT``,
#: while the boundary cells are demonstrably still moving; ``late`` is
#: ``EVAL_SPLIT_AT..ticks``, the regime every other trained reading on this map
#: is taken in. Both are reported, each with its own motion stamp, so the answer
#: does not rest on a choice of window.
EVAL_SPLIT_AT = 150
#: Ticks dropped immediately after a rearrangement, before the early window opens.
SETTLE = 20


@torch.no_grad()
def evaluate_split(agent, env, split_seeds, ticks, settle=SETTLE, split_at=EVAL_SPLIT_AT) -> dict:
    """Run the frozen surface on one slice and read what it agrees on.

    No rule is stepped: the maps are held exactly as training left them and only
    the tick runs, so this measures the agreement *these* maps achieve on *this*
    world rather than how fast they could adapt to it. Each seed is one
    arrangement of the world (`PlanarPushSandbox` has no episode boundary, so a
    reset is a rearrangement), and each arrangement gets its own windows, so the
    centred moment is taken **within** an arrangement rather than across the set
    of them.
    """
    interior = _interior_edges(agent.dome)
    width = int(agent.sheaf.maps.edge_width)
    keep_env, agent.env = agent.env, env
    bins = {"early": (settle, split_at), "late": (split_at, ticks)}
    got = {k: {"unc": [], "cen": [], "var": [], "motion": []} for k in bins}
    try:
        for seed in split_seeds:
            acc = {k: EdgeAgreement(len(agent.dome.edges), width) for k in bins}
            mot = {k: Motion(agent.dome, agent.sheaf.layout) for k in bins}
            for i, _ in enumerate(t0.run_ticks(agent, ticks, seed=seed)):
                for k, (lo, hi) in bins.items():
                    if lo <= i < hi:
                        acc[k].observe(agent)
                        mot[k].observe(agent.sheaf)
            for k in bins:
                u, c, v = acc[k].read()
                got[k]["unc"].append(u)
                got[k]["cen"].append(c)
                got[k]["var"].append(v)
                got[k]["motion"].append(mot[k].read())
    finally:
        agent.env = keep_env

    def summarise(g: dict, window: tuple) -> dict:
        unc, cen, var = np.stack(g["unc"]), np.stack(g["cen"]), np.stack(g["var"])
        # An edge whose two ends barely move carries no evidence about transport
        # of what varies, and its centred ratio is a ratio of two numerical zeros.
        alive = interior[None, :] & np.isfinite(cen) & (var > 1e-12)
        per_seed = np.array(
            [
                float(np.nanmean(cen[i][alive[i]])) if alive[i].any() else float("nan")
                for i in range(len(split_seeds))
            ]
        )
        flat_c = cen[alive]
        flat_u = unc[interior[None, :] & np.isfinite(unc)]
        return {
            "window_ticks": list(window),
            "edges_live_median": float(np.median(alive.sum(axis=1))),
            "edges_interior": int(interior.sum()),
            "rel_centred_mean": float(np.nanmean(flat_c)),
            "rel_centred_median": float(np.nanmedian(flat_c)),
            "rel_centred_p90": float(np.nanquantile(flat_c, 0.90)),
            "rel_uncentred_mean": float(np.nanmean(flat_u)),
            "rel_uncentred_median": float(np.nanmedian(flat_u)),
            "across_seed_sd": float(np.nanstd(per_seed)),
            "per_seed_centred_mean": [float(x) for x in per_seed],
            "world_motion_std_max_median": float(
                np.median([m.get("std_max", float("nan")) for m in g["motion"]])
            ),
            "world_motion_frac_moving_median": float(
                np.median(
                    [m.get("frac_components_moving", float("nan")) for m in g["motion"]]
                )
            ),
            "world_motion_per_seed": g["motion"],
        }

    return {
        "seeds": [int(x) for x in split_seeds],
        "ticks_per_seed": ticks,
        "windows": {k: summarise(got[k], bins[k]) for k in bins},
    }


def training_world_seeds(seed: int, ticks: int) -> list[int]:
    """The world seeds :func:`main`'s ladder actually reset on.

    `teaching_read` is handed `seed + seen` at the start of each leg and
    `run_ticks` resets the world with it, so the arrangements the surface was
    trained on are exactly these — which is what makes a true in-sample rung
    available without recording anything during training.
    """
    ladder = [c for c in CHECKPOINTS if c <= ticks]
    if ticks not in ladder:
        ladder.append(ticks)
    seen, out = 0, []
    for target in ladder:
        out.append(seed + seen)
        seen = target
    return out


def generalise_read(arm, seed, trained_agent, trained_ticks, ticks=400, n_seeds=6) -> dict:
    """[B25](#574)'s discriminator, which is cheaper than items 1-3 and decides more.

    Three readings on the same worlds: the **trained** surface, and an
    **untrained** one built from the same spec and seed as the control that says
    what "no agreement at all" costs on *these* arrangements. If the trained
    surface's advantage over the control is large on `train` and gone on the
    held-out slices, the lanes are fitted to the training slice rather than
    generic — which is a different diagnosis from either horn of the gauge
    question, and the one [B25](#574) supplied.

    The control is necessary because the slices are not equally hard: a bare
    train-versus-heldout gap on the trained surface alone confounds overfitting
    with the world being different. What is read is the **ratio**.
    """
    from patchworks.sandbox import PlanarPushSandbox
    from untrained_fixed_point import IMAGE_SIZE, dome_named

    base, _ = dome_named("real")
    seen_seeds = training_world_seeds(seed, trained_ticks)
    fresh_seeds = [seed + 7_000_000 + 101 * i for i in range(n_seeds)]
    control_env, control = arms_mod.build_arm(arm, seed)
    out = {
        "eval_ticks_per_seed": ticks,
        "settle": SETTLE,
        "split_at": EVAL_SPLIT_AT,
        "n_world_seeds": n_seeds,
        "seen_world_seeds": seen_seeds,
        "unseen_world_seeds": fresh_seeds,
        "splits": {},
    }
    try:
        for split in EVAL_SPLITS:
            seeds = seen_seeds if split == "train_seen" else fresh_seeds
            env = PlanarPushSandbox(
                split=SPLIT_OF[split], image_size=IMAGE_SIZE[base.patch_grid]
            )
            try:
                trained = evaluate_split(trained_agent, env, seeds, ticks)
                untrained = evaluate_split(control, env, seeds, ticks)
            finally:
                env.close()
            out["splits"][split] = {
                "trained": trained,
                "untrained": untrained,
                "improvement": {
                    w: untrained["windows"][w]["rel_centred_mean"]
                    / max(trained["windows"][w]["rel_centred_mean"], 1e-300)
                    for w in trained["windows"]
                },
                "improvement_uncentred": {
                    w: untrained["windows"][w]["rel_uncentred_mean"]
                    / max(trained["windows"][w]["rel_uncentred_mean"], 1e-300)
                    for w in trained["windows"]
                },
            }
            for w in ("early", "late"):
                tw, uw = trained["windows"][w], untrained["windows"][w]
                print(
                    f"[B28] generalise {split:>16} [{w:>5} {tw['window_ticks']}]: "
                    f"centred trained {tw['rel_centred_mean']:.5f} "
                    f"(sd {tw['across_seed_sd']:.5f}) untrained "
                    f"{uw['rel_centred_mean']:.5f} -> x"
                    f"{out['splits'][split]['improvement'][w]:.3f} | uncentred "
                    f"{tw['rel_uncentred_mean']:.5f} vs {uw['rel_uncentred_mean']:.5f} "
                    f"-> x{out['splits'][split]['improvement_uncentred'][w]:.1f} | "
                    f"world std_max {tw['world_motion_std_max_median']:.3e} moving "
                    f"{tw['world_motion_frac_moving_median']:.4f}",
                    flush=True,
                )
    finally:
        control_env.close()
    out["improvement_ratio_to_train_seen"] = {
        w: {
            s: out["splits"][s]["improvement"][w]
            / max(out["splits"]["train_seen"]["improvement"][w], 1e-300)
            for s in EVAL_SPLITS
        }
        for w in ("early", "late")
    }
    return out


def _sib_live(sib: dict) -> dict:
    """The non-degenerate pairs where there are any, else the whole set."""
    return sib.get("non_degenerate", sib)


# -- the run ------------------------------------------------------------------


def reads(agent, chains, traffic, seed, label) -> dict:
    started = time.time()
    out = {
        "label": label,
        "orbit": orbit_read(agent, chains, seed=seed),
        "sibling": sibling_read(agent, seed=seed),
    }
    if traffic.count:
        out["traffic"] = traffic_read(agent, traffic, seed=seed)
    out["read_minutes"] = (time.time() - started) / 60.0
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", default="reserve_p16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ticks", type=int, default=20_000)
    p.add_argument("--prefix", default="577")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    out = args.out or _HERE / f"{args.prefix}-gauge-{args.arm}-seed{args.seed}-{args.ticks}.json"
    if out.exists():
        print(f"[B28] {out.name} already at the horizon, skipping", flush=True)
        return
    inflight = out.with_suffix(".inflight.json")

    env, agent = arms_mod.build_arm(args.arm, args.seed)
    started = time.time()
    try:
        dome = agent.dome
        chains = t2.rim_chains(dome)
        traffic = Traffic(dome, agent.sheaf.layout)
        motion = Motion(dome, agent.sheaf.layout)
        record = {
            "issue": 577,
            "reading": "gauge invariance, traffic alignment, sibling structure",
            "arm": args.arm,
            "seed": args.seed,
            "ticks": args.ticks,
            "window": WINDOW,
            "n": int(dome.shape.n),
            "chains": len(chains),
            "privacy": arms_mod.privacy_read(dome, arms_mod.ARMS[args.arm][1]),
            "widths": arms_mod.widths_read(dome, chains),
            "gauge_group": {
                "per_cell": "O(k_v) on the exposed block",
                "per_edge": "O(m_e) on the edge stalk",
                "fixed_at": "boundary cells and wholly-pinned incidences",
                "free_cells": int(
                    sum(
                        1
                        for c in dome.cells
                        if not c.is_boundary and not pinned_incidence(dome)[c.id]
                    )
                ),
                "fixed_cells": int(
                    sum(
                        1
                        for c in dome.cells
                        if c.is_boundary or pinned_incidence(dome)[c.id]
                    )
                ),
            },
            "checkpoints": [],
        }
        record["at_construction"] = reads(agent, chains, traffic, args.seed, "construction")
        o = record["at_construction"]["orbit"]
        print(
            f"[B28] construction: ER {o['before']['composed_er']['median']:.6f} | "
            f"worst orbit delta {o['worst_abs_delta']:.3e} | "
            f"energy rel delta {o['worst_energy_rel_delta']:.3e} | "
            f"sibling lead {_sib_live(record['at_construction']['sibling'])['leading_cos']['median']:.4f} "
            f"(z {_sib_live(record['at_construction']['sibling'])['leading_cos_z_vs_haar']['median']:+.2f})",
            flush=True,
        )

        bias = PredictionRule(agent.sheaf)
        transport = TransportRule(agent.sheaf)
        ladder = [c for c in CHECKPOINTS if c <= args.ticks]
        if args.ticks not in ladder:
            ladder.append(args.ticks)
        seen = 0
        for target in ladder:
            window_starts = target - WINDOW
            traffic.reset()
            motion.reset()
            for outcome in t0.teaching_read(
                agent, target - seen, args.seed + seen, _NullRecorder(), bias, transport
            ):
                if agent.sheaf.ticks > window_starts:
                    traffic.observe(agent.sheaf)
                    motion.observe(agent.sheaf)
            seen = target
            entry = reads(agent, chains, traffic, args.seed, f"@{target}")
            entry["world_motion"] = motion.read()
            entry["ticks"] = target
            entry["elapsed_minutes"] = (time.time() - started) / 60.0
            record["checkpoints"].append(entry)
            inflight.write_text(json.dumps(record, indent=1))
            o, tr, sb = entry["orbit"], entry.get("traffic", {}), entry["sibling"]
            nd = tr.get("non_degenerate", {})
            sbl = sb.get("non_degenerate", sb)
            print(
                f"[B28] @{target:>6}: ER {o['before']['composed_er']['median']:.6f} | "
                f"orbit delta {o['worst_abs_delta']:.3e} | "
                f"traffic[live {nd.get('lanes', 0)}] pct med "
                f"{nd.get('percentile', {}).get('median', float('nan')):.3f} "
                f"z {nd.get('z', {}).get('median', float('nan')):+.2f} "
                f"(capture {nd.get('capture', {}).get('median', float('nan')):.4f} vs null "
                f"{nd.get('null_median', {}).get('median', float('nan')):.4f}, ceiling "
                f"{nd.get('ceiling', {}).get('median', float('nan')):.4f}) | "
                f"sibling[live {sbl.get('pairs', 0)}] lead "
                f"{sbl['leading_cos']['median']:.4f} "
                f"z {sbl['leading_cos_z_vs_haar']['median']:+.2f} "
                f"| world std p90 {entry['world_motion'].get('std_p90', float('nan')):.2e} "
                f"({entry['elapsed_minutes']:.1f} min)",
                flush=True,
            )
        inflight.replace(out)
        print(f"[B28] wrote {out.name}", flush=True)

        # The trained maps are what every read here is a function of, and nothing
        # else on the rig saves them (#537's premise 1). ~4 MB, so the evaluation
        # protocol can be revised without paying for the ticks again.
        maps_out = out.with_name(
            out.name.replace("-gauge-", "-maps-").replace(".json", ".pt")
        )
        torch.save(
            {
                "arm": args.arm,
                "seed": args.seed,
                "ticks": args.ticks,
                "maps": agent.sheaf.maps.maps.detach().clone(),
            },
            maps_out,
        )
        print(f"[B28] wrote {maps_out.name}", flush=True)

        gen_name = out.name.replace("-gauge-", "-generalise-")
        if gen_name == out.name:
            gen_name = f"{out.stem}-generalise{out.suffix}"
        gen_out = out.with_name(gen_name)
        if gen_out.exists():
            print(f"[B28] {gen_out.name} already present, skipping item 4", flush=True)
        else:
            gen = generalise_read(args.arm, args.seed, agent, args.ticks)
            gen.update(
                {
                    "issue": 577,
                    "reading": "item 4 -- does the trained agreement generalise?",
                    "arm": args.arm,
                    "seed": args.seed,
                    "trained_ticks": args.ticks,
                    "note": (
                        "The trained surface is the one the ladder above left at "
                        "the horizon; no rule is stepped during evaluation."
                    ),
                }
            )
            gen_out.write_text(json.dumps(gen, indent=1))
            print(f"[B28] wrote {gen_out.name}", flush=True)
    finally:
        env.close()


class _NullRecorder:
    """`teaching_read` wants a recorder; this reading takes its own measurements."""

    def observe(self) -> None:  # pragma: no cover - trivial
        return None


if __name__ == "__main__":
    main()
