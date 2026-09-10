"""B57 (#629): the first reading of the graded agreement instrument.

`#629 <https://github.com/NGL321/patchworks/issues/629>`_, under
`the map <https://github.com/NGL321/patchworks/issues/532>`_. The instrument was
specified by `B52 (#622) <https://github.com/NGL321/patchworks/issues/622>`_,
which deliberately took no reading; this takes it.

**The statistic.** Let `x_t` be the assembled node-stalk configuration over
predicting cells at tick `t` -- `sheaf.evidence()` flattened, which is
exactly the column layout `diagnostics.Diagnostics` lays `delta` out in -- and
`C = (1/T) sum_t x_t x_t^T` the traffic covariance over a window. Let `delta` be
the coboundary against the learned maps and `D` the block-diagonal
`sum_{e in v} F_e^T F_e` at each cell. For each eigendirection `u_i` of `C` with
weight `w_i = lambda_i / sum lambda`:

    q_i = (u_i^T L u_i) / (u_i^T D u_i)     L = delta^T delta
    A(theta) = sum_{q_i <= theta} w_i
    N(theta) = (sum_{q_i <= theta} w_i)^2 / sum_{q_i <= theta} w_i^2

**`C` is never formed.** `rank C <= T` and a direction with zero weight cannot
enter either sum, so the eigendirections that matter are the right singular
vectors of the `[T, columns]` traffic matrix `X`, and `lambda_i = s_i^2 / T`.
That also means `u_i^T L u_i = ||delta u_i||^2` and `u_i^T D u_i = ||G u_i||^2`
for a `G` that stacks each *predicting endpoint's* map in its own row block --
one matrix application each, instead of a `[4800, 4800]` product.

**`G` is not `delta` unsigned.** `delta` gives an edge's two endpoints the same
rows, because it differences them; `D` sums `F_e^T F_e` per endpoint, so `G`
must give every endpoint its own rows or a two-predicting-endpoint edge would
have its denominator read as `||F_u x_u + F_v x_v||^2`. Both layouts are built
here from the same walk over `dome.edges` and the coboundary half is checked
against :meth:`patchworks.diagnostics.Diagnostics.whole_graph` on the real
surface before anything is read (:func:`check_delta`).

**The window is `T = 1,000` ticks**, and it is not invented here: it is
`T0/run.py`'s `WINDOW`, the read window every cold-start figure on this rig has
been taken over. `#629` asks for the window to be defended rather than assumed,
so :func:`window_stability` re-reads one checkpoint at `T = 250` and `T = 2,000`
and the readout quotes the spread.

**Three `theta` levels**, per B52 -- `0.05 / 0.10 / 0.25`, and the profile is
primary. `q = 1` is the no-relationship point (`||F_u x_u - F_v x_v||^2` equal
to `||F_u x_u||^2 + ||F_v x_v||^2`), so agreement is a fraction of 1 and the
levels are read against a null that measures where 1 actually sits rather than
against a constant this file asserts.

Usage::

    PYTHONPATH=src python prototypes/cold-start/T6/b57_agreement.py --stage accept
    PYTHONPATH=src python prototypes/cold-start/T6/b57_agreement.py --stage trained --condition baseline
    PYTHONPATH=src python prototypes/cold-start/T6/b57_agreement.py --stage trained --condition flat
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
_T0, _T1, _T2, _T3, _T4 = (_HERE.parent / n for n in ("T0", "T1", "T2", "T3", "T4"))
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
t3 = _load("t3_run", _T3 / "run.py")
b50 = _load("b50_spectra", _T4 / "b50_spectra.py")
b40_routes = _load("b40_routes", _HERE / "b40_routes.py")

from patchworks.diagnostics import Diagnostics  # noqa: E402
from patchworks.learning import PredictionRule, TransportRule  # noqa: E402
from patchworks.restriction import RestrictionMaps, pair_index  # noqa: E402
from untrained_fixed_point import build  # noqa: E402

#: T0's read window, unchanged. See the header.
WINDOW = 1_000
#: B52's "three levels so nothing rests on a value".
THETAS = (0.05, 0.10, 0.25)
#: The profile, published whole. `q` is bounded above by 2 (anti-alignment).
PROFILE_GRID = (
    0.001, 0.003, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25,
    0.35, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00,
)
_SIGN = (1.0, -1.0)


# -- the two layouts ----------------------------------------------------------


class Layout:
    """Where every predicting endpoint's map goes, in both matrices.

    Built once off the dome, because the *shape* of both is a construction fact
    and only the numbers in them move -- the same argument
    :class:`patchworks.diagnostics.Diagnostics` makes for its own cached layout.
    """

    def __init__(self, dome) -> None:
        n = dome.shape.n
        self.n = n
        self.row_of = {cell_id: i for i, cell_id in enumerate(dome.predicting)}
        self.columns = len(dome.predicting) * n
        self.delta_rows = sum(edge.m for edge in dome.edges)

        delta_blocks: list[tuple[int, int, int, int, float]] = []
        gram_blocks: list[tuple[int, int, int, int]] = []
        at = 0
        gram_at = 0
        for edge in dome.edges:
            for side, cell_id in enumerate((edge.u, edge.v)):
                if cell_id not in self.row_of:
                    continue
                pair = pair_index(edge.id, side)
                column = self.row_of[cell_id] * n
                delta_blocks.append((pair, at, edge.m, column, _SIGN[side]))
                gram_blocks.append((pair, gram_at, edge.m, column))
                gram_at += edge.m
            at += edge.m
        self.delta_blocks = tuple(delta_blocks)
        self.gram_blocks = tuple(gram_blocks)
        self.gram_rows = gram_at

    def _assemble(self, weights: np.ndarray, blocks, rows: int, signed: bool) -> np.ndarray:
        out = np.zeros((rows, self.columns), dtype=np.float64)
        for block in blocks:
            if signed:
                pair, at, m, column, sign = block
            else:
                pair, at, m, column = block
                sign = 1.0
            out[at : at + m, column : column + self.n] += (
                sign * weights[pair, :m, : self.n]
            )
        return out

    def delta(self, maps) -> np.ndarray:
        """`delta_P`: the coboundary over the predicting-cell subcomplex."""
        w = maps.maps.detach().to(torch.float64).numpy()
        return self._assemble(w, self.delta_blocks, self.delta_rows, True)

    def gram(self, maps) -> np.ndarray:
        """`G` with `G^T G = D`: one row block per predicting endpoint."""
        w = maps.maps.detach().to(torch.float64).numpy()
        return self._assemble(w, self.gram_blocks, self.gram_rows, False)


def check_delta(agent, layout: Layout) -> dict:
    """Hold this file's `delta_P` against the one already on `main`.

    `Diagnostics.whole_graph` assembles the same matrix by its own route and
    reports `dim H0 = columns - rank delta_P`. If the two disagree, one of them
    has a dropped block, a flipped sign or a shifted offset, and the whole
    reading below is of nothing. Costs one decomposition and is worth it.
    """
    diag = Diagnostics(agent.sheaf)
    reading = diag.whole_graph()
    d = layout.delta(agent.sheaf.maps)
    s = np.linalg.svd(d, compute_uv=False)
    tol = max(d.shape) * float(s[0]) * np.finfo(np.float64).eps
    rank = int((s > tol).sum())
    return {
        "diagnostics_rank": int(reading.rank),
        "here_rank": rank,
        "diagnostics_dim_h0": int(reading.dim_h0),
        "here_dim_h0": int(layout.columns - rank),
        "agrees": bool(reading.rank == rank),
        "columns": int(layout.columns),
        "delta_rows": int(layout.delta_rows),
        "gram_rows": int(layout.gram_rows),
    }


# -- the traffic --------------------------------------------------------------


class Traffic:
    """A ring buffer of the last `window` assembled node-stalk configurations.

    `sheaf.evidence()` is `[predicting cells, n]` gathered from
    `layout.predicting_positions`, which is `dome.predicting` order -- the order
    :class:`Layout` lays its columns out in -- so the flatten *is* the assembly
    and there is no second convention to keep in step. It is read after the
    tick, so it is the node stalk reconciliation left behind: what every page is
    actually holding, which is B52's `x_t`.
    """

    def __init__(self, agent, window: int = WINDOW) -> None:
        self.agent = agent
        self.window = window
        self.buffer: list[np.ndarray] = []

    @torch.no_grad()
    def observe(self) -> None:
        x = self.agent.sheaf.evidence().detach().reshape(-1)
        self.buffer.append(np.asarray(x.to(torch.float64)))
        if len(self.buffer) > self.window:
            self.buffer.pop(0)

    def matrix(self, window: int | None = None) -> np.ndarray:
        """`X`, `[T, columns]`. `window` truncates to the *last* `window` ticks."""
        rows = self.buffer if window is None else self.buffer[-window:]
        return np.stack(rows) if rows else np.zeros((0, 0))


def directions(x: np.ndarray, *, centred: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """`C`'s eigendirections and weights, from `X` and without forming `C`.

    `C = X^T X / T`, so the right singular vectors of `X` are `C`'s
    eigenvectors and `lambda_i = s_i^2 / T`. Directions carrying no weight are
    dropped: `w_i = 0` contributes nothing to `A` or to `N`, and keeping them
    would put 3,800 numerically arbitrary `q_i` into a profile.

    **`centred` subtracts the window's own mean configuration first**, and both
    readings are taken everywhere. B52 wrote `C = (1/T) sum_t x_t x_t^T` and
    called it a covariance; it is a second moment *about zero*, and the node
    stalk is not a zero-mean object -- `body.py:392`'s `encode` is affine ->
    ReLU -> affine, and `body.py:528-531` says in as many words that *"any
    nonzero mean in its stalk is permanently unreachable error"*, which is why
    `decode_output_bias` exists at all. A standing baseline `mu` puts a
    **rank-one** `mu mu^T` into the uncentered moment, so an uncentered ceiling
    may be reporting the baseline rather than the variety riding on it. B53's
    advisory on #629 asks for the two side by side and #633 is blocked on the
    answer; neither is quoted here in preference to the other.
    """
    if x.size == 0:
        return np.zeros((0,)), np.zeros((0, 0))
    if centred:
        x = x - x.mean(axis=0, keepdims=True)
    _, s, vh = np.linalg.svd(x, full_matrices=False)
    lam = s.astype(np.float64) ** 2
    total = float(lam.sum())
    w = lam / max(total, 1e-300)
    keep = w > 1e-12
    return w[keep], vh[keep].T


# -- the statistic ------------------------------------------------------------


def soft_count(w: np.ndarray) -> float:
    """`(sum w)^2 / sum w^2` -- ADR-0010's participation ratio, deliberately."""
    if w.size == 0:
        return 0.0
    return float(w.sum() ** 2 / max(float((w**2).sum()), 1e-300))


def agreement(w: np.ndarray, u: np.ndarray, delta: np.ndarray, gram: np.ndarray) -> dict:
    """B52's instrument on one `(C, maps)` pair.

    `q_i` is left as `nan` where the denominator vanishes -- a direction no
    incident map can see at all is not *agreed about*, it is invisible, and
    counting it would be the one way this statistic could be gamed by closing
    the mask. Those directions are reported separately and never enter `A`.
    """
    if w.size == 0:
        return {"directions": 0}
    num = np.square(delta @ u).sum(axis=0)
    den = np.square(gram @ u).sum(axis=0)
    seen = den > 1e-24 * max(float(den.max()), 1e-300)
    q = np.full(w.shape, np.nan)
    q[seen] = num[seen] / den[seen]

    profile = {}
    for theta in PROFILE_GRID:
        below = seen & (q <= theta)
        profile[f"{theta:g}"] = {
            "A": float(w[below].sum()),
            "N": soft_count(w[below]),
            "directions": int(below.sum()),
        }
    headline = {f"{t:g}": profile[f"{t:g}"] for t in THETAS}

    order = np.argsort(-w)
    return {
        "directions": int(w.size),
        "unseen_directions": int((~seen).sum()),
        "unseen_weight": float(w[~seen].sum()),
        "traffic_effective_rank": soft_count(w),
        "weight_top": float(w.max()),
        "weight_top5": float(np.sort(w)[::-1][:5].sum()),
        "profile": profile,
        "headline": headline,
        "q_of_leading": None if not seen[order[0]] else float(q[order[0]]),
        "q_weighted_mean": float(np.nansum(w[seen] * q[seen]) / max(float(w[seen].sum()), 1e-300)),
        "q_quantiles": {
            k: (None if not seen.any() else float(np.quantile(q[seen], v)))
            for k, v in (("p10", 0.1), ("median", 0.5), ("p90", 0.9))
        },
        # The eight heaviest directions, level beside weight, so the profile can
        # be read as a spectrum rather than as a curve.
        "top_directions": [
            {"w": float(w[i]), "q": None if not seen[i] else float(q[i])}
            for i in order[:8]
        ],
    }


def both(x: np.ndarray, delta: np.ndarray, gram: np.ndarray) -> dict:
    """The statistic read on the uncentered moment and on the covariance proper.

    B52's form is `uncentred`; `centred` is B53's advisory. Reported side by
    side on every row, and the readout quotes neither alone.
    """
    out = {}
    for name, flag in (("uncentred", False), ("centred", True)):
        w, u = directions(x, centred=flag)
        out[name] = agreement(w, u, delta, gram)
    return out


# -- the joint report ---------------------------------------------------------


def exposure(dome, maps, layout: Layout) -> dict:
    """Effective exposure per cell, rank-measured -- B48's third column.

    `rank(sum_{e in v} F_e^T F_e)` is how many of a cell's `n` components its
    neighbours can see at all, and the participation ratio of the same spectrum
    is the graded form. **The spec field is not read**: the map's note says
    effective exposure, and B42's 14.8 of 32 is this number.
    """
    w = maps.maps.detach().to(torch.float64).numpy()
    n = layout.n
    hard, soft = [], []
    for cell_id in dome.predicting:
        blocks = []
        for edge_id, edge in enumerate(dome.edges):
            for side, other in enumerate((edge.u, edge.v)):
                if other == cell_id and edge.m > 0:
                    blocks.append(w[pair_index(edge_id, side)][: edge.m, :n])
        if not blocks:
            continue
        stack = np.concatenate(blocks, axis=0)
        s = np.linalg.svd(stack, compute_uv=False)
        tol = max(stack.shape) * float(s[0]) * np.finfo(np.float64).eps if s.size else 0.0
        hard.append(int((s > tol).sum()))
        soft.append(soft_count(s**2))
    return {
        "n": int(n),
        "cells": len(hard),
        "rank_mean": float(np.mean(hard)),
        "rank_median": float(np.median(hard)),
        "effective_mean": float(np.mean(soft)),
        "effective_median": float(np.median(soft)),
    }


def joint(agent, layout: Layout, chains, x: np.ndarray, label: str, *, seed: int = 0) -> dict:
    """One row of the report: agreement, differentiation, exposure. Never one alone.

    The acceptance scramble rides along on every row rather than only on the
    construction surface. B52 pre-registers the check on the instrument, and a
    surface with no agreement to read cannot tell a pinned statistic from an
    insensitive one -- so it is re-run wherever the reading is taken.
    """
    dome, maps = agent.dome, agent.sheaf.maps
    delta, gram = layout.delta(maps), layout.gram(maps)
    reading = both(x, delta, gram)
    scrambles = {
        name: both(xs, delta, gram)
        for name, xs in scrambled_traffic(x, layout, seed).items()
    }
    return {
        "label": label,
        "window_ticks": int(x.shape[0]) if x.size else 0,
        "agreement": reading["uncentred"],
        "agreement_centred": reading["centred"],
        "scrambles": {k: v["uncentred"] for k, v in scrambles.items()},
        "scrambles_centred": {k: v["centred"] for k, v in scrambles.items()},
        "audience_differentiation": b50.audience_differentiation(dome, maps, chains),
        "exposure": exposure(dome, maps, layout),
    }


def generic_maps(dome, seed: int):
    """The matched-generic null: same `n`, same `m_e`, same mask, redrawn.

    `RestrictionMaps.__init__` applies the support mask and the gauge norm, and
    `project()` puts the draw through the same floor, band and incoherence step
    the built surface goes through at construction. Same room, phrasebooks
    redrawn -- and nothing about the traffic is touched, which is the point.
    """
    maps = RestrictionMaps(dome, generator=torch.Generator().manual_seed(seed + 4242))
    maps.project()
    return maps


def against_generic(agent, layout: Layout, x: np.ndarray, seed: int, draws: int = 3) -> dict:
    """`N_gen(theta)` and the excess, both curves published.

    Averaged over `draws` independent redraws, because a single generic surface
    is a sample and the excess is the number the map would quote.
    """
    out = {"draws": draws}
    drawn = [generic_maps(agent.dome, seed + 100 * d) for d in range(draws)]
    for variant, flag in (("uncentred", False), ("centred", True)):
        w, u = directions(x, centred=flag)
        per_draw = [
            agreement(w, u, layout.delta(maps), layout.gram(maps)) for maps in drawn
        ]
        entry = {"per_draw": per_draw}
        for key in ("A", "N"):
            entry[f"{key}_mean"] = {
                g: float(np.mean([p["profile"][g][key] for p in per_draw]))
                for g in per_draw[0]["profile"]
            }
        entry["q_weighted_mean"] = float(
            np.mean([p["q_weighted_mean"] for p in per_draw])
        )
        out[variant] = entry
    # B52's form stays at the top level so the record reads the way the spec is
    # written; the centred half sits beside it and neither is the default.
    out.update(out["uncentred"])
    return out


# -- item 1: the acceptance check ---------------------------------------------


def scrambled_traffic(x: np.ndarray, layout: Layout, seed: int) -> dict:
    """Redraw the traffic, hold every map fixed. B52's pre-registered check.

    Two redraws, because they fail differently and only the first is
    pre-registered:

    * `permute` -- each tick's configuration has its *cells* permuted, so every
      page in the room is still a page somebody was holding and only the
      arrangement is destroyed. This is "one scramble of the held
      configuration" as B52 wrote it.
    * `isotropic` -- a white configuration of the same total power, which is the
      degenerate end and says what the statistic reads with no traffic
      structure at all.

    `earned` provably cannot move under either: it never sees a page.
    """
    rng = np.random.default_rng(seed)
    cells = len(layout.row_of)
    n = layout.n
    frames = x.reshape(x.shape[0], cells, n)
    permuted = np.stack([f[rng.permutation(cells)] for f in frames]).reshape(x.shape)
    white = rng.standard_normal(x.shape)
    white *= np.linalg.norm(x) / max(np.linalg.norm(white), 1e-300)
    return {"permute": permuted, "isotropic": white}


# -- the arms -----------------------------------------------------------------


def build_arm(condition: str, seed: int):
    """T3's two arms, plus B40's flat bundle installed on the baseline arm."""
    base = "baseline" if condition == "flat" else condition
    arm = t3.CONDITIONS[base]
    env, agent = build("real", "train", seed)
    info = None
    if arm["pin"]:
        t1.pin_drive_edges(agent)
    if condition == "flat":
        info = b40_routes.install_flat_bundle(agent.dome, agent.sheaf.maps, seed)
    return env, agent, arm, info


def collect(agent, traffic: Traffic, ticks: int, seed: int, recorder, bias, transport, *, learn: bool):
    """Tick, recording the traffic every tick. `learn=False` freezes the maps."""
    if learn:
        for _ in t0.teaching_read(agent, ticks, seed, recorder, bias, transport):
            traffic.observe()
    else:
        for _ in t0.run_ticks(agent, ticks, seed=seed):
            recorder.observe()
            traffic.observe()


def _line(tag: str, row: dict) -> str:
    a = row["agreement"]
    h = a.get("headline", {})
    ad = row["audience_differentiation"]
    ex = row["exposure"]
    c = row.get("agreement_centred", {})
    parts = " ".join(f"N@{t:g} {h.get(f'{t:g}', {}).get('N', float('nan')):.4f}" for t in THETAS)
    parts += " | cen " + " ".join(
        f"N@{t:g} {c.get('headline', {}).get(f'{t:g}', {}).get('N', float('nan')):.4f}"
        for t in THETAS
    )
    return (
        f"  {tag:<28} traffic-ER {a.get('traffic_effective_rank', float('nan')):.4f}"
        f"/{c.get('traffic_effective_rank', float('nan')):.4f} "
        f"| {parts} | q_w {a.get('q_weighted_mean', float('nan')):.4f}"
        f"/{c.get('q_weighted_mean', float('nan')):.4f} "
        f"| aud-diff {ad.get('median', float('nan')):.4f} "
        f"| exposure {ex['effective_median']:.2f}/{ex['n']}"
    )


def stage_accept(condition: str, seed: int, out: Path, window: int = WINDOW) -> None:
    """Item 1, on its own and first: it can end the ticket.

    Read at construction on a frozen surface, because the acceptance check is
    about the *instrument* and not about training: if the number does not move
    when only the traffic is redrawn, no amount of training makes it stalk-side.
    """
    env, agent, arm, _ = build_arm(condition, seed)
    try:
        layout = Layout(agent.dome)
        chains = t2.rim_chains(agent.dome)
        record = {
            "issue": 629,
            "stage": "accept",
            "condition": condition,
            "seed": seed,
            "window": window,
            "surface": t0.surface(),
            "delta_check": check_delta(agent, layout),
        }
        print(f"[B57] delta check: {record['delta_check']}", flush=True)
        traffic = Traffic(agent, window)
        recorder = t2.EdgeRecorder(agent)
        collect(agent, traffic, window, seed, recorder, None, None, learn=False)
        x = traffic.matrix()
        held = joint(agent, layout, chains, x, f"{condition} held @construction", seed=seed)
        record["held"] = held
        print(_line("held", held), flush=True)

        maps = agent.sheaf.maps
        delta, gram = layout.delta(maps), layout.gram(maps)
        record["scrambles"] = held["scrambles"]
        record["scrambles_centred"] = held["scrambles_centred"]
        for name in record["scrambles"]:
            for variant, table in (
                ("uncentred", record["scrambles"]),
                ("centred", record["scrambles_centred"]),
            ):
                a = table[name]
                print(
                    f"  scramble/{name}/{variant:<12} traffic-ER "
                    f"{a['traffic_effective_rank']:.4f} | A@1 {a['profile']['1']['A']:.4f} "
                    f"N@1 {a['profile']['1']['N']:.4f} | q_w {a['q_weighted_mean']:.4f}",
                    flush=True,
                )
        # The window defence, on the same held configuration.
        record["window_stability"] = {}
        for w_len in (window // 4, window // 2, window):
            xw = traffic.matrix(w_len)
            record["window_stability"][str(w_len)] = both(xw, delta, gram)
        record["generic"] = against_generic(agent, layout, x, seed)
        out.write_text(json.dumps(record, indent=1))
    finally:
        env.close()
    print(f"[B57] wrote {out.name}", flush=True)


def stage_trained(condition: str, seed: int, ticks: int, out: Path, window: int = WINDOW) -> None:
    """Items 2-5 across the checkpoint ladder, on one arm."""
    started = time.time()
    env, agent, arm, flat_info = build_arm(condition, seed)
    try:
        layout = Layout(agent.dome)
        chains = t2.rim_chains(agent.dome)
        record = {
            "issue": 629,
            "stage": "trained",
            "condition": condition,
            "arm": {"rho1_drive_edges": bool(arm["pin"]), "c_learning_rate": float(arm["c"])},
            "flat_bundle": flat_info,
            "seed": seed,
            "ticks": ticks,
            "window": window,
            "thetas": list(THETAS),
            "surface": t0.surface(),
            "delta_check": check_delta(agent, layout),
            "checkpoints": [],
        }
        print(f"[B57] delta check: {record['delta_check']}", flush=True)

        traffic = Traffic(agent, window)
        recorder = t2.EdgeRecorder(agent)
        bias = PredictionRule(agent.sheaf, operator_rate_ratio=arm["c"])
        transport = TransportRule(agent.sheaf)

        # At construction: the same window, frozen maps, no learning. B49's
        # construction-to-trained rule is paid rather than inherited.
        collect(agent, traffic, window, seed, recorder, None, None, learn=False)
        x = traffic.matrix()
        entry = joint(agent, layout, chains, x, f"{condition} s{seed} @construction", seed=seed)
        entry["generic"] = against_generic(agent, layout, x, seed)
        record["at_construction"] = entry
        print(_line(f"{condition} s{seed} @0", entry), flush=True)
        out.write_text(json.dumps(record, indent=1))

        ladder = [cp for cp in t3.CHECKPOINTS if cp <= ticks]
        if ticks not in ladder:
            ladder.append(ticks)
        seen = 0
        for target in ladder:
            collect(
                agent, traffic, target - seen, seed + seen, recorder, bias, transport, learn=True
            )
            seen = target
            x = traffic.matrix()
            entry = joint(agent, layout, chains, x, f"{condition} s{seed} @{target}", seed=seed)
            entry["generic"] = against_generic(agent, layout, x, seed)
            entry["ticks"] = target
            entry["elapsed_minutes"] = (time.time() - started) / 60.0
            record["checkpoints"].append(entry)
            out.write_text(json.dumps(record, indent=1))
            print(
                _line(f"{condition} s{seed} @{target}", entry)
                + f" ({entry['elapsed_minutes']:.1f} min)",
                flush=True,
            )
    finally:
        env.close()
    print(f"[B57] wrote {out.name}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("accept", "trained"), required=True)
    p.add_argument("--condition", default="baseline", choices=("baseline", "winner", "flat"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ticks", type=int, default=20_000)
    p.add_argument("--window", type=int, default=WINDOW)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    if args.stage == "accept":
        out = args.out or _HERE / f"629-accept-{args.condition}-seed{args.seed}.json"
        stage_accept(args.condition, args.seed, out, args.window)
    else:
        out = args.out or _HERE / f"629-{args.condition}-seed{args.seed}-{args.ticks}.json"
        stage_trained(args.condition, args.seed, args.ticks, out, args.window)


if __name__ == "__main__":
    main()
