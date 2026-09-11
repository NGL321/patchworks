"""Is the tick heat or waves? The propagator spectrum and an impulse response.

The question (#532, 2026-09-11): the motivating image speaks of ripples in a
dissipative chamber. The tick as implemented is one Jacobi gradient step on
Dirichlet energy against a one-tick-delayed neighbour term
(`src/patchworks/tick.py`, *message_passing_phase*), with no momentum and no
second-order term, and the predicting stalk is overwritten by `decode` before
that step runs. That predicts pure diffusion: a real, positive propagator
spectrum inside the unit disc, no oscillatory modes, and an impulse whose peak
arrives later at farther cells without ever changing sign. This probe reads
that instead of arguing it.

Four readings, three surfaces, one held world:

1. **The orbit.** Whether the held world is at rest at all. `hold_still` for
   `HOLD` ticks, then 400 more, recording the whole carried state. Reported:
   the per-tick change as a fraction of the state's norm, its autocorrelation
   (a strong negative value at lag 2 and a positive one near lag 7 is a period
   of about 3.5 ticks), the lag of the first autocorrelation peak, and the
   per-cell ring amplitude. **If this is not near zero the single-tick
   Jacobian below is a linearisation on a moving orbit and its eigenvalues are
   not growth rates**; the reading that stands is the next one.

2. **Finite-time growth spectrum along the orbit.** Twelve orthonormal
   perturbations carried through the true tick for 300 ticks with a QR
   re-orthonormalisation each tick (Benettin's method), reported as per-tick
   moduli. A modulus at 1 is a mode nothing drains; above 1 is growth. These
   are finite-time rates on this orbit and named so -- not the long-run limit
   `CONTEXT.md` reserves the word *Lyapunov exponent* for.

3. **Propagator spectrum at one tick.** The one-tick map on the carried state
   -- `(stalks, charts, broadcast)`; `incoming`, `prediction` and the two
   `prior_*` tensors are a tick's outputs, not its inputs, and a check below
   confirms it -- linearised by central finite differences, world write
   included. Kept because it is exact where the orbit is stationary and its
   complex angles give the ring period directly; read it beside reading 1.
   The quantity is the *propagator spectrum*: not the spectral radius
   `CONTEXT.md` reserves for `K`.

4. **Impulse response.** `benchmarks/detectability.branch` with a unit rim
   impulse, the whole `[ticks, cells]` trace kept. Per cell: hop distance from
   the source, tick of peak, sign changes of the deviation projected on its own
   peak direction, and the dominant period of that projection. Peak time
   growing with distance squared is diffusion; growing linearly is a front; a
   shared period at every cell is a ring.

5. **Decomposition.** The held tick with one phase skipped -- inference alone,
   reconciliation alone -- and reconciliation with `broadcast` refreshed before
   the step so the neighbour term carries no unit delay. Says which loop the
   ring belongs to. Instrument, never mechanism.

Surfaces: the constructor's draw (`untrained`), `holonomy_read.flat_maps`
installed (`flat`), and `untrained_fixed_point.taught` at 2,000 ticks
(`taught2000`). Small dome, float64, seed 42. Output one JSON per surface,
next to this file. Nothing under `src/` changes; no benchmark CLI `read` is
called, so nothing files to GitHub.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
sys.path.insert(0, str(ROOT / "tests"))

import detectability as det  # noqa: E402
import holonomy_read  # noqa: E402
import untrained_fixed_point as ufp  # noqa: E402
from conftest import SMALL  # noqa: E402
from patchworks.agent import Agent  # noqa: E402
from patchworks.graph import build_graph  # noqa: E402
from patchworks.sandbox import PlanarPushSandbox  # noqa: E402

#: The state a tick reads. Everything else in `ufp._TICK_STATE` is written by
#: the tick before it is read again, which `check_state_closure` verifies.
CARRIED = ("stalks", "charts", "broadcast")

SEED = 42
HOLD = 100
WINDOW = 256
EPS = 1e-6
UNIT_TOL = 1e-9


# -- surfaces ----------------------------------------------------------------


def build(surface: str):
    env = PlanarPushSandbox(split="train", image_size=16)
    agent = Agent(
        env, dome=build_graph(SMALL), generator=torch.Generator().manual_seed(SEED)
    )
    observation, _info = env.reset(seed=SEED)
    agent.observe(observation)
    if surface == "flat":
        flat = holonomy_read.flat_maps(agent.dome, torch.Generator().manual_seed(SEED))
        with torch.no_grad():
            agent.sheaf.maps.maps.copy_(flat.maps.to(agent.sheaf.maps.maps.dtype))
    elif surface.startswith("taught"):
        ticks = int(surface[len("taught"):])
        ufp.taught(agent, ticks, seed=SEED)
        observation, _info = env.reset(seed=SEED)
        agent.observe(observation)
    elif surface != "untrained":
        raise ValueError(surface)
    det.double_precision(agent.sheaf)
    applied = np.zeros(env.action_space.shape, dtype=np.float64)
    det.hold_still(agent, observation, applied, HOLD)
    return env, agent, observation, applied


# -- the one-tick map --------------------------------------------------------


def pack(sheaf) -> torch.Tensor:
    return torch.cat([getattr(sheaf, name).reshape(-1) for name in CARRIED])


def unpack(sheaf, vector: torch.Tensor, base: dict) -> None:
    offset = 0
    for name in CARRIED:
        shape = base[name].shape
        size = base[name].numel()
        setattr(sheaf, name, vector[offset : offset + size].reshape(shape).clone())
        offset += size


def step(agent, base: dict, observation, applied, vector: torch.Tensor) -> torch.Tensor:
    """Restore the held snapshot, overwrite the carried state, tick, write."""
    ufp.restore(agent.sheaf, base)
    unpack(agent.sheaf, vector, base)
    agent.sheaf.tick()
    agent.write(observation, applied)
    return pack(agent.sheaf)


def check_state_closure(agent, base, observation, applied) -> float:
    """The tick is a function of `CARRIED` alone: scrambling the rest changes nothing."""
    x = pack(agent.sheaf)
    reference = step(agent, base, observation, applied, x)
    scrambled = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in base.items()}
    g = torch.Generator().manual_seed(7)
    for name in ufp._TICK_STATE:
        if name not in CARRIED:
            scrambled[name] = torch.randn(base[name].shape, generator=g, dtype=base[name].dtype)
    other = step(agent, scrambled, observation, applied, x)
    ufp.restore(agent.sheaf, base)
    return float((reference - other).abs().max())


def jacobian(agent, base, observation, applied) -> tuple[np.ndarray, np.ndarray]:
    x0 = pack(agent.sheaf).clone()
    dim = x0.numel()
    J = np.zeros((dim, dim), dtype=np.float64)
    started = time.time()
    for j in range(dim):
        e = torch.zeros_like(x0)
        e[j] = EPS
        plus = step(agent, base, observation, applied, x0 + e)
        minus = step(agent, base, observation, applied, x0 - e)
        J[:, j] = ((plus - minus) / (2 * EPS)).numpy()
        if j % 500 == 0:
            print(f"  column {j}/{dim}  {time.time() - started:.0f}s", flush=True)
    ufp.restore(agent.sheaf, base)
    return J, x0.numpy()


def layout_blocks(agent, base) -> dict[str, tuple[int, int]]:
    blocks = {}
    offset = 0
    for name in CARRIED:
        size = base[name].numel()
        blocks[name] = (offset, offset + size)
        offset += size
    return blocks


def spectrum_reading(J: np.ndarray, loops: dict[int, int]) -> dict:
    values = np.linalg.eigvals(J)
    modulus = np.abs(values)
    on_circle = np.abs(modulus - 1.0) <= UNIT_TOL
    above = modulus > 1.0 + UNIT_TOL
    complex_mask = (np.abs(values.imag) > UNIT_TOL) & (modulus > 0.5)
    inside = modulus < 1.0 - UNIT_TOL
    slowest = float(modulus[inside].max()) if inside.any() else float("nan")
    efold = -1.0 / math.log(slowest) if inside.any() else float("nan")
    # The five slowest complex modes, if any, with the period their angle implies.
    complex_rows = []
    if complex_mask.any():
        idx = np.argsort(-modulus[complex_mask])[:5]
        for v in values[complex_mask][idx]:
            angle = abs(math.atan2(v.imag, v.real))
            complex_rows.append(
                {
                    "modulus": float(abs(v)),
                    "angle_rad": float(angle),
                    "period_ticks": float(2 * math.pi / angle) if angle > 0 else float("inf"),
                }
            )
    # Real-negative modes near the circle are the delayed-Jacobi alternation, if it exists.
    alternating = int(((values.real < -0.5) & (np.abs(values.imag) <= UNIT_TOL)).sum())
    hist_edges = [0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0 - UNIT_TOL, 1.0 + UNIT_TOL, np.inf]
    hist = np.histogram(modulus, bins=hist_edges)[0].tolist()
    world = np.array(sorted(loops.values()), dtype=float)
    return {
        "dimension": int(J.shape[0]),
        "on_unit_circle": int(on_circle.sum()),
        "above_one": int(above.sum()),
        "largest_modulus": float(modulus.max()),
        "complex_above_half": int(complex_mask.sum()),
        "complex_slowest": complex_rows,
        "real_negative_below_minus_half": alternating,
        "slowest_draining_modulus": slowest,
        "slowest_efold_ticks": efold,
        "world_loop_min": float(world.min()),
        "world_loop_median": float(np.median(world)),
        "world_loop_max": float(world.max()),
        "modulus_histogram": {
            "edges": [float(e) if np.isfinite(e) else "inf" for e in hist_edges],
            "counts": hist,
        },
        "max_imaginary_part": float(np.abs(values.imag).max()),
        "min_real_part": float(values.real.min()),
    }


def circle_modes(J: np.ndarray, blocks: dict[str, tuple[int, int]], agent) -> dict:
    """Where the unit-circle modes live: how much of each eigenvector is in each block,
    and, within the stalks, in the private reserve."""
    values, vectors = np.linalg.eig(J)
    keep = np.abs(np.abs(values) - 1.0) <= UNIT_TOL
    if not keep.any():
        return {"count": 0}
    V = vectors[:, keep]
    weight = {}
    for name, (a, b) in blocks.items():
        weight[name] = float((np.abs(V[a:b]) ** 2).sum() / (np.abs(V) ** 2).sum())
    # The private reserve inside the stalks: the trailing p components of each
    # predicting stalk, which reconciliation leaves exactly invariant.
    layout = agent.sheaf.layout
    p = agent.dome.spec.private_reserve
    private = np.zeros(J.shape[0], dtype=bool)
    a, _ = blocks["stalks"]
    for cell in agent.dome.predicting:
        sl = layout.slice(cell)
        private[a + sl.stop - p : a + sl.stop] = True
    weight["stalks_private_reserve"] = float(
        (np.abs(V[private]) ** 2).sum() / (np.abs(V) ** 2).sum()
    )
    return {"count": int(keep.sum()), "energy_share_by_block": weight}


# -- the orbit and the growth spectrum ---------------------------------------

ORBIT_TICKS = 400
GROWTH_TICKS = 300
GROWTH_VECTORS = 12


def orbit_reading(agent, base, observation, applied, blocks) -> tuple[dict, list[torch.Tensor]]:
    """Is the held world at rest? The carried state over ORBIT_TICKS, from the snapshot."""
    ufp.restore(agent.sheaf, base)
    xs = [pack(agent.sheaf).clone()]
    for _ in range(ORBIT_TICKS):
        agent.sheaf.tick()
        agent.write(observation, applied)
        xs.append(pack(agent.sheaf).clone())
    ufp.restore(agent.sheaf, base)
    X = torch.stack(xs).numpy()
    dX = np.diff(X, axis=0)
    tail = slice(-100, None)
    change = float(np.linalg.norm(dX[tail], axis=1).mean() / np.linalg.norm(X[tail], axis=1).mean())
    Y = X[-300:] - X[-300:].mean(axis=0)
    denominator = float((Y * Y).sum())
    ac = [1.0] + [float((Y[:-k] * Y[k:]).sum() / denominator) for k in range(1, 40)]
    peaks = [k for k in range(2, 39) if ac[k] > ac[k - 1] and ac[k] > ac[k + 1] and ac[k] > 0.3]
    by_block = {
        name: float(np.linalg.norm(dX[tail, a:b], axis=1).mean() / max(np.linalg.norm(X[tail, a:b], axis=1).mean(), 1e-300))
        for name, (a, b) in blocks.items()
    }
    layout = agent.sheaf.layout
    a, _ = blocks["stalks"]
    ring = []
    for cell in agent.dome.cells:
        sl = layout.slice(cell.id)
        seg = dX[tail, a + sl.start : a + sl.stop]
        ring.append(
            {
                "cell": cell.id,
                "kind": cell.kind.name,
                "level": int(cell.index.level),
                "per_tick_change_rms": float(np.sqrt((seg**2).sum(axis=1).mean())),
                "stalk_norm": float(np.linalg.norm(X[tail, a + sl.start : a + sl.stop], axis=1).mean()),
            }
        )
    ring.sort(key=lambda r: -r["per_tick_change_rms"])
    return (
        {
            "ticks": ORBIT_TICKS,
            "per_tick_change_over_state_norm": change,
            "per_tick_change_by_block": by_block,
            "autocorrelation_lags_0_to_12": [round(v, 3) for v in ac[:13]],
            "autocorrelation_first_peak_lag": peaks[0] if peaks else None,
            "autocorrelation_peaks": peaks[:5],
            "ring_by_cell": ring,
            "cells_ringing_above_1e-3": int(sum(1 for r in ring if r["per_tick_change_rms"] > 1e-3)),
        },
        xs,
    )


def growth_reading(agent, base, observation, applied, x0: torch.Tensor) -> dict:
    """Finite-time growth rates along the held orbit, Benettin's method."""
    dim = x0.numel()
    g = torch.Generator().manual_seed(3)
    Q = torch.linalg.qr(torch.randn(dim, GROWTH_VECTORS, generator=g, dtype=torch.float64))[0]
    logs = np.zeros(GROWTH_VECTORS)
    x = x0.clone()
    for _ in range(GROWTH_TICKS):
        xq = step(agent, base, observation, applied, x)
        cols = [(step(agent, base, observation, applied, x + EPS * Q[:, j]) - xq) / EPS for j in range(GROWTH_VECTORS)]
        Q, R = torch.linalg.qr(torch.stack(cols, 1))
        logs += np.log(np.abs(np.diag(R.numpy())) + 1e-300)
        x = xq
    ufp.restore(agent.sheaf, base)
    rates = logs / GROWTH_TICKS
    moduli = np.exp(rates)
    neutral = int((np.abs(rates) < 1.0 / GROWTH_TICKS).sum())
    decaying = rates[rates <= -1.0 / GROWTH_TICKS]
    return {
        "ticks": GROWTH_TICKS,
        "vectors": GROWTH_VECTORS,
        "per_tick_moduli": [float(m) for m in moduli],
        "growing": int((rates >= 1.0 / GROWTH_TICKS).sum()),
        "neutral_within_resolution": neutral,
        "resolution_per_tick": 1.0 / GROWTH_TICKS,
        "slowest_decaying_efold_ticks": float(-1.0 / decaying.max()) if decaying.size else None,
    }


def decomposition_reading(agent, base, observation, applied, blocks) -> dict:
    """Which loop rings: the tick's two phases run alone, and reconciliation with no delay.

    Instrument, never mechanism: each variant is the held tick with one phase
    skipped, or with `broadcast` refreshed from the current stalks before the
    step so the neighbour term carries no unit delay. The per-tick change as a
    fraction of the state and the autocorrelation are read exactly as in
    :func:`orbit_reading`.
    """

    def variant(kind: str) -> dict:
        ufp.restore(agent.sheaf, base)
        xs = []
        for _ in range(300):
            with torch.no_grad():
                if kind == "reconcile_no_delay":
                    gathered = agent.sheaf.stalks[agent.sheaf.layout.pair_positions]
                    agent.sheaf.broadcast = agent.sheaf.maps.restrict(gathered)
                if kind in ("full", "inference"):
                    agent.sheaf.inference_phase()
                if kind in ("full", "reconcile", "reconcile_no_delay"):
                    agent.sheaf.message_passing_phase()
            agent.write(observation, applied)
            xs.append(pack(agent.sheaf).clone())
        ufp.restore(agent.sheaf, base)
        X = torch.stack(xs).numpy()
        tail = slice(-100, None)
        change = float(np.linalg.norm(np.diff(X[tail], axis=0), axis=1).mean() / np.linalg.norm(X[tail], axis=1).mean())
        Y = X[-200:] - X[-200:].mean(axis=0)
        d = float((Y * Y).sum())
        ac = [float((Y[:-k] * Y[k:]).sum() / d) if d > 0 else None for k in range(1, 9)]
        by_block = {
            name: float(np.linalg.norm(np.diff(X[tail, a:b], axis=0), axis=1).mean() / max(np.linalg.norm(X[tail, a:b], axis=1).mean(), 1e-300))
            for name, (a, b) in blocks.items()
        }
        return {"per_tick_change_over_state_norm": change, "by_block": by_block, "autocorrelation_lags_1_to_8": ac}

    return {kind: variant(kind) for kind in ("full", "inference", "reconcile", "reconcile_no_delay")}


# -- the impulse -------------------------------------------------------------


def hops_from(dome, source: int) -> dict[int, int]:
    distance = {source: 0}
    frontier = [source]
    while frontier:
        onward = []
        for cell in frontier:
            for edge_id in dome.incident[cell]:
                other = dome.edges[edge_id].other(cell)
                if other not in distance:
                    distance[other] = distance[cell] + 1
                    onward.append(other)
        frontier = onward
    return distance


def impulse_reading(agent, base, observation, applied) -> dict:
    dome = agent.dome
    cells = det.population(dome)
    sites = det.reading_sites(dome)
    source = det.rim(dome)[0]
    nudge = ((source, det.unit(dome.cells[source].stalk, torch.Generator().manual_seed(1))),)
    _, quiet = det.branch(agent, base, observation, applied, WINDOW, None, record=cells)
    _, moved = det.branch(agent, base, observation, applied, WINDOW, nudge, record=cells)
    ufp.restore(agent.sheaf, base)
    hops = hops_from(dome, source)
    rows = []
    total = np.zeros(WINDOW)
    for cell in cells:
        dev = (moved[cell] - quiet[cell]).numpy()  # [ticks, stalk]
        norm = np.linalg.norm(dev, axis=-1)
        site = np.linalg.norm(dev * sites[cell].numpy(), axis=-1)
        total += norm**2
        if norm.max() <= 0:
            rows.append({"cell": cell, "hops": hops.get(cell), "reached": False})
            continue
        peak = int(norm.argmax())
        direction = dev[peak] / np.linalg.norm(dev[peak])
        scalar = dev @ direction
        live = np.abs(scalar) > 1e-9 * np.abs(scalar).max()
        s = np.sign(scalar[live])
        sign_changes = int((s[1:] * s[:-1] < 0).sum())
        first = int(np.argmax(norm > 1e-12 * norm.max()))
        centred = scalar - scalar.mean()
        power = np.abs(np.fft.rfft(centred))
        power[0] = 0.0
        freq = np.fft.rfftfreq(len(centred))
        period = float(1.0 / freq[power.argmax()]) if power.max() > 0 else None
        rows.append(
            {
                "cell": cell,
                "hops": hops.get(cell),
                "reached": True,
                "first_tick": first,
                "peak_tick": peak,
                "peak_norm": float(norm[peak]),
                "end_norm": float(norm[-1]),
                "site_peak_tick": int(site.argmax()) if site.max() > 0 else None,
                "sign_changes_on_peak_direction": sign_changes,
                "dominant_period_ticks": period,
                "norm_rises_after_peak": int((np.diff(norm[peak:]) > 1e-12 * norm[peak]).sum()),
            }
        )
    reached = [r for r in rows if r["reached"] and r["hops"] is not None and r["hops"] > 0]
    d = np.array([r["hops"] for r in reached], dtype=float)
    t = np.array([r["peak_tick"] for r in reached], dtype=float)

    def fit(x):
        A = np.stack([x, np.ones_like(x)], axis=1)
        coef, res, *_ = np.linalg.lstsq(A, t, rcond=None)
        resid = float(((A @ coef - t) ** 2).sum())
        return {"slope": float(coef[0]), "intercept": float(coef[1]), "sse": resid}

    by_hop = {}
    for r in reached:
        by_hop.setdefault(r["hops"], []).append(r["peak_tick"])
    peak_total = int(total.argmax())
    return {
        "source": source,
        "window": WINDOW,
        "cells": rows,
        "fit_linear_in_hops": fit(d) if len(d) > 2 else None,
        "fit_quadratic_in_hops": fit(d**2) if len(d) > 2 else None,
        "median_peak_tick_by_hop": {int(k): float(np.median(v)) for k, v in sorted(by_hop.items())},
        "total_energy_peak_tick": peak_total,
        "total_energy_rises_after_peak": int((np.diff(total[peak_total:]) > 1e-12 * total[peak_total]).sum()),
        "total_energy_end_over_peak": float(total[-1] / total[peak_total]),
        "cells_with_sign_changes": int(sum(1 for r in reached if r["sign_changes_on_peak_direction"] > 0)),
        "median_dominant_period_ticks": float(np.median([r["dominant_period_ticks"] for r in reached if r["dominant_period_ticks"]])) if reached else None,
        "cells_reached": len(reached),
    }


# -- main --------------------------------------------------------------------


def run(surface: str, out: Path) -> dict:
    print(f"== {surface}", flush=True)
    env, agent, observation, applied = build(surface)
    base = ufp.snapshot(agent.sheaf)
    blocks = layout_blocks(agent, base)
    closure = check_state_closure(agent, base, observation, applied)
    print(f"  state closure residual {closure:.3e}  dim {pack(agent.sheaf).numel()}", flush=True)
    orbit, xs = orbit_reading(agent, base, observation, applied, blocks)
    print(
        f"  orbit: per-tick change/state {orbit['per_tick_change_over_state_norm']:.3e}  "
        f"ac lags 1-8 {orbit['autocorrelation_lags_0_to_12'][1:9]}  first peak lag {orbit['autocorrelation_first_peak_lag']}  "
        f"ringing cells {orbit['cells_ringing_above_1e-3']}/{len(orbit['ring_by_cell'])}  "
        f"loudest {orbit['ring_by_cell'][0]['cell']} ({orbit['ring_by_cell'][0]['kind']}, L{orbit['ring_by_cell'][0]['level']}) rms {orbit['ring_by_cell'][0]['per_tick_change_rms']:.3e}",
        flush=True,
    )
    growth = growth_reading(agent, base, observation, applied, xs[0])
    print(
        f"  growth along orbit: moduli {[round(m, 4) for m in growth['per_tick_moduli']]}  "
        f"growing {growth['growing']}  neutral {growth['neutral_within_resolution']}  slowest e-fold {growth['slowest_decaying_efold_ticks']}",
        flush=True,
    )
    decomposition = decomposition_reading(agent, base, observation, applied, blocks)
    for kind, row in decomposition.items():
        print(f"  {kind:18s} change/state {row['per_tick_change_over_state_norm']:.3e}  ac1-4 {[round(a, 2) if a is not None else None for a in row['autocorrelation_lags_1_to_8'][:4]]}", flush=True)
    J, _ = jacobian(agent, base, observation, applied)
    loops = det.world_loops(agent.dome)
    spectrum = spectrum_reading(J, loops)
    spectrum["unit_circle_modes"] = circle_modes(J, blocks, agent)
    spectrum["blocks"] = {k: list(v) for k, v in blocks.items()}
    spectrum["state_closure_residual"] = closure
    spectrum["read_at_a_fixed_point"] = bool(orbit["per_tick_change_over_state_norm"] < 1e-6)
    impulse = impulse_reading(agent, base, observation, applied)
    result = {
        "surface": surface,
        "seed": SEED,
        "dome": "SMALL",
        "hold": HOLD,
        "eps": EPS,
        "orbit": orbit,
        "growth_along_orbit": growth,
        "decomposition": decomposition,
        "spectrum_at_one_tick": spectrum,
        "impulse": impulse,
    }
    out.write_text(json.dumps(result, indent=1))
    s = spectrum
    print(
        f"  on circle {s['on_unit_circle']}  above 1 {s['above_one']} (max {s['largest_modulus']:.6f})  "
        f"complex>0.5 {s['complex_above_half']}  real<-0.5 {s['real_negative_below_minus_half']}  "
        f"slowest {s['slowest_draining_modulus']:.6f} -> {s['slowest_efold_ticks']:.1f} ticks  "
        f"world_loop min/med {s['world_loop_min']:.0f}/{s['world_loop_median']:.0f}",
        flush=True,
    )
    i = impulse
    print(
        f"  impulse: reached {i['cells_reached']}  sign-change cells {i['cells_with_sign_changes']}  median period {i['median_dominant_period_ticks']}  "
        f"peak-by-hop {i['median_peak_tick_by_hop']}  lin sse {i['fit_linear_in_hops']['sse']:.1f}  "
        f"quad sse {i['fit_quadratic_in_hops']['sse']:.1f}  energy end/peak {i['total_energy_end_over_peak']:.3e}",
        flush=True,
    )
    env.close()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--surfaces", nargs="+", default=["untrained", "flat", "taught2000"])
    args = parser.parse_args(argv)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    for surface in args.surfaces:
        run(surface, HERE / f"{surface}-seed{SEED}.json")


if __name__ == "__main__":
    main()
