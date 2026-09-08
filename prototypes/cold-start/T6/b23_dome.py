"""T6 (#572 / B23): does a cell's *level* predict anything about what it holds?

The dome is a wager — that forcing semi-hierarchical connectivity forces
abstraction to develop. This instrument reads the wager on the trained arms.
It measures, **per predicting cell**, structure the dome fixes at construction
against dynamics that training produces, and asks whether the second is a
function of the first.

Structural, fixed at construction (the dome's own gradient claim,
`docs/spec/06-graph-topology.md`, *Private dimension is a gradient*):

* ``level``       — `cell.index.level`, 1 at the rim, 7 at the apex.
* ``p_v``         — private dimension, `max(0, n - sum_e m_e)`.
* ``k_v``         — the exposed (permitted) block width.
* ``sum_m``       — lane budget spent on this cell's edges.
* ``degree``      — incident edge count.
* ``fan_in``      — boundary cells in this cell's downward cone (receptive field).
* ``hops_to_rim`` — BFS distance to the nearest boundary cell.

Dynamical, measured over the last `WINDOW` ticks at each checkpoint:

* ``tau_e`` / ``tau_int`` / ``lag1`` — **emergent timescale**. Nothing in the
  running architecture sets a per-cell rate: `patchworks.timescale.ClockDivisor`
  is an *instrument*, held by no part of `src/` outside its own module and its
  tests, so any timescale gradient is produced by the dynamics or is absent.
  The statistic is a vector autocorrelation of the **centred** trajectory,
  `r(L) = <x_t, x_{t+L}> / sqrt(|x_t|^2 |x_{t+L}|^2)`, on the chart `h` and on
  the node stalk `v`.
* ``pr_total_centred`` etc. — state rank, T0's `excitation_reads`, centred and
  uncentred both, because [B17](#565) found the uncentred statistic reads ~1
  whatever the rank of the variation.
* ``mean_share`` — `|mean|^2 / energy`, how much of the cell is static.

Nothing here is a composed-rank reading and nothing here edits `src/`.

Usage::

    PYTHONPATH=src python prototypes/cold-start/T6/b23_dome.py --arms reserve shipped --ticks 20000
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
_T0, _T2, _T3, _T4 = (_HERE.parent / n for n in ("T0", "T2", "T3", "T4"))
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
t3 = _load("t3_run", _T3 / "run.py")
arms_mod = _load("t6_arms", _HERE / "arms.py")
excitation = _load("t0_excitation", _T0 / "excitation.py")

from patchworks.graph import CellKind, EdgeKind  # noqa: E402
from patchworks.learning import PredictionRule, TransportRule  # noqa: E402
from patchworks.restriction import pair_index  # noqa: E402

#: Lags the autocorrelation is evaluated at. The window is 1000 ticks, so the
#: longest lag still averages over 500 samples.
LAGS = (1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 500)


# -- structure ----------------------------------------------------------------


def structure(agent) -> dict:
    """Per predicting cell: what the dome fixed before a tick ran."""
    dome = agent.dome
    predicting = list(dome.predicting)
    boundary = set(dome.boundary)
    drive = {cid for cid, c in enumerate(dome.cells) if c.kind == CellKind.DRIVE}
    n = dome.shape.n

    # BFS from the boundary over every edge: hop distance to the world.
    dist = {c: 0 for c in boundary}
    queue = deque(boundary)
    while queue:
        cur = queue.popleft()
        for nb in dome.neighbours(cur):
            if nb not in dist:
                dist[nb] = dist[cur] + 1
                queue.append(nb)

    # The same distance to the **sensory** rim only. The drive cell is a
    # boundary cell attached at the apex (ADR-0009, *a drive is a motor edge
    # attached deep*), so a plain boundary BFS reports the apex as one hop from
    # the world and is not a depth axis at all. This one is.
    sensory = boundary - drive
    sdist = {c: 0 for c in sensory}
    queue = deque(sensory)
    while queue:
        cur = queue.popleft()
        for nb in dome.neighbours(cur):
            if nb in drive or nb in sdist:
                continue
            sdist[nb] = sdist[cur] + 1
            queue.append(nb)

    # The downward cone: boundary cells reachable without ever going up a level.
    level_of = {c.id: c.index.level for c in dome.cells}

    def cone(cell_id: int) -> int:
        seen = {cell_id}
        stack = [cell_id]
        hit = 0
        while stack:
            cur = stack.pop()
            for nb in dome.neighbours(cur):
                if nb in seen:
                    continue
                if nb in boundary:
                    seen.add(nb)
                    hit += 1
                    continue
                if level_of[nb] < level_of[cur]:
                    seen.add(nb)
                    stack.append(nb)
        return hit

    rows = []
    for row, cid in enumerate(predicting):
        cell = dome.cells[cid]
        incident = dome.incident[cid]
        lanes = [int(dome.edges[e].m) for e in incident]
        interior = [
            int(dome.edges[e].m)
            for e in incident
            if dome.edges[e].kind is EdgeKind.INTERIOR
        ]
        rows.append(
            {
                "row": row,
                "cell_id": int(cid),
                "level": int(cell.index.level),
                "column": str(cell.index.column),
                "degree": int(dome.degrees[cid]),
                "k_v": int(dome._permitted[cid]),
                "p_v": int(n - dome._permitted[cid])
                if dome._permitted[cid] <= n
                else 0,
                "p_v_mask": int(dome.private_mask[row].sum()),
                "sum_m": int(sum(lanes)),
                "sum_m_interior": int(sum(interior)),
                "m_max": int(max(lanes)) if lanes else 0,
                "hops_to_rim": int(dist.get(cid, -1)),
                "hops_to_sensory_rim": int(sdist.get(cid, -1)),
                "fan_in": int(cone(cid)),
                "boundary_adjacent": int(
                    sum(1 for nb in dome.neighbours(cid) if nb in boundary)
                ),
                "drive_adjacent": int(any(nb in drive for nb in dome.neighbours(cid))),
            }
        )
    return {"n": int(n), "k": int(dome.shape.k), "cells": rows}


def emission_gain(agent) -> list[float]:
    """Per cell: the summed leading singular value of every map it sends on.

    An operator-side proxy for *how loudly a cell speaks*, so that reach has some
    reading here at all. The causal question — how far a perturbation actually
    travels — is [B21](#570)'s and is not attempted here.
    """
    dome = agent.dome
    maps = agent.sheaf.maps
    out = []
    for cid in dome.predicting:
        total = 0.0
        for eid in dome.incident[cid]:
            side = 0 if dome.edges[eid].u == cid else 1
            f = maps.maps[pair_index(eid, side)].detach().double()
            if f.numel():
                total += float(torch.linalg.svdvals(f)[0])
        out.append(total)
    return out


# -- dynamics -----------------------------------------------------------------


@torch.no_grad()
def autocorrelation(stream: torch.Tensor) -> dict:
    """`[T, cells, d]` in: the centred vector autocorrelation, per cell.

    Returns `lag1`, `tau_e` (the lag where `r` first falls below `1/e`, linearly
    interpolated) and `tau_int` (`1 + 2*sum r`, trapezoidal over the evaluated
    lags, truncated at the first non-positive `r`). A cell whose centred
    trajectory is identically zero reads `nan`.
    """
    x = stream.double()
    x = x - x.mean(dim=0, keepdim=True)
    T = x.shape[0]
    cells = x.shape[1]
    lags = [L for L in LAGS if L < T]
    r = np.full((len(lags), cells), np.nan)
    for i, L in enumerate(lags):
        a, b = x[: T - L], x[L:]
        num = (a * b).sum(dim=(0, 2))
        den = (a.pow(2).sum(dim=(0, 2)) * b.pow(2).sum(dim=(0, 2))).sqrt()
        live = den > 1e-300
        vals = torch.where(live, num / den.clamp(min=1e-300), torch.full_like(num, float("nan")))
        r[i] = vals.numpy()

    thresh = float(np.exp(-1.0))
    tau_e = np.full(cells, np.nan)
    tau_int = np.full(cells, np.nan)
    for c in range(cells):
        rc = r[:, c]
        if np.isnan(rc).all():
            continue
        prev_lag, prev_val = 0.0, 1.0
        for L, val in zip(lags, rc):
            if np.isnan(val):
                break
            if val < thresh:
                span = prev_val - val
                frac = (prev_val - thresh) / span if span > 0 else 0.0
                tau_e[c] = prev_lag + frac * (L - prev_lag)
                break
            prev_lag, prev_val = float(L), float(val)
        else:
            tau_e[c] = float(lags[-1])  # censored: still above 1/e at the last lag
        area, prev_lag, prev_val = 0.0, 0.0, 1.0
        for L, val in zip(lags, rc):
            if np.isnan(val) or val <= 0.0:
                break
            area += 0.5 * (prev_val + float(val)) * (L - prev_lag)
            prev_lag, prev_val = float(L), float(val)
        tau_int[c] = 1.0 + 2.0 * area
    return {
        "lags": lags,
        "r": [[None if np.isnan(v) else float(v) for v in row] for row in r],
        "lag1": [None if np.isnan(v) else float(v) for v in r[0]],
        "tau_e": [None if np.isnan(v) else float(v) for v in tau_e],
        "tau_int": [None if np.isnan(v) else float(v) for v in tau_int],
        "censored": int(np.sum(tau_e >= lags[-1])),
    }


@torch.no_grad()
def decode_r2(state: torch.Tensor, target: np.ndarray, lag: int) -> list[float]:
    """Per cell: the share of `target`'s variance a linear read of `state` explains.

    `state` is `[T, cells, d]` and `target` is `[T, q]`, both in tick order. At
    `lag > 0` the state at `t` is fitted against the target at `t + lag`, which
    asks what a cell's present state says about the world's near future rather
    than its present.

    This is the ticket's own question in its most literal form — *does anything
    about a cell's position predict what it represents* — with the caveat that a
    linear decode is a lower bound on content, and that `R^2` inflates by about
    `d / T` on noise. :func:`decode_null` gives that inflation empirically by
    fitting a circularly shifted target, and the two are reported together.
    """
    T = state.shape[0]
    if lag:
        x = state[: T - lag].double()
        y = torch.from_numpy(np.ascontiguousarray(target[lag:])).double()
    else:
        x = state.double()
        y = torch.from_numpy(np.ascontiguousarray(target)).double()
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    total = float(y.pow(2).sum())
    if total <= 0.0 or x.shape[0] < 4 * x.shape[2]:
        return [float("nan")] * state.shape[1]
    out = []
    for c in range(x.shape[1]):
        xc = x[:, c, :]
        if float(xc.pow(2).sum()) <= 1e-300:
            out.append(float("nan"))
            continue
        beta = torch.linalg.lstsq(xc, y).solution
        resid = float((y - xc @ beta).pow(2).sum())
        out.append(1.0 - resid / total)
    return out


@torch.no_grad()
def read_cells(agent, recorder, world: dict | None = None) -> dict:
    """Everything measured, at one checkpoint."""
    h, e, v, ticks = recorder.window()
    bl = excitation.blocks(agent)
    out = {"window_ticks": int(ticks)}
    out["autocorr_h"] = autocorrelation(h)
    out["autocorr_v"] = autocorrelation(v)
    out["autocorr_v_private"] = autocorrelation(v * bl.private.to(v.dtype).unsqueeze(0))
    out["autocorr_v_exposed"] = autocorrelation(v * bl.exposed.to(v.dtype).unsqueeze(0))

    stream = v.transpose(0, 1)  # [cells, T, n]
    ex = excitation.excitation_reads(stream, bl) if hasattr(excitation, "excitation_reads") else t0.excitation_reads(stream, bl)
    for name, val in ex.items():
        out[name] = [float(x) for x in val]

    x = stream.double()
    mu = x.mean(dim=1, keepdim=True)
    energy = x.pow(2).sum(dim=(1, 2)).clamp(min=1e-300)
    out["mean_share"] = [float(y) for y in (mu.pow(2).sum(dim=(1, 2)) * x.shape[1]) / energy]
    out["v_rms"] = [float(y) for y in x.norm(dim=-1).pow(2).mean(1).sqrt()]

    hh = h.double()
    hmu = hh.mean(dim=0, keepdim=True)
    henergy = hh.pow(2).sum(dim=(0, 2)).clamp(min=1e-300)
    out["h_mean_share"] = [
        float(y) for y in (hmu.pow(2).sum(dim=(0, 2)) * hh.shape[0]) / henergy
    ]
    out["h_pr_centred"] = [
        float(y) for y in excitation.participation_ratio(hh.transpose(0, 1), centred=True)
    ]
    out["h_pr"] = [float(y) for y in excitation.participation_ratio(hh.transpose(0, 1))]
    out["h_rms"] = [float(y) for y in hh.norm(dim=-1).pow(2).mean(0).sqrt()]
    out["e_rms"] = [float(y) for y in e.double().norm(dim=-1).pow(2).mean(0).sqrt()]
    out["emission_gain"] = emission_gain(agent)

    # The record's own timescale quantity: `rho(used)`, "the quantity timescale
    # reads (#143)" (`body.py:789`), and the whole eigenvalue-modulus spectrum
    # beside it, because ADR-0028 says a cell holds a spectrum of retention
    # constants rather than one. `tau = -1 / ln rho` is the e-folding time in
    # ticks; a radius at or above one reads `inf` and is reported as None.
    ops = agent.sheaf.operators
    rho_used = ops.radii().detach().double()
    rho_raw = ops.raw_radii().detach().double()
    moduli = torch.linalg.eigvals(ops.used().detach().double()).abs()
    moduli, _ = torch.sort(moduli, dim=-1, descending=True)
    out["rho_used"] = [float(x) for x in rho_used]
    out["rho_raw"] = [float(x) for x in rho_raw]
    out["moduli_used"] = [[float(y) for y in row] for row in moduli]

    def _tau(r: torch.Tensor) -> list:
        vals = []
        for x in r:
            x = float(x)
            vals.append(None if x <= 0.0 or x >= 1.0 else -1.0 / float(np.log(x)))
        return vals

    out["tau_spectral"] = _tau(rho_used)
    out["tau_spectral_raw"] = _tau(rho_raw)

    # What the cell represents, as far as a linear read can say. `puck` is the
    # one target no cell is ever handed: it reaches the graph only through the
    # rendered image, so decoding it is the nearest thing this rig has to an
    # abstraction reading. `proprio` and `touch` are written straight onto rim
    # boundary cells, so they are the low-level comparison.
    if world:
        rows = min(len(world["ticks"]), h.shape[0])
        for name, cols in world["groups"].items():
            block = np.asarray(world["values"])[-rows:][:, cols]
            for lag in (0, 50):
                out[f"decode_{name}_h_lag{lag}"] = decode_r2(h[-rows:], block, lag)
                out[f"decode_{name}_v_lag{lag}"] = decode_r2(v[-rows:], block, lag)
            shifted = np.roll(block, block.shape[0] // 2, axis=0)
            out[f"decode_{name}_h_null"] = decode_r2(h[-rows:], shifted, 0)
            out[f"decode_{name}_v_null"] = decode_r2(v[-rows:], shifted, 0)
            # Whether the target moved at all over the window. A target that
            # barely moves makes every `R^2` above it noise, and the readout is
            # not entitled to quote one without this beside it.
            out[f"world_std_{name}"] = [float(x) for x in block.std(axis=0)]
    return out


# -- the run ------------------------------------------------------------------


def _world_groups(env) -> dict[str, list[int]]:
    """Column blocks of the world row: proprioception, touch, and the pucks.

    `proprio` and `touch` are written straight onto rim boundary cells every
    tick. `puck` is **privileged** — `PlanarPushSandbox.puck_pose` is the demo's
    accessor, and the pucks reach the graph only through the rendered image — so
    it is the one target that has to be inferred rather than read.
    """
    pucks = len(getattr(env, "_puck_qadr", []))
    groups = {"proprio": list(range(0, 6)), "touch": list(range(6, 9))}
    groups["puck"] = list(range(9, 9 + 3 * pucks))
    return {k: v for k, v in groups.items() if v}


def _world_row(env, outcome) -> list[float]:
    obs = outcome.observation
    row = list(map(float, obs["qpos"])) + list(map(float, obs["qvel"]))
    row += list(map(float, obs["touch"]))
    for i in range(len(getattr(env, "_puck_qadr", []))):
        row += [float(x) for x in env.puck_pose(i)]
    return row


def run_seed(arm: str, condition: str, seed: int, ticks: int, out: Path) -> dict:
    started = time.time()
    inflight = t0.stage(out)
    cond = t3.CONDITIONS[condition]
    pin, c = cond["pin"], cond["c"]
    env, agent = arms_mod.build_arm(arm, seed)
    try:
        if pin:
            _load("t1_run", _HERE.parent / "T1" / "run.py").pin_drive_edges(agent)
        record = {
            "issue": 572,
            "reading": "B23: does level predict anything a cell holds?",
            "arm": arm,
            "condition": condition,
            "seed": seed,
            "ticks": ticks,
            "window": t0.WINDOW,
            "lags": list(LAGS),
            "surface": _surface(),
            "timescale_note": (
                "No per-cell rate exists in the running architecture: "
                "patchworks.timescale.ClockDivisor is held by nothing outside "
                "its own module. Any timescale gradient here is emergent."
            ),
            "structure": structure(agent),
            "checkpoints": [],
        }
        record["world_groups"] = _world_groups(env)
        world = {"ticks": [], "values": [], "groups": record["world_groups"]}
        recorder = t0.Recorder(agent)
        bias = PredictionRule(agent.sheaf, operator_rate_ratio=c)
        transport = TransportRule(agent.sheaf)

        ladder = [cp for cp in t3.CHECKPOINTS if cp <= ticks]
        if ticks not in ladder:
            ladder.append(ticks)
        seen = 0
        for target in ladder:
            for outcome in t0.teaching_read(
                agent, target - seen, seed + seen, recorder, bias, transport
            ):
                world["ticks"].append(agent.sheaf.ticks)
                world["values"].append(_world_row(env, outcome))
                if len(world["ticks"]) > t0.WINDOW:
                    del world["ticks"][0]
                    del world["values"][0]
            seen = target
            entry = read_cells(agent, recorder, world)
            entry["ticks"] = target
            entry["elapsed_minutes"] = (time.time() - started) / 60.0
            record["checkpoints"].append(entry)
            inflight.write_text(json.dumps(record, indent=1))
            _report(record["structure"], entry, f"{arm} s{seed} @{target:>6}")
        inflight.replace(out)
        return record
    finally:
        env.close()


def _surface() -> dict:
    import subprocess

    def git(*args: str) -> str:
        try:
            return (
                subprocess.run(["git", *args], capture_output=True, check=True, cwd=_ROOT)
                .stdout.decode("utf-8", "replace")
                .strip()
            )
        except Exception:
            return "unknown"

    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(git("status", "--porcelain", "--", "src")),
        "lane_allocation": "per-edge, graph.allocate_lane_widths (#548)",
    }


def _by_level(structure: dict, values: list) -> dict[int, float]:
    levels = np.array([r["level"] for r in structure["cells"]])
    vals = np.array([np.nan if v is None else v for v in values], dtype=float)
    out = {}
    for lv in sorted(set(levels.tolist())):
        sel = vals[levels == lv]
        sel = sel[~np.isnan(sel)]
        out[int(lv)] = float(np.median(sel)) if sel.size else float("nan")
    return out


def _report(structure: dict, entry: dict, label: str) -> None:
    tau = _by_level(structure, entry["autocorr_v"]["tau_e"])
    rho = _by_level(structure, entry["rho_used"])
    pr = _by_level(structure, entry["pr_total_centred"])
    ms = _by_level(structure, entry["mean_share"])
    fmt = lambda d: " ".join(f"L{k}:{v:.3g}" for k, v in sorted(d.items()))
    print(
        f"  {label} ({entry['elapsed_minutes']:.1f} min)\n"
        f"    tau_e(v)  {fmt(tau)}\n"
        f"    pr_centred {fmt(pr)}\n"
        f"    mean_share {fmt(ms)}\n"
        f"    rho_used   {fmt(rho)}",
        flush=True,
    )


def write_structure(arm: str, seed: int) -> Path:
    """The construction half alone, for an arm whose run is already on disk.

    Cheap enough to re-take rather than re-run: nothing here depends on a tick.
    """
    env, agent = arms_mod.build_arm(arm, seed)
    try:
        out = _HERE / f"572-structure-{arm}.json"
        out.write_text(json.dumps({"arm": arm, "seed": seed, **structure(agent)}, indent=1))
        return out
    finally:
        env.close()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--structure-only",
        action="store_true",
        help="write 572-structure-<arm>.json and stop; no ticks are run",
    )
    p.add_argument("--arms", nargs="+", default=["reserve", "shipped"])
    p.add_argument("--condition", default="baseline", choices=sorted(t3.CONDITIONS))
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--ticks", type=int, default=20_000)
    args = p.parse_args()
    if args.structure_only:
        for arm in args.arms:
            print(f"  wrote {write_structure(arm, args.seeds[0]).name}", flush=True)
        return
    for arm in args.arms:
        for seed in args.seeds:
            out = _HERE / f"572-dome-{arm}-{args.condition}-seed{seed}-{args.ticks}.json"
            if out.exists():
                print(f"[B23] {out.name} already at the horizon, skipping", flush=True)
                continue
            print(f"[B23] {arm} seed {seed}, {args.ticks} ticks -> {out.name}", flush=True)
            run_seed(arm, args.condition, seed, args.ticks, out)
            print(f"  wrote {out.name}", flush=True)


if __name__ == "__main__":
    main()
