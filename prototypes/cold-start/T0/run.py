"""T0 (#518): mechanism reads on the live surface, before any intervention.

Is the apex collapse the outer-product mechanism, what is the apex's persistent
error made of, and what does the collapse read on the surface `main` actually
has? Five pre-registered reads (`#518
<https://github.com/NGL321/patchworks/issues/518>`_), one run.

**Surface.** Full dome, `map/cold-start` at HEAD -- forward normalisation
(#466, PR #513) and `(interior_m, boundary_m) = (3, 4)` (#474). Seeds 42/43/44,
20k ticks, `read.py`'s checkpoint ladder, both rules on, no intervention. The
loop is #496's loop with a read inserted between the tick and the rule step,
which is the one moment the prediction rule's own `(prior chart, prior
evidence, prediction, target)` are all live and unedited.

**Logged per cell, per checkpoint.**

* the raw `K` and the used `K` (`CellOperators.used`), and `rho` of both --
  #496 read `rho(K)` on a surface where the stored `K` *was* the in-band
  operator, so on this one the used operator is its comparator and the raw one
  is reported beside it;
* `h_bar`, the mean `encode` output (the fused chart `K` advances) over the
  last 1,000 ticks; recomputed from `prior_charts`/`prior_evidence` by the
  body's own `encode`, which is the same arithmetic the tick ran;
* `e_bar`, the mean prediction-error vector `prediction - evidence()` over the
  same window, read where the rule reads it;
* the evidence stream split into private / interior / drive pieces
  (`excitation.Blocks`), with excitation rank per piece and the window's
  energy and variance shares.

**And one read the ticket did not name, because it is the mechanism sentence
itself at zero cost:** the prediction rule's gradient on `K` at the checkpoint
tick, its participation ratio and the alignment of its top right-singular
vector with that tick's `h`. The claim is that the update is an outer product
with row space along `h`; this reads whether it is.

Usage::

    PYTHONPATH=src python prototypes/cold-start/T0/run.py --seeds 42 --ticks 20000
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

torch.set_num_threads(1)

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
for extra in (_ROOT / "prototypes" / "chart-double-duty-166", _ROOT / "benchmarks", _HERE):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import read as rig  # noqa: E402
from excitation import Blocks, blocks, excitation_rank  # noqa: E402
from patchworks import agent as agent_module  # noqa: E402
from patchworks.agent import run as run_ticks  # noqa: E402
from patchworks.graph import CellKind  # noqa: E402
from patchworks.learning import PredictionRule, TransportRule  # noqa: E402
from untrained_fixed_point import build  # noqa: E402

CHECKPOINTS = rig.CHECKPOINTS
#: The read window: the last 1,000 ticks before a checkpoint. Checkpoints
#: earlier than that read the whole run so far, and say so in `window_ticks`.
WINDOW = 1_000
#: Consecutive non-overlapping blocks of this length carry the direction
#: stability read (P3: cos between consecutive window means). Equal to WINDOW
#: so that at a checkpoint on a block boundary the ring *is* the block.
BLOCK = 1_000


def surface() -> dict:
    def git(*args: str) -> str:
        try:
            return subprocess.run(
                ("git", *args), cwd=_ROOT, capture_output=True, check=True
            ).stdout.decode("utf-8", "replace").strip()
        except Exception:
            return "unknown"

    from patchworks.graph import DEFAULT_SPEC

    return {
        "commit": git("rev-parse", "HEAD"),
        "describe": git("log", "-1", "--format=%h %ad %s", "--date=short"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(git("status", "--porcelain", "--", "src")),
        "interior_m": int(DEFAULT_SPEC.interior_m),
        "boundary_m": int(DEFAULT_SPEC.boundary_m),
        "band": "forward normalisation in CellOperators.used (#466, PR #513)",
    }


def cell_context(agent) -> dict:
    """As #496's, verbatim: level, column, degree, private width, adjacency."""
    dome = agent.dome
    predicting = list(dome.predicting)
    drive_cells = {cid for cid, cell in enumerate(dome.cells) if cell.kind == CellKind.DRIVE}
    boundary = set(dome.boundary)
    levels, columns, degrees, drive_adj, boundary_adj = [], [], [], [], []
    for c in predicting:
        cell = dome.cells[c]
        levels.append(int(cell.index.level))
        columns.append(str(cell.index.column))
        degrees.append(int(dome.degrees[c]))
        neighbours = set(dome.neighbours(c))
        drive_adj.append(int(bool(neighbours & drive_cells)))
        boundary_adj.append(int(len(neighbours & boundary)))
    p_v = [int(x) for x in dome.private_dimensions]
    return {
        "cell_ids": [int(c) for c in predicting],
        "levels": levels,
        "columns": columns,
        "degrees": degrees,
        "p_v": p_v,
        "drive_adjacent": drive_adj,
        "boundary_adjacent": boundary_adj,
    }


def groups(context: dict) -> dict[str, list[int]]:
    """#496's groups, by column and adjacency (#181), as row indices."""
    cols, lv, drv, bnd = (
        context["columns"], context["levels"], context["drive_adjacent"], context["boundary_adjacent"]
    )
    out = {"soma": [], "vision": [], "apex": [], "core": []}
    for i, (c, l, d, b) in enumerate(zip(cols, lv, drv, bnd)):
        if c == "somatomotor" and l == 1 and b > 0:
            out["soma"].append(i)
        elif c == "vision" and l == 1:
            out["vision"].append(i)
        elif c == "core" and d:
            out["apex"].append(i)
        elif c == "core" and 3 <= l <= 6:
            out["core"].append(i)
    return out


def _cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """`[cells]`: |cos| between rows of `a` and `b`; 0 where either is zero."""
    na, nb = a.norm(dim=-1), b.norm(dim=-1)
    dot = (a * b).sum(dim=-1).abs()
    live = (na > 0) & (nb > 0)
    return torch.where(live, dot / (na * nb).clamp(min=1e-300), torch.zeros_like(dot))


class Recorder:
    """The ring of the last WINDOW ticks, and the per-BLOCK series.

    Reads `h`, `e` and `v` once per tick and holds them; every heavier
    statistic is taken at block boundaries and checkpoints only.
    """

    def __init__(self, agent) -> None:
        sheaf = agent.sheaf
        self.agent = agent
        cells, k, n = sheaf.operators.cells, sheaf.dome.shape.k, sheaf.dome.shape.n
        self.h = torch.zeros(WINDOW, cells, k)
        self.e = torch.zeros(WINDOW, cells, n)
        self.v = torch.zeros(WINDOW, cells, n)
        self.ptr = 0
        self.filled = 0
        self.tick_count = 0
        self.blocks: list[dict] = []
        self.checked = False

    @torch.no_grad()
    def observe(self) -> None:
        sheaf = self.agent.sheaf
        v = sheaf.prior_evidence
        h = sheaf.body.encode(sheaf.prior_charts, v, sheaf.biases)
        e = sheaf.prediction - sheaf.evidence()
        if not self.checked:
            # The read is where the rule reads: the prediction recomputed from
            # the very inputs the rule differentiates through equals the one
            # the tick left behind, to float tolerance.
            _adv, pred = sheaf.body.advance(h, sheaf.biases, sheaf.operators)
            if not torch.allclose(pred, sheaf.prediction, atol=1e-5, rtol=1e-4):
                raise RuntimeError("recomputed prediction differs from the tick's")
            self.checked = True
        self.h[self.ptr] = h
        self.e[self.ptr] = e
        self.v[self.ptr] = v
        self.ptr = (self.ptr + 1) % WINDOW
        self.filled = min(WINDOW, self.filled + 1)
        self.tick_count += 1
        if self.tick_count % BLOCK == 0:
            self.blocks.append(self._block())

    def window(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """The last `filled` ticks in time order: `(h, e, v, ticks)`."""
        if self.filled < WINDOW:
            sl = slice(0, self.filled)
            return self.h[sl], self.e[sl], self.v[sl], self.filled
        order = torch.arange(self.ptr, self.ptr + WINDOW) % WINDOW
        return self.h[order], self.e[order], self.v[order], WINDOW

    @torch.no_grad()
    def _block(self) -> dict:
        h, e, v, ticks = self.window()
        bl = blocks(self.agent)
        return {
            "end_tick": self.tick_count,
            "mean_h": h.mean(0),
            "mean_e": e.mean(0),
            "norm_e": e.mean(0).norm(dim=-1),
            "rms_e": e.norm(dim=-1).pow(2).mean(0).sqrt(),
            **excitation_reads(v.transpose(0, 1), bl),
        }


@torch.no_grad()
def excitation_reads(stream: torch.Tensor, bl: Blocks) -> dict:
    """`[cells, T, n]` in: excitation rank per piece, energy and variance shares."""
    pieces = bl.decompose(stream)
    out = {}
    for name, m in (("total", None), ("private", bl.private), ("exposed", bl.exposed)):
        out[f"pr_{name}"] = excitation_rank(stream, m)
        out[f"pr_{name}_centred"] = excitation_rank(stream, m, centred=True)
    out["pr_interior"] = excitation_rank(pieces["interior"])
    out["pr_interior_centred"] = excitation_rank(pieces["interior"], centred=True)
    energy_total = stream.double().pow(2).sum(dim=(1, 2)).clamp(min=1e-300)
    centred = stream.double() - stream.double().mean(dim=1, keepdim=True)
    var_total = centred.pow(2).sum(dim=(1, 2)).clamp(min=1e-300)
    for name, piece in pieces.items():
        p = piece.double()
        out[f"energy_share_{name}"] = p.pow(2).sum(dim=(1, 2)) / energy_total
        pc = p - p.mean(dim=1, keepdim=True)
        out[f"variance_share_{name}"] = pc.pow(2).sum(dim=(1, 2)) / var_total
    return out


@torch.no_grad()
def checkpoint_read(agent, recorder: Recorder, rule: PredictionRule, seed_context: dict) -> tuple[dict, dict]:
    """Everything P1-P4 need at one checkpoint. Returns `(json_entry, arrays)`."""
    sheaf = agent.sheaf
    ops = sheaf.operators
    K_raw = ops.K.detach().double()
    K_used = ops.used().detach().double()
    levels, p_v = seed_context["levels"], seed_context["p_v"]

    entry = {
        "raw": rig.read(rig.spectra(ops), levels, p_v, rig.moduli(ops), rig.nonnormality(ops)),
    }
    used_ns = SimpleNamespace(K=K_used)
    entry["used"] = rig.read(
        rig.spectra(used_ns), levels, p_v, rig.moduli(used_ns), rig.nonnormality(used_ns)
    )
    entry["raw_norms"] = [float(x) for x in ops.raw_norms.detach()]
    entry["used_norms"] = [float(x) for x in ops.norms.detach()]

    h, e, v, ticks = recorder.window()
    h_bar, e_bar = h.mean(0).double(), e.mean(0).double()
    _u, s, vh = torch.linalg.svd(K_used)
    v1, u1 = vh[:, 0, :], _u[:, :, 0]
    _ur, sr, vhr = torch.linalg.svd(K_raw)
    # Dominant eigenvector alignment, as context: the retained direction rather
    # than the most-amplified one. Complex in general; |cos| on the real part
    # is meaningless for a complex pair, so it is reported only where the top
    # eigenvalue is real.
    eig, vec = torch.linalg.eig(K_used)
    top = eig.abs().argmax(dim=-1)
    top_vec = vec[torch.arange(vec.shape[0]), :, top]
    top_real = eig[torch.arange(vec.shape[0]), top].imag.abs() < 1e-9
    eig_cos = torch.where(top_real, _cos(top_vec.real, h_bar), torch.full((K_used.shape[0],), float("nan"), dtype=torch.float64))

    bl = blocks(agent)
    e_priv = e_bar * bl.private.double()
    e_exp = e_bar * bl.exposed.double()
    e_drive_along = (e_exp * bl.drive_direction.double()).sum(-1)
    e_int_rows = torch.einsum("cij,cj->ci", bl.interior_rowspace.double(), e_bar)
    e_norm2 = e_bar.pow(2).sum(-1).clamp(min=1e-300)

    # P3: direction stability against the previous block's mean.
    blk = recorder.blocks
    if len(blk) >= 2 and blk[-1]["end_tick"] == recorder.tick_count:
        stability = _cos(blk[-1]["mean_e"].double(), blk[-2]["mean_e"].double())
        h_stability = _cos(blk[-1]["mean_h"].double(), blk[-2]["mean_h"].double())
    else:
        stability = torch.full((e_bar.shape[0],), float("nan"), dtype=torch.float64)
        h_stability = stability.clone()

    # The unregistered read: the gradient's own shape at this tick.
    g = rule.gradient()["operators.K"].detach().double()
    gs = torch.linalg.svdvals(g)
    g_pr = gs.pow(2).sum(-1).pow(2) / gs.pow(4).sum(-1).clamp(min=1e-300)
    _gu, _gs, gvh = torch.linalg.svd(g)
    h_now = recorder.h[(recorder.ptr - 1) % WINDOW].double()
    g_cos_h = _cos(gvh[:, 0, :], h_now)
    g_cos_hbar = _cos(gvh[:, 0, :], h_bar)

    ex = excitation_reads(v.transpose(0, 1), bl)

    per_cell = {
        "rho_raw": [float(x) for x in ops.raw_radii().detach()],
        "rho_used": [float(x) for x in ops.radii().detach()],
        "p1_cos_v1_hbar": [float(x) for x in _cos(v1, h_bar)],
        "cos_u1_hbar": [float(x) for x in _cos(u1, h_bar)],
        "cos_v1raw_hbar": [float(x) for x in _cos(vhr[:, 0, :], h_bar)],
        "cos_eigvec_hbar": [float(x) for x in eig_cos],
        "sigma_used": [[float(y) for y in row] for row in s],
        "hbar_norm": [float(x) for x in h_bar.norm(dim=-1)],
        "h_rms": [float(x) for x in h.double().norm(dim=-1).pow(2).mean(0).sqrt()],
        "p3_ebar_norm": [float(x) for x in e_bar.norm(dim=-1)],
        "e_rms": [float(x) for x in e.double().norm(dim=-1).pow(2).mean(0).sqrt()],
        "p3_direction_stability": [float(x) for x in stability],
        "h_direction_stability": [float(x) for x in h_stability],
        "p4_ebar_share_private": [float(x) for x in e_priv.pow(2).sum(-1) / e_norm2],
        "p4_ebar_share_exposed": [float(x) for x in e_exp.pow(2).sum(-1) / e_norm2],
        "p4_ebar_share_drive": [float(x) for x in e_drive_along.pow(2) / e_norm2],
        "p4_ebar_share_interior_rowspace": [float(x) for x in e_int_rows.pow(2).sum(-1) / e_norm2],
        "grad_pr": [float(x) for x in g_pr],
        "grad_cos_v1_h": [float(x) for x in g_cos_h],
        "grad_cos_v1_hbar": [float(x) for x in g_cos_hbar],
        "grad_norm": [float(x) for x in torch.linalg.matrix_norm(g, ord="fro")],
    }
    for name, val in ex.items():
        per_cell[name] = [float(x) for x in val]
    entry["window_ticks"] = ticks
    entry["per_cell"] = per_cell
    arrays = {
        "K_raw": K_raw.float().numpy(),
        "K_used": K_used.float().numpy(),
        "h_bar": h_bar.float().numpy(),
        "e_bar": e_bar.float().numpy(),
        "v1_used": v1.float().numpy(),
        "drive_direction": bl.drive_direction.numpy(),
        "grad_K": g.float().numpy(),
    }
    return entry, arrays


def stage(target: Path) -> Path:
    inflight = target.with_suffix(".inflight.json")
    if inflight.exists():
        index = 0
        while (kept := target.with_suffix(f".killed-{index}.json")).exists():
            index += 1
        inflight.replace(kept)
    return inflight


def teaching_read(agent, ticks: int, seed: int, recorder: Recorder, bias: PredictionRule, transport: TransportRule):
    """#496's `teaching`, with the read between the tick and the rules."""
    for outcome in run_ticks(agent, ticks, seed=seed):
        recorder.observe()
        bias.step()
        if agent.sheaf.ticks > 1:
            transport.step()
        yield outcome


def run_seed(name: str, split: str, seed: int, ticks: int, out: Path) -> dict:
    started = time.time()
    inflight = stage(out)
    npz_path = out.with_suffix(".npz")
    env, agent = build(name, split, seed)
    try:
        context = cell_context(agent)
        record = {
            "issue": 518,
            "condition": "baseline",
            "dome": name,
            "split": split,
            "seed": seed,
            "ticks": ticks,
            "surface": surface(),
            "cells": int(agent.sheaf.operators.cells),
            "k": int(agent.sheaf.operators.shape.k),
            "n": int(agent.dome.shape.n),
            "band": [1.0 / agent.sheaf.operators.rho_k, 1.0],
            "drive_assertion": float(agent_module.DRIVE_ASSERTION),
            "window": WINDOW,
            "block": BLOCK,
            "context": context,
            "groups": groups(context),
            "checkpoints": [],
        }
        recorder = Recorder(agent)
        bias = PredictionRule(agent.sheaf)
        transport = TransportRule(agent.sheaf)
        arrays: dict[str, np.ndarray] = {}

        ladder = [c for c in CHECKPOINTS if c <= ticks]
        if ticks not in ladder:
            ladder.append(ticks)
        seen = 0
        carry = None
        travel_total = 0.0
        for target in ladder:
            window = target - seen
            poses = [] if carry is None else [carry]
            for outcome in teaching_read(agent, window, seed + seen, recorder, bias, transport):
                poses.append(outcome.observation["qpos"].copy())
            pose = np.asarray(poses)
            window_travel = float(np.abs(np.diff(pose, axis=0)).sum()) if len(pose) > 1 else 0.0
            travel_total += window_travel
            carry = poses[-1]
            seen = target

            entry, arr = checkpoint_read(agent, recorder, bias, context)
            entry["ticks"] = target
            entry["travel_window"] = window_travel
            entry["travel_cumulative"] = travel_total
            entry["travel_per_tick"] = window_travel / max(1, window)
            record["checkpoints"].append(entry)
            record["elapsed_minutes"] = (time.time() - started) / 60.0
            for key, val in arr.items():
                arrays[f"t{target}_{key}"] = val
            # The block series, so P3's stability and the excitation-rank
            # trajectory can be read at every 1,000 ticks, not only at checkpoints.
            record["blocks"] = [
                {
                    "end_tick": b["end_tick"],
                    **{
                        key: [float(x) for x in b[key]]
                        for key in b
                        if key not in ("end_tick", "mean_h", "mean_e")
                    },
                }
                for b in recorder.blocks
            ]
            if recorder.blocks:
                arrays["block_end_ticks"] = np.array([b["end_tick"] for b in recorder.blocks])
                arrays["block_mean_e"] = np.stack([b["mean_e"].numpy() for b in recorder.blocks])
                arrays["block_mean_h"] = np.stack([b["mean_h"].numpy() for b in recorder.blocks])

            inflight.write_text(json.dumps(record, indent=1))
            np.savez_compressed(npz_path, **arrays)

            g = record["groups"]
            pc = entry["per_cell"]
            rho = np.array(pc["rho_used"])
            print(
                f"  seed {seed} @ {target:>6}: rho_used apex {np.median(rho[g['apex']]):.3f} "
                f"core {np.median(rho[g['core']]):.3f} vis {np.median(rho[g['vision']]):.3f} "
                f"soma {np.median(rho[g['soma']]):.3f} | P1 apex {np.mean(np.array(pc['p1_cos_v1_hbar'])[g['apex']]):.3f} "
                f"core {np.mean(np.array(pc['p1_cos_v1_hbar'])[g['core']]):.3f} | "
                f"|e| apex {np.median(np.array(pc['p3_ebar_norm'])[g['apex']]):.2e} | "
                f"PR apex {np.median(np.array(pc['pr_total'])[g['apex']]):.2f} vis {np.median(np.array(pc['pr_total'])[g['vision']]):.2f} "
                f"({record['elapsed_minutes']:.1f} min)",
                flush=True,
            )
        inflight.replace(out)
        return record
    finally:
        env.close()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--ticks", type=int, default=20_000)
    p.add_argument("--dome", default="real")
    p.add_argument("--split", default="train")
    p.add_argument("--tag", default="")
    args = p.parse_args()
    for seed in args.seeds:
        tag = f"-{args.tag}" if args.tag else ""
        out = _HERE / f"518-baseline{tag}-{args.dome}-{args.split}-seed{seed}-{args.ticks}.json"
        print(f"[T0] seed {seed}, {args.ticks} ticks -> {out.name}", flush=True)
        run_seed(args.dome, args.split, seed, args.ticks, out)
        print(f"  wrote {out.name}", flush=True)


if __name__ == "__main__":
    main()
