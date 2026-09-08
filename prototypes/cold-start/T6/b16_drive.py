"""T6 (#564 / B16): what the concentration drive is a function of.

[B16](#564) asks why training drives composed rank *below* the surface's own
generic prediction, given [B13](#560) showed `p` dials the decay away by forcing
incident lanes to share dimensions. `p = 16` removes the gap by denying training
the room to erode; the question is what training *wants*.

**The hypothesis this instrument tests.** The transport rule descends
`relative_disagreement(F_v x_v, y_e)` where `x_v` is the **cell's own node
stalk**. Whatever the objective is a function of downstream, its gradient with
respect to `F_v` factors as

    `∂L/∂F_v = (∂L/∂o) · x_vᵀ`

— **rank one, with `x_v` on the input side**. A relay cell holds *both* incident
maps and feeds *both* from that same `x_v`, so both maps' row spaces (the
carried subspaces `V_in`, `V_out` whose principal angles are #533's composed
object) accumulate their updates along the **same** traffic directions. The
leading principal-angle cosine is therefore driven up by construction of the
rule, and the composed effective rank down.

If that is the mechanism, the drive is a function of one quantity: the
**effective rank of the cell's own node-stalk traffic**. Two predictions
follow, and the readouts below are exactly them:

1. **Across cells at a fixed `k_v`** — where [B6](#546)'s headroom law is a
   constant and can predict nothing — a cell's leading cosine should rise with
   the *anisotropy* of its traffic. `traffic_er` low => `cos_leading` high.
2. Both carried subspaces should rotate **onto the traffic's own top
   direction**, not merely toward each other, and should do so from tick ~100
   (once ADR-0032's band closes) onward.

The alternatives B16 lists are recorded beside them so the ticket can rule
between them: elapsed **ticks**, the transport **objective value** itself, the
**gradient norm**, and prediction-error reduction (`bias_loss`).

`sweep_c` is not used, per [B9](#551) and #555's notes.

Usage::

    PYTHONPATH=src python prototypes/cold-start/T6/b16_drive.py --arms reserve reserve_p16 --ticks 20000
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
departure = _load("t6_departure", _HERE / "departure.py")

import construction_grading as cg  # noqa: E402
from patchworks.learning import (  # noqa: E402
    PredictionRule,
    TransportRule,
    relative_disagreement,
)

#: Checkpoints. `100` is B16's stated entry point — the surface *reaches* its
#: generic prediction once ADR-0032's band closes, then leaves it.
CHECKPOINTS = [100, 300, 1_000, 3_000, 10_000, 20_000, 50_000, 100_000]


def effective_rank_eigs(w: np.ndarray) -> float:
    """`(Σλ)² / Σλ²` on eigenvalues of a covariance — #436's quantity on a Gram."""
    w = np.clip(w.astype(np.float64), 0.0, None)
    return float(w.sum() ** 2 / max(float((w**2).sum()), 1e-300))


class TrafficRecorder:
    """Per relay cell, the second moment of its own node stalk over a window.

    The whole hypothesis lives here: this is the `x_v` that is the input-side
    factor of *every* transport update both of that cell's incident maps take.
    Restricted to the cell's shared structural mask, because that is the
    ambient the carried subspaces live in (`angles.py`'s header).
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
        self.count = 0

    @torch.no_grad()
    def observe(self) -> None:
        sh = self.agent.sheaf
        for c in self.cells:
            h = np.asarray(sh.stalks[sh.layout.slice(c)].double())[self.masks[c]]
            self.moment[c] += np.outer(h, h)
        self.count += 1

    def read(self) -> dict:
        out = {}
        for c in self.cells:
            m = self.moment[c] / max(self.count, 1)
            w, v = np.linalg.eigh(m)
            order = np.argsort(w)[::-1]
            out[c] = {"er": effective_rank_eigs(w), "top": v[:, order[0]], "power": float(w.sum())}
        return out


def _incident(dome, edge_id: int, cell_id: int) -> bool:
    try:
        cg.side_of(dome, edge_id, cell_id)
        return True
    except Exception:
        return False


@torch.no_grad()
def cell_read(dome, maps, cell: int, edge_in: int, edge_out: int, traffic: dict) -> dict:
    """One relay cell: its two carried subspaces, their angle, and the traffic."""
    _, v_in, _ = angles.carried(dome, maps, edge_in, cell)
    _, v_out, _ = angles.carried(dome, maps, edge_out, cell)
    cos = np.linalg.svd(v_out.T @ v_in, compute_uv=False)
    top = traffic["top"]
    return {
        "cell": int(cell),
        "k_v": int(v_in.shape[0]),
        "m_in": int(v_in.shape[1]),
        "m_out": int(v_out.shape[1]),
        "cos_leading": float(np.sort(cos)[::-1][0]),
        "cos_second": float(np.sort(cos)[::-1][1]) if len(cos) > 1 else 0.0,
        "traffic_er": traffic["er"],
        "traffic_power": traffic["power"],
        # Prediction 2: does each carried subspace rotate onto the traffic's own
        # top direction? `‖V ᵀ t‖` is the cosine between `t` and the subspace.
        "align_in": float(np.linalg.norm(v_in.T @ top)),
        "align_out": float(np.linalg.norm(v_out.T @ top)),
    }


def _q(a) -> dict:
    a = np.asarray(a, dtype=np.float64)
    return {
        "median": float(np.median(a)),
        "p10": float(np.quantile(a, 0.10)),
        "p90": float(np.quantile(a, 0.90)),
        "mean": float(a.mean()),
    }


def _corr(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.std() < 1e-12 or y.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


@torch.no_grad()
def rule_read(agent, transport: TransportRule) -> dict:
    """The alternatives B16 lists: the objective's own value and its gradient."""
    sh = agent.sheaf
    if sh.ticks < 2:
        return {"disagreement_median": None, "grad_norm_median": None}
    gathered, incoming = transport.inputs()
    out = transport.path(gathered)
    d = relative_disagreement(out, incoming).double().numpy()
    g = transport.gradient().double()
    gn = torch.linalg.matrix_norm(g).numpy()
    return {
        "disagreement_median": float(np.median(d)),
        "disagreement_mean": float(d.mean()),
        "grad_norm_median": float(np.median(gn)),
        "grad_norm_mean": float(gn.mean()),
    }


def run_seed(arm: str, seed: int, ticks: int, out: Path) -> dict:
    started = time.time()
    inflight = out.with_suffix(".inflight.json")
    env, agent = arms_mod.build_arm(arm, seed)
    try:
        dome = agent.dome
        chains = t2.rim_chains(dome)
        # Every (edge_in, cell, edge_out) relay triple the chains actually use.
        triples: dict[int, tuple[int, int]] = {}
        for ch in chains:
            for edge_in, cell, edge_out in cg.hops_of(dome, tuple(ch["edges"])):
                triples.setdefault(int(cell), (int(edge_in), int(edge_out)))
        cells = sorted(triples)
        _, reserve_p = arms_mod.ARMS[arm]
        record = {
            "issue": 564,
            "arm": arm,
            "reserve_p": reserve_p,
            "seed": seed,
            "ticks": ticks,
            "n": int(dome.shape.n),
            "relay_cells": len(cells),
            "hypothesis": (
                "the transport gradient is rank-one with the cell's own node stalk "
                "on the input side, and a relay cell feeds both incident maps from "
                "that one stalk, so both carried subspaces rotate onto the traffic's "
                "dominant directions and the leading principal cosine rises"
            ),
            "checkpoints": [],
        }

        recorder = t2.EdgeRecorder(agent)
        bias = PredictionRule(agent.sheaf)
        transport = TransportRule(agent.sheaf)
        traffic_rec = TrafficRecorder(agent, cells)

        def snapshot(label: str, tick: int) -> dict:
            traffic = traffic_rec.read()
            rows = [cell_read(dome, agent.sheaf.maps, c, *triples[c], traffic[c]) for c in cells]
            surf = angles.read_surface(agent, chains, f"{label} {arm} s{seed}")
            entry = {
                "ticks": tick,
                "composed_er": surf["composed_er"],
                "cos_leading_per_hop": surf["cos_leading_per_hop"],
                "sigma": departure.sigma_read(agent),
                "rule": rule_read(agent, transport),
                "cos_leading": _q([r["cos_leading"] for r in rows]),
                "traffic_er": _q([r["traffic_er"] for r in rows]),
                "align_in": _q([r["align_in"] for r in rows]),
                "align_out": _q([r["align_out"] for r in rows]),
                # Prediction 1, the discriminator: at a **fixed** `k_v` this
                # correlation is something B6's headroom law cannot produce.
                "corr_trafficer_cosleading": _corr(
                    [r["traffic_er"] for r in rows], [r["cos_leading"] for r in rows]
                ),
                "corr_trafficer_alignout": _corr(
                    [r["traffic_er"] for r in rows], [r["align_out"] for r in rows]
                ),
                "k_v_distinct": sorted({r["k_v"] for r in rows}),
                "cells": rows,
            }
            entry["elapsed_minutes"] = (time.time() - started) / 60.0
            return entry

        traffic_rec.observe()  # one frame so construction has a defined traffic read
        record["checkpoints"].append(snapshot("construction", 0))
        c0 = record["checkpoints"][0]
        print(
            f"  {arm} s{seed} construction: ER {c0['composed_er']['median']:.6f} "
            f"cos_lead {c0['cos_leading']['median']:.4f} "
            f"traffic_er {c0['traffic_er']['median']:.3f} k_v {c0['k_v_distinct']}",
            flush=True,
        )

        ladder = [c for c in CHECKPOINTS if c <= ticks]
        if ticks not in ladder:
            ladder.append(ticks)
        seen = 0
        for target in ladder:
            traffic_rec.reset()
            for _ in t0.run_ticks(agent, target - seen, seed=seed + seen):
                recorder.observe()
                traffic_rec.observe()
                bias.step()
                if agent.sheaf.ticks > 1:
                    transport.step()
            seen = target
            entry = snapshot(f"@{target}", target)
            record["checkpoints"].append(entry)
            inflight.write_text(json.dumps(record, indent=1))
            print(
                f"  {arm} s{seed} @{target:>6}: ER {entry['composed_er']['median']:.6f} "
                f"| cos_lead {entry['cos_leading']['median']:.4f} "
                f"| traffic_er {entry['traffic_er']['median']:.3f} "
                f"| align_out {entry['align_out']['median']:.4f} "
                f"| corr(traffic_er, cos_lead) {entry['corr_trafficer_cosleading']:+.3f} "
                f"| disag {entry['rule']['disagreement_median']:.5f} "
                f"| |g| {entry['rule']['grad_norm_median']:.3e} "
                f"| sigma {entry['sigma']['sigma_min_over_max_median']:.4f} "
                f"({entry['elapsed_minutes']:.1f} min)",
                flush=True,
            )
        inflight.replace(out)
        return record
    finally:
        env.close()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arms", nargs="+", default=["reserve", "reserve_p16"])
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--ticks", type=int, default=20_000)
    args = p.parse_args()
    for arm in args.arms:
        for seed in args.seeds:
            out = _HERE / f"564-drive-{arm}-seed{seed}-{args.ticks}.json"
            if out.exists():
                print(f"[T6] {out.name} already at the horizon, skipping", flush=True)
                continue
            print(f"[T6] B16 drive probe: {arm} seed {seed}, {args.ticks} ticks -> {out.name}", flush=True)
            run_seed(arm, seed, args.ticks, out)
            print(f"  wrote {out.name}", flush=True)


if __name__ == "__main__":
    main()
