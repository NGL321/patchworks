"""B44 (#610): does an O(1) return bend `τ̂`, and does `ρ(K)` move as the objective stands?

Two readings, one training run.

**Reading 1 — the return.** [B39](https://github.com/NGL321/patchworks/issues/601)
§2 argues `τ̂` is blind to a *uniform* rescale (peak-to-`1/e` against the
deviation's own peak cancels it exactly) but **not** to the ratio between the
locally-decaying part of the deviation at `c` and whatever returns around the
world loop — that ratio bends the curve's shape, which is the only thing `τ̂` can
see. B38's two arms sat at bottleneck ratios of `1.83e-12` and `8.35e-10`, eight
to twelve orders below unity, so nothing could return and the argument was never
measured.

[B42](https://github.com/NGL321/patchworks/issues/605) supplies the one object on
this map's record that puts the return at O(1): the **reserved-frame flat
bundle**, `identification` 0.0000 and `channel_return` 1.0000 at `sigma_max`
1.000. It is refused as architecture and retained as this map's **null**, which
is exactly what it is used as here.

**A 2x2, because gain and return are two things.** B39's claim is about the
*ratio*, so an arm that moves both at once cannot test it. All four
configurations are installed on the **same trained surface** — one training run,
one `K`, one body, one graph, one dome, so `world_loop(c)` is identical in every
arm and only the transport maps differ:

| arm | return | gain |
|---|---|---|
| `control` | whatever training left | whatever training left |
| `flat_unit` | **exact** (`channel_return` 1.0) | **unit** (`sigma_max` 1.0) |
| `haar_unit` | chance | **unit** |
| `flat_banded` | **exact** | ADR-0032's band (~7.7e-4) |
| `haar_banded` | chance | ADR-0032's band |

`haar_*` is `holonomy_read.flat_maps` — the same block structure and the same
isometry, drawn per edge-side instead of off a shared cell frame, so what
separates it from `flat_*` is cycle-consistency **alone**. Every arm's
`identification` / `channel_return` / `sigma_max` is *measured* on the wide local
cycles and recorded beside its `τ̂`; nothing here is asserted from B42's table.

**The falsifying end is pre-registered by the ticket**: if `τ̂` is unchanged at
`channel_return ≈ 1`, the return term is not what sets the decay and §2 of B39's
resolution is wrong.

**Reading 2 — `ρ(K)`.** `learning.py::PredictionRule.step` descends
`K ← K − η_K·∇K` every tick, ADR-0015 bands `σ_max(used) ∈ [1/ρ_K, 1]` and keeps
`ρ(K)` as *"the reported spectral quantity — it is what timescale wants"*, and
ADR-0026 records `λ = 0.99 → τ ≈ 99.5` as in-band. So the parameterisation holds
a retention lever and the rule descends on it; what is unknown is whether
one-step prediction error **presses** it. Sampled at checkpoints across the
horizon off the body's own accessors (`raw_radii`, `radii`, `norms`) — no
re-implementation.

**What is *not* read.** [B39](https://github.com/NGL321/patchworks/issues/601)
struck ADR-0026's path quantifiers, so no max-min-over-paths conduction verdict
is reported: `Trial.conduction` is carried for provenance only and the reading
taken is the **per-cell** `τ̂_c / world_loop(c)`, which B39 left untouched. No
`--file` is passed anywhere. The sandbox is not enriched.

**The horizon is stamped, not inherited.** [B38](https://github.com/NGL321/patchworks/issues/599)
found the stall horizon per-*run* and varying 13x between seeds of one arm, so
`b33.Motion` is read over this run's own last window and recorded.

**Name the surface** (#455): `reserve_p12`, `capacity_budget = 63`,
`private_reserve = 12`, which is `b33.ARM`.

Usage::

    PYTHONPATH=src python prototypes/cold-start/T6/b44_return.py train
    PYTHONPATH=src python prototypes/cold-start/T6/b44_return.py read
    PYTHONPATH=src python prototypes/cold-start/T6/b44_return.py read --arms control flat_unit
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
for extra in (_ROOT / "prototypes" / "chart-double-duty-166", _ROOT / "benchmarks"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))


def _load(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


b33 = _load("b44_b33", _HERE / "b33_coexist.py")
b29 = _load("b44_b29", _HERE / "b29_holonomy.py")

import detectability as det  # noqa: E402
import untrained_fixed_point as ufp  # noqa: E402
import loop_length as ll  # noqa: E402
import holonomy_read as hr  # noqa: E402
import construction_grading as cg  # noqa: E402
from patchworks.restriction import pair_index  # noqa: E402

#: Training horizon. B33's, and B43's — the world's own motion has stopped long
#: before it (B38 measured tick ~125 on this arm), so a longer run buys a longer
#: stall rather than a better-trained surface. Every arm shares this one run.
LEARN = 2000

#: Where `ρ(K)` is sampled. Dense early because that is where the world is still
#: moving and therefore where a pressure on retention would show.
CHECKPOINTS = (0, 1, 5, 10, 25, 50, 100, 150, 250, 500, 1000, 2000)

#: Trials per arm. The contrast is between four map configurations on **one**
#: trained surface at one seed with one rotation, so the per-trial spread is
#: reported rather than averaged away.
TRIALS = 3

ARMS = ("control", "flat_unit", "haar_unit", "flat_banded", "haar_banded")


# -- the transport configurations ---------------------------------------------


def reserved_frames(dome, seed: int) -> dict[int, torch.Tensor]:
    """One orthogonal frame per predicting cell, built **inside** the reserve.

    B42's correction of B40: a frame drawn in the ambient `n` and then truncated
    by the reserve mask has rows that are no longer orthonormal, which is what
    B40 read as the reserve breaking exactness. #556/#562 made `k_v = n − p` the
    same leading block on every incident edge, so the permitted window is one
    fixed coordinate subspace per cell — build the frame in it and the mask has
    nothing to cut.
    """
    n = dome.shape.n
    gen = torch.Generator().manual_seed(seed + 9001)
    frames: dict[int, torch.Tensor] = {}
    for cell in dome.predicting:
        width = int(dome._permitted[cell])
        a = torch.randn(width, width, generator=gen, dtype=torch.float64)
        q, r = torch.linalg.qr(a)
        q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)
        full = torch.zeros(n, n, dtype=torch.float64)
        full[:width, :width] = q
        frames[cell] = full
    return frames


def install_frames(dome, maps, frames: dict[int, torch.Tensor]) -> int:
    """Set each edge endpoint `(e, c)` to the edge's first `m_e` rows of `R_c`.

    Every edge at a cell takes rows of the *same* frame, so a hop is the top-left
    `m_out x m_in` block of `R_c R_cᵀ = I` and a cycle telescopes exactly at any
    widths. `m_e = n` is not required — B40 established that and B42 kept it.
    """
    installed = 0
    with torch.no_grad():
        for edge_id, edge in enumerate(dome.edges):
            for cell in (edge.u, edge.v):
                if cell not in frames:
                    continue
                try:
                    idx = pair_index(edge_id, cg.side_of(dome, edge_id, cell))
                except Exception:
                    continue
                block = frames[cell][: edge.m]
                t = maps.maps[idx]
                rr = min(block.shape[0], t.shape[0])
                cc = min(block.shape[1], t.shape[1])
                t[:rr, :cc] = block[:rr, :cc].to(t.dtype)
                installed += 1
    return installed


def configure(arm: str, dome, maps, trained: torch.Tensor, seed: int) -> dict:
    """Write `arm`'s transport into `maps`, starting from the trained tensor.

    The trained maps are restored first in every arm, so an arm that overwrites
    only the endpoints it can index leaves the rest as training left it rather
    than as the previous arm did.
    """
    with torch.no_grad():
        maps.maps.copy_(trained)
    if arm == "control":
        return {"arm": arm, "note": "the trained maps, untouched"}
    if arm.startswith("flat"):
        n_installed = install_frames(dome, maps, reserved_frames(dome, seed))
        info = {"arm": arm, "installed_endpoints": n_installed}
    else:
        gen = torch.Generator().manual_seed(seed + 9001)
        fm = hr.flat_maps(dome, gen)
        with torch.no_grad():
            maps.maps.copy_(fm.maps.to(maps.maps.dtype))
        info = {"arm": arm, "note": "independent isometries; no shared cell frame"}
    if arm.endswith("banded"):
        maps.project()
        info["projected"] = True
    else:
        info["projected"] = False
    return info


def surface(dome, maps, wide, label: str) -> dict:
    """`identification` / `channel_return` / `sigma_max` on the wide local cycles.

    Measured per arm rather than quoted from B42's table: the whole reading rests
    on which arm actually has an O(1) return, so that is read on the object in
    hand.
    """
    s = b29.surface_read(dome, maps, wide, label)["subsets"]["wide"]
    return {
        "identification": s["identification"]["median"],
        "channel_return": s["channel_return"]["median"],
        "sigma_max": s["sigma_max"]["median"],
        "cycles": s["cycles"],
    }


# -- reading 2: rho(K) --------------------------------------------------------


def spectra(operators) -> dict:
    """`ρ(K)`, `ρ(used)` and `σ_max(used)` over cells, as distributions.

    Off the body's own accessors. `raw_radii` is the learned parameter's radius —
    the quantity ADR-0015 calls *reported* and timescale reads — and `radii` is
    the used operator's, which is what the cell computes with after #433's
    in-forward-path band.
    """
    with torch.no_grad():
        raw = operators.raw_radii().to(torch.float64).cpu().numpy()
        used = operators.radii().to(torch.float64).cpu().numpy()
        norms = operators.norms.to(torch.float64).cpu().numpy()

    def q(a: np.ndarray) -> dict:
        a = np.asarray(a, dtype=float)
        return {
            "min": float(a.min()),
            "median": float(np.median(a)),
            "max": float(a.max()),
            "mean": float(a.mean()),
            "std": float(a.std()),
        }

    # ADR-0026's own currency: the retention constant a radius implies. `rho >= 1`
    # is reported as `inf` rather than clamped, which is the honesty the ADR asks
    # for at that face.
    with np.errstate(divide="ignore", invalid="ignore"):
        tau = np.where(
            (used > 0.0) & (used < 1.0), -1.0 / np.log(np.clip(used, 1e-300, 1.0)), np.inf
        )
    finite = tau[np.isfinite(tau)]
    return {
        "rho_raw": q(raw),
        "rho_used": q(used),
        "sigma_used": q(norms),
        "tau_from_rho_used": {
            "median": float(np.median(finite)) if finite.size else float("inf"),
            "max": float(finite.max()) if finite.size else float("inf"),
            "cells_at_or_above_one": int((used >= 1.0).sum()),
        },
        "rho_raw_per_cell": [float(x) for x in raw],
    }


def train(seed: int, learn: int, out: Path) -> dict:
    """One training run: stamp the motion horizon and `ρ(K)` across it."""
    started = time.time()
    env, agent = b33.arms_mod.build_arm(b33.ARM, seed)
    dome = agent.dome
    motion = b33.Motion(env)
    print(
        f"[B44] {b33.ARM} seed {seed}: {len(dome.cells)} cells, "
        f"{len(dome.edges)} edges; training {learn} ticks",
        flush=True,
    )
    record = {
        "issue": 610,
        "reading": "rho(K) across the training horizon, as the objective stands",
        "surface": b33.ARM,
        "seed": seed,
        "learn": learn,
        "n": int(dome.shape.n),
        "k_v": int(dome._permitted[dome.predicting[0]]),
        "rho_k_band": float(agent.sheaf.operators.rho_k),
        "checkpoints": {},
    }
    record["checkpoints"]["0"] = spectra(agent.sheaf.operators)
    wanted = set(CHECKPOINTS)
    for i, _ in enumerate(ufp.teaching(agent, learn, seed)):
        t = i + 1
        if t in wanted or t == learn:
            record["checkpoints"][str(t)] = spectra(agent.sheaf.operators)
            ck = record["checkpoints"][str(t)]
            print(
                f"  tick {t:>5}: rho(K) med {ck['rho_raw']['median']:.6f} "
                f"[{ck['rho_raw']['min']:.6f}, {ck['rho_raw']['max']:.6f}] "
                f"| rho(used) med {ck['rho_used']['median']:.6f} "
                f"| sigma(used) med {ck['sigma_used']['median']:.6f}",
                flush=True,
            )
            out.write_text(json.dumps(record, indent=1), encoding="utf-8")
        if i >= learn - b33.READ_WINDOW:
            motion.observe()
    record["motion"] = motion.read()
    record["read_window"] = len(motion.buffer)
    record["minutes"] = (time.time() - started) / 60.0

    # The movement statistic the ticket asks for: not the level of `rho`, but
    # whether the objective moved it at all.
    first = np.array(record["checkpoints"]["0"]["rho_raw_per_cell"], dtype=float)
    last = np.array(record["checkpoints"][str(learn)]["rho_raw_per_cell"], dtype=float)
    delta = np.abs(last - first)
    record["rho_movement"] = {
        "cells": int(delta.size),
        "at_construction": float(first[0]),
        "construction_is_uniform": bool(np.allclose(first, first[0])),
        "abs_delta_median": float(np.median(delta)),
        "abs_delta_max": float(delta.max()),
        "cells_moved_above_1e_6": int((delta > 1e-6).sum()),
        "cells_moved_above_1e_3": int((delta > 1e-3).sum()),
        "relative_delta_median": float(np.median(delta / max(first[0], 1e-300))),
    }
    out.write_text(json.dumps(record, indent=1), encoding="utf-8")

    # The trained surface every arm of reading 1 shares.
    torch.save(
        {
            "state": ufp.snapshot(agent.sheaf),
            "maps": agent.sheaf.maps.maps.detach().clone(),
            "operators": {
                k: v.detach().clone()
                for k, v in agent.sheaf.operators.state_dict().items()
            },
            "biases": {
                k: v.detach().clone()
                for k, v in agent.sheaf.biases.state_dict().items()
            },
        },
        _HERE / f"610-trained-seed{seed}-{learn}.pt",
    )
    m = record["rho_movement"]
    print(
        f"[B44] rho(K) at construction {m['at_construction']:.6f}; "
        f"|delta| median {m['abs_delta_median']:.3e} max {m['abs_delta_max']:.3e}; "
        f"{m['cells_moved_above_1e_3']} of {m['cells']} cells moved > 1e-3 "
        f"in {record['minutes']:.1f} min -> {out.name}",
        flush=True,
    )
    return record


# -- reading 1: tau-hat per arm -----------------------------------------------


def per_cell_ratio(
    dome, tau: np.ndarray, censored: np.ndarray, resolved: np.ndarray
) -> dict:
    """`τ̂_c / world_loop(c)` per cell — the part of ADR-0026 B39 left standing.

    The path quantifier is struck and not reported. `world_loop(c)` is identical
    in every arm here (one dome, no relay), so a difference between arms is a
    difference in `τ̂` and nothing else — which is what makes this a clean test of
    the return rather than of the measuring stick.
    """
    loops = ll.world_loops(dome).lengths
    cells = det.population(dome)
    vals, taus, cens, res = [], [], [], []
    for row, cell in enumerate(cells):
        if cell not in loops:
            continue
        vals.append(tau[row] / loops[cell])
        taus.append(tau[row])
        cens.append(bool(censored[row]))
        res.append(bool(resolved[row]))
    v = np.array(vals, dtype=float)
    t = np.array(taus, dtype=float)
    gate = np.array(res, dtype=bool)
    finite = v[np.isfinite(v)]
    gated = v[gate & np.isfinite(v)]
    gated_tau = t[gate]
    return {
        # #224's gate per cell, not just counted: 85-95% of cells read as
        # unreadable at runtime precision on this map's surfaces, so a median
        # over the whole population is largely a median over arithmetic noise.
        # The paired reading downstream is taken both ways and both are reported.
        "readable_per_cell": res,
        "gated_cells": int(gate.sum()),
        "gated_ratio_median": float(np.median(gated)) if gated.size else float("nan"),
        "gated_tau_median": (
            float(np.median(gated_tau)) if gated_tau.size else float("nan")
        ),
        "cells": int(v.size),
        "ratio_median": float(np.median(finite)) if finite.size else float("nan"),
        "ratio_mean": float(finite.mean()) if finite.size else float("nan"),
        "ratio_max": float(finite.max()) if finite.size else float("nan"),
        "ratio_above_one": int((finite >= 1.0).sum()),
        "tau_median": float(np.median(t)) if t.size else float("nan"),
        "tau_mean": float(t.mean()) if t.size else float("nan"),
        "tau_max": float(t.max()) if t.size else float("nan"),
        "tau_zero_cells": int((t == 0).sum()),
        "censored_cells": int(sum(cens)),
        "tau_per_cell": [float(x) for x in t],
    }


def read_arm(
    arm: str,
    seed: int,
    learn: int,
    trials: int,
    window: int,
    hold: int,
    out: Path,
) -> dict:
    started = time.time()
    env, agent = b33.arms_mod.build_arm(b33.ARM, seed)
    dome = agent.dome
    blob = torch.load(_HERE / f"610-trained-seed{seed}-{learn}.pt", weights_only=False)
    agent.sheaf.operators.load_state_dict(blob["operators"])
    agent.sheaf.biases.load_state_dict(blob["biases"])
    ufp.restore(agent.sheaf, blob["state"])
    trained_maps = blob["maps"].to(agent.sheaf.maps.maps.dtype)

    wide = b29.cycles_of(dome)["wide"]
    info = configure(arm, dome, agent.sheaf.maps, trained_maps, seed)
    read = surface(dome, agent.sheaf.maps, wide, arm)
    cast = det.double_precision(agent.sheaf)
    print(
        f"[B44] {arm}: ident {read['identification']:.4f}  "
        f"chan {read['channel_return']:.4f}  sigma {read['sigma_max']:.3e}  "
        f"({read['cycles']} wide cycles)",
        flush=True,
    )

    loops = ll.world_loops(dome).lengths
    record = {
        "issue": 610,
        "reading": "tau-hat against an O(1) return, on one trained surface",
        "arm": arm,
        "surface": b33.ARM,
        "seed": seed,
        "learn": learn,
        "trials": trials,
        "window": window,
        "hold": hold,
        "configuration": info,
        "cycle_surface": read,
        "double_precision_tensors": cast,
        "world_loop_median": float(np.median(list(loops.values()))),
        "world_loop_cells": len(loops),
        "b38_baseline_bottleneck_at_30k": 8.35e-10,
        "trials_out": [],
    }

    # The cohort and the rotation are fixed by the seed, not by the arm, so the
    # four arms are differenced against the same perturbation.
    picker = np.random.default_rng(seed)
    picks = det.sources(dome, "rim-to-apex", picker, trials)
    cohort = tuple(sorted(det.apex(dome)))
    record["cohort"] = [int(c) for c in cohort]
    generator = torch.Generator().manual_seed(seed + 1)

    for i in range(trials):
        observation, _info = env.reset(seed=seed * 1000 + i)
        agent.observe(observation)
        applied = np.zeros(env.action_space.shape, dtype=np.float64)
        det.hold_still(agent, observation, applied, hold)
        with torch.no_grad():
            settled = float(agent.sheaf.stalks.abs().max())
        t = det.trial(
            agent, observation, applied, picks[i], cohort, generator, window, det.PROBE
        )
        entry = {
            "trial": i,
            "source": [int(c) for c in t.source],
            "kind": t.kind,
            "state_max_after_hold": settled,
            # ADR-0021's amplitude, which carries no loop length. Recorded so the
            # arm's gain is visible beside its `τ̂`.
            "bottleneck": float(t.bottleneck),
            "target": int(t.target),
            "edge": det.name_edge(dome, int(t.edge)),
            "path_hops": len(t.path),
            "per_cell": per_cell_ratio(dome, t.tau, t.censored, t.resolved),
            "conduction_path_value_provenance_only": float(t.conduction),
            "readable_cells": int(np.sum(t.resolved)),
            "floor": float(t.floor),
            "elapsed_minutes": (time.time() - started) / 60.0,
        }
        record["trials_out"].append(entry)
        pc = entry["per_cell"]
        print(
            f"  trial {i + 1}/{trials} [{t.kind}] bottleneck {t.bottleneck:.3e} "
            f"| tau med {pc['tau_median']:.4g} max {pc['tau_max']:.4g} "
            f"| ratio med {pc['ratio_median']:.4g} >=1 {pc['ratio_above_one']} "
            f"| censored {pc['censored_cells']} readable {entry['readable_cells']}",
            flush=True,
        )
        out.write_text(json.dumps(record, indent=1), encoding="utf-8")

    taus = np.array([e["per_cell"]["tau_median"] for e in record["trials_out"]])
    ratios = np.array([e["per_cell"]["ratio_median"] for e in record["trials_out"]])
    bn = np.array([e["bottleneck"] for e in record["trials_out"]])
    record["summary"] = {
        "tau_median_of_trials": float(np.median(taus)),
        "tau_median_spread": [float(taus.min()), float(taus.max())],
        "ratio_median_of_trials": float(np.median(ratios)),
        "bottleneck_median": float(np.median(bn)),
    }
    record["minutes"] = (time.time() - started) / 60.0
    out.write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(
        f"[B44] {arm}: tau median {record['summary']['tau_median_of_trials']:.4g}, "
        f"bottleneck median {record['summary']['bottleneck_median']:.3e} "
        f"in {record['minutes']:.1f} min -> {out.name}",
        flush=True,
    )
    return record


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["train", "read"])
    p.add_argument("--arms", nargs="+", default=list(ARMS))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--learn", type=int, default=LEARN)
    p.add_argument("--trials", type=int, default=TRIALS)
    p.add_argument("--window", type=int, default=det.WINDOW)
    p.add_argument("--hold", type=int, default=det.HOLD)
    args = p.parse_args()

    if args.command == "train":
        out = _HERE / f"610-rho-seed{args.seed}-{args.learn}.json"
        if out.exists():
            print(f"[B44] {out.name} already taken, skipping", flush=True)
            return
        train(args.seed, args.learn, out)
        return

    for arm in args.arms:
        out = _HERE / f"610-tau-{arm}-seed{args.seed}-{args.learn}.json"
        if out.exists():
            print(f"[B44] {out.name} already taken, skipping", flush=True)
            continue
        read_arm(
            arm, args.seed, args.learn, args.trials, args.window, args.hold, out
        )


if __name__ == "__main__":
    main()
