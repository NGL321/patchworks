"""T6 (#555) item 2: train [#540](#540)'s per-edge stack and read the erosion at its own width.

[T5](#546)'s ``trained_width.py``, with two changes:

1. the dome is built by :func:`T6.stack.build_stack` -- **per-edge** ``m_e``,
   lateral lanes at ``m = 1``, invariant at ``2n - 1`` -- rather than at an
   overridden global ``interior_m``. That is the whole point of #555: B6 could
   only move one global constant, and it **saturates** short of the stack.
2. ``sweep_c`` is **dropped**, not trimmed. [B9](#551) showed its re-projection
   overstates by up to ~1.5e7x and it cost B6's item 5 its answer; #555's notes
   forbid it for anything ruling-bearing. Nothing here is worth its cost.

Two reads are added at every checkpoint, because #555 asks for them *beside* the
ER rather than after the fact:

* :func:`T6.departure.sigma_read` -- the band on each map's active block. At
  construction the built stack loses **0.48 of composed ER** to singular-value
  spread that #540's instrument idealised away (`555-departure.json`), and
  ADR-0032's band is a *training-time* projection, so whether that spread closes
  under training decides whether construction understates the stack.
* ``cap_read`` -- [B9](#551)'s sign constraint. The stack must buy its rank
  **without** raising incoherence.

**Run the arms sequentially and let each checkpoint land** -- parallel long runs
trip this box's low-memory guard, and the record is written at every checkpoint.

Usage::

    PYTHONPATH=src python prototypes/cold-start/T6/trained_stack.py --ticks 20000
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

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
angles = _load("t4_angles", _T4 / "angles.py")
t4_trained = _load("t4_trained", _T4 / "trained.py")
stack = _load("t6_stack", _HERE / "stack.py")
departure = _load("t6_departure", _HERE / "departure.py")

from patchworks.learning import PredictionRule, TransportRule  # noqa: E402


def run_seed(condition: str, seed: int, ticks: int, rung: str, out: Path) -> dict:
    started = time.time()
    inflight = out.with_suffix(".inflight.json")
    arm = t3.CONDITIONS[condition]
    pin, c = arm["pin"], arm["c"]
    cap, lateral_m, per_edge = stack.RUNGS[rung]
    env, agent = stack.build_stack(seed, cap=cap, lateral_m=lateral_m, per_edge=per_edge)
    try:
        if pin:
            t1.pin_drive_edges(agent)
        chains = t2.rim_chains(agent.dome)
        dome = agent.dome
        record = {
            "issue": 555,
            "condition": condition,
            "rung": rung,
            "stack": {"cap": cap, "lateral_m": lateral_m, "per_edge": per_edge},
            "arm": {"rho1_drive_edges": bool(pin), "c_learning_rate": float(c)},
            "note": (
                "`c` in `arm` is T3's learning-rate ratio, NOT ADR-0010's GAUGE_C. "
                "`sweep_c` is deliberately absent: B9 (#551) discredited it."
            ),
            "seed": seed,
            "ticks": ticks,
            "surface": t0.surface(),
            "k": int(agent.sheaf.operators.shape.k),
            "n": int(dome.shape.n),
            "widths": stack.chain_widths(dome, chains),
            "invariant": stack.invariant_read(dome, cap),
            "mask": stack.width_mod.mask_read(dome),
            "chains": chains,
            "checkpoints": [],
        }
        recorder = t2.EdgeRecorder(agent)
        bias = PredictionRule(agent.sheaf, operator_rate_ratio=c)
        transport = TransportRule(agent.sheaf)

        record["at_construction"] = {
            "angles": angles.read_surface(agent, chains, f"construction {rung} {condition} s{seed}"),
            "cap": t4_trained.cap_read(agent),
            "sigma": departure.sigma_read(agent),
            "generic": stack.generic_at(dome, chains),
        }
        base = record["at_construction"]["angles"]["composed_er"]["median"] - 1.0
        print(
            f"  {rung} {condition} s{seed} construction: ER med {base + 1:.6f} "
            f"(excess {base:.4e}) | generic {record['at_construction']['generic']['median']:.4f}",
            flush=True,
        )

        ladder = [cp for cp in t3.CHECKPOINTS if cp <= ticks]
        if ticks not in ladder:
            ladder.append(ticks)
        seen = 0
        for target in ladder:
            window = target - seen
            for _ in t0.teaching_read(agent, window, seed + seen, recorder, bias, transport):
                pass
            seen = target

            entry = {"ticks": target}
            entry["angles"] = angles.read_surface(
                agent, chains, f"{rung} {condition} s{seed} @{target}"
            )
            entry["cap"] = t4_trained.cap_read(agent)
            entry["sigma"] = departure.sigma_read(agent)
            entry["elapsed_minutes"] = (time.time() - started) / 60.0
            record["checkpoints"].append(entry)
            inflight.write_text(json.dumps(record, indent=1))

            a = entry["angles"]
            excess = a["composed_er"]["median"] - 1.0
            print(
                f"  {rung} {condition} s{seed} @{target:>6}: ER med "
                f"{a['composed_er']['median']:.6f} excess {excess:.4e} "
                f"erosion x{base / max(excess, 1e-300):.1f} | max {a['composed_er']['max']:.4f} "
                f"| cos lead {a['cos_leading_per_hop']['median']:.3f} "
                f"second {a['cos_second_per_hop']['median']:.3f} "
                f"| sigma {entry['sigma']['sigma_min_over_max_median']:.4f} "
                f"| cap p90 {entry['cap']['ratio_p90']:.4f} at-cap {entry['cap']['cells_at_cap']} "
                f"({entry['elapsed_minutes']:.1f} min)",
                flush=True,
            )
        inflight.replace(out)
        return record
    finally:
        env.close()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--condition", choices=sorted(t3.CONDITIONS), nargs="+", default=["baseline"])
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--rung", nargs="+", default=["b_invariant"])
    p.add_argument("--ticks", type=int, default=20_000)
    args = p.parse_args()
    for rung in args.rung:
        for condition in args.condition:
            for seed in args.seeds:
                out = _HERE / f"555-{rung}-{condition}-seed{seed}-{args.ticks}.json"
                if out.exists():
                    print(f"[T6] {out.name} already at the horizon, skipping", flush=True)
                    continue
                print(
                    f"[T6] {condition} {rung} seed {seed}, {args.ticks} ticks -> {out.name}",
                    flush=True,
                )
                run_seed(condition, seed, args.ticks, rung, out)
                print(f"  wrote {out.name}", flush=True)


if __name__ == "__main__":
    main()
