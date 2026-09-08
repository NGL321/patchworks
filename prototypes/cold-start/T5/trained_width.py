"""T5 (#546): the trained composed-ER ladder on a dome **rebuilt** at a given lane width.

`#546 <https://github.com/NGL321/patchworks/issues/546>`_. T4's ``trained.py``
verbatim -- same arms, same frozen world, same checkpoint ladder, same angle
read -- with exactly two changes:

1. The dome is built at an overridden ``interior_m`` (:mod:`T5.width`'s
   ``build_at``), so the lanes are genuinely wider rather than redrawn at a fixed
   mask. That is the whole point: #546 asks whether the erosion of the excess
   over one is a fixed **fraction** or a fixed **amount**, and only a rebuild
   moves the width the erosion would have to scale with.
2. ``sweep_m`` is **not** run. #546 warns in its own words that T4's ``sweep_m``
   block does not answer this ticket -- it is a generic rebuild-at-width sweep at
   a fixed ``k_v``, not a read of a trained surface at a wider lane. ``sweep_c``
   is kept, and trimmed to ``1 2 4``, because #546 item 5 asks whether ``c = 1``'s
   horizon advantage scales with width too.

**Run the widths sequentially and let each checkpoint land.** B1 lost two arms to
this box's low-memory guard; the record is written at every checkpoint, so a kill
costs the checkpoint in flight and nothing before it.

Usage::

    PYTHONPATH=src python prototypes/cold-start/T5/trained_width.py --interior-m 6 --ticks 20000
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
width_mod = _load("t5_width", _HERE / "width.py")

from patchworks.learning import PredictionRule, TransportRule  # noqa: E402


def run_seed(condition: str, seed: int, ticks: int, interior_m: int, out: Path, c_values) -> dict:
    started = time.time()
    inflight = out.with_suffix(".inflight.json")
    arm = t3.CONDITIONS[condition]
    pin, c = arm["pin"], arm["c"]
    env, agent = width_mod.build_at(interior_m, seed)
    try:
        if pin:
            t1.pin_drive_edges(agent)
        chains = t2.rim_chains(agent.dome)
        record = {
            "issue": 546,
            "condition": condition,
            "interior_m": int(interior_m),
            "arm": {"rho1_drive_edges": bool(pin), "c_learning_rate": float(c)},
            "note": (
                "`c` in `arm` is T3's learning-rate ratio, NOT ADR-0010's GAUGE_C; "
                "the latter is what `sweep_c` sweeps."
            ),
            "seed": seed,
            "ticks": ticks,
            "surface": t0.surface(),
            "k": int(agent.sheaf.operators.shape.k),
            "n": int(agent.dome.shape.n),
            "mask": width_mod.mask_read(agent.dome),
            "chains": chains,
            "checkpoints": [],
        }
        recorder = t2.EdgeRecorder(agent)
        bias = PredictionRule(agent.sheaf, operator_rate_ratio=c)
        transport = TransportRule(agent.sheaf)

        record["at_construction"] = {
            "angles": angles.read_surface(agent, chains, f"construction m={interior_m} {condition} seed {seed}"),
            "cap": t4_trained.cap_read(agent),
        }
        base = record["at_construction"]["angles"]["composed_er"]["median"] - 1.0
        print(
            f"  m={interior_m} {condition} s{seed} construction: ER med "
            f"{base + 1:.6f} (excess {base:.4e})",
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
            entry["angles"] = angles.read_surface(agent, chains, f"m={interior_m} {condition} seed {seed} @{target}")
            entry["cap"] = t4_trained.cap_read(agent)
            entry["sweep_c"] = angles.sweep_c(agent, chains, c_values)
            entry["elapsed_minutes"] = (time.time() - started) / 60.0
            record["checkpoints"].append(entry)
            inflight.write_text(json.dumps(record, indent=1))

            a = entry["angles"]
            excess = a["composed_er"]["median"] - 1.0
            print(
                f"  m={interior_m} {condition} s{seed} @{target:>6}: ER med "
                f"{a['composed_er']['median']:.8f} excess {excess:.4e} "
                f"erosion x{base / max(excess, 1e-300):.1f} | max {a['composed_er']['max']:.4f} "
                f"| cos med {a['cos_all']['median']:.3f} "
                f"| cap max {entry['cap']['ratio_max']:.3f} at-cap {entry['cap']['cells_at_cap']} "
                f"| c1 excess {entry['sweep_c'][0]['er_median'] - 1:.3e} "
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
    p.add_argument("--interior-m", type=int, nargs="+", default=[6])
    p.add_argument("--ticks", type=int, default=20_000)
    p.add_argument("--c-values", type=int, nargs="+", default=[1, 2, 4])
    args = p.parse_args()
    for interior_m in args.interior_m:
        for condition in args.condition:
            for seed in args.seeds:
                out = _HERE / f"546-{condition}-m{interior_m}-seed{seed}-{args.ticks}.json"
                if out.exists():
                    print(f"[T5] {out.name} already at the horizon, skipping", flush=True)
                    continue
                print(f"[T5] {condition} m={interior_m} seed {seed}, {args.ticks} ticks -> {out.name}", flush=True)
                run_seed(condition, seed, args.ticks, interior_m, out, args.c_values)
                print(f"  wrote {out.name}", flush=True)


if __name__ == "__main__":
    main()
