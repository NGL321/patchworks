"""T6 (#555) item 2: train [#556](#556)'s two arms and read the erosion at each.

The comparison is the point, and it is the user's own addition to #556's ruling:
the **doubling** and **reserve** arms sit at nearly the same construction ER with
a **22x difference in the `dim H⁰` floor**, so if the erosion differs between them
it is not the construction excess doing it. That is a cleaner test of
[B6](#546)'s headroom law than one arm can give.

Reads at every checkpoint, because #555 asks for them beside the ER rather than
after the fact:

* ``cap_read`` — [B9](#551)'s sign constraint. The reserve arm *narrows* `k_v`,
  which pushes incident maps together where tightening `c` pushed them apart, so
  the prediction is that it buys rank **without** raising incoherence.
* ``sigma_read`` — ADR-0032's band is a training-time projection, and at
  construction both arms lose composed ER to singular-value spread the generic
  instrument idealises away (`555-arms-departure.json`).

``sweep_c`` is **dropped**: [B9](#551) discredited it and #555's notes forbid it
for anything ruling-bearing.

**Run the arms sequentially and let each checkpoint land** — parallel long runs
trip this box's low-memory guard, and the record is written at every checkpoint.

Usage::

    PYTHONPATH=src python prototypes/cold-start/T6/trained_arms.py --arms doubling reserve
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
arms_mod = _load("t6_arms", _HERE / "arms.py")
departure = _load("t6_departure", _HERE / "departure.py")

from patchworks.graph import DEFAULT_SPEC  # noqa: E402
from patchworks.learning import PredictionRule, TransportRule  # noqa: E402


def surface() -> dict:
    """T0's `surface()`, minus the field #548 deleted.

    `DomeSpec.interior_m` is **gone** — [#548](#548) replaced the global constant
    with `allocate_lane_widths`, so `T0.run.surface()` raises on any dome built
    after that PR. The lane widths are per edge now and are recorded per arm in
    `widths`, so what belongs in the surface stamp is the two constants that are
    still constants.
    """
    stamp = {"lane_allocation": "per-edge, graph.allocate_lane_widths (#548)"}
    try:
        stamp.update(t0.surface())
    except AttributeError:
        stamp.update(
            {
                k: v
                for k, v in (
                    ("boundary_m", int(DEFAULT_SPEC.boundary_m)),
                    ("lateral_m", int(DEFAULT_SPEC.lateral_m)),
                    ("privacy_budget", int(DEFAULT_SPEC.privacy_budget)),
                )
            }
        )
        import subprocess

        def git(*args: str) -> str:
            try:
                return subprocess.run(
                    ["git", *args], capture_output=True, check=True, cwd=_ROOT
                ).stdout.decode("utf-8", "replace").strip()
            except Exception:
                return "unknown"

        stamp["commit"] = git("rev-parse", "HEAD")
        stamp["branch"] = git("rev-parse", "--abbrev-ref", "HEAD")
        stamp["dirty"] = bool(git("status", "--porcelain", "--", "src"))
    return stamp


def run_seed(arm: str, condition: str, seed: int, ticks: int, out: Path) -> dict:
    started = time.time()
    inflight = out.with_suffix(".inflight.json")
    cond = t3.CONDITIONS[condition]
    pin, c = cond["pin"], cond["c"]
    env, agent = arms_mod.build_arm(arm, seed)
    try:
        if pin:
            t1.pin_drive_edges(agent)
        dome = agent.dome
        chains = t2.rim_chains(dome)
        _, reserve_p = arms_mod.ARMS[arm]
        record = {
            "issue": 555,
            "arm": arm,
            "condition": condition,
            "reserve_p": reserve_p,
            "privacy_budget": int(dome.spec.privacy_budget),
            "conditions": {"rho1_drive_edges": bool(pin), "c_learning_rate": float(c)},
            "note": (
                "`c_learning_rate` is T3's learning-rate ratio, NOT ADR-0010's "
                "GAUGE_C. `sweep_c` is deliberately absent: B9 (#551) discredited it."
            ),
            "seed": seed,
            "ticks": ticks,
            "surface": surface(),
            "n": int(dome.shape.n),
            "privacy": arms_mod.privacy_read(dome, reserve_p),
            "widths": arms_mod.widths_read(dome, chains),
            "chains": chains,
            "checkpoints": [],
        }
        recorder = t2.EdgeRecorder(agent)
        bias = PredictionRule(agent.sheaf, operator_rate_ratio=c)
        transport = TransportRule(agent.sheaf)

        record["at_construction"] = {
            "angles": angles.read_surface(agent, chains, f"construction {arm} {condition} s{seed}"),
            "cap": t4_trained.cap_read(agent),
            "sigma": departure.sigma_read(agent),
            "generic": arms_mod.generic_at(dome, chains),
        }
        base = record["at_construction"]["angles"]["composed_er"]["median"] - 1.0
        print(
            f"  {arm} {condition} s{seed} construction: ER {base + 1:.6f} "
            f"(excess {base:.4e}) | generic "
            f"{record['at_construction']['generic']['median']:.4f} | priv tot "
            f"{record['privacy']['private_dim_total']}",
            flush=True,
        )

        ladder = [cp for cp in t3.CHECKPOINTS if cp <= ticks]
        if ticks not in ladder:
            ladder.append(ticks)
        seen = 0
        for target in ladder:
            for _ in t0.teaching_read(agent, target - seen, seed + seen, recorder, bias, transport):
                pass
            seen = target

            entry = {"ticks": target}
            entry["angles"] = angles.read_surface(agent, chains, f"{arm} {condition} s{seed} @{target}")
            entry["cap"] = t4_trained.cap_read(agent)
            entry["sigma"] = departure.sigma_read(agent)
            entry["elapsed_minutes"] = (time.time() - started) / 60.0
            record["checkpoints"].append(entry)
            inflight.write_text(json.dumps(record, indent=1))

            a = entry["angles"]
            excess = a["composed_er"]["median"] - 1.0
            print(
                f"  {arm} {condition} s{seed} @{target:>6}: ER {a['composed_er']['median']:.6f} "
                f"excess {excess:.4e} erosion x{base / max(excess, 1e-300):.1f} "
                f"| p90 {a['composed_er']['p90']:.4f} max {a['composed_er']['max']:.4f} "
                f"| lead {a['cos_leading_per_hop']['median']:.3f} "
                f"2nd {a['cos_second_per_hop']['median']:.3f} "
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
    p.add_argument("--arms", nargs="+", default=["doubling", "reserve"])
    p.add_argument("--condition", choices=sorted(t3.CONDITIONS), nargs="+", default=["baseline"])
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--ticks", type=int, default=20_000)
    args = p.parse_args()
    for arm in args.arms:
        for condition in args.condition:
            for seed in args.seeds:
                out = _HERE / f"555-{arm}-{condition}-seed{seed}-{args.ticks}.json"
                if out.exists():
                    print(f"[T6] {out.name} already at the horizon, skipping", flush=True)
                    continue
                print(f"[T6] {arm} {condition} seed {seed}, {args.ticks} ticks -> {out.name}", flush=True)
                run_seed(arm, condition, seed, args.ticks, out)
                print(f"  wrote {out.name}", flush=True)


if __name__ == "__main__":
    main()
