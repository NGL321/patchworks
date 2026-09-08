"""T6 (#577 / B28) item 4: does the transport rule's agreement hold on a *moving* world?

[B25](#574) named the test — *"does in-sample edge agreement generalise to held-out
episodes?"* — and `b28_gauge.py`'s in-process version of it **cannot answer it**, for a
reason worth recording rather than working around.

**Why the in-process evaluation is vacuous.** `PlanarPushSandbox` motion collapses within
a few hundred ticks of the start (#572's advisory; #518's own `travel_window` records the
same shape), and **`reset()` does not revive it**: the sampler is explicitly not allowed
to change the arm's pose (`sandbox/env.py`, `PLACEMENT_ATTEMPTS`), so an arm that has
stalled is still stalled after a rearrangement. Measured on the 20k surface: boundary-stalk
`std_max` reads **7e-08 with 0.0000 of components moving** in *every* evaluation window,
early and late, on all four rungs. Relative disagreement measured there is a ratio of two
numerical zeros, and says nothing about transport.

**What this script does instead.** The trained restriction maps are saved at the horizon
by `b28_gauge.py` (`577-maps-*.pt`), so they can be **transplanted into a freshly built
agent whose body has not yet stalled** and evaluated over the window where the world is
demonstrably moving. One fresh agent **per world seed**, because the stall arrives within
roughly 200 ticks of construction and a body reused across arrangements would carry it.

**What the transplant does and does not preserve.** It carries `sheaf.maps` — the object
every ruling on [#532](#532) is about — and nothing else: the chart operators `K`, the
biases and the stalks are the fresh agent's. So this asks *do the trained lanes agree on
the signal a moving body produces*, which is the question about **transport**; it is not
the same as replaying the trained agent's own trajectory, which the stall makes
unavailable at any horizon. Stated here rather than buried, because it is the difference
between this reading and the one the ticket asked for.

The control is the same fresh build with its maps left at construction, so the two differ
in exactly one thing.

Usage::

    PYTHONPATH=src python prototypes/cold-start/T6/b28_generalise.py --arm reserve_p16
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
_T0 = _HERE.parent / "T0"
for extra in (_ROOT / "prototypes" / "chart-double-duty-166", _ROOT / "benchmarks", _T0):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


t0 = _load("t0_run", _T0 / "run.py")
arms_mod = _load("t6_arms", _HERE / "arms.py")
gauge = _load("t6_b28_gauge", _HERE / "b28_gauge.py")

from patchworks.sandbox import PlanarPushSandbox  # noqa: E402
from untrained_fixed_point import IMAGE_SIZE, dome_named  # noqa: E402

EVAL_SPLITS = gauge.EVAL_SPLITS
SPLIT_OF = gauge.SPLIT_OF

#: Ticks per arrangement. The body is live for roughly the first 100-200 ticks of a
#: freshly built agent (`world std_max` 1.68 over ticks 0-100, 0.38 over 100-300), so
#: the window is kept inside that and its motion is stamped on every row.
EVAL_TICKS = 140
#: Dropped while the fresh sheaf's stalks come off their constructor zeros.
SETTLE = 10


def evaluate(arm, seed, world_seeds, trained_maps, split, ticks, settle) -> dict:
    """One rung, on a fresh body per arrangement."""
    base, _ = dome_named("real")
    unc, cen, var, moved = [], [], [], []
    interior = None
    for world_seed in world_seeds:
        env = PlanarPushSandbox(
            split=SPLIT_OF[split], image_size=IMAGE_SIZE[base.patch_grid]
        )
        build_env, agent = arms_mod.build_arm(arm, seed)
        build_env.close()
        agent.env = env
        try:
            if trained_maps is not None:
                with torch.no_grad():
                    agent.sheaf.maps.maps.copy_(trained_maps)
            if interior is None:
                interior = gauge._interior_edges(agent.dome)
            acc = gauge.EdgeAgreement(
                len(agent.dome.edges), int(agent.sheaf.maps.edge_width)
            )
            motion = gauge.Motion(agent.dome, agent.sheaf.layout)
            for i, _ in enumerate(t0.run_ticks(agent, ticks, seed=world_seed)):
                if i >= settle:
                    acc.observe(agent)
                    motion.observe(agent.sheaf)
            u, c, v = acc.read()
            unc.append(u)
            cen.append(c)
            var.append(v)
            moved.append(motion.read())
        finally:
            env.close()

    unc, cen, var = np.stack(unc), np.stack(cen), np.stack(var)
    alive = interior[None, :] & np.isfinite(cen) & (var > 1e-12)
    per_seed = np.array(
        [
            float(np.nanmean(cen[i][alive[i]])) if alive[i].any() else float("nan")
            for i in range(len(world_seeds))
        ]
    )
    flat_c = cen[alive]
    flat_u = unc[interior[None, :] & np.isfinite(unc)]
    return {
        "split": split,
        "world_seeds": [int(x) for x in world_seeds],
        "ticks_per_seed": ticks,
        "settle": settle,
        "edges_live_median": float(np.median(alive.sum(axis=1))),
        "edges_interior": int(interior.sum()),
        "rel_centred_mean": float(np.nanmean(flat_c)),
        "rel_centred_median": float(np.nanmedian(flat_c)),
        # `rel_centred = 1 - 2*cov/(var_u + var_v)`, so this is the normalised
        # covariance between the two ends' fluctuations: 1 is perfect agreement on
        # what varies, 0 is none at all, negative is anti-correlation.
        "agreement_coefficient": float(1.0 - np.nanmean(flat_c)),
        "rel_uncentred_mean": float(np.nanmean(flat_u)),
        "across_seed_sd": float(np.nanstd(per_seed)),
        "per_seed_centred_mean": [float(x) for x in per_seed],
        "world_motion_std_max_median": float(
            np.median([m.get("std_max", float("nan")) for m in moved])
        ),
        "world_motion_frac_moving_median": float(
            np.median([m.get("frac_components_moving", float("nan")) for m in moved])
        ),
        "world_motion_per_seed": moved,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", default="reserve_p16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ticks", type=int, default=20_000, help="the trained horizon to load")
    p.add_argument("--eval-ticks", type=int, default=EVAL_TICKS)
    p.add_argument("--settle", type=int, default=SETTLE)
    p.add_argument("--n-seeds", type=int, default=6)
    p.add_argument("--prefix", default="577")
    p.add_argument("--maps-prefix", default=None, help="defaults to --prefix")
    args = p.parse_args()

    mp = args.maps_prefix or args.prefix
    maps_path = _HERE / f"{mp}-maps-{args.arm}-seed{args.seed}-{args.ticks}.pt"
    if not maps_path.exists():
        raise SystemExit(f"no trained maps at {maps_path.name}; run b28_gauge.py first")
    blob = torch.load(maps_path, weights_only=False)
    trained_maps = blob["maps"]

    out = _HERE / f"{args.prefix}-transplant-{args.arm}-seed{args.seed}-{args.ticks}.json"
    if out.exists():
        print(f"[B28] {out.name} already present, skipping", flush=True)
        return
    inflight = out.with_suffix(".inflight.json")

    seen = gauge.training_world_seeds(args.seed, args.ticks)[: args.n_seeds]
    fresh = [args.seed + 7_000_000 + 101 * i for i in range(args.n_seeds)]
    record = {
        "issue": 577,
        "reading": "item 4 on a moving world, by transplanting the trained maps",
        "arm": args.arm,
        "seed": args.seed,
        "trained_ticks": args.ticks,
        "eval_ticks_per_seed": args.eval_ticks,
        "settle": args.settle,
        "seen_world_seeds": seen,
        "unseen_world_seeds": fresh,
        "caveat": (
            "The transplant carries `sheaf.maps` only; `K`, the biases and the stalks "
            "are the fresh agent's. The trained agent's own trajectory cannot be "
            "evaluated on a moving world at any horizon, because the body stalls "
            "within ~200 ticks and `reset()` does not restore the arm's pose."
        ),
        "splits": {},
    }
    started = time.time()
    for split in EVAL_SPLITS:
        world_seeds = seen if split == "train_seen" else fresh
        trained = evaluate(
            args.arm, args.seed, world_seeds, trained_maps, split, args.eval_ticks, args.settle
        )
        control = evaluate(
            args.arm, args.seed, world_seeds, None, split, args.eval_ticks, args.settle
        )
        record["splits"][split] = {
            "trained": trained,
            "untrained": control,
            "improvement_centred": control["rel_centred_mean"]
            / max(trained["rel_centred_mean"], 1e-300),
            "improvement_uncentred": control["rel_uncentred_mean"]
            / max(trained["rel_uncentred_mean"], 1e-300),
        }
        inflight.write_text(json.dumps(record, indent=1))
        print(
            f"[B28] {split:>16}: centred trained {trained['rel_centred_mean']:.5f} "
            f"(agree {trained['agreement_coefficient']:+.5f}, sd "
            f"{trained['across_seed_sd']:.5f}) untrained "
            f"{control['rel_centred_mean']:.5f} "
            f"(agree {control['agreement_coefficient']:+.5f}) | uncentred "
            f"{trained['rel_uncentred_mean']:.5f} vs {control['rel_uncentred_mean']:.5f} "
            f"| world std_max {trained['world_motion_std_max_median']:.3f} moving "
            f"{trained['world_motion_frac_moving_median']:.3f} "
            f"({(time.time() - started) / 60.0:.1f} min)",
            flush=True,
        )
    inflight.replace(out)
    print(f"[B28] wrote {out.name}", flush=True)


if __name__ == "__main__":
    main()
