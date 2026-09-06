"""T5 (#530): the tables, from `530-probe.json`.

Every figure is a mean over seeds 42/43/44 with the spread (max - min over
seeds) beside it, per #481's idiom. Aggregation is stated on every table.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _agg(values):
    a = np.asarray(values, dtype=float)
    return a.mean(), a.max() - a.min()


def main() -> None:
    with open(os.path.join(HERE, "530-probe.json"), encoding="utf-8") as fh:
        data = json.load(fh)
    seeds = data["seeds"]
    any_seed = seeds[next(iter(seeds))]

    print("== the body, and where the pose sits (identical across seeds) ==")
    limits = np.array(any_seed["joint_limits"])
    built = np.array(any_seed["built_pose"])
    parked = np.array([seeds[s]["parked_pose"] for s in seeds])
    print(f"joints            : {any_seed['joint_names']}")
    print(f"limits            : {limits.tolist()}")
    print(f"pose as built     : {built.tolist()}")
    print(f"  margin to limit : {np.array(any_seed['built_margin_to_nearest_limit']).round(4).tolist()}")
    print(f"  as a fraction of half-range: "
          f"{(np.array(any_seed['built_margin_to_nearest_limit']) / ((limits[:,1]-limits[:,0])/2)).round(4).tolist()}")
    print(f"pose after {any_seed['park_ticks']} ticks of #120's constant, per seed:")
    for s in seeds:
        m = np.array(seeds[s]["parked_margin_to_nearest_limit"])
        print(f"  seed {s}: pose {np.array(seeds[s]['parked_pose']).round(4).tolist()}  "
              f"margin {m.round(5).tolist()}")
    print("\n== the lock: #120's constant held again from the parked pose ==")
    print("(100 ticks, per-tick travel; travel32 is the rig's own float32 sum, "
          "travel64 MuJoCo's register)")
    for s in seeds:
        lk = seeds[s]["locked_under_untrained_constant"]
        print(f"  seed {s}: travel32/tick {lk['travel32_per_tick']:.3e}  "
              f"travel64/tick {lk['travel64_per_tick']:.3e}  "
              f"tip path {lk['tip_path']:.3e} m")

    print("\n== travel under a directly written command ==")
    print("per-tick joint travel (rad, summed over joints), tail of the hold "
          f"(last {20} of 100 ticks); mean over seeds 42/43/44, spread = max-min")
    rows = defaultdict(list)
    for s in seeds:
        for r in seeds[s]["rows"]:
            rows[(r["pose"], r["direction"], r["amplitude"])].append(r)
    for pose in ("built", "parked"):
        print(f"\n-- from the {pose} pose --")
        dirs = sorted({k[1] for k in rows if k[0] == pose})
        amps = sorted({k[2] for k in rows if k[0] == pose})
        head = "amp".ljust(9) + "".join(d.ljust(22) for d in dirs)
        print(head)
        for amp in amps:
            line = f"{amp:<9.0e}"
            for d in dirs:
                rs = rows[(pose, d, amp)]
                m, sp = _agg([r["tail_travel64_per_tick"] for r in rs])
                line += f"{m:.2e}+-{sp:.0e}".ljust(22)
            print(line)

    print("\n== tip (effector) net displacement over the 100-tick hold, metres ==")
    for pose in ("built", "parked"):
        print(f"\n-- from the {pose} pose --")
        dirs = sorted({k[1] for k in rows if k[0] == pose})
        amps = sorted({k[2] for k in rows if k[0] == pose})
        print("amp".ljust(9) + "".join(d.ljust(22) for d in dirs))
        for amp in amps:
            line = f"{amp:<9.0e}"
            for d in dirs:
                rs = rows[(pose, d, amp)]
                m, sp = _agg([r["tip_net"] for r in rs])
                line += f"{m:.2e}+-{sp:.0e}".ljust(22)
            print(line)

    print("\n== the map's shape: command -> tip displacement, central differences ==")
    print(f"  (the built pose is the arm stretched straight, a kinematic "
          f"singularity; the generic pose is {np.array(any_seed['generic_pose']).round(3).tolist()})")
    for name in ("jacobian_built", "jacobian_generic", "jacobian_parked"):
        svs = np.array([seeds[s][name]["singular_values"] for s in seeds])
        prs = np.array([seeds[s][name]["effective_rank_participation"] for s in seeds])
        print(f"{name}: delta {any_seed[name]['delta']}, {any_seed[name]['ticks']} ticks")
        print(f"  singular values (m per unit command): mean "
              f"{svs.mean(axis=0).round(4).tolist()}  spread "
              f"{(svs.max(axis=0)-svs.min(axis=0)).round(4).tolist()}")
        print(f"  participation-ratio effective rank: {prs.mean():.3f} "
              f"+- {prs.max()-prs.min():.3f}   (max possible 2, the effector is planar)")

    print("\n== dead zone: smallest command amplitude with non-zero tail travel ==")
    for d in any_seed["dead_zone"]:
        line = f"  {d:<7}"
        for s in seeds:
            z = seeds[s]["dead_zone"][d]
            line += f" seed {s}: still<={z['largest_still']!r} moves@{z['smallest_moving']!r}  "
        print(line)

    print("\n== does a *small* constant park the arm too? 5000-tick holds from the built pose ==")
    print("(tail = last 20 ticks; margin < 0 means past the limit, i.e. on the stop)")
    for key in any_seed["long_hold_5000"]:
        t64, sp64 = _agg([seeds[s]["long_hold_5000"][key]["tail_travel64_per_tick"] for s in seeds])
        m = np.array([seeds[s]["long_hold_5000"][key]["margin_to_nearest_limit"] for s in seeds])
        print(f"  {key:<12} tail travel64/tick {t64:.3e} +- {sp64:.0e}   "
              f"margin {m.mean(axis=0).round(5).tolist()}")

    print("\n== the rank read against its own horizon (built pose, seed-mean) ==")
    grid = defaultdict(list)
    for s in seeds:
        for e in seeds[s]["jacobian_grid"]:
            grid[(e["delta"], e["ticks"])].append(e)
    print("delta   ticks   sigma_1      sigma_2      cond        eff.rank(PR)")
    for (delta, ticks), es in sorted(grid.items()):
        sv = np.array([e["singular_values"] for e in es])
        pr = np.array([e["effective_rank_participation"] for e in es])
        print(f"{delta:<8}{ticks:<8}{sv[:,0].mean():<13.4g}{sv[:,1].mean():<13.4g}"
              f"{np.median([e['condition'] for e in es]):<12.4g}"
              f"{pr.mean():.3f} +- {pr.max()-pr.min():.3f}")

    print("\n== part 4: does the graph reach the commanded block? ==")
    for name, g in data["graph_side"].items():
        print(f"\n-- {name} dome --")
        if "failed" in g:
            print(f"  failed: {g['failed']}")
            continue
        for k in (
            "cells",
            "edges",
            "ticks",
            "commanded_before_any_tick",
            "commanded_after_one_reconciliation",
            "commanded_ever_nonzero",
            "commanded_max_abs_over_run",
            "commanded_tail_mean",
            "commanded_tail_sd",
            "pose_final",
            "travel64_total",
            "travel64_per_tick_last_100",
            "hand_written_commanded",
            "read_back_as_command",
            "applied_to_arm",
            "efference_after_act",
        ):
            print(f"  {k:<36}: {g[k]}")


if __name__ == "__main__":
    main()
