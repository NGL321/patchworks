"""`gain_v`'s denominator, read per level, and against the bound that actually binds.

#142 measured `bound / true lambda_max` over all 150 predicting cells and got
5.585 after 30k ticks -- the factor the gain gives away by bounding the block
`sum_e F^T F` instead of reading it. #155 spends that slack by swapping
`rho^2 * deg(v)` for `rho^2 * c`.

#158 then found the swap is inert at level 1: `gain_v = gamma / max(sum_e m_e,
rho^2 deg(v))`, and at level 1 all 70 cells take the `max` on the *other*
argument. #182 asks the question that leaves open: is there anything to
recover there at all -- i.e. how loose is `sum_e m_e` against the block's true
`lambda_max` at level 1 specifically?

This reads the denominator per level, reporting for each level:

  * which argument of the `max` binds, per cell;
  * `bound / true lambda_max` for the bound that binds -- what #142 reported,
    but split by depth rather than averaged over the taper;
  * `sum_e m_e / true lambda_max` unconditionally, which is the level-1 number
    #182 asks for and which is *not* the same quantity wherever
    `rho^2 deg(v)` is the binder;
  * `rho^2 deg(v) / true lambda_max`, so the two bounds can be compared where
    both are defined.

Untrained needs no sandbox: the maps are drawn at construction from the dome
and a seed. `--learn N` runs both rules for N ticks on the real dome first, and
that does need mujoco -- the container is the supported execution target.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from patchworks.graph import build_graph
from patchworks.tick import Sheaf


def read(sheaf: Sheaf, dome) -> list[dict]:
    """Per predicting cell: its level, degree, the two bounds, and the truth."""
    owner = sheaf.maps.owner
    rows = []
    for cell in dome.cells:
        if cell.is_boundary:
            continue
        mine = (owner == cell.id).nonzero().flatten().tolist()
        if not mine:
            continue
        width = cell.stalk
        block = torch.zeros((width, width), dtype=sheaf.maps.maps.dtype)
        widths = 0
        for i in mine:
            e = dome.edges[i // 2]
            live = sheaf.maps.maps[i, : e.m, :width]
            block = block + live.T @ live
            widths += e.m
        true_max = max(float(torch.linalg.eigvalsh(block)[-1]), 1e-12)
        rows.append(
            {
                "level": cell.index.level,
                "degree": len(mine),
                "sum_m": float(widths),
                "rho2deg": sheaf.maps.rho**2 * len(mine),
                "true_max": true_max,
            }
        )
    return rows


def report(rows: list[dict], label: str) -> None:
    print(f"\n== {label} ==")
    print(
        f"{'level':>5} {'cells':>5} {'deg':>5} {'sum_m':>7} {'rho2deg':>8} "
        f"{'binds':>8} {'binding/true':>13} {'sum_m/true':>11} {'rho2deg/true':>13}"
    )
    levels = sorted({r["level"] for r in rows})
    for level in levels:
        here = [r for r in rows if r["level"] == level]
        binder = ["sum_m" if r["sum_m"] >= r["rho2deg"] else "rho2deg" for r in here]
        which = "sum_m" if all(b == "sum_m" for b in binder) else (
            "rho2deg" if all(b == "rho2deg" for b in binder) else "mixed"
        )
        binding = np.array(
            [max(r["sum_m"], r["rho2deg"]) / r["true_max"] for r in here]
        )
        by_sum = np.array([r["sum_m"] / r["true_max"] for r in here])
        by_deg = np.array([r["rho2deg"] / r["true_max"] for r in here])
        print(
            f"{level:>5} {len(here):>5} {np.mean([r['degree'] for r in here]):>5.2f} "
            f"{np.mean([r['sum_m'] for r in here]):>7.1f} "
            f"{np.mean([r['rho2deg'] for r in here]):>8.1f} {which:>8} "
            f"{binding.mean():>13.3f} {by_sum.mean():>11.3f} {by_deg.mean():>13.3f}"
        )
    binding_all = np.array([max(r["sum_m"], r["rho2deg"]) / r["true_max"] for r in rows])
    print(
        f"  all {len(rows)} cells: binding/true mean {binding_all.mean():.3f} "
        f"median {np.median(binding_all):.3f} min {binding_all.min():.3f}"
    )
    level1 = [r for r in rows if r["level"] == 1]
    if level1:
        s = np.array([r["sum_m"] / r["true_max"] for r in level1])
        print(
            f"  level 1 only, {len(level1)} cells, sum_m/true: mean {s.mean():.3f} "
            f"median {np.median(s):.3f} min {s.min():.3f} max {s.max():.3f}"
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--learn",
        type=int,
        default=0,
        help="run both rules this many ticks first (needs the sandbox)",
    )
    args = parser.parse_args(argv)

    dome = build_graph()
    if args.learn <= 0:
        sheaf = Sheaf(dome, generator=torch.Generator().manual_seed(args.seed))
        report(read(sheaf, dome), f"untrained, seed {args.seed}")
        return

    # The taught reading needs the world, so it needs mujoco. Reuse #142's own
    # teaching loop rather than a second copy of it.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))
    from untrained_fixed_point import build, taught  # noqa: E402

    env, agent = build("full", args.split, args.seed)
    try:
        # `taught`, not `teaching`: the latter is a generator, and calling it
        # without consuming it runs no ticks at all -- which reads as an
        # untrained graph reported under a taught heading.
        taught(agent, args.learn, args.seed)
        report(read(agent.sheaf, agent.dome), f"taught {args.learn}, seed {args.seed}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
