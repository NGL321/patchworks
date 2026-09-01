"""What the gain's denominator gives away, split into factors that have owners.

#142 reported one number, `bound / true lambda_max` = 5.585 taught, and #155
proposes to spend it by swapping `rho^2 deg(v)` for `rho^2 c` under an ADR-0010
incoherence term. #182 asks whether anything is recoverable at level 1, where
`Sigma_e m_e` takes the `max` instead.

One ratio cannot answer that, because three different things are inside it.
Write the block as `B = Sigma_e F_e^T F_e`. Then

    Sigma_e m_e            Sigma_e m_e        Sigma_e ||F_e||_F^2    Sigma_e sigma_max(F_e)^2
    ----------------  =  -------------------  x  ---------------------  x  ------------------------
    lambda_max(B)        Sigma ||F_e||_F^2    Sigma sigma_max^2         lambda_max(B)

  * **width**  `Sigma m_e / Sigma ||F||_F^2` -- the gauge's headroom, plus the
    gap between `Sigma m_e` and the `rho^2` bound. Not an alignment fact at all.
  * **rank**   `Sigma ||F||_F^2 / Sigma sigma_max^2` -- what a near-rank-1 map
    is charged for and does not use. #142's effective rank, per cell.
  * **spread** `Sigma sigma_max^2 / lambda_max(B)` -- the incoherence proper:
    1.0 when every incident map loads the same input direction, `deg(v)` when
    they are mutually orthogonal. **This is the only factor an incoherence term
    in the projection can create**, and it is bounded above by `deg(v)`.

The last column is the one #182 needs: it says how much of the rim's slack is
alignment that a projection could hold, and how much is already spent.
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
        widths, frob2, smax2 = 0.0, 0.0, 0.0
        for i in mine:
            e = dome.edges[i // 2]
            live = sheaf.maps.maps[i, : e.m, :width].detach()
            block = block + live.T @ live
            widths += e.m
            frob2 += float((live**2).sum())
            smax2 += float(torch.linalg.matrix_norm(live, ord=2)) ** 2
        true_max = max(float(torch.linalg.eigvalsh(block)[-1]), 1e-12)
        rows.append(
            {
                "level": cell.index.level,
                "degree": len(mine),
                "sum_m": widths,
                "rho2deg": sheaf.maps.rho**2 * len(mine),
                "frob2": frob2,
                "smax2": smax2,
                "true_max": true_max,
            }
        )
    return rows


def report(rows: list[dict], label: str) -> None:
    print(f"\n== {label} ==")
    print(
        f"{'level':>5} {'cells':>5} {'deg':>5} | {'sum_m/true':>10} = "
        f"{'width':>7} x {'rank':>6} x {'spread':>7} | {'spread/deg':>10} "
        f"{'headroom':>9}"
    )
    for level in sorted({r["level"] for r in rows}):
        here = [r for r in rows if r["level"] == level]
        deg = np.mean([r["degree"] for r in here])
        total = np.mean([r["sum_m"] / r["true_max"] for r in here])
        width = np.mean([r["sum_m"] / r["frob2"] for r in here])
        rank = np.mean([r["frob2"] / r["smax2"] for r in here])
        spread = np.mean([r["smax2"] / r["true_max"] for r in here])
        # How much of the available incoherence is already taken, and what a
        # projection holding the maps mutually orthogonal could still add.
        frac = np.mean([(r["smax2"] / r["true_max"]) / r["degree"] for r in here])
        headroom = np.mean([r["degree"] / (r["smax2"] / r["true_max"]) for r in here])
        print(
            f"{level:>5} {len(here):>5} {deg:>5.2f} | {total:>10.2f} = "
            f"{width:>7.2f} x {rank:>6.2f} x {spread:>7.2f} | {frac:>10.2%} "
            f"{headroom:>9.2f}x"
        )
    print(
        "  width = gauge headroom + the sum_m/rho^2 gap; rank = charged-for and "
        "unused;\n  spread = the incoherence, 1.0 aligned to deg(v) orthogonal. "
        "headroom = deg/spread,\n  the factor a perfect incoherence term could "
        "still add on top of what is there."
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", default="train")
    parser.add_argument("--learn", type=int, default=0)
    args = parser.parse_args(argv)

    dome = build_graph()
    if args.learn <= 0:
        sheaf = Sheaf(dome, generator=torch.Generator().manual_seed(args.seed))
        report(read(sheaf, dome), f"untrained, seed {args.seed}")
        return

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))
    from untrained_fixed_point import build, taught  # noqa: E402

    env, agent = build("full", args.split, args.seed)
    try:
        taught(agent, args.learn, args.seed)
        report(read(agent.sheaf, agent.dome), f"taught {args.learn}, seed {args.seed}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
