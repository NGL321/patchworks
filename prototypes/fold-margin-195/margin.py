"""The fold margin, in the right space and against the new gain (ticket #195).

`02-tick-semantics.md` carries a per-cell check that a standing reconciliation
offset must not walk a cell across an activation boundary::

    gain_v x floor_v  <  margin_v

[#189](https://github.com/NGL321/patchworks/issues/189)'s dimensional pass found
it **well-formed but needlessly tight**, and
[#190](https://github.com/NGL321/patchworks/issues/190) then changed `gain_v`
out from under it. This script measures both corrections at once, because both
land on the same per-cell pass and doing them separately means paying for the
100,000-tick surface twice.

**Defect 1 -- the wrong space.** `bias_selection._fold_margin` reads Hanin &
Rolnick's distance to the nearest region boundary as `|z_i| / ||grad z_i||`,
with the gradient taken over `encode`'s *whole* input. `encode` eats
`cat(chart, node_stalk)` in `R^k x R^n`, so that row norm is over all `k + n =
44` directions. But reconciliation displaces the **node stalk alone** -- the
message-passing phase writes `stalks`, never `charts` -- so the displacement the
check guards against lives in an `n = 32` dimensional subspace of the map's
input. The distance to a fold *along the directions the offset can actually
move* is the same numerator over the row norm restricted to the stalk block,
`encode_hidden_weight[:, k:]`. That is never smaller, so the standing check is
conservative and never unsafe; isotropically it is `sqrt(44/32) = 1.173x` tight.

**Defect 2 -- the wrong quantity.** `02`'s prose writes the bound as
`gamma x floor < margin_v`, but what multiplies the floor in the code and in the
run is `gain_v = gamma / denominator_v`, not `gamma`. While the denominator was
`max(sum_e m_e, rho^2 deg(v))` the two moved together closely enough that nobody
separated them. #190 breaks that: `gamma` is untouched and `gain_v` rises 2.50x
at the apex to 6.10x at the rim. So the check must be re-run, and this script
reports the caps under both denominators to say by how much.

**What is measured, and on what.** The record has never paired a per-cell floor
with *that cell's own* fold margin: #155 measured margins on drawn candidates
and #178 paired a per-cell floor against the apex's global cap. Both halves are
read here on the same surface at the same instant, at the 100,000-tick horizon
#178 established -- 30,000 sat near a local high and the floor wanders 3.8x with
no trend below that.

Two subcommands::

    python prototypes/fold-margin-195/margin.py surface
    python prototypes/fold-margin-195/margin.py construction

**`surface`** teaches one agent to 100,000 ticks and, at each budget, holds the
world still (ADR-0007's protocol, as #158 and #178 run it) and reads every
predicting cell's floor and both margins off the operating point the run is
standing on. It writes an `.npz` per budget so the analysis is separable from
the 35 minutes of ticks.

**`construction`** is the continuity check: the same two margins read off
`bias_selection`'s construction sweep, which is where `02`'s standing 0.3502
comes from. It says what that published number becomes in the corrected space,
without a surface run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
_BENCH = str(_ROOT / "benchmarks")
if _BENCH not in sys.path:
    sys.path.append(_BENCH)

from untrained_fixed_point import (  # noqa: E402
    build,
    hold,
    restore,
    snapshot,
    teaching,
)

from patchworks.agent import Agent  # noqa: E402
from patchworks.bias_selection import (  # noqa: E402
    DemoHorizons,
    go_no_go,
)
from patchworks.body import CellBody  # noqa: E402
from patchworks.graph import DEFAULT_SPEC, Dome, build_graph  # noqa: E402
from patchworks.tick import Sheaf  # noqa: E402

#: #178's budgets. 30,000 is the control -- #158 read the floor there, and a run
#: that does not reproduce it is not the same run.
BUDGETS = (30000, 75000, 100000)

#: Ticks to settle inside each read, as #158's protocol and #178's trajectory
#: both use. The lag floor drains under a gain of order 1/16.
HOLD_TICKS = 400

#: `rho` in ADR-0010's gauge, `||F||_F <= rho` at a predicting cell.
GAUGE_RHO = 2.0

#: #190's global incoherence constant, declared alongside `rho` and enforced by
#: the same ADR-0010 projection.
INCOHERENCE_C = 2.0

#: Where the `.npz` reads land.
OUT = Path(__file__).resolve().parent


# -- the two margins --------------------------------------------------------


def margins_in_both_spaces(
    body: CellBody, chart: torch.Tensor, node_stalk: torch.Tensor, biases
) -> tuple[np.ndarray, np.ndarray]:
    """`(full, stalk)` fold margin per cell, at the operating point given.

    Both are `min_i |z_i| / ||g_i||` over `encode`'s hidden pre-activations, and
    they differ only in which directions `g_i` is allowed to run in:

    * **full** is `bias_selection._fold_margin` exactly -- the row norm over all
      `k + n` of `encode`'s inputs, which is the distance to the fold in the
      map's own input space and the quantity the record has published;
    * **stalk** restricts the gradient to `encode_hidden_weight[:, k:]`, the
      block the node stalk enters through, which is the only block a
      reconciliation offset moves.

    Read through `body.encode_parts` rather than through a re-implementation of
    `encode`, for the reason that method's own docstring gives: a rig that
    copied the forward path would keep measuring the body it was written against
    after one was swapped in underneath it.
    """
    pre, _ = body.encode_parts(chart, node_stalk, biases)
    weight = body.encode_hidden_weight
    k = body.shape.k
    full_rows = torch.linalg.vector_norm(weight, dim=-1).clamp(min=1e-12)
    stalk_rows = torch.linalg.vector_norm(weight[:, k:], dim=-1).clamp(min=1e-12)
    full = (pre.abs() / full_rows).min(dim=-1).values
    stalk = (pre.abs() / stalk_rows).min(dim=-1).values
    return full.numpy(), stalk.numpy()


def surface_margins(agent: Agent) -> tuple[np.ndarray, np.ndarray]:
    """The two margins at the operating point the surface is standing on.

    The chart is post-inference and `evidence()` is the node stalk the *next*
    `encode` will read, so this is the region the cell is about to be in --
    which is the one a standing offset would carry it out of.
    """
    sheaf = agent.sheaf
    with torch.no_grad():
        return margins_in_both_spaces(
            sheaf.body, sheaf.charts, sheaf.evidence(), sheaf.biases
        )


def margins_over_window(
    agent: Agent, observation, applied, ticks: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """`(full_med, stalk_med, full_min, stalk_min)` over a window of `ticks`.

    **A single instantaneous read is the wrong estimator, and the construction
    rig already knows it.** The margin is `min_i |z_i| / ||g_i||` over
    `encode`'s hidden units, so it is an extreme order statistic: over a
    population of cells, at any one instant some cell has a unit sitting
    arbitrarily close to its own fold, and the reported minimum is that
    coincidence rather than a property of the surface.
    `bias_selection.measure` handles this by taking the **median along the
    trajectory** -- `torch.stack(margins).median(dim=0)` -- and the number `02`
    publishes is that median. A surface read that did not do the same would be
    comparing a per-tick minimum against a published per-cell median and calling
    the difference a finding.

    So the margin is read at every tick of the same quiescent hold the floor is
    read in. The world is held still and the rules are off, exactly as #178's
    protocol has it, so this costs no extra ticks and disturbs nothing: what
    varies across the window is the surface settling under reconciliation alone,
    which is the distribution of operating points the standing offset acts at.
    The per-cell minimum over the window is carried alongside as the
    conservative reading.
    """

    def external() -> None:
        agent.write(observation, applied)

    full_all, stalk_all = [], []
    external()
    agent.command()
    for _ in range(ticks):
        agent.sheaf.tick()
        agent.command()
        external()
        full, stalk = surface_margins(agent)
        full_all.append(full)
        stalk_all.append(stalk)
    full_arr = np.stack(full_all)
    stalk_arr = np.stack(stalk_all)
    return (
        np.median(full_arr, axis=0),
        np.median(stalk_arr, axis=0),
        full_arr.min(axis=0),
        stalk_arr.min(axis=0),
    )


# -- the floor, as #158 and #178 read it ------------------------------------


def ungained_delta(sheaf: Sheaf) -> torch.Tensor:
    """`[components]`: the vector reconciliation scaled by `gain_v` and subtracted.

    Lifted unchanged from `prototypes/disagreement-floor/floor.py`, including
    its reason for reading `broadcast` and `incoming` rather than recomputing
    from the stalks: the stalks have already been edited by the step this is the
    delta of, so a recomputation would report the *next* step's delta.
    """
    contribution = sheaf.maps.spread(sheaf.broadcast - sheaf.incoming)
    delta = torch.zeros_like(sheaf.stalks)
    delta.index_add_(
        0, sheaf.layout.pair_positions.reshape(-1), contribution.reshape(-1)
    )
    return delta


def per_cell_floor(agent: Agent) -> np.ndarray:
    """`[predicting cells]`: the un-gained delta's norm on each cell's node stalk."""
    with torch.no_grad():
        delta = ungained_delta(agent.sheaf)
        layout = agent.sheaf.layout
        floors = []
        for cell in agent.dome.cells:
            if cell.is_boundary:
                continue
            floors.append(
                float(torch.linalg.vector_norm(delta[layout.slice(cell.id)]))
            )
    return np.array(floors)


# -- the two denominators ---------------------------------------------------


def denominators(dome: Dome) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """`(old, new, degree, level)` over the predicting cells, in row order.

    **old** is what `tick.reconciliation_gain` applies today,
    `max(sum_e m_e, rho^2 deg(v))`. **new** is #190's ruling, `g_v^2 c_v` with
    `g_v = rho` at a predicting cell and

        c_v = min( deg(v), max( c, ceil(deg(v) / n_v) ) )

    The pigeonhole floor and the `deg` ceiling both matter on a graph that is
    not this one; on this dome every predicting cell has `n = 32` against a
    degree under ten, so `c_v` is the declared `c = 2` at all 150 of them and the
    new denominator is a flat `rho^2 c = 8`. That flatness is #190's point --
    `gain_v` is uniform across the interior for the first time -- so it is
    computed from the formula rather than written in as a constant, and the
    assertion below is what says the two agree.
    """
    n = dome.spec.n
    old, new, degree, level = [], [], [], []
    for cell_id in dome.predicting:
        deg = dome.degrees[cell_id]
        old.append(max(float(dome.stalk_sums[cell_id]), GAUGE_RHO**2 * deg))
        c_v = min(deg, max(INCOHERENCE_C, -(-deg // n)))
        new.append(GAUGE_RHO**2 * c_v)
        degree.append(deg)
        level.append(dome.cells[cell_id].index.level)
    return np.array(old), np.array(new), np.array(degree), np.array(level)


# -- reporting --------------------------------------------------------------


def _q(x: np.ndarray, p: float) -> float:
    return float(np.percentile(x, p))


def report_cells(
    floors: np.ndarray,
    full: np.ndarray,
    stalk: np.ndarray,
    old: np.ndarray,
    new: np.ndarray,
    level: np.ndarray,
    label: str,
) -> None:
    """The four caps, per cell -- and the binding cell under each.

    Per cell and not per level, because #190's raise is **not uniform**: the rim
    takes 6.10x and the apex 2.50x, which is the inverse of the old shape, so a
    per-level summary can hide the binding cell behind a level median that moved
    the other way.
    """
    published = full * old  # what `fold_margin_check` returns today
    space_only = stalk * old  # defect 1 alone
    gain_only = full * new  # defect 2 alone
    both = stalk * new  # the ruling's own number

    print(f"\n=== {label} ===")
    print(f"  cells {floors.size}   floor: med {np.median(floors):.5f} "
          f"p95 {_q(floors, 95):.5f} max {floors.max():.5f}")
    print(f"  margin, full  R^44: med {np.median(full):.5f} min {full.min():.5f}")
    print(f"  margin, stalk R^32: med {np.median(stalk):.5f} min {stalk.min():.5f}")
    ratio = stalk / full
    print(f"  space correction:   mean {ratio.mean():.4f} median "
          f"{np.median(ratio):.4f}  (isotropic sqrt(44/32) = 1.1726)")

    print("\n  cap on gamma x floor, and the global gamma each permits:")
    print(f"    {'variant':<34} {'min cap':>9} {'binding':>8} {'lvl':>4} "
          f"{'gamma_cap':>10}")
    for name, cap in (
        ("published (full space, old denom)", published),
        ("defect 1 only (stalk, old denom)", space_only),
        ("defect 2 only (full, new denom)", gain_only),
        ("BOTH (stalk space, new denom)", both),
    ):
        per_cell = np.where(floors > 0, cap / np.maximum(floors, 1e-30), np.inf)
        binding = int(per_cell.argmin())
        print(f"    {name:<34} {cap.min():9.5f} {binding:>8} "
              f"{int(level[binding]):>4} {min(1.0, per_cell.min()):10.4f}")

    print("\n  per-cell gamma_cap under BOTH corrections, by level:")
    print(f"    {'level':>5} {'cells':>5} {'median':>9} {'min':>9} {'below 1':>8}")
    per_cell = np.where(floors > 0, both / np.maximum(floors, 1e-30), np.inf)
    for lv in sorted(set(level.tolist())):
        here = np.minimum(per_cell[level == lv], 1.0)
        print(f"    {lv:>5} {here.size:>5} {np.median(here):9.4f} "
              f"{here.min():9.4f} {int((here < 1.0).sum()):>8}")
    allc = np.minimum(per_cell, 1.0)
    print(f"    {'ALL':>5} {allc.size:>5} {np.median(allc):9.4f} "
          f"{allc.min():9.4f} {int((allc < 1.0).sum()):>8}")

    print("\n  the ten tightest cells under BOTH corrections:")
    print(f"    {'row':>4} {'lvl':>4} {'floor':>9} {'margin_s':>9} {'denom':>6} "
          f"{'cap':>9} {'gamma_cap':>10}")
    for row in np.argsort(per_cell)[:10]:
        print(f"    {row:>4} {int(level[row]):>4} {floors[row]:9.5f} "
              f"{stalk[row]:9.5f} {new[row]:6.1f} {both[row]:9.5f} "
              f"{min(1.0, per_cell[row]):10.4f}")


def report_estimator(
    floors: np.ndarray,
    stalk_med: np.ndarray,
    stalk_min: np.ndarray,
    stalk_instant: np.ndarray,
    new: np.ndarray,
    level: np.ndarray,
    budget: int,
) -> None:
    """How much of the verdict is the estimator rather than the surface.

    Three readings of the same cells' margins over the same window -- the
    median the construction rig publishes, the per-cell minimum over the window,
    and one instantaneous read. If they disagree by orders of magnitude the
    check's verdict is a statement about an order statistic, not about the
    body.
    """
    print("\n  the margin estimator, three ways (stalk space, new denominator):")
    print(f"    {'reading':<28} {'med margin':>11} {'min margin':>11} "
          f"{'min cap':>9} {'gamma_cap':>10} {'cells < 1':>10}")
    for name, margin in (
        ("median over the hold (02's)", stalk_med),
        ("min over the hold", stalk_min),
        ("one instant", stalk_instant),
    ):
        cap = margin * new
        per_cell = np.where(floors > 0, cap / np.maximum(floors, 1e-30), np.inf)
        print(f"    {name:<28} {np.median(margin):11.5f} {margin.min():11.5f} "
              f"{cap.min():9.5f} {min(1.0, per_cell.min()):10.4f} "
              f"{int((per_cell < 1.0).sum()):>10}")

    print("\n  level medians of the cap, which is 02's systematic claim:")
    cap = stalk_med * new
    row = []
    for lv in sorted(set(level.tolist())):
        row.append(f"L{lv} {np.median(cap[level == lv]):.4f}")
    print("    " + "  ".join(row))
    print(f"    (02 records the level medians of the *margin* falling with "
          f"depth, 1.25 -> 0.52)")
    row = []
    for lv in sorted(set(level.tolist())):
        row.append(f"L{lv} {np.median(stalk_med[level == lv]):.4f}")
    print("    margin: " + "  ".join(row))
    print(f"    budget {budget}")


# -- surface ----------------------------------------------------------------


def command_surface(
    name: str, split: str, seed: int, budgets, hold_ticks: int, tag: str = ""
) -> None:
    """One run to the largest budget, read at each; both halves on one surface."""
    _, agent = build(name, split, seed)
    dome = agent.dome
    old, new, degree, level = denominators(dome)
    assert np.allclose(new, GAUGE_RHO**2 * INCOHERENCE_C), (
        "on this dome #190's c_v is the declared c at every predicting cell; "
        f"got {sorted(set(new.tolist()))}"
    )

    budgets = sorted({int(b) for b in budgets if b > 0})
    total = budgets[-1]
    print(f"dome={name} split={split} seed={seed}")
    print(f"one trajectory to {total} ticks, both rules on, read at {len(budgets)} budgets")
    print(f"hold of {hold_ticks} ticks at each read; #178's protocol, #158's control\n")
    print(f"old denominator: min {old.min():.1f} max {old.max():.1f}")
    print(f"new denominator: flat {new[0]:.1f}  -> raise {old.min()/new[0]:.2f}x "
          f"to {old.max()/new[0]:.2f}x\n")

    reached = 0
    for outcome in teaching(agent, total, seed):
        reached += 1
        if reached not in budgets:
            continue
        state = snapshot(agent.sheaf)
        # One pass: `margins_over_window` *is* the hold -- same ticks, same
        # writes, same held world -- so the floor below is read where #178 and
        # #158 read theirs, and the margin window costs nothing extra.
        full, stalk, full_min, stalk_min = margins_over_window(
            agent, outcome.observation, outcome.applied, hold_ticks
        )
        floors = per_cell_floor(agent)
        instant_full, instant_stalk = surface_margins(agent)
        restore(agent.sheaf, state)

        np.savez(
            OUT / f"surface{tag}-{reached}.npz",
            floors=floors, full=full, stalk=stalk,
            full_min=full_min, stalk_min=stalk_min,
            instant_full=instant_full, instant_stalk=instant_stalk,
            old=old, new=new, degree=degree, level=level,
        )
        report_cells(floors, full, stalk, old, new, level, f"{reached} ticks")
        report_estimator(
            floors, stalk, stalk_min, instant_stalk, new, level, reached
        )
        sys.stdout.flush()


# -- construction -----------------------------------------------------------


#: `benchmarks/timescale_selection.py`'s two readings of the demo's horizons.
#: `05-timescales.md` fixes the derivation and not the numbers, and the two
#: supported readings differ by two orders of magnitude, so both are run: which
#: one `02`'s published 0.3502 came from is not recorded, and the correction
#: this ticket makes is a ratio that should not care.
HORIZONS = {
    "onset": (3.0, 14.0),
    "duration": (3.0, 750.0),
}


def command_construction(seed: int, draws: int) -> None:
    """`02`'s published 0.3502, and what it becomes in the corrected space.

    The construction sweep is where that number comes from, so it is the one
    place the correction can be stated against the record without a surface run.
    The sweep measures a *drawn* population rather than the built cells, which
    is the limitation #178 recorded; the surface read above is what removes it.
    """
    dome = build_graph(DEFAULT_SPEC)
    old, new, _, level = denominators(dome)
    isotropic = float(np.sqrt((dome.spec.k + dome.spec.n) / dome.spec.n))
    print(f"construction sweep: seed={seed} draws={draws}")
    print(f"  isotropic space correction: sqrt({dome.spec.k + dome.spec.n}/"
          f"{dome.spec.n}) = {isotropic:.4f}x\n")
    for label, (fastest, longest) in HORIZONS.items():
        body = CellBody(dome.shape, generator=torch.Generator().manual_seed(seed))
        result = go_no_go(
            dome,
            body,
            horizons=DemoHorizons(fastest=fastest, longest=longest),
            draws=draws,
            generator=torch.Generator().manual_seed(seed),
        )
        check = result.margin_check
        # `product_cap` is `margin x old denominator`; recover the margins it used.
        published = check.product_cap.detach().numpy()
        margin_full = published / old
        print(f"  horizons={label} ({fastest}, {longest})  "
              f"on_draws={result.margin_check_on_draws}")
        print(f"    published cap on gamma x floor: {check.cap:.4f} "
              f"(02 records 0.3502)")
        print(f"    defect 1 only (corrected space): "
              f"{float((margin_full * isotropic * old).min()):.4f}")
        print(f"    defect 2 only (new denominator): "
              f"{float((margin_full * new).min()):.4f}")
        print(f"    BOTH:                            "
              f"{float((margin_full * isotropic * new).min()):.4f}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    surface = sub.add_parser("surface")
    surface.add_argument("--dome", default="real", choices=("small", "real"))
    surface.add_argument("--split", default="train")
    surface.add_argument("--seed", type=int, default=42)
    surface.add_argument("--hold", type=int, default=HOLD_TICKS, dest="hold_ticks")
    surface.add_argument("--budgets", nargs="+", type=int, default=list(BUDGETS))
    surface.add_argument("--tag", default="", help="suffix for the .npz reads")

    construction = sub.add_parser("construction")
    construction.add_argument("--seed", type=int, default=42)
    construction.add_argument("--draws", type=int, default=8192)

    args = parser.parse_args(argv)
    if args.command == "surface":
        command_surface(
            args.dome, args.split, args.seed, args.budgets, args.hold_ticks,
            args.tag,
        )
    else:
        command_construction(args.seed, args.draws)


if __name__ == "__main__":
    main()
