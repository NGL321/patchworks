"""The per-level disagreement floor, derived at construction (ticket #158).

`02-tick-semantics.md` bounds reconciliation with `gain_v x floor < margin_v`,
and #155 priced its raise against that bound only to find the bound resolves to
a number nobody has ever measured. This script measures it.

**What "the floor" is, in the units the bound is written in.** The bound guards
a cell's *operating point*: a standing offset on the reconciled component of a
node stalk shifts `encode`'s input, and a large enough shift carries the cell
across a fold into a region with a different spectral radius (ADR-0007). The
offset reconciliation actually applies is

    offset_v  =  gain_v * || sum_{e in v} F_ev^T (F_ev x_v - y_e(t-1)) ||

and `tick.py`'s message-passing phase computes exactly that vector, multiplies
it by `gain_v`, and subtracts it. So `floor` is the norm of that sum **before**
the gain -- the un-gained delta -- and nothing here reconstructs it: the sheaf
keeps both terms of the difference (`broadcast`, `incoming`), so the number read
is the one the run actually applied.

**Why a construction-time read is the right read, and what it is not.** ADR-0007
divides disagreement into three floors and one residue. The *lag* floor drains
at rest, so the hold below removes it. The *settling* floor is a product of the
bias rule's parameter drift, and no rule runs here, so it is absent by
construction. What is left in the hold is the *static* floor -- curvature, rank,
scale ratio, self-intersection -- plus **model error**, the part learning is
supposed to remove. An untrained read therefore measures

    static floor  <=  what this reports  =  static floor + model error

which is the conservative direction for a *stability* bound (the offset is real
whether or not learning could later remove it) and the wrong direction for a
claim that the floor is *high*. The verdict says which side each threshold
needs.

Four subcommands::

    python prototypes/disagreement-floor/floor.py hold
    python prototypes/disagreement-floor/floor.py jitter
    python prototypes/disagreement-floor/floor.py sweep
    python prototypes/disagreement-floor/floor.py optimum

**`hold`** settles an untrained agent, then holds the world still and reports
the un-gained delta per level as it drains -- the lag floor going, the static
floor staying -- ending in the per-level floor and its verdict.

**`jitter`** is the sanity check the ticket asks for: the same quantity's
tick-to-tick variation with the sandbox *live*. A floor under the rig's own
noise is not a floor.

**`sweep`** repeats the hold across configurations and seeds, because a static
floor is positional and one pose reports on one point of the overlap
(ADR-0007's own protocol).

**`optimum`** is the structural control. The whole-graph minimum energy is the
disagreement no node-stalk assignment can clear -- but the *gradient* of that
energy at its own minimiser is zero, so a large irreducible energy need not
produce any standing offset at all. This measures both at one configuration and
says whether the offset tracks the energy or the distance from the minimiser.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_BENCH = str(Path(__file__).resolve().parents[2] / "benchmarks")
if _BENCH not in sys.path:
    sys.path.append(_BENCH)
from untrained_fixed_point import (  # noqa: E402
    build,
    hold,
    restore,
    snapshot,
    taught,
    teaching,
)

from patchworks.agent import Agent, run  # noqa: E402
from patchworks.diagnostics import Condition, Diagnostics  # noqa: E402
from patchworks.tick import Sheaf  # noqa: E402

#: #155's precondition-1 caps on `gamma x floor`, set by the apex cell. `POST`
#: is post-#157, where `step` is linear and the fold margin is read from
#: `encode` alone; `PRE` is today's body and a valid lower bound on it.
CAP_POST = 0.3278
CAP_PRE = 0.2600

#: The full denominator swap `rho^2 deg(v)` -> `rho^2 c` is worth this, from
#: #142's measured mean `bound / true lambda_max`.
FULL_FACTOR = 5.585

#: The floor at which the full factor is still recoverable, `CAP_POST / 5.585`.
BREAK_EVEN = CAP_POST / FULL_FACTOR

#: Ticks to settle before the hold. #120 measured zero travel from 5000 on.
SETTLE = 5000

#: Ticks of quiescent hold. The lag floor drains under a gain of order 1/16, so
#: a few hundred ticks is many time constants.
HOLD = 400


def settle(agent: Agent, ticks: int, seed: int):
    """Run to the fixed point; hand back the last tick's external write.

    `untrained_fixed_point.settle` reports the pose and the travel and drops the
    write itself, and the write is what the hold needs -- holding the world
    still means writing *that* observation every tick.
    """
    if ticks < 1:
        raise ValueError(
            f"there is no fixed point without ticks to settle it; got {ticks}"
        )
    outcome = None
    for outcome in run(agent, ticks, seed=seed):
        pass
    return outcome.observation, outcome.applied


def ungained_delta(sheaf: Sheaf) -> torch.Tensor:
    """`[components]`: the vector reconciliation scaled by `gain_v` and subtracted.

    Read off `broadcast` and `incoming`, which the message-passing phase leaves
    behind, rather than recomputed from the stalks -- the stalks have already
    been edited by the very step this is the delta of, so a recomputation would
    report the *next* step's delta and quietly answer a different question.
    """
    contribution = sheaf.maps.spread(sheaf.broadcast - sheaf.incoming)
    delta = torch.zeros_like(sheaf.stalks)
    delta.index_add_(
        0, sheaf.layout.pair_positions.reshape(-1), contribution.reshape(-1)
    )
    return delta


def per_cell_floor(agent: Agent) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(floor, offset, level)` over the predicting cells.

    `floor` is the un-gained delta's norm on that cell's own node stalk;
    `offset` is `gain_v * floor`, the standing shift on the operating point,
    which is the side of the bound that faces the fold margin.
    """
    delta = ungained_delta(agent.sheaf)
    layout = agent.sheaf.layout
    gain = agent.sheaf.gain
    floors, offsets, levels = [], [], []
    for cell in agent.dome.cells:
        if cell.is_boundary:
            continue
        norm = float(torch.linalg.vector_norm(delta[layout.slice(cell.id)]))
        floors.append(norm)
        offsets.append(norm * float(gain[cell.id]))
        levels.append(cell.index.level)
    return np.array(floors), np.array(offsets), np.array(levels)


def recoverable(floor: float, cap: float = CAP_POST) -> float:
    """#155's factor as a function of the floor: `min(5.585, cap / (gamma*floor))`."""
    if floor <= 0:
        return FULL_FACTOR
    return min(FULL_FACTOR, cap / floor)


def level_table(floors: np.ndarray, levels: np.ndarray, label: str) -> None:
    print(f"  {label}")
    print("    level  cells    median       p95       max   gamma_cap  recoverable")
    for level in sorted(set(levels.tolist())):
        here = floors[levels == level]
        median = float(np.median(here))
        gamma_cap = min(1.0, CAP_POST / median) if median > 0 else 1.0
        print(
            f"    {level:>5}  {here.size:>5}  {median:8.5f}  "
            f"{float(np.percentile(here, 95)):8.5f}  {float(here.max()):8.5f}  "
            f"{gamma_cap:9.3f}  {recoverable(median):11.3f}"
        )
    median = float(np.median(floors))
    print(
        f"    ALL    {floors.size:>5}  {median:8.5f}  "
        f"{float(np.percentile(floors, 95)):8.5f}  {float(floors.max()):8.5f}"
    )


def verdict(floors: np.ndarray, levels: np.ndarray) -> None:
    """The three-way ruling #158 asks for, per level and not only at the apex."""
    print("\n  Verdict against #155's thresholds (post-#157, cap = 0.3278):")
    print(f"    break-even for the full {FULL_FACTOR}x: floor <= {BREAK_EVEN:.4f}")
    print(f"    bound saturated at gamma = 1.0: floor >= {CAP_POST}")
    for level in sorted(set(levels.tolist())):
        median = float(np.median(floors[levels == level]))
        if median <= BREAK_EVEN:
            call = f"full {FULL_FACTOR}x recoverable"
        elif median < CAP_POST:
            call = f"partial, {recoverable(median):.2f}x"
        else:
            call = "OVER BUDGET at gamma = 1.0"
        print(f"    level {level}: floor {median:.5f} -> {call}")

    # The medians say what the taper does; `gamma` is global and one cell caps
    # it. Reported separately because the two answers can and do differ -- a
    # level whose median clears the break-even can still hold the cell that
    # puts `gamma` on the floor.
    worst = float(floors.max())
    print(
        f"\n    tightest cell: floor {worst:.5f} -> gamma_cap "
        f"{min(1.0, CAP_POST / worst) if worst > 0 else 1.0:.4f}"
    )
    print(
        "    (paired against the apex's cap, which is the global one. The exact\n"
        "     per-cell check joins each cell's floor to its *own* fold margin,\n"
        "     and #155's margins were measured on drawn candidates rather than\n"
        "     on the built cells, so that join is not available here.)"
    )


# -- hold ------------------------------------------------------------------


def command_hold(
    name: str, split: str, seed: int, ticks: int, hold_ticks: int, learn: int = 0
) -> None:
    _, agent = build(name, split, seed)
    if learn:
        outcome = taught(agent, learn, seed)
        observation, applied = outcome.observation, outcome.applied
        how = f"{learn} ticks with both rules on"
    else:
        observation, applied = settle(agent, ticks, seed)
        how = f"{ticks} ticks, untrained"

    print(f"dome={name} split={split} seed={seed}, {how}")
    floors, offsets, levels = per_cell_floor(agent)
    level_table(floors, levels, "driven, at the fixed point (lag floor present):")

    checkpoints = sorted({1, 10, 50, hold_ticks // 4, hold_ticks // 2, hold_ticks})
    previous = 0
    for at in checkpoints:
        hold(agent, observation, applied, None, at - previous)
        previous = at
        floors, offsets, levels = per_cell_floor(agent)
        print(f"\n  after {at} ticks of quiescent hold:")
        level_table(floors, levels, "un-gained delta (the floor):")
        print(
            f"    standing offset gain_v*floor: median "
            f"{float(np.median(offsets)):.6f}  max {float(offsets.max()):.6f}"
        )
    verdict(floors, levels)


# -- jitter ----------------------------------------------------------------


def command_jitter(
    name: str, split: str, seed: int, ticks: int, window: int, impulse: float = 0.0
) -> None:
    """The sandbox's own tick-to-tick noise, in the floor's units.

    `impulse` kicks a shoulder joint before the window opens, because the
    untrained fixed point is a *locked* one (#120: zero travel from tick 5000)
    and a noise floor read on a world that is not moving is not the world's
    noise floor. Read both: the still world says what the rig's arithmetic
    noise is, the disturbed one says what a moving world adds.
    """
    _, agent = build(name, split, seed)
    settle(agent, ticks, seed)
    if impulse:
        agent.env.disturb_arm(0, impulse)

    trace, levels = [], None
    for _ in run(agent, window, seed=seed):
        floors, _, levels = per_cell_floor(agent)
        trace.append(floors)
    trace = np.stack(trace)
    jitter = np.abs(np.diff(trace, axis=0)).mean(0)

    print(f"dome={name} split={split} seed={seed}, live sandbox, {window} ticks")
    print("    level  cells   floor(med)  jitter(med)   ratio")
    for level in sorted(set(levels.tolist())):
        here = levels == level
        floor = float(np.median(trace[:, here]))
        noise = float(np.median(jitter[here]))
        ratio = floor / noise if noise > 0 else float("inf")
        print(
            f"    {level:>5}  {int(here.sum()):>5}  {floor:10.5f}  "
            f"{noise:11.5f}  {ratio:6.1f}"
        )
    floor, noise = float(np.median(trace)), float(np.median(jitter))
    print(
        f"    ALL          {floor:10.5f}  {noise:11.5f}  "
        f"{floor / noise if noise > 0 else 0:6.1f}"
    )
    print(
        "\n  A floor under the rig's own noise is not a floor. Ratios well above"
        "\n  1 mean the number survives the sanity check."
    )


# -- sweep -----------------------------------------------------------------


def command_sweep(name: str, splits, seeds, ticks: int, hold_ticks: int) -> None:
    """A static floor is positional: sweep configurations, per ADR-0007."""
    print("    split   seed   level-1  level-4     apex   global-median")
    rows = []
    for split in splits:
        for seed in seeds:
            _, agent = build(name, split, seed)
            observation, applied = settle(agent, ticks, seed)
            hold(agent, observation, applied, None, hold_ticks)
            floors, _, levels = per_cell_floor(agent)
            apex = int(levels.max())
            rows.append((floors, levels))
            print(
                f"    {split:>5}  {seed:>5}  "
                f"{float(np.median(floors[levels == 1])):8.5f} "
                f"{float(np.median(floors[levels == 4])):8.5f} "
                f"{float(np.median(floors[levels == apex])):8.5f}  "
                f"{float(np.median(floors)):12.5f}"
            )
    every = np.concatenate([f for f, _ in rows])
    levels = np.concatenate([lv for _, lv in rows])
    print()
    level_table(every, levels, "pooled across configurations:")
    verdict(every, levels)


# -- optimum ---------------------------------------------------------------


def command_optimum(
    name: str, split: str, seed: int, ticks: int, hold_ticks: int, relax: int = 200
) -> None:
    """Does the offset track the irreducible energy, or the distance from its minimiser?"""
    _, agent = build(name, split, seed)
    observation, applied = settle(agent, ticks, seed)
    hold(agent, observation, applied, None, hold_ticks)

    diagnostics = Diagnostics(agent.sheaf)
    reading = diagnostics.read(Condition.QUIESCENT, whole_graph=True)
    whole = reading.whole_graph
    floors, offsets, _ = per_cell_floor(agent)

    total = float(reading.edges.energy.sum())
    print(f"dome={name} split={split} seed={seed}, held {hold_ticks} ticks")
    print(f"  dim H0 = {whole.dim_h0}   dim H1 = {whole.dim_h1}   rank = {whole.rank}")
    print(f"  minimum achievable Dirichlet energy: {whole.minimum_energy:.6g}")
    print(f"  total edge energy now:               {total:.6g}")
    print(
        f"  ratio total / minimum:               "
        f"{total / whole.minimum_energy if whole.minimum_energy else float('inf'):.4g}"
    )
    print(f"  un-gained delta, median over cells:  {float(np.median(floors)):.6g}")
    print(f"  standing offset, median over cells:  {float(np.median(offsets)):.6g}")

    # The control. Reconciliation on its own is a convergent Jacobi descent on
    # the same energy `minimum_energy` is the minimum of, so if the offset were
    # irreducible disagreement it would survive this; if it is the body pushing
    # the operating point off the consensus every tick, it collapses the moment
    # the body stops. Nothing here is a proposed mechanism -- the body cannot be
    # switched off in a run -- it is the read that tells the two apart.
    print(
        "\n  Reconciliation alone, the inference phase suppressed "
        f"({relax} steps):"
    )
    print("    step   median delta   max delta   total energy")
    for step in range(relax + 1):
        if step:
            agent.sheaf.message_passing_phase()
        floors, _, _ = per_cell_floor(agent)
        energy = float(Diagnostics(agent.sheaf).edge_reading().energy.sum())
        if step in {0, 1, 2, 5, 10, relax // 2, relax}:
            print(
                f"    {step:>4}   {float(np.median(floors)):12.6g}   "
                f"{float(floors.max()):9.6g}   {energy:12.6g}"
            )
    print(
        "\n  The minimum achievable energy is a residual whose gradient is zero\n"
        "  at its own minimiser, so an irreducible *energy* is not by itself a\n"
        "  standing *offset*. What the offset measures is how far the running\n"
        "  operating point sits from that minimiser -- and the body puts it\n"
        "  there afresh every tick."
    )


#: Budgets the trajectory reads at. Dense early, where #158 measured the fall
#: to be fastest, and out to 100,000 -- #120's own long-run budget, and the
#: one honest number to price a trade against.
BUDGETS = (1000, 2000, 5000, 10000, 20000, 30000, 50000, 75000, 100000)


# -- trajectory ------------------------------------------------------------


def trajectory_row(agent: Agent, observation, applied, hold_ticks: int):
    """Read the floor where the surface stands, without disturbing it.

    The read costs a quiescent hold, and a hold moves the stalks -- so a
    trajectory would otherwise be measuring a surface it had itself perturbed
    at every checkpoint. `snapshot`/`restore` put back everything a tick moves;
    the rules are off during the hold and the world is never stepped, so the
    parameters and the sandbox are untouched by construction and the run
    resumes on exactly the state it paused on.
    """
    state = snapshot(agent.sheaf)
    driven, _, _ = per_cell_floor(agent)
    hold(agent, observation, applied, None, hold_ticks)
    held, offsets, levels = per_cell_floor(agent)
    restore(agent.sheaf, state)
    return driven, held, offsets, levels


def command_trajectory(
    name: str, split: str, seed: int, hold_ticks: int, budgets
) -> None:
    """Where the floor settles: one run, read at many budgets (#178).

    #158 read the floor at 30,000 ticks and recorded that it was *still
    falling* there. That leaves the apex's 0.217 -- and with it #159's whole
    trade against `gamma` -- resting on a budget rather than on an asymptote.
    The break-even for #155's full 5.585x is 0.0587, a further 3.7x down, so
    which side of it the floor settles on decides whether there is a trade to
    make at all.

    Read as repeated `hold --learn N` this would cost the *sum* of the budgets,
    because `taught` restarts from scratch each time. Read here it costs the
    largest one: the teaching generator is paused at each checkpoint, the floor
    is read off a snapshot, and the run resumes.
    """
    _, agent = build(name, split, seed)
    budgets = sorted({int(b) for b in budgets if b > 0})
    total = budgets[-1]

    print(f"dome={name} split={split} seed={seed}")
    print(
        f"one trajectory to {total} ticks, both rules on, read at "
        f"{len(budgets)} budgets"
    )
    print(f"hold of {hold_ticks} ticks at each read; #158's 30k point is the control\n")

    apex = None
    rows = []
    reached = 0
    for outcome in teaching(agent, total, seed):
        reached += 1
        if reached not in budgets:
            continue
        driven, held, offsets, levels = trajectory_row(
            agent, outcome.observation, outcome.applied, hold_ticks
        )
        apex = int(levels.max())
        rows.append((reached, driven, held, offsets, levels))
        apex_floor = float(np.median(held[levels == apex]))
        print(
            f"  {reached:>7} ticks: apex {apex_floor:8.5f}  "
            f"recoverable {recoverable(apex_floor):5.2f}x  "
            f"all-cell median {float(np.median(held)):8.5f}  driven median "
            f"{float(np.median(driven)):8.5f}",
            flush=True,
        )

    if not rows:
        raise ValueError(f"no budget in {budgets} was reached in {total} ticks")

    print("\n  The floor per level, along the trajectory (median, after the hold):")
    header = "  ".join(f"{level:>8}" for level in range(1, apex + 1))
    print(f"    {'ticks':>7}  {header}  {'ALL':>8}")
    for reached, _, held, _, levels in rows:
        cells = "  ".join(
            f"{float(np.median(held[levels == level])):8.5f}"
            for level in range(1, apex + 1)
        )
        print(f"    {reached:>7}  {cells}  {float(np.median(held)):8.5f}")

    print(
        "\n  The tail, which is what caps the global gain "
        "(#158: a few mid-depth cells, not the apex):"
    )
    # The binding level is read per row rather than fixed at #158's level 4:
    # what caps a global `gamma` is whichever cell is worst *now*, and whether
    # that stays at mid-depth as training runs on is part of what is being
    # asked. On the real dome at 30k it was level 4; nothing guarantees it
    # still is at 100k, and a hard-coded 4 would hide the move.
    print(
        f"    {'ticks':>7}  {'level':>5}  {'med':>8}  {'p95':>8}  {'max':>8}  "
        f"{'gamma_cap':>9}"
    )
    for reached, _, held, _, levels in rows:
        binding = int(levels[int(np.argmax(held))])
        here = held[levels == binding]
        worst = float(held.max())
        print(
            f"    {reached:>7}  {binding:>5}  {float(np.median(here)):8.5f}  "
            f"{float(np.percentile(here, 95)):8.5f}  {worst:8.5f}  "
            f"{(min(1.0, CAP_POST / worst) if worst > 0 else 1.0):9.4f}"
        )

    print("\n  Ratio to the previous read -- is it still falling, and how fast:")
    print(f"    {'ticks':>7}  {'apex x prev':>12}  {'ALL x prev':>11}")
    for (_, _, prev, _, prev_levels), (after, _, held, _, levels) in zip(
        rows, rows[1:]
    ):
        was = float(np.median(prev[prev_levels == apex]))
        now = float(np.median(held[levels == apex]))
        was_all, now_all = float(np.median(prev)), float(np.median(held))
        print(
            f"    {after:>7}  {(was / now if now else float('inf')):12.3f}  "
            f"{(was_all / now_all if now_all else float('inf')):11.3f}"
        )

    reached, _, held, _, levels = rows[-1]
    print(f"\n  At the last budget ({reached} ticks):")
    level_table(held, levels, "un-gained delta (the floor):")
    verdict(held, levels)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dome", default="real", choices=("small", "real"))
    common.add_argument("--split", default="train")
    common.add_argument("--seed", type=int, default=42)
    common.add_argument("--ticks", type=int, default=SETTLE)
    common.add_argument("--hold", type=int, default=HOLD, dest="hold_ticks")

    held = subparsers.add_parser("hold", parents=[common])
    held.add_argument(
        "--learn",
        type=int,
        default=0,
        help="teach for this many ticks with both rules on first, instead of "
        "settling untrained. The one read that separates the static floor "
        "from model error.",
    )
    jitter = subparsers.add_parser("jitter", parents=[common])
    jitter.add_argument("--window", type=int, default=200)
    jitter.add_argument("--impulse", type=float, default=0.0)
    sweep = subparsers.add_parser("sweep", parents=[common])
    sweep.add_argument("--splits", nargs="+", default=["train", "heldout_pair", "heldout_sector"])
    sweep.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 42])
    optimum = subparsers.add_parser("optimum", parents=[common])
    optimum.add_argument("--relax", type=int, default=200)
    trajectory = subparsers.add_parser("trajectory", parents=[common])
    trajectory.add_argument(
        "--budgets",
        nargs="+",
        type=int,
        default=list(BUDGETS),
        help="training budgets to read the floor at, on one trajectory. The "
        "largest is what the run costs; the rest are free.",
    )

    args = parser.parse_args(argv)
    if args.command == "hold":
        command_hold(
            args.dome, args.split, args.seed, args.ticks, args.hold_ticks, args.learn
        )
    elif args.command == "jitter":
        command_jitter(
            args.dome, args.split, args.seed, args.ticks, args.window, args.impulse
        )
    elif args.command == "sweep":
        command_sweep(args.dome, args.splits, args.seeds, args.ticks, args.hold_ticks)
    elif args.command == "trajectory":
        command_trajectory(
            args.dome, args.split, args.seed, args.hold_ticks, args.budgets
        )
    else:
        command_optimum(
            args.dome, args.split, args.seed, args.ticks, args.hold_ticks, args.relax
        )


if __name__ == "__main__":
    main()
