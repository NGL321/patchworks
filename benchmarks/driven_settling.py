"""What drains and what fails to settle on a driven run (#351, for #324 and #329).

Two of the open-problem register's deferred bars are readings of the same driven
run, and running it twice would pay for it twice::

    python benchmarks/driven_settling.py read
    python benchmarks/driven_settling.py read --dome small --ticks 200 --no-file

**[#324](https://github.com/NGL321/patchworks/issues/324): disagreement draining
under drive, taking the instrument with it.** Per-edge Dirichlet energy falling
while per-edge effective rank — the participation ratio `(Σσᵢ²)² / Σσᵢ⁴` —
slides toward 1 across the fleet. The instrument is not owed by this ticket:
:meth:`patchworks.diagnostics.Diagnostics.edge_reading` is exactly this pair,
and it is the *only* way to obtain either half, because neither means anything
alone. What was owed is a **bar**, and the bar is a count.

`draining_effective_rank < 2` — *the median map on a draining edge has fewer
than two effective directions*. The participation ratio reads 1 for a rank-1
map and `m` for a uniform one, so 2 is not a level anybody chose: it is the
integer that separates *one* direction from *more than one*, and #324's failure
is stated as a slide toward 1. #156's discipline holds — a count, not a tuned
threshold.

**The conjunction lives in the population, not in a second clause.** The register
writes a threshold as one `<metric> <comparator> <number>`, so *energy falling*
cannot be a second comparison; it is the set the median is taken over. An edge is
draining if its energy in the closing window is below its energy in the opening
one, and the rank is read on the maps of those edges. **If no edge drains the
reading is `m`**, the largest value the ratio can take — the failure absent reads
maximally healthy, and never as a low number nobody looked at.

**A crossing is only a slide if the fleet was somewhere else to begin with**, so
the same population's rank in the *opening* window is reported beside it and
offered as `draining_effective_rank_opening`. #156's fourth trap is a bar that
fires on the architecture working, and a fleet already near rank-1 when the run
began has not slid anywhere; a verdict whose baseline lives in someone's shell
history is a verdict nobody can check. On `DEFAULT_SPEC` the maps are drawn at a
median effective rank of about 3.6, so there is real distance to fall.

**#91's float32 note needs no answer here**, and that is worth saying rather than
rediscovering: the instrument already takes the ratio on the unit-normalised map
in float64, precisely so that a scale-invariant quantity cannot come back `inf`
or be propped up by an epsilon. `diagnostics.edge_reading` argues it at length.
This rig reports the transmitting share alongside, so a fleet whose maps have
gone to zero is visible as that rather than as a collapse.

**[#329](https://github.com/NGL321/patchworks/issues/329): a learned `K` that
never settles.** Its spectrum wanders instead of converging on a retention
constant, so there is no stable `λ` to read a `τ` off. The quantity is free —
`patchworks.body.CellOperators.radii` is `ρ(K)` — and what was owed is again a
bar.

`tau_wander_over_loop >= 1` — *the cell's implied retention moves, within one
window, by more than the round trip it has to serve*. ADR-0026 fixes both halves
and neither is invented: `τ = −1/ln ρ` is `05`'s form, `|loop(c)|` is the ADR's
divisor recomputed by `benchmarks/loop_length.py`, and the 1 is ADR-0021's `k = 1`
for its stated reason — every invented constant in this map's history has later
been found to have none. A `τ` that wanders by more than `|loop(c)|` is not a
retention constant on the timescale the cell must hold, which is #329's failure
in its own words.

The spread is the **interquartile range**, not the variance #351 sketched. Two
reasons, both about the quantity rather than about taste: `τ` diverges as
`ρ → 1` — ADR-0026's amendment calls that spread the honest object — so a
variance is dominated by whichever tick came nearest the band's face; and `08`'s
own instrument for a spread is the IQR, which #156 reused rather than replacing.
Cells whose `τ` is not finite across the window are counted and reported rather
than dropped, because a cell pinned at the band's upper face is the failure and
not a gap in the data.

**#329 already carries a cutoff and this rig does not replace it.** Its
`measurement benchmarks/detectability.py conduction ratio >= 1` is the
precondition — the bar above is only meaningful on charts from a graph that
transmits — and the register's grammar gives a problem **one** `@cutoff`, so a
sequenced second one cannot be written down today. The bar is therefore stated,
measured and reported here, and the field on the issue is left where it is;
`#353` is where the register's missing mechanisms are owed.

Like every script here **it asserts nothing** and its exit code does not move.
Pass `--no-file` on any read that is not *the* read.
"""

from __future__ import annotations

import argparse
import collections
import math
import pathlib
import statistics
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))

from patchworks.agent import Agent, run  # noqa: E402
from patchworks.diagnostics import Diagnostics  # noqa: E402
from patchworks.graph import DEFAULT_SPEC, DomeSpec, build_graph  # noqa: E402
from patchworks.learning import PredictionRule, SparsityAnneal, TransportRule  # noqa: E402
from patchworks.sandbox import PlanarPushSandbox  # noqa: E402

import loop_length  # noqa: E402
from cutoff_report import report as report_cutoffs  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tests"))
from conftest import SMALL  # noqa: E402

IMAGE_SIZE = {DEFAULT_SPEC.patch_grid: 64, SMALL.patch_grid: 16}

#: Ticks the published read runs, matching `benchmarks/projection_firing.py`:
#: the two rigs answer different questions off the same kind of run, and a
#: different length would make their rows incomparable for no reason.
TICKS = 3000

#: One seed is a reading and three are a spread; #274's driven read published
#: nine. Per-seed rows are printed rather than averaged away.
SEEDS = (0, 1, 2)


def build(spec: DomeSpec, split: str, seed: int) -> tuple[PlanarPushSandbox, Agent]:
    env = PlanarPushSandbox(split=split, image_size=IMAGE_SIZE[spec.patch_grid])
    agent = Agent(
        env, dome=build_graph(spec), generator=torch.Generator().manual_seed(seed)
    )
    return env, agent


def window(spec: DomeSpec) -> int:
    """One apex round trip, `max |loop(c)|`, recomputed off this spec's mask.

    The same count `benchmarks/projection_firing.py` uses for its burn-in, and
    for the same reason: it is the longest a signal in this graph takes to come
    back, so it is the shortest interval over which "did this settle" is a
    question about the mechanism rather than about a transient.
    """
    lengths = loop_length.loops(build_graph(spec)).lengths
    return max(lengths.values()) if lengths else 1


def tau(radius: float) -> float:
    """`τ = −1/ln ρ`, `05-timescales.md`'s form. Non-finite at or above 1, said so."""
    if not 0.0 < radius < 1.0:
        return math.inf if radius >= 1.0 else 0.0
    return -1.0 / math.log(radius)


def iqr(values: list[float]) -> float:
    """`08`'s spread instrument, reused rather than replaced (#156's habit)."""
    if len(values) < 2:
        return 0.0
    lower, upper = np.percentile(np.asarray(values, dtype=float), (25, 75))
    return float(upper - lower)


def measure(spec: DomeSpec, split: str, seed: int, ticks: int) -> dict[str, object]:
    """One driven seed: the paired edge instrument, and `τ(ρ(K))` per cell per tick.

    The loop is `benchmarks/untrained_fixed_point.py`'s `teaching`, unchanged and
    for its reasons — `agent.tick()` then the rules, the prediction rule joining
    on the first tick and the transport rule on the second. The drive is asserted
    by `Agent.tick` itself, so this is a driven run with no second writer beside
    the world's.
    """
    env, agent = build(spec, split, seed)
    span = window(spec)
    opening: list[np.ndarray] = []
    closing: list[np.ndarray] = []
    ranks: np.ndarray | None = None
    first: np.ndarray | None = None
    transmitting = 0
    taus: list[list[float]] = []
    try:
        diagnostics = Diagnostics(agent.sheaf)
        bias = PredictionRule(agent.sheaf)
        transport = TransportRule(agent.sheaf, anneal=SparsityAnneal())
        for index, _outcome in enumerate(run(agent, ticks, seed=seed)):
            bias.step()
            if agent.sheaf.ticks > 1:
                transport.step()
            # The opening window is read *after* the burn-in the same length
            # gives, so "energy fell" is a statement about the run rather than
            # about the constructor's zeros draining.
            if span <= index < 2 * span:
                reading = diagnostics.edge_reading()
                opening.append(reading.energy.detach().numpy())
                # The rank in the *opening* window is what makes a crossing
                # readable. #156's fourth trap is a bar firing on the
                # architecture working, and a fleet that was already near
                # rank-1 when the run began has not slid anywhere. Carried by
                # the rig rather than computed on the side, because a verdict
                # whose baseline lives in someone's shell history is a verdict
                # nobody can check.
                first = reading.effective_rank.detach().to(torch.float64).numpy()
            if index >= ticks - span:
                reading = diagnostics.edge_reading()
                closing.append(reading.energy.detach().numpy())
                ranks = reading.effective_rank.detach().to(torch.float64).numpy()
                transmitting = int((ranks > 0).sum())
                taus.append(
                    [
                        tau(float(r))
                        for r in agent.sheaf.operators.radii().detach().tolist()
                    ]
                )
    finally:
        env.close()
    if not opening or not closing or ranks is None or first is None:
        return {"seed": seed, "counted": 0}
    return {
        "seed": seed,
        "counted": len(closing),
        "opening": np.median(np.stack(opening), axis=0),
        "closing": np.median(np.stack(closing), axis=0),
        "ranks": ranks,
        "first_ranks": first,
        "transmitting": transmitting,
        "taus": taus,
        "cells": list(agent.dome.predicting),
    }


def drain(result: dict[str, object], edge_width: int) -> dict[str, float]:
    """#324: the effective rank of the maps on the edges whose energy fell.

    Both columns of the rank reading are used and they are not averaged: an edge
    has two maps belonging to two different cells, and *across the fleet* in the
    spec's reading is a statement about the whole population of maps, which is
    what this median is taken over.
    """
    opening = np.asarray(result["opening"])
    closing = np.asarray(result["closing"])
    ranks = np.asarray(result["ranks"])
    was = np.asarray(result.get("first_ranks", ranks))
    draining = closing < opening
    share = float(draining.mean()) if draining.size else 0.0
    if not draining.any():
        # The failure absent reads maximally healthy, never as a low number
        # nobody looked at. `m` is the largest the participation ratio can take.
        return {"rank": float(edge_width), "was": float(edge_width), "share": share}
    return {
        "rank": float(np.median(ranks[draining].reshape(-1))),
        # The same population's rank in the opening window: a crossing is only
        # a *slide* if this is higher, and #156's fourth trap is a bar that
        # fires on a fleet which was already there when the run began.
        "was": float(np.median(was[draining].reshape(-1))),
        "share": share,
    }


def wander(result: dict[str, object], loops: dict[int, int]) -> dict[str, float]:
    """#329: how far each cell's implied `τ` moves within the window, over `|loop(c)|`."""
    rows = list(zip(*result["taus"])) if result["taus"] else []
    cells = list(result["cells"])
    ratios: list[float] = []
    unsettled = 0
    for cell, series in zip(cells, rows):
        finite = [value for value in series if math.isfinite(value)]
        if len(finite) < len(series):
            unsettled += 1
        if not finite or not loops.get(cell):
            continue
        ratios.append(iqr(finite) / loops[cell])
    if not ratios:
        return {"ratio": 0.0, "unsettled": float(unsettled)}
    return {"ratio": float(statistics.median(ratios)), "unsettled": float(unsettled)}


def readings(
    results: list[dict[str, object]], spec: DomeSpec
) -> dict[str, float]:
    """What this read has to offer a `measurement` cutoff, by name."""
    dome = build_graph(spec)
    edge_width = max((e.m for e in dome.edges), default=1)
    loops = loop_length.loops(dome).lengths
    drains = [drain(r, edge_width) for r in results if r.get("counted")]
    wanders = [wander(r, loops) for r in results if r.get("counted")]
    if not drains:
        return {}
    return {
        "draining_effective_rank": float(np.median([d["rank"] for d in drains])),
        "draining_effective_rank_opening": float(np.median([d["was"] for d in drains])),
        "draining_edge_share": float(np.median([d["share"] for d in drains])),
        "tau_wander_over_loop": float(np.median([w["ratio"] for w in wanders])),
    }


def read(
    spec: DomeSpec, split: str, seeds: tuple[int, ...], ticks: int, *, file: bool = True
) -> None:
    dome = build_graph(spec)
    edge_width = max((e.m for e in dome.edges), default=1)
    loops = loop_length.loops(dome).lengths
    span = window(spec)
    print(
        f"\n== a driven run's drain and its settling =="
        f"\n   {len(dome.edges)} edges, {len(dome.predicting)} predicting cells, "
        f"{ticks} ticks, window {span} (one apex |loop|)"
        f"\n   effective rank reads 1 for a rank-1 map and up to m = {edge_width}"
    )
    results = []
    for seed in seeds:
        result = measure(spec, split, seed, ticks)
        results.append(result)
        if not result.get("counted"):
            print(f"\n   seed {seed}: the run was shorter than two windows; nothing read.")
            continue
        drained = drain(result, edge_width)
        wandered = wander(result, loops)
        print(
            f"\n   seed {seed}:  draining edges {drained['share']:.1%}"
            f"   median effective rank on them {drained['was']:.3g} -> "
            f"{drained['rank']:.3g}"
            f"   maps transmitting {result['transmitting']}/{2 * len(dome.edges)}"
        )
        print(
            f"            median tau IQR / |loop(c)| {wandered['ratio']:.3g}"
            f"   cells with a non-finite tau in the window "
            f"{int(wandered['unsettled'])}/{len(dome.predicting)}"
        )
    print(
        "\n   #324 is the first row and #329 the second. Neither bar is a level "
        "anybody\n   chose: one is the count that separates one direction from "
        "more than one,\n   the other is ADR-0026's own divisor with ADR-0021's "
        "k = 1 over it."
    )
    report_cutoffs("driven_settling", readings(results, spec), file=file)


def main(argv: list[str] | None = None) -> int:
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=("read",))
    parser.add_argument("--dome", choices=("real", "small"), default="real")
    parser.add_argument("--split", default="train")
    parser.add_argument("--ticks", type=int, default=TICKS)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument(
        "--no-file",
        action="store_true",
        help=(
            "print the cutoff report and touch the tracker not at all. Pass it "
            "on any read that is not *the* read."
        ),
    )
    arguments = parser.parse_args(argv)
    spec = SMALL if arguments.dome == "small" else DEFAULT_SPEC
    read(
        spec,
        arguments.split,
        tuple(arguments.seeds),
        arguments.ticks,
        file=not arguments.no_file,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
