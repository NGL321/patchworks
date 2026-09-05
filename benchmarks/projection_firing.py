"""Retention against depth, per cell — and the projection that used to fire on it (#351, for #335).

**The mechanism this rig was built to watch no longer exists.**
[#433](https://github.com/NGL321/patchworks/issues/433) moved the band's
enforcement into the forward path, so there is no post-step projection and
nothing fires. The firing half of this read is therefore **undefined rather
than zero**, and reporting it as *0, fixed* would be measuring the instrument's
own removal. What survives is the half the firing was destroying: **`λ(K)` and
`τ` per cell against depth**, which is what
[#335](https://github.com/NGL321/patchworks/issues/335)'s claim was ultimately
about and what [#433](https://github.com/NGL321/patchworks/issues/433)
pre-registered a read on. See *What changed, and what this rig is now* below.

[#335](https://github.com/NGL321/patchworks/issues/335) said the band's
projection **can only shorten retention, never lengthen it**: ADR-0015 restored
the band by rescaling the whole operator, so every firing moved all of that
cell's eigenvalues by the same factor and shortened all its retention constants
together. Long retention at the apex is what ADR-0026's conduction ratio needs
(`|loop|` = 14 there), so the retention gradient stage 2 exists to produce was
fought by construction — *if* the mechanism fired hardest where retention was
most needed::

    python benchmarks/projection_firing.py read
    python benchmarks/projection_firing.py read --dome small --ticks 200 --no-file

**The bar was already stated and is a self-ratio; only the rig was owed.** #351
fixes it in those words — *apex firing rate at or above the rim while apex `τ` is
the shorter*. Both halves are ratios of the quantity to itself across depth, so
nothing here had to be invented, which is what
[#156](https://github.com/NGL321/patchworks/issues/156) found for the
falsification register's thresholds and what ADR-0029 exists to keep true.

**The two halves were combined by taking the lower**, which is
`benchmarks/detectability.py`'s convention for exactly this reason: a reading
that could say *the band is fighting retention* while one of its two clauses
failed is not the claim. So `band_fights_retention` was
`min(apex_firing / rim_firing, rim_τ / apex_τ)`, at or above 1 exactly when the
apex both fired at least as often as the rim **and** held the shorter `τ`.

## What changed, and what this rig is now

Since [#433](https://github.com/NGL321/patchworks/issues/433) this run reports
**`rim_tau_over_apex_tau` alone**. `apex_firing_over_rim` and the combined
`band_fights_retention` are **not reported**, and that is deliberate on two
grounds. The firing rate is not a number this build has: nothing fires, so a
reported 0 would be the instrument's removal masquerading as a fix. And
`band_fights_retention` takes the **lower** of two clauses, so supplying it from
the surviving clause alone would silently change what the name means — the
opposite of the convention above.

The consequence is stated rather than worked around: **#335's `@cutoff` names a
metric this rig no longer reports**, so `cutoff_report` prints it as unreported,
which is exactly the *cutoff nothing can fire on* that `open-problems.md`'s
second loud section exists to make visible. Re-pointing that bar at the
surviving clause would be **ruling #335**, which #433 explicitly forbade itself
and which this rig may not do sideways. #335 stays open, unruled, and is told
what its instrument now measures.

`recording()` is gone with `CellOperators.project`, and the statelessness note
it carried is now on
:meth:`~patchworks.body.CellOperators.used`, where it decided something: it is
the reason the forward normalisation takes a stateless batched spectral norm
rather than a warm-started power iteration's per-cell buffer.

**What the projection firing was, and why it is gone.** #138 initialises
`K = aI`, and the projection used to run after every prediction-rule step,
outside the transform, with no gradient —
`patchworks.learning.PredictionRule.step` named the firing rate as *the
observable that calls #138's named fallback from a dense `K` to a structured
one*. #422 read it on this rig at 100k ticks and found it **growing** rather
than shrinking, hardest at the apex, which is what #433 ruled on. The remedy
removed the observable along with the mechanism.

**The burn-in is a count off the graph, not a level**, and it survives the
change. #156's fourth trap is this one exactly — *entry 4 fires on the
architecture working*, because the fleet's first excursion against the band is
the selection settling rather than the cost. The exclusion here is one **apex
round trip**, `max |loop(c)|` from `benchmarks/loop_length.py`, which is 14
ticks on `DEFAULT_SPEC`: the longest path any signal in this graph takes to come
back, recomputed per spec rather than quoted. Below one loop no cell has yet
seen its own consequence, so nothing before it is a reading of the mechanism at
all. It now gates whether the run is long enough to read rather than which steps
enter a rate, since `λ(K)` is read off the trained operator at the end.

Like every script here **it asserts nothing** and its exit code does not move.
Pass `--no-file` on any read that is not *the* read — the small dome, a short
run — because filing records a run against #335 and a toy reading is not one.
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
from patchworks.graph import DEFAULT_SPEC, Dome, DomeSpec, build_graph  # noqa: E402
from patchworks.learning import PredictionRule, TransportRule  # noqa: E402
from patchworks.sandbox import PlanarPushSandbox  # noqa: E402

import loop_length  # noqa: E402
from cutoff_report import report as report_cutoffs  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tests"))
from conftest import SMALL  # noqa: E402

#: The render side each dome's patch grid wants, matching
#: `benchmarks/untrained_fixed_point.py`'s table rather than restating a rule.
IMAGE_SIZE = {DEFAULT_SPEC.patch_grid: 64, SMALL.patch_grid: 16}

#: Ticks the published read runs, and the same order as the driven reads #274
#: and #166 published on this dome.
#:
#: **This default is short for the thing the rig now reads, and #422 measured by
#: how much.** It ran a horizon ladder and found that at 3k this reading is
#: **CLEAR** and that it *reverses* by 100k — so a 3k run is not a cheap version
#: of the long one but a different and misleading one. #433's pre-registered
#: read is at `--ticks 100000` on seeds 0, 1, 2 for exactly that reason. The
#: default is left where it is because a rig carrying a cutoff runs often and
#: 100k is ~80 minutes a seed; a session reading retention against depth to
#: *decide* something passes the horizon explicitly.
TICKS = 3000

#: One seed is a reading and three are a spread. #274's driven read published
#: nine; three is what a rig carrying a cutoff can afford every time it runs,
#: and the per-seed rows are printed so the spread is visible rather than
#: averaged away.
SEEDS = (0, 1, 2)


def build(spec: DomeSpec, split: str, seed: int) -> tuple[PlanarPushSandbox, Agent]:
    env = PlanarPushSandbox(split=split, image_size=IMAGE_SIZE[spec.patch_grid])
    agent = Agent(
        env, dome=build_graph(spec), generator=torch.Generator().manual_seed(seed)
    )
    return env, agent


def taught(agent: Agent, ticks: int, seed: int) -> int:
    """Run with both rules on. Returns the number of steps taken.

    The loop is `benchmarks/untrained_fixed_point.py`'s `teaching`, unchanged
    and for its reasons: `agent.tick()` then the rules, the prediction rule
    joining on the first tick and the transport rule on the second. The drive is
    asserted by `Agent.tick` itself, so this is a driven run without a second
    writer standing beside the world's.

    It returned the per-step projection masks until #433 removed the projection.
    A count is what is left, and it is still worth returning: it is what the
    burn-in is checked against.
    """
    bias = PredictionRule(agent.sheaf)
    transport = TransportRule(agent.sheaf)
    steps = 0
    for _outcome in run(agent, ticks, seed=seed):
        bias.step()
        if agent.sheaf.ticks > 1:
            transport.step()
        steps += 1
    return steps


def burn_in(spec: DomeSpec) -> int:
    """One apex round trip, `max |loop(c)|`, recomputed off this spec's mask."""
    lengths = loop_length.loops(build_graph(spec)).lengths
    return max(lengths.values()) if lengths else 0


def tau(radius: float) -> float:
    """`τ = −1/ln ρ`, the retention constant ADR-0026 reads `|loop|` against.

    `05-timescales.md`'s form. It diverges as `ρ → 1` and is undefined at or
    above it, which ADR-0026's own amendment calls the honest object rather than
    a defect: a radius at the band's upper face is a cell that forgets nothing,
    and reporting `inf` says so where clamping it to a large number would
    quietly invent a retention.
    """
    if not 0.0 < radius < 1.0:
        return math.inf if radius >= 1.0 else 0.0
    return -1.0 / math.log(radius)


def by_depth(dome: Dome, values: dict[int, float]) -> dict[int, list[float]]:
    """Per-cell values grouped by construction level, shallowest first."""
    cells = {c.id: c for c in dome.cells}
    grouped: dict[int, list[float]] = collections.defaultdict(list)
    for cell, value in values.items():
        grouped[cells[cell].index.level].append(value)
    return dict(sorted(grouped.items()))


def _ratio(numerator: float, denominator: float) -> float:
    """A self-ratio that states what a zero denominator means rather than raising.

    Since #433 its one caller is the `τ` clause, where the degenerate cases are
    a rim that forgets instantly (`ρ = 0`, so `τ = 0`) against an apex that does
    not, and its converse. It reads `inf` and `0` respectively. Neither is a
    number to quote — both are statements about which side of the bar the run
    fell. `τ` at the band's *upper* face is `inf` rather than zero — ADR-0026's
    amendment calls that the honest object — and it is **not** this function's
    case: it arrives as a non-finite argument, so the ratio is `0` when the apex
    forgets nothing and `nan` when both faces do. The per-level print reports
    the finite median beside it, so a reader is never left with the ratio alone.
    """
    if numerator == 0.0:
        return 0.0
    if denominator == 0.0:
        return math.inf
    return numerator / denominator


def measure(spec: DomeSpec, split: str, seed: int, ticks: int) -> dict[str, object]:
    """One seed: per-cell `λ(K)` of the **used** operator, on a run past the burn-in.

    `radii()` reports the used operator since #433, which is the referent every
    reader of it meant — the band is a statement about what the cell computes
    with, and under the forward normalisation the raw parameter may sit outside
    it. `raw_radii()` is the other half if a reader ever wants the parameter.
    """
    env, agent = build(spec, split, seed)
    try:
        steps = taught(agent, ticks, seed)
    finally:
        env.close()
    counted = max(steps - burn_in(spec), 0)
    predicting = list(agent.dome.predicting)
    if not counted:
        return {"seed": seed, "counted": 0}
    operators = agent.sheaf.operators
    radii = operators.radii().detach().to(torch.float64)
    raw = operators.raw_norms.detach().to(torch.float64)
    return {
        "seed": seed,
        "counted": counted,
        "radius": {cell: float(radii[i]) for i, cell in enumerate(predicting)},
        # Nothing rescales the stored parameter any more, so how far it drifts
        # out of band is a question this build creates and the last one could
        # not ask. It is a diagnostic and not a reading: `sigma_max(used)` is in
        # band whatever this says, and no cutoff names it.
        "raw_norm": {cell: float(raw[i]) for i, cell in enumerate(predicting)},
    }


def readings(results: list[dict[str, object]], spec: DomeSpec) -> dict[str, float]:
    """What this read has to offer a `measurement` cutoff, by name.

    **One reading since #433**, and the two that are missing are missing on
    purpose. `apex_firing_over_rim` is not a number this build has — nothing
    fires — and `band_fights_retention` is the **lower** of two clauses, so
    handing it back from the surviving clause alone would change what the name
    means rather than report it. #335's bar names the combined metric and so
    goes unreported, which `cutoff_report` prints as such; re-pointing that bar
    is #335's to do and not this rig's. See the module docstring.

    Medians across cells within a level, then across seeds — the level is the
    row and the seed spread is printed rather than folded into the bar, which is
    #274's habit.
    """
    dome = build_graph(spec)
    tau_ratio: list[float] = []
    for result in results:
        if not result.get("counted"):
            continue
        radius = by_depth(dome, result["radius"])  # type: ignore[arg-type]
        if len(radius) < 2:
            continue
        rim, apex = min(radius), max(radius)
        taus = {level: [tau(r) for r in rows] for level, rows in radius.items()}
        tau_ratio.append(
            _ratio(statistics.median(taus[rim]), statistics.median(taus[apex]))
        )
    if not tau_ratio:
        return {}
    return {"rim_tau_over_apex_tau": float(np.median(tau_ratio))}


def read(
    spec: DomeSpec, split: str, seeds: tuple[int, ...], ticks: int, *, file: bool = True
) -> None:
    """The read: `λ(K)` and `τ` against depth, then the cutoff verdict."""
    dome = build_graph(spec)
    print(
        f"\n== the used operator's lambda(K) per cell against depth =="
        f"\n   {len(dome.predicting)} predicting cells, {ticks} ticks, "
        f"{len(seeds)} seed(s), burn-in {burn_in(spec)} (one apex |loop|)"
    )
    results = []
    for seed in seeds:
        result = measure(spec, split, seed, ticks)
        results.append(result)
        if not result.get("counted"):
            print(f"\n   seed {seed}: no steps past the burn-in; nothing to read.")
            continue
        radius = by_depth(dome, result["radius"])  # type: ignore[arg-type]
        raw = by_depth(dome, result["raw_norm"])  # type: ignore[arg-type]
        print(
            f"\n   seed {seed}   level  cells   median lambda   tau   "
            f"median raw sigma"
        )
        for level in radius:
            taus = [tau(r) for r in radius[level]]
            finite = [t for t in taus if math.isfinite(t)]
            shown = f"{statistics.median(finite):.3g}" if finite else "inf"
            print(
                f"            {level:>5}  {len(radius[level]):>5}   "
                f"{statistics.median(radius[level]):>13.6g}   {shown:>8}   "
                f"{statistics.median(raw[level]):>16.4g}"
            )
    print(
        "\n   Nothing fires since #433: the band is enforced in the forward path,"
        "\n   so the firing rate is undefined rather than zero and is not "
        "reported.\n   What is read is the retention it was cutting -- long tau "
        "at the apex is\n   what ADR-0026's conduction ratio needs (|loop| = 14 "
        "there)."
    )
    report_cutoffs("projection_firing", readings(results, spec), file=file)


def main(argv: list[str] | None = None) -> int:
    # #166's rig pins torch to one thread for the same reason: a rig that
    # spreads over every core measures the machine as much as the mechanism.
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
