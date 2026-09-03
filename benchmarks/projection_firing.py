"""How often ADR-0015's band projection fires, per cell, against depth (#351, for #335).

[#335](https://github.com/NGL321/patchworks/issues/335) says the band's
projection **can only shorten retention, never lengthen it**: ADR-0015 restores
the band by rescaling the whole operator, so every firing moves all of that
cell's eigenvalues by the same factor and shortens all its retention constants
together. Long retention at the apex is what ADR-0026's conduction ratio needs
(`|loop|` = 14 there), so the retention gradient stage 2 exists to produce is
fought by construction — *if* the mechanism fires hardest where retention is
most needed::

    python benchmarks/projection_firing.py read
    python benchmarks/projection_firing.py read --dome small --ticks 200 --no-file

**The bar was already stated and is a self-ratio; only the rig was owed.** #351
fixes it in those words — *apex firing rate at or above the rim while apex `τ` is
the shorter*. Both halves are ratios of the quantity to itself across depth, so
nothing here had to be invented, which is what
[#156](https://github.com/NGL321/patchworks/issues/156) found for the
falsification register's thresholds and what ADR-0029 exists to keep true.

**The two halves are combined by taking the lower**, which is
`benchmarks/detectability.py`'s convention for exactly this reason: a reading
that could say *the band is fighting retention* while one of its two clauses
failed is not the claim. So `band_fights_retention` is
`min(apex_firing / rim_firing, rim_τ / apex_τ)`, at or above 1 exactly when the
apex both fires at least as often as the rim **and** holds the shorter `τ`. The
two components are offered by name as well, so a later cutoff meaning one of
them can say so.

**What the projection firing is, and where it is read.** #138 initialises
`K = aI` and the projection runs after every prediction-rule step, outside the
transform, with no gradient — `patchworks.learning.PredictionRule.step` names the
firing rate as *the observable that calls #138's named fallback from a dense `K`
to a structured one*, and until #351 nothing could read it, because
:meth:`~patchworks.body.CellOperators.project` computed the mask and dropped it.
It now returns it. This rig records it without touching the rule: the rule is
documented as holding no state, and an instrument that made it hold some would
be changing the thing it measures.

**The burn-in is a count off the graph, not a level.** #156's fourth trap is
this one exactly — *entry 4 fires on the architecture working*, because the
fleet's first excursion against the band is the selection settling rather than
the cost. The exclusion here is one **apex round trip**, `max |loop(c)|` from
`benchmarks/loop_length.py`, which is 14 ticks on `DEFAULT_SPEC`: the longest
path any signal in this graph takes to come back, recomputed per spec rather
than quoted. Below one loop no cell has yet seen its own consequence, so nothing
before it is a reading of the mechanism at all.

Like every script here **it asserts nothing** and its exit code does not move.
Pass `--no-file` on any read that is not *the* read — the small dome, a short
run — because filing records a run against #335 and a toy reading is not one.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import math
import pathlib
import statistics
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))

from patchworks.agent import Agent, run  # noqa: E402
from patchworks.body import CellOperators  # noqa: E402
from patchworks.graph import DEFAULT_SPEC, Dome, DomeSpec, build_graph  # noqa: E402
from patchworks.learning import PredictionRule, SparsityAnneal, TransportRule  # noqa: E402
from patchworks.sandbox import PlanarPushSandbox  # noqa: E402

import loop_length  # noqa: E402
from cutoff_report import report as report_cutoffs  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tests"))
from conftest import SMALL  # noqa: E402

#: The render side each dome's patch grid wants, matching
#: `benchmarks/untrained_fixed_point.py`'s table rather than restating a rule.
IMAGE_SIZE = {DEFAULT_SPEC.patch_grid: 64, SMALL.patch_grid: 16}

#: Ticks the published read runs. Long enough that a per-cell firing *rate* is a
#: rate rather than a handful of events, and the same order as the driven reads
#: #274 and #166 published on this dome.
TICKS = 3000

#: One seed is a reading and three are a spread. #274's driven read published
#: nine; three is what a rig carrying a cutoff can afford every time it runs,
#: and the per-seed rows are printed so the spread is visible rather than
#: averaged away.
SEEDS = (0, 1, 2)


@contextlib.contextmanager
def recording(operators: CellOperators):
    """Capture every mask :meth:`CellOperators.project` returns, without state on the rule.

    :class:`~patchworks.learning.PredictionRule` is documented as holding no
    state whatsoever — *two calls with the same sheaf state produce the same
    step* — and hanging the last projection off it would make that false for the
    convenience of an instrument. So the recorder wraps the projection for the
    length of the read and unwraps it after: the rule is unchanged, the
    projection is unchanged, and what the mechanism did is visible.
    """
    fired: list[torch.Tensor] = []
    original = operators.project

    def watched() -> torch.Tensor:
        mask = original()
        fired.append(mask.clone())
        return mask

    operators.project = watched  # type: ignore[method-assign]
    try:
        yield fired
    finally:
        del operators.project


def build(spec: DomeSpec, split: str, seed: int) -> tuple[PlanarPushSandbox, Agent]:
    env = PlanarPushSandbox(split=split, image_size=IMAGE_SIZE[spec.patch_grid])
    agent = Agent(
        env, dome=build_graph(spec), generator=torch.Generator().manual_seed(seed)
    )
    return env, agent


def taught(agent: Agent, ticks: int, seed: int) -> list[torch.Tensor]:
    """Run with both rules on and hand back the projection mask from every step.

    The loop is `benchmarks/untrained_fixed_point.py`'s `teaching`, unchanged
    and for its reasons: `agent.tick()` then the rules, the prediction rule
    joining on the first tick and the transport rule on the second. The drive is
    asserted by `Agent.tick` itself, so this is a driven run without a second
    writer standing beside the world's.
    """
    bias = PredictionRule(agent.sheaf)
    transport = TransportRule(agent.sheaf, anneal=SparsityAnneal())
    with recording(agent.sheaf.operators) as fired:
        for _outcome in run(agent, ticks, seed=seed):
            bias.step()
            if agent.sheaf.ticks > 1:
                transport.step()
    return fired


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

    A rim that never fires and an apex that does is the failure at its most
    extreme, so it reads `inf`; an apex that never fires is the failure absent,
    so it reads 0 whatever the rim did. Neither is a number to quote — both are
    statements about which side of the bar the run fell.
    """
    if numerator == 0.0:
        return 0.0
    if denominator == 0.0:
        return math.inf
    return numerator / denominator


def measure(spec: DomeSpec, split: str, seed: int, ticks: int) -> dict[str, object]:
    """One seed: per-cell firing rate and `λ(K)`, past the burn-in."""
    env, agent = build(spec, split, seed)
    try:
        fired = taught(agent, ticks, seed)
    finally:
        env.close()
    skip = burn_in(spec)
    kept = fired[skip:]
    dome = agent.dome
    predicting = list(dome.predicting)
    if not kept:
        return {"seed": seed, "counted": 0}
    stacked = torch.stack(kept).to(torch.float64)
    rates = stacked.mean(dim=0)
    radii = agent.sheaf.operators.radii().detach().to(torch.float64)
    return {
        "seed": seed,
        "counted": len(kept),
        "firing": {cell: float(rates[i]) for i, cell in enumerate(predicting)},
        "radius": {cell: float(radii[i]) for i, cell in enumerate(predicting)},
    }


def readings(results: list[dict[str, object]], spec: DomeSpec) -> dict[str, float]:
    """What this read has to offer a `measurement` cutoff, by name.

    `band_fights_retention` is #335's bar. It is the **lower** of the two
    clauses, on `benchmarks/detectability.py`'s reasoning: a reading that says
    *holds* while one clause fails is not the claim being made. Medians across
    cells within a level, then across seeds — the level is the row and the seed
    spread is printed rather than folded into the bar, which is #274's habit.
    """
    dome = build_graph(spec)
    firing_ratio: list[float] = []
    tau_ratio: list[float] = []
    for result in results:
        if not result.get("counted"):
            continue
        firing = by_depth(dome, result["firing"])  # type: ignore[arg-type]
        radius = by_depth(dome, result["radius"])  # type: ignore[arg-type]
        if len(firing) < 2:
            continue
        rim, apex = min(firing), max(firing)
        firing_ratio.append(
            _ratio(statistics.median(firing[apex]), statistics.median(firing[rim]))
        )
        taus = {level: [tau(r) for r in rows] for level, rows in radius.items()}
        tau_ratio.append(
            _ratio(statistics.median(taus[rim]), statistics.median(taus[apex]))
        )
    if not firing_ratio:
        return {}
    apex_over_rim = float(np.median(firing_ratio))
    rim_over_apex = float(np.median(tau_ratio))
    return {
        "apex_firing_over_rim": apex_over_rim,
        "rim_tau_over_apex_tau": rim_over_apex,
        "band_fights_retention": min(apex_over_rim, rim_over_apex),
    }


def read(
    spec: DomeSpec, split: str, seeds: tuple[int, ...], ticks: int, *, file: bool = True
) -> None:
    """The read: firing rate and `λ(K)` against depth, then the cutoff verdict."""
    dome = build_graph(spec)
    print(
        f"\n== ADR-0015's projection, fired per cell against depth =="
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
        firing = by_depth(dome, result["firing"])  # type: ignore[arg-type]
        radius = by_depth(dome, result["radius"])  # type: ignore[arg-type]
        print(f"\n   seed {seed}   level  cells  firing rate   median lambda   tau")
        for level in firing:
            taus = [tau(r) for r in radius[level]]
            finite = [t for t in taus if math.isfinite(t)]
            shown = f"{statistics.median(finite):.3g}" if finite else "inf"
            print(
                f"            {level:>5}  {len(firing[level]):>5}  "
                f"{statistics.median(firing[level]):>11.3g}   "
                f"{statistics.median(radius[level]):>13.6g}   {shown:>8}"
            )
    print(
        "\n   The projection can only rescale down, so a level that fires more "
        "often is\n   a level whose retention is being cut more often. #335's "
        "claim is that this\n   is hardest at the apex, which is where "
        "ADR-0026 needs the longest tau."
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
