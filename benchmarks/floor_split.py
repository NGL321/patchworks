"""ADR-0007's two floors, and the third one #339 says is there (#351, for #339).

[#339](https://github.com/NGL321/patchworks/issues/339) says the transport rule
has **no fixed point at agreement**: a norm, unlike a squared norm, has a
gradient of unit magnitude arbitrarily close to zero, so every edge's maps orbit
their optimum at amplitude `~η` forever. That is a floor on disagreement which is
neither of ADR-0007's two kinds — not static (not a function of configuration)
and not lag (not a function of motion), but a function of the **rule itself** —
and it is the floor the acceptance demo reads through::

    python benchmarks/floor_split.py read
    python benchmarks/floor_split.py read --dome small --learn 200 --hold 50 --no-file

**Three segments, one run, and the differences are the floors.** ADR-0007's
quiescent hold already separates two of them, and adding a third segment that
differs only in whether the rule steps separates the third:

======================  ==========  ==========  ==================================
segment                 world       rule        per-edge energy it leaves
======================  ==========  ==========  ==================================
drive                   moving      stepping    `static + lag + wander`
hold, rule frozen       held        frozen      `static`
hold, rule stepping     held        stepping    `static + wander`
======================  ==========  ==========  ==================================

So `wander` is the second hold minus the first and `lag` is the drive minus the
second. **The two holds start from the same state**, which is what makes the
subtraction a difference of one variable: the frozen hold changes no learned
parameter, so the tick state is restored and the same segment is re-run with the
transport rule on. Running them the other way round would leave the second hold
reading a surface the first had moved.

`hold_still` and the 400-tick hold are `benchmarks/detectability.py`'s, unchanged
and cited rather than re-derived — the length is `untrained_fixed_point`'s too,
and it is there for exactly this reason: long enough that the lag floor has
drained out of the reading.

**The bar is a self-ratio and 1 is ADR-0021's `k = 1`.**
`wander_over_named_floors >= 1` says the floor ADR-0007 does not name is at least
as large as the two it does, together. That is the claim #339 actually makes —
the demo reads *through* this floor, and a taxonomy with two kinds in it is
wrong about what the demo measures the moment the third outweighs them. Nothing
was chosen: the numerator and the denominator are the same quantity, in the same
communication-lane units, off the same run.

**Fleet medians first, then the ratio.** A per-edge ratio has a denominator that
can be arbitrarily small on a single edge, and a median of such ratios is a
median of noise; the floors are populations and the bar is a statement about the
fleet. `benchmarks/detectability.py` takes its median the same way and for the
same reason.

**What this read cannot claim.** #339 says *on a converged edge*, and nothing in
this repository is trained: `--learn` buys an adapting surface, not a converged
one, which is the fence `benchmarks/detectability.py --learn` carries in the same
words. So the reading is a reading of the floors as they stand at the tick it was
taken, and the bar is written so that it can be re-read without change the day a
converged surface exists.

Like every script here **it asserts nothing** and its exit code does not move.
Pass `--no-file` on any read that is not *the* read.
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))

from patchworks.agent import Agent  # noqa: E402
from patchworks.diagnostics import Diagnostics  # noqa: E402
from patchworks.graph import DEFAULT_SPEC, DomeSpec, build_graph  # noqa: E402
from patchworks.learning import PredictionRule, TransportRule  # noqa: E402
from patchworks.sandbox import PlanarPushSandbox  # noqa: E402

import detectability  # noqa: E402
import untrained_fixed_point as ufp  # noqa: E402
from cutoff_report import report as report_cutoffs  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tests"))
from conftest import SMALL  # noqa: E402

IMAGE_SIZE = {DEFAULT_SPEC.patch_grid: 64, SMALL.patch_grid: 16}

#: Ticks of both rules before the floors are read. An adapting surface, not a
#: converged one — the module docstring carries the fence.
LEARN = 3000

#: Ticks of held world per segment. `benchmarks/detectability.py`'s `HOLD`,
#: named through it rather than copied so the two cannot drift.
HOLD = detectability.HOLD

#: Ticks at the end of each segment the energy is read over, so a floor is a
#: level rather than one tick's excursion. `benchmarks/detectability.py`'s
#: `WINDOW`, for the same reason and by the same route.
WINDOW = detectability.WINDOW

SEEDS = (0, 1, 2)


def build(spec: DomeSpec, split: str, seed: int) -> tuple[PlanarPushSandbox, Agent]:
    env = PlanarPushSandbox(split=split, image_size=IMAGE_SIZE[spec.patch_grid])
    agent = Agent(
        env, dome=build_graph(spec), generator=torch.Generator().manual_seed(seed)
    )
    return env, agent


def segment(
    agent: Agent,
    diagnostics: Diagnostics,
    ticks: int,
    window: int,
    observation,
    applied,
    transport: TransportRule | None,
) -> np.ndarray:
    """Hold the world for `ticks`, optionally stepping the rule, and read the floor.

    The hold is `benchmarks/detectability.py`'s `hold_still` with one addition:
    the transport rule steps between ticks when it is passed, which is the only
    difference between this rig's two hold segments and therefore the whole of
    what the subtraction attributes to the rule.
    """
    readings: list[np.ndarray] = []
    for index in range(ticks):
        agent.sheaf.tick()
        agent.write(observation, applied)
        if transport is not None and agent.sheaf.ticks > 1:
            transport.step()
        if index >= ticks - window:
            readings.append(diagnostics.edge_reading().energy.detach().numpy())
    return np.median(np.stack(readings), axis=0)


def measure(
    spec: DomeSpec, split: str, seed: int, learn: int, hold: int, window: int
) -> dict[str, object]:
    """One seed: the three segments, in the order the subtraction requires."""
    env, agent = build(spec, split, seed)
    try:
        observation, _info = env.reset(seed=seed)
        agent.observe(observation)
        diagnostics = Diagnostics(agent.sheaf)

        # -- drive: the world moving, both rules on.
        bias = PredictionRule(agent.sheaf)
        transport = TransportRule(agent.sheaf)
        driven: list[np.ndarray] = []
        applied = None
        for index, outcome in enumerate(ufp.run(agent, learn, seed=seed)):
            bias.step()
            if agent.sheaf.ticks > 1:
                transport.step()
            observation, applied = outcome.observation, outcome.command
            if index >= learn - window:
                driven.append(diagnostics.edge_reading().energy.detach().numpy())
        if not driven:
            return {"seed": seed, "counted": 0}
        drive_floor = np.median(np.stack(driven), axis=0)

        # -- hold, rule frozen. Changes no learned parameter, so the tick state
        #    restored below is the whole of what this segment moved.
        state = ufp.snapshot(agent.sheaf)
        static = segment(agent, diagnostics, hold, window, observation, applied, None)

        # -- hold, rule stepping, from the same state.
        ufp.restore(agent.sheaf, state)
        with_rule = segment(
            agent, diagnostics, hold, window, observation, applied, transport
        )
    finally:
        env.close()
    return {
        "seed": seed,
        "counted": 1,
        "static": float(np.median(static)),
        "wander": float(np.median(with_rule) - np.median(static)),
        "lag": float(np.median(drive_floor) - np.median(with_rule)),
        "driven": float(np.median(drive_floor)),
    }


def _ratio(numerator: float, denominator: float) -> float:
    """A self-ratio that states a zero denominator rather than raising.

    A wander at or below zero is the failure absent — the rule's motion did not
    add disagreement over the hold — and reads 0 whatever the named floors did.
    """
    if numerator <= 0.0:
        return 0.0
    if denominator <= 0.0:
        return math.inf
    return numerator / denominator


def readings(results: list[dict[str, object]]) -> dict[str, float]:
    """What this read has to offer a `measurement` cutoff, by name."""
    kept = [r for r in results if r.get("counted")]
    if not kept:
        return {}
    wander = float(np.median([r["wander"] for r in kept]))
    named = float(np.median([r["static"] + r["lag"] for r in kept]))
    driven = float(np.median([r["driven"] for r in kept]))
    return {
        "wander_over_named_floors": _ratio(wander, named),
        "wander_share_of_floor": _ratio(wander, driven),
    }


def read(
    spec: DomeSpec,
    split: str,
    seeds: tuple[int, ...],
    learn: int,
    hold: int,
    window: int,
    *,
    file: bool = True,
) -> None:
    dome = build_graph(spec)
    print(
        f"\n== ADR-0007's floors, and the rule's own =="
        f"\n   {len(dome.edges)} edges, {learn} driven ticks, {hold}-tick holds, "
        f"read over the closing {window}"
    )
    results = []
    for seed in seeds:
        result = measure(spec, split, seed, learn, hold, window)
        results.append(result)
        if not result.get("counted"):
            print(f"\n   seed {seed}: the run was shorter than a window; nothing read.")
            continue
        print(
            f"\n   seed {seed}:  static {result['static']:.4g}"
            f"   wander {result['wander']:+.4g}"
            f"   lag {result['lag']:+.4g}"
            f"   driven total {result['driven']:.4g}"
        )
    print(
        "\n   `static` is the held world with the rule frozen; `wander` is what "
        "the rule\n   adds to that same held state; `lag` is what the moving "
        "world adds on top.\n   The three sum to the floor the acceptance demo "
        "reads through."
    )
    report_cutoffs("floor_split", readings(results), file=file)


def main(argv: list[str] | None = None) -> int:
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=("read",))
    parser.add_argument("--dome", choices=("real", "small"), default="real")
    parser.add_argument("--split", default="train")
    parser.add_argument("--learn", type=int, default=LEARN)
    parser.add_argument("--hold", type=int, default=HOLD)
    parser.add_argument("--window", type=int, default=WINDOW)
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
        arguments.learn,
        arguments.hold,
        arguments.window,
        file=not arguments.no_file,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
