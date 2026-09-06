"""Did the spectral floor buy effective rank, and what did it cost the cap? (#435)

ADR-0032's **second pre-registration**, stated on the ADR before the constraint
existed and runnable now that [#432](https://github.com/NGL321/patchworks/issues/432)
built the floor into :meth:`patchworks.restriction.RestrictionMaps.project`::

    python benchmarks/spectral_floor_read.py read
    python benchmarks/spectral_floor_read.py read --dome small \
        --checkpoints 200 400 --no-file

This is a **read, not a decision**. The constraint is settled; this is what says
whether it paid. Two halves, and neither on its own:

1. **Flat maps must move effective rank toward `m`.** The per-edge participation
   ratio `(Σσᵢ²)²/Σσᵢ⁴`, which sat at **1.0009** on all three seeds at the
   shipped `λ = 0.4` ([#237](https://github.com/NGL321/patchworks/issues/237))
   and **2.913** against a mask ceiling of 4 at `λ = 0`
   ([#393](https://github.com/NGL321/patchworks/issues/393)). The instrument is
   :meth:`patchworks.diagnostics.Diagnostics.edge_reading`'s second half,
   unchanged, and the population is #324's: the maps on the edges whose energy
   fell.
2. **And it must not cost the incoherence cap.** ADR-0010 pre-registered that a
   cross-edge alignment pressure and `c` pull the same maps in opposite
   directions. :meth:`RestrictionMaps.gram_peaks` against
   :func:`patchworks.restriction.gain_denominators` is that collision arriving
   from the other side, **per cell, never a fleet aggregate** — so the worst
   cells are named and the whole distribution is published, and the aggregate
   rows below exist only to say how many cells are in each state.

**The horizon is the point, and it is not negotiable.** 100k ticks.
[#178](https://github.com/NGL321/patchworks/issues/178) has cost this map the
30k mistake three times, most recently on
[#416](https://github.com/NGL321/patchworks/issues/416), where the interior
scale ratio's meaning **reversed** between 30k and 100k. So 30k is read *en
route* on the same run and printed beside 100k — one trajectory, two
checkpoints, which is what makes "which way is it moving" a statement about the
same surface rather than about two runs.

**Flatness at exit is the first thing reported, and it is a residual.**
:meth:`RestrictionMaps.project` orders the floor **before** the incoherence cap,
so the floor is exact only where the cap does not bite: on a fresh draw that is
everywhere (flatness 0.999998), and under the fully-coherent worst arrangement
the floor is given up to 0.9427 and recovers monotonically across passes. What a
*trained* surface does was unmeasured before this rig.

**The nine unattainable masks are excluded by name, not averaged in.**
`RestrictionMaps.floored` is the population the floor reaches — on
`DEFAULT_SPEC`, 1339 of 1364 endpoints. A mask with `k_v < m_e` open columns
cannot contain a co-isometry at all and reads `flatness = 0` by construction;
averaging it in would report the floor failing where it was never applied. The
excluded count is printed so the exclusion is visible rather than silent.

**Per edge and per direction, never a graph-wide average** — #127's standing
rule, and [#181](https://github.com/NGL321/patchworks/issues/181)'s
per-edge-not-per-level form. The participation ratio is normalised by each
pair's *own* `m` where a fleet number is quoted, because the mask ceiling is not
4 everywhere and a raw fleet median silently weights the wide lanes.

**What this read cannot claim.** Nothing in this repository is converged; the
horizon buys an adapting surface, not a settled one, and that fence is
`benchmarks/detectability.py`'s in the same words. Like every script here **it
asserts nothing** and its exit code does not move.
"""

from __future__ import annotations

import argparse
import json
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
from patchworks.restriction import pair_index  # noqa: E402
from patchworks.sandbox import PlanarPushSandbox  # noqa: E402

import driven_settling  # noqa: E402
from untrained_fixed_point import run  # noqa: E402
from cutoff_report import report as report_cutoffs  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tests"))
from conftest import SMALL  # noqa: E402

IMAGE_SIZE = {DEFAULT_SPEC.patch_grid: 64, SMALL.patch_grid: 16}

#: Where the surface is read. 30k is reported **beside** 100k and never instead
#: of it, so which way a reading moves is visible; ADR-0032 states the horizon
#: and #178 is why the shorter one may not stand alone.
CHECKPOINTS = (30_000, 100_000)

#: One seed is a reading and three are a spread, `benchmarks/driven_settling.py`'s
#: count, named through it rather than re-chosen.
SEEDS = driven_settling.SEEDS


def build(
    spec: DomeSpec, split: str, seed: int, *, floor: bool = True
) -> tuple[PlanarPushSandbox, Agent]:
    env = PlanarPushSandbox(split=split, image_size=IMAGE_SIZE[spec.patch_grid])
    agent = Agent(
        env, dome=build_graph(spec), generator=torch.Generator().manual_seed(seed)
    )
    if not floor:
        # The control arm, and the *only* thing it changes. `project` keeps its
        # mask, its band and its cap; the floor alone stops running, so a
        # difference between the arms is attributable to the floor and to
        # nothing else. It is a benchmark-local disable and writes nothing back
        # into the module — the shipped surface is the floored one.
        agent.sheaf.maps._flatten = lambda: None
    return env, agent


def pair_widths(spec: DomeSpec) -> np.ndarray:
    """`[pairs]`: each endpoint's own `m`, the ceiling its participation ratio has.

    A fleet median of raw ratios weights the wide lanes; the ceiling is not 4
    everywhere, and #181's per-edge-not-per-level form is the reason this is
    carried per pair rather than assumed.
    """
    dome = build_graph(spec)
    widths = np.zeros(2 * len(dome.edges), dtype=float)
    for edge in dome.edges:
        for side in (0, 1):
            widths[pair_index(edge.id, side)] = float(edge.m)
    return widths


def snapshot(agent: Agent) -> dict[str, np.ndarray]:
    """The state-at-a-tick half of a checkpoint: flatness, and the cap's two sides.

    Read off the surface as it stands, with no window: flatness is the residual
    of the last projection and the Gram peak is the quantity the *next* gain
    divides by, so both are properties of this tick and neither is an average
    over a window.
    """
    maps = agent.sheaf.maps
    return {
        "flatness": maps.flatness().detach().to(torch.float64).numpy(),
        "floored": maps.floored.detach().numpy().copy(),
        "gram_peak": maps.gram_peaks().detach().to(torch.float64).numpy(),
        "cap": maps.overlap_target.detach().to(torch.float64).numpy(),
        "eps": float(torch.finfo(maps.maps.dtype).eps),
    }


def measure(
    spec: DomeSpec,
    split: str,
    seed: int,
    checkpoints: tuple[int, ...],
    sink=None,
    floor: bool = True,
) -> dict[str, object]:
    """One driven seed, read at every checkpoint on the way past.

    The loop is `benchmarks/untrained_fixed_point.py`'s `teaching` and
    `benchmarks/driven_settling.py`'s use of it, unchanged — `agent.tick()` then
    the rules, the prediction rule joining on the first tick and the transport
    rule on the second. One trajectory carries every checkpoint, which is the
    whole reason 30k costs nothing beside 100k.

    **`sink` is called the moment a checkpoint completes, and it is there
    because this run is hours long on a shared box.** A 100k read that only
    writes at exit loses the 30k reading too when the process is killed at 70k,
    and the 30k reading is a real one — it is what ADR-0032's horizon is stated
    *against*. Handing each checkpoint out as it lands makes the shorter read
    survive the longer one failing, which is the difference between a partial
    answer and no answer.
    """
    env, agent = build(spec, split, seed, floor=floor)
    span = driven_settling.window(spec)
    horizon = max(checkpoints)
    cell_ids = [cell.id for cell in agent.dome.cells]
    opening: list[np.ndarray] = []
    opening_ranks: np.ndarray | None = None
    pending: dict[int, list[np.ndarray]] = {c: [] for c in checkpoints}
    reads: dict[int, dict[str, object]] = {}
    try:
        diagnostics = Diagnostics(agent.sheaf)
        bias = PredictionRule(agent.sheaf)
        transport = TransportRule(agent.sheaf)
        for index, _outcome in enumerate(run(agent, horizon, seed=seed)):
            bias.step()
            if agent.sheaf.ticks > 1:
                transport.step()
            # The opening window sits after a burn-in of the same length, so
            # "energy fell" is a statement about the run and not about the
            # constructor's zeros draining (driven_settling's reason, its rig).
            if span <= index < 2 * span:
                reading = diagnostics.edge_reading()
                opening.append(reading.energy.detach().numpy())
                opening_ranks = (
                    reading.effective_rank.detach().to(torch.float64).numpy()
                )
            for checkpoint in checkpoints:
                if checkpoint - span <= index < checkpoint:
                    reading = diagnostics.edge_reading()
                    pending[checkpoint].append(reading.energy.detach().numpy())
                    if index == checkpoint - 1:
                        state = snapshot(agent)
                        state["closing"] = np.median(
                            np.stack(pending[checkpoint]), axis=0
                        )
                        state["ranks"] = (
                            reading.effective_rank.detach().to(torch.float64).numpy()
                        )
                        state["transmitting"] = int((state["ranks"] > 0).sum())
                        reads[checkpoint] = state
                        if sink is not None:
                            sink(
                                checkpoint,
                                {
                                    "seed": seed,
                                    "counted": 1,
                                    "opening": np.median(np.stack(opening), axis=0),
                                    "opening_ranks": opening_ranks,
                                    "reads": reads,
                                    "cells": cell_ids,
                                },
                            )
    finally:
        env.close()
    if not opening or opening_ranks is None or not reads:
        return {"seed": seed, "counted": 0}
    return {
        "seed": seed,
        "counted": 1,
        "opening": np.median(np.stack(opening), axis=0),
        "opening_ranks": opening_ranks,
        "reads": reads,
        "cells": cell_ids,
    }


def rank_half(
    result: dict[str, object], checkpoint: int, widths: np.ndarray
) -> dict[str, float]:
    """Half one: did effective rank move toward `m`, on #324's population?

    Both columns of the rank reading are used and they are not averaged — an
    edge has two maps belonging to two different cells, and *across the fleet*
    is a statement about the whole population of maps. `driven_settling.drain`'s
    reading, re-expressed per endpoint so each ratio can be put against its own
    ceiling as well as reported raw.
    """
    state = result["reads"][checkpoint]
    opening = np.asarray(result["opening"])
    closing = np.asarray(state["closing"])
    ranks = np.asarray(state["ranks"]).reshape(-1)
    was = np.asarray(result["opening_ranks"]).reshape(-1)
    draining = np.repeat(closing < opening, 2)
    share = float(draining.mean()) if draining.size else 0.0
    if not draining.any():
        # The failure absent reads maximally healthy, never as a low number
        # nobody looked at — driven_settling's convention, its words.
        return {
            "rank": float(widths.max()),
            "was": float(widths.max()),
            "share": share,
            "fraction": 1.0,
            "was_fraction": 1.0,
        }
    return {
        "rank": float(np.median(ranks[draining])),
        "was": float(np.median(was[draining])),
        "share": share,
        "fraction": float(np.median(ranks[draining] / widths[draining])),
        "was_fraction": float(np.median(was[draining] / widths[draining])),
    }


def flatness_half(result: dict[str, object], checkpoint: int) -> dict[str, float]:
    """Half one's other instrument: `σ_min/σ_max` at exit, on the floored population.

    The nine unattainable masks read 0 by construction and are excluded by
    `floored` rather than by a threshold, because a threshold would also drop a
    map the floor reached and lost.
    """
    state = result["reads"][checkpoint]
    flatness = np.asarray(state["flatness"])
    floored = np.asarray(state["floored"]).astype(bool)
    reached = flatness[floored]
    if not reached.size:
        return {"median": 0.0, "worst": 0.0, "p05": 0.0, "reached": 0, "excluded": 0}
    return {
        "median": float(np.median(reached)),
        "worst": float(reached.min()),
        "p05": float(np.percentile(reached, 5)),
        "reached": int(floored.sum()),
        "excluded": int((~floored).sum()),
    }


def cap_half(result: dict[str, object], checkpoint: int) -> dict[str, object]:
    """Half two: `λ_max(Σ_e F_evᵀF_ev)` against `g_v²·c_v`, **per cell**.

    The ratio is the whole reading. `project` orders the cap **last**, so `≤ 1`
    is supposed to hold exactly and by construction at every cell; a cell over 1
    is that guarantee broken, and a cell *at* 1 is the cap **biting** — the
    collision ADR-0010 pre-registered, and the place the floor is given up. The
    distribution is published and the worst cells are named; the counts below
    say how many cells are in each state and stand in for no cell.

    **`over` is counted against the dtype and not against 1 exactly.** The maps
    are float32, so a cell the cap has just pinned reads a hair above its own
    bound — two roundings, the Gram eigenvalue and this division, each worth an
    `eps`. Counting those as the guarantee broken would report a breach at every
    cell the cap is working perfectly at, which is the opposite of the reading.
    The tolerance is `2·eps` of the maps' own dtype, no constant is chosen, and
    `excess` publishes the largest overshoot in units of that `eps` so the count
    is checkable rather than trusted.
    """
    state = result["reads"][checkpoint]
    peak = np.asarray(state["gram_peak"])
    cap = np.asarray(state["cap"])
    eps = float(state.get("eps", np.finfo(np.float32).eps))
    live = cap > 0
    if not live.any():
        return {
            "max": 0.0,
            "median": 0.0,
            "over": 0,
            "biting": 0,
            "cells": 0,
            "excess": 0.0,
            "worst": [],
        }
    ratios = peak[live] / cap[live]
    cells = np.asarray(result["cells"])[live]
    ordered = np.argsort(-ratios)
    return {
        "max": float(ratios.max()),
        "median": float(np.median(ratios)),
        "over": int((ratios > 1.0 + 2.0 * eps).sum()),
        "biting": int((ratios > 0.99).sum()),
        "cells": int(live.sum()),
        "excess": float((ratios.max() - 1.0) / eps),
        "worst": [(int(cells[i]), float(ratios[i])) for i in ordered[:5]],
    }


def readings(
    results: list[dict[str, object]], spec: DomeSpec, checkpoint: int
) -> dict[str, float]:
    """What this read has to offer a `measurement` cutoff, by name."""
    widths = pair_widths(spec)
    counted = [r for r in results if r.get("counted")]
    if not counted:
        return {}
    ranks = [rank_half(r, checkpoint, widths) for r in counted]
    flats = [flatness_half(r, checkpoint) for r in counted]
    caps = [cap_half(r, checkpoint) for r in counted]
    return {
        "draining_effective_rank": float(np.median([r["rank"] for r in ranks])),
        "draining_effective_rank_opening": float(np.median([r["was"] for r in ranks])),
        "draining_effective_rank_fraction_of_m": float(
            np.median([r["fraction"] for r in ranks])
        ),
        "floored_flatness_median": float(np.median([f["median"] for f in flats])),
        "floored_flatness_worst": float(np.min([f["worst"] for f in flats])),
        "gram_peak_over_cap_max": float(np.max([c["max"] for c in caps])),
        "cells_over_cap": float(np.max([c["over"] for c in caps])),
    }


def read(
    spec: DomeSpec,
    split: str,
    seeds: tuple[int, ...],
    checkpoints: tuple[int, ...],
    *,
    file: bool = True,
    out: pathlib.Path | None = None,
    floor: bool = True,
) -> None:
    dome = build_graph(spec)
    widths = pair_widths(spec)
    print(
        f"\n== ADR-0032's second pre-registration: the floor's two halves =="
        f"\n   {len(dome.edges)} edges, {2 * len(dome.edges)} endpoints, "
        f"{len(dome.predicting)} predicting cells"
        f"\n   checkpoints {', '.join(str(c) for c in checkpoints)}; effective "
        f"rank reads 1 for a rank-1 map and up to that pair's own m "
        f"(max {int(widths.max())})"
    )
    payload: dict[str, object] = {"checkpoints": list(checkpoints), "seeds": {}}

    def flush() -> None:
        if out is None:
            return
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def landed(seed: int, checkpoint: int, result: dict[str, object]) -> None:
        """One checkpoint, printed and written the moment it exists."""
        ranked = rank_half(result, checkpoint, widths)
        flat = flatness_half(result, checkpoint)
        capped = cap_half(result, checkpoint)
        rows = payload["seeds"].setdefault(str(seed), {})
        rows[str(checkpoint)] = {
            "rank": ranked,
            "flatness": flat,
            "cap": capped,
            "transmitting": result["reads"][checkpoint]["transmitting"],
        }
        print(
            f"\n   seed {seed} @ {checkpoint}:"
            f"  draining edges {ranked['share']:.1%}"
            f"   effective rank on them {ranked['was']:.4g} -> "
            f"{ranked['rank']:.4g}"
            f"  ({ranked['was_fraction']:.3f} -> {ranked['fraction']:.3f} of m)",
            flush=True,
        )
        print(
            f"            flatness on the {flat['reached']} floored endpoints:"
            f" median {flat['median']:.6g}, 5th pct {flat['p05']:.6g},"
            f" worst {flat['worst']:.6g}"
            f"   ({flat['excluded']} unattainable, excluded)",
            flush=True,
        )
        print(
            f"            gram peak / cap: max {capped['max']:.6g},"
            f" median {capped['median']:.6g},"
            f" cells over 1: {capped['over']}/{capped['cells']}"
            f" (largest overshoot {capped['excess']:.2f} eps),"
            f" biting (>0.99): {capped['biting']}",
            flush=True,
        )
        worst = ", ".join(f"cell {c} {v:.4g}" for c, v in capped["worst"])
        print(f"            worst cells: {worst}", flush=True)
        flush()

    results = []
    for seed in seeds:
        result = measure(
            spec,
            split,
            seed,
            checkpoints,
            floor=floor,
            sink=lambda checkpoint, partial, seed=seed: landed(
                seed, checkpoint, partial
            ),
        )
        results.append(result)
        if not result.get("counted"):
            print(
                f"\n   seed {seed}: the run was shorter than two windows; "
                "nothing read.",
                flush=True,
            )
    print(
        "\n   Half one is the first two rows and half two is the third; neither "
        "half\n   carries the reading alone. A cell over 1 is the cap's "
        "construction guarantee\n   broken; a cell at 1 is the cap biting, which "
        "is where the floor is given up."
    )
    if out is not None:
        flush()
        print(f"\n   wrote {out}")
    report_cutoffs(
        "spectral_floor_read", readings(results, spec, max(checkpoints)), file=file
    )


def main(argv: list[str] | None = None) -> int:
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=("read",))
    parser.add_argument("--dome", choices=("real", "small"), default="real")
    parser.add_argument("--split", default="train")
    parser.add_argument("--checkpoints", type=int, nargs="+", default=list(CHECKPOINTS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--out", type=pathlib.Path, default=None)
    parser.add_argument(
        "--no-floor",
        action="store_true",
        help=(
            "the control arm: run with ADR-0032's spectral floor disabled and "
            "everything else — mask, band, incoherence cap — untouched, so a "
            "difference between the arms is the floor's and nothing else's."
        ),
    )
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
        tuple(sorted(arguments.checkpoints)),
        file=not arguments.no_file,
        out=arguments.out,
        floor=not arguments.no_floor,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
