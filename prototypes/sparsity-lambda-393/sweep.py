"""What `λ` buys and what it costs the channel: the sweep #393 asks for.

    python prototypes/sparsity-lambda-393/sweep.py --lam 0.4 --seeds 0 1 2
    python prototypes/sparsity-lambda-393/sweep.py --lam 0 --ticks 3000 --trials 4

**One training run per `(λ, seed)`, three readings off the same surface.** The
three quantities #393 names are all reads of a trained dome, and training is the
whole cost -- 30,000 ticks is ~21 minutes where every read together is under two.
Training each of them separately would pay for the run three times and, worse,
would report three quantities from three different surfaces, which is precisely
the joint reading the ticket asks for ("read **together**").

The three, and where each comes from:

1. **Effective rank** -- `benchmarks/driven_settling.py`'s
   `draining_effective_rank` and its `..._opening` baseline, computed by that
   module's own :func:`driven_settling.drain` imported rather than copied. The
   windows are its windows: an opening window one apex `|loop|` long placed
   after a burn-in of the same length, and a closing window at the end.
2. **`dim H0` and `rank d`** -- `Diagnostics.whole_graph`, over the
   predicting-cell subcomplex. `chi = dim H0 - dim H1` is fixed at construction,
   so it is asserted here as the free correctness check #393 asks for.
3. **The conduction ratio** -- `benchmarks/detectability.py`'s trial machinery,
   both directions, off the same trained surface cast to float64.

**A rig asserts nothing** (`benchmarks/run_reporting.py`). This reports a curve.
It does not file cutoffs and it does not gate: a `λ` sweep is not *the* read, and
filing a toy reading against #325/#329/#341 is what `--no-file` exists to
prevent. Nothing here writes to the problem registers.

**Per edge and per direction, never a graph-wide average** (#127's standing
rule). The rank reading is a median over the population of *maps* on draining
edges -- two per edge, uncollapsed, which is `driven_settling`'s own convention
-- and the per-map quantiles are published beside it so the median is never the
only thing on the record. Conduction is stated twice, rim-to-apex and
apex-to-rim, never pooled.

This is a prototype, not a published rig: #127's standing note is that an edit
hands off to its own ticket, and promoting this to `benchmarks/` would be one.
"""

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
for _part in ("src", "tools", "benchmarks", "tests"):
    _path = str(ROOT / _part)
    if _path not in sys.path:
        sys.path.insert(0, _path)

import detectability as det  # noqa: E402
import driven_settling as ds  # noqa: E402
import loop_length  # noqa: E402
import untrained_fixed_point as ufp  # noqa: E402
from patchworks.agent import run  # noqa: E402
from patchworks.diagnostics import Diagnostics  # noqa: E402
from patchworks.graph import DEFAULT_SPEC, build_graph  # noqa: E402
from patchworks.learning import (  # noqa: E402
    PredictionRule,
    SparsityAnneal,
    TransportRule,
)

#: #237's length, so the `λ = 0.4` column is comparable to the record's own
#: 30,000-tick reading of 1.0009 rather than to a number nobody else has.
TICKS = 30000

#: `benchmarks/detectability.py`'s published read, unchanged.
TRIALS = 24


def quantiles(values: np.ndarray) -> dict[str, float]:
    """p05/p25/median/p75/p95, the spread convention this rig's neighbours use."""
    flat = np.asarray(values, dtype=float).reshape(-1)
    if flat.size == 0:
        return {}
    p05, p25, p50, p75, p95 = np.percentile(flat, (5, 25, 50, 75, 95))
    return {
        "p05": float(p05),
        "p25": float(p25),
        "median": float(p50),
        "p75": float(p75),
        "p95": float(p95),
    }


def train(agent, lam: float, seed: int, ticks: int, span: int) -> dict:
    """One driven run at pressure `λ`, with `driven_settling`'s two windows.

    The loop is `untrained_fixed_point.teaching`'s, which `driven_settling`
    also runs, with one difference and only one: the anneal is constructed at
    `pressure=λ` instead of at its default. That is the independent variable and
    there is no second one.
    """
    diagnostics = Diagnostics(agent.sheaf)
    bias = PredictionRule(agent.sheaf)
    transport = TransportRule(agent.sheaf, anneal=SparsityAnneal(pressure=lam))
    opening: list[np.ndarray] = []
    closing: list[np.ndarray] = []
    first = None
    ranks = None
    for index, _outcome in enumerate(run(agent, ticks, seed=seed)):
        bias.step()
        if agent.sheaf.ticks > 1:
            transport.step()
        if span <= index < 2 * span:
            reading = diagnostics.edge_reading()
            opening.append(reading.energy.detach().numpy())
            first = reading.effective_rank.detach().to(torch.float64).numpy()
        if index >= ticks - span:
            reading = diagnostics.edge_reading()
            closing.append(reading.energy.detach().numpy())
            ranks = reading.effective_rank.detach().to(torch.float64).numpy()
    if not opening or not closing or ranks is None or first is None:
        raise RuntimeError(f"{ticks} ticks is too short for a window of {span}")
    return {
        "opening": np.median(np.stack(opening), axis=0),
        "closing": np.median(np.stack(closing), axis=0),
        "ranks": ranks,
        "first_ranks": first,
        "transmitting": int((ranks > 0).sum()),
        "maps": int(ranks.size),
    }


def conduct(
    env, agent, seed: int, trials: int, window: int, hold: int, probe: float
) -> dict:
    """`detectability`'s trial loop, both directions, on the trained surface.

    `detectability.read` trains its own surface and files cutoffs; neither is
    wanted here, so its parts are called directly. The cast to float64 is
    `detectability.double_precision`, unchanged -- the read's precision argument
    is not this ticket's to relitigate.
    """
    det.double_precision(agent.sheaf)
    dome = agent.dome
    ends = {"rim-to-apex": det.apex(dome), "apex-to-rim": det.rim(dome)}
    picker = np.random.default_rng(seed)
    picks = {d: det.sources(dome, d, picker, trials) for d in ends}
    generator = torch.Generator().manual_seed(seed + 1)
    out: dict[str, list[float]] = {d: [] for d in ends}
    bottleneck: dict[str, list[float]] = {d: [] for d in ends}
    for i in range(trials):
        observation, _info = env.reset(seed=seed * 1000 + i)
        agent.observe(observation)
        applied = np.zeros(env.action_space.shape, dtype=np.float64)
        det.hold_still(agent, observation, applied, hold)
        for direction, targets in ends.items():
            outcome = det.trial(
                agent,
                observation,
                applied,
                picks[direction][i],
                targets,
                generator,
                window,
                probe,
            )
            out[direction].append(float(outcome.conduction))
            bottleneck[direction].append(float(outcome.bottleneck))
    return {
        d: {
            "conduction": quantiles(np.array(out[d])),
            "bottleneck": quantiles(np.array(bottleneck[d])),
            "trials": out[d],
        }
        for d in ends
    }


def one(
    lam: float,
    seed: int,
    ticks: int,
    trials: int,
    window: int,
    hold: int,
    probe: float,
    split: str,
) -> dict:
    """One point of the sweep: train at `λ`, then read all three."""
    started = time.time()
    dome = build_graph(DEFAULT_SPEC)
    edge_width = max((e.m for e in dome.edges), default=1)
    span = max(loop_length.loops(dome).lengths.values())
    env, agent = ufp.build("full", split, seed)
    try:
        trained = train(agent, lam, seed, ticks, span)
        drained = ds.drain(trained, edge_width)

        whole = Diagnostics(agent.sheaf).whole_graph()
        # `chi` is fixed at construction and invariant under learning, so this
        # holds at every `λ` or the reading is wrong. #393 asks for it by name.
        chi = agent.dome.euler_characteristic
        if whole.dim_h0 - whole.dim_h1 != chi:
            raise AssertionError(
                f"dim H0 - dim H1 = {whole.dim_h0 - whole.dim_h1}, not chi = {chi}"
            )

        channel = conduct(env, agent, seed, trials, window, hold, probe)
    finally:
        env.close()
    return {
        "lam": lam,
        "seed": seed,
        "ticks": ticks,
        "split": split,
        "span": span,
        "edge_width": edge_width,
        "seconds": round(time.time() - started, 1),
        "rank": {
            "draining_effective_rank": drained["rank"],
            "draining_effective_rank_opening": drained["was"],
            "draining_edge_share": drained["share"],
            "closing_all_maps": quantiles(trained["ranks"]),
            "opening_all_maps": quantiles(trained["first_ranks"]),
            "transmitting": trained["transmitting"],
            "maps": trained["maps"],
        },
        "cohomology": {
            "rank_delta": whole.rank,
            "dim_h0": whole.dim_h0,
            "dim_h1": whole.dim_h1,
            "chi": chi,
            "minimum_energy": whole.minimum_energy,
        },
        "channel": channel,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lams", type=float, nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--ticks", type=int, default=TICKS)
    parser.add_argument("--trials", type=int, default=TRIALS)
    parser.add_argument("--window", type=int, default=det.WINDOW)
    parser.add_argument("--hold", type=int, default=det.HOLD)
    parser.add_argument("--probe", type=float, default=det.PROBE)
    parser.add_argument("--split", default="train")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--out", default=str(pathlib.Path(__file__).parent))
    args = parser.parse_args(argv)

    torch.set_num_threads(args.threads)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        for lam in args.lams:
            print(f"starting lam={lam:g} seed={seed} ({args.ticks} ticks)", flush=True)
            record = one(
                lam,
                seed,
                args.ticks,
                args.trials,
                args.window,
                args.hold,
                args.probe,
                args.split,
            )
            name = f"393-lam{lam:g}-seed{seed}-{args.ticks}.json"
            (out / name).write_text(json.dumps(record, indent=2), encoding="utf-8")
            rank = record["rank"]
            cohomology = record["cohomology"]
            channel = record["channel"]
            print(
                f"lam={lam:g} seed={seed}: "
                f"rank {rank['draining_effective_rank']:.4g} "
                f"(opening {rank['draining_effective_rank_opening']:.4g})  "
                f"dim H0 {cohomology['dim_h0']}  rank d {cohomology['rank_delta']}  "
                f"conduction in "
                f"{channel['rim-to-apex']['conduction']['median']:.3g} "
                f"out {channel['apex-to-rim']['conduction']['median']:.3g}  "
                f"[{record['seconds']}s] -> {name}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
