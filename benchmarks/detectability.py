"""Rim-to-core detectability: the bottleneck ratio, per edge, over trials (ticket #214).

ADR-0021 fixed the predicate and #214 runs it. Nothing here decides anything —
the predicate, the amplitude convention, the floor and the bar were all settled
before this file existed, and this is the read that returns a verdict against
them::

    python benchmarks/detectability.py read
    python benchmarks/detectability.py linearity

**The predicate.** `max` over rim-to-apex paths `P`, `min` over edges `e ∈ P`, of
`[A₀ · Π_{i ≤ e} hop_i] / floor_e ≥ 1`, with `A₀ = 1` and `k = 1`; per trial,
each trial reduced to the peak ratio at that edge; the bar at the median trial
with p05/p25/p75/p95 alongside; stated twice, rim→apex and apex→rim.

**The numerator is measured, not chained.** `A₀ · Π_{i ≤ e} hop_i` is *the
arriving deviation at edge `e`*, and with `A₀ = 1` a unit-norm deviation injected
at the source stalk makes the paired counterfactual read that product directly.
Multiplying seven separately-measured hops would assume the channel is the
product of graph-wide averages, which is the aggregation error #142 paid for.

**The paired counterfactual, and what the pairing buys.** Two branches forked
from one state, ticked against the same held world, differenced at the same tick:

    dev_e(t)   = ‖ d_e(t) | perturbed  −  d_e(t) | unperturbed ‖
    floor_e(t) = ‖ d_e(t) | unperturbed ‖

where `d_e` is :meth:`~patchworks.tick.Sheaf.disagreement`, the edge's own term of
`xᵀLx`. Both are the same object in the same edge-stalk units — one the
perturbation's contribution to it, the other what stands on the edge without the
perturbation — which is what makes the ratio well-formed at all, and it is the
defect ADR-0021 exists to repair. The floor is read **at the same tick** as the
numerator, which is where ADR-0021's `k = 1` gets its conservatism: excursions
are in the denominator rather than in an invented margin.

**Held world, because the floor is a quiescent-hold floor.** The numerator is
read under the same condition as the denominator: the world is not stepped, the
same observation is written every tick, and what is left is the graph. That
drains the lag floor, which is the driven part ADR-0007 separates and the part
ADR-0021 excludes — including it would make the perturbation beat a quantity it
partly causes. It also makes the counterfactual exact: both branches receive
byte-identical external writes, so every difference between them is the
perturbation, and two unperturbed branches differ by exactly zero.

**Trials sweep configurations.** `reset()` rearranges the world and never the
agent, so a trial is a new pose on the same surface. ADR-0007's static floor is
positional and one pose reports on one point of the overlap; that obligation has
been standing since ADR-0007 and this is the first read to honour it.

**Per edge, never per level** — the Notes' standing rule and ADR-0021's, one
level down from the rule against graph-wide averages. Nothing here is aggregated
across edges: the path reduction is a `min` along one path and a `max` across
paths, both of which pick out an edge rather than averaging over any set of them,
and the profile printed for the median trial names the edges of one path in
order rather than summarising a level.

**The read is taken in float64, and that is not a detail.** See
:func:`double_precision`. Taken in the float32 the architecture runs in, this
measurement reports its own rounding — on the real dome, `2.5e-4` where the
quantity is `1.1e-9`. `linearity` is the check that separates the two, and it is
worth running before believing any number here.

**What the widest path is.** Edge values in hand, the predicate is the classical
maximum-bottleneck path, computed exactly by a max-min Dijkstra rather than
sampled. `tests/test_detectability.py` holds it against closed forms.
"""

import argparse
import sys
from dataclasses import dataclass
from heapq import heappop, heappush
from pathlib import Path

import numpy as np
import torch

from patchworks.agent import Agent
from patchworks.graph import CellKind, Dome

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.append(_HERE)
# The fork's state list, the build and the training loop are imported rather
# than copied: `_TICK_STATE` is held against `Sheaf.assert_no_tape`'s own list by
# `tests/test_untrained_fixed_point.py`, and a second copy drifting from it would
# leave this script forking a branch that shares state with its own control.
import untrained_fixed_point as ufp  # noqa: E402

#: How many ticks each branch is ticked for after the fork. The dome is seven
#: levels deep and one hop is one tick, so this is several times the crossing.
#: The deviation does **not** decay to nothing — the two branches settle on
#: neighbouring trajectories rather than reconverging — so a longer window is
#: partly a maximum over more excursions and creeps upward with the horizon.
#: `read` prints the horizon ladder rather than leaving that in this constant.
WINDOW = 64

#: Ticks of held world before the fork, draining the lag floor so what is left is
#: ADR-0007's static and settling pair. `untrained_fixed_point.sensitivity`'s own
#: hold, for the same reason and at the same length.
HOLD = 400

#: The amplitude the paired difference is taken at, divided back out before
#: anything is reported. `1.0` is ADR-0021's convention taken literally, and in
#: float64 it sits inside the linear window (`linearity` measures that window as
#: `1e-2` to `1e1` on the real dome), so nothing is being rescaled from anywhere.
#: It is a knob only so that the window can be measured.
PROBE = 1.0


def double_precision(root, depth: int = 0, seen: set[int] | None = None) -> int:
    """Cast the trained sheaf to float64 for the read. The measurement needs it.

    **Not a change to the architecture, and not a claim that it should run this
    way.** The surface is trained in float32 exactly as it runs; the cast happens
    after training and before the fork, and it changes the precision the
    *measurement* is done at, not the model being measured.

    The reason is one this map has already paid for twice. #146 struck a depth
    claim on the finding that it located *float32's resolving limit, not the
    graph's*, and #183 re-read a drive hop for the same reason. This quantity is
    squarely in that territory: the deviation arriving at a deep edge is `1e-9`
    of a quiescent disagreement of order `1e-1`, which float32 cannot represent
    as a difference at all. Measured in float32 the two branches still differ —
    they differ by the *rounding* of two nearly-equal trajectories, which does
    not attenuate with depth. So the reading acquires a false floor that is
    largest exactly where the signal is smallest, and it flatters the
    architecture: on the real dome the float32 read returns `2.5e-4` for a
    quantity that is `1.1e-9`, five orders of magnitude of pure arithmetic.

    The tell is amplitude-independence, and it is what `linearity` prints. A
    transported deviation is linear in what was injected, so the bottleneck
    reported at `A₀ = 1` is flat in the amplitude it was measured at; rounding is
    not, so the column falls like `1/A₀`. In float32 on the real dome it falls
    like `1/A₀` across **nine** decades — there is no linear window at all, so no
    choice of amplitude rescues the float32 read. In float64 it is flat to five
    significant figures from `1e-2` to `1e1`.

    A crawl rather than an inventory, and checked rather than trusted: an
    explicit list of tensors is a list that drifts, so this walks what the sheaf
    actually holds, and `linearity`'s flat ladder is the evidence that nothing
    float32 was left behind to quantise the result.
    """
    if seen is None:
        seen = set()
    if id(root) in seen or depth > 6:
        return 0
    seen.add(id(root))
    cast = 0
    if isinstance(root, torch.nn.Module):
        root.double()
    for name in dir(root):
        if name.startswith("__"):
            continue
        try:
            value = getattr(root, name)
        except Exception:
            continue
        if isinstance(value, torch.Tensor):
            if value.dtype == torch.float32:
                try:
                    setattr(root, name, value.to(torch.float64))
                    cast += 1
                except Exception:
                    pass
        elif hasattr(value, "__dict__") and not isinstance(value, (str, bytes, type)):
            cast += double_precision(value, depth + 1, seen)
    return cast


@dataclass(frozen=True)
class Trial:
    """One trial's per-edge peak ratios, and what the widest path did."""

    source: int
    bottleneck: float
    target: int
    edge: int
    """The edge that binds the widest path — what the predicate says fails."""

    path: tuple[int, ...]
    """The binding path's edges, source-first: the descent, edge by edge."""

    ratios: np.ndarray
    """`[edges]`: the peak paired ratio at each edge."""

    peak_at: np.ndarray
    """`[edges]`: the tick each edge's peak landed on, for the window check."""

    horizons: tuple[tuple[int, float], ...]
    """`(ticks, bottleneck)`: the same trial read to several horizons."""


def rim(dome: Dome) -> tuple[int, ...]:
    """The sensorimotor rim: the cells the world writes and reads.

    The drive is excluded. It is a boundary cell and it sits at the *internal*
    rim by `CONTEXT.md`'s own words — at the apex level, not on the sensorimotor
    boundary — so a path that starts or ends there is not the path the predicate
    is about.
    """
    return tuple(
        c.id for c in dome.cells if c.is_boundary and c.kind is not CellKind.DRIVE
    )


def apex(dome: Dome) -> tuple[int, ...]:
    """The predicting cells at the deepest level."""
    deepest = max(c.index.level for c in dome.cells if not c.is_boundary)
    return tuple(
        c.id for c in dome.cells if not c.is_boundary and c.index.level == deepest
    )


def widest_path(
    dome: Dome, values: np.ndarray, source: int, targets: tuple[int, ...]
) -> tuple[float, int, int, tuple[int, ...]]:
    """`max` over paths, `min` over edges: the widest path, where it lands, what stops it.

    A max-min Dijkstra. The invariant that makes the greedy step correct is that
    a path's value cannot rise by extending it, so the unsettled cell of greatest
    value can never be improved by going through a cell of smaller value.

    The **binding edge** comes back with the value, because the predicate's whole
    content is that what fails is an edge rather than a level, and a verdict that
    could not name the edge would not be that predicate's verdict. So does the
    path itself, so the descent can be printed edge by edge.

    The source's own value is `inf` — a zero-edge path is bounded by no edge —
    which is what makes a target adjacent to the source come back with that
    edge's value rather than with `0`.
    """
    best = {source: float("inf")}
    binding = {source: -1}
    came: dict[int, int] = {}
    frontier = [(-float("inf"), source)]
    settled: set[int] = set()
    reached = tuple(t for t in targets if t != source)
    while frontier:
        value, cell = heappop(frontier)
        value = -value
        if cell in settled:
            continue
        settled.add(cell)
        if cell in reached:
            walk, at = [], cell
            while at in came:
                walk.append(came[at])
                at = dome.edges[came[at]].other(at)
            return value, cell, binding[cell], tuple(reversed(walk))
        for edge_id in dome.incident[cell]:
            other = dome.edges[edge_id].other(cell)
            if other in settled:
                continue
            crossing = float(values[edge_id])
            through = min(value, crossing)
            if through > best.get(other, -float("inf")):
                best[other] = through
                binding[other] = edge_id if crossing <= value else binding[cell]
                came[other] = edge_id
                heappush(frontier, (-through, other))
    return 0.0, -1, -1, ()


def unit(width: int, generator: torch.Generator) -> torch.Tensor:
    """`A₀ = 1`: a unit-norm deviation, direction drawn and magnitude fixed."""
    direction = torch.randn((width,), generator=generator, dtype=torch.float64)
    return direction / direction.norm().clamp_min(1e-12)


def branch(agent: Agent, state: dict, observation: dict, applied, ticks: int, nudge):
    """One branch of the fork: restore, optionally nudge, then hold for `ticks`.

    `[ticks, edges, m]` of disagreement comes back — the whole trace rather than
    a running maximum, because the two branches are differenced tick by tick and
    a maximum taken inside a branch would have thrown the pairing away.
    """
    ufp.restore(agent.sheaf, state)
    if nudge is not None:
        cell, deviation = nudge
        with torch.no_grad():
            agent.sheaf.stalks[agent.sheaf.layout.slice(cell)] += deviation
    trace = []
    for _ in range(ticks):
        agent.sheaf.tick()
        trace.append(agent.sheaf.disagreement().clone())
        # The world's write is the tick's last word and both branches get the
        # same one. Nothing is stepped: this is the hold.
        agent.write(observation, applied)
    return torch.stack(trace)


def ratios(
    agent: Agent,
    state: dict,
    quiet: torch.Tensor,
    observation: dict,
    applied,
    source: int,
    deviation: torch.Tensor,
    probe: float,
    window: int,
) -> np.ndarray:
    """`[ticks, edges]`: the paired ratio at every edge and tick, reported at `A₀ = 1`."""
    moved = branch(
        agent, state, observation, applied, window, (source, deviation * probe)
    )
    numerator = (moved - quiet).norm(dim=-1).numpy() / probe
    denominator = quiet.norm(dim=-1).numpy()
    # An edge standing at exactly zero is not a division to guard: it is an edge
    # carrying nothing, which anything at all is distinguishable from. It is
    # reported rather than clamped, because a graph full of them would mean the
    # hold had collapsed and not that transmission had succeeded.
    ratio = np.where(
        denominator > 0, numerator / np.maximum(denominator, 1e-300), np.inf
    )
    return np.where(numerator == 0, 0.0, ratio)


def horizons(window: int) -> tuple[int, ...]:
    """The ladder the bottleneck is reported on, so no single window is silent.

    The map's standing warning is against extrapolating a trend to its limit, and
    its dual applies here: the deviation does not decay to nothing, so a peak
    taken over a longer window is partly a maximum over more excursions. Printing
    the ladder is what makes the horizon's contribution visible instead of buried
    in the choice of :data:`WINDOW`.
    """
    ladder = [h for h in (8, 16, 32, 64, 128, 256) if h <= window]
    if window not in ladder:
        ladder.append(window)
    return tuple(ladder)


def trial(
    agent: Agent,
    observation: dict,
    applied,
    source: int,
    targets: tuple[int, ...],
    generator: torch.Generator,
    window: int,
    probe: float,
) -> Trial:
    """One perturbation: fork, inject, difference, reduce to a bottleneck."""
    state = ufp.snapshot(agent.sheaf)
    quiet = branch(agent, state, observation, applied, window, None)
    deviation = unit(agent.dome.cells[source].stalk, generator)
    ratio = ratios(
        agent, state, quiet, observation, applied, source, deviation, probe, window
    )
    ufp.restore(agent.sheaf, state)
    peaks = ratio.max(axis=0)
    value, target, edge, path = widest_path(agent.dome, peaks, source, targets)
    return Trial(
        source=source,
        bottleneck=value,
        target=target,
        edge=edge,
        path=path,
        ratios=peaks,
        peak_at=ratio.argmax(axis=0),
        horizons=tuple(
            (h, widest_path(agent.dome, ratio[:h].max(axis=0), source, targets)[0])
            for h in horizons(window)
        ),
    )


def quantiles(values: np.ndarray) -> dict:
    finite = values[np.isfinite(values)]
    return {
        "n": int(values.size),
        "infinite": int(values.size - finite.size),
        "p05": float(np.quantile(values, 0.05)),
        "p25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "p75": float(np.quantile(values, 0.75)),
        "p95": float(np.quantile(values, 0.95)),
    }


def sources(dome: Dome, direction: str, generator: np.random.Generator, count: int):
    """Which stalk the unit deviation is injected into, stratified across kinds.

    Stratified rather than drawn uniformly: 256 of the 263 rim cells are patch
    cells, so a uniform draw would report the render's fate under both columns'
    name, and proprioception — which #120 measured dying at a different level —
    would appear in about one trial in forty.
    """
    if direction == "apex-to-rim":
        pool = apex(dome)
        return [int(pool[i % len(pool)]) for i in range(count)]
    kinds = (CellKind.PATCH, CellKind.PROPRIOCEPTIVE, CellKind.TOUCH)
    strata = [[c.id for c in dome.cells if c.kind is kind] for kind in kinds]
    strata = [s for s in strata if s]
    return [int(generator.choice(strata[i % len(strata)])) for i in range(count)]


def hold_still(agent: Agent, observation: dict, applied, hold: int) -> None:
    """Hold the world still, so the lag floor drains out of the reading."""
    for _ in range(hold):
        agent.sheaf.tick()
        agent.write(observation, applied)


def prepared(name: str, split: str, seed: int, learn: int):
    """A surface trained in float32, then cast to float64 for the read."""
    env, agent = ufp.build(name, split, seed)
    print(f"dome {name}: {len(agent.dome.cells)} cells, {len(agent.dome.edges)} edges")
    print(f"training {learn} ticks with both rules, in float32...", flush=True)
    if learn:
        ufp.taught(agent, learn, seed)
    else:
        for _ in ufp.run(agent, 1, seed=seed):
            pass
    cast = double_precision(agent.sheaf)
    print(f"cast {cast} tensors to float64 for the read; see `double_precision`")
    return env, agent


def name_edge(dome: Dome, edge_id: int) -> str:
    if edge_id < 0:
        return "none"
    edge = dome.edges[edge_id]
    u, v = dome.cells[edge.u], dome.cells[edge.v]
    return f"#{edge_id} {edge.kind.value} m={edge.m} {u.index} -- {v.index}"


def linearity(
    name: str,
    split: str,
    seed: int,
    learn: int,
    window: int,
    hold: int,
    amplitudes: tuple[float, ...],
) -> None:
    """The read's own validity check: is the response linear, or is it rounding?

    A transported deviation is linear in what was injected, so the bottleneck
    reported *at `A₀ = 1`* is the same number whatever amplitude it was measured
    at — a flat column. Rounding is not, and shows as a column falling like
    `1/A₀`. Run this before believing anything `read` prints.
    """
    env, agent = prepared(name, split, seed, learn)
    observation, _info = env.reset(seed=seed * 1000)
    agent.observe(observation)
    applied = np.zeros(env.action_space.shape, dtype=np.float64)
    hold_still(agent, observation, applied, hold)
    state = ufp.snapshot(agent.sheaf)
    quiet = branch(agent, state, observation, applied, window, None)

    twice = branch(agent, state, observation, applied, window, None)
    print(f"\nthe fork is exact: max |quiet - quiet| = {float((twice - quiet).abs().max())}")

    source = sources(agent.dome, "rim-to-apex", np.random.default_rng(seed), 1)[0]
    deviation = unit(
        agent.dome.cells[source].stalk, torch.Generator().manual_seed(seed + 1)
    )
    targets = apex(agent.dome)
    print(f"injected at cell {source} ({agent.dome.cells[source].kind.value})")
    print("\n  A0         bottleneck at A0=1     median edge ratio")
    for a0 in amplitudes:
        ratio = ratios(
            agent, state, quiet, observation, applied, source, deviation, a0, window
        )
        peaks = ratio.max(axis=0)
        value, _target, _edge, _path = widest_path(agent.dome, peaks, source, targets)
        finite = peaks[np.isfinite(peaks)]
        print(f"  {a0:<10.3g} {value:<22.5g} {np.median(finite):.5g}")
    ufp.restore(agent.sheaf, state)


def report(dome: Dome, direction: str, outcomes: list[Trial], window: int) -> None:
    values = np.array([o.bottleneck for o in outcomes])
    q = quantiles(values)
    verdict = "HOLDS" if q["median"] >= 1.0 else "FAILS"
    print(f"== {direction}: rim-to-core detectability {verdict} ==")
    print(
        f"   bottleneck ratio over {q['n']} trials — "
        f"p05 {q['p05']:.3g}  p25 {q['p25']:.3g}  "
        f"median {q['median']:.3g}  p75 {q['p75']:.3g}  p95 {q['p95']:.3g}"
    )
    if q["median"] > 0:
        print(f"   short of the bar by {1.0 / q['median']:.3g}x at the median")
    at = np.array([o.peak_at[o.edge] for o in outcomes])
    print(
        f"   the binding edge peaks at tick {np.median(at):.0f} "
        f"(median over trials), in a window of {window}"
    )
    ladder = "  ".join(
        f"{h}:{np.median([dict(o.horizons)[h] for o in outcomes]):.3g}"
        for h, _ in outcomes[0].horizons
    )
    print(f"   median bottleneck by horizon (ticks:value) — {ladder}")
    middle = outcomes[int(np.argsort(values)[len(values) // 2])]
    print(f"   the median trial's binding edge: {name_edge(dome, middle.edge)}")
    profile = "  ".join(f"{middle.ratios[e]:.3g}" for e in middle.path)
    print(f"   its path, edge by edge from the source — {profile}")
    print()


def read(
    name: str,
    split: str,
    seed: int,
    learn: int,
    trials: int,
    window: int,
    hold: int,
    probe: float,
) -> None:
    """The whole read: train once, then sweep configurations, both directions."""
    env, agent = prepared(name, split, seed, learn)
    dome = agent.dome
    ends = {"rim-to-apex": apex(dome), "apex-to-rim": rim(dome)}
    picker = np.random.default_rng(seed)
    picks = {d: sources(dome, d, picker, trials) for d in ends}
    generator = torch.Generator().manual_seed(seed + 1)
    results: dict[str, list[Trial]] = {d: [] for d in ends}

    for i in range(trials):
        observation, _info = env.reset(seed=seed * 1000 + i)
        agent.observe(observation)
        applied = np.zeros(env.action_space.shape, dtype=np.float64)
        hold_still(agent, observation, applied, hold)
        for direction in ends:
            results[direction].append(
                trial(
                    agent,
                    observation,
                    applied,
                    picks[direction][i],
                    ends[direction],
                    generator,
                    window,
                    probe,
                )
            )
        print(
            f"  trial {i + 1}/{trials}: "
            + "  ".join(f"{d} {results[d][-1].bottleneck:.3g}" for d in ends),
            flush=True,
        )

    print()
    for direction, outcomes in results.items():
        report(dome, direction, outcomes, window)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for spoken, help_text in (
        ("read", "the rim-to-core detectability read"),
        ("linearity", "the read's own validity check"),
    ):
        p = commands.add_parser(spoken, help=help_text)
        p.add_argument("--dome", default="full", choices=("small", "full"))
        p.add_argument("--split", default="train")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--learn", type=int, default=30000)
        p.add_argument("--window", type=int, default=WINDOW)
        p.add_argument("--hold", type=int, default=HOLD)
        if spoken == "read":
            p.add_argument("--trials", type=int, default=24)
            p.add_argument("--probe", type=float, default=PROBE)
        else:
            p.add_argument(
                "--amplitudes",
                nargs="+",
                type=float,
                default=[1e-9, 1e-6, 1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2],
            )
    arguments = parser.parse_args(argv)
    if arguments.command == "linearity":
        linearity(
            arguments.dome,
            arguments.split,
            arguments.seed,
            arguments.learn,
            arguments.window,
            arguments.hold,
            tuple(arguments.amplitudes),
        )
        return
    read(
        arguments.dome,
        arguments.split,
        arguments.seed,
        arguments.learn,
        arguments.trials,
        arguments.window,
        arguments.hold,
        arguments.probe,
    )


if __name__ == "__main__":
    main()
