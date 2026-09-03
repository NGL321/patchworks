"""Rim-to-core detectability: the bottleneck ratio, per edge, over trials (#214, #232).

ADR-0021 fixed the predicate and #214 runs it. Nothing here decides anything —
the predicate, the amplitude convention, the floor and the bar were all settled
before this file existed, and this is the read that returns a verdict against
them::

    python benchmarks/detectability.py read
    python benchmarks/detectability.py corners
    python benchmarks/detectability.py linearity

**`read` carries the cutoff hook** (#284). Two open problems cut on this rig —
[#325](https://github.com/NGL321/patchworks/issues/325) and
[#329](https://github.com/NGL321/patchworks/issues/329) — and both write the
same bar, `conduction ratio >= 1`, because both wait on the same precondition:
*it needs charts from a graph that transmits*. That is the quantity
:func:`report` turns into `HOLDS` or `FAILS`, so `read` states the verdict
against each of them, records the run on the problem, and adds
`register:overdue` on a crossing. It asserts nothing and its exit code does not
move; `tools/cutoff_report.py` holds the argument, and the reading it hands over
is :func:`readings`.

Pass `--no-file` on any read that is not *the* read — a short `--learn`, the
small dome — because filing records a run against those two problems, and a toy
reading is not one.

**`corners` is #232 and it is the one thing here that chose something.** #214
read one corner of a two-axis space, and #230 found it the corner least
favourable to what the architecture claims: the stimulus is added once and the
rim sources are written boundary cells, so it cannot persist; and it enters at
one stalk, where #230 rules inbound rim influence is *collective*. `corners`
varies both axes and reports all four against #214's published baseline, which
stays reproducible because `read` is untouched. The predicate, the floor, `k`,
the median bar and per-edge indexing are ADR-0021's throughout — what changes is
the **stimulus convention** and nothing else. See :func:`injection` for what
*collective* was taken to mean, which is this file's only open call, and
:func:`branch` for what *sustained* was taken to mean.

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

# The cutoff hook lives in `tools/` and not here, because it shells `gh` and a
# network tool belongs on the far side of the line `tests/test_cli.py` defends.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from cutoff_report import report as report_cutoffs  # noqa: E402

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

    source: tuple[int, ...]
    """Every cell the deviation was injected into: one for #214, a stratum for #232."""

    kind: str
    """The stratum injected into, so the corners can be broken down by it."""

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
    dome: Dome, values: np.ndarray, source: int | tuple[int, ...], targets: tuple[int, ...]
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

    **Several sources are the same search, seeded wider.** A collective
    perturbation (#232) has no single source, and the predicate's `max` is over
    rim-to-apex paths, not over paths from one named cell — so every perturbed
    cell starts at `inf` and the search returns the widest path from any of them.
    That is the same object as a virtual super-source joined to each by an
    unbounded edge, and it degenerates to #214's search when there is one.
    """
    origins = (source,) if isinstance(source, int) else tuple(source)
    best = {s: float("inf") for s in origins}
    binding = {s: -1 for s in origins}
    came: dict[int, int] = {}
    frontier = [(-float("inf"), s) for s in origins]
    settled: set[int] = set()
    reached = tuple(t for t in targets if t not in set(origins))
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


def injection(
    dome: Dome,
    cells: tuple[int, ...],
    generator: torch.Generator,
    coherent: bool = True,
) -> tuple[tuple[int, torch.Tensor], ...]:
    """`A₀ = 1` spread over several rim cells: what *collective* is taken to mean.

    **This convention is #232's, and it is the ticket's one open call.** ADR-0021
    fixes `A₀ = 1` and #230 rules that inbound rim influence is *collective*;
    neither says what a coherent many-cell deviation is, so it is stated here
    with its reasoning, the way #214 stated its three instrument calls.

    Two clauses, and each is load-bearing:

    1. **Coherent means the same deviation in each cell's own coordinates.** The
       cells of one rim stratum are exchangeable copies of a single sensor type
       placed at different points of the surface — 256 patch cells of identical
       stalk width, built alike — so a disturbance that is uniform over the
       surface is, in their own coordinates, the same vector in each. That is
       the only common coordinate they have: there is no chart over the rim in
       which to write "the same direction" more strongly, and constructing one
       would be inventing structure the graph does not carry. The alternative
       drawn against this — an independent direction per cell — is available as
       `coherent=False`, and running both is what makes the choice a measurement
       rather than an assertion.

    2. **The whole injection is unit-norm, not each cell's share.** Each of `K`
       cells receives `1/√K` of the deviation, so `‖A₀‖ = 1` over the collective
       exactly as over #214's single stalk. Without this the collective corner
       would inject `√K = 16x` more and beat the baseline by arithmetic; the
       question is whether *coherence* transmits, not whether *more* does. This
       is the clause that keeps the four corners comparable, and it is also the
       clause that makes the collective corner a hard test.

    **Strata, not the whole rim.** The collective is all cells of one kind. The
    rim's kinds have stalk widths 48, 2 and 1, so there is no common coordinate
    across them and clause 1 has nothing to mean; kinds also die at different
    levels by #120, which is why #214 stratified in the first place. Patch is
    the only stratum that is genuinely many — proprioception and touch have
    three cells each — and the report breaks the corners down by kind so that
    the 256-cell read is never averaged into two 3-cell ones.
    """
    widths = {dome.cells[c].stalk for c in cells}
    if len(widths) != 1:
        raise ValueError(f"a collective needs one stalk width, got {sorted(widths)}")
    width = widths.pop()
    scale = 1.0 / np.sqrt(len(cells))
    if coherent:
        shared = unit(width, generator) * scale
        return tuple((c, shared.clone()) for c in cells)
    return tuple((c, unit(width, generator) * scale) for c in cells)


def branch(
    agent: Agent,
    state: dict,
    observation: dict,
    applied,
    ticks: int,
    nudge,
    sustained: dict[int, torch.Tensor] | None = None,
    record: tuple[int, ...] = (),
):
    """One branch of the fork: restore, optionally nudge, then hold for `ticks`.

    `[ticks, edges, m]` of disagreement comes back — the whole trace rather than
    a running maximum, because the two branches are differenced tick by tick and
    a maximum taken inside a branch would have thrown the pairing away. With
    `record`, the named cells' stalks come back alongside it, read at the end of
    each tick — which is what a later branch clamps against.

    **`nudge` is a sequence of `(cell, deviation)`**, one pair being #214's read.

    **`sustained` is what makes the stimulus persist.** #214's stimulus does not:
    the deviation is added once, and the rim sources are written boundary cells,
    so `agent.write` puts them back to quiet on the next tick and what is
    measured is an impulse response with the source clamped. Passed the quiet
    branch's own recorded stalks, this holds each source at *its counterfactual
    value plus the deviation* at the end of every tick — a constant offset from
    the control, maintained for the whole window.

    Clamping against the recorded quiet trace rather than re-adding the
    deviation is what makes *sustained* mean one thing in both directions. At a
    written rim cell the two coincide, since the write has already restored
    quiet. At an apex cell nothing writes, so re-adding would accumulate into a
    ramp — a different stimulus wearing the same name, and one whose growth
    would be read as transmission.
    """
    ufp.restore(agent.sheaf, state)
    layout = agent.sheaf.layout
    pairs = () if nudge is None else tuple(nudge)
    if pairs:
        with torch.no_grad():
            for cell, deviation in pairs:
                agent.sheaf.stalks[layout.slice(cell)] += deviation
    trace = []
    kept: dict[int, list[torch.Tensor]] = {c: [] for c in record}
    for step in range(ticks):
        agent.sheaf.tick()
        trace.append(agent.sheaf.disagreement().clone())
        # The world's write is the tick's last word and both branches get the
        # same one. Nothing is stepped: this is the hold.
        agent.write(observation, applied)
        if sustained is not None:
            with torch.no_grad():
                for cell, deviation in pairs:
                    agent.sheaf.stalks[layout.slice(cell)] = (
                        sustained[cell][step] + deviation
                    )
        for cell in record:
            kept[cell].append(agent.sheaf.stalks[layout.slice(cell)].clone())
    return torch.stack(trace), {c: torch.stack(v) for c, v in kept.items()}


def ratios(
    agent: Agent,
    state: dict,
    quiet: torch.Tensor,
    observation: dict,
    applied,
    nudge,
    probe: float,
    window: int,
    sustained: dict[int, torch.Tensor] | None = None,
) -> np.ndarray:
    """`[ticks, edges]`: the paired ratio at every edge and tick, reported at `A₀ = 1`."""
    scaled = tuple((cell, deviation * probe) for cell, deviation in nudge)
    moved, _ = branch(
        agent, state, observation, applied, window, scaled, sustained=sustained
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

    **The ladder runs past 256 for #232.** #214's flatness from 32 to 64 was
    measured on a decaying pulse and says nothing about a drive held for the
    whole window: a sustained stimulus has a settling time the ladder has to be
    long enough to show, and inheriting the pulse's flatness would assume the
    answer.
    """
    ladder = [h for h in (8, 16, 32, 64, 128, 256, 512, 1024) if h <= window]
    if window not in ladder:
        ladder.append(window)
    return tuple(ladder)


def trial(
    agent: Agent,
    observation: dict,
    applied,
    source,
    targets: tuple[int, ...],
    generator: torch.Generator,
    window: int,
    probe: float,
    stimulus: str = "impulse",
    coherent: bool = True,
    fork=None,
) -> Trial:
    """One perturbation: fork, inject, difference, reduce to a bottleneck.

    `source` is a cell or a tuple of them; `stimulus` is `impulse` (#214's, the
    deviation added once) or `sustained` (#232's, the source held at a constant
    offset for the window). `fork` carries the quiet branch and the recorded
    source stalks when several corners share one — a control run per corner
    would be the same arithmetic four times, and the pairing requires the
    branches be forked from one state anyway.
    """
    state = ufp.snapshot(agent.sheaf)
    cells = (source,) if isinstance(source, int) else tuple(source)
    if fork is None:
        quiet, held = branch(
            agent, state, observation, applied, window, None, record=cells
        )
    else:
        quiet, held = fork
    if len(cells) == 1:
        nudge = ((cells[0], unit(agent.dome.cells[cells[0]].stalk, generator)),)
    else:
        nudge = injection(agent.dome, cells, generator, coherent)
    ratio = ratios(
        agent,
        state,
        quiet,
        observation,
        applied,
        nudge,
        probe,
        window,
        sustained=held if stimulus == "sustained" else None,
    )
    ufp.restore(agent.sheaf, state)
    peaks = ratio.max(axis=0)
    value, target, edge, path = widest_path(agent.dome, peaks, cells, targets)
    return Trial(
        source=cells,
        kind=agent.dome.cells[cells[0]].kind.value,
        bottleneck=value,
        target=target,
        edge=edge,
        path=path,
        ratios=peaks,
        peak_at=ratio.argmax(axis=0),
        horizons=tuple(
            (h, widest_path(agent.dome, ratio[:h].max(axis=0), cells, targets)[0])
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


def sources(
    dome: Dome,
    direction: str,
    generator: np.random.Generator,
    count: int,
    collective: bool = False,
):
    """Which stalk the unit deviation is injected into, stratified across kinds.

    Stratified rather than drawn uniformly: 256 of the 263 rim cells are patch
    cells, so a uniform draw would report the render's fate under both columns'
    name, and proprioception — which #120 measured dying at a different level —
    would appear in about one trial in forty.

    With `collective`, a trial's source is the **whole stratum** rather than one
    cell drawn from it — #230's ruling that inbound rim influence is many-to-few,
    read at the same rotation across kinds so the corners are trial-for-trial
    comparable with #214's.
    """
    if direction == "apex-to-rim":
        pool = apex(dome)
        return [int(pool[i % len(pool)]) for i in range(count)]
    strata = rim_strata(dome)
    if collective:
        return [tuple(strata[i % len(strata)]) for i in range(count)]
    return [int(generator.choice(strata[i % len(strata)])) for i in range(count)]


def rim_strata(dome: Dome) -> list[list[int]]:
    """The rim's exchangeable sets: one list of cell ids per sensor kind."""
    kinds = (CellKind.PATCH, CellKind.PROPRIOCEPTIVE, CellKind.TOUCH)
    strata = [[c.id for c in dome.cells if c.kind is kind] for kind in kinds]
    return [s for s in strata if s]


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
    stimulus: str = "impulse",
    collective: bool = False,
) -> None:
    """The read's own validity check: is the response linear, or is it rounding?

    A transported deviation is linear in what was injected, so the bottleneck
    reported *at `A₀ = 1`* is the same number whatever amplitude it was measured
    at — a flat column. Rounding is not, and shows as a column falling like
    `1/A₀`. Run this before believing anything `read` prints.

    **Each corner needs its own run of this**, which is #232's requirement rather
    than a convenience: a sustained drive is held for the whole window instead of
    decaying, so it explores more of `encode`'s nonlinearity and has no claim on
    the flat window #214 measured on a pulse. `--stimulus` and `--collective`
    pick the corner, and the linear window is a property of the corner.
    """
    env, agent = prepared(name, split, seed, learn)
    observation, _info = env.reset(seed=seed * 1000)
    agent.observe(observation)
    applied = np.zeros(env.action_space.shape, dtype=np.float64)
    hold_still(agent, observation, applied, hold)
    state = ufp.snapshot(agent.sheaf)
    cells = pick(agent.dome, seed, collective)
    quiet, held = branch(
        agent, state, observation, applied, window, None, record=cells
    )

    twice, _ = branch(agent, state, observation, applied, window, None)
    print(f"\nthe fork is exact: max |quiet - quiet| = {float((twice - quiet).abs().max())}")

    generator = torch.Generator().manual_seed(seed + 1)
    if len(cells) == 1:
        nudge = ((cells[0], unit(agent.dome.cells[cells[0]].stalk, generator)),)
    else:
        nudge = injection(agent.dome, cells, generator)
    targets = apex(agent.dome)
    kind = agent.dome.cells[cells[0]].kind.value
    print(
        f"corner: {stimulus} x {'collective' if collective else 'single-source'}; "
        f"injected into {len(cells)} {kind} cell(s)"
    )
    print("\n  A0         bottleneck at A0=1     median edge ratio")
    for a0 in amplitudes:
        ratio = ratios(
            agent,
            state,
            quiet,
            observation,
            applied,
            nudge,
            a0,
            window,
            sustained=held if stimulus == "sustained" else None,
        )
        peaks = ratio.max(axis=0)
        value, _target, _edge, _path = widest_path(agent.dome, peaks, cells, targets)
        finite = peaks[np.isfinite(peaks)]
        print(f"  {a0:<10.3g} {value:<22.5g} {np.median(finite):.5g}")
    ufp.restore(agent.sheaf, state)


def pick(dome: Dome, seed: int, collective: bool) -> tuple[int, ...]:
    """`linearity`'s one source: #214's first stratified draw, or its whole stratum."""
    drawn = sources(
        dome, "rim-to-apex", np.random.default_rng(seed), 1, collective=collective
    )[0]
    return (drawn,) if isinstance(drawn, int) else tuple(drawn)


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
    strata = sorted({o.kind for o in outcomes})
    if len(strata) > 1:
        breakdown = "  ".join(
            f"{k}:{np.median([o.bottleneck for o in outcomes if o.kind == k]):.3g}"
            for k in strata
        )
        # Never averaged into one number: patch is 256 cells and the other two
        # are three each, so a collective read means something different in each.
        print(f"   median by stratum — {breakdown}")
    middle = outcomes[int(np.argsort(values)[len(values) // 2])]
    print(f"   the median trial's binding edge: {name_edge(dome, middle.edge)}")
    profile = "  ".join(f"{middle.ratios[e]:.3g}" for e in middle.path)
    print(f"   its path, edge by edge from the source — {profile}")
    print()


def readings(results: dict[str, list[Trial]]) -> dict[str, float]:
    """What this read has to offer a `measurement` cutoff, by name.

    **`conduction_ratio` is the lower of the two directions**, because a graph
    that transmits one way and not the other does not transmit: the bar #325 and
    #329 both write, `conduction ratio >= 1`, is the same quantity :func:`report`
    turns into `HOLDS` or `FAILS`, at the median trial, and taking the minimum
    is the reading that cannot say *holds* while a direction fails. The two
    directions are also offered by name, so a cutoff that means one of them can
    say so.
    """
    medians = {
        direction: float(np.median([o.bottleneck for o in outcomes]))
        for direction, outcomes in results.items()
        if outcomes
    }
    if not medians:
        return {}
    found = {
        f"conduction_ratio_{direction.replace('-', '_')}": value
        for direction, value in medians.items()
    }
    found["conduction_ratio"] = min(medians.values())
    return found


def read(
    name: str,
    split: str,
    seed: int,
    learn: int,
    trials: int,
    window: int,
    hold: int,
    probe: float,
    file_cutoffs: bool = True,
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
    # The measurement half of the cutoff mechanism (#284). #325 and #329 both
    # cut on this rig, and both wait on the same thing: a graph that transmits,
    # which is what this read measures. It states a verdict, files the run, and
    # asserts nothing.
    report_cutoffs("detectability", readings(results), file=file_cutoffs)


#: The two-axis space #232 reads, inbound. #214 measured the first row's first
#: column and it is the corner least favourable to what the architecture claims:
#: a stimulus that cannot persist, from a single sensor's private disturbance.
CORNERS = (
    ("impulse", "single-source"),
    ("impulse", "collective"),
    ("sustained", "single-source"),
    ("sustained", "collective"),
)

#: The collective corners run twice, coherent and not, so `injection`'s clause 1
#: is measured instead of asserted. Same cells, same total norm, same everything
#: else — the only difference is whether the cells were given one direction or
#: `K` of them, so the gap between the two rows *is* what coherence is worth.
CONTRAST = "collective (incoherent)"


def corners(
    name: str,
    split: str,
    seed: int,
    learn: int,
    trials: int,
    window: int,
    hold: int,
    probe: float,
    contrast: bool,
) -> None:
    """All four corners of the inbound stimulus space, against #214's baseline.

    **One quiet branch per trial, shared by every corner.** The control does not
    depend on the perturbation, and re-running it per corner would be the same
    arithmetic six times over — but more than that, the corners must be forked
    from *one* state to be compared with each other at all. The recorded stalks
    that come back with it are what the sustained corners clamp against.

    **Outbound stays single-source, in all four**, per #230: apex influence on
    the rim is claimed to be individually strong, so one apex cell is the correct
    instrument there and the collective axis has nothing to mean. It is read at
    both stimuli, because the sustained axis is about the stimulus and applies to
    either direction.
    """
    env, agent = prepared(name, split, seed, learn)
    dome = agent.dome
    inward, outward = apex(dome), rim(dome)
    picker = np.random.default_rng(seed)
    singles = sources(dome, "rim-to-apex", picker, trials)
    collectives = sources(
        dome, "rim-to-apex", np.random.default_rng(seed), trials, collective=True
    )
    outbound = sources(dome, "apex-to-rim", picker, trials)
    generator = torch.Generator().manual_seed(seed + 1)

    runs: list[tuple[str, str, str]] = [
        (f"rim-to-apex  {stimulus} x {breadth}", stimulus, breadth)
        for stimulus, breadth in CORNERS
    ]
    if contrast:
        runs += [
            (f"rim-to-apex  {stimulus} x {CONTRAST}", stimulus, CONTRAST)
            for stimulus in ("impulse", "sustained")
        ]
    runs += [
        (f"apex-to-rim  {stimulus} x single-source", stimulus, "outbound")
        for stimulus in ("impulse", "sustained")
    ]
    results: dict[str, list[Trial]] = {label: [] for label, _, _ in runs}

    for i in range(trials):
        observation, _info = env.reset(seed=seed * 1000 + i)
        agent.observe(observation)
        applied = np.zeros(env.action_space.shape, dtype=np.float64)
        hold_still(agent, observation, applied, hold)
        touched = tuple(
            sorted({singles[i], outbound[i], *collectives[i]})
        )
        state = ufp.snapshot(agent.sheaf)
        fork = branch(
            agent, state, observation, applied, window, None, record=touched
        )
        ufp.restore(agent.sheaf, state)
        for label, stimulus, breadth in runs:
            if breadth == "outbound":
                source, targets = outbound[i], outward
            elif breadth == "single-source":
                source, targets = singles[i], inward
            else:
                source, targets = collectives[i], inward
            results[label].append(
                trial(
                    agent,
                    observation,
                    applied,
                    source,
                    targets,
                    generator,
                    window,
                    probe,
                    stimulus=stimulus,
                    coherent=breadth != CONTRAST,
                    fork=fork,
                )
            )
        print(
            f"  trial {i + 1}/{trials}: "
            + "  ".join(
                f"{label.split('  ')[1]} {results[label][-1].bottleneck:.3g}"
                for label, _, _ in runs
                if label.startswith("rim")
            ),
            flush=True,
        )

    print()
    for label, _, _ in runs:
        report(dome, label, results[label], window)
    table(results, runs)


def table(results: dict[str, list[Trial]], runs) -> None:
    """The 2x2 the ticket asks for, with the baseline's own corner in place."""
    print("== the four corners, inbound: median bottleneck ratio ==")
    print(f"   {'':<24}{'impulse':<16}sustained")
    for breadth in ("single-source", "collective", CONTRAST):
        cells = []
        for stimulus in ("impulse", "sustained"):
            label = f"rim-to-apex  {stimulus} x {breadth}"
            if label in results:
                cells.append(f"{np.median([o.bottleneck for o in results[label]]):.3g}")
            else:
                cells.append("-")
        print(f"   {breadth:<24}{cells[0]:<16}{cells[1]}")
    print(
        "\n   #214's published baseline is impulse x single-source: 8.7e-10 "
        "rim->apex, 1.3e-8 apex->rim."
    )
    print("   The finding is the size of the gap between corners, not any one of them.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for spoken, help_text in (
        ("read", "the rim-to-core detectability read"),
        ("corners", "all four corners of the inbound stimulus space (#232)"),
        ("linearity", "the read's own validity check"),
    ):
        p = commands.add_parser(spoken, help=help_text)
        p.add_argument("--dome", default="full", choices=("small", "full"))
        p.add_argument("--split", default="train")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--learn", type=int, default=30000)
        p.add_argument("--window", type=int, default=WINDOW)
        p.add_argument("--hold", type=int, default=HOLD)
        if spoken in ("read", "corners"):
            p.add_argument("--trials", type=int, default=24)
            p.add_argument("--probe", type=float, default=PROBE)
        if spoken == "read":
            p.add_argument(
                "--no-file",
                dest="file_cutoffs",
                action="store_false",
                help=(
                    "print the cutoff report and file nothing on the tracker. "
                    "Use it for a read that is not the read -- a short --learn, "
                    "the small dome -- because filing records a run against "
                    "#325 and #329 and a toy reading is not one"
                ),
            )
        if spoken == "corners":
            p.add_argument(
                "--no-contrast",
                dest="contrast",
                action="store_false",
                help="skip the incoherent collective rows",
            )
        if spoken == "linearity":
            p.add_argument(
                "--stimulus", default="impulse", choices=("impulse", "sustained")
            )
            p.add_argument("--collective", action="store_true")
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
            arguments.stimulus,
            arguments.collective,
        )
        return
    if arguments.command == "corners":
        corners(
            arguments.dome,
            arguments.split,
            arguments.seed,
            arguments.learn,
            arguments.trials,
            arguments.window,
            arguments.hold,
            arguments.probe,
            arguments.contrast,
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
        arguments.file_cutoffs,
    )


if __name__ == "__main__":
    main()
