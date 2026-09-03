"""`|loop(c)|` on the mask, with and without ADR-0016's written-or-read split (#351, for #343).

ADR-0026's predicate is `τ̂_c / |loop(c)| ≥ 1`, and `|loop(c)|` is its divisor:
the tick length of the shortest cycle through `c` that reaches the rim and
returns. The ADR enumerated it once, on `DEFAULT_SPEC`, and recorded the ladder
`|loop(c)| = 2 · d(c, rim)` — 2 at L1 to 14 at the apex. **Nothing in
`benchmarks/` computed it**, which is the gap #351 names, and it is the gap that
left [#343](https://github.com/NGL321/patchworks/issues/343) `uncut`::

    python benchmarks/loop_length.py ladder
    python benchmarks/loop_length.py split
    python benchmarks/loop_length.py world

**`ladder`** recomputes ADR-0026's table from the graph rather than quoting it,
which is the ADR's own instruction — *"A changed `DomeSpec` changes these
numbers, and they are recomputed rather than quoted"*.

**`split` is #343's question and carries the cutoff hook.** #343 states that
under ADR-0016's ban — a boundary cell is written by the world or read by it,
never both — a motor panel's proprioceptive feedback must land on a *different*
boundary cell than the actuator that caused it, so the shortest command-to-
consequence loop grows by at least two ticks; and since `|loop(c)|` is ADR-0026's
divisor, the split raises the bar the architecture has to clear. That is a claim
about a divisor, and a divisor can be measured. `split` builds the counterfactual
graph in which the ban is lifted — the actuator fused into a joint's
proprioceptive cell, one written-and-read motor panel, which is the interface
the sandbox actually presents — and reads `|loop(c)|` on both.

**It asserts nothing**, like every script here. What it offers a `measurement`
cutoff is :func:`readings`, and the one metric that answers #343 is
`loop_split_cost`: the largest number of ticks the split adds to any predicting
cell's loop, over the fleet. It is a **count of ticks**, not a tuned level, and
#343's own words fix the bar at `>= 2` — *"grows by at least two ticks"* is the
failure as stated, so the problem stops being tolerable exactly when the graph
shows it. A reading of 0 says the divisor never moved and the claim is about
something other than ADR-0026's bar.

**`world` is [#368](https://github.com/NGL321/patchworks/issues/368)'s question,
and it is the other loop.** #362's grilling session found that `|loop(c)|` is not
the loop ADR-0026 argues for. The argument is *the cell still holds what it sent
by the time the answer gets back*, and for anything at the sensorimotor rim the
answer gets back through the **world**: an actuator writes, the world responds,
and under ADR-0016's ban a *different* boundary cell reads the consequence. That
loop is a construction-time graph quantity like its sibling::

    world_loop(c) = min over (a, p), a an actuator, p a proprioceptor, a != p
                    of  d(c, a) + w + d(p, c)

`|loop(c)|` collapses it by allowing `a = p` and setting `w = 0`. The metric this
arm offers a cutoff is `world_loop_excess`: the largest number of ticks
`world_loop(c)` exceeds `|loop(c)|` over the predicting fleet, at the world tick
the sandbox fixes. A reading of 0 would say the two names denote one quantity and
#368's failure does not occur; anything `>= 1` says the bar divides by the
shorter of two different lengths, which is the failure as stated. The bar is
#368's and was argued there, not here.

**The reading cannot be 0, and the reason is the same one that made `split` read
0** — `|loop(c)|` measures to the rim *as a set*. That set contains both `a` and
`p`, so `d(c, rim) ≤ min(d(c, a), d(c, p))` and

    world_loop(c) = d(c, a) + w + d(p, c) ≥ 2 · d(c, rim) + w = |loop(c)| + w

at every cell, on any `DomeSpec` carrying an actuator and a proprioceptor. So the
excess is at least `w` everywhere and the fleet max is at least `w` — the bar is
crossed by construction, not by this taper. What the rig measures is how far
*above* that floor the graph actually sits, which is the number a divisor
correction has to carry and is not derivable in closed form: it is the gap
between the nearest rim cell and the nearest **actuator**, and there is exactly
one actuator among 263 rim cells.

Both `ladder`'s and `split`'s readings are exact integers off the mask, and so is
this one. There is no seed, no run and no
world here: `|loop(c)|` is a construction-time quantity, which is why this rig is
cheap enough to carry a cutoff that would otherwise wait on a driven run.

**The reading on `DEFAULT_SPEC` is 0, and the reason is structural rather than a
coincidence of this taper.** `|loop(c)| = 2 · d(c, rim)` is a distance to the rim
**as a set**. Fusing two rim cells creates exactly one new shortcut, and that
shortcut runs *through* a cell already at distance 0 — so it can shorten no
predicting cell's distance to the set, and removing a vertex that no shortest
path to the set passes *through* can lengthen none either. Any fusion of rim
cells into rim cells therefore leaves the whole ladder fixed, on any `DomeSpec`.
The rig still computes it rather than asserting it, for ADR-0026's reason: the
argument is about the round-trip reading, and the numbers are the thing a later
`DomeSpec` changes.
"""

from __future__ import annotations

import argparse
import collections
import pathlib
import sys
from dataclasses import dataclass

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))

from patchworks.graph import (  # noqa: E402
    DEFAULT_SPEC,
    Cell,
    CellKind,
    Dome,
    DomeSpec,
    build_graph,
)

from cutoff_report import report as report_cutoffs  # noqa: E402

#: The rim `|loop(c)|` is measured to. ADR-0026 is explicit that the **drive**
#: boundary cell is not part of it: it sits at the internal rim, attached to all
#: eight apex cells, and its own distance to the sensorimotor rim is 8. Reading
#: the loop to the drive cell would hand the apex a `|loop|` of 2 and dissolve
#: the bar by redefining it.
SENSORIMOTOR = (
    CellKind.PATCH,
    CellKind.PROPRIOCEPTIVE,
    CellKind.TOUCH,
    CellKind.ACTUATOR,
)


@dataclass(frozen=True)
class Loops:
    """`|loop(c)|` for every predicting cell, and the rim it was measured to."""

    lengths: dict[int, int]
    rim: tuple[int, ...]
    unreachable: tuple[int, ...]
    """Predicting cells no walk from the rim reaches. Reported, never dropped."""


def adjacency(dome: Dome, fuse: dict[int, int] | None = None) -> dict[int, set[int]]:
    """Cell-to-cell adjacency, optionally with some cells fused into others.

    `fuse` maps a cell id onto the id it is merged into, which is how the
    counterfactual graph is built without a second constructor: fusing is a
    relabelling of endpoints, and every edge the fused cell carried becomes an
    edge on its host. That keeps the two arms of :func:`split` the *same* mask
    read two ways, rather than two builds that could differ for a second reason.
    """
    fuse = fuse or {}
    neighbours: dict[int, set[int]] = collections.defaultdict(set)
    for edge in dome.edges:
        u, v = fuse.get(edge.u, edge.u), fuse.get(edge.v, edge.v)
        if u == v:
            # A self-loop, which fusion can create and which carries no tick.
            continue
        neighbours[u].add(v)
        neighbours[v].add(u)
    return neighbours


def rim_of(dome: Dome, fuse: dict[int, int] | None = None) -> tuple[int, ...]:
    """The sensorimotor rim, after fusion, as cell ids in id order."""
    fuse = fuse or {}
    return tuple(
        sorted({fuse.get(c.id, c.id) for c in dome.cells if c.kind in SENSORIMOTOR})
    )


def loops(dome: Dome, fuse: dict[int, int] | None = None) -> Loops:
    """`|loop(c)| = 2 · d(c, rim)` for every predicting cell, by breadth-first sweep.

    The round-trip reading, which is ADR-0026's: the ADR checked it against the
    *genuine* vertex-disjoint cycle on the default dome and found the two agree
    at the apex, so the predicate does not turn on which is meant. This computes
    the round trip, states which reading it is, and leaves the other to the ADR
    that checked it — a rig that silently swapped readings between arms would
    make the comparison below meaningless.
    """
    fuse = fuse or {}
    neighbours = adjacency(dome, fuse)
    rim = rim_of(dome, fuse)
    distance: dict[int, int] = {cell: 0 for cell in rim}
    queue = collections.deque(rim)
    while queue:
        cell = queue.popleft()
        for other in neighbours[cell]:
            if other not in distance:
                distance[other] = distance[cell] + 1
                queue.append(other)
    predicting = [c for c in dome.predicting if fuse.get(c, c) == c]
    return Loops(
        lengths={c: 2 * distance[c] for c in predicting if c in distance},
        rim=rim,
        unreachable=tuple(c for c in predicting if c not in distance),
    )


def motor_fusion(dome: Dome) -> dict[int, int]:
    """ADR-0016's ban lifted: the actuator fused into a joint's proprioceptor.

    There is one actuator cell and `joints` proprioceptive cells, so the fusion
    is one-to-one in the graph even though the world has one interface per
    joint. Fusing every proprioceptor into a single host would merge the joints,
    which the world does not do. So the actuator is fused into the joint-0
    proprioceptor and the rest are left alone: that is the smallest edit that
    removes the split at one joint, and #343's claim is about *a* motor panel,
    not about all of them at once. If the split costs ticks anywhere, it costs
    them here.
    """
    proprioceptors = [c.id for c in dome.cells if c.kind is CellKind.PROPRIOCEPTIVE]
    actuators = [c.id for c in dome.cells if c.kind is CellKind.ACTUATOR]
    if not proprioceptors or not actuators:
        return {}
    return {actuator: proprioceptors[0] for actuator in actuators}


#: The world's own tick `w`, in the units `d(c, a)` and `d(p, c)` are counted in:
#: what the crossing from an actuator to the proprioceptor that reads its
#: consequence costs, in graph ticks.
#:
#: **The sandbox fixes it, and it fixes it at 1.** `src/patchworks/agent.py`'s
#: ordering is two graph phases and then the world: `tick()` runs `sheaf.tick()`,
#: reads the command off the actuator cell's stalk, steps the environment, and
#: writes what comes back onto the proprioceptive stalks as the same tick's last
#: word. So a command written on tick `t` is answered on `p`'s stalk at the end
#: of `t` and read by `p`'s neighbours on `t + 1` — exactly the latency of one
#: graph edge, which is why the crossing is a hop and not free. The world's
#: *settling* adds nothing on top of that: `FRAME_SKIP` physics steps of the
#: arena's `timestep` are one control tick by construction
#: (`CONTROL_HZ = PHYSICS_HZ / FRAME_SKIP`, `src/patchworks/sandbox/env.py`), so
#: the consequence is inside the same `env.step` that read the command. A world
#: whose response outlasted its own control period would raise this and nothing
#: else in the rig, which is why the excess is also reported at `w = 0`: the
#: graph-only half of the gap is then separable from the world's contribution.
WORLD_TICK = 1


@dataclass(frozen=True)
class WorldLoops:
    """`world_loop(c)` for every predicting cell, and what fixed it."""

    lengths: dict[int, int]
    world_tick: int
    actuators: tuple[int, ...]
    proprioceptors: tuple[int, ...]
    unreachable: tuple[int, ...]
    """Predicting cells no actuator-and-proprioceptor pair reaches. Reported."""


def distances_from(neighbours: dict[int, set[int]], source: int) -> dict[int, int]:
    """Hop distance from one cell, by breadth-first sweep. Unreached cells absent."""
    distance = {source: 0}
    queue = collections.deque([source])
    while queue:
        cell = queue.popleft()
        for other in neighbours[cell]:
            if other not in distance:
                distance[other] = distance[cell] + 1
                queue.append(other)
    return distance


def world_loops(dome: Dome, world_tick: int = WORLD_TICK) -> WorldLoops:
    """#368's loop: out through an actuator, across the world, back elsewhere.

    `world_loop(c) = min over (a, p), a an actuator, p a proprioceptor, a != p,
    of d(c, a) + w + d(p, c)`. This is the loop ADR-0026 *argues* for — *the cell
    still holds what it sent by the time the answer gets back* — and under
    ADR-0016's written-or-read ban the answer cannot return at the cell that
    sent it, so `a != p` is the ban rather than a modelling choice. `|loop(c)|`
    is what this collapses to when `a = p` is allowed and `w` is set to 0:
    exactly the two things ADR-0016 and the sandbox forbid.

    The `min` is the same `min` :func:`loops` takes over the rim — a cell sits on
    the shortest loop available to it, and a bar read against a longer route it
    could have avoided would be a bar about the route rather than about the
    cell. Distances are undirected because the mask is: `d(c, a)` and `d(a, c)`
    are one sweep, and the tick that carries a message up an edge carries one
    down it.

    Like `|loop(c)|` this is an exact integer off the mask — no seed, no run and
    no world — and `world_tick` is the world's only entry into it.
    """
    neighbours = adjacency(dome)
    actuators = tuple(sorted(c.id for c in dome.cells if c.kind is CellKind.ACTUATOR))
    proprioceptors = tuple(
        sorted(c.id for c in dome.cells if c.kind is CellKind.PROPRIOCEPTIVE)
    )
    reach = {
        cell: distances_from(neighbours, cell)
        for cell in sorted(set(actuators) | set(proprioceptors))
    }
    lengths: dict[int, int] = {}
    unreachable: list[int] = []
    for cell in dome.predicting:
        candidates = [
            reach[a][cell] + world_tick + reach[p][cell]
            for a in actuators
            for p in proprioceptors
            if a != p and cell in reach[a] and cell in reach[p]
        ]
        if candidates:
            lengths[cell] = min(candidates)
        else:
            unreachable.append(cell)
    return WorldLoops(
        lengths=lengths,
        world_tick=world_tick,
        actuators=actuators,
        proprioceptors=proprioceptors,
        unreachable=tuple(unreachable),
    )


def excesses(dome: Dome, world_tick: int = WORLD_TICK) -> dict[int, int]:
    """`world_loop(c) - |loop(c)|` per predicting cell, where both are defined.

    Reported as a `max` over the fleet by :func:`readings`, for #343's reason
    read at the other loop: the bar is `max` over paths of `min` over cells, so
    a claim about a worst case is answered by a worst case. The per-cell dict is
    kept because the distribution is the thing a later `DomeSpec` moves, and a
    single number would hide a fleet split between cells that sit on both loops
    at once and cells that do not.
    """
    graph_loop = loops(dome).lengths
    world = world_loops(dome, world_tick).lengths
    return {c: world[c] - graph_loop[c] for c in sorted(set(graph_loop) & set(world))}


def by_level(dome: Dome, found: Loops) -> dict[int, tuple[int, set[int]]]:
    """The ladder: cells and the `|loop|` values seen, keyed by construction level.

    ADR-0026 is emphatic that `|loop(c)| = 2 · level` is a fact about this
    graph's wiring and **not a licence to index by level** (#181). So the level
    is the row label and the `|loop|` values are a *set*: a level whose cells
    disagree prints more than one number rather than an average that would hide
    it.
    """
    ladder: dict[int, tuple[int, set[int]]] = {}
    cells: dict[int, Cell] = {c.id: c for c in dome.cells}
    for cell, length in found.lengths.items():
        level = cells[cell].index.level
        count, values = ladder.get(level, (0, set()))
        ladder[level] = (count + 1, values | {length})
    return ladder


def ladder(spec: DomeSpec) -> Loops:
    """Recompute ADR-0026's table from the mask and print it."""
    dome = build_graph(spec)
    found = loops(dome)
    rows = by_level(dome, found)
    print(f"\n== |loop(c)| on {len(dome.cells)} cells, {len(dome.edges)} edges ==")
    print(f"   sensorimotor rim: {len(found.rim)} cells")
    print("\n   level  cells  |loop(c)|")
    for level in sorted(rows):
        count, values = rows[level]
        shown = ", ".join(str(v) for v in sorted(values))
        print(f"   {level:>5}  {count:>5}  {shown}")
    if found.unreachable:
        print(f"\n   unreachable from the rim: {len(found.unreachable)} cells")
    print(
        "\n   ADR-0026's ladder, recomputed rather than quoted. A different "
        "`DomeSpec`\n   has a different ladder, and that is the point of "
        "computing it."
    )
    return found


def readings(spec: DomeSpec = DEFAULT_SPEC) -> dict[str, float]:
    """What `split` has to offer a `measurement` cutoff, by name.

    `loop_split_cost` is the metric #343 cuts on: **the largest number of ticks
    ADR-0016's split adds to any predicting cell's `|loop(c)|`**. The maximum
    rather than the median, because #343's claim is that the split raises *the
    bar*, and the bar is `max` over paths of `min` over cells — a claim about a
    worst case is answered by a worst case.

    `loop_apex` and `loop_apex_fused` are offered alongside so a cutoff that
    means the apex specifically can say so; the apex is where ADR-0026's
    shortfall is read.

    `world_loop_excess` is the metric #368 cuts on: **the largest number of ticks
    `world_loop(c)` exceeds `|loop(c)|` over the predicting fleet**, at the world
    tick the sandbox fixes
    (:data:`WORLD_TICK`). `world_loop_excess_w0` is the same reading with the
    world set to zero, so the graph-only half of the gap — the ban on returning
    at the cell that sent, and the actuator being one cell rather than the whole
    rim — is separable from the world's own contribution. The two differ by
    exactly :data:`WORLD_TICK` by construction; both are reported because *which
    half of the gap this is* is the question a reader of the bar arrives with.
    """
    dome = build_graph(spec)
    split_arm = loops(dome)
    fused = loops(dome, motor_fusion(dome))
    shared = set(split_arm.lengths) & set(fused.lengths)
    if not shared:
        return {}
    gaps = excesses(dome)
    zero_world = excesses(dome, world_tick=0)
    return {
        "loop_split_cost": float(
            max(split_arm.lengths[c] - fused.lengths[c] for c in shared)
        ),
        "loop_apex": float(max(split_arm.lengths.values())),
        "loop_apex_fused": float(max(fused.lengths.values())),
        **(
            {
                "world_loop_excess": float(max(gaps.values())),
                "world_loop_excess_w0": float(max(zero_world.values())),
            }
            if gaps
            else {}
        ),
    }


def split(spec: DomeSpec, *, file: bool = True) -> None:
    """#343: read `|loop(c)|` with the split and without it, and state the cost."""
    dome = build_graph(spec)
    fusion = motor_fusion(dome)
    kept = loops(dome)
    fused = loops(dome, fusion)
    shared = sorted(set(kept.lengths) & set(fused.lengths))

    print("\n== ADR-0016's written-or-read split, priced on |loop(c)| ==")
    print(
        f"   {len(fusion)} actuator cell(s) fused into a joint's proprioceptor: "
        f"one written-and-read motor panel."
    )
    print(f"   rim, split: {len(kept.rim)} cells; fused: {len(fused.rim)} cells")

    moved = [c for c in shared if kept.lengths[c] != fused.lengths[c]]
    print(f"\n   predicting cells compared: {len(shared)}")
    print(f"   cells whose |loop(c)| moved: {len(moved)}")
    if moved:
        cells = {c.id: c for c in dome.cells}
        for cell in moved[:10]:
            index = cells[cell].index
            print(
                f"     L{index.level} {index.column}{index.position}: "
                f"{fused.lengths[cell]} fused -> {kept.lengths[cell]} split"
            )
    print(
        f"\n   apex |loop|: {max(kept.lengths.values())} split, "
        f"{max(fused.lengths.values())} fused"
    )
    print(
        "\n   What this prices is ADR-0026's divisor and nothing else. A "
        "command-to-\n   consequence path through the *world* may well be "
        "longer under the ban;\n   #343's claim is that the split raises "
        "ADR-0026's bar, and the bar is this."
    )
    report_cutoffs("loop_length", readings(spec), file=file)


def world(spec: DomeSpec, *, file: bool = True) -> None:
    """#368: read `world_loop(c)` beside `|loop(c)|`, and state the excess."""
    dome = build_graph(spec)
    graph_loop = loops(dome)
    crossing = world_loops(dome)
    gaps = excesses(dome)
    zero_world = excesses(dome, world_tick=0)
    cells: dict[int, Cell] = {c.id: c for c in dome.cells}

    print("\n== ADR-0026's divisor against the loop it argues for ==")
    print(
        f"   {len(crossing.actuators)} actuator cell(s), "
        f"{len(crossing.proprioceptors)} proprioceptive cell(s); "
        f"a != p is ADR-0016's ban."
    )
    print(f"   world tick w = {crossing.world_tick}, fixed by the sandbox's ordering.")

    rows: dict[int, tuple[int, set[int], set[int], set[int]]] = {}
    for cell, gap in gaps.items():
        level = cells[cell].index.level
        count, loop_values, world_values, gap_values = rows.get(
            level, (0, set(), set(), set())
        )
        rows[level] = (
            count + 1,
            loop_values | {graph_loop.lengths[cell]},
            world_values | {crossing.lengths[cell]},
            gap_values | {gap},
        )

    print("\n   level  cells  |loop(c)|  world_loop(c)  excess")
    for level in sorted(rows):
        count, loop_values, world_values, gap_values = rows[level]
        print(
            f"   {level:>5}  {count:>5}  "
            f"{', '.join(str(v) for v in sorted(loop_values)):>9}  "
            f"{', '.join(str(v) for v in sorted(world_values)):>13}  "
            f"{', '.join(str(v) for v in sorted(gap_values)):>6}"
        )
    if crossing.unreachable:
        print(
            f"\n   reached by no (actuator, proprioceptor) pair: "
            f"{len(crossing.unreachable)} cells"
        )

    print(
        f"\n   excess over the fleet: max {max(gaps.values())}, "
        f"min {min(gaps.values())} ticks"
        f"\n   with the world at w = 0: max {max(zero_world.values())}, "
        f"min {min(zero_world.values())} ticks"
    )
    print(
        "\n   Where the excess is positive the conduction ratio divides by the "
        "shorter of\n   two different lengths, which is #368's failure. The bar "
        "is >= 1 because 1 is\n   the integer separating *the same quantity* "
        "from *two different quantities*."
    )
    report_cutoffs("loop_length", readings(spec), file=file)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=("ladder", "split", "world"))
    parser.add_argument(
        "--no-file",
        action="store_true",
        help=(
            "print the cutoff report and touch the tracker not at all. Pass it "
            "on any read that is not *the* read."
        ),
    )
    arguments = parser.parse_args(argv)
    if arguments.mode == "ladder":
        ladder(DEFAULT_SPEC)
    elif arguments.mode == "split":
        split(DEFAULT_SPEC, file=not arguments.no_file)
    else:
        world(DEFAULT_SPEC, file=not arguments.no_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
