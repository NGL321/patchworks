"""`|loop(c)|` on the mask, with and without ADR-0016's written-or-read split (#351, for #343).

ADR-0026's predicate is `τ̂_c / |loop(c)| ≥ 1`, and `|loop(c)|` is its divisor:
the tick length of the shortest cycle through `c` that reaches the rim and
returns. The ADR enumerated it once, on `DEFAULT_SPEC`, and recorded the ladder
`|loop(c)| = 2 · d(c, rim)` — 2 at L1 to 14 at the apex. **Nothing in
`benchmarks/` computed it**, which is the gap #351 names, and it is the gap that
left [#343](https://github.com/NGL321/patchworks/issues/343) `uncut`::

    python benchmarks/loop_length.py ladder
    python benchmarks/loop_length.py split

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

Both readings are exact integers off the mask. There is no seed, no run and no
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
    """
    dome = build_graph(spec)
    split_arm = loops(dome)
    fused = loops(dome, motor_fusion(dome))
    shared = set(split_arm.lengths) & set(fused.lengths)
    if not shared:
        return {}
    return {
        "loop_split_cost": float(
            max(split_arm.lengths[c] - fused.lengths[c] for c in shared)
        ),
        "loop_apex": float(max(split_arm.lengths.values())),
        "loop_apex_fused": float(max(fused.lengths.values())),
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=("ladder", "split"))
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
    else:
        split(DEFAULT_SPEC, file=not arguments.no_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
