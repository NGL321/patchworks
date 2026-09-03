"""`|loop(c)|` per cell, enumerated from the mask (#361, ADR-0026).

[#242](https://github.com/NGL321/patchworks/issues/242) flagged its own
`|loop(apex)| = 14` as **not verified** -- taken as *7 hops out and 7 back* from
#230's seven-hop figure rather than enumerated on the graph.
[ADR-0026](../../docs/adr/0026-rim-core-influence-is-a-conduction-ratio.md)
specified the enumeration and published its ladder; this is that enumeration as
**code**, so a session pairing `|loop(c)|` against a per-cell reading does not
re-key an ADR table by level.

**Per cell, never per level.** A loop is a property of the graph; a level is a
property of the shape imposed on it ([#181](https://github.com/NGL321/patchworks/issues/181)).
That `|loop(c)| = 2 * level` on `DEFAULT_SPEC` is a coincidence of the current
taper, and :func:`check_against_adr_0026` asserts the coincidence rather than
assuming it -- on a changed `DomeSpec` the assertion is expected to fail and the
per-cell numbers are still right.

Two readings, because ADR-0026 records both and says they agree on this dome:

- :func:`loop_lengths` -- `2 * d(c, rim)`, the **round trip**, where the outbound
  and return paths may retrace each other.
- :func:`disjoint_cycle_lengths` -- the shortest **genuine cycle** through `c`
  that reaches the rim and returns, outbound and return sharing no vertex but
  `c` itself. Solved exactly as a min-cost flow of two units out of `c` into the
  rim, with every vertex capacity 1 (`c` itself 2) so the two paths are
  vertex-disjoint by construction.

The **drive boundary cell is not part of the rim**: it sits at the internal rim,
attached to the apex, and ADR-0026 excludes it explicitly.

Usage::

    PYTHONPATH=src python prototypes/admissible-band-361/loops.py
"""

from __future__ import annotations

import heapq
from collections import deque

from patchworks.graph import DEFAULT_SPEC, CellKind, Dome, DomeSpec, build_graph

#: The sensorimotor rim. `DRIVE` is deliberately absent -- ADR-0026: *"The drive
#: boundary cell is not part of the rim for this purpose."*
RIM_KINDS = (
    CellKind.PATCH,
    CellKind.PROPRIOCEPTIVE,
    CellKind.TOUCH,
    CellKind.ACTUATOR,
)

#: ADR-0026's published ladder for `DEFAULT_SPEC`, kept only so
#: :func:`check_against_adr_0026` can assert this code reproduces it. Nothing
#: here reads `|loop|` off a level.
ADR_0026_LADDER = {1: 2, 2: 4, 3: 6, 4: 8, 5: 10, 6: 12, 7: 14}


def rim_ids(dome: Dome) -> tuple[int, ...]:
    """Cell ids of the sensorimotor rim, in cell order."""
    return tuple(c.id for c in dome.cells if c.kind in RIM_KINDS)


def adjacency(dome: Dome) -> list[list[int]]:
    """`[cells][neighbours]`, undirected, from the edge list."""
    neighbours: list[list[int]] = [[] for _ in dome.cells]
    for edge in dome.edges:
        neighbours[edge.u].append(edge.v)
        neighbours[edge.v].append(edge.u)
    return neighbours


def distance_to_rim(dome: Dome) -> list[int]:
    """`[cells]`: hops to the nearest sensorimotor rim cell, `-1` if unreachable.

    One breadth-first sweep seeded at the whole rim at once, which is the same
    thing as the per-cell minimum over rim sources and is one pass rather than
    263.
    """
    neighbours = adjacency(dome)
    distance = [-1] * len(dome.cells)
    queue: deque[int] = deque()
    for cell_id in rim_ids(dome):
        distance[cell_id] = 0
        queue.append(cell_id)
    while queue:
        current = queue.popleft()
        for other in neighbours[current]:
            if distance[other] < 0:
                distance[other] = distance[current] + 1
                queue.append(other)
    return distance


def loop_lengths(dome: Dome) -> dict[int, int]:
    """`{predicting cell id: |loop(c)|}` as `2 * d(c, rim)`, the round trip."""
    distance = distance_to_rim(dome)
    lengths = {}
    for cell_id in dome.predicting:
        if distance[cell_id] < 0:
            raise ValueError(f"cell {cell_id} never reaches the sensorimotor rim")
        lengths[cell_id] = 2 * distance[cell_id]
    return lengths


def _min_cost_two_paths(
    neighbours: list[list[int]], source: int, sinks: frozenset[int], cells: int
) -> int | None:
    """Total edges in the two cheapest **vertex-disjoint** `source`-to-rim paths.

    A min-cost flow of two units on the vertex-split graph: `v_in -> v_out` with
    capacity 1 carries the disjointness (the source gets 2), every graph edge
    becomes a unit-capacity arc in both directions costing 1, and each rim cell
    offers one unit to a super sink. `None` when two disjoint paths do not
    exist, which is a cell whose genuine cycle does not close at all.

    Successive shortest paths with Johnson potentials -- costs are non-negative
    to begin with and stay so under the potentials, so each augmentation is one
    Dijkstra rather than a Bellman-Ford.
    """
    # Node ids: v_in = 2v, v_out = 2v + 1, super sink = 2 * cells.
    sink = 2 * cells
    total = sink + 1
    graph: list[list[list[int]]] = [[] for _ in range(total)]

    def arc(u: int, v: int, capacity: int, cost: int) -> None:
        graph[u].append([v, capacity, cost, len(graph[v])])
        graph[v].append([u, 0, -cost, len(graph[u]) - 1])

    for cell_id in range(cells):
        arc(2 * cell_id, 2 * cell_id + 1, 2 if cell_id == source else 1, 0)
    for cell_id, adjacent in enumerate(neighbours):
        for other in adjacent:
            arc(2 * cell_id + 1, 2 * other, 1, 1)
    for cell_id in sinks:
        arc(2 * cell_id + 1, sink, 1, 0)

    start = 2 * source
    potential = [0] * total
    cost_total = 0
    for _ in range(2):
        distance: list[int | None] = [None] * total
        distance[start] = 0
        previous: list[tuple[int, int] | None] = [None] * total
        heap = [(0, start)]
        while heap:
            so_far, node = heapq.heappop(heap)
            if distance[node] is None or so_far > distance[node]:
                continue
            for index, (target, capacity, cost, _rev) in enumerate(graph[node]):
                if capacity <= 0:
                    continue
                reduced = cost + potential[node] - potential[target]
                nxt = so_far + reduced
                if distance[target] is None or nxt < distance[target]:
                    distance[target] = nxt
                    previous[target] = (node, index)
                    heapq.heappush(heap, (nxt, target))
        if distance[sink] is None:
            return None
        for node in range(total):
            if distance[node] is not None:
                potential[node] += distance[node]
        node = sink
        while node != start:
            parent, index = previous[node]
            graph[parent][index][1] -= 1
            graph[node][graph[parent][index][3]][1] += 1
            cost_total += graph[parent][index][2]
            node = parent
    return cost_total


def disjoint_cycle_lengths(dome: Dome) -> dict[int, int | None]:
    """`{predicting cell id: shortest genuine rim-returning cycle through c}`.

    `None` where no such cycle exists. Distinct from :func:`loop_lengths`, which
    permits the return to retrace the outbound path; ADR-0026 found the two
    agree at the apex on `DEFAULT_SPEC` and this reads them at every cell.
    """
    neighbours = adjacency(dome)
    sinks = frozenset(rim_ids(dome))
    cells = len(dome.cells)
    return {
        cell_id: _min_cost_two_paths(neighbours, cell_id, sinks, cells)
        for cell_id in dome.predicting
    }


def check_against_adr_0026(dome: Dome, lengths: dict[int, int]) -> dict[str, object]:
    """Does the per-cell enumeration reproduce ADR-0026's published ladder?

    Reported, not asserted, and expected to disagree on a changed `DomeSpec` --
    the ladder is `DEFAULT_SPEC`'s reading, and the point of computing per cell
    is that a different dome does not need the table rewritten first.
    """
    by_level: dict[int, set[int]] = {}
    for cell_id, length in lengths.items():
        level = dome.cells[cell_id].index.level
        by_level.setdefault(level, set()).add(length)
    exact = all(len(values) == 1 for values in by_level.values())
    matches = exact and all(
        next(iter(values)) == ADR_0026_LADDER.get(level)
        for level, values in by_level.items()
    )
    return {
        "one_length_per_level": exact,
        "matches_adr_0026_ladder": matches,
        "lengths_by_level": {
            level: sorted(values) for level, values in sorted(by_level.items())
        },
    }


def main(spec: DomeSpec = DEFAULT_SPEC) -> None:
    dome = build_graph(spec)
    rim = rim_ids(dome)
    lengths = loop_lengths(dome)
    cycles = disjoint_cycle_lengths(dome)

    kinds: dict[str, int] = {}
    for cell_id in rim:
        key = dome.cells[cell_id].kind.value
        kinds[key] = kinds.get(key, 0) + 1
    print(
        f"{len(dome.cells)} cells, {len(dome.edges)} edges; sensorimotor rim "
        f"{len(rim)} cells (" + ", ".join(f"{v} {k}" for k, v in kinds.items()) + ")"
    )
    print(f"{len(dome.predicting)} predicting cells\n")

    print("  level  cells  d(c,rim)  |loop(c)|  genuine cycle  agree")
    by_level: dict[int, list[int]] = {}
    for cell_id in dome.predicting:
        by_level.setdefault(dome.cells[cell_id].index.level, []).append(cell_id)
    for level, ids in sorted(by_level.items()):
        loops = sorted({lengths[c] for c in ids})
        genuine = sorted({cycles[c] for c in ids}, key=lambda v: (v is None, v))
        agree = all(cycles[c] == lengths[c] for c in ids)
        print(
            f"  L{level:<5} {len(ids):>5} {'/'.join(str(v // 2) for v in loops):>9} "
            f"{'/'.join(str(v) for v in loops):>10} "
            f"{'/'.join('none' if v is None else str(v) for v in genuine):>14} "
            f"{'yes' if agree else 'NO':>6}"
        )

    check = check_against_adr_0026(dome, lengths)
    print(
        f"\n  d(c, rim) exact within a level: {check['one_length_per_level']}; "
        f"reproduces ADR-0026's ladder: {check['matches_adr_0026_ladder']}"
    )
    unclosed = [c for c, v in cycles.items() if v is None]
    print(
        f"  Cells with no genuine rim-returning cycle: {len(unclosed)}"
        f"/{len(dome.predicting)}"
    )
    longer = [c for c, v in cycles.items() if v is not None and v > lengths[c]]
    shorter = [c for c, v in cycles.items() if v is not None and v < lengths[c]]
    print(
        f"  Genuine cycle longer than the round trip at {len(longer)}"
        f"/{len(dome.predicting)} cells, shorter at {len(shorter)}. A shorter one "
        "would\n  put a cell's true loop below what `2 * d(c, rim)` claims and "
        "weaken the bar;\n  none exists, so the round trip is the conservative "
        "reading at every cell."
    )
    drive = [c.id for c in dome.cells if c.kind is CellKind.DRIVE]
    if drive:
        distance = distance_to_rim(dome)
        print(f"  Drive boundary cell's own distance to the rim: {distance[drive[0]]}")


if __name__ == "__main__":
    main()
