"""A densely connected, undifferentiated graph with the dome's sensorimotor rim.

The user's ruling on #532 (2026-09-11): prototypes are no longer to be read on
the dome. The dome is an intuition-derived shape the record calls *explicitly
abandonable* (`docs/spec/06-graph-topology.md`, *The dome*), B27 (#576) called
its wager lost, and the graph the effort is aimed at is a dense undifferentiated
one carved toward sparsity -- the fallback `06` names and ADR-0031 records as
*rejected for this build, openable masks the reactivation condition*. This
module is that graph as a prototype surface, not as architecture: `src/` is
untouched and `build_graph` still builds the dome.

What is kept from the dome, on purpose: the whole sensorimotor rim, cell for
cell -- the patch tiling, the proprioceptive and touch cells, the actuator and
the drive -- with the same stalks and the same fixed lane widths, because the
rim is the world's shape and not the dome's. The actuator attaches by the
dome's own rule (one edge to the cell covering each joint's proprioception, so a
corrective twitch is three ticks).

What is not: there are no levels, no columns, no covering and no taper. One
population of predicting cells at construction level 1, every pair joined
(`interior` cells, degree `interior - 1`), every sensory cell attached to one
interior cell in a seeded round-robin, the drive attached to the `drive_cells`
interior cells carrying the fewest rim attachments.

**Lane widths are re-derived, not inherited.** `allocate_lane_widths` pins any
same-level interior edge to `spec.lateral_m = 1` -- #540's ruling (c1), which
`graph.py` marks *contingent, not general ... anyone replacing the dome must
re-derive it rather than inherit it*. On a graph with one level that rule would
make every interior lane one wide. So every interior lane here gets the width
the capacity budget affords it evenly: `(capacity_budget - fixed rim spend) //
degree`, minimised over cells, which is what the allocator's own max-min rule
gives a regular graph. The budget invariant is asserted exactly as
`build_graph` asserts it.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import replace

from patchworks.body import NODE_STALK_DIM
from patchworks.graph import (
    CellIndex,
    CellKind,
    Dome,
    DomeSpec,
    _Builder,
    allocate_lane_widths,
)


def build_dense(spec: DomeSpec, interior: int = 15, seed: int = 0, drive_cells: int = 3) -> Dome:
    b = _Builder(spec)

    # -- the sensorimotor rim, as build_graph lays it down --------------------
    patches = [
        b.cell(CellKind.PATCH, spec.patch_stalk, CellIndex(0, "vision", (r, c)))
        for r in range(spec.patch_grid)
        for c in range(spec.patch_grid)
    ]
    proprioceptive: list[int] = []
    somato_sensors: list[int] = []
    for j in range(spec.joints):
        proprioceptive.append(
            b.cell(CellKind.PROPRIOCEPTIVE, spec.proprioceptive_stalk, CellIndex(0, "somatomotor", (2 * j,)))
        )
        somato_sensors.append(proprioceptive[-1])
        somato_sensors.append(
            b.cell(CellKind.TOUCH, spec.touch_stalk, CellIndex(0, "somatomotor", (2 * j + 1,)))
        )
    actuator = b.cell(CellKind.ACTUATOR, spec.actuator_stalk, CellIndex(0, "somatomotor", (2 * spec.joints,)))

    # -- one undifferentiated population ----------------------------------------
    cells = [b.cell(CellKind.PREDICTING, NODE_STALK_DIM, CellIndex(1, "interior", (i,))) for i in range(interior)]
    drive = b.cell(CellKind.DRIVE, spec.drive_stalk, CellIndex(1, "internal rim", (0,)))

    rng = random.Random(seed)
    order = list(range(interior))
    rng.shuffle(order)
    covering: dict[int, int] = {}
    for k, sensory in enumerate(patches + somato_sensors):
        covering[sensory] = cells[order[k % interior]]
        b.edge(sensory, covering[sensory])
    for target in dict.fromkeys(covering[c] for c in proprioceptive):
        b.edge(actuator, target)
    load = Counter(covering.values())
    for target in sorted(cells, key=lambda c: (load.get(c, 0), c))[:drive_cells]:
        b.edge(drive, target)
    for i in range(interior):
        for j in range(i + 1, interior):
            b.edge(cells[i], cells[j])

    # -- lane widths, re-derived for a graph with one level ------------------
    fixed = Counter()
    for e in b.edges:
        if e.m:
            fixed[e.u] += e.m
            fixed[e.v] += e.m
    width = min((spec.capacity_budget - fixed.get(c, 0)) // (interior - 1) for c in cells)
    if width < 1:
        raise ValueError(f"no budget left for interior lanes at degree {interior - 1}")
    sized = allocate_lane_widths(b.cells, b.edges, replace(spec, lateral_m=width))
    spend = Counter()
    for e in sized:
        spend[e.u] += e.m
        spend[e.v] += e.m
    for cell in b.cells:
        if not cell.is_boundary and spend[cell.id] > spec.capacity_budget:
            raise ValueError(f"cell {cell.id} spends {spend[cell.id]} of {spec.capacity_budget}")
    return Dome._assemble(replace(spec, lateral_m=width), tuple(b.cells), sized)
