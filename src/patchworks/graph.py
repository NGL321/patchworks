"""The dome: graph construction, the structural masks, and what construction records.

The shape is a **taper from a two-dimensional boundary sheet to a deep core**
(`docs/spec/06-graph-topology.md`, *The dome*). The base is the sensorimotor
boundary — a 16x16 tiling of the render, plus the somatomotor cluster; levels
inward are successively coarser lattices; the apex is a small deep core which is
also the **internal rim**, where the drive boundary cell attaches.

:func:`build_graph` returns a :class:`Dome`. Everything a later ticket needs is
on it: the cells, the edges, the edge stalk dimensions, the structural masks,
and the private-component projection. :meth:`Dome.report` prints the
construction diagnostics — the numbers that say it was built right.

Three commitments are structural here rather than configurable.

* **The construction layout is an index, not an embedding.** Cells are indexed
  by level and lattice position (:class:`CellIndex`). No cell has a coordinate
  and no distance kernel generates the mask. The index generates the mask at
  construction; after that it has **no runtime role** and is used only for
  plotting. Every runtime accessor on :class:`Dome` reads a precomputed array,
  never a cell's index — asserted in `tests/test_graph.py`.
* **Boundary cells are exempt from `n`.** Their node stalk is the world's shape:
  48 for a sensory patch, 2 proprioceptive, 1 touch, 6 actuator, 1 drive
  (`docs/adr/0006-boundary-cell-stalks-are-world-shaped.md`). Every edge stalk
  in the graph, boundary-incident ones included, is ordinary and `m`-sized.
* **Sparsity is a property of the maps, not of the graph.** No edge is ever
  removed, and the mask closes and never re-opens. Nothing here deletes an edge
  or narrows a stalk, and nothing later may either.

Where the record leaves a construction rule open, this module chooses one and
says so at the point of choice. Those points are the somatomotor column's
internal wiring, the touch cell's stalk dimension, and the index order that puts
the actuator in the same L1 cell as a joint's proprioception.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import torch

from .body import BodyShape

__all__ = [
    "CellIndex",
    "CellKind",
    "Cell",
    "Edge",
    "EdgeKind",
    "Dome",
    "DomeSpec",
    "DEFAULT_SPEC",
    "build_graph",
]


class CellKind(str, Enum):
    """What kind of cell this is — a predicting cell, or one of the boundary kinds.

    `CONTEXT.md` has three kinds of boundary cell: sensory, actuator, and drive,
    the first two at the sensorimotor rim and the third at the internal rim.
    The sensory kinds are split by what the world writes into them, because that
    is what sizes their node stalk.
    """

    PREDICTING = "predicting"
    PATCH = "patch"
    PROPRIOCEPTIVE = "proprioceptive"
    TOUCH = "touch"
    ACTUATOR = "actuator"
    DRIVE = "drive"

    @property
    def is_boundary(self) -> bool:
        return self is not CellKind.PREDICTING


class EdgeKind(str, Enum):
    """Sorted by **who clears the disagreement**, which is ADR-0003's taxonomy.

    A drive edge is a motor edge by that test — its disagreement is cleared by
    the world moving, eventually rather than immediately — and is named
    separately only because its stalk width and attachment rule differ.
    """

    SENSORY = "sensory"
    MOTOR = "motor"
    DRIVE = "drive"
    INTERIOR = "interior"


@dataclass(frozen=True)
class CellIndex:
    """A cell's place in the **construction layout**: a level and a lattice position.

    An index, never a metric embedding. `position` is a tuple of integers — grid
    coordinates on a vision lattice, a single ordinal in the somatomotor column
    or a core level. Nothing here is a distance, and nothing reads it at runtime.
    """

    level: int
    column: str
    position: tuple[int, ...]

    def __str__(self) -> str:
        return f"L{self.level}/{self.column}{self.position}"


@dataclass(frozen=True)
class Cell:
    """One node of the graph.

    `stalk` is the node stalk dimension: `n` for a predicting cell, the world's
    shape for a boundary cell. `index` is the construction layout entry and is
    plot-only.
    """

    id: int
    kind: CellKind
    stalk: int
    index: CellIndex

    @property
    def is_boundary(self) -> bool:
        return self.kind.is_boundary


@dataclass(frozen=True)
class Edge:
    """One edge, carrying an edge stalk of dimension `m`.

    `m` is fixed here and never changes: 4 between two predicting cells, 8 on a
    boundary-incident edge, 1 on a drive edge. Boundary edges are wider because
    a patch cell's edges are the only route that patch's information ever takes;
    the drive edge is narrower because a map out of a one-dimensional stalk has
    rank at most one.
    """

    id: int
    u: int
    v: int
    m: int
    kind: EdgeKind

    def other(self, cell_id: int) -> int:
        if cell_id == self.u:
            return self.v
        if cell_id == self.v:
            return self.u
        raise ValueError(f"cell {cell_id} is not an endpoint of edge {self.id}")


@dataclass(frozen=True)
class DomeSpec:
    """Every count in the dome is a construction parameter, not a constant.

    The defaults are `docs/spec/06-graph-topology.md`'s. `core_degree` and
    `apex_degree` are targets the lateral fill works to: the apex is lower
    because L7 has no predicting level above it and loses its up-edges by
    construction, which is what the private-dimension gradient depends on.
    """

    patch_grid: int = 16
    """Side of the sensory tiling. 16x16 4x4-px patches over a 64x64 render."""

    vision_sides: tuple[int, ...] = (8, 4)
    """Sides of the L1 and L2 vision lattices, each over a 2x2 block below."""

    somatomotor_sizes: tuple[int, ...] = (6, 4)
    """Cells in the somatomotor column at L1 and L2."""

    core_sizes: tuple[int, ...] = (16, 14, 12, 10, 8)
    """Cells at L3-L7. The last entry is the apex, and the internal rim."""

    joints: int = 3
    """Arm joints, one proprioceptive and one touch boundary cell each."""

    n: int = 32
    """Node stalk dimension of a predicting cell."""

    k: int = 12
    """Chart dimension."""

    interior_m: int = 4
    boundary_m: int = 8
    drive_m: int = 1

    patch_stalk: int = 48
    """4x4 px RGB, raw. The world writes it with no compressor in between."""

    proprioceptive_stalk: int = 2
    """Angle and velocity."""

    touch_stalk: int = 1
    """One contact scalar per joint.

    **Chosen here, not recorded.** `06-graph-topology.md`'s dimension table sizes
    the patch, proprioceptive, actuator and drive stalks and omits touch. ADR-0006
    settles the rule rather than the number — a boundary cell's stalk has whatever
    dimension the thing writing it gives it — and the sandbox's `touch` observation
    is `(3,)`, one scalar per joint, over three touch cells.
    """

    actuator_stalk: int = 6
    """Three commanded, three efference."""

    drive_stalk: int = 1
    """Valence, not specification."""

    core_degree: int = 6
    apex_degree: int = 4

    def __post_init__(self) -> None:
        if self.patch_grid % 2 or self.patch_grid < 2:
            raise ValueError(f"patch_grid must be even and >= 2, got {self.patch_grid}")
        sides = (self.patch_grid,) + tuple(self.vision_sides)
        for below, above in zip(sides, sides[1:]):
            if below != 2 * above:
                raise ValueError(
                    "each vision lattice covers a 2x2 block of the one below; "
                    f"{below} does not taper to {above}"
                )
        if not self.vision_sides:
            raise ValueError("the taper needs at least one vision lattice")
        if len(self.somatomotor_sizes) != len(self.vision_sides):
            raise ValueError(
                "the somatomotor column runs parallel to the vision lattices, so "
                "it has a level for each of them; got "
                f"{len(self.somatomotor_sizes)} against {len(self.vision_sides)}"
            )
        if len(self.core_sizes) < 2:
            raise ValueError("the core needs at least two levels")
        if min(self.core_sizes) < 2:
            raise ValueError(f"every core level holds at least two cells, got {self.core_sizes}")
        if self.apex_degree >= self.core_degree:
            raise ValueError(
                "the apex is lower-degree than the rest of the core by "
                "construction (docs/spec/06-graph-topology.md, Connectivity)"
            )


DEFAULT_SPEC = DomeSpec()


def _covers(i: int, below: int, above: int, fan: int) -> tuple[int, ...]:
    """The `fan` cells at the level above that cover cell `i` of the level below.

    A proportional covering over the two levels' indices: cell `i` maps to
    `floor(i * above / below)` and, for `fan > 1`, the cells following it
    cyclically. At `fan = 1` this is a partition — every cell below has exactly
    one up-edge, which is what the recorded cut capacities require of L0, L1 and
    L2. Inside the core `fan = 2`, which is what makes the apex's lost up-edges
    worth two of its six.

    On a level narrower than the fan the offsets collide. The collision is
    deduplicated rather than laid down twice: two cells are joined by one edge or
    by none, and there are no parallel edges anywhere in the dome.
    """
    base = (i * above) // below
    return tuple(dict.fromkeys((base + t) % above for t in range(fan)))


def _lateral_fill(size: int, deficit: list[int]) -> tuple[list[tuple[int, int]], int]:
    """Sparse lateral edges within one core level, to a per-cell degree target.

    Candidate pairs are enumerated by stride over the level's cyclic index —
    `(i, i+1)`, then `(i, i+2)`, and so on — and accepted when both endpoints
    are still short of the level's target degree. Index-generated, deterministic,
    and no lattice: the core's laterals are a circulant fill, not a grid.

    The stride sweep is greedy and can strand a cell whose every partner is
    already satisfied, so a repair pass follows it, pairing the cells furthest
    from their target first. Returns the edges and how much degree the level is
    still short — which the caller refuses to build on, because the guaranteed
    private dimension the taper exists to produce is read straight off these
    degrees.
    """
    remaining = list(deficit)
    taken: set[tuple[int, int]] = set()
    edges: list[tuple[int, int]] = []

    def join(a: int, b: int) -> None:
        pair = (min(a, b), max(a, b))
        taken.add(pair)
        edges.append(pair)
        remaining[a] -= 1
        remaining[b] -= 1

    for stride in range(1, size // 2 + 1):
        # Within a stride the cells furthest from their target are offered the
        # edge first. Taking the level in plain index order instead lets an
        # early cell spend a partner a later one had no other way to reach.
        for i in sorted(range(size), key=lambda c: (-remaining[c], c)):
            j = (i + stride) % size
            if i == j or (min(i, j), max(i, j)) in taken:
                continue
            if remaining[i] > 0 and remaining[j] > 0:
                join(i, j)

    while True:
        short = sorted(
            (i for i in range(size) if remaining[i] > 0),
            key=lambda c: (-remaining[c], c),
        )
        if len(short) < 2:
            break
        head, rest = short[0], short[1:]
        partner = next(
            (c for c in rest if (min(head, c), max(head, c)) not in taken), None
        )
        if partner is None:
            break
        join(head, partner)

    return edges, sum(remaining)


class _Builder:
    """Assembles cells and edges. Reads the construction layout; nothing else does."""

    def __init__(self, spec: DomeSpec) -> None:
        self.spec = spec
        self.cells: list[Cell] = []
        self.edges: list[Edge] = []

    def cell(self, kind: CellKind, stalk: int, index: CellIndex) -> int:
        cell = Cell(id=len(self.cells), kind=kind, stalk=stalk, index=index)
        self.cells.append(cell)
        return cell.id

    def edge(self, u: int, v: int) -> None:
        """Add one edge, sizing its stalk and sorting its kind from its endpoints."""
        kinds = (self.cells[u].kind, self.cells[v].kind)
        if CellKind.DRIVE in kinds:
            m, kind = self.spec.drive_m, EdgeKind.DRIVE
        elif CellKind.ACTUATOR in kinds:
            m, kind = self.spec.boundary_m, EdgeKind.MOTOR
        elif any(k.is_boundary for k in kinds):
            m, kind = self.spec.boundary_m, EdgeKind.SENSORY
        else:
            m, kind = self.spec.interior_m, EdgeKind.INTERIOR
        self.edges.append(Edge(id=len(self.edges), u=u, v=v, m=m, kind=kind))


def _grid_lateral(ids: dict[tuple[int, int], int], side: int) -> list[tuple[int, int]]:
    """Four-neighbour lateral edges within one vision lattice."""
    pairs = []
    for r in range(side):
        for c in range(side):
            if c + 1 < side:
                pairs.append((ids[(r, c)], ids[(r, c + 1)]))
            if r + 1 < side:
                pairs.append((ids[(r, c)], ids[(r + 1, c)]))
    return pairs


def build_graph(spec: DomeSpec = DEFAULT_SPEC) -> "Dome":
    """Build the dome: cells, edges, edge stalks, structural masks.

    Deterministic and free of any draw — the whole thing follows from the level
    and lattice indices. Two calls with the same spec give identical graphs.
    """
    b = _Builder(spec)

    # -- L0, the sensorimotor boundary -----------------------------------
    # The sensory tiling. Adjacent patches are adjacent cells, so retinotopy
    # falls out of the index rather than being designed.
    patch_ids: dict[tuple[int, int], int] = {}
    for r in range(spec.patch_grid):
        for c in range(spec.patch_grid):
            patch_ids[(r, c)] = b.cell(
                CellKind.PATCH, spec.patch_stalk, CellIndex(0, "vision", (r, c))
            )

    # The somatomotor cluster, in one index order. The actuator sits at the head
    # of it, which is what puts it in the same L1 cell as the first joint's
    # proprioception and makes the reflex loop three ticks by construction
    # rather than by hand-wiring. The order is a construction-layout choice the
    # record leaves open; it is made here to satisfy a requirement the record
    # does state.
    somato_l0: list[int] = [
        b.cell(CellKind.ACTUATOR, spec.actuator_stalk, CellIndex(0, "somatomotor", (0,)))
    ]
    for j in range(spec.joints):
        somato_l0.append(
            b.cell(
                CellKind.PROPRIOCEPTIVE,
                spec.proprioceptive_stalk,
                CellIndex(0, "somatomotor", (2 * j + 1,)),
            )
        )
        somato_l0.append(
            b.cell(
                CellKind.TOUCH,
                spec.touch_stalk,
                CellIndex(0, "somatomotor", (2 * j + 2,)),
            )
        )

    # -- L1 and L2, the vision lattices and the parallel somatomotor column --
    vision_levels: list[dict[tuple[int, int], int]] = []
    somato_levels: list[list[int]] = []
    for depth, (side, column_size) in enumerate(
        zip(spec.vision_sides, spec.somatomotor_sizes), start=1
    ):
        ids = {
            (r, c): b.cell(CellKind.PREDICTING, spec.n, CellIndex(depth, "vision", (r, c)))
            for r in range(side)
            for c in range(side)
        }
        vision_levels.append(ids)
        somato_levels.append(
            [
                b.cell(
                    CellKind.PREDICTING, spec.n, CellIndex(depth, "somatomotor", (p,))
                )
                for p in range(column_size)
            ]
        )

    # -- L3 to L7, the core; the apex is also the internal rim ---------------
    core_levels: list[list[int]] = []
    first_core = 1 + len(spec.vision_sides)
    for offset, size in enumerate(spec.core_sizes):
        depth = first_core + offset
        core_levels.append(
            [
                b.cell(CellKind.PREDICTING, spec.n, CellIndex(depth, "core", (i,)))
                for i in range(size)
            ]
        )
    apex = core_levels[-1]
    apex_level = first_core + len(spec.core_sizes) - 1

    drive = b.cell(
        CellKind.DRIVE, spec.drive_stalk, CellIndex(apex_level, "internal rim", (0,))
    )

    # -- Vertical: the block covered below, the cell covering above ---------
    # Each L0, L1 and L2 cell has exactly one up-edge. That is what the recorded
    # cut capacities say: 263 x 8, 70 x 4, 20 x 4.
    for (r, c), patch in patch_ids.items():
        b.edge(patch, vision_levels[0][(r // 2, c // 2)])
    for p, cell in enumerate(somato_l0):
        (target,) = _covers(p, len(somato_l0), len(somato_levels[0]), fan=1)
        b.edge(cell, somato_levels[0][target])

    for below, above in zip(vision_levels, vision_levels[1:]):
        for (r, c), cell in below.items():
            b.edge(cell, above[(r // 2, c // 2)])
    for below_col, above_col in zip(somato_levels, somato_levels[1:]):
        for p, cell in enumerate(below_col):
            (target,) = _covers(p, len(below_col), len(above_col), fan=1)
            b.edge(cell, above_col[target])

    # L2 into L3, where the modalities first share a cell. The vision lattice
    # maps one-to-one onto the core level by row-major index; the column, four
    # cells against sixteen, spreads evenly around it, so the cells that hear
    # both modalities are spread through the core rather than bunched at one
    # seam.
    l2_vision, l2_somato, l3 = vision_levels[-1], somato_levels[-1], core_levels[0]
    side = spec.vision_sides[-1]
    for (r, c), cell in l2_vision.items():
        (target,) = _covers(r * side + c, len(l2_vision), len(l3), fan=1)
        b.edge(cell, l3[target])
    for j, cell in enumerate(l2_somato):
        b.edge(cell, l3[(j * len(l3)) // len(l2_somato)])

    # Within the core each cell has two up-edges, to the two cells covering it.
    # The apex has none — there is no predicting level above it — which is
    # exactly the two edges of six that its degree of four is short.
    for below, above in zip(core_levels, core_levels[1:]):
        for i, cell in enumerate(below):
            for target in _covers(i, len(below), len(above), fan=2):
                b.edge(cell, above[target])

    # -- Lateral -----------------------------------------------------------
    for ids, side in zip(vision_levels, spec.vision_sides):
        for u, v in _grid_lateral(ids, side):
            b.edge(u, v)

    # The somatomotor column is laterally complete at each of its levels.
    # **Chosen here, not recorded**: the record gives four-neighbour laterals for
    # the vision lattices and "sparse" for the core, and says nothing about the
    # column. Complete is what keeps a column cell's degree in the same range as
    # its vision neighbours' and so keeps L1's guaranteed private dimension at
    # zero, which is the recorded gradient. At six and four cells it is also the
    # only rule on a set this small that is not an arbitrary ordering.
    for column in somato_levels:
        for a in range(len(column)):
            for c in range(a + 1, len(column)):
                b.edge(column[a], column[c])

    # The core's laterals fill each cell to its level's degree target.
    degree = [0] * len(b.cells)
    for e in b.edges:
        degree[e.u] += 1
        degree[e.v] += 1
    for level_index, level in enumerate(core_levels):
        target = (
            spec.apex_degree if level_index == len(core_levels) - 1 else spec.core_degree
        )
        deficit = [max(0, target - degree[cell]) for cell in level]
        lateral, shortfall = _lateral_fill(len(level), deficit)
        # One cell may be a single edge short when the level's total deficit is
        # odd, which no simple graph can absorb. More than that means these
        # construction parameters cannot hold the level at its degree, and the
        # private-dimension gradient would silently stop being what the record
        # says it is — so it is refused rather than built and reported.
        if shortfall > sum(deficit) % 2:
            raise ValueError(
                f"the core level of {len(level)} cells cannot reach degree "
                f"{target}: {shortfall} short. Its cells already carry "
                f"{[degree[cell] for cell in level]} vertical edges."
            )
        for i, j in lateral:
            b.edge(level[i], level[j])

    # -- The drive, at the apex level, entire ------------------------------
    # A rule rather than a hand-pick, which is why no apex cell is singled out.
    # Strength is fan-out, not width: every drive edge is m = 1.
    for cell in apex:
        b.edge(drive, cell)

    return Dome._assemble(spec, tuple(b.cells), tuple(b.edges))


@dataclass(frozen=True)
class Dome:
    """The built graph, and everything a later ticket runs on.

    The runtime surface — :attr:`incident`, :attr:`degrees`, :attr:`stalk_sums`,
    :attr:`private_mask`, :attr:`private_projection`, :meth:`restriction_mask` —
    is fixed at construction and reads no cell's :class:`CellIndex`. The index is
    on the cells for plotting and for the diagnostics below, which are reporting
    rather than runtime.
    """

    spec: DomeSpec
    cells: tuple[Cell, ...]
    edges: tuple[Edge, ...]
    incident: tuple[tuple[int, ...], ...]
    """Edge ids incident on each cell, by cell id."""

    degrees: tuple[int, ...]
    """Number of incident edges, by cell id."""

    stalk_sums: tuple[int, ...]
    """`sum_e m_e` over the edges incident on each cell, by cell id."""

    predicting: tuple[int, ...]
    """Cell ids of the predicting cells, in the row order of :attr:`private_mask`."""

    boundary: tuple[int, ...]
    """Cell ids of the boundary cells."""

    _permitted: tuple[int, ...] = field(repr=False)
    """How many leading node stalk directions a cell exposes on its edges, by cell id."""

    @classmethod
    def _assemble(
        cls, spec: DomeSpec, cells: tuple[Cell, ...], edges: tuple[Edge, ...]
    ) -> "Dome":
        incident: list[list[int]] = [[] for _ in cells]
        stalk_sums = [0] * len(cells)
        for e in edges:
            for endpoint in (e.u, e.v):
                incident[endpoint].append(e.id)
                stalk_sums[endpoint] += e.m

        # The structural mask. At a predicting cell the incident edges together
        # can carry at most `sum_e m_e` independent directions of a node stalk of
        # `n`; the mask makes that bound structural by permitting the leading
        # `min(n, sum_e m_e)` directions on every incident edge and masking the
        # rest out everywhere. Those are the cell's private features, and they
        # make `dim H^0 >= sum_v max(0, n - sum_e m_e)` hold on the mask alone
        # rather than on the maps' ranks. Learned rank-deficiency only enlarges
        # `H^0` past it; nothing shrinks it.
        #
        # A boundary cell is not masked. Its stalk is world-shaped rather than
        # `n`-shaped and its restriction is the compression of what the world
        # wrote — a patch cell's 48 -> 8 is that compression, and masking it
        # would throw the patch away instead of compressing it.
        permitted = [
            c.stalk if c.is_boundary else min(c.stalk, stalk_sums[c.id]) for c in cells
        ]
        return cls(
            spec=spec,
            cells=cells,
            edges=edges,
            incident=tuple(tuple(ids) for ids in incident),
            degrees=tuple(len(ids) for ids in incident),
            stalk_sums=tuple(stalk_sums),
            predicting=tuple(c.id for c in cells if not c.is_boundary),
            boundary=tuple(c.id for c in cells if c.is_boundary),
            _permitted=tuple(permitted),
        )

    # -- runtime surface ---------------------------------------------------

    @property
    def shape(self) -> BodyShape:
        """The predicting population's `n` and `k`, for the shared frozen body."""
        return BodyShape(n=self.spec.n, k=self.spec.k)

    def restriction_mask(self, edge_id: int, cell_id: int) -> torch.Tensor:
        """`[stalk]` bool: which node stalk directions this cell may put on this edge.

        The map itself is `[m_e, stalk]`; the columns this mask clears are
        structurally zero and stay that way. The mask closes and never re-opens.
        """
        edge = self.edges[edge_id]
        edge.other(cell_id)  # raises if the cell is not an endpoint
        stalk = self.cells[cell_id].stalk
        permitted = torch.zeros(stalk, dtype=torch.bool)
        permitted[: self._permitted[cell_id]] = True
        return permitted

    @property
    def private_mask(self) -> torch.Tensor:
        """`[cells, n]` bool over predicting cells: True where the direction is private.

        A direction masked out on every incident edge participates on no edge, so
        it cannot disagree on any edge and lies in `H^0` by construction
        (`docs/spec/01-cell-and-sheaf.md`, *`H^0` is the private features*). Rows
        are indexed by :attr:`predicting`, not by cell id.

        Built from :attr:`_permitted` on each read rather than held as a tensor,
        so it is the same fact :meth:`restriction_mask` reports and the two can
        never drift apart. The mask closes and never re-opens: writing into what
        this returns changes nothing.
        """
        mask = torch.ones((len(self.predicting), self.spec.n), dtype=torch.bool)
        for row, cell_id in enumerate(self.predicting):
            mask[row, : self._permitted[cell_id]] = False
        return mask

    @property
    def private_projection(self) -> torch.Tensor:
        """`[cells, n]` of 0.0/1.0 over predicting cells: the private component.

        A fixed projection, known at construction and invariant under learning.
        Multiplying a `[cells, n]` node stalk batch by it keeps exactly the
        directions reconciliation cannot move — the cell's `H^0` component, which
        is what makes slow state and commitment possible at all.
        """
        return self.private_mask.to(torch.float32)

    @property
    def private_dimensions(self) -> torch.Tensor:
        """`[cells]` int over predicting cells: `max(0, n - sum_e m_e)`."""
        return self.private_mask.sum(dim=-1)

    def neighbours(self, cell_id: int) -> tuple[int, ...]:
        return tuple(self.edges[e].other(cell_id) for e in self.incident[cell_id])

    # -- construction diagnostics -----------------------------------------

    @property
    def euler_characteristic(self) -> int:
        """`chi = sum_v n - sum_e m_e = dim H^0 - dim H^1`.

        The node term runs over **predicting cells only** — including boundary
        cells swamps it, 256 cells of nominally private state the world
        overwrites every tick. The edge term runs over **all** edges,
        boundary-incident ones included, because those edge stalks are ordinary
        and are the route the boundary's information actually takes
        (`docs/spec/06-graph-topology.md`, *Private dimension is a gradient*).

        Fixed at construction and invariant under learning: no learned parameter
        appears in it. A diagnostic, not a budget, and nothing branches on it.
        """
        return len(self.predicting) * self.spec.n - sum(e.m for e in self.edges)

    @property
    def cut_capacities(self) -> tuple[tuple[str, int], ...]:
        """Numbers per tick across the render and each of the taper's cuts.

        The first entry is the render itself, which the world writes into the
        sensory boundary's node stalks raw. The rest are `sum m_e` over the
        edges crossing between consecutive levels, in order, so the run from the
        render down to the first core level is what the whole sensory boundary
        reaches the core through. Drive edges are not a cut of the taper: the
        drive is not the world and its edges cross no level.
        """
        render = sum(c.stalk for c in self.cells if c.kind is CellKind.PATCH)
        cuts: list[tuple[str, int]] = [("render", render)]
        deepest = max(c.index.level for c in self.cells)
        for level in range(deepest):
            crossing = sum(
                e.m
                for e in self.edges
                if e.kind is not EdgeKind.DRIVE
                and {self.cells[e.u].index.level, self.cells[e.v].index.level}
                == {level, level + 1}
            )
            if crossing:
                cuts.append((f"L{level} -> L{level + 1}", crossing))
        return tuple(cuts)

    def private_dimension_rows(self) -> tuple[tuple[str, str, str, str], ...]:
        """The private-dimension table, grouped the way the record groups it.

        Rows are `(cell group, degree, sum_e m_e, guaranteed private dimension)`,
        each a single value or a range. Measured from the built graph.
        """
        groups: dict[str, list[int]] = {}
        for cell_id in self.predicting:
            groups.setdefault(self._group(cell_id), []).append(cell_id)

        def span(values: list[int]) -> str:
            lo, hi = min(values), max(values)
            return str(lo) if lo == hi else f"{lo}-{hi}"

        rows = []
        for name, ids in groups.items():
            rows.append(
                (
                    f"{name} ({len(ids)})",
                    span([self.degrees[i] for i in ids]),
                    span([self.stalk_sums[i] for i in ids]),
                    span([max(0, self.spec.n - self.stalk_sums[i]) for i in ids]),
                )
            )
        return tuple(rows)

    def _group(self, cell_id: int) -> str:
        """The row of the recorded table a predicting cell belongs to.

        Reporting only — it reads the construction layout, which is what the
        layout is for once the mask is generated.
        """
        index = self.cells[cell_id].index
        if index.column == "vision":
            side = self.spec.vision_sides[index.level - 1]
            r, c = index.position
            edges_on = (r in (0, side - 1)) + (c in (0, side - 1))
            place = {0: "interior", 1: "lattice edge", 2: "lattice corner"}[edges_on]
            return f"L{index.level} vision ({place})"
        if index.column == "somatomotor":
            return f"L{index.level} somatomotor"
        apex = 1 + len(self.spec.vision_sides) + len(self.spec.core_sizes) - 1
        return "L%d apex" % apex if index.level == apex else "L%d core" % index.level

    def report(self) -> str:
        """The construction diagnostics, measured from the built graph."""
        spec = self.spec
        lines = ["The dome", "========", ""]

        levels: dict[int, dict[str, int]] = {}
        for cell in self.cells:
            levels.setdefault(cell.index.level, {}).setdefault(cell.index.column, 0)
            levels[cell.index.level][cell.index.column] += 1
        lines.append("cells")
        for level in sorted(levels):
            parts = ", ".join(f"{n} {col}" for col, n in levels[level].items())
            lines.append(f"  L{level}: {parts}")
        lines.append(
            f"  {len(self.predicting)} predicting, {len(self.boundary)} boundary, "
            f"{len(self.cells)} in all"
        )
        lines.append("")

        by_kind: dict[str, int] = {}
        for e in self.edges:
            by_kind[e.kind.value] = by_kind.get(e.kind.value, 0) + 1
        mean = sum(self.degrees[i] for i in self.predicting) / len(self.predicting)
        apex_level = 1 + len(spec.vision_sides) + len(spec.core_sizes) - 1
        apex_ids = [
            c.id
            for c in self.cells
            if c.index.level == apex_level and c.index.column == "core"
        ]
        apex_degrees = {self.degrees[i] for i in apex_ids}
        apex_undriven = {
            self.degrees[i]
            - sum(1 for e in self.incident[i] if self.edges[e].kind is EdgeKind.DRIVE)
            for i in apex_ids
        }
        lines.append("edges")
        for kind, count in sorted(by_kind.items()):
            lines.append(f"  {count} {kind}")
        lines.append(f"  {len(self.edges)} in all")
        lines.append(f"  mean degree over predicting cells: {mean:.2f}")
        lines.append(
            f"  apex degree: {sorted(apex_undriven)} without the drive edge, "
            f"{sorted(apex_degrees)} with it"
        )
        lines.append("")

        lines.append("dimensions")
        lines.append(
            f"  n = {spec.n}, k = {spec.k}, interior m = {spec.interior_m}, "
            f"boundary m = {spec.boundary_m}, drive m = {spec.drive_m}"
        )
        lines.append(
            "  boundary stalks: patch "
            f"{spec.patch_stalk}, proprioceptive {spec.proprioceptive_stalk}, "
            f"touch {spec.touch_stalk}, actuator {spec.actuator_stalk}, "
            f"drive {spec.drive_stalk}"
        )
        lines.append("")

        lines.append(f"chi = {self.euler_characteristic:+d}")
        lines.append(
            f"  over {len(self.predicting)} predicting-cell node terms and all "
            f"{len(self.edges)} edges, boundary-incident included"
        )
        lines.append("")

        cuts = self.cut_capacities
        to_core = cuts[: 2 + len(spec.vision_sides)]
        lines.append("cut capacities, numbers per tick")
        lines.append("  " + " -> ".join(f"{value:,}" for _, value in cuts))
        for name, value in cuts:
            lines.append(f"    {name}: {value:,}")
        lines.append(
            f"  the whole sensory boundary reaches the core through {to_core[-1][1]:,} "
            f"numbers per tick, a {to_core[0][1] / to_core[-1][1]:.0f}:1 squeeze"
        )
        lines.append("")

        lines.append("guaranteed private dimension")
        header = ("cell", "degree", "sum m_e", "dim H^0 >=")
        rows = (header,) + self.private_dimension_rows()
        widths = [max(len(row[i]) for row in rows) for i in range(4)]
        for i, row in enumerate(rows):
            lines.append("  " + "  ".join(v.ljust(w) for v, w in zip(row, widths)))
            if i == 0:
                lines.append("  " + "  ".join("-" * w for w in widths))
        total = int(self.private_dimensions.sum())
        lines.append(f"  sum over predicting cells: {total}")
        return "\n".join(lines)


def main() -> None:  # pragma: no cover - the entry point behind `python -m patchworks`
    print(build_graph().report())
