"""Dome construction and the structural masks (ticket #83).

What these tests hold down is the construction and what construction records:
the populations, the cut capacities, the private-dimension gradient, `chi`, and
the two commitments that are easy to lose later -- that the construction layout
has no runtime role, and that no edge is ever removed.

Every figure asserted here was **measured** from the built graph. Where a
measurement differs from `docs/spec/06-graph-topology.md`'s rounded figure the
test says so at the assertion, so a later change that moves it is caught rather
than absorbed.
"""

import dataclasses
from collections import Counter, deque

import pytest
import torch

from patchworks.body import BodyShape
from patchworks.graph import (
    DEFAULT_SPEC,
    Cell,
    CellIndex,
    CellKind,
    Dome,
    DomeSpec,
    EdgeKind,
    build_graph,
)

APEX_LEVEL = 7


@pytest.fixture(scope="module")
def dome():
    return build_graph()


def _by_level(dome, level, column=None):
    return [
        c.id
        for c in dome.cells
        if c.index.level == level and (column is None or c.index.column == column)
    ]


def _shortest_path(dome, source, target):
    previous = {source: None}
    queue = deque([source])
    while queue:
        cell = queue.popleft()
        if cell == target:
            path = []
            while cell is not None:
                path.append(cell)
                cell = previous[cell]
            return path[::-1]
        for neighbour in dome.neighbours(cell):
            if neighbour not in previous:
                previous[neighbour] = cell
                queue.append(neighbour)
    raise AssertionError(f"no path from {source} to {target}")


class TestThePopulations:
    def test_predicting_and_boundary_counts(self, dome):
        assert len(dome.predicting) == 150
        assert len(dome.boundary) == 264

    def test_the_levels(self, dome):
        assert len(_by_level(dome, 0, "vision")) == 256
        assert len(_by_level(dome, 0, "somatomotor")) == 7
        assert len(_by_level(dome, 1)) == 70
        assert len(_by_level(dome, 2)) == 20
        assert [len(_by_level(dome, level, "core")) for level in range(3, 8)] == [
            16,
            14,
            12,
            10,
            8,
        ]

    def test_the_internal_rim_is_one_drive_cell_at_the_apex(self, dome):
        rim = _by_level(dome, APEX_LEVEL, "internal rim")
        assert len(rim) == 1
        assert dome.cells[rim[0]].kind is CellKind.DRIVE

    def test_every_boundary_cell_runs_no_body_and_holds_no_chart(self, dome):
        # A boundary cell is exempt from `n`; nothing here gives it a `k`.
        assert all(dome.cells[i].kind.is_boundary for i in dome.boundary)
        assert all(dome.cells[i].kind is CellKind.PREDICTING for i in dome.predicting)


class TestConnectivity:
    def test_edge_count(self, dome):
        # Measured, and now what the record carries: the rounded ~698 was retired
        # in favour of the built graph's own count.
        assert len(dome.edges) == 682

    def test_mean_degree_is_about_seven(self, dome):
        mean = sum(dome.degrees[i] for i in dome.predicting) / len(dome.predicting)
        assert 6.5 <= mean <= 7.5
        assert round(mean, 2) == 7.27

    def test_the_core_is_uniform_at_six_except_the_apex_at_four(self, dome):
        for level in range(3, APEX_LEVEL):
            assert {dome.degrees[i] for i in _by_level(dome, level, "core")} == {6}
        # The apex loses its up-edges by construction -- L7 has no predicting
        # level above it -- which is the two edges of six it is short.
        apex = _by_level(dome, APEX_LEVEL, "core")
        undriven = {
            dome.degrees[i]
            - sum(
                1
                for e in dome.incident[i]
                if dome.edges[e].kind is EdgeKind.DRIVE
            )
            for i in apex
        }
        assert undriven == {4}

    def test_every_sensory_boundary_cell_has_one_edge_and_the_actuator_has_three(
        self, dome
    ):
        # 2,120 = 262 sensory cells x 8, plus the actuator's one motor edge per
        # joint x 8. The actuator is the single exception and it is a motor cell:
        # one edge per joint is what makes every joint's reflex loop three ticks.
        sensory = [
            i
            for i in dome.boundary
            if dome.cells[i].index.level == 0
            and dome.cells[i].kind is not CellKind.ACTUATOR
        ]
        assert {dome.degrees[i] for i in sensory} == {1}
        actuator = next(c.id for c in dome.cells if c.kind is CellKind.ACTUATOR)
        assert dome.degrees[actuator] == DEFAULT_SPEC.joints == 3

    def test_vision_lattices_are_four_neighbour(self, dome):
        for level, side in zip((1, 2), DEFAULT_SPEC.vision_sides):
            lateral = [
                e
                for e in dome.edges
                if dome.cells[e.u].index.level == level
                and dome.cells[e.v].index.level == level
                and dome.cells[e.u].index.column == "vision"
                and dome.cells[e.v].index.column == "vision"
            ]
            assert len(lateral) == 2 * side * (side - 1)
            for edge in lateral:
                a = dome.cells[edge.u].index.position
                b = dome.cells[edge.v].index.position
                assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1

    def test_the_dome_is_one_connected_graph(self, dome):
        # Nothing is stranded: every cell, boundary cells included, is reachable
        # from every other. Fragmentation is never structural here.
        seen = {dome.cells[0].id}
        stack = [dome.cells[0].id]
        while stack:
            for neighbour in dome.neighbours(stack.pop()):
                if neighbour not in seen:
                    seen.add(neighbour)
                    stack.append(neighbour)
        assert len(seen) == len(dome.cells)
        assert min(dome.degrees) >= 1

    def test_no_edge_is_ever_removed(self, dome):
        # There is no removal path: the built graph is frozen, and the mask
        # closes and never re-opens (`06-graph-topology.md`, *No edge is ever
        # removed*).
        with pytest.raises(dataclasses.FrozenInstanceError):
            dome.edges = ()
        with pytest.raises(dataclasses.FrozenInstanceError):
            dome.edges[0].m = 99
        assert isinstance(dome.edges, tuple)


class TestTheSomatomotorColumn:
    def test_the_column_is_parallel_through_l1_and_l2(self, dome):
        crossing = [
            e
            for e in dome.edges
            if {dome.cells[e.u].index.column, dome.cells[e.v].index.column}
            == {"vision", "somatomotor"}
        ]
        assert crossing == []

    def test_vision_and_proprioception_first_share_a_cell_at_l3(self, dome):
        def columns_below(cell_id):
            level = dome.cells[cell_id].index.level
            return {
                dome.cells[dome.edges[e].other(cell_id)].index.column
                for e in dome.incident[cell_id]
                if dome.cells[dome.edges[e].other(cell_id)].index.level < level
            }

        for level in (1, 2):
            assert all(
                len(columns_below(i)) == 1 for i in _by_level(dome, level)
            ), "no cell below the core hears two modalities"
        sharing = [
            i
            for i in _by_level(dome, 3, "core")
            if columns_below(i) == {"vision", "somatomotor"}
        ]
        assert len(sharing) == 4

    def test_the_reflex_loop_is_three_ticks_and_purely_somatomotor(self, dome):
        actuator = next(
            c.id for c in dome.cells if c.kind is CellKind.ACTUATOR
        )
        proprioceptive = [
            c.id for c in dome.cells if c.kind is CellKind.PROPRIOCEPTIVE
        ]
        paths = [_shortest_path(dome, p, actuator) for p in proprioceptive]
        # Proprioceptive boundary cell -> an L1 somatomotor cell -> actuator
        # boundary cell. Three cells, so three ticks, and at **every** joint: a
        # corrective twitch never waits on vision and never waits a tick longer
        # at one joint than another.
        assert len(paths) == DEFAULT_SPEC.joints == 3
        for path in paths:
            assert len(path) == 3
            assert [dome.cells[i].index.level for i in path] == [0, 1, 0]
            # Purely somatomotor: no cell on any of these paths is a vision cell.
            assert all(dome.cells[i].index.column == "somatomotor" for i in path)
        # And each joint reaches the rim through its own L1 cell.
        assert len({path[1] for path in paths}) == 3


class TestTheDrive:
    def test_it_attaches_at_the_apex_level_entire(self, dome):
        drive = next(c.id for c in dome.cells if c.kind is CellKind.DRIVE)
        reached = {dome.edges[e].other(drive) for e in dome.incident[drive]}
        assert reached == set(_by_level(dome, APEX_LEVEL, "core"))
        assert len(reached) == 8

    def test_strength_is_fan_out_not_width(self, dome):
        drive = next(c.id for c in dome.cells if c.kind is CellKind.DRIVE)
        assert dome.cells[drive].stalk == 1
        assert {dome.edges[e].m for e in dome.incident[drive]} == {1}
        # A drive edge's share of an apex cell's reconciliation pull. It rose
        # from 0.059 when #474 narrowed the interior lanes: the drive's own
        # width did not move, and every other lane into the apex got thinner.
        apex = _by_level(dome, APEX_LEVEL, "core")[0]
        assert round(1 / dome.stalk_sums[apex], 3) == 0.077


class TestDimensionsAndTheBoundaryExemption:
    def test_boundary_stalks_are_world_shaped(self, dome):
        stalks = {}
        for cell in dome.cells:
            if cell.kind.is_boundary:
                stalks.setdefault(cell.kind, set()).add(cell.stalk)
        assert stalks == {
            CellKind.PATCH: {48},
            CellKind.PROPRIOCEPTIVE: {2},
            CellKind.TOUCH: {1},
            CellKind.ACTUATOR: {6},
            CellKind.DRIVE: {1},
        }

    def test_predicting_stalks_are_all_n(self, dome):
        assert {dome.cells[i].stalk for i in dome.predicting} == {32}
        assert dome.shape == BodyShape(n=32, k=12)

    def test_every_lane_is_ordinary(self, dome):
        # The world touches node stalks only. Boundary-incident lanes are
        # ordinary and m-sized like any other.
        widths = {}
        for edge in dome.edges:
            widths.setdefault(edge.kind, set()).add(edge.m)
        assert widths == {
            EdgeKind.SENSORY: {4},
            EdgeKind.MOTOR: {4},
            EdgeKind.INTERIOR: {3},
            EdgeKind.DRIVE: {1},
        }


class TestRecordedDiagnostics:
    def test_cut_capacities(self, dome):
        named = dict(dome.cut_capacities)
        assert named["render"] == 12_288
        assert named["L0 -> L1"] == 1_060
        assert named["L1 -> L2"] == 210
        assert named["L2 -> L3"] == 60
        # The whole sensory boundary reaches the core through sixty numbers a
        # tick: a 205:1 squeeze at a single cut. It was 2_120 / 280 / 80 and
        # 154:1 until #474 narrowed both lane widths for the private floor --
        # the taper's capacities are set by m (ADR-0030), so this is the
        # arithmetic of that ruling rather than a separate finding.
        assert round(named["render"] / named["L2 -> L3"]) == 205

    def test_euler_characteristic(self, dome):
        chi = dome.euler_characteristic
        assert chi == len(dome.predicting) * 32 - sum(e.m for e in dome.edges)
        # Measured, and now what the record carries. What is load-bearing about
        # chi is its invariance under *learning*, not its value -- so a
        # construction change moves it freely, and #474 moved it from +1036 to
        # here by narrowing both lane widths. The old +980/+1096 band was an
        # estimate retired in favour of the measurement at (4, 8) and does not
        # travel to this surface.
        assert chi == 2505

    def test_the_node_term_is_predicting_cells_and_the_edge_term_is_all_edges(
        self, dome
    ):
        boundary_incident = sum(
            e.m
            for e in dome.edges
            if dome.cells[e.u].kind.is_boundary or dome.cells[e.v].kind.is_boundary
        )
        assert boundary_incident == 1_068  # 265 x 4 sensorimotor, 8 x 1 drive
        # Dropping the boundary edges as well as the boundary nodes is the wrong
        # computation the record corrects; it gives +3573 against the +2505 this
        # graph carries. The gap narrowed when #474 halved the boundary lanes --
        # it read +3164 against +1036 before -- and the error did not.
        assert dome.euler_characteristic + boundary_incident == 3_573

    def test_private_dimension_gradient(self, dome):
        def dims(cell_ids):
            rows = {dome.predicting.index(i) for i in cell_ids}
            return {int(dome.private_dimensions[r]) for r in rows}

        # Nowhere zero since #474: the pair (interior_m, boundary_m) = (3, 4)
        # is derived from `sum_e m_e <= n - 1` at every predicting cell, so the
        # floor is p_v >= 1 by construction. The L1 vision cells read 1 / 4 / 7
        # by degree 9 / 8 / 7, and 1 is the thinnest cell in the graph.
        assert dims(_by_level(dome, 1, "vision")) == {1, 4, 7}
        assert dims(_by_level(dome, 1, "somatomotor")) == {6, 10}

        side = DEFAULT_SPEC.vision_sides[-1]
        corners = [
            i
            for i in _by_level(dome, 2, "vision")
            if all(p in (0, side - 1) for p in dome.cells[i].index.position)
        ]
        assert len(corners) == 4
        assert dims(corners) == {11}

        for level in range(3, APEX_LEVEL):
            assert dims(_by_level(dome, level, "core")) == {14}
        assert dims(_by_level(dome, APEX_LEVEL, "core")) == {19}

    def test_the_l2_somatomotor_cells_are_not_in_the_recorded_table(self, dome):
        # Measured and reported rather than transcribed: the record's table has
        # rows for the vision lattices, the core and the apex. Four cells sit
        # outside it -- the L2 somatomotor column, a four-cell level whose
        # degree the taper cannot lift -- and they carry more structural privacy
        # than the L2 vision corners do.
        rows = {dome.predicting.index(i) for i in _by_level(dome, 2, "somatomotor")}
        assert {int(dome.private_dimensions[r]) for r in rows} == {14, 17}

    def test_the_whole_private_dimension_distribution(self, dome):
        # The per-group table is a range table, so it can read unmoved while the
        # cells behind it move. This pins every cell. It is what says the
        # actuator's three motor edges left the gradient alone: the three L1
        # somatomotor cells covering proprioception carry one motor edge more
        # than their siblings and land at 6 where those read 10.
        #
        # **No cell reads zero.** It was {0: 82, 4: 4, 8: 54, 12: 2, 15: 8},
        # summing to 592, until #474 set (interior_m, boundary_m) = (3, 4) from
        # `sum_e m_e <= n - 1`. The 0 key is gone, which is the whole content of
        # that ruling, and the minimum key is 1.
        histogram = Counter(int(v) for v in dome.private_dimensions)
        assert dict(sorted(histogram.items())) == {
            1: 36,
            4: 24,
            5: 4,
            6: 3,
            7: 4,
            8: 8,
            10: 3,
            11: 4,
            14: 54,
            17: 2,
            19: 8,
        }
        assert min(histogram) == 1
        assert int(dome.private_dimensions.sum()) == 1278

    def test_the_bound_is_met_with_equality_by_the_mask(self, dome):
        for row, cell_id in enumerate(dome.predicting):
            expected = max(0, 32 - dome.stalk_sums[cell_id])
            assert int(dome.private_dimensions[row]) == expected

    def test_report_prints_the_numbers(self, dome):
        text = dome.report()
        for fragment in (
            "150 predicting, 264 boundary",
            "chi = +2505",
            "12,288 -> 1,060 -> 210 -> 60",
            "guaranteed private dimension",
        ):
            assert fragment in text


class TestTheStructuralMasks:
    def test_the_mask_closes_and_never_re_opens(self, dome):
        # The mask is derived from one stored fact rather than held as a tensor,
        # so writing into what a read hands back re-opens nothing and cannot
        # drift away from what `restriction_mask` reports.
        original = build_graph().private_mask
        handed_back = dome.private_mask
        handed_back[:] = False
        assert torch.equal(dome.private_mask, original)
        cell_id = dome.predicting[-1]
        union = torch.zeros(32, dtype=torch.bool)
        for edge_id in dome.incident[cell_id]:
            union |= dome.restriction_mask(edge_id, cell_id)
        row = dome.predicting.index(cell_id)
        assert torch.equal(union, ~dome.private_mask[row])

    def test_the_private_component_is_a_fixed_projection(self, dome):
        projection = dome.private_projection
        assert projection.shape == (150, 32)
        assert set(projection.unique().tolist()) == {0.0, 1.0}
        assert torch.equal(projection, dome.private_mask.to(torch.float32))
        # Idempotent, as a projection: applying it twice is applying it once.
        stalks = torch.randn(150, 32)
        once = stalks * projection
        assert torch.equal(once, once * projection)

    def test_private_directions_participate_on_no_edge(self, dome):
        for row, cell_id in enumerate(dome.predicting):
            private = dome.private_mask[row]
            for edge_id in dome.incident[cell_id]:
                permitted = dome.restriction_mask(edge_id, cell_id)
                assert not bool((permitted & private).any())
            # And every direction that is not private participates somewhere.
            union = torch.zeros(32, dtype=torch.bool)
            for edge_id in dome.incident[cell_id]:
                union |= dome.restriction_mask(edge_id, cell_id)
            assert torch.equal(union, ~private)

    def test_a_boundary_cell_is_not_masked(self, dome):
        # A patch cell's 48 -> 8 restriction is the compression of that patch.
        # Masking it would throw the patch away instead of compressing it.
        for cell_id in dome.boundary:
            for edge_id in dome.incident[cell_id]:
                mask = dome.restriction_mask(edge_id, cell_id)
                assert mask.shape == (dome.cells[cell_id].stalk,)
                assert bool(mask.all())

    def test_a_mask_is_refused_for_a_cell_not_on_the_edge(self, dome):
        with pytest.raises(ValueError):
            dome.restriction_mask(0, dome.predicting[-1])


class Poisoned:
    """Stands in for a construction layout that must not be consulted."""

    def __getattr__(self, name):
        raise AssertionError(f"the construction layout was consulted at runtime ({name})")


class TestTheLayoutIsAnIndexNotAnEmbedding:
    def test_positions_are_integer_indices(self, dome):
        for cell in dome.cells:
            assert isinstance(cell.index, CellIndex)
            assert isinstance(cell.index.level, int)
            for coordinate in cell.index.position:
                assert type(coordinate) is int

    def test_no_distance_kernel_is_available_to_be_consulted(self):
        import patchworks.graph as module

        names = [name.lower() for name in dir(module)]
        assert not [
            name
            for name in names
            if "distance" in name or "coord" in name or "embedding" in name
        ]
        source = [f.name for f in dataclasses.fields(Cell)]
        assert "coordinate" not in source and "position" not in source

    def test_the_runtime_surface_never_reads_the_layout(self, dome):
        """Poison every cell's index, then run the whole runtime surface."""
        before = (
            dome.private_mask.clone(),
            dome.private_projection.clone(),
            dome.private_dimensions.clone(),
            dome.degrees,
            dome.stalk_sums,
            dome.euler_characteristic,
        )
        masks_before = [
            dome.restriction_mask(e.id, e.u).clone() for e in dome.edges
        ]
        indices = [cell.index for cell in dome.cells]
        try:
            for cell in dome.cells:
                object.__setattr__(cell, "index", Poisoned())
            after = (
                dome.private_mask,
                dome.private_projection,
                dome.private_dimensions,
                dome.degrees,
                dome.stalk_sums,
                dome.euler_characteristic,
            )
            for cell_id in dome.predicting:
                dome.neighbours(cell_id)
            masks_after = [
                dome.restriction_mask(e.id, e.u) for e in dome.edges
            ]
            assert dome.shape == BodyShape(n=32, k=12)
        finally:
            for cell, index in zip(dome.cells, indices):
                object.__setattr__(cell, "index", index)

        assert torch.equal(before[0], after[0])
        assert torch.equal(before[1], after[1])
        assert torch.equal(before[2], after[2])
        assert before[3:] == after[3:]
        assert all(torch.equal(a, b) for a, b in zip(masks_before, masks_after))


class TestConstruction:
    def test_it_is_deterministic(self):
        assert build_graph() == build_graph()

    def test_there_are_no_parallel_edges(self, dome):
        pairs = [(min(e.u, e.v), max(e.u, e.v)) for e in dome.edges]
        assert len(set(pairs)) == len(pairs)

    def test_halving_the_core_keeps_the_recorded_gradient(self):
        # Halving the core is a construction-parameter change, not a code change
        # (docs/spec/06-graph-topology.md, "Why 150 and not 500"), and it is one
        # of the two conditions of the falsification sweep. The degree targets
        # and so the private-dimension step have to survive it.
        halved = build_graph(DomeSpec(core_sizes=(8, 7, 6, 5, 4)))
        assert len(halved.predicting) == 120
        assert len(halved.boundary) == 264
        assert dict(halved.cut_capacities)["L2 -> L3"] == 60
        for level in range(3, APEX_LEVEL):
            assert {halved.degrees[i] for i in _by_level(halved, level, "core")} == {6}
        apex = _by_level(halved, APEX_LEVEL, "core")
        assert {halved.degrees[i] - 1 for i in apex} == {4}
        rows = {halved.predicting.index(i) for i in apex}
        assert {int(halved.private_dimensions[r]) for r in rows} == {19}

    def test_a_core_level_that_cannot_hold_its_degree_is_refused(self):
        # Rather than built and reported: the guaranteed private dimension is
        # read straight off these degrees, so a level that quietly missed its
        # target would make the recorded gradient stop being true with nothing
        # saying so. It can miss in either direction, and both are refused.
        with pytest.raises(ValueError, match="overshoots degree"):
            build_graph(DomeSpec(core_sizes=(2, 2)))
        with pytest.raises(ValueError, match="cannot reach degree"):
            build_graph(
                DomeSpec(
                    patch_grid=4,
                    vision_sides=(2, 1),
                    somatomotor_sizes=(2, 1),
                    core_sizes=(2, 2),
                )
            )

    @pytest.mark.parametrize(
        "core_sizes",
        [(16, 14, 12, 10, 8), (8, 7, 6, 5, 4), (20, 16, 12, 8, 4), (7, 5, 4, 3, 2)],
    )
    def test_every_core_that_builds_holds_its_degree_exactly(self, core_sizes):
        # A level is realised when a simple graph can carry its degrees, not when
        # a sweep in index order happens to find the pairing. These four all can.
        built = build_graph(DomeSpec(core_sizes=core_sizes))
        for level in range(3, APEX_LEVEL):
            assert {built.degrees[i] for i in _by_level(built, level, "core")} == {6}
        assert {
            built.degrees[i] - 1 for i in _by_level(built, APEX_LEVEL, "core")
        } == {4}

    def test_an_empty_somatomotor_level_is_refused(self):
        with pytest.raises(ValueError, match="at least one cell"):
            DomeSpec(somatomotor_sizes=(0, 0))

    def test_the_column_runs_parallel_to_the_vision_lattices(self):
        with pytest.raises(ValueError, match="a level for each of them"):
            DomeSpec(vision_sides=(8, 4), somatomotor_sizes=(6,))
        with pytest.raises(ValueError, match="at least one vision lattice"):
            DomeSpec(patch_grid=2, vision_sides=(), somatomotor_sizes=())

    def test_a_coarser_tiling_is_a_construction_parameter(self):
        # The other condition of the falsification sweep: 8x8 px patches, so a
        # patch cell does see a whole puck.
        coarse = build_graph(
            DomeSpec(patch_grid=8, vision_sides=(4, 2), patch_stalk=192)
        )
        assert len([c for c in coarse.cells if c.kind is CellKind.PATCH]) == 64
        assert dict(coarse.cut_capacities)["render"] == 12_288

    def test_a_taper_that_is_not_a_two_by_two_block_is_refused(self):
        with pytest.raises(ValueError, match="2x2 block"):
            DomeSpec(vision_sides=(8, 3))

    def test_the_apex_must_be_lower_degree_than_the_core(self):
        with pytest.raises(ValueError, match="lower-degree"):
            DomeSpec(apex_degree=6)

    def test_the_dome_is_frozen(self, dome):
        assert isinstance(dome, Dome)
        with pytest.raises(dataclasses.FrozenInstanceError):
            dome.spec = None
