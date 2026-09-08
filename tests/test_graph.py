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
        # from 0.059 to 0.077 when #474 narrowed the interior lanes, #548 took
        # it back down to 0.048 by widening them again, and #562 halved it
        # again by doubling the budget those lanes are allocated under -- the
        # drive's own width has never moved, so this number is entirely a
        # readout of what the lanes around it are doing.
        apex = _by_level(dome, APEX_LEVEL, "core")[0]
        assert round(1 / dome.stalk_sums[apex], 3) == 0.024


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
        # An interior lane is no longer one number. #548 wrote #540's ruling:
        # the width is allocated per edge, the largest both endpoints can
        # afford under the construction invariant, so the set here is a spread
        # rather than a constant. The 1 in it is `lateral_m` -- every same-level
        # lane, narrowed by #540's lever (c1).
        #
        # **The top of the spread is now a cap rather than an affordance.** It
        # was 18, what the apex could afford out of a budget of 31. At 63 the
        # water-fill would hand two lanes 32, so #556 capped every lane at the
        # permitted block `n - p = 20`: a lane wider than what the mask permits
        # carries nothing extra, which is `drive_m`'s own reasoning. 8 of the
        # 409 interior lanes bind on that cap.
        assert widths == {
            EdgeKind.SENSORY: {4},
            EdgeKind.MOTOR: {4},
            EdgeKind.INTERIOR: {1, 11, 12, 15, 16, 19, 20},
            EdgeKind.DRIVE: {1},
        }
        cap = 32 - dome.spec.private_reserve
        interior = [e for e in dome.edges if e.kind is EdgeKind.INTERIOR]
        assert max(e.m for e in interior) == cap
        assert sum(1 for e in interior if e.m == cap) == 8

    def test_the_invariant_holds_at_every_predicting_cell(self, dome):
        # What makes the allocation admissible at all, and the thing #548
        # checked before writing anything: the reallocation spends idle budget
        # and must not spend a cell past its own. `build_graph` raises if it
        # does; this pins it from outside.
        #
        # **The budget is 2n - 1 since #562, and it no longer carries the
        # floor.** #540 ruled the doubling and #548 refused to ship it, because
        # under the union mask 63 read zero private dimension at 104 of 150
        # cells. #556 unwelded the two, so the floor below is `private_reserve`
        # and is unaffected by anything this invariant permits -- which is
        # exactly why the doubling became shippable.
        budget = dome.spec.capacity_budget
        assert budget == 63
        assert max(dome.stalk_sums[i] for i in dome.predicting) == budget
        assert min(int(p) for p in dome.private_dimensions) == dome.spec.private_reserve


class TestRecordedDiagnostics:
    def test_cut_capacities(self, dome):
        named = dict(dome.cut_capacities)
        assert named["render"] == 12_288
        assert named["L0 -> L1"] == 1_060
        assert named["L1 -> L2"] == 872
        assert named["L2 -> L3"] == 266
        # The whole sensory boundary reaches the core through 266 numbers a
        # tick: a 46:1 squeeze at a single cut. It was 2_120 / 280 / 80 and
        # 154:1 before #474, then 210 / 60 and 205:1 after it; #548 widened it
        # to 392 / 118 and 104:1 by allocating interior lanes per edge. #562
        # widened it again, and by a different mechanism: the budget these
        # lanes are allocated under went `n - 1` to `2n - 1`, which #540 ruled
        # and #548 could not ship while the same number was also supplying
        # privacy. The taper's capacities are set by m (ADR-0030), so every one
        # of these numbers is the arithmetic of a lane ruling rather than a
        # separate finding.
        assert round(named["render"] / named["L2 -> L3"]) == 46

    def test_euler_characteristic(self, dome):
        chi = dome.euler_characteristic
        assert chi == len(dome.predicting) * 32 - sum(e.m for e in dome.edges)
        # Measured, and now what the record carries. What is load-bearing about
        # chi is its invariance under *learning*, not its value -- so a
        # construction change moves it freely: #474 moved it from +1036 to
        # +2505 by narrowing both lane widths, #548 moved it to +2323 by
        # allocating the interior ones per edge, and #562 moved it to here by
        # doubling the budget those lanes are allocated under. The old
        # +980/+1096 band was an estimate retired in favour of the measurement
        # at (4, 8) and does not travel to this surface.
        assert chi == 939

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
        # computation the record corrects; it gives +2007 against the +939 this
        # graph carries. The gap narrowed when #474 halved the boundary lanes --
        # it read +3164 against +1036 before -- and the *gap itself* is unmoved
        # by #548 and #562 alike, both of which touched interior lanes only and
        # left every boundary lane at 4. It is the same 1_068 either side.
        assert dome.euler_characteristic + boundary_incident == 2_007

    def test_the_private_dimension_is_flat_and_there_is_no_gradient(self, dome):
        """#556's ruling, and the whole of what the reserve mask changes.

        This test asserted a *gradient* until #562: 9 / 8 / 7 across the L1
        vision lattice by degree, 1-3 through the deep core, 11-15 at the apex.
        There is no gradient now. A predicting cell withholds its trailing
        `p` directions from **every** incident edge, so `p_v = p` at all 150 of
        them whatever their degree and whatever their lanes sum to.

        **That is a better guarantee and a worse gradient, and both halves are
        the record.** Better, because `p_v` no longer depends on the allocation
        at all -- it was a residual of `n - sum_e m_e`, so any lane ruling moved
        it, and #548's reallocation moved it 1278 -> 914 without anyone
        choosing that. Worse, because the graded column is gone, and
        `05-timescales.md` wanted the grading for slow state living deep.

        **Nothing supplies that grading now, and nothing did.** #572 measured
        it: `level` predicts none of the timescale statistics, the apex is the
        *fastest* place in the graph, and training drives it there. So this is
        not a gradient the reserve spent -- it is one the record credited to a
        mechanism that was never delivering it. #594 carries the candidate
        replacement.
        """

        def dims(cell_ids):
            rows = {dome.predicting.index(i) for i in cell_ids}
            return {int(dome.private_dimensions[r]) for r in rows}

        p = DEFAULT_SPEC.private_reserve
        for level, column in (
            (1, "vision"),
            (1, "somatomotor"),
            (2, "vision"),
            (2, "somatomotor"),
            (3, "core"),
            (4, "core"),
            (5, "core"),
            (6, "core"),
            (APEX_LEVEL, "core"),
        ):
            assert dims(_by_level(dome, level, column)) == {p}

        # The apex is no longer distinguishable by this column, and it was the
        # one cell group the old gradient was clearest about. It still loses
        # its up-edges by construction -- that shows in `sum_e m_e`, which is
        # now a capacity reading rather than a privacy one.
        apex_sums = {dome.stalk_sums[i] for i in _by_level(dome, APEX_LEVEL, "core")}
        assert max(apex_sums) < DEFAULT_SPEC.capacity_budget

    def test_the_l2_somatomotor_cells_read_the_same_as_everything_else(self, dome):
        # The record's table has rows for the vision lattices, the core and the
        # apex, and four cells sit outside it -- the L2 somatomotor column, a
        # four-cell level whose degree the taper cannot lift. Whether they are
        # in the table used to matter, because they carried a different private
        # width from every row in it (more than the L2 vision corners before
        # #548, less after). Since #562 they carry `p`, like everything else,
        # and the table's omission of them costs the reader nothing.
        rows = {dome.predicting.index(i) for i in _by_level(dome, 2, "somatomotor")}
        assert {int(dome.private_dimensions[r]) for r in rows} == {
            DEFAULT_SPEC.private_reserve
        }

    def test_the_whole_private_dimension_distribution(self, dome):
        # The per-group table is a range table, so it can read unmoved while the
        # cells behind it move. This pins every cell.
        #
        # **The distribution is one bar.** It was {0: 82, 4: 4, 8: 54, 12: 2,
        # 15: 8}, summing to 592, until #474 set (interior_m, boundary_m) =
        # (3, 4) from `sum_e m_e <= n - 1`; #474's own distribution summed to
        # 1278 with a spike of 36 cells at the floor; #548 reallocated per edge
        # and it summed to 914 across thirteen distinct values. #562 collapses
        # it to a single value at every cell.
        #
        # **The total is 1800 = 150p, and it is an equality rather than a
        # bound.** The old floor was `sum_v max(0, n - sum_e m_e)`, an
        # inequality that happened to be tight; this one is `sum_v p`, exact
        # and independent of every lane width in the graph. It is also the
        # highest the floor has ever read -- 1800 against 914 shipped, and
        # against the 54 that #540's doubling would have left under the union
        # mask, which is why #548 declined to ship the doubling and #562 can.
        histogram = Counter(int(v) for v in dome.private_dimensions)
        assert dict(histogram) == {DEFAULT_SPEC.private_reserve: 150}
        assert int(dome.private_dimensions.sum()) == 1_800
        assert (
            int(dome.private_dimensions.sum())
            == len(dome.predicting) * DEFAULT_SPEC.private_reserve
        )

    def test_the_bound_is_met_with_equality_by_the_mask(self, dome):
        """The mask still meets a bound exactly -- a different one.

        It used to meet `max(0, n - sum_e m_e)`, the union bound, which the
        mask made structural by permitting the leading `min(n, sum_e m_e)`
        directions. Since #556 the mask reserves `p` outright, so what it meets
        with equality is `p` itself, and `sum_e m_e` has dropped out of the
        statement. The old bound is superseded rather than violated -- it still
        holds, since `p >= max(0, n - sum_e m_e)` at every cell on this dome --
        and this asserts both, so the supersession is visible rather than
        asserted in prose alone.
        """
        for row, cell_id in enumerate(dome.predicting):
            private = int(dome.private_dimensions[row])
            assert private == DEFAULT_SPEC.private_reserve
            assert private >= max(0, 32 - dome.stalk_sums[cell_id])

    def test_report_prints_the_numbers(self, dome):
        text = dome.report()
        for fragment in (
            "150 predicting, 264 boundary",
            "chi = +939",
            "12,288 -> 1,060 -> 872 -> 266",
            "interior m: allocated per edge",
            "private reserve p = 12",
            "dim H^0 >= 1800",
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

    def test_halving_the_core_keeps_the_degree_targets_and_the_floor(self):
        # Halving the core is a construction-parameter change, not a code change
        # (docs/spec/06-graph-topology.md, "Why 150 and not 500"), and it is one
        # of the two conditions of the falsification sweep. The degree targets
        # have to survive it.
        halved = build_graph(DomeSpec(core_sizes=(8, 7, 6, 5, 4)))
        assert len(halved.predicting) == 120
        assert len(halved.boundary) == 264
        assert dict(halved.cut_capacities)["L2 -> L3"] == 236
        for level in range(3, APEX_LEVEL):
            assert {halved.degrees[i] for i in _by_level(halved, level, "core")} == {6}
        apex = _by_level(halved, APEX_LEVEL, "core")
        assert {halved.degrees[i] - 1 for i in apex} == {4}
        # This used to be the sharpest statement of what #548 changed: the
        # apex read the same 11 / 15 on a halved core as on the full dome,
        # because private dimension was a function of the cell's own
        # neighbourhood rather than of a global constant. Since #562 it is a
        # function of neither -- it is `p`, on every cell of every dome -- so
        # the sweep's question is answered trivially and the interesting
        # invariance moved to `dim H^0`, which is now `p` per predicting cell
        # exactly and so scales with the cell count rather than with the
        # widths.
        rows = {halved.predicting.index(i) for i in apex}
        p = halved.spec.private_reserve
        assert {int(halved.private_dimensions[r]) for r in rows} == {p}
        assert int(halved.private_dimensions.sum()) == 120 * p

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
