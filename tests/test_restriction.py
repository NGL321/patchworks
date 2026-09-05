"""The restriction maps (ticket #86, docs/spec/01-cell-and-sheaf.md, *The sheaf*).

What these hold down is the shape construction gave the maps -- the mask, the
gauge, and the endpoint indexing the tick's unit delay is built on. Nothing
here trains anything; the transport rule is #89's.
"""

import collections

import pytest
import torch

from patchworks.graph import build_graph
from patchworks.restriction import (
    GAUGE_C,
    GAUGE_RHO,
    RestrictionMaps,
    cell_gauges,
    gain_denominators,
    map_is_pinned,
    overlap_counts,
    pinned_incidence,
    pair_index,
)

from conftest import SMALL


@pytest.fixture
def dome():
    return build_graph(SMALL)


@pytest.fixture
def maps(dome):
    return RestrictionMaps(dome, generator=torch.Generator().manual_seed(0))


class TestEndpointIndexing:
    def test_a_map_per_edge_endpoint_not_per_edge(self, dome, maps):
        # The two ends of an edge are independent maps belonging to different
        # cells. If they were tied, agreement would be definitional and
        # disagreement could carry no information.
        assert maps.pairs == 2 * len(dome.edges)

    def test_the_partner_of_a_pair_is_its_index_flipped(self, dome, maps):
        for edge in dome.edges:
            u, v = pair_index(edge.id, 0), pair_index(edge.id, 1)
            assert u ^ 1 == v and v ^ 1 == u
            assert maps.owner[u] == edge.u and maps.owner[v] == edge.v

    def test_the_padded_tensor_covers_the_widest_edge_and_stalk(self, dome, maps):
        assert maps.edge_width == max(e.m for e in dome.edges)
        assert maps.stalk_width == max(c.stalk for c in dome.cells)
        assert maps.maps.shape == (maps.pairs, maps.edge_width, maps.stalk_width)


class TestTheMask:
    def test_a_map_is_zero_outside_its_structural_mask(self, dome, maps):
        assert torch.all(maps.maps[~maps.support] == 0)

    def test_the_mask_is_the_dome_s_own(self, dome, maps):
        # Read back per endpoint rather than trusted in bulk: the mask closes
        # and never re-opens, and it is the graph's, not this module's.
        for edge in dome.edges:
            for side, cell_id in enumerate((edge.u, edge.v)):
                permitted = dome.restriction_mask(edge.id, cell_id)
                row = maps.support[pair_index(edge.id, side)]
                assert torch.equal(row[: edge.m, : permitted.numel()].all(dim=0), permitted)
                assert not row[edge.m :].any()
                assert not row[:, permitted.numel() :].any()

    def test_padding_and_masking_are_the_same_kind_of_zero(self, maps):
        # Which is what lets one padded tensor stand in for 108 ragged ones.
        assert torch.equal(maps.maps != 0, (maps.maps != 0) & maps.support)


class TestTheGauge:
    def test_a_boundary_cell_s_own_maps_carry_the_exact_gauge(self, dome, maps):
        norms = maps.norms().detach()
        for pair, cell_id in enumerate(maps.owner.tolist()):
            if dome.cells[cell_id].is_boundary:
                assert float(norms[pair]) == pytest.approx(1.0, abs=1e-5)

    def test_the_predicting_end_of_a_sensory_edge_is_an_ordinary_interior_map(
        self, dome, maps
    ):
        # The gauge sorts by who *holds* the map, not by what the edge
        # connects: a boundary cell has no metric individuality to protect, and
        # its neighbour still does.
        lower, upper = maps.gauge_bounds
        for edge in dome.edges:
            for side, cell_id in enumerate((edge.u, edge.v)):
                pair = pair_index(edge.id, side)
                if dome.cells[cell_id].is_boundary:
                    assert (lower[pair], upper[pair]) == (1.0, 1.0)
                else:
                    assert (lower[pair], upper[pair]) == (1.0 / GAUGE_RHO, GAUGE_RHO)

    def test_every_map_starts_inside_the_band(self, maps):
        lower, upper = maps.gauge_bounds
        norms = maps.norms()
        assert torch.all(norms >= lower - 1e-5)
        assert torch.all(norms <= upper + 1e-5)

    def test_projection_pulls_a_map_back_into_the_band(self, maps):
        with torch.no_grad():
            maps.maps.mul_(10.0)
        maps.project()
        _lower, upper = maps.gauge_bounds
        assert torch.allclose(maps.norms(), upper, atol=1e-5)

    def test_projection_pushes_a_shrunken_map_back_up(self, maps):
        with torch.no_grad():
            maps.maps.mul_(1e-3)
        maps.project()
        lower, _upper = maps.gauge_bounds
        assert torch.allclose(maps.norms(), lower, atol=1e-5)

    def test_projection_re_applies_the_mask(self, maps):
        # A step that walked a weight outside the mask and a step that grew a
        # map past rho are the same kind of event, and neither is ever wanted.
        with torch.no_grad():
            maps.maps.add_(1.0)
        maps.project()
        assert torch.all(maps.maps[~maps.support] == 0)

    def test_a_gauge_band_below_one_is_refused(self, dome):
        with pytest.raises(ValueError, match="rho"):
            RestrictionMaps(dome, rho=0.5)


class TestRestrictAndSpread:
    def test_spread_is_the_transpose_of_restrict(self, maps):
        generator = torch.Generator().manual_seed(3)
        x = torch.randn(maps.pairs, maps.stalk_width, generator=generator)
        y = torch.randn(maps.pairs, maps.edge_width, generator=generator)
        # <Fx, y> == <x, F^T y>, per pair.
        assert torch.allclose(
            (maps.restrict(x) * y).sum(dim=-1),
            (x * maps.spread(y)).sum(dim=-1),
            atol=1e-5,
        )

    def test_a_masked_direction_comes_back_zero(self, dome, maps):
        y = torch.ones(maps.pairs, maps.edge_width)
        back = maps.spread(y)
        assert torch.all(back[~maps.support.any(dim=1)] == 0)

    def test_a_stalk_the_wrong_width_is_refused(self, maps):
        with pytest.raises(ValueError, match="stalks must be"):
            maps.restrict(torch.zeros(maps.pairs, maps.stalk_width + 1))
        with pytest.raises(ValueError, match="edge_values must be"):
            maps.spread(torch.zeros(maps.pairs, maps.edge_width + 1))


def test_the_real_dome_s_maps_are_all_gauge_fixed():
    """The sizes that matter: 682 edges, 1364 maps, none outside its band."""
    dome = build_graph()
    maps = RestrictionMaps(dome, generator=torch.Generator().manual_seed(0))
    lower, upper = maps.gauge_bounds
    norms = maps.norms()
    assert maps.pairs == 2 * len(dome.edges) == 1364
    assert torch.all((norms >= lower - 1e-5) & (norms <= upper + 1e-5))
    assert torch.all(maps.maps[~maps.support] == 0)



def _turn_the_maps_to_face_one_way(maps: RestrictionMaps) -> None:
    """The worst arrangement the projection has to survive.

    Every map loaded onto the same node stalk direction is the fully-coherent
    case the old denominator assumed and the new one refuses to. Nothing the
    transport rule does reaches this in one step; it is here because a bound
    that only holds for arrangements the rule happens to produce is not a bound.
    """
    with torch.no_grad():
        maps.maps.mul_(0.2)
        maps.maps[:, :, 0] += 1.0
        maps.maps.mul_(maps.support)


class TestTheIncoherenceTerm:
    """ADR-0010, *Incoherence is gauge-fixed too*, and #220's half of it.

    `gain_v` divides by `g_v^2 . c_v`, and what makes that a bound rather than a
    hope is this: the projection holds the surface to it after every transport
    step. The tests below are written against the *denominator the gain uses*
    rather than against a number, because the two being one thing is the whole
    of the change.
    """

    def test_the_bound_holds_at_every_cell_that_holds_it(self, dome, maps):
        _turn_the_maps_to_face_one_way(maps)
        maps.project()
        held = maps.gram_peaks()[maps.holding]
        target = gain_denominators(dome)[maps.holding]
        assert torch.all(held <= target * (1 + 1e-5))

    def test_the_worst_arrangement_breaks_the_bound_without_the_projection(
        self, dome, maps
    ):
        # Otherwise the test above passes on a surface that never needed it.
        _turn_the_maps_to_face_one_way(maps)
        held = maps.gram_peaks()[maps.holding]
        assert torch.any(held > gain_denominators(dome)[maps.holding])

    def test_the_mask_survives_the_transform(self, dome, maps):
        # The transform is one shared right-multiply per cell, and a right-
        # multiply is exactly the operation that could write into a column the
        # mask closed. It cannot here, because all of a cell's incident maps
        # share one mask and the transform is supported on it.
        _turn_the_maps_to_face_one_way(maps)
        maps.project()
        assert torch.all(maps.maps[~maps.support] == 0)

    def test_a_pinned_map_is_left_exactly_where_the_gauge_put_it(self, dome, maps):
        # A boundary cell's maps carry the exact gauge, so there is no scale
        # freedom for the transform to spend and it does not try. They come out
        # of the projection at 1, to the last bit the float has.
        _turn_the_maps_to_face_one_way(maps)
        maps.project()
        assert torch.allclose(
            maps.norms()[maps.pinned],
            torch.ones(int(maps.pinned.sum())),
            atol=1e-6,
        )

    @staticmethod
    def _energy(dome, maps):
        """`[cells]`: `sum_e ||F_ev||_F^2`, which is the Gram's trace."""
        return torch.zeros(len(dome.cells)).index_add_(
            0, maps.owner, maps.norms().detach() ** 2
        )

    @staticmethod
    def _lean_the_maps_one_way(maps, amount):
        with torch.no_grad():
            maps.maps[:, :, 0] += amount
            maps.maps.mul_(maps.support)
            lower, upper = maps.gauge_bounds
            norms = maps.norms().clamp_min(1e-12)
            maps.maps.mul_(
                (norms.clamp(lower, upper) / norms).unsqueeze(-1).unsqueeze(-1)
            )

    def test_a_correctable_cell_keeps_all_of_its_energy(self, dome, maps):
        # `sum_e ||F_ev||_F^2` is the Gram's trace and it is what the band
        # holds. Where the cell's other directions have room to take the excess,
        # the step moves it there and the trace does not move at all: this is a
        # redistribution, not a cap.
        self._lean_the_maps_one_way(maps, 0.5)
        before = self._energy(dome, maps)
        maps.project()
        held = maps.holding
        assert torch.all(maps.gram_peaks()[held] <= gain_denominators(dome)[held] * 1.00001)
        assert torch.allclose(self._energy(dome, maps)[held], before[held], rtol=1e-4)

    def test_the_worst_arrangement_still_keeps_nearly_all_of_it(self, dome, maps):
        # Turned fully one way there is more excess than headroom, so some is
        # lost -- but 10% of it, against the 71% a bare cap would take out of
        # the same cell. That difference is the whole reason for the fill: a cap
        # would take energy out on every tick and ratchet the maps down onto the
        # band's lower edge, which is the map collapse the band exists to
        # prevent, arriving by a different road.
        _turn_the_maps_to_face_one_way(maps)
        self._lean_the_maps_one_way(maps, 0.0)
        before = self._energy(dome, maps)
        maps.project()
        held = maps.holding
        after = self._energy(dome, maps)
        assert torch.all(after[held] >= before[held] * 0.85)

    def test_a_cell_already_inside_the_bound_is_not_touched(self, maps):
        # The draw is incoherent enough to start inside, so the cap has nothing
        # to do and must do nothing: a transform applied every tick to a surface
        # that does not need it is a slow leak, not a projection. Read against
        # `_push_apart` rather than the whole projection, because the spectral
        # floor in front of it is *not* a no-op on the draw -- a random map is
        # nowhere near flat, and flattening it is the point.
        before = maps.maps.detach().clone()
        maps._push_apart()
        assert torch.allclose(maps.maps, before, atol=1e-6)

    def test_the_band_still_holds_after_the_transform(self, maps):
        _turn_the_maps_to_face_one_way(maps)
        maps.project()
        lower, upper = maps.gauge_bounds
        norms = maps.norms()
        assert torch.all(norms <= upper + 1e-5)
        assert torch.all(norms >= lower - 1e-5)

    def test_projection_is_idempotent(self, maps):
        _turn_the_maps_to_face_one_way(maps)
        maps.project()
        once = maps.maps.detach().clone()
        maps.project()
        assert torch.allclose(maps.maps, once, atol=1e-6)


class TestTheDenominatorTheGainDividesBy:
    def test_the_gauge_is_the_band_inside_and_exactly_one_at_the_boundary(self, dome):
        gauges = cell_gauges(dome)
        for cell in dome.cells:
            assert float(gauges[cell.id]) == (1.0 if cell.is_boundary else GAUGE_RHO)

    def test_the_pigeonhole_floor_is_what_saves_the_drive(self):
        # The drive carries deg = 8 maps on a stalk of dimension 1. Eight
        # directions cannot be mutually orthogonal in one dimension, so a bare
        # global c = 2 there is an unsafe bound and not a loose one -- and the
        # projection cannot fix it, because the drive's cell is a boundary cell
        # and its maps are pinned. The floor is what makes the number true.
        dome = build_graph()
        counts = overlap_counts(dome)
        drive = [
            c.id for c in dome.cells if c.stalk == 1 and dome.degrees[c.id] > GAUGE_C
        ]
        assert drive, "no cell with more incident maps than stalk dimensions"
        for cell_id in drive:
            assert float(counts[cell_id]) == dome.degrees[cell_id]

    def test_the_outer_min_keeps_it_inside_what_the_band_already_gave(self):
        dome = build_graph()
        counts = overlap_counts(dome)
        for cell in dome.cells:
            assert float(counts[cell.id]) <= dome.degrees[cell.id]

    def test_the_interior_is_the_global_c_everywhere_on_this_dome(self):
        dome = build_graph()
        counts = overlap_counts(dome)
        interior = [c.id for c in dome.cells if not c.is_boundary]
        assert {float(counts[i]) for i in interior} == {float(GAUGE_C)}

    def test_a_wholly_pinned_incidence_takes_deg(self):
        # #228. The projection spends scale freedom and a pinned map has none,
        # so where a cell has none anywhere on its incidence nothing enforces a
        # count below deg(v) -- while the exact gauge makes it true unaided.
        dome = build_graph()
        counts = overlap_counts(dome)
        pinned = pinned_incidence(dome)
        assert any(pinned), "no cell with a wholly pinned incidence"
        for cell in dome.cells:
            if pinned[cell.id]:
                assert float(counts[cell.id]) == dome.degrees[cell.id]

    def test_the_actuator_is_the_one_cell_the_rule_moves_on_this_dome(self):
        # deg = 3 on a stalk of 6, so the pigeonhole floor leaves it at the
        # global c = 2 while nothing pushes its three maps apart. Every other
        # boundary cell already agreed with deg(v), by deg = 1 or by the floor.
        dome = build_graph()
        counts = overlap_counts(dome)

        def superseded(cell):
            degree = dome.degrees[cell.id]
            return min(degree, max(GAUGE_C, -(-degree // cell.stalk)))

        moved = [c.id for c in dome.cells if float(counts[c.id]) != superseded(c)]
        assert len(moved) == 1
        actuator = moved[0]
        assert dome.degrees[actuator] == 3 and dome.cells[actuator].stalk == 6
        assert float(counts[actuator]) == 3.0
        assert float(gain_denominators(dome)[actuator]) == 3.0

    def test_the_boundary_correction_is_uniform(self):
        # Both denominators are proportional to deg(v) on a lane 8 wide -- the
        # superseded `sum_e m_e` is 8.deg, the new `g_v^2.c_v` is deg -- so the
        # correction is 8.00x at every such cell and the actuator is no longer
        # the graded exception (`02-tick-semantics.md`). The drive is the one
        # boundary cell outside the figure, and for a reason about lane width
        # rather than about the clamp: its eight lanes are m = 1, so the two
        # denominators already agreed there.
        dome = build_graph()
        denominators = gain_denominators(dome)
        corrections = collections.Counter(
            dome.stalk_sums[cell_id] / float(denominators[cell_id])
            for cell_id in dome.boundary
        )
        assert corrections == {8.0: 263, 1.0: 1}
        drive = [
            c.id for c in dome.cells if c.stalk == 1 and dome.degrees[c.id] > GAUGE_C
        ]
        assert len(drive) == 1
        assert dome.stalk_sums[drive[0]] / float(denominators[drive[0]]) == 1.0

    def test_the_condition_is_read_one_map_at_a_time(self):
        # The rule is stated over pinned incidence rather than `is_boundary` so
        # that a partly-pinned cell is visibly uncovered on a graph that is not
        # this dome. Nothing here is partly pinned, and that is the claim.
        dome = build_graph()
        for cell in dome.cells:
            per_map = [map_is_pinned(dome, e, cell.id) for e in dome.incident[cell.id]]
            assert all(per_map) or not any(per_map)
            assert pinned_incidence(dome)[cell.id] == all(per_map)

    def test_the_denominator_is_the_two_terms_multiplied(self, dome):
        assert torch.allclose(
            gain_denominators(dome), cell_gauges(dome) ** 2 * overlap_counts(dome)
        )


class TestTheSpectralFloor:
    """ADR-0032, and the three mechanics it left to the build (#432).

    The maps are learning isometric transport, and the constraint that expresses
    it is a per-map floor at `σ_min ≥ ‖F‖_F/√m` — attainable only with equality
    throughout, so the floor and the projection onto the nearest scaled
    co-isometry are one operation. These read the property on the *real* dome,
    where the attainability argument was made, and not on `SMALL`.
    """

    @pytest.fixture(scope="class")
    def real(self):
        return build_graph()

    @pytest.fixture
    def real_maps(self, real):
        return RestrictionMaps(real, generator=torch.Generator().manual_seed(0))

    def test_every_map_the_floor_reaches_comes_out_flat(self, real_maps):
        real_maps.project()
        assert torch.all(real_maps.flatness()[real_maps.floored] > 1 - 1e-4)

    def test_the_floor_preserves_the_frobenius_norm_exactly(self, real_maps):
        # This is why the floor sits beside ADR-0010's gauge rather than against
        # it: the band holds `‖F‖_F` and the floor spends none of it, only
        # spreading it evenly across the singular values.
        before = real_maps.norms().detach().clone()
        real_maps._flatten()
        assert torch.allclose(real_maps.norms(), before, rtol=1e-5)

    def test_the_floor_lifts_a_dead_direction_and_the_cap_cannot(self, real_maps):
        # The whole difference between a floor and a cap. `_push_apart` gates
        # its water-fill on `live = eigenvalues > peak * 1e-9` because scaling
        # zero leaves zero, so it flattens survivors and can never resurrect one.
        with torch.no_grad():
            real_maps.maps.zero_()
            real_maps.maps[:, 0, 0] = 1.0
            real_maps.maps.mul_(real_maps.support)
        assert torch.all(real_maps.flatness()[real_maps.floored] < 1e-5)
        # The cap does *move* this surface -- it is the fully-coherent
        # arrangement, so the one live direction is over the target and gets
        # rescaled. What it cannot do is change the rank, which is the property
        # at issue: every map is still rank 1 when it comes out.
        real_maps._push_apart()
        assert torch.all(real_maps.flatness()[real_maps.floored] < 1e-5)
        real_maps._flatten()
        assert torch.all(real_maps.flatness()[real_maps.floored] > 1 - 1e-4)

    def test_the_padding_and_the_mask_come_back_exactly_zero(self, real_maps):
        # A batched SVD over the padded tensor would flatten the *padding*, and
        # a co-isometry fitted to structural zeros writes weights into rows and
        # columns construction closed. Grouping by `(m, k)` is what prevents it,
        # and `exactly` is the word: not `allclose`, zero.
        real_maps.project()
        assert torch.all(real_maps.maps[~real_maps.support] == 0)

    def test_the_six_unattainable_masks_are_excluded_by_name(self, real, real_maps):
        # ADR-0032's mask-attainability read: `k < m` means the mask cannot
        # contain a co-isometry at all, so `σ_min = 0` whatever the projection
        # does and the projection would shrink `‖F‖_F` by `√(k/m)`, fighting the
        # exact gauge. Three touch and three proprioceptive.
        #
        # **They were nine, and #474 released three of them.** The actuator's
        # three maps sat at `(m = 8, k = 6)` and were unattainable by two
        # columns; `boundary_m` 8 -> 4 puts them at `(4, 6)`, where the mask
        # does contain a co-isometry, so they take the floor like any other map.
        # Nothing in `RestrictionMaps` was edited to do that -- the population is
        # computed from the mask, which is the property #415 built it for.
        # ADR-0032's ledger names nine, read on the `boundary_m = 8` surface.
        unattainable = []
        for edge in real.edges:
            for side, cell_id in enumerate((edge.u, edge.v)):
                k = int(real.restriction_mask(edge.id, cell_id).sum())
                if k < edge.m:
                    unattainable.append((edge.m, k, pair_index(edge.id, side)))
        assert sorted({(m, k) for m, k, _ in unattainable}) == [(4, 1), (4, 2)]
        assert len(unattainable) == 6
        where = [i for _m, _k, i in unattainable]
        assert not bool(real_maps.floored[where].any())
        # And they are all pinned, which is why the ADR could state the floor on
        # the banded maps without leaving a banded map behind.
        assert bool(real_maps.pinned[where].all())

    def test_the_floor_reaches_attainable_pinned_maps_too(self, real_maps):
        # The exclusion is by attainability, not by pinning: `_push_apart` skips
        # a pinned map for want of *scale* freedom, and the floor needs none,
        # because it preserves `‖F‖_F`. Skipping them would leave the rim's own
        # end of 256 sensory edges unflattened, and isometry is a property of
        # the edge *pair*, so one flat end buys nothing.
        # 256 sensory edge-ends, plus the actuator's three, which #474 made
        # attainable when `boundary_m` went 8 -> 4 (see the exclusion test
        # above). It read 256 on the `boundary_m = 8` surface.
        assert int((real_maps.floored & real_maps.pinned).sum()) == 259

    def test_a_pinned_map_still_leaves_the_projection_at_exactly_one(self, real_maps):
        real_maps.project()
        assert torch.allclose(
            real_maps.norms()[real_maps.pinned],
            torch.ones(int(real_maps.pinned.sum())),
            atol=1e-6,
        )

    def test_the_cap_still_holds_at_exit_with_the_floor_in_front_of_it(
        self, real, real_maps
    ):
        # The ordering decision (#432, mechanic 1). The floor and the cap cannot
        # both be exactly true at exit, and the cap takes the last slot because
        # `reconciliation_gain` divides by it on every tick (#220). So this is
        # the invariant that must hold exactly, under the worst arrangement, and
        # `flatness()` is what reads what the floor gave up for it.
        _turn_the_maps_to_face_one_way(real_maps)
        real_maps.project()
        held = real_maps.holding
        assert torch.all(
            real_maps.gram_peaks()[held] <= gain_denominators(real)[held] * (1 + 1e-5)
        )

    def test_flatness_reads_zero_where_the_floor_cannot_reach(self, real_maps):
        real_maps.project()
        widths = torch.tensor(
            [[float(edge.m)] * 2 for edge in real_maps.dome.edges]
        ).reshape(-1)
        unreachable = ~real_maps.floored & real_maps.pinned & (widths > 1)
        # Six since #474, and nine before it: the actuator's three left this set
        # for the floored one when `boundary_m` went 8 -> 4.
        assert int(unreachable.sum()) == 6
        assert torch.all(real_maps.flatness()[unreachable] < 1e-5)

    def test_where_the_cap_bites_the_floor_is_given_up_and_recovered(self, real_maps):
        # The price of the ordering, measured rather than asserted away. Under
        # the fully-coherent arrangement the cap binds at every holding cell, so
        # the projection is *not* a one-step fixed point the way it is when only
        # the band and the mask are live: the cap un-flattens what the floor
        # flattened, and the next projection flattens it again. What matters is
        # the direction -- flatness climbs monotonically toward 1 across passes
        # while the cap stays exactly held -- so the two steps converge on a
        # surface satisfying both rather than oscillating between them.
        _turn_the_maps_to_face_one_way(real_maps)
        seen = []
        for _pass in range(4):
            real_maps.project()
            seen.append(float(real_maps.flatness()[real_maps.floored].amin()))
        assert seen[0] > 0.9
        assert seen == sorted(seen)
