"""The restriction maps (ticket #86, docs/spec/01-cell-and-sheaf.md, *The sheaf*).

What these hold down is the shape construction gave the maps -- the mask, the
gauge, and the endpoint indexing the tick's unit delay is built on. Nothing
here trains anything; the transport rule is #89's.
"""

import pytest
import torch

from patchworks.graph import build_graph
from patchworks.restriction import (
    GAUGE_C,
    GAUGE_RHO,
    RestrictionMaps,
    cell_gauges,
    gain_denominators,
    overlap_counts,
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
        # The draw is incoherent enough to start inside, so the projection has
        # nothing to do and must do nothing: a transform applied every tick to a
        # surface that does not need it is a slow leak, not a projection.
        before = maps.maps.detach().clone()
        maps.project()
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

    def test_the_denominator_is_the_two_terms_multiplied(self, dome):
        assert torch.allclose(
            gain_denominators(dome), cell_gauges(dome) ** 2 * overlap_counts(dome)
        )
