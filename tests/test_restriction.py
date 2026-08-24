"""The restriction maps (ticket #86, docs/spec/01-cell-and-sheaf.md, *The sheaf*).

What these hold down is the shape construction gave the maps -- the mask, the
gauge, and the endpoint indexing the tick's unit delay is built on. Nothing
here trains anything; the transport rule is #89's.
"""

import pytest
import torch

from patchworks.graph import DomeSpec, build_graph
from patchworks.restriction import GAUGE_RHO, RestrictionMaps, pair_index

# A dome small enough to iterate over in a test, built by the same rules as the
# real one: a taper from a sensory tiling through a vision lattice to a core.
SMALL = DomeSpec(
    patch_grid=4,
    vision_sides=(2,),
    somatomotor_sizes=(4,),
    core_sizes=(4, 3),
    core_degree=4,
    apex_degree=3,
)


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
