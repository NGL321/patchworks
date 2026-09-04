"""`benchmarks/floor_price.py`: the instrument that prices ADR-0032's floor (#436).

**Nothing here pins a reading of a run.** What the floor cost and what it bought
is a reading of a surface later tickets are expected to move, and a test holding
today's numbers would have to be deleted by whoever moves it —
`tests/test_graph_transmission.py`'s reasoning, and #150's before it.

What is worth holding is that the instrument is **right**, because this read is a
pre-registration whose verdict is a comparison of two runs, and every way of
getting a two-run comparison wrong produces a plausible number:

* **`without_floor` removes the floor and nothing else, and puts it back.** The
  `before` surface has to be the surface `main` has today. A patch that leaked
  into the `after` run — or into the rest of the suite — would make the two runs
  the same run and the whole read would report a difference of seeds.
* **The derived per-direction quantities are the quantities they are named
  after.** `effective rank`, `off-channel share` and #142's ratio are each
  checked against an operator whose spectrum is known by construction, at both
  ends of their range: a rank-1 operator and a flat one.
* **The chain is a composition and not a product of gains.** Composing the hops
  and reading the spectrum of the composition is the whole reason the chain
  section can net cost against benefit; multiplying the per-hop top gains would
  assume away #233's composition gap, and the two differ.
* **The routes are the same in both runs.** `chain_paths` is a function of the
  dome alone, so it cannot move between the runs — held directly, because the
  module docstring's second limit is a claim about the instrument.
"""

import floor_price as fp
import numpy as np
import pytest
import torch

from conftest import SMALL
from patchworks.graph import build_graph
from patchworks.restriction import RestrictionMaps


@pytest.fixture(scope="module")
def dome():
    return build_graph(SMALL)


def diagonal(values: list[float], rows: int | None = None) -> torch.Tensor:
    """An operator whose singular values are exactly `values`."""
    rows = rows if rows is not None else len(values)
    operator = torch.zeros(rows, len(values), dtype=torch.float64)
    for i, value in enumerate(values):
        operator[i, i] = value
    return operator


class TestTheFloorIsRemovedAndRestored:
    """`without_floor` is the whole of what makes the two runs comparable."""

    def test_inside_the_context_flatten_does_nothing(self, dome):
        maps = RestrictionMaps(dome, generator=torch.Generator().manual_seed(0))
        with torch.no_grad():
            maps.maps.normal_(0.0, 1.0, generator=torch.Generator().manual_seed(1))
            maps.maps.mul_(maps.support)
        before = maps.maps.clone()
        with fp.without_floor():
            maps._flatten()
        assert torch.equal(maps.maps, before)

    def test_outside_the_context_flatten_flattens(self, dome):
        maps = RestrictionMaps(dome, generator=torch.Generator().manual_seed(0))
        with fp.without_floor():
            pass
        maps.project()
        reached = maps.floored
        # `project` runs the cap after the floor, so flatness is exact only
        # where the cap does not bite; that it moved at all is the claim here.
        assert float(maps.flatness()[reached].min()) > 0.0

    def test_the_patch_is_undone_even_when_the_body_raises(self, dome):
        original = RestrictionMaps._flatten
        with pytest.raises(RuntimeError):
            with fp.without_floor():
                raise RuntimeError("the training run died")
        assert RestrictionMaps._flatten is original

    def test_the_patch_does_not_outlive_the_context(self):
        original = RestrictionMaps._flatten
        with fp.without_floor():
            assert RestrictionMaps._flatten is not original
        assert RestrictionMaps._flatten is original


class TestThePerDirectionQuantities:
    """Each is checked at both ends of its range against a known spectrum."""

    def test_effective_rank_of_a_rank_one_operator_is_one(self):
        assert fp.effective_rank(np.array([3.0, 0.0, 0.0, 0.0])) == pytest.approx(1.0)

    def test_effective_rank_of_r_equal_values_is_r(self):
        assert fp.effective_rank(np.array([2.0] * 4)) == pytest.approx(4.0)

    def test_off_channel_share_is_zero_for_a_rank_one_operator(self):
        assert fp.off_channel_share(np.array([3.0, 0.0, 0.0])) == pytest.approx(0.0)

    def test_off_channel_share_of_a_flat_operator_is_one_minus_one_over_m(self):
        assert fp.off_channel_share(np.array([2.0] * 4)) == pytest.approx(0.75)

    def test_the_isotropic_ratio_is_sqrt_m_for_a_rank_one_operator(self):
        # The ceiling: all of `‖M‖_F` sits on one direction.
        assert fp.channel_over_isotropic(
            np.array([5.0, 0.0, 0.0, 0.0]), 4
        ) == pytest.approx(2.0)

    def test_the_isotropic_ratio_is_one_for_a_flat_operator(self):
        # The floor: a flat operator has no preferred direction to ride, which
        # is ADR-0032's flagged consequence for #142's sentence.
        assert fp.channel_over_isotropic(np.array([5.0] * 4), 4) == pytest.approx(1.0)

    def test_carried_counts_directions_at_a_tenth_of_the_channel(self):
        assert fp.carried(np.array([1.0, 0.5, 0.1, 0.09])) == 3

    def test_carried_is_one_for_a_near_rank_one_operator(self):
        # The instrument's whole point: at a machine-relative cut this would be
        # 4, which is the numerical rank and says nothing about the trade.
        assert fp.carried(np.array([1.0, 1e-3, 1e-4, 1e-5])) == 1

    def test_carried_is_m_for_a_flat_operator(self):
        assert fp.carried(np.array([2.0] * 4)) == 4

    def test_a_dead_operator_carries_nothing(self):
        assert fp.carried(np.zeros(4)) == 0


class TestTheHopsAndTheChains:
    """The route half: what is read, and that it is the same in both runs."""

    def test_every_hop_is_a_pair_of_distinct_edges_at_an_interior_cell(self, dome):
        for edge_in, cell, edge_out in fp.hops_of_graph(dome):
            assert edge_in != edge_out
            assert not dome.cells[cell].is_boundary
            assert edge_in in dome.incident[cell]
            assert edge_out in dome.incident[cell]

    def test_the_hop_count_is_every_ordered_pair_at_every_interior_cell(self, dome):
        expected = sum(
            len(dome.incident[c.id]) * (len(dome.incident[c.id]) - 1)
            for c in dome.cells
            if not c.is_boundary
        )
        assert len(fp.hops_of_graph(dome)) == expected

    def test_a_chain_runs_from_a_rim_cell_to_an_apex_cell(self, dome):
        import detectability as det

        apex, rim = set(det.apex(dome)), set(det.rim(dome))
        for source, path in fp.chain_paths(dome).items():
            assert source in rim
            reached = source
            for edge_id in path:
                edge = dome.edges[edge_id]
                assert reached in (edge.u, edge.v)
                reached = edge.v if edge.u == reached else edge.u
            assert reached in apex

    def test_a_chain_never_transits_another_boundary_cell(self, dome):
        for source, path in fp.chain_paths(dome).items():
            import detectability as det

            apex = set(det.apex(dome))
            reached = source
            for edge_id in path[:-1]:
                edge = dome.edges[edge_id]
                reached = edge.v if edge.u == reached else edge.u
                assert (not dome.cells[reached].is_boundary) or reached in apex

    def test_the_routes_are_a_function_of_the_dome_alone(self, dome):
        # The module docstring's second limit, held: a route that could move
        # between the two runs would mix a change of route into a change of
        # gain, at exactly the place the comparison has to be like-for-like.
        assert fp.chain_paths(dome) == fp.chain_paths(build_graph(SMALL))

    def test_the_chain_is_the_composition_and_not_the_product_of_gains(self, dome):
        import construction_grading as cg
        from patchworks.restriction import GAUGE_RHO
        from patchworks.tick import DEFAULT_GAMMA, reconciliation_gain

        maps = RestrictionMaps(dome, generator=torch.Generator().manual_seed(3))
        maps.project()
        gains = reconciliation_gain(dome, gamma=DEFAULT_GAMMA, rho=GAUGE_RHO)
        source, path = next(
            (s, p) for s, p in fp.chain_paths(dome).items() if len(p) > 2
        )
        composed = fp.chain_operator(dome, maps, gains, path)
        hops = [h for h in cg.hops_of(dome, path) if not dome.cells[h[1]].is_boundary]

        product = 1.0
        for key in hops:
            product *= float(fp.spectrum(fp.hop_operator(dome, maps, gains, key))[0])
        top = float(fp.spectrum(composed)[0])

        # Sub-multiplicativity is the arithmetic; that the two are not equal is
        # #233's composition gap, and it is why the chain is composed.
        assert top <= product * (1 + 1e-9)
        assert top != pytest.approx(product, rel=1e-6)

    def test_the_chain_operator_maps_the_first_lane_to_the_last(self, dome):
        from patchworks.restriction import GAUGE_RHO
        from patchworks.tick import DEFAULT_GAMMA, reconciliation_gain

        maps = RestrictionMaps(dome, generator=torch.Generator().manual_seed(4))
        gains = reconciliation_gain(dome, gamma=DEFAULT_GAMMA, rho=GAUGE_RHO)
        for source, path in fp.chain_paths(dome).items():
            composed = fp.chain_operator(dome, maps, gains, path)
            if composed is None:
                continue
            assert composed.shape == (dome.edges[path[-1]].m, dome.edges[path[0]].m)


class TestTheBookedPrice:
    """ADR-0032's arithmetic, so the measured column has something to sit beside."""

    def test_the_booked_price_is_the_product_of_sqrt_m_over_the_path(self, dome):
        path = next(iter(fp.chain_paths(dome).values()))
        expected = float(np.prod([np.sqrt(dome.edges[e].m) for e in path]))
        assert fp.booked_cost(dome, path) == pytest.approx(expected)

    def test_an_empty_path_is_priced_at_one(self, dome):
        assert fp.booked_cost(dome, ()) == pytest.approx(1.0)
