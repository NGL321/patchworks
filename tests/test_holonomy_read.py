"""`benchmarks/holonomy_read.py`: what the departure measure means (#453, for #315).

**Nothing here pins a reading of a run.** The numbers the rig prints are
readings of a surface later tickets are expected to move; what is held is the
meaning of the columns, because every way of getting a departure-from-identity
measure wrong produces a plausible number:

* **A cycle is a cycle.** The basis has exactly the cycle rank of the interior
  subgraph, every cycle closes, and every cell on one is a predicting cell --
  a hop through a boundary cell is not transport (ADR-0016).
* **The identity reads 0 and chance reads 1.** A surface whose maps agree
  around a loop must read 0, or the rig cannot see the thing ADR-0032 asserts;
  and the analytic null `E‖Q − I‖_F² = 2m` on `O(m)` must be what the
  flat-and-independent draw actually reads, or the normalisation is decoration.
* **The columns do not read the gauge.** ADR-0010 fixes map scale and leaves an
  edge's frame free, so re-choosing every edge stalk's basis must move no
  column. This is the argument the module makes -- holonomy conjugates -- and
  it is checkable rather than asserted.
* **The reconciliation gain is absent on purpose**, so scaling any cell's maps
  must leave the identification column exactly where it was.
"""

import holonomy_read as hr
import numpy as np
import pytest
import torch

from conftest import SMALL
from patchworks.graph import build_graph
from patchworks.restriction import pair_index


@pytest.fixture(scope="module")
def dome():
    return build_graph(SMALL)


@pytest.fixture(scope="module")
def cycles(dome):
    return hr.cycle_basis(dome)


class TestTheBasis:
    """The cycles are the graph's, and they close."""

    def test_the_basis_has_the_cycle_rank(self, dome, cycles):
        interior, edges = hr.interior_graph(dome)
        # One connected interior subgraph on `SMALL` as on `DEFAULT_SPEC`; the
        # rig checks the rank itself and this holds the arithmetic down here.
        assert len(cycles) == len(edges) - len(interior) + 1

    def test_every_cycle_closes_through_interior_cells_only(self, dome, cycles):
        for cycle in cycles:
            hops = hr.cycle_hops(dome, cycle)
            assert len(hops) == len(cycle)
            assert all(not dome.cells[hop[1]].is_boundary for hop in hops)
            # The last hop hands back to the first edge: the walk is closed.
            assert hops[-1][2] == hops[0][0]

    def test_no_cycle_routes_through_a_boundary_cell(self, dome, cycles):
        interior, _ = hr.interior_graph(dome)
        inside = set(interior)
        for cycle in cycles:
            for edge_id in cycle:
                edge = dome.edges[edge_id]
                assert edge.u in inside and edge.v in inside


class TestWhatTheColumnsMean:
    """0 at the identity, 1 at chance -- or the rig is reading nothing."""

    def test_the_identity_reads_zero(self):
        operator = torch.eye(4, dtype=torch.float64)
        row = hr.departures(operator)
        assert row["identification"] == pytest.approx(0.0, abs=1e-12)
        assert row["flatness"] == pytest.approx(1.0)
        assert row["channel_return"] == pytest.approx(1.0)

    def test_scale_alone_is_not_a_departure(self):
        """`H = 3I` is the identity transport at another scale, not a rotation.

        The gain the rig leaves out is exactly this: a positive scalar per cell.
        """
        row = hr.departures(3.0 * torch.eye(4, dtype=torch.float64))
        assert row["identification"] == pytest.approx(0.0, abs=1e-12)
        assert row["sigma_max"] == pytest.approx(3.0)

    def test_minus_the_identity_is_maximally_opposed(self):
        row = hr.departures(-torch.eye(4, dtype=torch.float64))
        assert row["identification"] == pytest.approx(np.sqrt(2.0))

    def test_chance_reads_one(self):
        """`E‖Q − I‖_F² = 2m` for Haar-uniform `Q`, which is why 1 is the null."""
        generator = torch.Generator().manual_seed(7)
        squares = []
        for _ in range(400):
            draw = torch.empty(4, 4, dtype=torch.float64).normal_(generator=generator)
            u, _, vh = torch.linalg.svd(draw)
            squares.append(hr.departures(u @ vh)["identification"] ** 2)
        assert np.mean(squares) == pytest.approx(1.0, abs=0.05)


class TestASurfaceThatAgrees:
    """The end-to-end check: maps that agree around a loop must read 0.

    ADR-0032 asserts identification agreement is bought by the objective. The
    rig can only report that it was not if it would report that it *was* --
    otherwise a reading at chance is consistent with an instrument that always
    reads chance, and the ticket's whole question has no negative.
    """

    def aligned(self, dome):
        """Every cell sends every incident lane into the *same* frame.

        `F_{v,e}` is the first `m_e` rows of the identity on the cell's own
        columns, so a hop `F_out F_in^T` is the identity wherever the two lanes
        are the same width, and the holonomy around such a cycle is exactly `I`.
        This is what agreement looks like; nothing learns it here.
        """
        from patchworks.restriction import RestrictionMaps

        maps = RestrictionMaps(dome, generator=torch.Generator().manual_seed(1))
        with torch.no_grad():
            maps.maps.zero_()
            for edge in dome.edges:
                for side, cell_id in enumerate((edge.u, edge.v)):
                    i = pair_index(edge.id, side)
                    for row in range(edge.m):
                        maps.maps[i, row, row] = 1.0
                    maps.maps[i] *= maps.support[i]
        return maps

    def test_an_agreeing_surface_reads_the_identity(self, dome, cycles):
        maps = self.aligned(dome)
        rows = hr.read_surface(dome, maps, cycles)
        constant = [
            row
            for row, cycle in zip(rows, cycles)
            if len({dome.edges[e].m for e in cycle}) == 1
        ]
        assert constant, "no cycle of constant lane width to check against"
        for row in constant:
            assert row["identification"] == pytest.approx(0.0, abs=1e-6)
            assert row["flatness"] == pytest.approx(1.0, abs=1e-6)
            assert row["channel_return"] == pytest.approx(1.0, abs=1e-6)


class TestTheGauge:
    """ADR-0010 leaves an edge's frame free; no column may read it."""

    def test_re_framing_every_edge_stalk_moves_nothing(self, dome, cycles):
        maps = hr.flat_maps(dome, torch.Generator().manual_seed(3))
        before = hr.read_surface(dome, maps, cycles)
        generator = torch.Generator().manual_seed(11)
        with torch.no_grad():
            for edge in dome.edges:
                draw = torch.empty(edge.m, edge.m).normal_(generator=generator)
                frame, _, _ = torch.linalg.svd(draw)
                # The same frame at *both* endpoints: this is a change of basis
                # on the edge stalk, which is what the gauge leaves free.
                for side in (0, 1):
                    i = pair_index(edge.id, side)
                    block = maps.maps[i][: edge.m].clone()
                    maps.maps[i][: edge.m] = frame @ block
        after = hr.read_surface(dome, maps, cycles)
        for old, new in zip(before, after):
            # Exact in exact arithmetic. The maps are stored in float32 and the
            # read casts to float64, so the re-framed surface is the same
            # surface to float32 and no closer -- the tolerance is the storage's
            # and not the measure's.
            assert new["identification"] == pytest.approx(
                old["identification"], abs=1e-6
            )
            assert new["flatness"] == pytest.approx(old["flatness"], rel=1e-4)

    def test_scaling_a_cell_moves_no_identification(self, dome, cycles):
        """The reconciliation gain cancels out of the polar factor, as claimed."""
        maps = hr.flat_maps(dome, torch.Generator().manual_seed(5))
        before = hr.read_surface(dome, maps, cycles)
        with torch.no_grad():
            for edge in dome.edges:
                if not dome.cells[edge.u].is_boundary:
                    maps.maps[pair_index(edge.id, 0)] *= 2.5
        after = hr.read_surface(dome, maps, cycles)
        for old, new in zip(before, after):
            assert new["identification"] == pytest.approx(
                old["identification"], abs=1e-6
            )
