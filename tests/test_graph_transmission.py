"""`benchmarks/graph_transmission.py`: the two estimators, and that the script runs (#150).

#150 is a measurement, not a decision, so nothing here pins its verdict — the
rim-to-apex resistance, the gauge's share of the hop and the curvature table are
readings of a graph that later tickets are expected to change, and a test
holding today's numbers would have to be deleted by whoever changes it. That is
`tests/test_untrained_fixed_point.py`'s reasoning and it applies unchanged.

What is worth holding is that the two estimators are **right**, because the
whole ticket is arithmetic and an off-by-one in either would produce a
plausible, wrong diagnosis that nothing downstream would catch. Both are checked
against closed forms rather than against themselves:

* **Effective resistance** by Foster's theorem — `sum_e w_e R(u,v) = n - 1` over
  any connected graph, exactly — plus series and parallel, which fix the scale
  and the weighting convention that theorem alone would let drift together.
* **Balanced Forman curvature** against graphs whose curvature Definition 1
  gives in closed form: `n/(n-1)` on a complete graph, zero on a long cycle,
  zero at a degree-one endpoint.

And that the script still runs against the API, in the shape
`tests/test_untrained_fixed_point.py` and `tests/test_agent.py::TestBenchmark`
ask it of the other benchmarks. It is graph-side and needs no sandbox, so the
whole run is affordable in the suite rather than smoke-tested on a small dome.
"""

import numpy as np
import pytest

import graph_transmission as transmission
from patchworks.graph import build_graph

from conftest import SMALL


def adjacency_of(edges: list[tuple[int, int]], size: int, weight: float = 1.0):
    a = np.zeros((size, size))
    for u, v in edges:
        a[u, v] += weight
        a[v, u] += weight
    return a


def neighbours_of(edges: list[tuple[int, int]], size: int) -> list[set[int]]:
    sets: list[set[int]] = [set() for _ in range(size)]
    for u, v in edges:
        sets[u].add(v)
        sets[v].add(u)
    return sets


class TestEffectiveResistance:
    """The estimator, against the closed forms that fix its scale and convention."""

    def test_series_adds(self):
        """Three unit edges in a line: end to end is 3."""
        r = transmission.effective_resistance(adjacency_of([(0, 1), (1, 2), (2, 3)], 4))
        assert r[0, 3] == pytest.approx(3.0)
        assert r[1, 2] == pytest.approx(1.0)

    def test_parallel_divides(self):
        """A conductance of 4 across one pair is a resistance of 1/4.

        The weight is a **conductance**: `w_e = m_e` says a wider lane is
        an easier route, not a harder one. Getting that backwards would invert
        the whole lane-weighted table without changing its shape.
        """
        r = transmission.effective_resistance(adjacency_of([(0, 1)], 2, weight=4.0))
        assert r[0, 1] == pytest.approx(0.25)

    def test_foster(self):
        """`sum_e w_e R(u,v) = n - 1` over a connected graph, exactly (Foster, 1949).

        The one global check on the pseudoinverse: it constrains every entry the
        edges touch at once, and it is what makes the printed `edge cut share`
        a probability rather than an index — the shares are a spanning tree's
        edge distribution and sum to `n - 1`.
        """
        dome = build_graph(SMALL)
        for weight in (lambda e: 1.0, lambda e: float(e.m)):
            adjacency = transmission.weighted_adjacency(dome, weight)
            r = transmission.effective_resistance(adjacency)
            total = sum(weight(e) * r[e.u, e.v] for e in dome.edges)
            assert total == pytest.approx(len(dome.cells) - 1, rel=1e-9)

    def test_symmetric_and_zero_on_the_diagonal(self):
        dome = build_graph(SMALL)
        r = transmission.effective_resistance(
            transmission.weighted_adjacency(dome, lambda e: 1.0)
        )
        assert np.allclose(r, r.T)
        assert np.allclose(np.diag(r), 0.0, atol=1e-9)

    def test_a_leaf_edge_is_a_bridge(self):
        """A boundary patch has one edge, so its cut share is exactly 1.

        The claim the sensory row of the printed table rests on, checked where
        it is provable rather than where it is measured.
        """
        dome = build_graph(SMALL)
        r = transmission.effective_resistance(
            transmission.weighted_adjacency(dome, lambda e: float(e.m))
        )
        leaves = [c.id for c in dome.cells if dome.degrees[c.id] == 1]
        assert leaves, "the small dome should still have leaf boundary cells"
        for cell_id in leaves:
            edge = dome.edges[dome.incident[cell_id][0]]
            assert edge.m * r[edge.u, edge.v] == pytest.approx(1.0)


class TestBalancedForman:
    """Definition 1, against the graphs it evaluates in closed form."""

    @pytest.mark.parametrize("n", [3, 4, 5, 6])
    def test_complete_graph(self, n):
        """`K_n`: every edge has `n - 2` triangles and no squares, so `Ric = n/(n-1)`.

        Substituting `d = n - 1` and `|T| = n - 2` into Definition 1 leaves
        `(4 + 3(n-2))/(n-1) - 2`, which is `n/(n-1)`.
        """
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        adjacency = neighbours_of(edges, n)
        for u, v in edges:
            assert transmission.edge_curvature(adjacency, u, v) == pytest.approx(
                n / (n - 1)
            )

    def test_long_cycle_is_flat(self):
        """`C_6`: degree 2, no triangles and no squares, so the first three terms alone."""
        n = 6
        edges = [(i, (i + 1) % n) for i in range(n)]
        adjacency = neighbours_of(edges, n)
        for u, v in edges:
            assert transmission.edge_curvature(adjacency, u, v) == pytest.approx(0.0)

    def test_a_degree_one_endpoint_is_zero(self):
        """Definition 1 is stated for `d_i, d_j > 1` and the rest is zero by convention.

        Which is why the sensory row of the printed table is 0 and not negative:
        every patch cell is a leaf, so curvature is **silent** there rather than
        favourable, and the cut share is what speaks instead.
        """
        adjacency = neighbours_of([(0, 1), (1, 2), (2, 0), (0, 3)], 4)
        assert transmission.edge_curvature(adjacency, 0, 3) == pytest.approx(0.0)

    def test_every_dome_edge_is_finite(self):
        dome = build_graph(SMALL)
        curvature = transmission.balanced_forman(dome)
        assert curvature.shape == (len(dome.edges),)
        assert np.isfinite(curvature).all()


class TestAnalyticHop:
    """The predicted hop, and the one direction the stalk table asserts."""

    def test_it_agrees_with_the_drawn_hop(self, capsys):
        """The closed form and the Monte Carlo are the same number to a few percent.

        `analytic_hop` prices construction parameters the sheaf would have to be
        rebuilt to measure, so it is only worth anything if it reproduces the
        drawn reading on a dome that *is* built. Both are at `rho = 1`, the
        norm every map is initialised at.
        """
        dome = build_graph(SMALL)
        transmission.gauge_section(dome)
        printed = capsys.readouterr().out
        drawn = float(printed.split("->  one hop")[1].split()[0])
        assert transmission.analytic_hop(dome) == pytest.approx(drawn, rel=0.1)

    def test_a_wider_lane_lowers_the_hop(self):
        """At a fixed Frobenius gauge, `m` costs transmission and buys rank.

        The direction the stalk table reports, and the whole of what it claims:
        the same map norm over more rows is less per row (`1/sqrt(m)`), and the
        gain's `sum_e m_e` arm takes another `1/m` above `m = rho^2`.

        **This is the one assertion here that a later ticket may have to
        delete**, and deliberately so: it holds because the gauge fixes a
        Frobenius norm independent of `m` (ADR-0010). A gauge that scaled with
        the lane would reverse it, and whoever writes that should find
        this test in the way.
        """
        import dataclasses

        hops = [
            transmission.analytic_hop(
                build_graph(dataclasses.replace(SMALL, interior_m=interior_m))
            )
            for interior_m in (2, 4, 6)
        ]
        assert hops == sorted(hops, reverse=True)


class TestHopDistance:
    def test_breadth_first_reaches_every_cell(self):
        dome = build_graph(SMALL)
        apex = max(c.index.level for c in dome.cells if not c.is_boundary)
        sources = [c.id for c in dome.cells if not c.is_boundary and c.index.level == apex]
        distance = transmission.hop_distance(dome, sources)
        assert (distance >= 0).all(), "the dome is connected"
        assert set(distance[sources]) == {0}


class TestBenchmark:
    """The script runs end to end against the current API, and reports its three parts."""

    def test_main(self, capsys):
        transmission.main()
        printed = capsys.readouterr().out
        for heading in (
            "### effective resistance, unweighted",
            "### effective resistance, stalk m_e",
            "### the gauge's share of one hop",
            "### balanced Forman curvature",
        ):
            assert heading in printed

    def test_the_frobenius_identity_holds(self, capsys):
        """The split into gauge / dilution / gain is only meaningful if it does.

        `E|F u| ~= |F|_F / sqrt(d)` for a uniform direction, and the script
        prints the drawn ratio against the predicted one. This asks that the
        agreement is close, not that it is exact: the ratio is the mean of a
        chi distribution over `d` degrees of freedom against its own square
        root, which sits a few percent low and narrows with `d`.
        """
        transmission.gauge_section(build_graph(SMALL))
        printed = capsys.readouterr().out
        ratios = [
            float(line.split("ratio mean")[1].split()[0])
            for line in printed.splitlines()
            if "ratio mean" in line
        ]
        assert len(ratios) == 2
        assert all(0.9 < ratio < 1.05 for ratio in ratios)
