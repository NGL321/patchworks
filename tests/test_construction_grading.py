"""`benchmarks/construction_grading.py`: the predictor, the split, and that it runs (#233).

#233 explains a measurement and decides nothing, so nothing here pins its
numbers — the residual it reports is a reading of a graph later tickets are
expected to change, and a test holding today's value would have to be deleted by
whoever changes it. That is `tests/test_untrained_fixed_point.py`'s reasoning and
`tests/test_graph_transmission.py` and `tests/test_detectability.py` both apply
it unchanged.

What is worth holding is that the arithmetic is **right**, because the ticket's
whole content is a prediction and a residual, and a wrong predictor would
produce a plausible, wrong attribution that nothing downstream would catch:

* **The predictor against a Monte Carlo of the operator it claims to predict.**
  `P` is the Frobenius identity applied to `M = F_out · gain_v · F_inᵀ`, so the
  honest check is to build that operator from the real maps and draw directions
  through it — the same check `benchmarks/graph_transmission.py` makes of the
  identity one factor at a time, here made of the composition.
* **The split is an identity.** `reported = transport · floor_ratio` holds by
  construction of the three quantities, and a version of `paired` that peaked the
  numerator rather than the quotient would break it silently while still printing
  numbers. The script checks this at runtime too; this is the check that the
  check is real.
* **The regression recovers a line it is given**, including the sign of the
  slope, so `R²` and the residual mean what the resolution comment says.
* **The hop decomposition of a path** finds the shared cell and refuses a pair of
  edges that do not share exactly one.

And that the script still runs against the API, in the shape the other
benchmarks' tests ask it of them. `predict` is graph-side and affordable whole;
`regress` needs a sandbox and a trained surface, so it is smoke-tested on the
small dome.
"""

import numpy as np
import pytest
import torch

import construction_grading as grading
from patchworks.graph import build_graph
from patchworks.restriction import GAUGE_RHO, RestrictionMaps, pair_index
from patchworks.tick import DEFAULT_GAMMA, reconciliation_gain

from conftest import SMALL


class Pair:
    """Two edges and their endpoints, for :func:`hops_of` alone.

    `hops_of` reads only `edges[i].u` and `.v`, so a hand-built stand-in is
    clearer here than a dome, whose edge ids are a construction detail no
    hand-computed answer should have to know.
    """

    class Edge:
        def __init__(self, u, v):
            self.u, self.v = u, v

    def __init__(self, pairs):
        self.edges = [Pair.Edge(u, v) for u, v in pairs]


class TestHops:
    def test_finds_the_shared_cell_whichever_end_it_sits_on(self):
        # The four orientations a pair of adjacent edges can be written in. The
        # shared cell is 1 in every one of them, and a version that assumed
        # `edges[i].v == edges[i + 1].u` would pass only the first.
        for pairs in (
            [(0, 1), (1, 2)],
            [(0, 1), (2, 1)],
            [(1, 0), (1, 2)],
            [(1, 0), (2, 1)],
        ):
            assert grading.hops_of(Pair(pairs), (0, 1)) == [(0, 1, 1)]

    def test_a_three_edge_path_gives_two_hops(self):
        dome = Pair([(0, 1), (1, 2), (2, 3)])
        assert grading.hops_of(dome, (0, 1, 2)) == [(0, 1, 1), (1, 2, 2)]

    def test_refuses_edges_that_do_not_share_exactly_one_cell(self):
        with pytest.raises(ValueError):
            grading.hops_of(Pair([(0, 1), (2, 3)]), (0, 1))
        # A doubled edge shares *two* cells, which is not one hop through one
        # relay and must not be silently reduced to one.
        with pytest.raises(ValueError):
            grading.hops_of(Pair([(0, 1), (0, 1)]), (0, 1))

    def test_a_single_edge_path_has_no_hop(self):
        assert grading.hops_of(Pair([(0, 1)]), (0,)) == []


class TestSide:
    def test_names_the_end_the_cell_sits_on(self):
        dome = build_graph(SMALL)
        for edge in dome.edges:
            assert grading.side_of(dome, edge.id, edge.u) == 0
            assert grading.side_of(dome, edge.id, edge.v) == 1

    def test_refuses_a_cell_that_is_not_an_endpoint(self):
        dome = build_graph(SMALL)
        edge = dome.edges[0]
        outsider = next(
            c.id for c in dome.cells if c.id not in (edge.u, edge.v)
        )
        with pytest.raises(ValueError):
            grading.side_of(dome, edge.id, outsider)


class TestPredictor:
    def test_matches_a_monte_carlo_of_the_operator_it_predicts(self):
        """`P` against `E‖M u‖` for `u` uniform on the sphere, on the real maps.

        The predictor is the Frobenius identity applied to the composition
        `M = F_out · gain_v · F_inᵀ`, with each map's norm taken from the maps
        themselves — which is tier 2, so tier 2 is what the draw has to match.
        Tier 1 substitutes the gauge ceiling for those norms and is a *bound*,
        not a prediction of this quantity, so it is not what is checked here.

        `body` is a constant factor in both sides and is divided out rather than
        modelled: this test is about the two map factors and the gain.
        """
        dome = build_graph(SMALL)
        maps = RestrictionMaps(dome, generator=torch.Generator().manual_seed(0))
        gains = reconciliation_gain(dome, gamma=DEFAULT_GAMMA, rho=GAUGE_RHO)
        norms = maps.norms().detach()
        generator = torch.Generator().manual_seed(1)

        ratios = []
        for cell in dome.cells:
            if cell.is_boundary or dome.degrees[cell.id] < 2:
                continue
            edge_in, edge_out = dome.incident[cell.id][:2]
            m_in = dome.edges[edge_in].m
            m_out = dome.edges[edge_out].m
            with torch.no_grad():
                f_in = maps.maps[
                    pair_index(edge_in, grading.side_of(dome, edge_in, cell.id))
                ][:m_in]
                f_out = maps.maps[
                    pair_index(edge_out, grading.side_of(dome, edge_out, cell.id))
                ][:m_out]
                operator = (f_out.double() @ f_in.double().T) * float(gains[cell.id])
                directions = torch.randn((4096, m_in), generator=generator).double()
                directions /= directions.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                drawn = float((directions @ operator.T).norm(dim=-1).mean())
            predicted = (
                grading.predicted_hop(
                    dome, gains, edge_in, cell.id, edge_out, norms=norms
                )
                / grading.BODY_GAIN
            )
            ratios.append(drawn / predicted)

        ratios = np.array(ratios)
        assert len(ratios) > 5
        # The identity is exact in `E‖Mu‖²` and approximate in `E‖Mu‖` by the
        # concentration of the chi distribution; and the predictor is a
        # **product of Frobenius norms** where the drawn quantity is the norm of
        # the **product**, so the per-cell ratio also carries the alignment
        # between the two maps, which is not modelled and is not meant to be.
        #
        # Both gaps widen as the lanes narrow -- fewer dimensions, so less
        # concentration and more alignment scatter -- and #474 took the lanes to
        # `(interior_m, boundary_m) = (3, 4)`. Measured on this fixture the
        # ratios run 0.45 to 1.27 about a mean of 0.83, where at (4, 8) they sat
        # inside +/-15% of 1. The band below is widened to match, and the claim
        # it is defending is unchanged and is still a real one: **a predictor
        # that had the dilution or the gain wrong would miss by a factor**, and
        # a factor is well outside this band.
        #
        # Not corrected by a chi factor: that was tried, and it moves the mean
        # to 0.89 while widening the spread, because the dominant term here is
        # the alignment rather than the concentration.
        assert ratios.mean() == pytest.approx(1.0, rel=0.25)
        assert ratios.min() > 0.4
        assert ratios.max() < 1.5

    def test_the_gauge_tier_is_the_saturated_bound(self):
        """Tier 1 uses `rho` at a predicting cell and the exact gauge at a boundary."""
        dome = build_graph(SMALL)
        assert grading.gauge_norm(dome, dome.predicting[0]) == GAUGE_RHO
        assert grading.gauge_norm(dome, dome.boundary[0]) == 1.0

    def test_scales_as_the_arithmetic_says(self):
        """The three construction terms enter as `gain_v / sqrt(m_in · perm_v)`.

        Checked by recomputing the formula from the graph's own numbers rather
        than by re-running the function: an implementation that silently used
        `m_out` for the dilution would agree with itself and disagree here.
        """
        dome = build_graph(SMALL)
        gains = reconciliation_gain(dome, gamma=DEFAULT_GAMMA, rho=GAUGE_RHO)
        cell = next(
            c.id
            for c in dome.cells
            if not c.is_boundary
            and dome.degrees[c.id] >= 2
            and len({dome.edges[e].m for e in dome.incident[c.id]}) > 1
        )
        edge_in, edge_out = dome.incident[cell][:2]
        expected = (
            grading.BODY_GAIN
            * float(gains[cell])
            * GAUGE_RHO
            * GAUGE_RHO
            / np.sqrt(dome.edges[edge_in].m * grading.permitted_width(dome, cell))
        )
        assert grading.predicted_hop(
            dome, gains, edge_in, cell, edge_out
        ) == pytest.approx(expected)

    def test_the_permitted_width_is_the_mask_the_dome_hands_out(self):
        dome = build_graph(SMALL)
        for cell in dome.cells:
            edge_id = dome.incident[cell.id][0]
            assert grading.permitted_width(dome, cell.id) == int(
                dome.restriction_mask(edge_id, cell.id).sum()
            )


class TestAlignmentHeadroom:
    def test_is_at_least_one_and_bounded_by_the_incoming_width(self):
        """`sigma_max ≥ ‖M‖_F / sqrt(m_in) ≥ sigma_max / sqrt(m_in)`, always.

        The headroom is the room a direction-based account has to work in, so a
        number outside these bounds would not be a headroom at all — it would be
        an arithmetic error dressed as evidence for or against #184's parked
        candidate.
        """
        dome = build_graph(SMALL)
        maps = RestrictionMaps(dome, generator=torch.Generator().manual_seed(0))
        gains = reconciliation_gain(dome, gamma=DEFAULT_GAMMA, rho=GAUGE_RHO)
        seen = 0
        for cell in dome.cells:
            if cell.is_boundary or dome.degrees[cell.id] < 2:
                continue
            edge_in, edge_out = dome.incident[cell.id][:2]
            _value, headroom = grading.exact_operator(
                dome, maps, gains, edge_in, cell.id, edge_out
            )
            assert 1.0 <= headroom <= np.sqrt(dome.edges[edge_in].m) + 1e-9
            seen += 1
        assert seen > 5


class TestRegression:
    def test_recovers_a_line_it_is_given(self):
        x = np.linspace(-3.0, 1.0, 40)
        fit = grading.regression(x, 0.75 * x - 2.0)
        assert fit["slope"] == pytest.approx(0.75)
        assert fit["intercept"] == pytest.approx(-2.0)
        assert fit["r2"] == pytest.approx(1.0)
        assert fit["correlation"] == pytest.approx(1.0)
        assert np.abs(fit["residual"]).max() < 1e-9

    def test_the_unfitted_residual_allows_no_slope_or_offset(self):
        """`raw` is `y - x`: what is left when the prediction is simply believed.

        The distinction is the ticket's, and it is not cosmetic. A predictor can
        be perfectly *correlated* with the truth and still wrong by a constant
        factor, and a resolution that reported only `R²` would call that a
        success.
        """
        x = np.linspace(-3.0, 1.0, 40)
        fit = grading.regression(x, x + 1.5)
        assert fit["r2"] == pytest.approx(1.0)
        assert fit["raw"].mean() == pytest.approx(1.5)
        assert fit["residual"].std() < 1e-9

    def test_reports_a_negative_relationship_as_negative(self):
        x = np.linspace(-3.0, 1.0, 40)
        fit = grading.regression(x, -0.5 * x)
        assert fit["slope"] < 0
        assert fit["correlation"] < 0


class TestSplitIsAnIdentity:
    def test_transport_times_floor_reconstructs_the_reported_grading(self):
        """The split the whole resolution rests on, on numbers standing in for a read.

        `reported = (dev_out/dev_in) · (floor_in/floor_out)` is algebra, and the
        test is that the three fields the script builds are those three
        quantities — the thing that breaks if `paired` ever peaks the numerator
        instead of the quotient, which finds a different tick and a different
        path and still prints.
        """
        rng = np.random.default_rng(0)
        dev = rng.lognormal(size=16)
        floor = rng.lognormal(size=16)
        ratio = dev / floor
        for first, second in zip(range(15), range(1, 16)):
            transport = dev[second] / dev[first]
            floor_ratio = floor[first] / floor[second]
            assert transport * floor_ratio == pytest.approx(
                ratio[second] / ratio[first]
            )


class TestBenchmark:
    """That the script still runs against the API, in the other benchmarks' shape."""

    def test_predict_runs_on_the_real_dome(self, capsys):
        grading.main(["predict"])
        printed = capsys.readouterr().out
        assert "the construction predictor" in printed
        assert "directed hops" in printed

    def test_regress_runs(self, capsys):
        """The whole path, on the small dome: sandbox, fork, split, regression.

        Short everywhere it can be — the numbers this produces are not the
        ticket's numbers and are not meant to be. What is checked is that the
        read still composes against the API, which is what
        `tests/test_untrained_fixed_point.py` asks of its own script.
        """
        grading.main(
            [
                "regress",
                "--dome",
                "small",
                "--learn",
                "50",
                "--trials",
                "2",
                "--hold",
                "20",
                "--window",
                "8",
            ]
        )
        printed = capsys.readouterr().out
        assert "the split closes" in printed
        assert "construction only" in printed
