"""`benchmarks/detectability.py`: the estimator, the fork, and that the script runs (#214).

#214 is a measurement, not a decision, so nothing here pins its verdict — the
bottleneck ratio is a reading of a graph that #155 and #190 are expected to
change, and a test holding today's numbers would have to be deleted by whoever
changes it. That is `tests/test_untrained_fixed_point.py`'s reasoning and it
applies unchanged.

What is worth holding is that the instrument is **right**, because the whole
ticket is one arithmetic reduction and a wrong one would produce a plausible,
wrong verdict that nothing downstream would catch:

* **The widest path** against closed forms — series, parallel, a path that has to
  be found the long way round, and the edge the answer must name. It is the
  classical maximum-bottleneck path, so its value is checkable by hand on small
  graphs.
* **The fork is exact** — two unperturbed branches from one state differ by
  identically zero. Every number the read produces is a difference of two
  branches, so a fork that leaked would put a floor under all of them.
* **The read is linear in the injected amplitude**, which is the check that says
  the deviation being differenced is the perturbation and not the arithmetic's
  rounding. This is the property `double_precision` exists to obtain, and the one
  #146 and #183 both found missing in float32 readings.
"""

import numpy as np
import pytest
import torch

import detectability as det
from patchworks.agent import Agent
from patchworks.graph import build_graph
from patchworks.sandbox import PlanarPushSandbox

from conftest import SMALL


class Line:
    """A hand-built stand-in for `Dome`'s adjacency, for the estimator alone.

    `widest_path` reads three things off a dome — `incident`, `edges`, and each
    edge's `other` — so a graph the closed forms can be computed on by hand is
    cheaper and clearer here than a built dome, whose edge ids are a construction
    detail no closed form should have to know.
    """

    class Edge:
        def __init__(self, u, v):
            self.u, self.v = u, v

        def other(self, cell):
            return self.v if cell == self.u else self.u

    def __init__(self, pairs, size):
        self.edges = [Line.Edge(u, v) for u, v in pairs]
        incident: list[list[int]] = [[] for _ in range(size)]
        for i, edge in enumerate(self.edges):
            incident[edge.u].append(i)
            incident[edge.v].append(i)
        self.incident = tuple(tuple(e) for e in incident)


class TestWidestPath:
    """The max-min reduction, against values a reader can check by hand."""

    def test_series_takes_the_minimum(self):
        """A chain is worth its weakest edge, and the answer names that edge."""
        graph = Line([(0, 1), (1, 2), (2, 3)], 4)
        value, target, edge, path = det.widest_path(
            graph, np.array([0.9, 0.2, 0.7]), 0, (3,)
        )
        assert value == pytest.approx(0.2)
        assert target == 3
        assert edge == 1
        assert path == (0, 1, 2)

    def test_parallel_takes_the_better_route(self):
        """Two routes, and the max-min takes the one whose weakest edge is larger."""
        graph = Line([(0, 1), (1, 3), (0, 2), (2, 3)], 4)
        value, _target, edge, _path = det.widest_path(
            graph, np.array([0.9, 0.1, 0.5, 0.6]), 0, (3,)
        )
        assert value == pytest.approx(0.5)
        assert edge == 2

    def test_the_long_way_round_can_win(self):
        """The widest path is not the shortest one, and greedy-by-hops would miss it."""
        graph = Line([(0, 3), (0, 1), (1, 2), (2, 3)], 4)
        value, _target, _edge, path = det.widest_path(
            graph, np.array([0.01, 0.8, 0.8, 0.8]), 0, (3,)
        )
        assert value == pytest.approx(0.8)
        assert len(path) == 3

    def test_an_adjacent_target_is_bounded_by_its_own_edge(self):
        """The source's zero-edge path is bounded by nothing, so `inf` cannot leak out."""
        graph = Line([(0, 1)], 2)
        value, _target, _edge, _path = det.widest_path(graph, np.array([0.3]), 0, (1,))
        assert value == pytest.approx(0.3)

    def test_the_nearest_target_of_several_is_not_assumed(self):
        """Several targets: the best of them wins, not the first one reached."""
        graph = Line([(0, 1), (0, 2)], 3)
        value, target, _edge, _path = det.widest_path(
            graph, np.array([0.2, 0.6]), 0, (1, 2)
        )
        assert value == pytest.approx(0.6)
        assert target == 2

    def test_an_unreachable_target_is_zero(self):
        """A disconnected target carries no channel, and says so rather than raising."""
        graph = Line([(0, 1)], 3)
        value, target, edge, path = det.widest_path(graph, np.array([0.5]), 0, (2,))
        assert value == 0.0
        assert (target, edge, path) == (-1, -1, ())


class TestTheFork:
    """The paired counterfactual, on the small dome: exact, and linear."""

    @pytest.fixture(scope="class")
    def held(self):
        env = PlanarPushSandbox(split="train", image_size=16)
        agent = Agent(
            env, dome=build_graph(SMALL), generator=torch.Generator().manual_seed(0)
        )
        observation, _info = env.reset(seed=0)
        agent.observe(observation)
        det.double_precision(agent.sheaf)
        applied = np.zeros(env.action_space.shape, dtype=np.float64)
        det.hold_still(agent, observation, applied, 60)
        return agent, observation, applied

    def test_two_unperturbed_branches_are_identical(self, held):
        """Identically zero, not merely small: the fork carries no state between runs."""
        agent, observation, applied = held
        state = det.ufp.snapshot(agent.sheaf)
        one = det.branch(agent, state, observation, applied, 8, None)
        two = det.branch(agent, state, observation, applied, 8, None)
        assert float((one - two).abs().max()) == 0.0

    def test_the_cast_reaches_everything_a_tick_touches(self, held):
        """A float32 tensor left behind would quantise the difference being read."""
        agent, _observation, _applied = held
        assert agent.sheaf.stalks.dtype is torch.float64
        assert agent.sheaf.disagreement().dtype is torch.float64

    def test_the_reading_is_linear_in_the_amplitude(self, held):
        """The tell that the difference is the perturbation and not the rounding.

        Reported at `A₀ = 1`, the same number has to come back from an injection
        a hundred times larger. In float32 this fails on the real dome by five
        orders of magnitude, which is the whole reason `double_precision` exists.
        """
        agent, observation, applied = held
        state = det.ufp.snapshot(agent.sheaf)
        quiet = det.branch(agent, state, observation, applied, 16, None)
        source = det.rim(agent.dome)[0]
        deviation = det.unit(
            agent.dome.cells[source].stalk, torch.Generator().manual_seed(1)
        )
        reads = [
            det.ratios(
                agent, state, quiet, observation, applied, source, deviation, a0, 16
            )
            for a0 in (1e-2, 1.0)
        ]
        finite = np.isfinite(reads[0]) & np.isfinite(reads[1]) & (reads[1] > 0)
        assert finite.any()
        assert np.allclose(reads[0][finite], reads[1][finite], rtol=1e-6)


class TestBenchmark:
    """That the script still runs against the API, on the small dome."""

    def test_read_runs(self, capsys):
        det.main(
            [
                "read",
                "--dome",
                "small",
                "--trials",
                "2",
                "--learn",
                "50",
                "--hold",
                "20",
                "--window",
                "8",
            ]
        )
        printed = capsys.readouterr().out
        assert "rim-to-apex" in printed
        assert "apex-to-rim" in printed

    def test_linearity_runs(self, capsys):
        det.main(
            [
                "linearity",
                "--dome",
                "small",
                "--learn",
                "50",
                "--hold",
                "20",
                "--window",
                "8",
                "--amplitudes",
                "1.0",
                "10.0",
            ]
        )
        assert "bottleneck at A0=1" in capsys.readouterr().out
