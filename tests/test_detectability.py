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

ADR-0026's reduction (#379) is held to the same standard and for the same reason,
one predicate over:

* **`|loop(c)|` is enumerated, not quoted.** ADR-0026 prints a ladder for
  `DEFAULT_SPEC` and says in terms that no session should quote `14` at a dome it
  has not checked. The enumeration is re-derived here against that published
  ladder, which is the one number in the predicate that had never been checked
  when #242 wrote it.
* **`τ̂` is peak-to-`1/e`**, against decays whose e-fold time is known by
  construction, plus the two readings that are not measurements: a censored
  window and a deviation that never arrived.
* **The conducting path** is the same max-min against closed forms, valued at
  cells instead of edges.
* **`readings()` publishes the two quantities under their own names**, which is
  the regression #379 exists to prevent: they were one key, they are eight orders
  apart, and nothing downstream could have caught the swap.
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

    def test_several_sources_search_from_all_of_them(self):
        """A collective perturbation has no single source, and the `max` is over
        rim-to-apex paths rather than over paths from one named cell. Seeding
        every perturbed cell at `inf` is the same object as a virtual
        super-source, and the wider seed can only find a wider path."""
        graph = Line([(0, 1), (2, 1)], 3)
        alone, _target, _edge, _path = det.widest_path(
            graph, np.array([0.2, 0.9]), 0, (1,)
        )
        together, _target, _edge, _path = det.widest_path(
            graph, np.array([0.2, 0.9]), (0, 2), (1,)
        )
        assert alone == pytest.approx(0.2)
        assert together == pytest.approx(0.9)

    def test_an_unreachable_target_is_zero(self):
        """A disconnected target carries no channel, and says so rather than raising."""
        graph = Line([(0, 1)], 3)
        value, target, edge, path = det.widest_path(graph, np.array([0.5]), 0, (2,))
        assert value == 0.0
        assert (target, edge, path) == (-1, -1, ())


class TestTheLoopEnumeration:
    """`|loop(c)|` off the mask, against ADR-0026's published ladder."""

    def test_the_default_dome_reproduces_the_published_ladder(self):
        """ADR-0026's own table, re-derived rather than quoted.

        The ADR enumerated this on `DEFAULT_SPEC` and made re-derivation the
        rule: *"A changed `DomeSpec` changes these numbers, and they are
        recomputed rather than quoted."* This is that recomputation, and it is
        what would fail first if the taper's wiring moved under a bar that still
        said 14.
        """
        from patchworks.graph import DEFAULT_SPEC

        dome = build_graph(DEFAULT_SPEC)
        loops = det.loop_lengths(dome)
        ladder: dict[int, set[int]] = {}
        for cell, length in loops.items():
            ladder.setdefault(dome.cells[cell].index.level, set()).add(length)
        assert ladder == {
            1: {2},
            2: {4},
            3: {6},
            4: {8},
            5: {10},
            6: {12},
            7: {14},
        }

    def test_every_predicting_cell_has_a_closing_loop(self):
        """No cell is absent, so the ladder above is the whole graph's."""
        from patchworks.graph import DEFAULT_SPEC

        dome = build_graph(DEFAULT_SPEC)
        assert set(det.loop_lengths(dome)) == set(dome.predicting)

    def test_the_distance_is_exact_and_not_a_spread(self):
        """ADR-0026's second enumeration finding: no cell short-cuts its level."""
        dome = build_graph(SMALL)
        loops = det.loop_lengths(dome)
        for cell, length in loops.items():
            assert length == 2 * dome.cells[cell].index.level

    def test_the_boundary_cells_are_not_in_it(self):
        """`|loop|` is defined at predicting cells; the rim is where it starts."""
        dome = build_graph(SMALL)
        assert not set(det.loop_lengths(dome)) & set(det.rim(dome))


class TestTauHat:
    """Peak-to-`1/e` in ticks, and the two readings that are not measurements."""

    def test_an_exponential_reads_its_own_e_fold_time(self):
        """`exp(-t/8)` crosses `1/e` at `t = 8`, and the sample there is the read."""
        ticks = np.arange(64.0)
        decay = np.exp(-ticks / 8.0)[:, None]
        tau, censored, peak_at = det.tau_hat(decay)
        assert peak_at[0] == 0
        assert tau[0] == 8.0
        assert not censored[0]

    def test_the_peak_is_found_before_the_decay_is_read(self):
        """`τ̂` is measured from the peak, not from the fork."""
        rise = np.concatenate([np.zeros(5), np.exp(-np.arange(59.0) / 4.0)])[:, None]
        tau, _censored, peak_at = det.tau_hat(rise)
        assert peak_at[0] == 5
        assert tau[0] == 4.0

    def test_a_window_that_ends_first_is_censored_and_flagged(self):
        """A lower bound, and it says so: it can cost a pass, never manufacture one."""
        flat = np.ones((16, 1))
        tau, censored, peak_at = det.tau_hat(flat)
        assert censored[0]
        assert tau[0] == 15.0 - peak_at[0]

    def test_a_deviation_that_never_arrived_is_zero_and_not_censored(self):
        """`0 / |loop|` is the falsification reading, not a short decay."""
        tau, censored, _peak = det.tau_hat(np.zeros((16, 1)))
        assert tau[0] == 0.0
        assert not censored[0]


class TestTheRuntimePrecisionGate:
    """#224's gate: is the `1/e` crossing above float32's granularity?"""

    def test_a_crossing_above_the_floor_is_readable(self):
        deviation = np.exp(-np.arange(32.0) / 4.0)[:, None]
        state = np.ones((32, 1))
        tau, _c, peak_at = det.tau_hat(deviation)
        assert det.readable(deviation, state, tau, peak_at)[0]

    def test_a_crossing_below_the_floor_is_not(self):
        """The same decay, scaled under `eps · ‖state‖`: real in float64, absent in float32."""
        deviation = det.EPS_F32 * 1e-3 * np.exp(-np.arange(32.0) / 4.0)[:, None]
        state = np.ones((32, 1))
        tau, _c, peak_at = det.tau_hat(deviation)
        assert not det.readable(deviation, state, tau, peak_at)[0]

    def test_the_floor_scales_with_the_state_it_is_read_at(self):
        """Granularity is `eps` *at the magnitude the state sits at*, not absolute."""
        deviation = np.exp(-np.arange(32.0) / 4.0)[:, None]
        tau, _c, peak_at = det.tau_hat(deviation)
        assert det.readable(deviation, np.ones((32, 1)), tau, peak_at)[0]
        loud = np.full((32, 1), 1.0 / det.EPS_F32 * 10.0)
        assert not det.readable(deviation, loud, tau, peak_at)[0]


class TestTheConductingPath:
    """The same max-min reduction, valued at cells instead of edges."""

    def test_series_takes_the_minimum_cell(self):
        line = Line([(0, 1), (1, 2), (2, 3)], 4)
        value, target, cell, walk = det.conducting_path(
            line, {1: 0.8, 2: 0.3, 3: 0.9}, 0, (3,)
        )
        assert (value, target, cell) == (0.3, 3, 2)
        assert walk == (0, 1, 2, 3)

    def test_a_cell_with_no_value_is_unbounded(self):
        """Boundary cells hold no private features, so they bound no path."""
        line = Line([(0, 1), (1, 2)], 3)
        value, _target, cell, _walk = det.conducting_path(line, {2: 0.4}, 0, (2,))
        assert (value, cell) == (0.4, 2)

    def test_the_better_route_is_taken(self):
        line = Line([(0, 1), (1, 4), (0, 2), (2, 3), (3, 4)], 5)
        value, _target, cell, walk = det.conducting_path(
            line, {1: 0.1, 2: 0.7, 3: 0.6, 4: 0.9}, 0, (4,)
        )
        assert (value, cell) == (0.6, 3)
        assert walk == (0, 2, 3, 4)

    def test_the_source_bounds_the_path_when_it_is_a_predicting_cell(self):
        """`min` over the cells of `P` includes the ends, which is outbound's case."""
        line = Line([(0, 1), (1, 2)], 3)
        value, _target, cell, _walk = det.conducting_path(
            line, {0: 0.2, 1: 0.9, 2: 0.9}, 0, (2,)
        )
        assert (value, cell) == (0.2, 0)

    def test_an_unreachable_target_is_zero(self):
        line = Line([(0, 1)], 3)
        assert det.conducting_path(line, {1: 0.5}, 0, (2,)) == (0.0, -1, -1, ())

    def test_a_cell_whose_loop_never_closes_bounds_the_path_at_zero(self):
        """Absent from `loop_lengths` means `0.0`, the falsification, not unbounded."""
        line = Line([(0, 1), (1, 2)], 3)
        value, _target, cell, _walk = det.conducting_path(
            line, {1: 0.0, 2: 0.9}, 0, (2,)
        )
        assert (value, cell) == (0.0, 1)


class TestTheTwoQuantitiesAreNotOneKey:
    """#379's regression: what `conduction_ratio` names, and what it does not.

    The defect was silent — a plausible, wrong number under the right name — so
    the check has to be that the key tracks the right attribute, which nothing
    downstream of `readings` could ever have told apart.
    """

    @staticmethod
    def outcome(bottleneck: float, conduction: float) -> det.Trial:
        empty = np.zeros(1)
        return det.Trial(
            source=(0,),
            kind="patch",
            bottleneck=bottleneck,
            target=0,
            edge=0,
            path=(),
            ratios=empty,
            peak_at=empty,
            horizons=(),
            conduction=conduction,
            cell=0,
            walk=(),
            tau=empty,
            censored=empty.astype(bool),
            resolved=empty.astype(bool),
            private=empty.astype(bool),
            conduction_horizons=(),
        )

    @pytest.fixture
    def results(self):
        return {
            "rim-to-apex": [self.outcome(8.7e-10, 0.2), self.outcome(8.7e-10, 0.4)],
            "apex-to-rim": [self.outcome(1.3e-8, 0.8), self.outcome(1.3e-8, 0.9)],
        }

    def test_conduction_ratio_is_adr_0026s_quantity(self, results):
        found = det.readings(results)
        assert found["conduction_ratio_rim_to_apex"] == pytest.approx(0.3)
        assert found["conduction_ratio_apex_to_rim"] == pytest.approx(0.85)

    def test_the_bottleneck_is_still_published_under_its_own_name(self, results):
        found = det.readings(results)
        assert found["bottleneck_ratio_rim_to_apex"] == pytest.approx(8.7e-10)
        assert found["bottleneck_ratio_apex_to_rim"] == pytest.approx(1.3e-8)

    def test_the_bare_key_is_the_lower_direction(self, results):
        found = det.readings(results)
        assert found["conduction_ratio"] == pytest.approx(0.3)
        assert found["bottleneck_ratio"] == pytest.approx(8.7e-10)

    def test_inbound_and_outbound_name_the_two_directions(self, results):
        """#341 writes `outbound conduction ratio` and had no key to land on."""
        found = det.readings(results)
        assert found["inbound_conduction_ratio"] == found["conduction_ratio_rim_to_apex"]
        assert found["outbound_conduction_ratio"] == found["conduction_ratio_apex_to_rim"]

    def test_the_two_quantities_do_not_share_a_key(self, results):
        """The defect itself: one name, two quantities, eight orders apart."""
        found = det.readings(results)
        assert found["conduction_ratio"] != found["bottleneck_ratio"]

    def test_an_empty_read_offers_nothing(self):
        assert det.readings({"rim-to-apex": []}) == {}


class TestTheStructuralZero:
    """A cell with no private dimension reads `0` by construction (#385)."""

    def test_every_l1_cell_of_the_default_dome_holds_no_private_features(self):
        """So the inbound predicate is pinned at `0` before any retention is read.

        Not a decision here — ADR-0026 is settled and this is its literal
        reading — but it is the fact `report` prints and #385 owns, and a test is
        what stops it being rediscovered a third time.
        """
        from patchworks.graph import DEFAULT_SPEC

        dome = build_graph(DEFAULT_SPEC)
        dimensions = dome.private_dimensions
        levels = [c.index.level for c in dome.cells if not c.is_boundary]
        assert [
            int(dimensions[row]) for row, level in enumerate(levels) if level == 1
        ] == [0] * 70

    def test_the_rim_touches_nothing_but_l1(self):
        """Which is what makes the pin structural: every path starts through one."""
        from patchworks.graph import DEFAULT_SPEC

        dome = build_graph(DEFAULT_SPEC)
        touched = {
            dome.cells[dome.edges[e].other(cell)].index.level
            for cell in det.rim(dome)
            for e in dome.incident[cell]
        }
        assert touched == {1}

    def test_no_path_between_the_rim_and_the_apex_avoids_a_zero_private_cell(self):
        """So **both** directions are pinned, not only the inbound one.

        The reduction is a `min` over the cells of a path. Every predicting cell
        adjacent to the rim is zero-private, and no apex cell is adjacent to the
        rim, so any walk between the two ends crosses one of them — whichever way
        it is walked. That makes the pin independent of retention, training,
        seed, stimulus and horizon, which is the claim #385 rests on and the one
        a reader would otherwise have to take on trust.
        """
        from patchworks.graph import DEFAULT_SPEC

        dome = build_graph(DEFAULT_SPEC)
        dimensions = dome.private_dimensions
        rows = {cell: row for row, cell in enumerate(dome.predicting)}
        zero = {c for c in dome.predicting if dimensions[rows[c]] == 0}
        adjacent = {
            dome.edges[e].other(cell)
            for cell in det.rim(dome)
            for e in dome.incident[cell]
        }
        assert adjacent and adjacent <= zero
        assert not adjacent & set(det.apex(dome))


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
        one, _ = det.branch(agent, state, observation, applied, 8, None)
        two, _ = det.branch(agent, state, observation, applied, 8, None)
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
        quiet, _ = det.branch(agent, state, observation, applied, 16, None)
        source = det.rim(agent.dome)[0]
        nudge = (
            (
                source,
                det.unit(
                    agent.dome.cells[source].stalk, torch.Generator().manual_seed(1)
                ),
            ),
        )
        reads = [
            det.ratios(agent, state, quiet, observation, applied, nudge, a0, 16)[0]
            for a0 in (1e-2, 1.0)
        ]
        finite = np.isfinite(reads[0]) & np.isfinite(reads[1]) & (reads[1] > 0)
        assert finite.any()
        assert np.allclose(reads[0][finite], reads[1][finite], rtol=1e-6)

    def test_a_sustained_source_is_held_at_a_constant_offset(self, held):
        """What *sustained* means, checked on the stalk rather than inferred.

        The rim sources are written boundary cells, so #214's stimulus is erased
        on the tick after it lands; the sustained branch clamps them to the quiet
        branch's own value plus the deviation. So the difference between the two
        branches at the source is exactly the deviation, at **every** tick — and
        under the impulse it is the deviation at the first tick and zero after.
        """
        agent, observation, applied = held
        state = det.ufp.snapshot(agent.sheaf)
        source = det.rim(agent.dome)[0]
        deviation = det.unit(
            agent.dome.cells[source].stalk, torch.Generator().manual_seed(1)
        )
        nudge = ((source, deviation),)
        _quiet, quiet_stalks = det.branch(
            agent, state, observation, applied, 8, None, record=(source,)
        )
        _moved, sustained = det.branch(
            agent,
            state,
            observation,
            applied,
            8,
            nudge,
            sustained=quiet_stalks,
            record=(source,),
        )
        _moved, impulse = det.branch(
            agent, state, observation, applied, 8, nudge, record=(source,)
        )
        offset = sustained[source] - quiet_stalks[source]
        assert torch.allclose(offset, deviation.expand_as(offset), atol=1e-12)
        erased = impulse[source] - quiet_stalks[source]
        assert float(erased.abs().max()) == pytest.approx(0.0, abs=1e-12)


class TestTheCollective:
    """#232's injection convention: unit-norm over the stratum, and coherent."""

    def stratum(self):
        dome = build_graph(SMALL)
        return dome, tuple(det.rim_strata(dome)[0])

    def test_the_whole_injection_is_unit_norm(self):
        """`A₀ = 1` over the collective, so the corners compare like with like.

        Without this clause the collective corner would inject `√K` times more
        than #214 did and beat the baseline by arithmetic rather than by
        coherence — which is the thing the read is asking about.
        """
        dome, cells = self.stratum()
        assert len(cells) > 1
        for coherent in (True, False):
            nudge = det.injection(
                dome, cells, torch.Generator().manual_seed(0), coherent
            )
            assert {c for c, _ in nudge} == set(cells)
            total = torch.cat([d for _, d in nudge]).norm()
            assert float(total) == pytest.approx(1.0, rel=1e-12)

    def test_coherent_is_one_direction_and_incoherent_is_many(self):
        """The contrast row differs from the corner in this and nothing else."""
        dome, cells = self.stratum()
        shared = det.injection(dome, cells, torch.Generator().manual_seed(0), True)
        apart = det.injection(dome, cells, torch.Generator().manual_seed(0), False)
        first = shared[0][1]
        assert all(torch.allclose(d, first) for _, d in shared)
        assert not torch.allclose(apart[0][1], apart[1][1])

    def test_a_ragged_collective_is_refused_rather_than_averaged(self):
        """Coherence is *the same vector in each cell's own coordinates*, so it
        has no meaning across stalks of different width, and the strata are what
        keep the widths equal. A mixed set is a bug, not a reading."""
        dome = build_graph(SMALL)
        strata = det.rim_strata(dome)
        mixed = (strata[0][0], strata[1][0])
        with pytest.raises(ValueError):
            det.injection(dome, mixed, torch.Generator().manual_seed(0))


class TestBenchmark:
    """That the script still runs against the API, on the small dome."""

    def test_read_runs(self, capsys):
        det.main(
            [
                "read",
                "--no-file",
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

    def test_corners_runs(self, capsys):
        det.main(
            [
                "corners",
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
        for corner in ("impulse x single-source", "sustained x collective"):
            assert corner in printed
        assert "the four corners, inbound" in printed

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
