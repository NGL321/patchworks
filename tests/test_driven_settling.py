"""`benchmarks/driven_settling.py`: what the two bars are made of (#351, for #324 and #329).

**Nothing here pins a reading of a run**, for `tests/test_graph_transmission.py`'s
reason. What is held is the arithmetic between the instrument and the bar, which
is where a plausible wrong number would come from:

* **#324's population is the draining edges**, and *no edge draining* must read
  maximally healthy rather than as a low number nobody looked at. That is the
  one branch capable of turning the failure's absence into a crossing.
* **Both rank columns are used and not averaged.** An edge has two maps
  belonging to two different cells, and `diagnostics.EdgeReading` is explicit
  that collapsing them hides the case the instrument exists for.
* **#329's ratio is `τ`'s spread over that cell's own `|loop(c)|`**, so a cell
  deep in the graph is held to a longer round trip than a rim cell. Reading
  every cell against one loop length would be indexing by level, which #181
  forbids and ADR-0026 restates.
* **A `τ` that is not finite is counted, not dropped.** A cell pinned at the
  band's upper face is #329's failure, and silently discarding it would make the
  rig quietest exactly where the problem is loudest.
"""

import math

import driven_settling
import numpy as np
import pytest

from conftest import SMALL
from patchworks.graph import build_graph


def result(opening, closing, ranks, taus=None, cells=None):
    return {
        "seed": 0,
        "counted": 1,
        "opening": np.asarray(opening, dtype=float),
        "closing": np.asarray(closing, dtype=float),
        "ranks": np.asarray(ranks, dtype=float),
        "transmitting": 2 * len(opening),
        "taus": taus or [],
        "cells": cells or [],
    }


class TestTheDrain:
    """#324: effective rank on the edges whose energy fell."""

    def test_it_reads_only_the_draining_edges(self):
        # Edge 0 drains and its maps are rank-1; edge 1 does not and its maps
        # are healthy. The healthy edge must not lift the reading.
        found = driven_settling.drain(
            result([1.0, 1.0], [0.5, 2.0], [[1.0, 1.0], [4.0, 4.0]]), edge_width=4
        )
        assert found["rank"] == pytest.approx(1.0)
        assert found["share"] == pytest.approx(0.5)

    def test_no_edge_draining_reads_maximally_healthy(self):
        found = driven_settling.drain(
            result([1.0, 1.0], [2.0, 2.0], [[1.0, 1.0], [1.0, 1.0]]), edge_width=4
        )
        assert found["rank"] == 4.0
        assert found["share"] == 0.0

    def test_both_columns_enter_the_median(self):
        """One end concentrating and the other not is the case that must show."""
        found = driven_settling.drain(
            result([1.0], [0.5], [[1.0, 3.0]]), edge_width=4
        )
        assert found["rank"] == pytest.approx(2.0)

    def test_a_whole_fleet_of_rank_one_maps_crosses_the_bar(self):
        found = driven_settling.drain(
            result([1.0, 1.0], [0.5, 0.5], [[1.0, 1.0], [1.0, 1.0]]), edge_width=4
        )
        assert found["rank"] < 2.0


class TestTheWander:
    """#329: the spread of `tau` within a window, over that cell's own loop."""

    def test_a_settled_cell_reads_zero(self):
        found = driven_settling.wander(
            result([1.0], [1.0], [[1.0, 1.0]], taus=[[5.0], [5.0], [5.0]], cells=[0]),
            loops={0: 10},
        )
        assert found["ratio"] == 0.0
        assert found["unsettled"] == 0.0

    def test_the_ratio_is_against_that_cells_own_loop(self):
        taus = [[0.0], [20.0], [0.0], [20.0]]
        shallow = driven_settling.wander(
            result([1.0], [1.0], [[1.0, 1.0]], taus=taus, cells=[0]), loops={0: 2}
        )
        deep = driven_settling.wander(
            result([1.0], [1.0], [[1.0, 1.0]], taus=taus, cells=[0]), loops={0: 14}
        )
        assert shallow["ratio"] > deep["ratio"]
        assert deep["ratio"] == pytest.approx(shallow["ratio"] * 2 / 14)

    def test_a_non_finite_tau_is_counted_rather_than_dropped(self):
        found = driven_settling.wander(
            result(
                [1.0], [1.0], [[1.0, 1.0]], taus=[[math.inf], [5.0]], cells=[0]
            ),
            loops={0: 10},
        )
        assert found["unsettled"] == 1.0

    def test_a_cell_with_no_loop_is_left_out_rather_than_divided_by_zero(self):
        found = driven_settling.wander(
            result([1.0], [1.0], [[1.0, 1.0]], taus=[[0.0], [20.0]], cells=[0]),
            loops={},
        )
        assert found["ratio"] == 0.0


class TestTau:
    def test_it_is_05s_form_and_says_inf_at_the_face(self):
        assert driven_settling.tau(0.5) == pytest.approx(-1.0 / math.log(0.5))
        assert driven_settling.tau(1.0) == math.inf
        assert driven_settling.tau(0.0) == 0.0


class TestTheWindow:
    def test_it_is_one_apex_round_trip(self):
        import loop_length

        assert driven_settling.window(SMALL) == max(
            loop_length.loops(build_graph(SMALL)).lengths.values()
        )


class TestTheScriptRuns:
    def test_read_runs_without_touching_the_tracker(self, capsys):
        assert (
            driven_settling.main(
                ["read", "--dome", "small", "--ticks", "60", "--seeds", "0", "--no-file"]
            )
            == 0
        )
        assert "drain and its settling" in capsys.readouterr().out
