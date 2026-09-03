"""`benchmarks/projection_firing.py`: the observable, and what the bar is made of (#351).

**Nothing here pins a reading of a run.** The firing rate against depth is a
reading of a surface later tickets are expected to move, and a test holding
today's numbers would have to be deleted by whoever moves it — which is
`tests/test_graph_transmission.py`'s reasoning and #150's before it.

What is worth holding is that the instrument is **right**, because #335's bar is
a ratio of two medians and every way of getting a ratio wrong produces a
plausible number:

* **The projection reports what it moved.** :meth:`CellOperators.project` now
  returns the mask, and a mask that were always true — or always false — would
  make the whole rig read a constant. Held against operators placed deliberately
  in and out of band.
* **The rule is not changed by being watched.** The recorder wraps the
  projection for the length of a read and unwraps it after;
  :class:`~patchworks.learning.PredictionRule` is documented as holding no state
  whatsoever, and an instrument that made that false would be changing the thing
  it measures.
* **The two clauses combine by taking the lower**, which is what makes the bar
  unable to say *holds* while one of them fails.
"""

import math

import projection_firing
import pytest
import torch

from conftest import SMALL
from patchworks.body import BodyShape, CellOperators
from patchworks.graph import build_graph

#: A body small enough to write a `K` out by hand and read the band off it.
SHAPE = BodyShape(n=8, k=4)


def operators(norms: list[float]) -> CellOperators:
    """`len(norms)` cells whose `K` is a multiple of the identity at each norm."""
    built = CellOperators(SHAPE, len(norms))
    with torch.no_grad():
        for cell, norm in enumerate(norms):
            built.K[cell] = torch.eye(SHAPE.k) * norm
    return built


class TestTheObservable:
    """`project` returns which cells it moved, and that is the rig's whole input."""

    def test_a_cell_above_the_band_is_reported_moved(self):
        built = operators([4.0])
        assert bool(built.project()[0]) is True

    def test_a_cell_inside_the_band_is_reported_untouched(self):
        built = operators([0.9])
        assert bool(built.project()[0]) is False

    def test_a_cell_below_the_band_is_reported_moved(self):
        built = operators([1e-6])
        assert bool(built.project()[0]) is True

    def test_the_mask_is_one_bool_per_cell(self):
        mask = operators([4.0, 0.9, 0.5]).project()
        assert mask.shape == (3,)
        assert mask.dtype is torch.bool

    def test_projection_is_idempotent_and_says_so_the_second_time(self):
        built = operators([4.0])
        assert bool(built.project()[0]) is True
        assert bool(built.project()[0]) is False


class TestTheRecorder:
    """Watching the projection must not leave anything behind on the rule."""

    def test_it_collects_one_mask_per_call(self):
        built = operators([4.0, 0.9])
        with projection_firing.recording(built) as fired:
            built.project()
            built.project()
        assert len(fired) == 2
        assert fired[0].tolist() == [True, False]

    def test_it_unwraps_afterwards(self):
        built = operators([4.0])
        with projection_firing.recording(built):
            pass
        assert "project" not in vars(built)
        assert built.project().tolist() == [True]

    def test_it_unwraps_even_when_the_read_raises(self):
        built = operators([4.0])
        with pytest.raises(RuntimeError):
            with projection_firing.recording(built):
                raise RuntimeError("the read fell over")
        assert "project" not in vars(built)


class TestTau:
    """`tau = -1/ln rho`, and what it says at the band's faces."""

    def test_it_is_05s_form(self):
        assert projection_firing.tau(0.5) == pytest.approx(-1.0 / math.log(0.5))

    def test_at_or_above_one_it_is_infinite_rather_than_clamped(self):
        assert projection_firing.tau(1.0) == math.inf
        assert projection_firing.tau(1.5) == math.inf

    def test_a_dead_operator_retains_nothing(self):
        assert projection_firing.tau(0.0) == 0.0


class TestTheBar:
    """The two clauses, and the minimum that combines them."""

    def test_a_zero_denominator_is_stated_rather_than_raised(self):
        assert projection_firing._ratio(0.5, 0.0) == math.inf
        assert projection_firing._ratio(0.0, 0.0) == 0.0
        assert projection_firing._ratio(0.0, 0.5) == 0.0

    def test_the_bar_is_the_lower_of_the_two_clauses(self):
        dome = build_graph(SMALL)
        levels = sorted({c.index.level for c in dome.cells if not c.is_boundary})
        rim, apex = levels[0], levels[-1]
        cells = {c.id: c for c in dome.cells}
        # The apex fires twice as often as the rim and holds a shorter tau: both
        # clauses hold, so the bar reads the smaller of the two ratios.
        firing = {
            cell: (1.0 if cells[cell].index.level == apex else 0.5)
            for cell in dome.predicting
        }
        radius = {
            cell: (0.5 if cells[cell].index.level == apex else 0.9)
            for cell in dome.predicting
        }
        result = [{"seed": 0, "counted": 1, "firing": firing, "radius": radius}]
        found = projection_firing.readings(result, SMALL)
        assert found["apex_firing_over_rim"] == pytest.approx(2.0)
        assert found["rim_tau_over_apex_tau"] > 1.0
        assert found["band_fights_retention"] == pytest.approx(
            min(found["apex_firing_over_rim"], found["rim_tau_over_apex_tau"])
        )
        assert rim != apex

    def test_a_failing_clause_pulls_the_bar_down_alone(self):
        """The apex fires more but holds the *longer* tau: the claim does not hold."""
        dome = build_graph(SMALL)
        cells = {c.id: c for c in dome.cells}
        levels = sorted({c.index.level for c in dome.cells if not c.is_boundary})
        apex = levels[-1]
        firing = {
            cell: (1.0 if cells[cell].index.level == apex else 0.5)
            for cell in dome.predicting
        }
        radius = {
            cell: (0.9 if cells[cell].index.level == apex else 0.5)
            for cell in dome.predicting
        }
        result = [{"seed": 0, "counted": 1, "firing": firing, "radius": radius}]
        found = projection_firing.readings(result, SMALL)
        assert found["apex_firing_over_rim"] > 1.0
        assert found["rim_tau_over_apex_tau"] < 1.0
        assert found["band_fights_retention"] < 1.0


class TestTheBurnIn:
    """A count off the graph, not a level anybody chose."""

    def test_it_is_one_apex_round_trip(self):
        import loop_length

        assert projection_firing.burn_in(SMALL) == max(
            loop_length.loops(build_graph(SMALL)).lengths.values()
        )


class TestTheScriptRuns:
    """A smoke test on the same footing as the other rigs', on the small dome."""

    def test_read_runs_without_touching_the_tracker(self, capsys):
        assert (
            projection_firing.main(
                ["read", "--dome", "small", "--ticks", "60", "--seeds", "0", "--no-file"]
            )
            == 0
        )
        assert "projection, fired per cell against depth" in capsys.readouterr().out
