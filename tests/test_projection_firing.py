"""`benchmarks/projection_firing.py`: what the reading is made of (#351, #433).

**Nothing here pins a reading of a run.** Retention against depth is a reading
of a surface later tickets are expected to move, and a test holding today's
numbers would have to be deleted by whoever moves it — which is
`tests/test_graph_transmission.py`'s reasoning and #150's before it.

What is worth holding is that the instrument is **right**, because the reading
is a ratio of two medians and every way of getting a ratio wrong produces a
plausible number.

**Two classes went with #433's forward normalisation**, and are recorded here
rather than deleted quietly. `TestTheObservable` held that the projection
reported what it moved; `TestTheRecorder` held that watching it left nothing on
the rule. There is no projection now and nothing to watch, so both pinned
behaviour that no longer exists — the statelessness they protected is preserved
where it now decides something, in
:meth:`~patchworks.body.CellOperators.used`'s choice of a stateless spectral
norm over a warm-started power iteration's per-cell buffer.

**The bar's combining rule is held here too, and is now held negatively:** the
rig must *not* report `band_fights_retention` from the surviving clause alone,
because taking the lower of two clauses is what made it unable to say *holds*
while one of them failed.
"""

import math

import projection_firing
import pytest

from conftest import SMALL
from patchworks.graph import build_graph


class TestTau:
    """`tau = -1/ln rho`, and what it says at the band's faces."""

    def test_it_is_05s_form(self):
        assert projection_firing.tau(0.5) == pytest.approx(-1.0 / math.log(0.5))

    def test_at_or_above_one_it_is_infinite_rather_than_clamped(self):
        assert projection_firing.tau(1.0) == math.inf
        assert projection_firing.tau(1.5) == math.inf

    def test_a_dead_operator_retains_nothing(self):
        assert projection_firing.tau(0.0) == 0.0


def _radius_by_level(apex_radius: float, rim_radius: float) -> list[dict]:
    """One synthetic seed: every apex cell at one radius, every other at another."""
    dome = build_graph(SMALL)
    cells = {c.id: c for c in dome.cells}
    apex = max(c.index.level for c in dome.cells if not c.is_boundary)
    radius = {
        cell: (apex_radius if cells[cell].index.level == apex else rim_radius)
        for cell in dome.predicting
    }
    return [{"seed": 0, "counted": 1, "radius": radius}]


class TestTheBar:
    """The surviving clause, and the two that must stay unreported."""

    def test_a_zero_denominator_is_stated_rather_than_raised(self):
        assert projection_firing._ratio(0.5, 0.0) == math.inf
        assert projection_firing._ratio(0.0, 0.0) == 0.0
        assert projection_firing._ratio(0.0, 0.5) == 0.0

    def test_a_short_apex_tau_reads_above_one(self):
        # The apex forgets faster than the rim, which is the shape #335 claimed.
        found = projection_firing.readings(_radius_by_level(0.5, 0.9), SMALL)
        assert found["rim_tau_over_apex_tau"] > 1.0

    def test_a_long_apex_tau_reads_below_one(self):
        found = projection_firing.readings(_radius_by_level(0.9, 0.5), SMALL)
        assert found["rim_tau_over_apex_tau"] < 1.0

    def test_the_firing_clause_and_the_combined_bar_are_not_reported(self):
        """#433: nothing fires, so a reported number would be the removal itself.

        `band_fights_retention` is the **lower** of two clauses. Supplying it
        from the surviving clause alone would change what the name means rather
        than report it, and would let #335's bar read *holds* on half a claim.
        The rig reports neither, and `cutoff_report` says so out loud.
        """
        found = projection_firing.readings(_radius_by_level(0.5, 0.9), SMALL)
        assert set(found) == {"rim_tau_over_apex_tau"}


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
        assert "lambda(K) per cell against depth" in capsys.readouterr().out
