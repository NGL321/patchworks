"""`benchmarks/floor_split.py`: the subtraction, and what it may not claim (#351, for #339).

**Nothing here pins a reading of a run.** The three floors are readings of a
surface later tickets are expected to move. What is held is the shape of the
subtraction, because the whole rig is one difference of one variable and every
way of getting it wrong yields a plausible floor:

* **The two holds start from the same state.** The frozen hold moves no learned
  parameter, so restoring the tick state puts the second hold on the same
  surface. If it did not, `wander` would be measuring the first hold's leftovers.
* **The only difference between the two holds is whether the rule steps.** That
  is what the attribution to *the rule itself* rests on, and it is checkable
  without a world: a frozen segment leaves the maps untouched, a stepping one
  does not.
* **A wander at or below zero reads 0, not a negative ratio.** On an
  unconverged surface the rule still has real gradient to descend and the hold
  can *reduce* disagreement; that is the failure absent. #339's claim is about a
  converged edge, and the rig must not manufacture a crossing out of its
  opposite.
* **The lengths are `benchmarks/detectability.py`'s**, named through it rather
  than copied, so the two rigs' holds cannot drift apart.
"""

import math

import detectability
import floor_split
import pytest
import torch
import untrained_fixed_point as ufp

from conftest import SMALL
from patchworks.diagnostics import Diagnostics
from patchworks.learning import SparsityAnneal, TransportRule


class TestTheBar:
    """A self-ratio, and what it says at its edges."""

    def test_wander_at_or_below_zero_is_the_failure_absent(self):
        assert floor_split._ratio(-1.0, 2.0) == 0.0
        assert floor_split._ratio(0.0, 2.0) == 0.0

    def test_no_named_floor_left_is_stated_rather_than_raised(self):
        assert floor_split._ratio(1.0, 0.0) == math.inf

    def test_it_crosses_when_the_unnamed_floor_outweighs_the_named_ones(self):
        results = [{"counted": 1, "static": 1.0, "lag": 1.0, "wander": 3.0, "driven": 5.0}]
        found = floor_split.readings(results)
        assert found["wander_over_named_floors"] == pytest.approx(1.5)
        assert found["wander_share_of_floor"] == pytest.approx(0.6)

    def test_it_is_clear_when_the_named_floors_dominate(self):
        results = [{"counted": 1, "static": 4.0, "lag": 4.0, "wander": 1.0, "driven": 9.0}]
        assert floor_split.readings(results)["wander_over_named_floors"] < 1.0

    def test_a_read_with_no_segment_offers_nothing(self):
        assert floor_split.readings([{"seed": 0, "counted": 0}]) == {}

    def test_the_seeds_are_combined_by_median_not_by_sum(self):
        rows = [
            {"counted": 1, "static": 1.0, "lag": 1.0, "wander": w, "driven": 5.0}
            for w in (1.0, 2.0, 3.0)
        ]
        assert floor_split.readings(rows)["wander_over_named_floors"] == pytest.approx(1.0)


class TestTheSegments:
    """The one variable the two holds differ in."""

    def test_a_frozen_segment_leaves_the_maps_where_a_stepping_one_moves_them(self):
        """Both arms, on one small run, because the difference *is* the reading."""
        env, agent = floor_split.build(SMALL, "train", 0)
        try:
            observation, _info = env.reset(seed=0)
            agent.observe(observation)
            diagnostics = Diagnostics(agent.sheaf)
            transport = TransportRule(agent.sheaf, anneal=SparsityAnneal())
            applied = agent.command()

            before = agent.sheaf.maps.maps.detach().clone()
            state = ufp.snapshot(agent.sheaf)
            floor_split.segment(agent, diagnostics, 8, 4, observation, applied, None)
            assert torch.equal(agent.sheaf.maps.maps.detach(), before)

            ufp.restore(agent.sheaf, state)
            floor_split.segment(
                agent, diagnostics, 8, 4, observation, applied, transport
            )
            assert not torch.equal(agent.sheaf.maps.maps.detach(), before)
        finally:
            env.close()

    def test_the_hold_lengths_are_detectabilitys_own(self):
        assert floor_split.HOLD == detectability.HOLD
        assert floor_split.WINDOW == detectability.WINDOW


class TestTheScriptRuns:
    def test_read_runs_without_touching_the_tracker(self, capsys):
        assert (
            floor_split.main(
                [
                    "read",
                    "--dome",
                    "small",
                    "--learn",
                    "60",
                    "--hold",
                    "20",
                    "--window",
                    "10",
                    "--seeds",
                    "0",
                    "--no-file",
                ]
            )
            == 0
        )
        assert "floors, and the rule's own" in capsys.readouterr().out
