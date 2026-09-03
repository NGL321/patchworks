"""`benchmarks/loop_length.py`: the ladder, and the split's price on it (#351, for #343).

Nothing here pins a reading of a run — there is no run. `|loop(c)|` is a
construction-time integer off the mask, so what the rig computes is either right
or wrong, and both arms are checkable in closed form:

* **The ladder is ADR-0026's**, recomputed. The ADR enumerated 414 cells, 682
  edges, a 263-cell sensorimotor rim and `|loop(c)| = 2 · level` from 2 at L1 to
  14 at the apex. That table is the one thing in the predicate the ADR says had
  never been checked, so a rig that reproduces it is worth a test even though
  the numbers move with `DomeSpec` — this is the *default* spec, and the ADR's
  reading is of the default spec.
* **`d(c, rim)` is exact, not a minimum with a spread.** The ADR settles this
  and it is not safe to assume: one level whose cells disagreed would mean the
  taper had opened a shortcut past a level.
* **Fusing rim cells into rim cells moves nothing**, which is #343's answer and
  is structural rather than a fact about this taper. Held on the small dome as
  well as the default one, because a claim that holds on one graph for a reason
  about *distance to a set* should hold on both, and a rig that agreed only on
  the graph it was written against would be reading a coincidence.

The last of these is the one the cutoff rests on, so it is held twice: once as
the fleet reading `loop_split_cost == 0`, and once cell by cell.
"""

import loop_length
import pytest

from conftest import SMALL
from patchworks.graph import DEFAULT_SPEC, CellKind, build_graph

#: ADR-0026's enumerated ladder on `DEFAULT_SPEC`, level to `|loop(c)|`.
LADDER = {1: 2, 2: 4, 3: 6, 4: 8, 5: 10, 6: 12, 7: 14}

#: Cells per level, same table. Held alongside the lengths because a rig that
#: swept the wrong cell set could reproduce every length and still be wrong
#: about which cells have them.
POPULATION = {1: 70, 2: 20, 3: 16, 4: 14, 5: 12, 6: 10, 7: 8}


class TestTheLadder:
    """ADR-0026's table, recomputed from the mask rather than quoted."""

    def test_the_default_dome_is_the_one_the_adr_enumerated(self):
        dome = build_graph(DEFAULT_SPEC)
        assert (len(dome.cells), len(dome.edges)) == (414, 682)
        assert len(loop_length.rim_of(dome)) == 263

    def test_the_ladder_is_two_times_the_level(self):
        dome = build_graph(DEFAULT_SPEC)
        rows = loop_length.by_level(dome, loop_length.loops(dome))
        assert {level: count for level, (count, _) in rows.items()} == POPULATION
        assert {level: sorted(v) for level, (_, v) in rows.items()} == {
            level: [length] for level, length in LADDER.items()
        }

    def test_every_predicting_cell_is_reached_from_the_rim(self):
        dome = build_graph(DEFAULT_SPEC)
        found = loop_length.loops(dome)
        assert found.unreachable == ()
        assert len(found.lengths) == len(dome.predicting) == 150

    def test_the_drive_cell_is_not_part_of_the_rim(self):
        """ADR-0026 excludes it explicitly, and including it would hand the apex a 2."""
        dome = build_graph(DEFAULT_SPEC)
        rim = set(loop_length.rim_of(dome))
        drive = {c.id for c in dome.cells if c.kind is CellKind.DRIVE}
        assert drive and not (drive & rim)
        assert max(loop_length.loops(dome).lengths.values()) == 14


class TestTheSplit:
    """#343: what ADR-0016's written-or-read ban costs ADR-0026's divisor."""

    @pytest.mark.parametrize("spec", [DEFAULT_SPEC, SMALL], ids=["default", "small"])
    def test_fusing_the_actuator_moves_no_cells_loop(self, spec):
        dome = build_graph(spec)
        fusion = loop_length.motor_fusion(dome)
        assert fusion, "the fusion must actually fuse something to be a counterfactual"
        kept = loop_length.loops(dome)
        fused = loop_length.loops(dome, fusion)
        shared = set(kept.lengths) & set(fused.lengths)
        assert shared
        assert all(kept.lengths[c] == fused.lengths[c] for c in shared)

    @pytest.mark.parametrize("spec", [DEFAULT_SPEC, SMALL], ids=["default", "small"])
    def test_the_reading_the_cutoff_reads_is_zero(self, spec):
        found = loop_length.readings(spec)
        assert found["loop_split_cost"] == 0.0
        assert found["loop_apex"] == found["loop_apex_fused"]

    def test_the_fused_arm_drops_exactly_the_actuator_from_the_rim(self):
        """The counterfactual must be one cell smaller, not a differently-built graph."""
        dome = build_graph(DEFAULT_SPEC)
        fusion = loop_length.motor_fusion(dome)
        assert len(loop_length.rim_of(dome, fusion)) == 263 - len(fusion)

    def test_fusion_never_creates_a_self_loop_edge(self):
        """A fused endpoint pair collapses; the walk must not spend a tick on it."""
        dome = build_graph(DEFAULT_SPEC)
        neighbours = loop_length.adjacency(dome, loop_length.motor_fusion(dome))
        assert all(cell not in ends for cell, ends in neighbours.items())


class TestTheWorldLoop:
    """#368: the loop ADR-0026 argues for, out through the world and back elsewhere."""

    @pytest.mark.parametrize("spec", [DEFAULT_SPEC, SMALL], ids=["default", "small"])
    def test_the_excess_is_at_least_the_world_tick_at_every_cell(self, spec):
        """`d(c, rim) <= min(d(c, a), d(c, p))`, because both are in the rim.

        The floor is closed-form and holds on any dome carrying an actuator and
        a proprioceptor, which is why it is checked on both and cell by cell
        rather than on the fleet maximum alone.
        """
        dome = build_graph(spec)
        gaps = loop_length.excesses(dome)
        assert gaps
        assert min(gaps.values()) >= loop_length.WORLD_TICK

    @pytest.mark.parametrize("spec", [DEFAULT_SPEC, SMALL], ids=["default", "small"])
    def test_the_world_tick_shifts_every_cell_by_itself_and_nothing_else(self, spec):
        """`w` enters once, additively. A `w` that moved the `min` would be a bug."""
        dome = build_graph(spec)
        at_zero = loop_length.excesses(dome, world_tick=0)
        assert loop_length.excesses(dome) == {
            c: gap + loop_length.WORLD_TICK for c, gap in at_zero.items()
        }

    def test_the_ban_is_enforced_the_answer_returns_at_another_cell(self):
        """`a != p`. One actuator and `joints` proprioceptors, so the pairs are real."""
        dome = build_graph(DEFAULT_SPEC)
        found = loop_length.world_loops(dome)
        assert len(found.actuators) == 1
        assert len(found.proprioceptors) == DEFAULT_SPEC.joints
        assert not set(found.actuators) & set(found.proprioceptors)
        assert found.unreachable == ()
        assert len(found.lengths) == len(dome.predicting) == 150

    def test_the_reading_the_cutoff_reads(self):
        """#368's bar is `world_loop_excess >= 1`; the default dome reads 13."""
        found = loop_length.readings(DEFAULT_SPEC)
        assert found["world_loop_excess"] == 13.0
        assert found["world_loop_excess_w0"] == 12.0

    def test_the_other_two_readings_do_not_move(self):
        """#369's instruction in terms: `ladder`'s and `split`'s readings stay put."""
        found = loop_length.readings(DEFAULT_SPEC)
        assert found["loop_split_cost"] == 0.0
        assert found["loop_apex"] == found["loop_apex_fused"] == 14.0


class TestTheScriptRuns:
    """A smoke test on the same footing as the other rigs' (`--no-file` throughout)."""

    def test_ladder_runs(self, capsys):
        assert loop_length.main(["ladder"]) == 0
        assert "|loop(c)|" in capsys.readouterr().out

    def test_split_runs_without_touching_the_tracker(self, capsys):
        assert loop_length.main(["split", "--no-file"]) == 0
        assert "written-or-read split" in capsys.readouterr().out

    def test_world_runs_without_touching_the_tracker(self, capsys):
        assert loop_length.main(["world", "--no-file"]) == 0
        assert "the loop it argues for" in capsys.readouterr().out
