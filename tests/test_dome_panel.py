"""The dome panel: the bands, and every mark drawn in them (#93, #94).

Six things are held down here, and the first three are the panel's whole
claim on a viewer.

* **Depth reads as height.** One band per level, L0 at the bottom and the apex
  at the top, each at its own lattice shape, every position taken from the
  construction index and nothing else.
* **Colour is prediction error, normalised per cell.** Two cells at wildly
  different raw scales light the same when each is the same distance above its
  *own* baseline -- and a chronically wrong cell renders calm, which is the
  consequence `10-the-demo-surface.md` accepts rather than fixes. The raw map
  is behind the debug flag, and shows what the normalised one hides.
* **The panel says it is warming up** until each cell has a baseline, on
  screen, in the frame.
* **The trail decays at each cell's own measured persistence** -- over ticks,
  not frames -- and it is **not** driven by `‖Δ private‖`: asserted twice, once
  on the frames and once on the module's source, because the panel's ability to
  contradict the thesis depends on it.
* **Closing it changes nothing but the view.** Bit for bit over a run with the
  world in it, and structurally: no agent, sheaf or world is reachable from a
  panel at all.
* **The persistence is `05-timescales.md`'s estimate**, not a second definition
  of timescale: it is `bias_selection.measure`'s, over the run's own body and
  biases, and taking it changes no parameter.

#94 adds the marks a **boundary cell** gets, and they are one claim: **every
mark draws a quantity that is honestly earned.** A boundary cell runs no body
and makes no prediction, so it has no prediction error and never appears on
that map -- what it has is an edge, and edge disagreement is drawn on the same
colormap. L0 draws the world instead: the agent's own render, tiled into the
patch lattice. The actuator draws decomposed, which renders `04`'s
route-selection falsification test rather than describing it, and the edge
overlay is thresholded from the tick's own scale with no magnitude in the rule
at all.
"""

import ast
import dataclasses
import inspect
import pathlib
import types

import numpy as np
import pytest
import torch

import patchworks
from patchworks import bias_selection
from patchworks.agent import Agent
from patchworks.graph import DomeSpec, build_graph
from patchworks.sandbox import PlanarPushSandbox, snapshot
from patchworks.sandbox.env import Task
from patchworks.sandbox.state import Snapshot
from patchworks.surface import (
    BandLayout,
    DomePanel,
    Recorder,
    Renderer,
    TickRecord,
    Trace,
    colormap,
    measured_persistence,
)
from patchworks.surface import dome_panel as dome_panel_module
from patchworks.tick import Sheaf

TICKS = 24


@pytest.fixture(scope="module")
def dome():
    return build_graph()


@pytest.fixture
def env():
    world = PlanarPushSandbox(split="any")
    yield world
    world.close()


def started(env, dome, seed=0):
    """An agent on an arranged world, one external write in and no tick yet."""
    agent = Agent(env, dome=dome, generator=torch.Generator().manual_seed(seed))
    observation, _info = env.reset(seed=seed)
    agent.observe(observation)
    return agent


STATE = Snapshot(
    physics=np.zeros(1),
    task=Task(
        puck_xy=np.zeros((2, 2)), puck_theta=np.zeros(2), goal_puck=0, goal_zone=0
    ),
    rng={},
)
"""A record's state, which the panel never reads: it draws the two arrays."""


def record(tick, error, private=None, disagreement=None, actuator=None):
    """A tick record carrying `error`, and whatever else the test wants.

    `disagreement` and `actuator` default to **not captured** rather than to
    zeros, which is what a record built by hand for a test about the dome's
    channel honestly holds: the panel draws no boundary mark from one.
    """
    error = np.asarray(error, dtype=float)
    return TickRecord(
        tick=tick,
        state=STATE,
        prediction_error=error,
        private_delta=(
            np.zeros_like(error) if private is None else np.asarray(private, dtype=float)
        ),
        disagreement=(
            np.zeros(0) if disagreement is None else np.asarray(disagreement, dtype=float)
        ),
        actuator=(
            np.zeros(0) if actuator is None else np.asarray(actuator, dtype=float)
        ),
    )


SMALL_SPEC = DomeSpec(
    patch_grid=4, vision_sides=(2,), somatomotor_sizes=(3,), core_sizes=(4, 3)
)
"""The colour and trail tests are about arithmetic per cell, not about the
taper, so they run on a dome small enough to name every cell in: 14 predicting
cells over four levels."""


@pytest.fixture(scope="module")
def small():
    return build_graph(SMALL_SPEC)


@pytest.fixture(scope="module")
def small_sheaf(small):
    """A sheaf on the small dome: a body, and one bias vector per cell."""
    return Sheaf(small, generator=torch.Generator().manual_seed(0))


def mark(panel, frame, cell_id):
    """The colour of one cell's mark."""
    top, left, _size = panel.rect(cell_id)
    return tuple(int(channel) for channel in frame[top, left])


class TestBandsStackWithDepthAsHeight:
    def test_l0_is_at_the_bottom_and_the_apex_at_the_top(self, dome):
        layout = BandLayout(dome)
        deepest = max(cell.index.level for cell in dome.cells)
        rows = {}
        for slot in layout.slots:
            rows.setdefault(slot.level, []).append(slot.row)
        assert min(rows[0]) > max(rows[deepest])
        # Not just the two ends: every band is above the one below it, which is
        # what makes the vertical axis hop distance from the rim.
        tops = [min(rows[level]) for level in sorted(rows)]
        assert tops == sorted(tops, reverse=True)

    def test_one_band_per_level_and_no_two_bands_overlap(self, dome):
        layout = BandLayout(dome)
        levels = {cell.index.level for cell in dome.cells}
        assert [level for level, _top, _rows in layout.bands] == sorted(levels, reverse=True)
        spans = [range(top, top + rows) for _level, top, rows in layout.bands]
        for above, below in zip(spans, spans[1:]):
            assert above.stop < below.start

    def test_every_cell_has_exactly_one_slot_and_no_slot_is_shared(self, dome):
        layout = BandLayout(dome)
        assert {slot.cell for slot in layout.slots} == {cell.id for cell in dome.cells}
        places = {(slot.row, slot.column) for slot in layout.slots}
        assert len(places) == len(layout.slots)

    def test_a_vision_lattice_is_drawn_at_its_own_lattice_shape(self, dome):
        """The 8x8 of L1 is an 8x8 of slots, in the index's own arrangement."""
        layout = BandLayout(dome)
        placed = {
            dome.cells[slot.cell].index.position: (slot.row, slot.column)
            for slot in layout.slots
            if slot.level == 1 and slot.column_name == "vision"
        }
        side = dome.spec.vision_sides[0]
        assert len(placed) == side * side
        origin = placed[(0, 0)]
        for (r, c), (row, column) in placed.items():
            assert (row - origin[0], column - origin[1]) == (r, c)

    def test_the_somatomotor_column_is_drawn_beside_the_vision_lattice(self, dome):
        """Where the cluster actually attaches: one region of the band, not the lattice."""
        layout = BandLayout(dome)
        for level in (0, 1, 2):
            vision = [
                slot for slot in layout.slots
                if slot.level == level and slot.column_name == "vision"
            ]
            column = [
                slot for slot in layout.slots
                if slot.level == level and slot.column_name == "somatomotor"
            ]
            assert column
            assert min(slot.column for slot in column) > max(slot.column for slot in vision)
            # Beside it, inside the band the lattice's shape sets.
            assert max(slot.row for slot in column) <= max(slot.row for slot in vision)

    def test_a_core_level_is_drawn_as_the_row_of_indices_it_is(self, dome):
        """The core is not a lattice, so its band is the index, laid along it."""
        layout = BandLayout(dome)
        core = [
            slot for slot in layout.slots if slot.level == 3 and slot.column_name == "core"
        ]
        assert len({slot.row for slot in core}) == 1
        order = sorted(core, key=lambda slot: slot.column)
        assert [dome.cells[slot.cell].index.position for slot in order] == [
            (i,) for i in range(len(core))
        ]

    def test_the_drive_gets_a_slot_at_the_apex_beside_the_core(self, dome):
        """Its mark is a later ticket's; the frame keeps its place at the apex."""
        layout = BandLayout(dome)
        (drive,) = [slot for slot in layout.slots if slot.column_name == "internal rim"]
        apex = [slot for slot in layout.slots if slot.level == drive.level and slot is not drive]
        assert drive.column > max(slot.column for slot in apex)

    def test_the_positions_come_from_the_construction_index(self):
        """Change the index and the drawing changes with it. Nothing else feeds it."""
        wider = build_graph(
            DomeSpec(
                patch_grid=8,
                vision_sides=(4, 2),
                somatomotor_sizes=(6, 4),
                patch_stalk=192,
                core_sizes=(7, 5, 4, 3, 2),
            )
        )
        layout = BandLayout(wider)
        lattice = [
            slot for slot in layout.slots if slot.level == 1 and slot.column_name == "vision"
        ]
        assert len({slot.row for slot in lattice}) == 4
        assert len({slot.column for slot in lattice}) == 4

    def test_a_mark_is_the_slot_less_the_gap_that_separates_it(self, dome):
        layout = BandLayout(dome, pitch=10)
        assert layout.mark == 9
        assert layout.width == layout.columns * 10
        assert layout.height == layout.rows * 10

    def test_a_slot_is_a_whole_number_of_pixels_across(self, dome):
        for refused in (0, 1, -4, 2.5, True):
            with pytest.raises(ValueError):
                BandLayout(dome, pitch=refused)


class TestColourIsPredictionErrorNormalisedPerCell:
    """Per cell, against that cell's own running statistics -- not a raw norm.

    The whole acceptance claim rests on this: the sensory funnel carries 12,288
    numbers at the base and eight core cells carry 32 dimensions each, so a raw
    map would show the taper's shape and "which level lit up" would stop meaning
    anything.
    """

    def two_cells(self, small, quiet, loud):
        """A panel fed one cell at scale `quiet` and one at scale `loud`."""
        cells = len(small.predicting)
        panel = DomePanel(small, np.full(cells, 1.0))
        rng = np.random.default_rng(0)
        base = np.zeros(cells)
        for tick in range(1, 40):
            error = base.copy()
            error[0] = quiet * (1.0 + 0.1 * rng.standard_normal())
            error[1] = loud * (1.0 + 0.1 * rng.standard_normal())
            panel.frame(record(tick, error))
        return panel

    def test_the_same_deviation_from_each_cells_own_baseline_lights_the_same(self, small):
        """Scales 10,000 apart; the same surprise; the same mark."""
        cells = len(small.predicting)
        panel = self.two_cells(small, quiet=0.001, loud=10.0)
        error = panel.mean.copy()
        error[0] += 3.0 * panel.spread[0]
        error[1] += 3.0 * panel.spread[1]
        frame = panel.frame(record(100, error))
        quiet, loud = small.predicting[0], small.predicting[1]
        assert mark(panel, frame, quiet) == mark(panel, frame, loud)
        assert panel.glow[0] == pytest.approx(panel.glow[1], abs=1e-9)

    def test_a_raw_norm_ten_thousand_times_larger_is_not_thereby_brighter(self, small):
        panel = self.two_cells(small, quiet=0.001, loud=10.0)
        # Both cells sitting exactly where they always sit.
        frame = panel.frame(record(100, panel.mean))
        quiet, loud = small.predicting[0], small.predicting[1]
        assert mark(panel, frame, quiet) == mark(panel, frame, loud)

    def test_a_chronically_wrong_cell_renders_calm(self, small):
        """The consequence the record accepts rather than fixes."""
        cells = len(small.predicting)
        panel = DomePanel(small, np.full(cells, 1.0))
        error = np.zeros(cells)
        error[0] = 50.0
        for tick in range(1, 40):
            frame = panel.frame(record(tick, error))
        assert panel.glow[0] == pytest.approx(0.0, abs=1e-6)
        wrong = small.predicting[0]
        assert mark(panel, frame, wrong) == mark(panel, frame, small.predicting[1])

    def test_a_cell_below_its_own_baseline_is_calm_rather_than_lit(self, small):
        """Prediction error is a magnitude: quieter than usual is not surprise."""
        cells = len(small.predicting)
        panel = DomePanel(small, np.full(cells, 1.0))
        rng = np.random.default_rng(1)
        for tick in range(1, 40):
            error = np.full(cells, 1.0) + 0.1 * rng.standard_normal(cells)
            panel.frame(record(tick, error))
        panel.frame(record(100, np.zeros(cells)))
        assert np.allclose(panel.glow, 0.0)

    def test_a_boundary_cells_slot_carries_no_prediction_error(self, small):
        """Boundary cells make no prediction (ADR-0006); colouring them would be a lie.

        A record carrying nothing but prediction error leaves every one of them
        empty -- not calm, which is a reading, and not bright, which is the
        fabrication. What they are drawn from when a record carries it is
        `TestNoBoundaryCellIsEverAssignedAPredictionError`, below.
        """
        cells = len(small.predicting)
        panel = DomePanel(small, np.full(cells, 1.0))
        frame = panel.frame(record(1, np.full(cells, 7.0)))
        for cell_id in small.boundary:
            assert mark(panel, frame, cell_id) == dome_panel_module._EMPTY

    @pytest.mark.parametrize("gone", [np.inf, np.nan, -np.inf])
    def test_a_cell_with_no_reading_is_drawn_as_one_and_not_as_calm(self, small, gone):
        """A diverging cell is what the display exists to show, not to absorb."""
        cells = len(small.predicting)
        panel = DomePanel(small, np.full(cells, 2.0))
        rng = np.random.default_rng(11)
        for tick in range(1, 10):
            panel.frame(record(tick, np.abs(rng.standard_normal(cells))))
        settled = panel.mean.copy(), panel.spread.copy()

        error = np.abs(rng.standard_normal(cells))
        error[0] = gone
        frame = panel.frame(record(10, error))
        assert panel.no_reading[0] and not panel.no_reading[1:].any()
        assert mark(panel, frame, small.predicting[0]) == dome_panel_module._NO_READING
        # And the run carries on: the cell's own statistics stand where they
        # were, and nothing leaked into anyone else's.
        assert panel.mean[0] == settled[0][0]
        assert panel.spread[0] == settled[1][0]
        assert np.all(np.isfinite(panel.mean)) and np.all(np.isfinite(panel.glow))

    def test_one_infinity_does_not_black_out_the_raw_map(self, small):
        cells = len(small.predicting)
        panel = DomePanel(small, np.full(cells, 1e6), raw=True)
        first = np.zeros(cells)
        first[1] = 4.0
        panel.frame(record(1, first))
        gone = np.zeros(cells)
        gone[0], gone[1] = np.inf, 4.0
        panel.frame(record(2, gone))
        assert panel.glow[1] == pytest.approx(1.0)

    def test_a_cell_that_never_read_never_has_a_baseline(self, small):
        cells = len(small.predicting)
        panel = DomePanel(small, np.full(cells, 1.0))
        rng = np.random.default_rng(12)
        for tick in range(1, 10):
            error = np.abs(rng.standard_normal(cells))
            error[0] = np.nan
            panel.frame(record(tick, error))
        assert not panel.baseline[0]
        assert panel.baseline[1:].all()
        assert panel.warming_up == 1

    def test_a_marks_place_in_the_frame_is_the_lattice_under_the_notice(self, small):
        """`rect()` is what a caller indexes a frame with; the layout's is not."""
        cells = len(small.predicting)
        panel = DomePanel(small, np.full(cells, 1.0))
        offsets = set()
        for cell_id in (small.predicting[0], small.predicting[-1], small.boundary[0]):
            top, left, size = panel.rect(cell_id)
            lattice_top, lattice_left, lattice_size = panel.layout.rect(cell_id)
            assert size == lattice_size
            assert top == lattice_top + (panel.height - panel.layout.height)
            offsets.add(left - lattice_left)
            assert 0 <= top and top + size <= panel.height
            assert 0 <= left and left + size <= panel.width
        # One offset for every mark: the lattice is placed, not each cell.
        (offset,) = offsets
        # And it is the lattice **and the motor strip beside it** that are
        # centred, so the taper does not shift by whatever the strip needs.
        strip_top, strip_left, strip_height, strip_width = panel.motor_strip
        assert strip_left >= offset + panel.layout.width
        assert abs(offset - (panel.width - strip_left - strip_width)) <= 1
        assert strip_top >= panel.height - panel.layout.height
        assert strip_top + strip_height <= panel.height

    def test_a_frame_is_one_uint8_image_of_the_panels_own_size(self, small):
        cells = len(small.predicting)
        panel = DomePanel(small, np.full(cells, 1.0))
        frame = panel.frame(record(1, np.zeros(cells)))
        assert frame.shape == (panel.height, panel.width, 3)
        assert frame.dtype == np.uint8

    def test_the_colormap_is_monotone_in_luminance(self):
        """Calm reads dark and lit reads bright, at a glance and in greyscale."""
        ramp = colormap(np.linspace(0.0, 1.0, 64)).astype(float)
        luminance = ramp @ np.array([0.2126, 0.7152, 0.0722])
        assert np.all(np.diff(luminance) > 0)

    def test_no_reading_is_far_from_every_colour_the_map_can_produce(self):
        """Or *no reading* would be indistinguishable from some reading."""
        ramp = colormap(np.linspace(0.0, 1.0, 512)).astype(float)
        distance = np.linalg.norm(
            ramp - np.array(dome_panel_module._NO_READING, dtype=float), axis=1
        )
        assert distance.min() > 150.0


class TestTheRawMapIsBehindADebugFlag:
    def base(self, small, raw):
        cells = len(small.predicting)
        panel = DomePanel(small, np.full(cells, 1.0), raw=raw)
        error = np.zeros(cells)
        error[0], error[1] = 50.0, 0.5
        for tick in range(1, 20):
            frame = panel.frame(record(tick, error))
        return panel, frame

    def test_the_raw_map_shows_the_chronic_failure_the_normalised_one_hides(self, small):
        big, small_cell = small.predicting[0], small.predicting[1]
        panel, frame = self.base(small, raw=True)
        assert mark(panel, frame, big) != mark(panel, frame, small_cell)
        assert panel.glow[0] > panel.glow[1]

        normalised, calm = self.base(small, raw=False)
        assert mark(normalised, calm, big) == mark(normalised, calm, small_cell)

    def test_the_raw_map_puts_every_cell_on_one_shared_scale(self, small):
        cells = len(small.predicting)
        panel = DomePanel(small, np.full(cells, 1.0), raw=True)
        error = np.zeros(cells)
        error[0], error[1] = 1.0, 0.5
        panel.frame(record(1, error))
        assert panel.glow[0] == pytest.approx(1.0)
        assert panel.glow[1] == pytest.approx(0.5)

    def test_a_new_largest_error_rescales_the_trail_along_with_the_map(self, small):
        """One shared scale, the trail included -- or two cells stop comparable."""
        cells = len(small.predicting)
        panel = DomePanel(small, np.full(cells, 1e6), raw=True)
        first = np.zeros(cells)
        first[0] = 1.0
        panel.frame(record(0, first))
        assert panel.glow[0] == pytest.approx(1.0)
        later = np.zeros(cells)
        later[1] = 100.0
        panel.frame(record(1, later))
        assert panel.glow[1] == pytest.approx(1.0)
        assert panel.glow[0] == pytest.approx(0.01, rel=1e-4)

    def test_switching_the_flag_clears_the_trail_it_cannot_carry(self, small):
        """The two maps' values are different quantities; a glow is one of them."""
        cells = len(small.predicting)
        panel = DomePanel(small, np.full(cells, 1e6))
        rng = np.random.default_rng(7)
        for tick in range(1, 8):
            panel.frame(record(tick, np.abs(rng.standard_normal(cells))))
        assert panel.glow.any()
        panel.raw = True
        assert not panel.glow.any()
        panel.frame(record(9, np.abs(rng.standard_normal(cells))))
        lit = panel.glow.copy()
        panel.raw = True
        assert np.array_equal(panel.glow, lit)

    def test_the_flag_changes_what_is_drawn_and_never_the_statistics(self, small):
        cells = len(small.predicting)
        rng = np.random.default_rng(2)
        errors = [np.abs(rng.standard_normal(cells)) for _ in range(12)]
        steady = DomePanel(small, np.full(cells, 1.0))
        switched = DomePanel(small, np.full(cells, 1.0))
        for tick, error in enumerate(errors, start=1):
            steady.frame(record(tick, error))
            switched.raw = tick == 6
            switched.frame(record(tick, error))
        assert np.array_equal(steady.mean, switched.mean)
        assert np.array_equal(steady.spread, switched.spread)
        assert np.array_equal(steady.baseline, switched.baseline)

    def test_the_raw_map_says_nothing_about_warming_up(self, small):
        """There are no per-cell statistics on it to be warming."""
        cells = len(small.predicting)
        panel = DomePanel(small, np.full(cells, 1.0), raw=True)
        frame = panel.frame(record(1, np.full(cells, 3.0)))
        notice = frame[: panel.height - panel.layout.height]
        assert np.array_equal(
            np.unique(notice.reshape(-1, 3), axis=0),
            np.array([dome_panel_module._BACKGROUND]),
        )


class TestThePanelSaysItIsWarmingUp:
    """On screen, until each cell has a baseline, rather than pretending."""

    def notice(self, panel, frame):
        return frame[: panel.height - panel.layout.height]

    def has_ink(self, panel, frame):
        strip = self.notice(panel, frame)
        return bool((strip != np.array(dome_panel_module._BACKGROUND)).any())

    def test_a_panel_opened_on_a_cold_run_says_so(self, small):
        cells = len(small.predicting)
        panel = DomePanel(small, np.full(cells, 8.0))
        frame = panel.frame(record(1, np.full(cells, 1.0)))
        assert panel.warming_up == cells
        assert self.has_ink(panel, frame)

    def test_a_cell_has_no_baseline_until_the_statistics_span_its_own_persistence(
        self, small
    ):
        """Per cell, and the slow cells are the ones still warming."""
        cells = len(small.predicting)
        persistence = np.full(cells, 2.0)
        persistence[0] = 200.0
        panel = DomePanel(small, persistence)
        rng = np.random.default_rng(3)
        for tick in range(1, 12):
            frame = panel.frame(record(tick, np.abs(rng.standard_normal(cells))))
        assert panel.baseline[1:].all()
        assert not panel.baseline[0]
        assert panel.warming_up == 1
        assert self.has_ink(panel, frame)

    def test_the_notice_goes_away_once_every_cell_has_one(self, small):
        cells = len(small.predicting)
        panel = DomePanel(small, np.full(cells, 4.0))
        rng = np.random.default_rng(4)
        for tick in range(1, 12):
            frame = panel.frame(record(tick, np.abs(rng.standard_normal(cells))))
        assert panel.warming_up == 0
        assert not self.has_ink(panel, frame)

    def test_two_observations_are_not_a_baseline_for_a_slow_cell(self, small):
        """A spread measured over a stretch the cell barely moved in is not one."""
        cells = len(small.predicting)
        panel = DomePanel(small, np.full(cells, 50.0))
        panel.frame(record(1, np.full(cells, 1.0)))
        panel.frame(record(2, np.full(cells, 2.0)))
        assert panel.warming_up == cells

    def test_the_frame_is_one_size_for_the_whole_run(self, small):
        """A capture whose frames change shape halfway through is not a capture."""
        cells = len(small.predicting)
        panel = DomePanel(small, np.full(cells, 2.0))
        shapes = set()
        rng = np.random.default_rng(5)
        for tick in range(1, 20):
            shapes.add(panel.frame(record(tick, np.abs(rng.standard_normal(cells)))).shape)
        assert len(shapes) == 1

    def test_the_notice_fits_in_the_frame(self, small):
        cells = len(small.predicting)
        panel = DomePanel(small, np.full(cells, 8.0))
        assert panel.width >= dome_panel_module._text_width(
            panel._notice(cells), panel._notice_scale
        )


class TestTheTrail:
    """One exponential per cell, at that cell's own measured persistence."""

    def spiked(self, small, persistence):
        """A panel with one raw spike in it, on the raw map so the value is known."""
        cells = len(small.predicting)
        panel = DomePanel(small, np.asarray(persistence, dtype=float), raw=True)
        panel.frame(record(0, np.ones(cells)))
        return panel

    def test_glow_decays_at_the_cells_own_measured_persistence(self, small):
        cells = len(small.predicting)
        persistence = np.linspace(2.0, 100.0, cells)
        panel = self.spiked(small, persistence)
        assert np.allclose(panel.glow, 1.0)
        panel.frame(record(10, np.zeros(cells)))
        assert np.allclose(panel.glow, np.exp(-10.0 / persistence))

    def test_a_rim_cell_goes_dark_while_a_core_cell_stays_lit(self, small):
        """The multi-timescale claim, rendered for free."""
        cells = len(small.predicting)
        persistence = np.full(cells, 1.0)
        persistence[-1] = 400.0
        panel = self.spiked(small, persistence)
        panel.frame(record(40, np.zeros(cells)))
        assert panel.glow[0] < 1e-9
        assert panel.glow[-1] > 0.9

    def test_the_decay_is_over_ticks_and_not_over_frames(self, small):
        """A 10 Hz capture decimates the frames, never the clock they decay on."""
        cells = len(small.predicting)
        persistence = np.linspace(3.0, 60.0, cells)
        every_tick = DomePanel(small, persistence, raw=True)
        decimated = DomePanel(small, persistence, raw=True)
        every_tick.frame(record(0, np.ones(cells)))
        decimated.frame(record(0, np.ones(cells)))
        for tick in range(1, 21):
            every_tick.frame(record(tick, np.zeros(cells)))
            if tick % 5 == 0:
                decimated.frame(record(tick, np.zeros(cells)))
        assert np.allclose(every_tick.glow, decimated.glow)

    def test_a_fresh_value_takes_the_glow_up_but_never_down(self, small):
        cells = len(small.predicting)
        panel = self.spiked(small, np.full(cells, 100.0))
        panel.frame(record(1, np.zeros(cells)))
        held = panel.glow.copy()
        assert np.all(held > 0.9)
        panel.frame(record(2, np.zeros(cells)))
        assert np.all(panel.glow < held)

    def test_the_trail_is_not_driven_by_delta_private(self, small):
        """The claim the display tests must not be the number the display decays on.

        `‖Δ private‖` is the private-component panel's readout and `08`'s depth
        measurement. Drive the trail with it and the panel could never contradict
        the thesis it exists to test.
        """
        cells = len(small.predicting)
        rng = np.random.default_rng(6)
        errors = [np.abs(rng.standard_normal(cells)) for _ in range(15)]
        plain = DomePanel(small, np.linspace(2.0, 50.0, cells))
        scrambled = DomePanel(small, np.linspace(2.0, 50.0, cells))
        for tick, error in enumerate(errors, start=1):
            left = plain.frame(record(tick, error, private=np.zeros(cells)))
            right = scrambled.frame(
                record(tick, error, private=1e6 * np.abs(rng.standard_normal(cells)))
            )
            assert np.array_equal(left, right), f"at tick {tick}"
            assert np.array_equal(plain.glow, scrambled.glow)

    def test_the_module_never_reads_the_private_delta_at_all(self):
        """The structural half: there is no line for the coupling to creep into.

        Over the parsed module rather than its text, so the prose above -- which
        says at length that the trail is not driven by it -- is not what passes
        this.
        """
        tree = ast.parse(inspect.getsource(dome_panel_module))
        read = {
            node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
        } | {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        assert "private_delta" not in read

    def test_the_panel_is_fed_in_order_and_each_record_once(self, small):
        cells = len(small.predicting)
        panel = DomePanel(small, np.full(cells, 5.0))
        panel.frame(record(10, np.zeros(cells)))
        for refused in (10, 9, 0):
            with pytest.raises(ValueError, match="in order"):
                panel.frame(record(refused, np.zeros(cells)))


class TestClosingItChangesNothingButTheView:
    def test_a_run_watched_by_a_panel_is_bit_identical_to_an_unwatched_one(self, dome):
        """The acceptance criterion, with the world in it -- panel open, then closed."""
        plain_env = PlanarPushSandbox(split="any")
        watched_env = PlanarPushSandbox(split="any")
        try:
            plain = started(plain_env, dome)
            watched = started(watched_env, dome)
            recorder = Recorder(watched)
            panel = DomePanel(dome, np.full(len(dome.predicting), 20.0))
            for tick in range(TICKS):
                left = plain.tick()
                right = watched.tick()
                captured = recorder.observe()
                if captured is not None and not panel.closed:
                    panel.frame(captured)
                if tick == TICKS // 2:
                    panel.close()
                where = f"at tick {tick}"
                assert np.array_equal(left.command, right.command), where
                assert np.array_equal(left.applied, right.applied), where
                for name in ("stalks", "charts", "prediction", "broadcast", "incoming"):
                    assert torch.equal(
                        getattr(plain.sheaf, name), getattr(watched.sheaf, name)
                    ), f"{name} differs {where}"
                assert np.array_equal(
                    snapshot(plain_env).physics, snapshot(watched_env).physics
                ), where
        finally:
            plain_env.close()
            watched_env.close()

    def test_a_closed_panel_drains_its_feed_rather_than_stopping_it(self, small):
        """A live feed is a run being driven; a closed window must not stop one."""
        cells = len(small.predicting)
        panel = DomePanel(small, np.full(cells, 5.0))
        drawn = []

        def feed():
            for tick in range(1, 11):
                if tick == 4:
                    panel.close()
                drawn.append(tick)
                yield record(tick, np.zeros(cells))

        frames = list(panel.frames(feed()))
        assert drawn == list(range(1, 11))
        assert len(frames) == 3

    def test_a_closed_panel_draws_nothing(self, small):
        cells = len(small.predicting)
        panel = DomePanel(small, np.full(cells, 5.0))
        panel.close()
        panel.close()
        assert panel.closed
        with pytest.raises(ValueError, match="closed"):
            panel.frame(record(1, np.zeros(cells)))

    def test_it_closes_on_the_way_out_of_a_with_block(self, small):
        cells = len(small.predicting)
        with DomePanel(small, np.full(cells, 5.0)) as panel:
            panel.frame(record(1, np.zeros(cells)))
        assert panel.closed

    def test_no_agent_sheaf_or_world_is_reachable_from_a_panel(self, env, dome):
        """The structural half of *no cell reads anything the surface computes*.

        The panel is handed a record and holds statistics; if the run itself is
        not reachable from it, there is nothing for it to touch.
        """
        agent = started(env, dome)
        panel = DomePanel(dome, measured_persistence(agent.sheaf, ticks=8, burn_in=2))
        forbidden = (Agent, type(agent.sheaf), PlanarPushSandbox, Recorder, Trace)
        seen, stack, found = set(), [("<root>", panel)], []
        while stack:
            path, obj = stack.pop()
            if id(obj) in seen:
                continue
            seen.add(id(obj))
            assert len(seen) < 200_000, "the object walk ran away"
            if isinstance(obj, forbidden):
                found.append(path)
                continue
            leaf = (str, bytes, int, float, bool, type(None))
            if isinstance(obj, leaf + (torch.Tensor, np.ndarray)):
                continue
            if isinstance(obj, type) or isinstance(obj, types.ModuleType):
                continue
            if isinstance(obj, (list, tuple, set, frozenset)):
                stack.extend((f"{path}[{i}]", v) for i, v in enumerate(obj))
            elif isinstance(obj, dict):
                for key, value in obj.items():
                    stack.append((f"{path}[{key!r}]", value))
            else:
                for name, value in getattr(obj, "__dict__", {}).items():
                    stack.append((f"{path}.{name}", value))
        assert found == []


class TestThePersistenceIsTimescalesOwnEstimate:
    def test_one_positive_finite_persistence_per_predicting_cell(self, small_sheaf):
        measured = measured_persistence(
            small_sheaf, ticks=8, burn_in=2, generator=torch.Generator().manual_seed(0)
        )
        assert measured.shape == (len(small_sheaf.dome.predicting),)
        assert np.all(np.isfinite(measured)) and np.all(measured > 0)

    def test_it_is_bias_selections_estimate_and_not_a_second_one(self, small_sheaf):
        """`05-timescales.md`'s, over this run's own body and biases."""
        mine = measured_persistence(
            small_sheaf, ticks=8, burn_in=2, generator=torch.Generator().manual_seed(1)
        )
        theirs = bias_selection.measure(
            small_sheaf.body,
            small_sheaf.biases,
            ticks=8,
            burn_in=2,
            generator=torch.Generator().manual_seed(1),
        ).effective_timescale
        assert np.allclose(mine, theirs.numpy())

    def test_measuring_it_changes_no_parameter_and_leaves_no_tape(self, small_sheaf):
        before = {
            name: parameter.detach().clone()
            for name, parameter in small_sheaf.biases.named_parameters()
        }
        measured_persistence(small_sheaf, ticks=8, burn_in=2)
        for name, parameter in small_sheaf.biases.named_parameters():
            assert torch.equal(parameter.detach(), before[name]), name
            assert parameter.grad is None, name
        small_sheaf.assert_no_tape()

    @pytest.mark.parametrize("field", ["rho_median", "finite"])
    def test_a_cell_with_no_persistence_is_refused_rather_than_clamped(
        self, small_sheaf, monkeypatch, field
    ):
        """An expansive region has no `tau`, and the clamp is not a timescale.

        Handed to a trail, the clamp is a cell lit forever; a cell that
        overflowed comes back the *fastest* in the dome, because a NaN reads
        downstream as no unit active. Both would be a display stating a rate
        nothing measured.
        """
        real = bias_selection.measure

        def hobbled(*args, **kwargs):
            measurement = real(*args, **kwargs)
            broken = getattr(measurement, field).clone()
            broken[0] = 1.5 if field == "rho_median" else False
            return dataclasses.replace(measurement, **{field: broken})

        monkeypatch.setattr(dome_panel_module, "measure", hobbled)
        with pytest.raises(ValueError, match="no measured persistence"):
            measured_persistence(small_sheaf, ticks=8, burn_in=2)

    def test_a_persistence_that_is_not_one_per_cell_is_refused(self, small):
        cells = len(small.predicting)
        with pytest.raises(ValueError, match="per predicting cell"):
            DomePanel(small, np.full(cells + 1, 5.0))

    def test_a_persistence_that_is_not_a_number_of_ticks_is_refused(self, small):
        cells = len(small.predicting)
        for refused in (np.zeros(cells), np.full(cells, -1.0), np.full(cells, np.inf)):
            with pytest.raises(ValueError, match="positive, finite"):
                DomePanel(small, refused)

    def test_a_record_from_another_graph_is_refused(self, small):
        cells = len(small.predicting)
        panel = DomePanel(small, np.full(cells, 5.0))
        with pytest.raises(ValueError, match="different graphs"):
            panel.frame(record(1, np.zeros(cells + 3)))


class TestTheArchitectureDoesNotImportTheSurface:
    """What keeps a measured persistence out of the architecture at runtime.

    `tests/test_timescale.py`'s scan catches a module that *names* a schedule --
    `divisor`, `timescale`, `clock`, `schedule`, `cadence` -- and
    `measured_persistence` names none of them, nor do `surface`, `dome_panel` or
    `DomePanel`. So the naming scan cannot see this door, and this is the one
    that shuts it: the surface is not imported by anything but the surface, and
    `patchworks/__init__.py` does not import it either. A learning rule reaching
    for a per-cell timescale would have to add an import that shows up here.
    """

    def imports(self, path):
        names = set()
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                names.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                names.add(node.module or "")
                names.update(f"{node.module or ''}.{a.name}" for a in node.names)
        return names

    def test_no_module_outside_the_surface_imports_it(self):
        package = pathlib.Path(patchworks.__file__).resolve().parent
        offences = {}
        for path in sorted(package.rglob("*.py")):
            relative = path.relative_to(package).as_posix()
            if relative.startswith("surface/"):
                continue
            hit = sorted(
                name
                for name in self.imports(path)
                if "surface" in name.split(".")
            )
            if hit:
                offences[relative] = hit
        assert offences == {}

    def test_the_scan_catches_a_rule_that_reaches_for_one(self, tmp_path):
        offender = tmp_path / "rule.py"
        offender.write_text(
            "from patchworks.surface import measured_persistence\n"
            "def rule(sheaf):\n    return measured_persistence(sheaf)\n"
        )
        assert [
            name for name in self.imports(offender) if "surface" in name.split(".")
        ]


class TestOneRendererOverATickRecord:
    def test_a_live_feed_and_a_trace_off_disk_draw_the_same_frames(
        self, env, dome, tmp_path
    ):
        """Live and replay are two iterables, not two code paths."""
        agent = started(env, dome)
        recorder = Recorder(agent)
        persistence = np.full(len(dome.predicting), 20.0)
        live = DomePanel(dome, persistence)
        drawn = [live.frame(r) for r in recorder.watch(TICKS, seed=0)]
        path = recorder.trace.save(tmp_path / "run")
        replayed = DomePanel(dome, persistence)
        again = list(replayed.frames(Trace.load(path)))
        assert len(drawn) == len(again) > 0
        for i, (left, right) in enumerate(zip(drawn, again)):
            assert np.array_equal(left, right), f"frame {i}"

    def test_the_boundary_band_comes_back_off_disk_too(self, env, dome, tmp_path):
        """The band is re-rendered from the record's own state, at capture time.

        A record holds no frame, so the picture at the bottom of the panel is
        the scene renderer's over the same file -- which is what keeps the
        capture resolution a choice made when rendering rather than one baked
        into the recording.
        """
        agent = started(env, dome)
        recorder = Recorder(agent)
        list(recorder.watch(TICKS, seed=0))
        path = recorder.trace.save(tmp_path / "run")
        persistence = np.full(len(dome.predicting), 20.0)
        frames = []
        for _ in range(2):
            panel = DomePanel(dome, persistence)
            with Renderer(size=dome.spec.patch_grid * 4) as scene:
                frames.append(list(panel.frames(Trace.load(path), renderer=scene.frame)))
        assert len(frames[0]) > 0
        for i, (left, right) in enumerate(zip(*frames)):
            assert np.array_equal(left, right), f"frame {i}"
        # And the band is a picture rather than the empty slots of a panel that
        # was handed no render.
        patch = only(dome, "patch")[len(only(dome, "patch")) // 2]
        assert mark(panel, frames[0][-1], patch) != dome_panel_module._EMPTY


class TestTheFourWaysThePanelCouldHaveFlatteredUs:
    """One test per finding of #93's review, each one red before its fix.

    Grouped because they are one class of defect rather than four unrelated
    ones: each is the panel claiming something it had not earned -- a baseline
    it had not watched for, a trail it had not been fed, a persistence it had
    validated and then let go of, and a measurement it took out of the run's
    own random stream.
    """

    def test_a_cell_that_read_nothing_for_most_of_the_run_is_still_warming_up(
        self, small
    ):
        """A cell's baseline spans that cell's readings, not the panel's ticks.

        Recovering from divergence is the case this panel exists to show, and
        the notice clearing early would colour a slow cell against a two-sample
        baseline -- the pretending `10-the-demo-surface.md` rules out.
        """
        cells = len(small.predicting)
        panel = DomePanel(small, np.full(cells, 100.0))
        rng = np.random.default_rng(11)
        for tick in range(1, 300):
            error = np.abs(rng.standard_normal(cells)) + 1.0
            error[0] = np.nan
            panel.frame(record(tick, error))
        # Every other cell has been read for 299 ticks and has a baseline.
        assert panel.baseline[1:].all()
        assert not panel.baseline[0]
        for tick in (300, 301):
            panel.frame(record(tick, np.abs(rng.standard_normal(cells)) + 1.0))
        assert panel.spread[0] > 0.0, "two readings, so it does have a spread"
        assert not panel.baseline[0], "but two readings are not 100 ticks of watching"
        assert panel.warming_up == 1

    def test_a_diverged_cells_trail_decays_on_the_raw_map_too(self, small):
        """No reading is kept out of the glow on both maps, not just one.

        On the raw map a cell standing at its own mean is a positive raw norm,
        and feeding that to the trail would pin a diverged cell's glow at a
        constant forever -- and draw it, on recovery, from a brightness it
        never produced.
        """
        cells = len(small.predicting)
        panel = DomePanel(small, np.full(cells, 5.0), raw=True)
        error = np.zeros(cells)
        error[0], error[1] = 1.0, 10.0
        panel.frame(record(0, error))
        assert panel.glow[0] == pytest.approx(0.1)
        gone = error.copy()
        gone[0] = np.nan
        for tick in range(1, 61):
            panel.frame(record(tick, gone))
        # Pinned at 0.1 before the fix; now the plain exponential it should be.
        assert panel.glow[0] == pytest.approx(0.1 * np.exp(-60.0 / 5.0), rel=1e-9)
        assert panel.glow[0] < 1e-6
        assert panel.no_reading[0]

    def test_the_persistence_is_not_the_callers_array_to_change(self, small):
        """Validated once, so it must not still be reachable to invalidate.

        A negative persistence turns `exp(-elapsed / tau)` into growth without
        decay -- a trail that brightens on its own, past the check that exists
        to refuse it.
        """
        cells = len(small.predicting)
        mine = np.full(cells, 5.0)
        panel = DomePanel(small, mine)
        mine[:] = -1.0
        assert np.all(panel.persistence == 5.0)
        with pytest.raises(ValueError):
            panel.persistence[0] = -1.0

    def test_measuring_the_persistence_does_not_touch_the_global_rng(
        self, small_sheaf
    ):
        """#77: switching the surface off must change no trajectory.

        `measure` draws a batch of normals per tick of its trajectory. Taken
        from the global stream, opening a panel would change the parameters of
        every `Sheaf` or `Agent` built after it without a generator of its own.
        """
        torch.manual_seed(1234)
        expected = torch.randn(3)

        torch.manual_seed(1234)
        measured_persistence(small_sheaf, ticks=8, burn_in=2)
        assert torch.equal(torch.randn(3), expected)

    def test_the_default_measurement_is_repeatable(self, small_sheaf):
        """A private generator's seed is fixed, so inert does not cost repeatable."""
        first = measured_persistence(small_sheaf, ticks=8, burn_in=2)
        torch.manual_seed(99)
        second = measured_persistence(small_sheaf, ticks=8, burn_in=2)
        assert np.array_equal(first, second)


# -- #94: the marks a boundary cell gets ------------------------------------


def full(tick, cells, edges, *, error=0.0, disagreement=0.0, actuator=None):
    """A record carrying every array, sized to a dome.

    The counterpart of :func:`record` for the marks a boundary cell gets: those
    are drawn from edge disagreement and from the actuator's own two rows, so a
    test about them has to hand over a record that carries both.
    """
    return record(
        tick,
        np.full(cells, error) if np.isscalar(error) else np.asarray(error, dtype=float),
        disagreement=(
            np.full(edges, disagreement)
            if np.isscalar(disagreement)
            else np.asarray(disagreement, dtype=float)
        ),
        actuator=np.zeros((2, 3)) if actuator is None else actuator,
    )


def painted(panel, frame, colour):
    """`(top row, bottom row)` of everything painted `colour` on the motor strip.

    Measured off the pixels rather than computed from the panel's own geometry:
    what the bars claim is that a fill falls short of its outline, and a test
    that asked the panel where it drew them would be asking the same arithmetic
    twice.
    """
    top, left, height, width = panel.motor_strip
    region = frame[top : top + height, left : left + width]
    hit = np.all(region == np.array(colour, dtype=np.uint8), axis=-1)
    rows = np.flatnonzero(hit.any(axis=1))
    return (int(rows.min()), int(rows.max())) if rows.size else None


def standing_bar(panel, frame):
    """`(top row, bottom row)` of the strip's disagreement bar.

    Everything on the strip that is neither the ground nor one of the strip's
    own inks: the fourth bar is drawn on the colormap, so it is the only thing
    there in a colour the strip does not otherwise use.
    """
    top, left, height, width = panel.motor_strip
    region = frame[top : top + height, left : left + width]
    known = (
        dome_panel_module._BACKGROUND,
        dome_panel_module._ZERO_LINE,
        dome_panel_module._COMMANDED,
        dome_panel_module._APPLIED,
    )
    hit = np.ones(region.shape[:2], dtype=bool)
    for ink in known:
        hit &= ~np.all(region == np.array(ink, dtype=np.uint8), axis=-1)
    rows = np.flatnonzero(hit.any(axis=1))
    return (int(rows.min()), int(rows.max())) if rows.size else None


def kind_of(dome, cell_id):
    return dome.cells[cell_id].kind.value


def only(dome, kind):
    """The cell ids of one boundary kind, in id order."""
    return [cell.id for cell in dome.cells if cell.kind.value == kind]


class TestTheBoundaryBandDrawsTheRender:
    """L0 draws the agent's own render, tiled into the patch lattice.

    Boundary cells run no body and make no prediction, so the largest and most
    eye-catching band on screen cannot be drawn from prediction error without
    fabricating it. The render costs nothing -- it already exists every tick --
    and it ties the abstract stack to the world.
    """

    def picture(self, small, side=4):
        """A render whose every patch block is a flat colour of its own."""
        grid = small.spec.patch_grid
        image = np.zeros((grid * side, grid * side, 3), dtype=np.uint8)
        for r in range(grid):
            for c in range(grid):
                image[r * side : (r + 1) * side, c * side : (c + 1) * side] = (
                    20 + 50 * r,
                    30 + 50 * c,
                    200,
                )
        return image

    def test_each_patch_cell_draws_its_own_block_of_the_render(self, small):
        cells = len(small.predicting)
        panel = DomePanel(small, np.full(cells, 4.0))
        image = self.picture(small)
        frame = panel.frame(record(1, np.zeros(cells)), render=image)
        side = image.shape[0] // small.spec.patch_grid
        for cell_id in only(small, "patch"):
            r, c = small.cells[cell_id].index.position
            block = image[r * side : (r + 1) * side, c * side : (c + 1) * side]
            assert mark(panel, frame, cell_id) == tuple(int(v) for v in block[0, 0])

    def test_the_band_carries_no_prediction_error_colour(self, small):
        """The same render draws the same band, whatever the dome is doing."""
        cells = len(small.predicting)
        image = self.picture(small)
        quiet = DomePanel(small, np.full(cells, 4.0))
        loud = DomePanel(small, np.full(cells, 4.0))
        calm = quiet.frame(record(1, np.zeros(cells)), render=image)
        burning = loud.frame(record(1, np.full(cells, 1e6)), render=image)
        for cell_id in only(small, "patch"):
            assert mark(quiet, calm, cell_id) == mark(loud, burning, cell_id)

    def test_without_a_render_the_band_is_empty_rather_than_coloured(self, small):
        cells = len(small.predicting)
        panel = DomePanel(small, np.full(cells, 4.0))
        frame = panel.frame(
            full(1, cells, len(small.edges), error=9.0, disagreement=3.0)
        )
        for cell_id in only(small, "patch"):
            assert mark(panel, frame, cell_id) == dome_panel_module._EMPTY

    def test_a_render_the_cells_were_never_cut_from_is_refused(self, small):
        cells = len(small.predicting)
        panel = DomePanel(small, np.full(cells, 4.0))
        for refused in (
            np.zeros((17, 17, 3), dtype=np.uint8),  # does not tile
            np.zeros((16, 12, 3), dtype=np.uint8),  # not square
            np.zeros((16, 16), dtype=np.uint8),  # no channels
        ):
            with pytest.raises(ValueError):
                panel.frame(record(1, np.zeros(cells)), render=refused)

    def test_the_tile_is_the_block_the_world_writes_into_that_cell(self, env, dome):
        """The panel's tiling is the agent's, not a second one that agrees today.

        What a patch cell is looking at is what the world wrote into its node
        stalk, so a tile drawn anywhere else would be a picture of the arena
        that no cell is reading.
        """
        agent = Agent(env, dome=dome, generator=torch.Generator().manual_seed(0))
        observation, _info = env.reset(seed=0)
        agent.observe(observation)
        image = observation["image"]
        panel = DomePanel(dome, np.full(len(dome.predicting), 20.0))
        frame = panel.frame(record(1, np.zeros(len(dome.predicting))), render=image)
        side = image.shape[0] // dome.spec.patch_grid
        patches = only(dome, "patch")
        for cell_id in (patches[0], patches[len(patches) // 3], patches[-1]):
            r, c = dome.cells[cell_id].index.position
            block = image[r * side : (r + 1) * side, c * side : (c + 1) * side]
            written = agent.sheaf.stalk(cell_id).numpy().reshape(side, side, 3)
            assert np.allclose(written * 255.0, block, atol=1e-3)
            top, left, size = panel.rect(cell_id)
            drawn = frame[top : top + size, left : left + size]
            # Nearest neighbour: every drawn pixel is one of this block's, and
            # the corners are the block's corners.
            assert set(map(tuple, drawn.reshape(-1, 3))) <= set(
                map(tuple, block.reshape(-1, 3))
            )
            assert tuple(drawn[0, 0]) == tuple(block[0, 0])
            assert tuple(drawn[-1, -1]) == tuple(block[-1, -1])


class TestTheStripAndTheDriveMarkDrawEdgeDisagreement:
    """What a boundary cell has instead of a prediction error is an edge.

    Three proprioceptive, three touch and one actuator beside the tiled render,
    and the drive on the apex band -- all on the same colormap, all honestly
    earned.
    """

    def settled(self, small, ticks=40, seed=0):
        """A panel whose marks have a baseline, over quiet random disagreement."""
        cells, edges = len(small.predicting), len(small.edges)
        panel = DomePanel(small, np.full(cells, 4.0))
        rng = np.random.default_rng(seed)
        for tick in range(1, ticks):
            panel.frame(
                full(
                    tick,
                    cells,
                    edges,
                    error=np.abs(rng.standard_normal(cells)),
                    disagreement=1.0 + 0.05 * rng.standard_normal(edges),
                )
            )
        return panel

    def test_the_marks_are_the_somatomotor_cluster_and_the_drive(self, small):
        cells = len(small.predicting)
        panel = DomePanel(small, np.full(cells, 4.0))
        kinds = sorted(kind_of(small, cell_id) for cell_id in panel.boundary_marks)
        assert kinds == [
            "actuator",
            "drive",
            "proprioceptive",
            "proprioceptive",
            "proprioceptive",
            "touch",
            "touch",
            "touch",
        ]

    def test_a_mark_lights_when_its_own_edge_disagrees(self, small):
        """Touch marks light on contact, which is this, in the world's terms."""
        cells, edges = len(small.predicting), len(small.edges)
        panel = self.settled(small)
        touch = only(small, "touch")[0]
        (edge,) = small.incident[touch]
        loud = np.full(edges, 1.0)
        loud[edge] = 20.0
        panel.frame(full(100, cells, edges, disagreement=loud))
        row = panel.boundary_marks.index(touch)
        lit = panel.boundary_lit
        assert lit[row] > 0.9
        assert all(value < 0.5 for i, value in enumerate(lit) if i != row)

    def test_the_drive_mark_is_the_drive_edges_disagreement(self, small):
        """`08`'s task-invariant near-miss is diagnosed off this mark."""
        cells, edges = len(small.predicting), len(small.edges)
        panel = self.settled(small)
        (drive,) = only(small, "drive")
        loud = np.full(edges, 1.0)
        for edge in small.incident[drive]:
            loud[edge] = 12.0
        frame = panel.frame(full(100, cells, edges, disagreement=loud))
        row = panel.boundary_marks.index(drive)
        assert panel.boundary_lit[row] > 0.9
        assert mark(panel, frame, drive) == tuple(
            int(channel) for channel in colormap(panel.boundary_lit[row])
        )

    def test_a_mark_carries_no_trail(self, small):
        """A boundary cell has no measured persistence for a glow to decay at.

        It runs no body, so `05-timescales.md`'s estimate is not defined for
        one, and a glow fading at a rate nothing measured is exactly the
        fabrication these marks exist to avoid.
        """
        cells, edges = len(small.predicting), len(small.edges)
        panel = self.settled(small)
        panel.frame(full(100, cells, edges, error=40.0, disagreement=40.0))
        assert panel.boundary_lit.max() > 0.9
        assert panel.glow.max() > 0.9
        panel.frame(full(101, cells, edges, error=0.0, disagreement=1.0))
        assert panel.boundary_lit.max() < 0.1, "this tick, and nothing carried over"
        assert panel.glow.max() > 0.5, "while a predicting cell's trail decays"

    def test_a_mark_with_no_reading_is_drawn_as_one(self, small):
        cells, edges = len(small.predicting), len(small.edges)
        panel = self.settled(small)
        gone = np.full(edges, 1.0)
        touch = only(small, "touch")[0]
        (edge,) = small.incident[touch]
        gone[edge] = np.nan
        frame = panel.frame(full(100, cells, edges, disagreement=gone))
        assert mark(panel, frame, touch) == dome_panel_module._NO_READING

    def test_the_notice_counts_the_marks_beside_the_cells(self, small):
        """Two readings is a boundary mark's baseline, and it says so until then."""
        cells, edges = len(small.predicting), len(small.edges)
        panel = DomePanel(small, np.full(cells, 4.0))
        panel.frame(full(1, cells, edges, disagreement=1.0))
        assert panel.boundary_warming_up == len(panel.boundary_marks)
        assert "MARKS" in panel._notice(panel.warming_up, panel.boundary_warming_up)
        panel.frame(full(2, cells, edges, disagreement=2.0))
        assert panel.boundary_warming_up == 0

    def test_a_mark_never_read_is_not_thereby_warming_up(self, small):
        """No disagreement in the record is nothing drawn, not a statistic settling."""
        cells = len(small.predicting)
        panel = DomePanel(small, np.full(cells, 1.0))
        for tick in range(1, 8):
            panel.frame(record(tick, np.full(cells, float(tick))))
        assert panel.boundary_warming_up == 0
        assert not panel.boundary_baseline.any()


class TestTheActuatorDrawsDecomposed:
    """Three paired bars: commanded as an outline, applied as a fill.

    `04-action-and-the-boundary.md`'s efference copy made visible -- and the
    place where that spec's route-selection falsification test is *rendered*
    rather than described.
    """

    def bars(self, small, commanded, applied, *, ticks=30, disagreement=1.0, seed=0):
        cells, edges = len(small.predicting), len(small.edges)
        panel = DomePanel(small, np.full(cells, 4.0))
        rng = np.random.default_rng(seed)
        actuator = np.array([commanded, applied], dtype=float)
        frame = None
        for tick in range(1, ticks):
            values = (
                disagreement + 0.02 * rng.standard_normal(edges)
                if np.isscalar(disagreement)
                else np.asarray(disagreement, dtype=float)
                + 0.02 * rng.standard_normal(edges)
            )
            frame = panel.frame(
                full(
                    tick,
                    cells,
                    edges,
                    error=np.abs(rng.standard_normal(cells)),
                    disagreement=values,
                    actuator=actuator,
                )
            )
        return panel, frame

    def test_saturation_reads_as_the_fill_falling_short_of_its_outline(self, small):
        panel, frame = self.bars(small, [5.0, 0.0, 0.0], [3.0, 0.0, 0.0])
        outline = painted(panel, frame, dome_panel_module._COMMANDED)
        fill = painted(panel, frame, dome_panel_module._APPLIED)
        assert outline is not None and fill is not None
        # Up from the zero line, so a shorter bar starts lower down the frame.
        assert fill[0] > outline[0]

    def test_an_unclipped_command_fills_its_outline(self, small):
        panel, frame = self.bars(small, [3.0, 0.0, 0.0], [3.0, 0.0, 0.0])
        outline = painted(panel, frame, dome_panel_module._COMMANDED)
        fill = painted(panel, frame, dome_panel_module._APPLIED)
        assert outline[0] == fill[0]

    def test_the_rows_the_strip_drew_are_the_records_own(self, small):
        panel, _frame = self.bars(small, [1.0, -2.0, 0.5], [1.0, -2.0, 0.5])
        assert np.array_equal(
            panel.torque, np.array([[1.0, -2.0, 0.5], [1.0, -2.0, 0.5]])
        )

    def test_the_stall_signature_is_an_outline_near_zero_beside_a_standing_bar(
        self, small
    ):
        """`04`'s falsification signature, rendered rather than described.

        Near-zero commanded torque with standing motor-side disagreement is the
        blend of swing-left and swing-right, which is *stay put*. It has to be
        visible **while it stands**, which is why the standing bar is drawn raw:
        on the normalised map a stall that lasts habituates to its own baseline
        and renders calm -- the consequence `10-the-demo-surface.md` accepts for
        the colour channel, and the one thing a falsification test cannot
        accept.
        """
        cells, edges = len(small.predicting), len(small.edges)
        (actuator,) = only(small, "actuator")
        rng = np.random.default_rng(5)
        panel = DomePanel(small, np.full(cells, 4.0))
        quiet = np.full(edges, 0.05)
        stalled = quiet.copy()
        for edge in small.incident[actuator]:
            stalled[edge] = 6.0

        def swing(tick, motor):
            """One tick: no torque commanded, and whatever the motor edges say."""
            return panel.frame(
                full(
                    tick,
                    cells,
                    edges,
                    error=np.abs(rng.standard_normal(cells)),
                    disagreement=motor * (1.0 + 0.02 * rng.standard_normal(edges)),
                    actuator=np.zeros((2, 3)),
                )
            )

        for tick in range(1, 300):
            swing(tick, quiet)
        arrival = swing(300, stalled)
        row = panel.boundary_marks.index(actuator)
        struck, first = panel.boundary_lit[row], standing_bar(panel, arrival)
        for tick in range(301, 600):
            frame = swing(tick, stalled)

        outline = painted(panel, frame, dome_panel_module._COMMANDED)
        # The commanded outline is one pixel high, at the zero line.
        assert outline[1] - outline[0] <= 1
        # The disagreement bar beside it stands, most of the way up the strip,
        # and stands **exactly as high as it did when the stall began**: it is
        # drawn raw, so three hundred ticks of it change nothing.
        bar = standing_bar(panel, frame)
        assert bar is not None and first is not None
        assert bar[1] - bar[0] > 0.5 * outline[0], "a bar that stands, not a stub"
        assert abs((bar[1] - bar[0]) - (first[1] - first[0])) <= 1
        # While the mark beside it has faded from the same disagreement, which
        # is the habituation this bar is drawn raw to escape.
        assert struck > 0.99
        assert panel.boundary_lit[row] < 0.75 * struck

    def test_the_onset_count_is_drawn_on_the_strip(self, small):
        """Onset is read off the strip rather than reconstructed afterward."""
        cells, edges = len(small.predicting), len(small.edges)
        panel = DomePanel(small, np.full(cells, 4.0))
        blank = panel.frame(full(1, cells, edges, disagreement=1.0), since=None)
        counted = panel.frame(full(2, cells, edges, disagreement=1.0), since=17)
        top, left, height, width = panel.motor_strip
        text = slice(top + height - dome_panel_module._FONT_HEIGHT, top + height)
        assert not np.array_equal(
            blank[text, left : left + width], counted[text, left : left + width]
        )


class TestEdgesAreThresholdedAndOffByDefault:
    """The route, shown as a route -- and not as an artifact of a constant."""

    def loud(self, small, seed=0):
        """A record whose edges carry wildly different disagreement."""
        rng = np.random.default_rng(seed)
        return np.abs(rng.standard_normal(len(small.edges))) ** 3

    def test_the_overlay_is_off_by_default(self, small):
        cells, edges = len(small.predicting), len(small.edges)
        disagreement = self.loud(small)
        panel = DomePanel(small, np.full(cells, 4.0))
        assert panel.edges is False
        drawn = panel.frame(full(1, cells, edges, disagreement=disagreement))
        overlaid = DomePanel(small, np.full(cells, 4.0), edges=True)
        lit = overlaid.frame(full(1, cells, edges, disagreement=disagreement))
        assert not np.array_equal(drawn, lit), "the toggle draws something"

    def test_the_threshold_is_this_ticks_own_scale(self, small):
        cells, edges = len(small.predicting), len(small.edges)
        panel = DomePanel(small, np.full(cells, 4.0), edges=True)
        disagreement = self.loud(small)
        panel.frame(full(1, cells, edges, disagreement=disagreement))
        assert panel.edge_threshold == pytest.approx(
            disagreement.mean() + disagreement.std()
        )
        assert set(panel.drawn_edges) == set(
            np.flatnonzero(disagreement > panel.edge_threshold).tolist()
        )
        assert 0 < len(panel.drawn_edges) < edges, "the most, not all and not none"

    def test_scaling_every_edge_draws_exactly_the_same_route(self, small):
        """A hand-set constant would make the route an artifact of the constant.

        This one has no magnitude in it at all: both terms are homogeneous in
        the disagreement, so there is nothing here that could be tuned until a
        route appeared.
        """
        cells, edges = len(small.predicting), len(small.edges)
        disagreement = self.loud(small)
        routes = []
        for factor in (1.0, 1e-4, 1e4):
            panel = DomePanel(small, np.full(cells, 4.0), edges=True)
            panel.frame(full(1, cells, edges, disagreement=disagreement * factor))
            routes.append(panel.drawn_edges)
        assert routes[0] == routes[1] == routes[2]
        assert routes[0]

    def test_an_evenly_disagreeing_tick_has_no_route_to_draw(self, small):
        cells, edges = len(small.predicting), len(small.edges)
        panel = DomePanel(small, np.full(cells, 4.0), edges=True)
        panel.frame(full(1, cells, edges, disagreement=3.0))
        assert panel.drawn_edges == ()

    def test_a_diverged_edge_does_not_empty_the_overlay(self, small):
        """A NaN threshold would draw the quietest picture of a diverged graph."""
        cells, edges = len(small.predicting), len(small.edges)
        disagreement = self.loud(small)
        disagreement[0] = np.nan
        panel = DomePanel(small, np.full(cells, 4.0), edges=True)
        panel.frame(full(1, cells, edges, disagreement=disagreement))
        assert panel.drawn_edges
        assert 0 not in panel.drawn_edges, "it cleared no threshold"

    def test_an_edge_with_no_reading_is_drawn_rather_than_dropped(self, small):
        """The loudest thing on the graph is not the one edge left out.

        Keeping a NaN out of the statistics and keeping it out of the drawing
        are two decisions, and only the first is justified: a route drawn
        *around* a divergence is the quiet picture arrived at more slowly.
        """
        cells, edges = len(small.predicting), len(small.edges)
        disagreement = self.loud(small)
        disagreement[0] = np.nan
        panel = DomePanel(small, np.full(cells, 4.0), edges=True)
        drawn = panel.frame(full(1, cells, edges, disagreement=disagreement))
        assert panel.unread_edges == (0,)
        ink = np.array(dome_panel_module._NO_READING, dtype=np.uint8)
        assert np.all(drawn == ink, axis=-1).any()


class TestNoBoundaryCellIsEverAssignedAPredictionError:
    """#94's acceptance criterion, held down three ways.

    Structurally, because the record has no row for one and the panel has no
    index from a boundary cell into the array that would carry it; and on the
    pixels, because a record whose prediction error swings wildly draws the
    same boundary marks either way.
    """

    def test_the_record_has_no_row_for_a_boundary_cell(self, small):
        cells = len(small.predicting)
        panel = DomePanel(small, np.full(cells, 1.0))
        assert set(panel.boundary_marks).isdisjoint(small.predicting)
        with pytest.raises(ValueError):
            panel.frame(record(1, np.zeros(len(small.cells))))

    def test_no_boundary_cell_indexes_the_prediction_error_map(self, small):
        cells = len(small.predicting)
        panel = DomePanel(small, np.full(cells, 1.0))
        # `_row` is the one map from a cell id into the array prediction error
        # is drawn from, and it is exactly the predicting population.
        assert {
            cell.id for cell in small.cells if panel._row.get(cell.id) is not None
        } == set(small.predicting)

    def test_a_boundary_mark_does_not_move_when_prediction_error_does(self, small):
        cells, edges = len(small.predicting), len(small.edges)
        rng = np.random.default_rng(2)
        disagreement = [1.0 + 0.1 * rng.standard_normal(edges) for _ in range(20)]
        # Two runs of the dome's own channel over the same run of the
        # boundary's: quiet, and one that lights half the dome on the last tick.
        quietly = [np.abs(rng.standard_normal(cells)) for _ in range(20)]
        loudly = [error.copy() for error in quietly]
        loudly[-1][: cells // 2] += 1e6
        image = np.zeros((small.spec.patch_grid * 4,) * 2 + (3,), dtype=np.uint8)
        image[:] = 90
        drawn = []
        for errors in (quietly, loudly):
            panel = DomePanel(small, np.full(cells, 4.0))
            for tick, (values, error) in enumerate(zip(disagreement, errors), start=1):
                frame = panel.frame(
                    full(tick, cells, edges, error=error, disagreement=values),
                    render=image,
                )
            drawn.append((panel, frame))
        (quiet, calm), (loud, burning) = drawn
        assert not np.array_equal(calm, burning), "the predicting cells did move"
        assert np.array_equal(quiet.boundary_lit, loud.boundary_lit)
        for cell_id in small.boundary:
            assert mark(quiet, calm, cell_id) == mark(loud, burning, cell_id)
        # Including the strip's bars, which are the record's own rows.
        assert np.array_equal(quiet.torque, loud.torque)


class TestTheWaysTheBoundaryMarksCouldHaveFlatteredUs:
    """One test per finding of #94's review, each one red before its fix.

    The same class of defect #93's review found, in the marks #94 added: each
    is the panel reporting something calmer or tidier than what it was handed --
    an empty slot for a divergence, a zero torque for a number that was not one,
    a calm mark for a quantity nobody captured, a baseline that moved on a
    record the panel then refused, a black arena for a render it could not read,
    a route drawn around a diverged edge, and a saturated count that reads as a
    latency.
    """

    def test_a_marks_first_reading_being_no_reading_is_still_no_reading(self, small):
        """A graph that has already diverged when the panel opens.

        Drawn as an empty slot -- *this record carried nothing* -- the one
        thing on screen would be the panel reporting that it had not been told,
        about the event it exists to show.
        """
        cells, edges = len(small.predicting), len(small.edges)
        panel = DomePanel(small, np.full(cells, 4.0))
        gone = np.full(edges, 1.0)
        touch = only(small, "touch")[0]
        (edge,) = small.incident[touch]
        gone[edge] = np.inf
        frame = panel.frame(full(1, cells, edges, disagreement=gone))
        assert mark(panel, frame, touch) == dome_panel_module._NO_READING

    def test_a_torque_that_is_not_a_number_is_not_drawn_as_zero_torque(self, small):
        """The stall signature's own half, fabricated out of no reading.

        A bar encodes its quantity in a height and no height means *not a
        number*, so a non-finite pair takes the whole column in its own colour
        -- across the zero line, which no torque bar ever crosses.
        """
        cells, edges = len(small.predicting), len(small.edges)
        panel = DomePanel(small, np.full(cells, 4.0))
        rest = panel.frame(
            full(1, cells, edges, disagreement=1.0, actuator=np.zeros((2, 3)))
        )
        lost = np.zeros((2, 3))
        lost[0, 0] = np.nan
        gone = panel.frame(full(2, cells, edges, disagreement=1.0, actuator=lost))
        top, left, height, width = panel.motor_strip
        ink = np.array(dome_panel_module._NO_READING, dtype=np.uint8)

        def unread(frame):
            region = frame[top : top + height, left : left + width]
            return np.all(region == ink, axis=-1)

        assert not unread(rest).any(), "a torque of zero is a reading"
        # A whole column of it, crossing the zero line, where the first joint's
        # pair would have stood.
        columns = np.flatnonzero(unread(gone).any(axis=0))
        assert columns.size == panel.layout.mark
        assert unread(gone)[:, columns[0]].sum() > 2

    def test_a_record_that_carries_no_disagreement_empties_the_marks_again(
        self, small
    ):
        """*Not captured* is a fact about this record, not about the run.

        A mark left standing at the calmest stop on the ramp because the last
        capture happened to leave the array out is a graph reported as agreeing
        on every edge by nobody.
        """
        cells, edges = len(small.predicting), len(small.edges)
        panel = DomePanel(small, np.full(cells, 4.0))
        panel.frame(full(1, cells, edges, disagreement=2.0))
        panel.frame(full(2, cells, edges, disagreement=3.0))
        frame = panel.frame(record(3, np.zeros(cells)))
        for cell_id in panel.boundary_marks:
            assert mark(panel, frame, cell_id) == dome_panel_module._EMPTY
        assert panel.boundary_warming_up == 0

    def test_a_refused_record_leaves_the_statistics_where_they_were(self, small):
        """A record is checked whole before any of it is observed.

        Refused half-way through, the tick is already in every cell's Welford
        statistics and the trail has already advanced -- and the corrected
        record is then refused as out of order, so the panel is permanently one
        tick off with no recovery short of a fresh one.
        """
        cells, edges = len(small.predicting), len(small.edges)
        panel = DomePanel(small, np.full(cells, 4.0))
        panel.frame(full(1, cells, edges, error=1.0, disagreement=1.0))
        settled = panel.mean.copy(), panel.glow.copy(), panel.boundary_lit.copy()
        for wrong in (
            full(2, cells, edges + 1, error=9.0, disagreement=9.0),
            full(2, cells, edges, error=9.0, actuator=np.zeros((2, 5))),
        ):
            with pytest.raises(ValueError):
                panel.frame(wrong)
        assert np.array_equal(panel.mean, settled[0])
        assert np.array_equal(panel.glow, settled[1])
        assert np.array_equal(panel.boundary_lit, settled[2])
        # And tick 2 is still there to be drawn, by the record that fits.
        panel.frame(full(2, cells, edges, error=9.0, disagreement=9.0))
        assert not np.array_equal(panel.mean, settled[0])

    def test_the_numbers_beside_the_pixels_say_no_reading_too(self, small):
        """`boundary_lit` is what a sweep reads, and zero is a reading.

        A replay of a trace saved before the disagreement array existed hands
        the panel records that carry none; a zero there reads back as a graph
        agreeing on every edge, which is the fabrication the pixels already
        refuse by drawing an empty slot.
        """
        cells, edges = len(small.predicting), len(small.edges)
        panel = DomePanel(small, np.full(cells, 4.0))
        panel.frame(full(1, cells, edges, disagreement=1.0))
        panel.frame(full(2, cells, edges, disagreement=4.0))
        assert np.isfinite(panel.boundary_lit).all()
        panel.frame(record(3, np.zeros(cells)))
        assert np.isnan(panel.boundary_lit).all()
        # And per mark, for the one whose own reading was not a number.
        gone = np.full(edges, 1.0)
        touch = only(small, "touch")[0]
        (edge,) = small.incident[touch]
        gone[edge] = np.nan
        panel.frame(full(4, cells, edges, disagreement=gone))
        row = panel.boundary_marks.index(touch)
        assert np.isnan(panel.boundary_lit[row])
        assert np.isfinite(np.delete(panel.boundary_lit, row)).all()

    def test_a_render_that_is_not_the_worlds_own_is_refused(self, small):
        """A normalised image assigned into the frame draws a black arena.

        Not an empty slot -- a picture, of an arena with the lights off, which
        is a reading of the world that nobody took.
        """
        cells = len(small.predicting)
        panel = DomePanel(small, np.full(cells, 4.0))
        scaled = np.full((small.spec.patch_grid * 4,) * 2 + (3,), 0.5, dtype=np.float32)
        with pytest.raises(ValueError, match="uint8"):
            panel.frame(record(1, np.zeros(cells)), render=scaled)

    def test_a_replay_draws_the_counter_the_live_panel_drew(self, small):
        """Onset is read off the strip, and a replay is not a second code path."""
        cells, edges = len(small.predicting), len(small.edges)
        feed = [full(tick, cells, edges, disagreement=1.0) for tick in (1, 2, 3)]
        counted = list(
            DomePanel(small, np.full(cells, 4.0)).frames(
                feed, since=lambda record: record.tick * 3
            )
        )
        plain = list(DomePanel(small, np.full(cells, 4.0)).frames(feed))
        assert len(counted) == len(plain) == 3
        assert not np.array_equal(counted[-1], plain[-1])
        one_by_one = DomePanel(small, np.full(cells, 4.0))
        by_hand = [
            one_by_one.frame(record, since=record.tick * 3) for record in feed
        ]
        for i, (fed, hand) in enumerate(zip(counted, by_hand)):
            assert np.array_equal(fed, hand), f"frame {i}"

    def test_an_onset_count_past_the_strip_is_not_quietly_saturated(self, small):
        """A plausible number that is not the reading is the worst outcome.

        The count is drawn in full and runs off the strip, which is visibly
        wrong; a harness that misses `OnsetCounter.restart()` is the realistic
        way to get a count that large, and that class documents a restore as
        invisible to it.
        """
        cells, edges = len(small.predicting), len(small.edges)
        panel = DomePanel(small, np.full(cells, 4.0))
        big = panel.frame(full(1, cells, edges, disagreement=1.0), since=999999)
        bigger = panel.frame(full(2, cells, edges, disagreement=1.0), since=12345678)
        top, _left, height, _width = panel.motor_strip
        text = slice(top + height - dome_panel_module._FONT_HEIGHT, top + height)
        assert not np.array_equal(big[text], bigger[text])
