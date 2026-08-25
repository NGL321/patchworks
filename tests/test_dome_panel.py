"""The dome panel: bands, prediction error, and the trail (ticket #93).

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


def record(tick, error, private=None):
    """A tick record carrying `error`, and whatever `private` the test wants."""
    error = np.asarray(error, dtype=float)
    return TickRecord(
        tick=tick,
        state=STATE,
        prediction_error=error,
        private_delta=(
            np.zeros_like(error) if private is None else np.asarray(private, dtype=float)
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

    def test_a_boundary_cells_slot_is_laid_out_and_left_empty(self, small):
        """Boundary cells make no prediction (ADR-0006); colouring them would be a lie."""
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
        for cell_id in (small.predicting[0], small.predicting[-1], small.boundary[0]):
            top, left, size = panel.rect(cell_id)
            lattice_top, lattice_left, lattice_size = panel.layout.rect(cell_id)
            assert size == lattice_size
            assert top == lattice_top + (panel.height - panel.layout.height)
            assert left == lattice_left + (panel.width - panel.layout.width) // 2
            assert 0 <= top and top + size <= panel.height
            assert 0 <= left and left + size <= panel.width

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
