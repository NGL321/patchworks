"""The private-component panel (ticket #95).

Four things are held down here.

* **The axis is hop distance from the sensorimotor rim**, counted on the graph
  and agreeing with the construction layout's levels over the whole
  population.
* **The readout is a measured trace.** The marks are what the run did to `H⁰`
  between two consecutive ticks -- no eigenvalue, no stored rate, and no
  per-cell factor of the panel's own.
* **It stays off the dome's marks.** The panel is its own image and its own
  numbers, and prediction error is not reachable from any of it.
* **A scatter against depth either slopes or it does not.** A flat trace and a
  sloping one are two different pictures, and a common factor on every cell is
  not.
"""

import ast
import dataclasses
import inspect

import numpy as np
import pytest
import torch

from patchworks.agent import Agent
from patchworks.body import NODE_STALK_DIM
from patchworks.graph import (
    Cell,
    CellIndex,
    CellKind,
    Dome,
    DomeSpec,
    Edge,
    EdgeKind,
    build_graph,
)
from patchworks.sandbox import PlanarPushSandbox
from patchworks.surface import (
    PANEL_HEIGHT,
    PANEL_WIDTH,
    PrivateComponentPanel,
    Recorder,
    Trace,
    hop_distance,
)
from patchworks.surface import private_component as panel_module
from patchworks.surface import record as record_module

TICKS = 24


@pytest.fixture(scope="module")
def dome():
    return build_graph()


@pytest.fixture
def env():
    world = PlanarPushSandbox(split="any")
    yield world
    world.close()


@pytest.fixture
def agent(env, dome):
    agent = Agent(env, dome=dome, generator=torch.Generator().manual_seed(0))
    observation, _info = env.reset(seed=0)
    agent.observe(observation)
    return agent


def run_watched(recorder, ticks=TICKS):
    """Drive the agent by hand, the way a viewer's event loop would."""
    records = []
    for _ in range(ticks):
        recorder.agent.tick()
        record = recorder.observe()
        if record is not None:
            records.append(record)
    return records


def private_component(sheaf):
    """`[predicting cells, n]`: the node stalk directions no incident edge carries."""
    return sheaf.evidence() * sheaf.dome.private_projection


def with_deltas(record, moved):
    """The same record, carrying deltas chosen here."""
    return dataclasses.replace(record, private_delta=np.asarray(moved, dtype=float))


def identifiers(module):
    """Every name the module's *code* mentions. Docstrings and comments are not
    code, and this readout's provenance is a question about what runs."""
    names = set()
    for node in ast.walk(ast.parse(inspect.getsource(module))):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
            names.update(alias.name for alias in node.names)
    return names


class TestTheAxisIsHopDistanceFromTheSensorimotorRim:
    def test_the_hop_count_is_the_construction_layout_s_level(self, dome):
        """Counted on the graph, checked against the index.

        A level *is* hop distance from the sensorimotor boundary
        (`CONTEXT.md`, *Level*), so the two agree -- but the panel counts hops
        rather than reading the index, and this is where that is worth
        anything: it is a check on the construction, over the whole
        population, and it would catch a taper wired to the wrong level.
        """
        depth = hop_distance(dome)
        levels = [dome.cells[cell_id].index.level for cell_id in dome.predicting]
        assert depth.tolist() == levels

    def test_the_shallowest_predicting_cells_are_one_hop_out(self, dome):
        depth = hop_distance(dome)
        assert depth.min() == 1
        assert depth.max() == len(dome.spec.vision_sides) + len(dome.spec.core_sizes)

    def test_the_apex_is_the_deepest_and_the_drive_does_not_make_it_shallow(self, dome):
        """The drive attaches at the apex, and it is not a rim to count from.

        Counting from the internal rim as well would put the apex one hop out
        -- the deepest cells drawn as the shallowest, which is the axis
        inverted on the one band the timescale claim rests on.
        """
        depth = hop_distance(dome)
        apex = max(
            cell.index.level
            for cell in dome.cells
            if cell.kind is CellKind.DRIVE
        )
        rows = [
            row
            for row, cell_id in enumerate(dome.predicting)
            if dome.cells[cell_id].index.level == apex
        ]
        assert rows, "this dome has no apex predicting cells"
        assert (depth[rows] == depth.max()).all()

    def test_the_internal_rim_is_not_a_route_between_deep_cells(self):
        """Barred from the walk, not only from its start.

        Built by hand because the default spec cannot show it: every apex cell
        is the same distance out, so the drive's bridge shortens nothing there.
        Here the far cell is four hops along the graph and would be three
        through the drive.
        """
        spec = DomeSpec()
        cells = (
            Cell(0, CellKind.PATCH, spec.patch_stalk, CellIndex(0, "vision", (0, 0))),
            Cell(1, CellKind.PREDICTING, NODE_STALK_DIM, CellIndex(1, "core", (0,))),
            Cell(2, CellKind.PREDICTING, NODE_STALK_DIM, CellIndex(2, "core", (1,))),
            Cell(3, CellKind.PREDICTING, NODE_STALK_DIM, CellIndex(3, "core", (2,))),
            Cell(4, CellKind.PREDICTING, NODE_STALK_DIM, CellIndex(4, "core", (3,))),
            Cell(
                5,
                CellKind.DRIVE,
                spec.drive_stalk,
                CellIndex(4, "internal rim", (0,)),
            ),
        )
        edges = (
            Edge(0, 0, 1, spec.boundary_m, EdgeKind.SENSORY),
            Edge(1, 1, 2, spec.interior_m, EdgeKind.INTERIOR),
            Edge(2, 2, 3, spec.interior_m, EdgeKind.INTERIOR),
            Edge(3, 3, 4, spec.interior_m, EdgeKind.INTERIOR),
            Edge(4, 5, 1, spec.drive_m, EdgeKind.DRIVE),
            Edge(5, 5, 4, spec.drive_m, EdgeKind.DRIVE),
        )
        # `_assemble` is the only constructor for a graph this shape; the
        # builder generates the dome and nothing else.
        strung_out = Dome._assemble(spec, cells, edges)
        assert hop_distance(strung_out).tolist() == [1, 2, 3, 4]

    def test_the_axis_is_fixed_at_construction(self, agent, dome):
        panel = PrivateComponentPanel(dome)
        records = run_watched(Recorder(agent))
        first, last = panel.scatter(records[0]), panel.scatter(records[-1])
        assert np.array_equal(first.depth, last.depth)
        assert np.array_equal(first.depth, PrivateComponentPanel(dome).depth)
        # Shared by every scatter, so it is sealed rather than trusted.
        with pytest.raises(ValueError):
            first.depth[0] = 99


class TestTheReadoutIsAMeasuredTrace:
    def test_the_marks_are_what_one_tick_did_to_the_private_component(
        self, agent, dome
    ):
        """Re-derived from outside: the norm of what one tick did to `H⁰`."""
        recorder = Recorder(agent, every=1)
        panel = PrivateComponentPanel(dome)
        agent.tick()
        recorder.observe()  # primes
        before = private_component(agent.sheaf).clone()
        agent.tick()
        record = recorder.observe()
        after = private_component(agent.sheaf)
        expected = (after - before).norm(dim=-1)
        assert np.allclose(panel.scatter(record).moved, expected.numpy())
        assert not torch.equal(before, after)

    def test_the_projection_is_fixed_and_the_reading_is_per_tick(self, agent, dome):
        """`A fixed projection computed per tick.` Both halves, together.

        The private component is the node-stalk directions masked out on every
        incident edge, known at construction (`05-timescales.md`,
        *Demonstrating it*): the mask closes and never re-opens, so what is
        read is the same subspace on every tick of a run that is learning the
        whole time. What is *read off* it moves every tick.
        """
        projection = dome.private_projection.clone()
        moved = [
            PrivateComponentPanel(dome).scatter(record).moved
            for record in run_watched(Recorder(agent))
        ]
        assert torch.equal(dome.private_projection, projection)
        assert not np.array_equal(moved[0], moved[-1])

    def test_the_panel_applies_no_factor_of_its_own(self, agent, dome):
        """A per-cell factor is where a stored rate would enter. There is none."""
        record = run_watched(Recorder(agent))[-1]
        marks = PrivateComponentPanel(dome).scatter(record)
        assert np.array_equal(marks.moved, record.private_delta)

    def test_a_mark_is_a_trace_and_not_a_cell_attribute(self, agent, dome):
        """The same cell reads differently on different ticks.

        A rate -- an eigenvalue of a region, a `tau` kept per cell, anything
        selected at construction -- would be the same number every tick, and
        the panel would be drawing the dome's construction rather than its
        run.
        """
        panel = PrivateComponentPanel(dome)
        moved = np.stack(
            [panel.scatter(record).moved for record in run_watched(Recorder(agent))]
        )
        assert moved.shape[0] > 1
        assert (moved.max(axis=0) > moved.min(axis=0)).any()

    @pytest.mark.parametrize("module", [panel_module, record_module])
    def test_no_eigenvalue_and_no_stored_rate_is_reachable(self, module):
        """`It was never an eigenvalue, and it must not become one.`

        The readout is a difference of two measured node stalks, so nothing
        between the sheaf and the mark may decompose anything or consult a rate
        that was measured once and kept. Read off the code rather than off the
        prose: an eigendecomposition, a spectral radius or an import of the
        modules where a cell's `tau` is measured and banded
        (:mod:`patchworks.bias_selection`) or a rate is set by hand
        (:mod:`patchworks.timescale`).
        """
        mentioned = identifiers(module)
        for banned in (
            "eig",
            "eigh",
            "eigvals",
            "eigvalsh",
            "svd",
            "spectrum",
            "spectral_radius",
            "tau",
            "rho",
            "bias_selection",
            "patchworks.bias_selection",
            "timescale",
            "patchworks.timescale",
            "ClockDivisor",
            "Measurement",
        ):
            assert banned not in mentioned, f"{module.__name__} reaches {banned}"


class TestADivergedRunIsNotDrawnAsAQuietOne:
    """The failure that flatters is the dangerous one.

    A NaN peak sends every mark back to the baseline and an `inf` gets there
    by the other arithmetic, so an unguarded panel draws a diverged run as a
    graph perfectly at rest -- the most reassuring picture it has, and the one
    thing this panel exists to be unable to say by accident.
    """

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_a_non_finite_reading_is_refused(self, agent, dome, bad):
        record = run_watched(Recorder(agent))[-1]
        panel = PrivateComponentPanel(dome)
        moved = np.asarray(record.private_delta, dtype=float).copy()
        moved[3] = bad
        with pytest.raises(ValueError, match="non-finite"):
            panel.scatter(with_deltas(record, moved))
        with pytest.raises(ValueError, match="non-finite"):
            panel.draw(with_deltas(record, moved))

    def test_it_would_otherwise_draw_exactly_the_at_rest_picture(self, agent, dome):
        """Why the refusal is not fussiness: the two pictures were identical."""
        record = run_watched(Recorder(agent))[-1]
        panel = PrivateComponentPanel(dome)
        sloping = np.asarray(1.0 / panel.depth, dtype=float)
        at_rest = panel.draw(with_deltas(record, np.zeros(panel.depth.shape)))
        drawn = panel.draw(with_deltas(record, sloping))
        assert not np.array_equal(drawn, at_rest)
        # The same trace with one cell diverged, drawn the way an unguarded
        # panel would have: back on the baseline, indistinguishable from rest.
        poisoned = sloping.copy()
        poisoned[3] = np.nan
        peak = float(poisoned.max())
        assert not peak > 0  # NaN, so the zeros branch would have fired
        with pytest.raises(ValueError):
            panel.draw(with_deltas(record, poisoned))


class TestItStaysOffTheDomeSMarks:
    def test_the_panel_is_its_own_image(self, agent, dome):
        record = run_watched(Recorder(agent))[-1]
        panel = PrivateComponentPanel(dome)
        image = panel.draw(record)
        assert image.shape == (PANEL_HEIGHT, PANEL_WIDTH, 3)
        assert image.dtype == np.uint8

    def test_prediction_error_is_not_reachable_from_the_panel(self):
        """The other array of the record, and the dome's channel (#93).

        Two quantities on one mark makes neither readable, which is why the
        panel is drawn separately at all. Here that is structural: the word
        does not appear in the panel's code, so there is nothing to fold in.
        """
        assert "prediction_error" not in identifiers(panel_module)

    def test_the_marks_do_not_move_when_prediction_error_does(self, agent, dome):
        record = run_watched(Recorder(agent))[-1]
        panel = PrivateComponentPanel(dome)
        lit = dataclasses.replace(
            record, prediction_error=record.prediction_error * 1000 + 7
        )
        assert np.array_equal(panel.scatter(lit).moved, panel.scatter(record).moved)
        assert np.array_equal(panel.draw(lit), panel.draw(record))

    def test_a_panel_holds_no_agent_no_sheaf_and_no_world(self, dome):
        """Nothing a cell's computation is handed can reach one, because a
        panel holds nothing but a graph's arithmetic."""
        panel = PrivateComponentPanel(dome)
        for name, held in vars(panel).items():
            assert isinstance(held, (int, tuple, np.ndarray)), f"{name} is {held!r}"


class TestASlopeOrNotASlope:
    def test_a_flat_trace_and_a_sloping_one_are_different_pictures(self, agent, dome):
        """The failure `08` counts even when every recovery looks perfect."""
        record = run_watched(Recorder(agent))[-1]
        panel = PrivateComponentPanel(dome)
        flat = panel.draw(with_deltas(record, np.ones(panel.depth.shape)))
        sloping = panel.draw(with_deltas(record, 1.0 / panel.depth))
        assert not np.array_equal(flat, sloping)

    def test_a_sloping_trace_draws_its_deep_marks_below_its_shallow_ones(
        self, agent, dome
    ):
        record = run_watched(Recorder(agent))[-1]
        panel = PrivateComponentPanel(dome)
        image = panel.draw(with_deltas(record, 1.0 / panel.depth))
        ink = np.flatnonzero((image == panel_module.INK).all(axis=-1).any(axis=0))
        rows = [
            np.flatnonzero((image[:, x] == panel_module.INK).all(axis=-1)).min()
            for x in ink
        ]
        # Top of the image is row zero, so a falling trace is a rising row
        # index. Read left to right, the rim first.
        assert rows == sorted(rows)

    def test_a_flat_trace_draws_every_mark_at_one_height(self, agent, dome):
        record = run_watched(Recorder(agent))[-1]
        panel = PrivateComponentPanel(dome)
        image = panel.draw(with_deltas(record, np.full(panel.depth.shape, 0.4)))
        rows = np.flatnonzero((image == panel_module.INK).all(axis=-1).any(axis=1))
        assert rows.max() - rows.min() <= 2  # the mark is three pixels tall

    def test_a_common_factor_moves_no_mark(self, agent, dome):
        """The vertical scale is the tick's own, so the picture reads shape.

        What that costs is the absolute magnitude, and what it buys is a panel
        whose picture is not an artifact of a hand-set constant. The magnitude
        is still in the scatter.
        """
        record = run_watched(Recorder(agent))[-1]
        panel = PrivateComponentPanel(dome)
        # A power of two, so that the factor divides back out exactly and the
        # assertion is about the panel rather than about rounding.
        louder = with_deltas(record, record.private_delta * 16.0)
        assert np.array_equal(panel.draw(louder), panel.draw(record))
        assert not np.array_equal(
            panel.scatter(louder).moved, panel.scatter(record).moved
        )

    def test_a_tick_on_which_nothing_moved_draws_on_the_baseline(self, agent, dome):
        record = run_watched(Recorder(agent))[-1]
        panel = PrivateComponentPanel(dome)
        image = panel.draw(with_deltas(record, np.zeros(panel.depth.shape)))
        rows = np.flatnonzero((image == panel_module.INK).all(axis=-1).any(axis=1))
        baseline = np.flatnonzero((image == panel_module.AXIS).all(axis=-1).any(axis=1))
        assert baseline.size == 1
        assert abs(int(rows.mean()) - baseline[0]) <= 1


class TestOnePanelTwoFeeds:
    def test_live_and_replay_are_one_code_path(self, agent, dome, tmp_path):
        recorder = Recorder(agent)
        run_watched(recorder)
        panel = PrivateComponentPanel(dome)
        live = list(panel.frames(recorder.trace))
        replay = list(panel.frames(Trace.load(recorder.trace.save(tmp_path / "run"))))
        assert live and len(live) == len(replay)
        assert all(np.array_equal(a, b) for a, b in zip(live, replay))

    def test_the_size_is_chosen_when_drawing(self, agent, dome):
        record = run_watched(Recorder(agent))[-1]
        assert PrivateComponentPanel(dome, width=120, height=60).draw(record).shape == (
            60,
            120,
            3,
        )

    @pytest.mark.parametrize("size", [0, -1, 15, 4.0, True, "wide"])
    def test_a_panel_too_small_to_draw_axes_in_is_refused(self, dome, size):
        with pytest.raises(ValueError, match="pixels"):
            PrivateComponentPanel(dome, width=size)

    def test_a_record_from_another_dome_is_refused(self, agent, dome):
        record = run_watched(Recorder(agent))[-1]
        panel = PrivateComponentPanel(dome)
        with pytest.raises(ValueError, match="row order"):
            panel.scatter(with_deltas(record, np.zeros(3)))
