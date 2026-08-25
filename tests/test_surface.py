"""The tick record, and one renderer (ticket #92).

Four things are held down here, and the third is the one the whole demo
surface rests on.

* **The record is the contract plus the arrays.** A tick holds
  `03-the-sandbox.md`'s snapshot/restore state -- `mjSTATE_INTEGRATION`, the
  task, the sampler's RNG -- plus per-cell prediction error and
  `‖Δ private‖`, plus (from #94) per-edge disagreement and the actuator's
  commanded/applied pair, plus the markers the hands drop. Not a new format:
  what comes back off disk restores a world, bit for bit.
* **One renderer, two feeds.** Live is a recorder's `watch()`; replay is a
  `Trace` off disk; the renderer cannot tell which, and the frames are
  identical.
* **Switching the surface off changes no trajectory.** Asserted bit for bit
  over a run with the world in it, and structurally as well: nothing a cell's
  computation is handed can reach a recorder, a trace or a renderer.
* **The live budget is near zero.** A 10 Hz capture does not measurably slow
  the run it is watching.
"""

import dataclasses
import json
import types

import numpy as np
import pytest
import torch

from patchworks.agent import Agent
from patchworks.graph import build_graph
from patchworks.sandbox import PlanarPushSandbox, Snapshot, restore, snapshot
from patchworks.sandbox.env import CONTROL_HZ
from patchworks.surface import (
    CAPTURE_EVERY,
    CAPTURE_HZ,
    Event,
    EventKind,
    Recorder,
    Renderer,
    TickRecord,
    Trace,
)

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


@pytest.fixture
def agent(env, dome):
    return started(env, dome)


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


class TestTheRecordIsTheContractPlusTheArrays:
    def test_a_record_is_the_contract_the_arrays_and_the_markers(self):
        """Literally: the fields, and no others. Not a new format.

        #94 added the two the boundary band's marks are drawn from -- per-edge
        disagreement, and the actuator's commanded/applied pair. They are the
        same kind of thing as the two already here (privileged state, read,
        never fed back) at the same order of size, and they are here rather
        than handed to the panel beside a record because a quantity that
        reached the panel any other way would draw live and not off disk.
        """
        assert [field.name for field in dataclasses.fields(TickRecord)] == [
            "tick",
            "state",
            "prediction_error",
            "private_delta",
            "disagreement",
            "actuator",
            "events",
        ]

    def test_the_state_is_the_snapshot_restore_contract_unchanged(self, agent):
        record = run_watched(Recorder(agent))[0]
        assert isinstance(record.state, Snapshot)
        # #81's contract as it stands, not a copy of it that could drift.
        assert [field.name for field in dataclasses.fields(Snapshot)] == [
            "physics",
            "task",
            "rng",
        ]
        taken = snapshot(agent.env)
        assert record.state.physics.shape == taken.physics.shape
        assert record.state.rng.keys() == taken.rng.keys()

    def test_the_two_arrays_are_one_float_per_predicting_cell(self, agent, dome):
        recorder = Recorder(agent)
        record = run_watched(recorder)[0]
        cells = len(dome.predicting)
        assert record.prediction_error.shape == (cells,)
        assert record.private_delta.shape == (cells,)

    def test_disagreement_is_one_float_per_edge_not_per_cell(self, agent, dome):
        """Edge-owned and spatial, where prediction error is cell-owned."""
        record = run_watched(Recorder(agent))[0]
        assert record.disagreement.shape == (len(dome.edges),)
        assert np.all(record.disagreement >= 0.0)

    def test_the_actuator_row_pair_is_what_the_cell_holds(self, agent):
        """Commanded and applied, off the actuator boundary cell's own stalk.

        Not a second reading of the torque: the six numbers the efference copy
        is made of, so a saturating command is a row that parts from its
        partner and nothing else has to be consulted to see it.
        """
        # Stopped on the tick the last record was taken from, so the cell's
        # stalk still holds what that record says it did.
        record = run_watched(Recorder(agent), ticks=CAPTURE_EVERY)[-1]
        assert record.tick == agent.sheaf.ticks
        assert record.actuator.shape == (2, agent.joints)
        stalk = agent.sheaf.stalk(agent.actuator_cell).numpy()
        assert np.array_equal(record.actuator.reshape(-1), stalk)

    def test_the_actuator_rows_are_not_a_view_on_the_running_stalk(self, agent):
        """A record is a reading, and a reading that moves is not one."""
        recorder = Recorder(agent)
        record = run_watched(recorder)[0]
        taken = record.actuator.copy()
        run_watched(recorder, ticks=CAPTURE_EVERY * 2)
        assert np.array_equal(record.actuator, taken)

    def test_a_record_carries_no_frame(self, agent):
        recorder = Recorder(agent)
        record = run_watched(recorder)[0]
        # A trace is state, not frames: nothing in a record is an image, which
        # is what lets the capture resolution be chosen at render time.
        for value in vars(record).values():
            assert not (
                isinstance(value, np.ndarray)
                and (value.dtype == np.uint8 or value.ndim >= 3)
            )

    def test_a_record_restores_the_world_it_was_taken_from(self, agent, dome):
        recorder = Recorder(agent)
        record = run_watched(recorder)[-1]
        elsewhere = PlanarPushSandbox(split="any")
        try:
            elsewhere.reset(seed=7, options={"reset_arm": True})
            restore(elsewhere, record.state)
            assert np.array_equal(snapshot(elsewhere).physics, record.state.physics)
            assert elsewhere.np_random.bit_generator.state == record.state.rng
        finally:
            elsewhere.close()


class TestWhatTheTwoArraysHold:
    def test_prediction_error_is_the_tick_s_own(self, agent):
        recorder = Recorder(agent)
        # Stopped on a capture tick, so the sheaf still holds exactly what the
        # last record was read off.
        run_watched(recorder, CAPTURE_EVERY)
        sheaf = agent.sheaf
        expected = (sheaf.prediction - sheaf.evidence()).norm(dim=-1)
        assert np.array_equal(recorder.trace[-1].prediction_error, expected.numpy())

    def test_the_delta_is_the_private_component_moving(self, agent):
        """Re-derived from outside: the norm of what one tick did to `H⁰`."""
        recorder = Recorder(agent, every=1)
        agent.tick()
        recorder.observe()  # primes
        before = private_component(agent.sheaf).clone()
        agent.tick()
        record = recorder.observe()
        after = private_component(agent.sheaf)
        expected = (after - before).norm(dim=-1)
        assert np.allclose(record.private_delta, expected.numpy())
        assert not torch.equal(before, after)

    def test_a_cell_with_no_private_dimension_never_moves(self, agent, dome):
        record = run_watched(Recorder(agent))[-1]
        none = (dome.private_dimensions == 0).numpy()
        assert none.any(), "this dome has no cell without private dimension"
        assert (record.private_delta[none] == 0.0).all()

    def test_a_cell_with_private_dimension_does_move(self, agent, dome):
        record = run_watched(Recorder(agent))[-1]
        some = (dome.private_dimensions > 0).numpy()
        assert some.any()
        assert (record.private_delta[some] > 0.0).any()

    def test_the_delta_is_tick_to_tick_and_not_capture_to_capture(self, dome):
        """Two capture rates, one quantity.

        `‖Δ private‖` is a difference between consecutive ticks. If it were
        differenced between captures instead, the number would change meaning
        with the display's rate -- so the same tick read at two rates has to
        give the same number.
        """
        recorders = []
        worlds = [PlanarPushSandbox(split="any") for _ in range(2)]
        try:
            for world, every in zip(worlds, (1, CAPTURE_EVERY)):
                recorder = Recorder(started(world, dome), every=every)
                run_watched(recorder, TICKS)
                recorders.append(recorder)
        finally:
            for world in worlds:
                world.close()
        fast, slow = recorders
        by_tick = {r.tick: r.private_delta for r in fast.trace}
        for record in slow.trace:
            assert np.allclose(by_tick[record.tick], record.private_delta)


class TestTheCaptureRate:
    def test_the_default_rate_is_the_spec_s_ten_hertz(self):
        assert CAPTURE_HZ == 10.0
        assert CAPTURE_EVERY == round(CONTROL_HZ / CAPTURE_HZ)

    def test_one_tick_in_every_is_kept(self, agent):
        recorder = Recorder(agent)
        records = run_watched(recorder, TICKS)
        assert [r.tick for r in records] == list(
            range(CAPTURE_EVERY, TICKS + 1, CAPTURE_EVERY)
        )

    def test_the_first_tick_seen_primes_the_delta_rather_than_faking_one(self, agent):
        recorder = Recorder(agent, every=1)
        agent.tick()
        assert recorder.observe() is None
        agent.tick()
        assert recorder.observe() is not None

    def test_a_skipped_tick_is_refused(self, agent):
        recorder = Recorder(agent)
        agent.tick()
        recorder.observe()
        agent.tick()
        agent.tick()
        with pytest.raises(ValueError, match="consecutive"):
            recorder.observe()

    def test_a_gap_can_be_taken_deliberately_without_losing_the_trace(self, agent):
        """`reprime()` is the way past the refusal that keeps what was captured."""
        recorder = Recorder(agent, every=1)
        run_watched(recorder, 3)
        captured = len(recorder.trace)
        assert captured > 0
        agent.tick()
        agent.tick()
        with pytest.raises(ValueError, match="reprime"):
            recorder.observe()
        recorder.reprime()
        agent.tick()
        assert recorder.observe() is None  # primes again, states no delta it cannot
        agent.tick()
        assert recorder.observe() is not None
        assert len(recorder.trace) == captured + 1

    def test_reprime_keeps_the_markers_waiting(self, agent):
        recorder = Recorder(agent, every=1)
        run_watched(recorder, 2)
        fired = recorder.mark(EventKind.DISTURB_ARM, 1, 0.25)
        recorder.reprime()
        assert recorder.pending == (fired,)

    @pytest.mark.parametrize("every", [0, -1, 1.5, True])
    def test_a_capture_keeps_one_tick_in_a_whole_number(self, agent, every):
        with pytest.raises(ValueError, match="every"):
            Recorder(agent, every=every)


class TestTheEventMarkers:
    def test_a_marker_carries_the_tick_the_hand_fired_on(self, agent):
        recorder = Recorder(agent)
        run_watched(recorder, CAPTURE_EVERY + 2)
        fired = recorder.mark(EventKind.PERTURB, 1, 0.1, -0.2)
        assert fired.tick == agent.sheaf.ticks
        agent.tick()
        record = recorder.observe()
        assert record.events == (fired,)
        # The record is a later tick than the marker, and the marker keeps its
        # own: onset latency is counted in ticks, not in captures.
        assert record.tick != fired.tick

    def test_a_tick_that_fires_a_marker_is_always_captured(self, agent):
        recorder = Recorder(agent)
        run_watched(recorder, 1)
        assert not recorder.trace
        recorder.mark(EventKind.DISTURB_ARM, 0, 0.5)
        agent.tick()
        record = recorder.observe()
        assert record is not None and record.tick % CAPTURE_EVERY != 0

    def test_a_marker_from_a_gesture_survives_being_written(self, agent, tmp_path):
        """A hand bound to a gesture hands over what it picked off an array.

        Left as it arrived, a `np.float32` is accepted by `mark()` and then
        refused by the file at the end of the run, taking the whole trace with
        it. It is coerced where it arrives instead.
        """
        recorder = Recorder(agent)
        fired = recorder.mark(
            EventKind.PERTURB, np.int64(1), np.float32(0.1), np.float64(-0.2)
        )
        assert all(type(value) is float for value in fired.detail)
        run_watched(recorder, CAPTURE_EVERY)
        back = Trace.load(recorder.trace.save(tmp_path / "run"))
        assert back[0].events == (fired,)

    def test_a_marker_not_yet_captured_is_visible(self, agent):
        """The one tick in which a marker exists and no record holds it."""
        recorder = Recorder(agent)
        run_watched(recorder, 1)
        fired = recorder.mark(EventKind.RETARGET, 0, 2)
        assert recorder.pending == (fired,)
        agent.tick()
        recorder.observe()
        assert recorder.pending == ()

    def test_the_markers_are_the_three_hands(self):
        assert {kind.value for kind in EventKind} == {
            "disturb_arm",
            "perturb",
            "retarget",
        }

    def test_a_hand_nobody_has_is_refused(self, agent):
        with pytest.raises(ValueError):
            Recorder(agent).mark("wave")


class TestSwitchingItOffChangesNoTrajectory:
    def test_a_watched_run_is_bit_identical_to_an_unwatched_one(self, dome):
        """The acceptance criterion, with the world in it.

        Two identically constructed agents on two identically arranged worlds,
        one watched and one not, ticked side by side: every tensor the graph
        holds, every torque the arm was given, and the whole physics state,
        bit for bit, on every tick.
        """
        plain_env = PlanarPushSandbox(split="any")
        watched_env = PlanarPushSandbox(split="any")
        try:
            plain = started(plain_env, dome)
            watched = started(watched_env, dome)
            recorder = Recorder(watched)
            for tick in range(TICKS):
                left = plain.tick()
                right = watched.tick()
                recorder.observe()
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
            assert len(recorder.trace) == TICKS // CAPTURE_EVERY
        finally:
            plain_env.close()
            watched_env.close()

    def test_rendering_mid_run_leaves_the_live_world_alone(self, agent):
        """A renderer restores into its own world, never the one being lived in."""
        recorder = Recorder(agent)
        record = run_watched(recorder)[0]
        before = snapshot(agent.env)
        with Renderer(size=32) as renderer:
            renderer.frame(record)
        after = snapshot(agent.env)
        assert np.array_equal(before.physics, after.physics)

    def test_the_recorder_never_puts_anything_on_a_tape(self, agent):
        recorder = Recorder(agent)
        run_watched(recorder)
        agent.sheaf.assert_no_tape()


class TestTheSurfaceIsNotReachableFromAnyCellComputation:
    """The structural half of *no cell reads anything the surface computes*.

    A rule, a body, a map and a cell are handed the sheaf, the dome, the body,
    the biases and the maps, because those are the only objects there are. If
    no recorder, trace or renderer is reachable from any of them, nothing
    inside the graph can consult one -- whatever it wants.
    """

    SURFACE = (Recorder, Trace, Renderer, TickRecord, Event)

    def walk(self, root, limit=200_000):
        seen, stack, found = set(), [("<root>", root)], []
        while stack:
            path, obj = stack.pop()
            if id(obj) in seen:
                continue
            seen.add(id(obj))
            assert len(seen) < limit, "the object walk ran away"
            if isinstance(obj, self.SURFACE):
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
        return found

    def test_nothing_the_graph_holds_can_reach_the_surface(self, agent):
        recorder = Recorder(agent)
        run_watched(recorder)
        sheaf = agent.sheaf
        for root in (sheaf, sheaf.dome, sheaf.body, sheaf.biases, sheaf.maps):
            assert self.walk(root) == []

    def test_the_walk_would_catch_a_recorder_stashed_on_the_sheaf(self, agent):
        agent.sheaf.instrument = Recorder(agent)
        assert self.walk(agent.sheaf) == ["<root>.instrument"]


class TestTheFile:
    def test_the_file_holds_the_contract_and_the_arrays_and_nothing_else(
        self, agent, tmp_path
    ):
        recorder = Recorder(agent)
        run_watched(recorder)
        path = recorder.trace.save(tmp_path / "run")
        with np.load(path) as stored:
            assert set(stored) == {
                "tick",
                "physics",
                "puck_xy",
                "puck_theta",
                "goal",
                "prediction_error",
                "private_delta",
                "disagreement",
                "actuator",
                "rng",
                "events",
            }
            # No frames: nothing in the file is an image.
            assert not any(stored[name].dtype == np.uint8 for name in stored)

    def test_a_trace_survives_the_round_trip_exactly(self, agent, tmp_path):
        recorder = Recorder(agent)
        recorder.mark(EventKind.RETARGET, 2, 1)
        run_watched(recorder)
        path = recorder.trace.save(tmp_path / "run")
        back = Trace.load(path)
        assert len(back) == len(recorder.trace)
        for read, written in zip(back, recorder.trace):
            assert read.tick == written.tick
            assert np.array_equal(read.state.physics, written.state.physics)
            assert np.array_equal(read.state.task.puck_xy, written.state.task.puck_xy)
            assert np.array_equal(
                read.state.task.puck_theta, written.state.task.puck_theta
            )
            assert read.state.task.pair == written.state.task.pair
            assert read.state.rng == written.state.rng
            assert np.array_equal(read.prediction_error, written.prediction_error)
            assert np.array_equal(read.private_delta, written.private_delta)
            assert np.array_equal(read.disagreement, written.disagreement)
            assert np.array_equal(read.actuator, written.actuator)
            assert read.events == written.events

    def test_the_rng_survives_at_full_width(self, agent, tmp_path):
        """PCG64's state is a 128-bit integer, and a lossy trip is a replay that
        diverges rather than an error."""
        recorder = Recorder(agent)
        run_watched(recorder)
        path = recorder.trace.save(tmp_path / "run")
        with np.load(path) as stored:
            written = json.loads(str(stored["rng"].item()))
        assert written[0] == recorder.trace[0].state.rng
        assert written[0]["state"]["state"] > 2**64

    def test_a_record_off_disk_restores_a_world(self, agent, tmp_path):
        recorder = Recorder(agent)
        run_watched(recorder)
        path = recorder.trace.save(tmp_path / "run")
        record = Trace.load(path)[-1]
        elsewhere = PlanarPushSandbox(split="any")
        try:
            elsewhere.reset(seed=3, options={"reset_arm": True})
            restore(elsewhere, record.state)
            assert np.array_equal(snapshot(elsewhere).physics, record.state.physics)
            assert elsewhere.np_random.bit_generator.state == record.state.rng
        finally:
            elsewhere.close()

    def test_an_empty_trace_survives_the_round_trip(self, tmp_path):
        assert len(Trace.load(Trace().save(tmp_path / "nothing"))) == 0


class TestOneRenderer:
    def test_live_and_replay_are_one_code_path(self, agent, tmp_path):
        """The acceptance criterion: the renderer cannot tell the two feeds apart.

        The same run is drawn twice -- once from the records as they come off
        the live recorder, once from the file they were written to -- and the
        frames are identical arrays.
        """
        recorder = Recorder(agent)
        live_feed = run_watched(recorder)
        path = recorder.trace.save(tmp_path / "run")
        with Renderer(size=48) as renderer:
            live = list(renderer.frames(live_feed))
            replayed = list(renderer.frames(Trace.load(path)))
        assert len(live) == len(recorder.trace) > 0
        for drawn, redrawn in zip(live, replayed):
            assert np.array_equal(drawn, redrawn)

    def test_a_live_run_can_be_drawn_as_it_happens(self, agent):
        """`watch()` is the live feed, and it is the same argument `frames()` takes."""
        recorder = Recorder(agent)
        with Renderer(size=32) as renderer:
            frames = list(renderer.frames(recorder.watch(TICKS, seed=0)))
        assert len(frames) == TICKS // CAPTURE_EVERY
        assert all(frame.shape == (32, 32, 3) for frame in frames)

    def test_a_stretch_of_a_trace_feeds_the_renderer_like_a_whole_one(self, agent):
        """The README's front door is two short loops off one run."""
        recorder = Recorder(agent)
        run_watched(recorder)
        stretch = recorder.trace[:2]
        assert isinstance(stretch, Trace) and len(stretch) == 2
        with Renderer(size=32) as renderer:
            assert len(list(renderer.frames(stretch))) == 2

    def test_the_resolution_is_chosen_when_rendering(self, agent):
        """One record, two sizes. Nothing about the recording changes."""
        record = run_watched(Recorder(agent))[-1]
        with Renderer(size=64) as small, Renderer(size=192) as large:
            assert small.frame(record).shape == (64, 64, 3)
            assert large.frame(record).shape == (192, 192, 3)

    def test_the_scene_is_re_rendered_from_the_state(self, agent):
        """The frame is the world that record was taken from, not an approximation.

        At the observation's own size the offscreen re-render reproduces the
        picture the agent was actually given on that tick -- to within a
        rasterisation least-significant bit, on a handful of edge pixels, and
        not bit for bit. **That is a property of two GL contexts, not of the
        record**: the same renderer redraws the same record identically, and
        the state it draws from is restored bit-exactly (asserted above). A
        display is allowed to differ in the last bit of an edge pixel; a
        trajectory is not, which is what
        `TestSwitchingItOffChangesNoTrajectory` holds down separately.
        """
        recorder = Recorder(agent)
        outcome = None
        for _ in range(CAPTURE_EVERY):
            outcome = agent.tick()
            record = recorder.observe()
        with Renderer(size=agent.env.image_size) as renderer:
            drawn = renderer.frame(record)
            again = renderer.frame(record)
        live = outcome.observation["image"]
        assert np.array_equal(drawn, again)
        assert drawn.shape == live.shape and drawn.max() > 0
        difference = np.abs(drawn.astype(int) - live.astype(int))
        assert difference.max() <= 1
        assert (difference.sum(axis=-1) > 0).mean() < 0.01

    @pytest.mark.parametrize("size", [0, -1, 2.5, True])
    def test_a_resolution_is_a_positive_number_of_pixels(self, size):
        with pytest.raises(ValueError, match="pixels"):
            Renderer(size=size)

    def test_a_resolution_the_arena_cannot_draw_is_refused_at_construction(self):
        """Not on the first frame, which a live feed reaches only mid-run."""
        world = PlanarPushSandbox(split="any")
        try:
            limit = int(world.model.vis.global_.offwidth)
        finally:
            world.close()
        with pytest.raises(ValueError, match="framebuffer"):
            Renderer(size=limit + 1)


class TestTheLiveBudget:
    """A 10 Hz capture decimated off a state log does not threaten the run.

    What is measured is the capture's **own** cost against the cost of the tick
    it is watching, both taken in the same loop on the same run. That is the
    quantity `10-the-demo-surface.md`'s budget argument is about -- *it keeps
    the live budget near zero* -- and it is the only form of this measurement
    that survives the machine it runs on: a watched run timed against an
    unwatched one measures the arm wandering into contact and whatever else the
    laptop is doing at least as much as it measures the recorder.

    The statistic is the **median**, for the same reason: a laptop running the
    rest of a build alongside this stalls a tick now and then, and a total lets
    one such stall decide the answer.

    What the assertion catches is a capture that has started doing real work
    per tick -- a render, a copy of the graph, a write to disk. Each of those
    is an order of magnitude, not the few per cent the reading actually shows.
    """

    TICKS = 60

    def test_the_capture_is_a_small_fraction_of_the_tick_it_watches(
        self, dome, capsys
    ):
        import statistics
        import time

        env = PlanarPushSandbox(split="any")
        try:
            agent = started(env, dome)
            recorder = Recorder(agent)
            for _ in range(5):  # warm the renderer and the graph up
                agent.tick()
                recorder.observe()
            ticking, capturing = [], []
            for _ in range(self.TICKS):
                start = time.perf_counter()
                agent.tick()
                middle = time.perf_counter()
                recorder.observe()
                ticking.append(middle - start)
                capturing.append(time.perf_counter() - middle)
        finally:
            env.close()
        tick = statistics.median(ticking)
        capture = statistics.median(capturing)
        # The capture really did run over the measured stretch.
        assert len(recorder.trace) >= self.TICKS // CAPTURE_EVERY
        with capsys.disabled():
            print(
                f"\n  live tick {1e6 * tick:.0f} us, "
                f"{CAPTURE_HZ:.0f} Hz capture {1e6 * capture:.0f} us "
                f"({100 * capture / tick:.1f}% of a tick)"
            )
        assert capture < 0.1 * tick
