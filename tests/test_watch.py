"""`patchworks watch`: the panel, in a window (ticket #122).

**No test here opens a window**, and that is a constraint rather than an
omission. What can be held down without one is everything up to the blit:

* **The frame stream round-trips.** What :class:`FrameWindow` writes is what
  :func:`frames` reads back, in order, bit for bit -- so the only untested step
  between a panel and a screen is `glDrawPixels` itself.
* **The run never waits on the display, and never dies with it.** A window that
  stopped reading costs frames and not ticks; a window that has gone closes the
  panel; a panel that raises is warned about and the feed keeps draining. All
  three are what *closing the panel changes nothing but the view* means in code.
* **Closing the panel changes no trajectory**, asserted the way
  `tests/test_surface.py` asserts the recorder's: two identically constructed
  agents on two identically arranged worlds, one drawn into a window that is
  shut mid-run and one not, ticked side by side and compared bit for bit.
* **Live and replay are one code path with two feeds.** The same run drawn from
  a live loop and from its own trace off disk gives identical frames.

**What rests on a human at the screen**, and is asserted nowhere:

* that the window opens at all, and that what is blitted into it is the frame
  that was sent -- orientation, colour order, and the fit into a resized window;
* that the panel is legible beside the scene window at the default scale, and
  that the two windows do not fight over focus;
* that closing the panel window with the mouse ends the panel and not the run.
  The *mechanism* under that -- a broken stream closes the panel and drains the
  feed -- is asserted below; that the close button reaches it is GLFW plumbing
  that only a hand on a mouse exercises.
"""

import contextlib
import io
import os
import threading
import time

import numpy as np
import pytest
import torch

from patchworks.agent import Agent
from patchworks.graph import build_graph
from patchworks.sandbox import PlanarPushSandbox, snapshot
from patchworks.surface import (
    DomePanel,
    OnsetCounter,
    Recorder,
    Renderer,
    Trace,
    measured_persistence,
)
from patchworks.surface import watch as watch_module
from patchworks.surface.watch import (
    compose,
    frame_size,
    live,
    main,
    paced,
    readout_for,
    replay,
    show,
)
from patchworks.surface.window import MAGIC, FrameWindow, frames, open_window
from patchworks.tick import Sheaf

TICKS = 24

#: What :func:`measured_persistence` is given here. It is the run's own
#: estimator either way (`docs/spec/05-timescales.md`); what is shortened is the
#: trajectory it is read off, which no assertion in this file is about.
MEASURED = dict(ticks=8, burn_in=2)


@pytest.fixture(scope="module")
def dome():
    return build_graph()


@pytest.fixture
def env():
    world = PlanarPushSandbox(split="any")
    yield world
    world.close()


def started(env, dome, seed=0):
    """An agent on an arranged world, one external write in and no tick yet.

    `tests/test_surface.py`'s helper, for the same reason it has one: the
    generator is explicit, so a run is the run its seed names.
    """
    agent = Agent(env, dome=dome, generator=torch.Generator().manual_seed(seed))
    observation, _info = env.reset(seed=seed)
    agent.observe(observation)
    return agent


def panels(dome, sheaf):
    """A dome panel and the readout that stacks under it, at test speed.

    Built **before** the run, as :func:`~patchworks.surface.watch.live` builds
    them: a persistence is a number about the biases and the biases adapt, so
    one measured after a run is a different number from one measured before it.
    """
    panel = DomePanel(dome, measured_persistence(sheaf, **MEASURED))
    return panel, readout_for(panel, dome)


def recorded(agent, ticks=TICKS):
    """`ticks` whole ticks, and the records they captured."""
    recorder = Recorder(agent)
    records = []
    for _ in range(ticks):
        agent.tick()
        record = recorder.observe()
        if record is not None:
            records.append(record)
    return recorder, records


class Pipe:
    """A real `os.pipe`, and a thread decoding the frames off the far end."""

    def __init__(self):
        read_fd, write_fd = os.pipe()
        self.frames: list[np.ndarray] = []
        self.write = os.fdopen(write_fd, "wb")
        self._read = os.fdopen(read_fd, "rb")
        self._reader = threading.Thread(target=self._drain, daemon=True)
        self._reader.start()

    def _drain(self):
        for frame in frames(self._read):
            self.frames.append(frame)

    def arrived(self, count, timeout=10.0):
        """Wait until `count` frames have been decoded.

        The mailbox is one frame deep and drops rather than queues, so a test
        that wants every frame it sends to arrive sends them one at a time.
        That is the API working, not a limitation being worked around.
        """
        deadline = time.monotonic() + timeout
        while len(self.frames) < count and time.monotonic() < deadline:
            time.sleep(0.002)
        assert len(self.frames) >= count, f"only {len(self.frames)} of {count} arrived"

    def finish(self, timeout=10.0):
        """Wait for the decoder to reach end of stream. The writer closes it."""
        self._reader.join(timeout)
        assert not self._reader.is_alive(), "the frame decoder never saw end of stream"
        self._read.close()
        return self.frames


class Recording:
    """A stream that keeps what was written, and can hold a write open.

    The mailbox is one frame deep and the sender thread is what empties it, so
    *dropped rather than queued* is only observable while a write is in
    progress. Timing it would be a race; holding the write open is not.
    """

    def __init__(self):
        self.written: list[bytes] = []
        self.entered = threading.Event()
        self.release = threading.Event()
        self.hold = False

    def write(self, payload):
        if self.hold:
            self.entered.set()
            self.release.wait(10.0)
        self.written.append(bytes(payload))
        return len(payload)

    def flush(self):
        pass

    def close(self):
        pass


class Broken:
    """A stream that has already gone: every write is a broken pipe."""

    def write(self, payload):
        raise BrokenPipeError(32, "Broken pipe")

    def flush(self):
        pass

    def close(self):
        pass


class TestTheFrameStreamRoundTrips:
    def test_what_the_window_writes_is_what_the_reader_reads(self):
        """The only untested step between a panel and a screen is the blit."""
        rng = np.random.default_rng(0)
        sent = [rng.integers(0, 256, (7, 5, 3), dtype=np.uint8) for _ in range(4)]
        pipe = Pipe()
        with FrameWindow(pipe.write, 7, 5) as window:
            for at, frame in enumerate(sent, start=1):
                window.show(frame)
                pipe.arrived(at)
        got = pipe.finish()
        assert len(got) == len(sent)
        for drawn, expected in zip(got, sent):
            assert np.array_equal(drawn, expected)
        assert window.dropped == 0

    def test_the_last_frame_of_a_run_is_sent_and_not_dropped(self):
        """Closing flushes the mailbox: the final picture is the final one."""
        pipe = Pipe()
        window = FrameWindow(pipe.write, 4, 4)
        last = np.full((4, 4, 3), 9, dtype=np.uint8)
        window.show(np.zeros((4, 4, 3), np.uint8))
        pipe.arrived(1)
        window.show(last)
        window.close()
        got = pipe.finish()
        assert np.array_equal(got[-1], last)

    def test_a_stream_that_is_not_a_frame_stream_is_refused(self):
        assert MAGIC not in b"not a frame stream at all"
        with pytest.raises(ValueError, match="does not start with"):
            list(frames(io.BytesIO(b"not a frame stream at all")))

    def test_a_stream_that_ends_before_it_starts_yields_nothing(self):
        assert list(frames(io.BytesIO(b""))) == []

    def test_a_stream_cut_mid_frame_ends_rather_than_raises(self):
        """A cut pipe is a thing that happens to a display, not an error."""
        stream = Recording()
        window = FrameWindow(stream, 4, 4)
        window.show(np.zeros((4, 4, 3), np.uint8))
        window.close()
        cut = b"".join(stream.written) + b"\x01\x02\x03"
        assert len(list(frames(io.BytesIO(cut)))) == 1

    def test_a_frame_of_the_wrong_shape_is_refused(self):
        """A fixed-size stream with one wrong frame in it never resynchronises."""
        window = FrameWindow(Recording(), 4, 4)
        try:
            with pytest.raises(ValueError, match="4x4 uint8"):
                window.show(np.zeros((5, 4, 3), np.uint8))
            with pytest.raises(ValueError, match="4x4 uint8"):
                window.show(np.zeros((4, 4, 3), np.float32))
        finally:
            window.close()

    @pytest.mark.parametrize("height,width", [(0, 4), (4, 0), (-1, 4), (70000, 4)])
    def test_a_window_is_a_positive_number_of_pixels_each_way(self, height, width):
        with pytest.raises(ValueError):
            FrameWindow(Recording(), height, width)

    def test_a_window_opens_at_a_whole_number_of_screen_pixels_per_frame_pixel(self):
        """Refused before anything is spawned, which is why this opens nothing."""
        with pytest.raises(ValueError, match="whole number"):
            open_window(4, 4, scale=0)


class TestTheRunNeverWaitsOnTheDisplay:
    def test_a_frame_handed_over_while_one_is_in_flight_replaces_it(self):
        """The mailbox is one deep: a slow window costs frames, never ticks."""
        stream = Recording()
        window = FrameWindow(stream, 2, 2)
        stream.hold = True
        window.show(np.full((2, 2, 3), 1, dtype=np.uint8))
        assert stream.entered.wait(10.0), "the sender never started writing"
        window.show(np.full((2, 2, 3), 2, dtype=np.uint8))
        window.show(np.full((2, 2, 3), 3, dtype=np.uint8))
        assert window.dropped == 1
        stream.hold = False
        stream.release.set()
        window.close()
        # [0] is the header; the second frame never went, and the third did.
        assert [payload[:1] for payload in stream.written[1:]] == [b"\x01", b"\x03"]

    def test_a_window_that_has_gone_is_closed_and_shows_nothing(self):
        window = FrameWindow(Broken(), 2, 2)
        try:
            assert window.closed
            window.show(np.zeros((2, 2, 3), np.uint8))
            assert window.closed
        finally:
            window.close()


class TestClosingThePanelChangesNothingButTheView:
    def test_a_window_that_has_gone_closes_the_panel_and_drains_the_feed(
        self, env, dome
    ):
        """A live feed is a run: stopping consuming it would stop the run."""
        agent = started(env, dome)
        panel, readout = panels(dome, agent.sheaf)
        _recorder, records = recorded(agent)
        assert len(records) >= 2
        height, width = frame_size(panel, readout)
        window = FrameWindow(Broken(), height, width)
        drained = []

        def feed():
            for record in records:
                drained.append(record.tick)
                yield record

        try:
            show(feed(), window, panel=panel, readout=readout)
        finally:
            window.close()
        assert drained == [record.tick for record in records]
        assert panel.closed

    def test_a_panel_that_raises_warns_and_the_feed_carries_on(self, env, dome):
        """A display is not entitled to end a run by being wrong about it."""
        agent = started(env, dome)
        panel, readout = panels(dome, agent.sheaf)
        _recorder, records = recorded(agent)
        assert len(records) >= 2
        # A non-finite `‖Δ private‖` is the case the readout refuses rather than
        # draws (#95): every mark on the baseline is a graph perfectly at rest.
        records[0].private_delta[0] = np.nan
        height, width = frame_size(panel, readout)
        pipe = Pipe()
        drained = []

        def feed():
            for record in records:
                drained.append(record.tick)
                yield record

        with FrameWindow(pipe.write, height, width) as window:
            with pytest.warns(UserWarning, match="carrying on without it"):
                show(feed(), window, panel=panel, readout=readout)
        assert drained == [record.tick for record in records]
        assert panel.closed
        assert pipe.finish() == []

    def test_a_run_watched_through_a_window_is_bit_identical_to_an_unwatched_one(
        self, dome
    ):
        """The acceptance criterion, with the world in it.

        Two identically constructed agents on two identically arranged worlds,
        one drawn into a window that is shut after the first frame and one not,
        ticked side by side: every tensor the graph holds, every torque the arm
        was given, and the whole physics state, bit for bit, on every tick.
        """
        plain_env = PlanarPushSandbox(split="any")
        watched_env = PlanarPushSandbox(split="any")
        pipe = Pipe()
        try:
            plain = started(plain_env, dome)
            watched = started(watched_env, dome)
            recorder = Recorder(watched)
            panel, readout = panels(dome, watched.sheaf)
            height, width = frame_size(panel, readout)
            counter = OnsetCounter()
            drawn = 0
            with Renderer(size=64) as scene, FrameWindow(
                pipe.write, height, width
            ) as window:
                for tick in range(TICKS):
                    left = plain.tick()
                    right = watched.tick()
                    record = recorder.observe()
                    if record is not None:
                        show(
                            [record],
                            window,
                            panel=panel,
                            readout=readout,
                            scene=scene.frame,
                            since=counter.count,
                        )
                        drawn += 1
                        if drawn == 1:
                            # The human closes the panel, mid-run.
                            window.close()
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
            assert drawn > 1, "the run stopped when the window did"
            assert panel.closed
            assert len(pipe.finish()) == 1
        finally:
            plain_env.close()
            watched_env.close()


class TestTheSecondWindowHoldsWhatTwoWindowsSaysItHolds:
    def test_the_frame_is_the_dome_panel_over_the_private_component_readout(
        self, env, dome
    ):
        """`10-the-demo-surface.md`: a second panel, **below the dome**."""
        agent = started(env, dome)
        panel, readout = panels(dome, agent.sheaf)
        _recorder, records = recorded(agent)
        assert readout.width == panel.width
        top = panel.frame(records[0])
        bottom = readout.draw(records[0])
        frame = compose(top, bottom)
        assert frame.shape == (panel.height + readout.height, panel.width, 3)
        assert frame_size(panel, readout) == frame.shape[:2]
        assert np.array_equal(frame[: panel.height], top)
        assert np.array_equal(frame[panel.height :], bottom)

    def test_two_panels_of_different_widths_are_not_stacked(self):
        with pytest.raises(ValueError, match="share a width"):
            compose(np.zeros((4, 8, 3), np.uint8), np.zeros((4, 9, 3), np.uint8))


class TestLiveAndReplayAreOneCodePathWithTwoFeeds:
    def test_a_run_and_its_own_trace_draw_identical_frames(self, env, dome, tmp_path):
        """#92's claim, carried through to the window.

        The trail is the one thing a trace cannot carry -- it decays at each
        cell's measured persistence, and a trace holds no biases -- so replay
        measures a sheaf of its own. Seeded to match, that is the same body and
        the frames are identical; this is the assertion that says so, and the
        one that would fail if replay ever became a second code path.
        """
        agent = started(env, dome)
        panel, readout = panels(dome, agent.sheaf)
        recorder, records = recorded(agent)
        assert len(records) >= 2
        with Renderer(size=64) as scene:
            drew = [
                compose(
                    panel.frame(record, render=scene.frame(record)),
                    readout.draw(record),
                )
                for record in records
            ]
            path = recorder.trace.save(tmp_path / "run.npz")
            rebuilt = Sheaf(dome, generator=torch.Generator().manual_seed(0))
            assert np.array_equal(
                measured_persistence(rebuilt, **MEASURED),
                measured_persistence(agent.sheaf, **MEASURED),
            ), "a sheaf on the same seed measures the same persistences"
            again, readout_again = panels(dome, rebuilt)
            replayed = [
                compose(
                    again.frame(record, render=scene.frame(record)),
                    readout_again.draw(record),
                )
                for record in Trace.load(path)
            ]
        assert len(replayed) == len(drew)
        for at, (one, other) in enumerate(zip(drew, replayed)):
            assert np.array_equal(one, other), f"frame {at} differs"

    def test_the_pacing_is_in_the_feed_and_passes_every_record_through(
        self, monkeypatch
    ):
        """Replay's half of *live and replay differ only in the feed*."""
        slept = []
        monkeypatch.setattr(watch_module.time, "sleep", slept.append)
        feed = list(range(5))
        assert list(paced(feed, 0)) == feed
        assert slept == []
        assert list(paced(feed, 1000.0)) == feed
        assert slept and all(nap > 0 for nap in slept)


class FakeViewer:
    """Enough of `mujoco.viewer`'s handle to run :func:`drive` with no display.

    `tests/test_gestures.py` has the full one, with a script of what the human
    did; nothing here is about a gesture, so this is the handle and no script.
    """

    def __init__(self, model, data, key_callback):
        import mujoco

        self.model, self.data, self.key_callback = model, data, key_callback
        self.perturb = mujoco.MjvPerturb()
        self.cam = mujoco.MjvCamera()
        self.syncs = 0

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        return False

    def is_running(self):
        return True

    @contextlib.contextmanager
    def lock(self):
        yield

    def sync(self):
        self.syncs += 1


@pytest.fixture
def scene_window(monkeypatch):
    """`launch_passive`, with no window behind it. Returns the handles made."""
    import mujoco.viewer

    handles = []

    def launch_passive(model, data, *, key_callback=None, **kwargs):
        handles.append(FakeViewer(model, data, key_callback))
        return handles[-1]

    monkeypatch.setattr(mujoco.viewer, "launch_passive", launch_passive)
    return handles


@pytest.fixture
def panel_window(monkeypatch):
    """`open_window`, with a pipe where the child process would be."""
    opened = []

    def open_window(height, width, *, title="patchworks", scale=2):
        pipe = Pipe()
        opened.append(pipe)
        return FrameWindow(pipe.write, height, width)

    monkeypatch.setattr(watch_module, "open_window", open_window)
    return opened


class TestTheTwoEntryPointsDrawTheRun:
    def test_live_runs_the_agent_in_the_scene_window_and_draws_beside_it(
        self, scene_window, panel_window, tmp_path
    ):
        """`live` is `drive` plus this window, and `--save` leaves the trace."""
        path = tmp_path / "run.npz"
        live(ticks=12, split="any", save=path, hold=False)
        assert len(scene_window) == 1, "the scene window is MuJoCo's, opened once"
        assert scene_window[0].syncs >= 12
        assert len(panel_window) == 1
        drawn = panel_window[0].finish()
        assert drawn, "the panel drew nothing beside the scene"
        assert path.exists()
        assert len(Trace.load(path)) == len(drawn)

    def test_replay_draws_the_same_window_from_a_file_and_opens_no_world(
        self, scene_window, panel_window, tmp_path
    ):
        """The sibling flag: same window, same panels, a file for a feed."""
        path = tmp_path / "run.npz"
        live(ticks=12, split="any", save=path, hold=False)
        assert len(panel_window) == 1
        live_frames = panel_window[0].finish()

        replay(path, fps=0, hold=False)
        assert len(scene_window) == 1, "replay opened no scene window"
        replayed = panel_window[1].finish()
        assert len(replayed) == len(live_frames)

    def test_a_replay_is_reproducible_from_its_seed(
        self, scene_window, panel_window, tmp_path
    ):
        """A trace holds no biases, so the trail's persistences are measured here.

        Which makes `--seed` load-bearing rather than decorative: measured off
        an unseeded sheaf, two replays of one file would draw two different
        trails over the same recorded marks.
        """
        path = tmp_path / "run.npz"
        live(ticks=12, split="any", save=path, hold=False)
        panel_window[-1].finish()

        replay(path, seed=0, fps=0, hold=False)
        once = panel_window[-1].finish()
        replay(path, seed=0, fps=0, hold=False)
        again = panel_window[-1].finish()
        assert once and len(once) == len(again)
        for at, (one, other) in enumerate(zip(once, again)):
            assert np.array_equal(one, other), f"frame {at} differs between replays"


class TestTheParsingIsSeparableFromTheDoing:
    """#119 adopts this as a subcommand by calling `live` and `replay`.

    So what is asserted is that :func:`main` decides nothing except which of
    them to call and with what -- there is no work in the parser for a
    dispatcher to have to reproduce.
    """

    def test_no_argument_means_live(self, monkeypatch):
        called = {}
        monkeypatch.setattr(watch_module, "live", lambda **kw: called.update(kw))
        monkeypatch.setattr(
            watch_module, "replay", lambda *a, **k: pytest.fail("replayed")
        )
        main([])
        assert called["ticks"] == 100_000
        assert called["seed"] == 0
        assert called["hold"] is True
        assert called["save"] is None

    def test_the_replay_flag_reads_a_trace_and_opens_no_world(self, monkeypatch):
        called = {}

        def replayed(path, **kw):
            called["path"] = path
            called.update(kw)

        monkeypatch.setattr(watch_module, "replay", replayed)
        monkeypatch.setattr(watch_module, "live", lambda **k: pytest.fail("ran a world"))
        main(["--replay", "run.npz", "--fps", "3", "--no-hold", "--edges", "--raw"])
        assert called["path"] == "run.npz"
        assert called["fps"] == 3
        assert called["hold"] is False
        assert called["edges"] is True
        assert called["raw"] is True

    def test_the_doing_takes_no_argv_and_opens_nothing_by_itself(self):
        """`show` is the seam a dispatcher calls: everything is an argument."""
        import inspect

        assert list(inspect.signature(show).parameters) == [
            "feed",
            "window",
            "panel",
            "readout",
            "scene",
            "since",
        ]
        assert "argv" not in inspect.signature(live).parameters
        assert "argv" not in inspect.signature(replay).parameters


class TestNoDisplayIsTouchedUntilOneIsOpened:
    def test_importing_the_surface_imports_no_graphics_library(self):
        """`import patchworks.surface` must not need a display to succeed.

        glfw and pyopengl arrive with `mujoco`, which `pyproject.toml` pins
        exactly, so neither is a new dependency -- but a headless sweep that
        imports the record should not be loading a window toolkit, and
        `patchworks.surface.gestures` defers `mujoco.viewer` for the same
        reason.
        """
        import subprocess
        import sys

        probe = (
            "import sys, patchworks.surface, patchworks.surface.window; "
            "print(int('glfw' in sys.modules), int('OpenGL' in sys.modules))"
        )
        done = subprocess.run(
            [sys.executable, "-c", probe], capture_output=True, text=True, check=True
        )
        assert done.stdout.split() == ["0", "0"], done.stdout
