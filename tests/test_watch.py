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
import subprocess
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
from patchworks.surface import window as window_module
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
        self.broken = False
        self.closed = False

    def write(self, payload):
        if self.hold:
            self.entered.set()
            self.release.wait(10.0)
        if self.broken:
            raise BrokenPipeError(32, "Broken pipe")
        self.written.append(bytes(payload))
        return len(payload)

    def flush(self):
        pass

    def close(self):
        self.closed = True


class Child:
    """Enough of `Popen` to be killed. Killing it releases the stream it feeds.

    Which is what a real child's death does: it breaks the pipe, and the write
    the sender is stuck in fails.
    """

    def __init__(self, stream=None):
        self.stream = stream
        self.killed = False
        self.returncode = None

    def kill(self):
        self.killed = True
        self.returncode = -9
        if self.stream is not None:
            self.stream.broken = True
            self.stream.release.set()

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        if self.returncode is None:
            raise subprocess.TimeoutExpired("child", timeout)
        return self.returncode


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

    def test_everything_refusable_is_refused_before_a_child_is_spawned(
        self, monkeypatch
    ):
        """A raise after the `Popen` leaves a process nobody has a handle to.

        Which is why this test can call `open_window` at all: nothing it passes
        gets as far as spawning, and the `Popen` is replaced so that a
        regression is a failure here rather than a stray window.
        """
        monkeypatch.setattr(
            window_module.subprocess,
            "Popen",
            lambda *a, **k: pytest.fail("a child was spawned for a refused window"),
        )
        with pytest.raises(ValueError, match="whole number"):
            open_window(4, 4, scale=0)
        with pytest.raises(ValueError, match="65535"):
            open_window(70000, 4)
        with pytest.raises(ValueError, match="positive number of pixels"):
            open_window(0, 4)


class TestTheFrameIsFittedToTheWindowWithoutFallingOutOfIt:
    """The blit's arithmetic, which is the half of it a display is not needed for.

    OpenGL marks a raster position outside the view volume invalid and then
    ignores the whole `glDrawPixels` -- a black window and no error anywhere --
    so a coordinate a hair past `-1` is a panel that silently stops drawing.
    """

    def test_the_raster_position_never_leaves_the_view_volume(self):
        """Swept, because the overshoot is an ulp and lands on some widths only.

        `width * (fb_width / width)` can exceed `fb_width`, which puts the left
        edge very slightly below zero and the coordinate below `-1`. A window
        480 across at a framebuffer of 962 is one of the widths where it does,
        and a handful in every hundred are -- which is why this sweeps rather
        than picking the one that was found.
        """
        for fb_width in range(1, 2000):
            for fb_height in (1, 3, 1080, 4000):
                zoom, x, y = window_module.fitted(424, 480, fb_width, fb_height)
                where = f"{fb_width}x{fb_height}"
                assert -1.0 <= x <= 1.0, f"x = {x!r} at {where}"
                assert -1.0 <= y <= 1.0, f"y = {y!r} at {where}"
                # Fitted rather than stretched: the drawn extent is inside the
                # framebuffer in **both** axes, which is what letterboxing is.
                assert 480 * zoom <= fb_width + 1e-9, where
                assert 424 * zoom <= fb_height + 1e-9, where

    def test_an_exact_fit_starts_at_the_top_left_corner(self):
        """The common case, and the one the orientation depends on."""
        zoom, x, y = window_module.fitted(40, 24, 48, 80)
        assert zoom == 2.0
        assert (x, y) == (-1.0, 1.0)

    def test_a_frame_narrower_than_its_window_is_centred_in_it(self):
        """Letterboxed on the sides, and still starting at the top."""
        zoom, x, y = window_module.fitted(40, 24, 96, 80)
        assert zoom == 2.0  # height-limited: 80/40 beats 96/24
        assert x == pytest.approx(-0.5)  # 24 pixels of margin each side of 96
        assert y == 1.0


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

    def test_closing_kills_a_child_the_sender_is_stuck_writing_to(self):
        """The deadlock this design walks straight into if the order is wrong.

        A write blocked on a full pipe holds the stream's own lock, so closing
        that stream waits behind the write instead of interrupting it. What
        unblocks it is the child going away. Order it the other way -- close,
        then kill -- and the kill is unreachable and `close()` never returns.

        Reachable rather than theoretical: the child reads nothing until GLFW
        has opened a window, and one panel frame is bigger than a pipe buffer.
        """
        stream = Recording()
        child = Child(stream)
        window = FrameWindow(stream, 2, 2, process=child, grace=0.05)
        stream.hold = True
        window.show(np.zeros((2, 2, 3), np.uint8))
        assert stream.entered.wait(10.0), "the sender never started writing"
        finished = threading.Event()
        threading.Thread(
            target=lambda: (window.close(), finished.set()), daemon=True
        ).start()
        assert finished.wait(10.0), "close() is behind the stuck write"
        assert child.killed, "the child was not killed out of the sender's way"
        # And once the kill has broken the pipe under the sender, the stream is
        # closed as usual -- the kill is what makes that reachable.
        assert stream.closed
        assert window.closed

    def test_closing_returns_even_with_no_child_to_kill(self):
        """A bare stream has nothing to unblock a stuck sender.

        So the choice is a leaked file descriptor until the process ends, or a
        caller deadlocked on `close()`. It takes the first, and the stream is
        left alone rather than closed under a thread that is still inside it.
        """
        stream = Recording()
        window = FrameWindow(stream, 2, 2, grace=0.05)
        stream.hold = True
        window.show(np.zeros((2, 2, 3), np.uint8))
        assert stream.entered.wait(10.0)
        finished = threading.Event()
        threading.Thread(
            target=lambda: (window.close(), finished.set()), daemon=True
        ).start()
        assert finished.wait(10.0), "close() did not return"
        assert not stream.closed, "a stream a thread is still inside was closed"
        stream.release.set()

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
                    held = ("stalks", "charts", "prediction", "broadcast", "incoming")
                    for name in held:
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


class Held(FrameWindow):
    """A window that counts how long it was asked to stay open, and never does.

    :meth:`FrameWindow.wait` blocks on a child exiting, and a test that let it
    would be a test that hangs -- which is the very failure being asserted
    against. So the wait is counted instead of taken.
    """

    waits = 0

    def wait(self, timeout=None):
        self.waits += 1
        return True


@pytest.fixture
def panel_window(monkeypatch):
    """`open_window`, with a pipe where the child process would be."""
    opened = []

    def open_window(height, width, *, title="patchworks", scale=2):
        pipe = Pipe()
        pipe.window = Held(pipe.write, height, width)
        opened.append(pipe)
        return pipe.window

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

    def test_a_capture_the_patch_lattice_cannot_tile_is_refused_up_front(self):
        """Not left to fail once inside the loop, where a display must not stop.

        `DomePanel.frame` refuses a render the patch cells were never cut from,
        and inside :func:`show` that refusal closes the panel and lets the run
        carry on -- which is right there and wrong here: the whole run would go
        by with a dead window. So the number is checked where it is chosen.
        """
        dome = build_graph()
        with pytest.raises(ValueError, match="does not divide 100"):
            watch_module.check_capture(100, dome)
        watch_module.check_capture(64, dome)

    def test_live_refuses_that_capture_before_it_arranges_anything(
        self, scene_window, panel_window
    ):
        with pytest.raises(ValueError, match="patch lattice"):
            live(ticks=12, capture=100)
        assert scene_window == [] and panel_window == []

    def test_live_refuses_a_bad_scale_before_it_arranges_anything(
        self, scene_window, panel_window
    ):
        """The same rule as the capture, for the same reason it has one.

        Left to `open_window`, a scale of nothing is refused after the world,
        the agent and a 64-tick persistence measurement have all been built for
        a window that will not open.
        """
        with pytest.raises(ValueError, match="whole number"):
            live(ticks=12, scale=0)
        assert scene_window == [] and panel_window == []

    def test_a_run_that_drew_nothing_holds_no_window(
        self, scene_window, panel_window
    ):
        """Otherwise the command sits forever on a window that never opened.

        The child creates no window until the first frame arrives -- the frame
        is what says how big it should be -- so waiting on it after a run that
        drew nothing is a wait nothing can end. `--ticks 4` reaches it on the
        defaults: a capture is one tick in five and the first primes.
        """
        live(ticks=4, split="any", hold=True)
        pipe = panel_window[0]
        assert pipe.window.shown == 0
        assert pipe.window.waits == 0, "held a window that never opened"

    def test_a_run_that_drew_something_does_hold_it(self, scene_window, panel_window):
        """The other half: `hold` is the default and it has to mean something."""
        live(ticks=12, split="any", hold=True)
        pipe = panel_window[0]
        assert pipe.window.shown > 0
        assert pipe.window.waits == 1

    def test_an_interrupted_run_still_leaves_its_trace(
        self, scene_window, panel_window, tmp_path, monkeypatch
    ):
        """The default is 100,000 ticks; the usual way to end one is ctrl-C.

        A save that only ran on the success path would throw the run's one
        artefact away in exactly the case the human meant to stop watching.
        """
        path = tmp_path / "run.npz"

        def interrupted(feed, window, **arguments):
            for _record in feed:
                raise KeyboardInterrupt
            raise AssertionError("the feed produced nothing to interrupt")

        monkeypatch.setattr(watch_module, "show", interrupted)
        with pytest.raises(KeyboardInterrupt):
            live(ticks=12, split="any", save=path, hold=False)
        assert path.exists()

    def test_replay_measures_its_trail_off_the_sheaf_the_seed_names(
        self, scene_window, panel_window, tmp_path, monkeypatch
    ):
        """A trace holds no biases, so which body the trail decays at is a choice.

        `--seed` is what makes it the run's own, and it is asserted here rather
        than through the frames on purpose: on the **untrained** body the frames
        cannot show it. Every cell's measured persistence is under two ticks
        (0.5 to 1.7 on the default spec, seeded), a capture is one tick in five,
        so a glow has decayed to at most 6% of itself by the next frame and the
        trail is invisible whichever body was measured. That is a fact about the
        body rather than about this module -- `05-timescales.md`'s go/no-go is
        what has something to say about it -- and it is exactly the reason this
        assertion is on the call and not on the picture. A frame-level assertion
        would have passed with the seed unwired.
        """
        path = tmp_path / "run.npz"
        live(ticks=12, split="any", save=path, hold=False)
        seeds = []
        built = watch_module.Sheaf

        def spy(dome, **arguments):
            generator = arguments.get("generator")
            seeds.append(None if generator is None else generator.initial_seed())
            return built(dome, **arguments)

        monkeypatch.setattr(watch_module, "Sheaf", spy)
        replay(path, seed=7, fps=0, hold=False)
        assert seeds == [7]


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
        monkeypatch.setattr(
            watch_module, "live", lambda **k: pytest.fail("ran a world")
        )
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
