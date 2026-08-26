"""The second window: a process with a GLFW window in it, fed uint8 frames.

`docs/spec/10-the-demo-surface.md`, *Two windows*. The scene window is MuJoCo's
passive viewer (:func:`patchworks.surface.gestures.drive`); this is the other
one. It draws whatever `[height, width, 3]` uint8 array it is handed --
:meth:`patchworks.surface.dome_panel.DomePanel.frame`'s, in practice -- and it
knows nothing else. It holds no dome, no agent, no record and no world.

**How the two windows share the main thread on macOS: they do not, because they
cannot.** This is the arrangement the module exists to work around, and every
line of it was measured on the reference laptop rather than reasoned about:

* On macOS the scene window needs `mjpython`, whose own docstring says what it
  does -- it "executes a native binary that runs the CPython interpreter entry
  point in a separate thread, thus leaving the macOS main thread free for Cocoa
  GUI calls". So **under `mjpython` no Python thread is the Cocoa main thread**,
  including the one Python calls `MainThread`. Measured: `NSThread.isMainThread`
  is false there, while `threading.main_thread()` is the thread asking.
* GLFW creates an `NSWindow`, and Cocoa refuses to make one off the Cocoa main
  thread. Measured, from the script thread under `mjpython`: the process dies
  with `NSInternalInconsistencyException ... 'NSWindow should only be
  instantiated on the main thread!'`, thrown out of `_glfwCreateWindowCocoa`.
  Not an exception a caller can catch -- it aborts the process, and it would
  abort the run the panel was watching.
* Off `mjpython`, on a plain interpreter, a *second* thread is no better:
  `glfw.init()` called from one never returns. Measured, by joining it with a
  timeout and finding it still alive.
* `mujoco.viewer` has one hook onto the Cocoa main thread, `_MJPYTHON`, and it
  dispatches exactly one call -- `launch_on_ui_thread(model, data, ...)`. There
  is no general "run this on the UI thread" for a second window to use.

So the panel window lives in **its own process**, where it is the Cocoa main
thread and the constraint above is satisfied by construction. Measured: a child
spawned from under `mjpython` reports `NSThread.isMainThread` true and creates
its window. That also settles the platform question in one place -- the
arrangement is the same on Linux and Windows, where the split is unnecessary but
harmless, rather than one code path per platform.

**The run never waits on the display.** Each side holds a mailbox of one frame:
:meth:`FrameWindow.show` drops a frame into the parent's and returns, a sender
thread writes it down the pipe, and the child's reader thread fills the child's
while its main thread draws whatever is in it. Neither mailbox ever queues, so a
slow or stalled window costs dropped frames and never a stalled tick. It is not
a nicety: on macOS dragging or resizing a window blocks that process's event
loop for as long as the mouse is down, and a display that could stall the loop
`drive` runs would be a display that changed the run's timing.

**Closing the window changes nothing but the view.** The child exits, the pipe
breaks, the next :meth:`FrameWindow.show` sees it and sets :attr:`closed`, and
the caller stops drawing. Nothing here holds an agent, a sheaf, a recorder or a
world, so there is nothing else for closing to reach -- `tests/test_watch.py`
asserts the trajectory is bit-identical either way.

**No new dependency.** `glfw` and `pyopengl` are both requirements of
`mujoco==3.10.0`, which `pyproject.toml` pins exactly, and `mujoco.viewer`
imports `glfw` at module scope -- so any environment that can open the scene
window already has both, at a version that pin fixes. Neither is imported until
a window is actually opened (:func:`serve`), for the reason
:func:`~patchworks.surface.gestures.drive` defers `mujoco.viewer`.

Neither is **named** in `pyproject.toml`, and that follows the file's own
recorded rule rather than being an omission: `pyyaml` is listed there because
"nothing else in the dependency set brings a YAML parser with it". Something
else brings these two, and an exact pin is what makes that a guarantee instead
of a hope.

**Run as a script, not as a module.** The child is started as
`python .../surface/window.py`, a path rather than `-m`, so it imports this file
alone and not the package around it: what the child needs is numpy, glfw and
pyopengl, and going in through `patchworks.surface` would drag torch, mujoco and
the whole graph into a process that draws pixels. Nothing in this file imports
from the package, which is what keeps that true.
"""

from __future__ import annotations

import os
import struct
import subprocess
import sys
import threading
from typing import BinaryIO, Iterator

import numpy as np

__all__ = ["MAGIC", "FrameWindow", "frames", "open_window", "serve"]

#: The stream's first eight bytes: a magic, then the frame shape. Every frame
#: after it is exactly `height * width * 3` bytes of C-ordered RGB, and there is
#: no per-frame header because a panel's size is fixed for a whole run
#: (:class:`~patchworks.surface.dome_panel.DomePanel`). The magic is here so
#: that a stream wired to the wrong end fails on its first read with something
#: that names the problem, rather than opening a window on noise.
MAGIC = b"PWPN"
_HEADER = struct.Struct("<4sHH")

#: Seconds the child waits between redraws when no frame has arrived. It is a
#: ceiling on how long the window can look ignored -- a close, a drag or an
#: expose is handled the moment it happens, because the wait ends on any event.
_IDLE = 1.0 / 60.0

#: Seconds :meth:`FrameWindow.close` gives the child to go away on its own
#: before killing it. It has only to fall out of a read and destroy a window.
_GRACE = 5.0


def _read_exactly(stream: BinaryIO, count: int) -> bytes | None:
    """`count` bytes, or `None` at end of stream. Short reads are not an end."""
    chunks: list[bytes] = []
    got = 0
    while got < count:
        chunk = stream.read(count - got)
        if not chunk:
            return None
        chunks.append(chunk)
        got += len(chunk)
    return b"".join(chunks)


def frames(stream: BinaryIO) -> Iterator[np.ndarray]:
    """The frames on `stream`, as `[height, width, 3]` uint8 arrays.

    The child's whole reading of the protocol, and the half of it that can be
    tested without a display: hand it the read end of a pipe
    :class:`FrameWindow` is writing to and the arrays come back out.

    Ends at end of stream. A stream that ends **mid-frame** ends here too, and
    silently: the writer was killed or the pipe was cut, which is a thing that
    happens to a display and not a thing to raise about. A stream whose header
    is not this protocol's is refused, because that is a wiring mistake and
    drawing it would be a window full of noise.
    """
    header = _read_exactly(stream, _HEADER.size)
    if header is None:
        return
    magic, height, width = _HEADER.unpack(header)
    if magic != MAGIC:
        raise ValueError(
            f"this stream does not start with {MAGIC!r} and is not a frame "
            f"stream; it starts {magic!r}. A frame stream is what "
            "`FrameWindow` writes -- see patchworks/surface/window.py."
        )
    if not height or not width:
        raise ValueError(
            f"a frame stream declares a frame of {height}x{width}, which has no "
            "pixels in it"
        )
    count = height * width * 3
    while True:
        payload = _read_exactly(stream, count)
        if payload is None:
            return
        yield np.frombuffer(payload, dtype=np.uint8).reshape(height, width, 3)


class FrameWindow:
    """The parent's end: frames in, and a window in another process.

    ::

        with open_window(panel.height, panel.width) as window:
            for record in feed:
                window.show(panel.frame(record))
                if window.closed:
                    break

    Built on a stream rather than on a process, because the stream is the whole
    of what this class does with it -- :func:`open_window` is the one that knows
    about a child, and a test hands in a pipe instead.

    **:meth:`show` never blocks, and nothing the display does makes it raise.**
    It leaves the frame in a mailbox of one and returns; a sender thread writes
    whatever is in the mailbox down the stream, and a frame that arrives while
    one is still waiting replaces it. So the caller's pace is its own, and a
    window that has stopped reading costs frames rather than ticks
    (:attr:`dropped` counts them). What it *does* raise on is a frame of the
    wrong shape, which is the caller's mistake and not the window's state.
    """

    def __init__(
        self,
        stream: BinaryIO,
        height: int,
        width: int,
        *,
        process: subprocess.Popen | None = None,
    ) -> None:
        for name, size in (("height", height), ("width", width)):
            if isinstance(size, bool) or not isinstance(size, int) or size < 1:
                raise ValueError(
                    f"a frame is a positive number of pixels {name}; got {size!r}"
                )
        if height > 0xFFFF or width > 0xFFFF:
            raise ValueError(
                f"a frame stream carries a shape in two 16-bit fields, so a frame is "
                f"at most 65535 pixels each way; got {height}x{width}"
            )
        self.height = height
        self.width = width
        self.dropped = 0
        """Frames :meth:`show` was handed while an unsent one was still waiting."""
        self._stream = stream
        self._process = process
        self._state = threading.Condition()
        self._pending: bytes | None = None
        self._stopping = False
        self._closed = False
        # The header goes down synchronously: it is eight bytes, so it cannot
        # block on any pipe, and writing it here means a caller that never
        # shows a frame has still declared the window's size.
        try:
            stream.write(_HEADER.pack(MAGIC, height, width))
            stream.flush()
        except (BrokenPipeError, OSError, ValueError):
            self._closed = True
        self._sender = threading.Thread(
            target=self._send, name="patchworks-frames", daemon=True
        )
        self._sender.start()

    def __repr__(self) -> str:
        state = "closed" if self._closed else "open"
        return (
            f"FrameWindow({self.width}x{self.height}, {state}, "
            f"{self.dropped} dropped)"
        )

    def __enter__(self) -> "FrameWindow":
        return self

    def __exit__(self, *exception) -> None:
        self.close()

    @property
    def closed(self) -> bool:
        """Whether the window has gone -- the human closed it, or :meth:`close` did.

        It is observed rather than asked: the child's exit breaks the pipe, and
        what notices is the sender thread failing to write. So it turns true on
        the first frame shown after the window went away, not at the instant it
        did, and a caller that shows nothing never notices at all. That is the
        right way round -- nothing here polls a display on the run's behalf.
        """
        with self._state:
            return self._closed

    def show(self, frame: np.ndarray) -> None:
        """Hand the window a frame. Returns at once, and draws nothing itself.

        Refuses a frame that is not this window's shape, because a stream of
        fixed-size frames with one wrong-sized frame in it is a stream that
        never resynchronises -- every frame after it would be drawn from the
        bytes of two.

        A frame handed to a closed window is dropped, quietly: the run goes on
        after the window does, and a caller that has not looked at
        :attr:`closed` yet is not doing anything wrong.
        """
        array = np.asarray(frame)
        if array.dtype != np.uint8 or array.shape != (self.height, self.width, 3):
            raise ValueError(
                f"this window draws {self.height}x{self.width} uint8 RGB frames "
                f"and was handed {array.shape} of {array.dtype}. A window's frame "
                "size is fixed when it opens, as a panel's is."
            )
        payload = np.ascontiguousarray(array).tobytes()
        with self._state:
            if self._closed or self._stopping:
                return
            if self._pending is not None:
                self.dropped += 1
            self._pending = payload
            self._state.notify()

    def wait(self, timeout: float | None = None) -> bool:
        """Block until the child process exits. `True` if it has.

        What holds the window open after a feed has run out, so that the last
        frame of a replay is a thing a human can look at rather than a thing
        that flashes. `False` on timeout, and `True` immediately when there is
        no child to wait for -- a window built on a bare stream is nobody's
        process to outlive.
        """
        if self._process is None:
            return True
        try:
            self._process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return False
        return True

    def close(self) -> None:
        """Close the stream, stop the sender, and reap the child. Idempotent.

        A frame still waiting in the mailbox is **sent** before the stream goes:
        it is the run's last one, and the last frame of a run is the one worth
        leaving on screen. Dropping it would have made the final picture of
        every capture and every replay the second-to-last one.
        """
        with self._state:
            if self._stopping:
                return
            self._stopping = True
            self._state.notify()
        self._sender.join(timeout=_GRACE)
        try:
            self._stream.close()
        except (BrokenPipeError, OSError, ValueError):
            pass
        if self._process is not None:
            try:
                self._process.wait(timeout=_GRACE)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
        with self._state:
            self._closed = True

    def _send(self) -> None:
        """The sender thread: the mailbox, down the stream, one frame at a time.

        It stops when the mailbox is empty *and* the window is closing, so the
        frame :meth:`close` was racing with goes down the pipe rather than into
        the bin.
        """
        while True:
            with self._state:
                while self._pending is None and not self._stopping:
                    self._state.wait()
                payload, self._pending = self._pending, None
                if payload is None:
                    return
            try:
                self._stream.write(payload)
                self._stream.flush()
            except (BrokenPipeError, OSError, ValueError):
                # The window went away. That is the human closing it, and the
                # only thing it changes is that there is nothing to draw on.
                with self._state:
                    self._closed = True
                return


def open_window(
    height: int,
    width: int,
    *,
    title: str = "patchworks",
    scale: int = 2,
    executable: str | None = None,
) -> FrameWindow:
    """Open the window in a child process and return the parent's end.

    `scale` is how many screen pixels a frame pixel opens at -- the panel is
    laid out in lattice slots and is a few hundred pixels across, which is a
    postage stamp on a modern display. It sizes the window and nothing else: the
    child fits the frame to whatever size the window ends up, so resizing it, or
    a display with a scale factor of its own, costs nothing here.

    The child is started by **path** rather than by `-m` (see this module's
    docstring), which assumes this file is on a filesystem the child can open --
    true of an editable install and of an unpacked wheel, and not of a zipped
    one. A zipped install fails at `Popen` with the interpreter's own complaint
    about the path; nothing here can draw pixels out of a zip either way.

    `executable` is the interpreter to start it with, defaulting to this one.
    Under `mjpython` that default is already the plain interpreter -- the
    trampoline sets `sys.executable` to the venv's python before it execs --
    which is what the child wants: it draws pixels and has no use for the Cocoa
    dispatch `mjpython` exists to provide.
    """
    if isinstance(scale, bool) or not isinstance(scale, int) or scale < 1:
        raise ValueError(
            "a window opens at a whole number of screen pixels per frame pixel, at "
            f"least 1; got {scale!r}"
        )
    command = [
        executable or sys.executable,
        os.path.abspath(__file__),
        "--title",
        str(title),
        "--scale",
        str(scale),
    ]
    # stdout and stderr are inherited on purpose: a GLFW that cannot reach a
    # display says so there, and a human who ran this from a terminal is owed
    # the message rather than a window that never appears.
    child = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert child.stdin is not None
    return FrameWindow(child.stdin, height, width, process=child)


def serve(stream: BinaryIO, *, title: str = "patchworks", scale: int = 2) -> int:
    """The child: open a window, and draw the frames on `stream` until it ends.

    Returns a process exit status. **Nothing in the test suite calls this** --
    it opens a window, and no test opens a window. What it does is
    :func:`frames` (tested, on a pipe) and a blit (not tested, and not testable
    without a display).

    Two threads, for the reason :class:`FrameWindow` has two: this one draws and
    handles events at :data:`_IDLE`, and a reader thread blocks on the stream
    and leaves what it gets in a mailbox of one. A feed that stops does not
    freeze the window -- the close button still works -- and a window being
    dragged does not stop the reader draining the pipe.

    **It ends at end of stream**, which is the parent closing the pipe. That is
    what makes *hold the window open after the feed* the parent's to decide and
    not a mode here: a parent that wants the last frame left on screen keeps the
    stream open and waits (:meth:`FrameWindow.wait`), and a parent that wants
    the window gone closes it (:meth:`FrameWindow.close`).
    """
    import glfw
    from OpenGL import GL

    if not glfw.init():
        print(
            "could not initialise GLFW; the panel window cannot open",
            file=sys.stderr,
        )
        return 1

    state = threading.Condition()
    latest: list[np.ndarray | None] = [None]
    done = [False]

    def read() -> None:
        try:
            for frame in frames(stream):
                with state:
                    latest[0] = frame
                    state.notify()
        except (ValueError, OSError) as complaint:
            print(complaint, file=sys.stderr)
        finally:
            with state:
                done[0] = True
                state.notify()

    reader = threading.Thread(target=read, name="patchworks-frames", daemon=True)
    reader.start()

    # The first frame is what says how big the window should be, so the window
    # is not opened until one arrives. A feed that ends without sending one
    # opens nothing, which is the right answer to a run that drew nothing.
    with state:
        while latest[0] is None and not done[0]:
            state.wait()
        first, latest[0] = latest[0], None
    if first is None:
        glfw.terminate()
        return 0

    height, width = first.shape[:2]
    window = glfw.create_window(width * scale, height * scale, title, None, None)
    if not window:
        print("could not open the panel window", file=sys.stderr)
        glfw.terminate()
        return 1
    glfw.make_context_current(window)
    glfw.swap_interval(1)

    # The frame on screen is redrawn every pass rather than only when a new one
    # arrives: a resize, an expose or a move leaves the back buffer undefined,
    # and a panel updating at 10 Hz would then show whatever was behind it for
    # a tenth of a second at a time. The pass is paced twice over -- the event
    # wait times out at :data:`_IDLE`, and the swap waits for vertical retrace
    # -- because either alone can fail to throttle an occluded window.
    current = first
    while not glfw.window_should_close(window):
        with state:
            fresh, finished, latest[0] = latest[0], done[0], None
        if fresh is not None:
            current = fresh
        elif finished:
            break
        _blit(GL, glfw, window, current)
        glfw.wait_events_timeout(_IDLE)

    glfw.destroy_window(window)
    glfw.terminate()
    return 0


def _blit(GL, glfw, window, frame: np.ndarray) -> None:
    """Draw one frame, fitted to the window and centred, and swap.

    `glDrawPixels` with a pixel zoom rather than a texture and a quad: the
    context GLFW gives on macOS without hints is a legacy 2.1 one, which has
    both, and this is the one that needs no shader, no vertex buffer and no
    state to leak. The vertical zoom is negative and the raster position is the
    top edge, because a frame's row 0 is its top row and OpenGL's is its bottom.

    Fitted rather than stretched: the aspect ratio is the panel's layout, and a
    window the human has made the wrong shape should letterbox rather than lie
    about where a mark sits.
    """
    height, width = frame.shape[:2]
    fb_width, fb_height = glfw.get_framebuffer_size(window)
    if fb_width < 1 or fb_height < 1:
        return
    zoom = min(fb_width / width, fb_height / height)
    across, down = width * zoom, height * zoom
    left, bottom = (fb_width - across) / 2.0, (fb_height - down) / 2.0
    GL.glViewport(0, 0, fb_width, fb_height)
    GL.glClearColor(0.0, 0.0, 0.0, 1.0)
    GL.glClear(GL.GL_COLOR_BUFFER_BIT)
    GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
    GL.glPixelZoom(zoom, -zoom)
    GL.glRasterPos2f(
        2.0 * left / fb_width - 1.0, 2.0 * (bottom + down) / fb_height - 1.0
    )
    GL.glDrawPixels(width, height, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, frame)
    glfw.swap_buffers(window)


def main(argv: list[str] | None = None) -> int:
    """`python patchworks/surface/window.py` -- the child, on stdin.

    Not an entry point for a human: :func:`open_window` starts it, and what it
    reads is a frame stream rather than anything typed. `patchworks watch` is
    the command a human runs (:mod:`patchworks.surface.watch`).
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Draw a patchworks frame stream from stdin."
    )
    parser.add_argument("--title", default="patchworks")
    parser.add_argument("--scale", type=int, default=2)
    arguments = parser.parse_args(argv)
    return serve(sys.stdin.buffer, title=arguments.title, scale=arguments.scale)


if __name__ == "__main__":
    sys.exit(main())
