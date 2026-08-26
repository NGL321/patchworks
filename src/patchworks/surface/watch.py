"""`patchworks watch`: the panel, in a window, live and from a trace.

`docs/spec/10-the-demo-surface.md`, *Two windows*, had one until this module.
:meth:`~patchworks.surface.dome_panel.DomePanel.frame` has returned a
`[height, width, 3]` uint8 array since #93 and nothing in `src/` put one on a
screen; :func:`~patchworks.surface.gestures.drive` opens the scene window and
says outright that the panel is the other one and is not opened there. This is
the other one.

**Display plumbing, and deliberately nothing else.** The seam was already
built: `drive` yields
:class:`~patchworks.surface.record.TickRecord`\\ s at ~10 Hz, a
:class:`~patchworks.surface.record.Trace` off disk holds the same records, and
every panel here consumes exactly those. So **live and replay differ only in the
feed** (`10-the-demo-surface.md`, *The trace*): :func:`show` is handed an
iterable and cannot tell which it has, and the two entry points below differ in
one expression each.

**What the second window holds** is what *Two windows* says it holds: the dome,
the private-component readout, and the motor strip. The strip is drawn inside
the dome panel (#94); the readout is its own panel (#95), *below the dome* as
that section puts it, and :func:`compose` is the whole of the stacking.

**How the two windows share the main thread on macOS: they do not.** They are
two processes, and :mod:`patchworks.surface.window` is where that is measured
and explained -- the short of it is that under `mjpython` no Python thread is
the Cocoa main thread, and Cocoa aborts the process rather than make an
`NSWindow` off it. So the scene window is drawn on the Cocoa main thread of the
process running the agent, the panel window on the Cocoa main thread of a child,
and the run's own loop is on neither.

**The surface stays off the architecture** (`10-the-demo-surface.md`'s opening
constraint, and #77). Nothing drawn here is fed back: a panel holds no agent, no
sheaf and no world, :func:`~patchworks.surface.dome_panel.measured_persistence`
draws from a private generator rather than the global torch stream, and the
window is a pipe to another process. Two consequences are worth stating rather
than assuming:

* **Closing the panel changes no trajectory.** The window going away sets
  :attr:`~patchworks.surface.window.FrameWindow.closed`, :func:`show` closes the
  panels and goes on draining the feed -- a live feed is a run being driven, and
  a display that stopped consuming would stop the run.
  `tests/test_watch.py` asserts the run is bit-identical either way.
* **A panel that fails does not take the run with it.** A draw that raises is
  warned about, once, and the panels close; the feed keeps draining. A display
  is not entitled to end a run by being wrong about it.

**What replay cannot reproduce, and it matters for one mark.** A trace holds
state and the record's arrays -- not the sheaf's biases, which is right, since
`10-the-demo-surface.md` is explicit that a trace is the snapshot/restore
contract plus those arrays and *not a new format*. But the dome panel's **trail**
decays at each cell's measured persistence, and that is measured from a body and
its biases (:func:`~patchworks.surface.dome_panel.measured_persistence`). So
:func:`replay` builds a sheaf of its own to measure and says so here rather than
implying the number came off the file: with `--seed` matching the run's, the
biases are the same draw and the trail is the one the live panel drew; with a
different seed it is a different body's trail over the recorded run's marks.
Every other mark on both panels comes off the record and is exact.
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Callable, Iterable, Iterator

import numpy as np
import torch

from patchworks.agent import Agent
from patchworks.graph import DEFAULT_SPEC, Dome, DomeSpec, build_graph
from patchworks.sandbox import PlanarPushSandbox
from patchworks.sandbox.env import IMAGE_SIZE
from patchworks.tick import Sheaf

from .dome_panel import DEFAULT_PITCH, DomePanel, measured_persistence
from .gestures import drive
from .onset import OnsetCounter
from .private_component import PrivateComponentPanel
from .record import CAPTURE_HZ, Recorder, TickRecord, Trace
from .renderer import Renderer
from .window import FrameWindow, open_window

__all__ = [
    "DEFAULT_SCALE",
    "compose",
    "frame_size",
    "live",
    "main",
    "paced",
    "readout_for",
    "replay",
    "show",
]

#: How many screen pixels one frame pixel opens at, by default. The panel is
#: laid out in lattice slots -- a few hundred pixels across the default dome --
#: and one-to-one on a modern display is a postage stamp. It is the window's
#: size and nothing else: no mark moves with it.
DEFAULT_SCALE = 2


def readout_for(panel: DomePanel, dome: Dome) -> PrivateComponentPanel:
    """The private-component panel that stacks under `panel`.

    Its width is the dome panel's, because the two are one frame and a
    composite has one width. Its height keeps the 2:1 the module's own defaults
    have (`PANEL_WIDTH`/`PANEL_HEIGHT`), which is the proportion the scatter was
    drawn to read at.
    """
    return PrivateComponentPanel(dome, width=panel.width, height=max(16, panel.width // 2))


def compose(dome_frame: np.ndarray, readout_frame: np.ndarray) -> np.ndarray:
    """The second window's frame: the dome panel, and the readout below it.

    `10-the-demo-surface.md`, *The private-component panel*: **a second panel,
    below the dome**. Stacked and not blended -- that section's whole argument
    is that folding `‖Δ private‖` into the dome's marks makes neither readable,
    and this function is the one place the two pictures meet, so there is
    nowhere for one to reach the other.
    """
    if dome_frame.shape[1] != readout_frame.shape[1]:
        raise ValueError(
            f"the two panels are stacked into one frame and so share a width; the "
            f"dome panel is {dome_frame.shape[1]} across and the readout is "
            f"{readout_frame.shape[1]}. `readout_for(panel, dome)` is what builds one "
            "at the other's width."
        )
    return np.concatenate((dome_frame, readout_frame), axis=0)


def frame_size(panel: DomePanel, readout: PrivateComponentPanel) -> tuple[int, int]:
    """`(height, width)` of the composed frame, known before any record arrives.

    Both panels fix their size at construction, so a window can be opened at the
    right size before the run produces anything -- and a capture is one shape
    for a whole run rather than changing halfway through.
    """
    return panel.height + readout.height, panel.width


def show(
    feed: Iterable[TickRecord],
    window: FrameWindow,
    *,
    panel: DomePanel,
    readout: PrivateComponentPanel,
    scene: Callable[[TickRecord], np.ndarray] | None = None,
    since: Callable[[TickRecord], int | None] | None = None,
) -> None:
    """Draw `feed` into `window`. The doing, with no argument parsing in it.

    `feed` is a live loop's records or a trace off disk and this cannot tell
    which, which is the whole of what *not two code paths* buys. `scene` is what
    the boundary band's picture comes from --
    :meth:`patchworks.surface.renderer.Renderer.frame` -- and `since` is the
    onset counter, both on exactly the footing
    :meth:`patchworks.surface.dome_panel.DomePanel.frames` takes them.

    **It always drains the feed.** A closed window, a closed panel or a panel
    that raised all stop the drawing and none of them stop the loop: a live feed
    is a run being driven, so returning early would stop the run rather than
    close a window (`10-the-demo-surface.md`, *Two windows*: closing the panel
    changes nothing but the view).

    This is the seam a CLI dispatcher calls (#119): everything it needs is an
    argument, and nothing here reads `sys.argv` or opens anything.
    """
    for record in feed:
        if panel.closed:
            continue
        if window.closed:
            panel.close()
            continue
        try:
            frame = compose(
                panel.frame(
                    record,
                    render=None if scene is None else scene(record),
                    since=None if since is None else since(record),
                ),
                readout.draw(record),
            )
        except Exception as failure:  # noqa: BLE001 -- see below
            # A display is not entitled to end a run by being wrong about it.
            # Both panels refuse rather than draw something reassuring -- a
            # non-finite `‖Δ private‖` puts every mark on the baseline, which is
            # a graph perfectly at rest -- and that refusal arriving here must
            # close the view and not the run. Warned rather than swallowed, and
            # once rather than per frame, because the panels are closed with it.
            panel.close()
            warnings.warn(
                f"the panel stopped drawing at tick {record.tick} and the run is "
                f"carrying on without it: {failure!r}",
                stacklevel=2,
            )
            continue
        window.show(frame)


def paced(feed: Iterable[TickRecord], fps: float) -> Iterator[TickRecord]:
    """`feed`, slowed to `fps` records a second. `fps <= 0` is as fast as it comes.

    Replay's half of *live and replay differ only in the feed*: a live feed
    arrives at the pace the world runs at, and a file arrives as fast as numpy
    can index it. So the pacing is **in the feed**, where the difference between
    the two already is, rather than in :func:`show`, which would then have a
    live branch and a replay branch after all.

    It never tries to catch up. A frame that took longer than its slot moves the
    next slot along instead of being followed by a burst, because the thing
    being watched is a run at ~10 Hz and a burst is not what it looked like.
    """
    if fps <= 0:
        yield from feed
        return
    period = 1.0 / fps
    due = time.monotonic()
    for record in feed:
        slack = due - time.monotonic()
        if slack > 0:
            time.sleep(slack)
        due = max(due + period, time.monotonic())
        yield record


def live(
    *,
    ticks: int = 100_000,
    seed: int = 0,
    split: str = "train",
    spec: DomeSpec = DEFAULT_SPEC,
    pitch: int = DEFAULT_PITCH,
    scale: int = DEFAULT_SCALE,
    capture: int = IMAGE_SIZE,
    edges: bool = False,
    raw: bool = False,
    hold: bool = True,
    save: str | Path | None = None,
) -> None:
    """Run the agent in the scene window with the panel open beside it.

    The scene window, the hands and the camera are
    :func:`~patchworks.surface.gestures.drive`'s entire -- this adds the second
    window and nothing else. On macOS that means `mjpython`, which is MuJoCo's
    requirement for the passive viewer and not this module's.

    `save` writes the run's trace, which is what :func:`replay` reads back.

    The agent is seeded rather than left to the global torch stream, so that
    `--seed` names the whole run: the same seed gives the same body, the same
    biases and therefore the same trail on replay.
    """
    dome = build_graph(spec)
    world = PlanarPushSandbox(split=split)
    try:
        agent = Agent(world, dome=dome, generator=torch.Generator().manual_seed(seed))
        recorder = Recorder(agent)
        panel = DomePanel(
            dome, measured_persistence(agent.sheaf), pitch=pitch, edges=edges, raw=raw
        )
        readout = readout_for(panel, dome)
        height, width = frame_size(panel, readout)
        with Renderer(size=capture) as scene, open_window(
            height, width, title="patchworks — the dome panel", scale=scale
        ) as window:
            show(
                drive(recorder, ticks, seed=seed),
                window,
                panel=panel,
                readout=readout,
                scene=scene.frame,
                since=OnsetCounter().count,
            )
            if hold and not window.closed:
                print("the run has ended; close the panel window to finish.")
                window.wait()
        if save is not None:
            print(f"trace written to {recorder.trace.save(save)}")
    finally:
        world.close()


def replay(
    path: str | Path,
    *,
    seed: int = 0,
    spec: DomeSpec = DEFAULT_SPEC,
    pitch: int = DEFAULT_PITCH,
    scale: int = DEFAULT_SCALE,
    capture: int = IMAGE_SIZE,
    fps: float = CAPTURE_HZ,
    edges: bool = False,
    raw: bool = False,
    hold: bool = True,
) -> None:
    """Replay a saved trace through the same window. No world, no agent.

    The only differences from :func:`live` are the feed and where the trail's
    persistences come from: a trace holds no biases, so a sheaf is built here to
    measure them, and `seed` is what makes that the run's own body. See this
    module's docstring.

    There is still a :class:`~patchworks.surface.renderer.Renderer`, because the
    boundary band draws the scene and a record holds state rather than frames --
    the scratch world it owns is the one the record restores into.
    """
    dome = build_graph(spec)
    sheaf = Sheaf(dome, generator=torch.Generator().manual_seed(seed))
    panel = DomePanel(
        dome, measured_persistence(sheaf), pitch=pitch, edges=edges, raw=raw
    )
    readout = readout_for(panel, dome)
    height, width = frame_size(panel, readout)
    trace = Trace.load(path)
    with Renderer(size=capture) as scene, open_window(
        height, width, title=f"patchworks — {Path(path).name}", scale=scale
    ) as window:
        show(
            paced(trace, fps),
            window,
            panel=panel,
            readout=readout,
            scene=scene.frame,
            since=OnsetCounter().count,
        )
        if hold and not window.closed:
            print(f"replayed {len(trace)} records; close the panel window to finish.")
            window.wait()


def main(argv: list[str] | None = None) -> None:
    """`mjpython -m patchworks.surface.watch` -- the panel, beside the scene.

    ::

        mjpython -m patchworks.surface.watch --ticks 2000 --save run.npz
        python   -m patchworks.surface.watch --replay run.npz

    Live opens two windows: MuJoCo's passive viewer over the arena, with the
    hands bound to it (`--help` on
    :func:`~patchworks.surface.gestures.main` for the gestures), and the panel
    beside it. On macOS the passive viewer needs `mjpython`; replay opens no
    scene window and runs under plain `python`.

    **The parsing is here and the doing is in :func:`live` and :func:`replay`.**
    #119 is the dispatcher this becomes a subcommand of, and what it will call
    is those two functions -- this parser is a stand-in for its `watch`
    subcommand's, not a CLI framework for it to build on.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="patchworks watch",
        description="The dome panel, in a window: live beside the scene, or from a trace.",
    )
    parser.add_argument(
        "--replay",
        metavar="TRACE.npz",
        help="replay a saved trace instead of running live; no scene window, no agent",
    )
    parser.add_argument("--ticks", type=int, default=100_000, help="live: how many ticks to run")
    parser.add_argument("--seed", type=int, default=0, help="the run's seed; on replay, the body whose persistences the trail decays at")
    parser.add_argument("--split", default="train", help="live: the sandbox's task split")
    parser.add_argument("--save", metavar="TRACE.npz", help="live: write the run's trace here")
    parser.add_argument("--fps", type=float, default=CAPTURE_HZ, help="replay: records a second; 0 for as fast as they come")
    parser.add_argument("--pitch", type=int, default=DEFAULT_PITCH, help="pixels per lattice slot")
    parser.add_argument("--scale", type=int, default=DEFAULT_SCALE, help="screen pixels per frame pixel")
    parser.add_argument("--capture", type=int, default=IMAGE_SIZE, help="the scene render the boundary band is tiled from, in pixels")
    parser.add_argument("--edges", action="store_true", help="draw the thresholded edge overlay")
    parser.add_argument("--raw", action="store_true", help="the un-normalised prediction-error map, behind the debug flag")
    parser.add_argument("--no-hold", action="store_true", help="close the panel window when the feed ends instead of leaving the last frame up")
    arguments = parser.parse_args(argv)

    common = dict(
        seed=arguments.seed,
        pitch=arguments.pitch,
        scale=arguments.scale,
        capture=arguments.capture,
        edges=arguments.edges,
        raw=arguments.raw,
        hold=not arguments.no_hold,
    )
    if arguments.replay is not None:
        replay(arguments.replay, fps=arguments.fps, **common)
    else:
        live(ticks=arguments.ticks, split=arguments.split, save=arguments.save, **common)


if __name__ == "__main__":
    main()
