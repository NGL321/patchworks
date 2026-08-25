"""One renderer, over a tick record.

`docs/spec/10-the-demo-surface.md`, *The trace*: **live mode feeds it that
record directly at ~10 Hz; replay mode feeds it from disk. Not two code
paths.** That sentence is the shape of this module. :meth:`Renderer.frames`
takes an iterable of :class:`~patchworks.surface.record.TickRecord`, and the
two modes are two iterables:

    live    renderer.frames(recorder.watch(ticks=600, seed=0))
    replay  renderer.frames(Trace.load(path))

There is no live branch and no replay branch to drift apart, and
`tests/test_surface.py` asserts the two produce identical frames from the same
run.

**State, not frames.** A record holds no image (:mod:`patchworks.surface.record`),
so the scene is re-rendered here, from MuJoCo, offscreen, at whatever size this
renderer was built for. Capture resolution is therefore chosen when rendering
rather than baked into the recording -- one file, and the README capture, a
falsification sweep and a debugging pass all read it at whatever size each
wants.

**The scene the renderer draws is not the scene the run is in.** A renderer
owns its own sandbox and restores each record into that, never into the env the
agent is living in. Restoring into a running world would rewind it, which is
the one way a display could change a trajectory; owning a second world is what
makes "switching the surface off changes no trajectory" true of a renderer that
is being watched live, mid-run. It costs one more MuJoCo model and one more GL
context, and nothing else -- the two worlds share no state, and the scratch one
is never stepped.

**The panel is not here.** `10-the-demo-surface.md`'s dome bands, its
somatomotor strip and its private-component panel are #93, #94 and #95; what
they add is marks drawn from the arrays a record already carries. This module
owns the scene window and the seam the panel plugs into.
"""

from __future__ import annotations

from typing import Iterable, Iterator

import numpy as np

from patchworks.sandbox.env import IMAGE_SIZE, PlanarPushSandbox
from patchworks.sandbox.state import restore

from .record import TickRecord

__all__ = ["Renderer"]


class Renderer:
    """Draws the scene of a tick record, at a size chosen here.

    ::

        with Renderer(size=256) as renderer:
            frames = list(renderer.frames(Trace.load(path)))

    The size is this renderer's, not the record's, and nothing about the record
    changes with it: build a second renderer at another size and the same file
    gives the same run at that size.

    **One consumer constrains the number.** A frame fed to the dome panel's
    boundary band is cut along the patch lattice, so a size the lattice does
    not divide is refused there rather than resampled
    (:meth:`patchworks.surface.dome_panel.DomePanel.frame`). Nothing here
    enforces that -- a renderer has no lattice -- but the constraint is on this
    side of the seam, where the number is chosen: with a 16x16 lattice, 64, 128
    and 256 tile and 100 does not.
    """

    def __init__(self, *, size: int = IMAGE_SIZE, seed: int = 0) -> None:
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            raise ValueError(
                f"a capture resolution is a positive number of pixels; got {size!r}"
            )
        self.size = size
        #: The scratch world every record is restored into. Its own, never the
        #: agent's -- see this module's docstring. `render_obs=False` because
        #: nothing here reads an observation: this world is never stepped, and
        #: the only picture wanted of it is `render()`'s.
        self._scene = PlanarPushSandbox(
            split="any", image_size=size, render_mode="rgb_array", render_obs=False
        )
        # The arena declares its own offscreen framebuffer, and MuJoCo refuses
        # a render larger than it. Refused here rather than there: `frames()`
        # is lazy, so a live feed has already arranged the world and started
        # the run before the first frame is drawn, and a resolution that cannot
        # be drawn should not cost a run to find out.
        limit = min(
            int(self._scene.model.vis.global_.offwidth),
            int(self._scene.model.vis.global_.offheight),
        )
        if size > limit:
            self._scene.close()
            raise ValueError(
                f"the arena's offscreen framebuffer is {limit} pixels square and "
                f"MuJoCo will not render {size}. Capture smaller, or raise "
                "<global offwidth= offheight=> in the arena -- which is the "
                "sandbox's file, not the surface's."
            )
        # A restore needs a world that has been arranged: `restore()` refuses
        # one that has never held a task, because a restore rewinds a run
        # rather than starting one. The layout this draws is overwritten by the
        # first record, and the seed only keeps that throwaway draw repeatable.
        self._scene.reset(seed=seed, options={"reset_arm": True})

    def __enter__(self) -> "Renderer":
        return self

    def __exit__(self, *exception) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"Renderer(size={self.size})"

    def frame(self, record: TickRecord) -> np.ndarray:
        """`[size, size, 3]` uint8: the scene that record was taken from.

        The record's state goes into the scratch world and the top-down camera
        draws it. Everything a restore restores is restored here too -- the
        goal zone is lit from the task, the friction field is re-derived from
        where the pucks are -- so the picture is the world as it stood on that
        tick and not an approximation of it.
        """
        restore(self._scene, record.state)
        return self._scene.render()

    def frames(self, feed: Iterable[TickRecord]) -> Iterator[np.ndarray]:
        """One frame per record of `feed`, in order.

        `feed` is a live recorder's :meth:`~patchworks.surface.record.Recorder.watch`
        or a :class:`~patchworks.surface.record.Trace` off disk, and this method
        cannot tell which -- which is the whole of what "not two code paths"
        buys. Lazy, so a live feed is drawn as the run produces it rather than
        after it.
        """
        for record in feed:
            yield self.frame(record)

    def close(self) -> None:
        """Close the scratch world and its GL context."""
        self._scene.close()
