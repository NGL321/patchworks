"""The demo surface: the tick record, and what reads it.

`docs/spec/10-the-demo-surface.md` is the whole of what this package
implements, and its opening constraint governs every module in it:

    **Nothing in this file is part of the architecture.** The panel reads
    privileged state -- prediction error, private components, edge
    disagreement -- on exactly the footing `03-the-sandbox.md` gives `info`:
    for looking at, never fed back. No cell reads anything the surface
    computes, and switching the whole surface off must change no trajectory.

Four modules, and the seam between them is the record:

* :mod:`patchworks.surface.record` -- the tick record. The snapshot/restore
  contract, unchanged, plus per-cell prediction error and `‖Δ private‖`, plus
  the markers the human's hands drop. Not a new format.
  Plus, since #94, the two arrays a boundary cell's marks are drawn from:
  per-edge disagreement, and the actuator's commanded and applied rows.
* :mod:`patchworks.surface.renderer` -- the one renderer. It consumes an
  iterable of records; a live run and a file off disk are two iterables, not
  two code paths.
* :mod:`patchworks.surface.dome_panel` -- the second window's dome: bands
  stacked with the sensorimotor boundary at the bottom, prediction error
  normalised per cell as colour, and a trail decaying at each cell's own
  measured persistence. It reads a record and draws marks; it holds no agent,
  no sheaf and no world, so closing it changes nothing but the view.
  A boundary cell's marks are drawn from edge disagreement and from the world
  instead -- the tiled render at L0, the somatomotor strip and its decomposed
  actuator, the drive mark, and the thresholded edge overlay -- because a
  boundary cell runs no body and has no prediction error to draw.
* :mod:`patchworks.surface.private_component` -- the private-component panel:
  `‖Δ(private component)‖` per cell against hop distance from the sensorimotor
  rim, drawn as its own panel so that it can disagree with the claim it tests.
* :mod:`patchworks.surface.onset` -- what the surface owes onset latency: the
  hands, bound so that firing one drops its marker, and the tick counter the
  motor strip runs from the most recent one.
* :mod:`patchworks.surface.gestures` -- the gestures those hands are bound to,
  and the live viewer they are bound in: ctrl-drag a link or a puck, click a
  puck and then a zone, `r` to rearrange, and the number keys as the headless
  and scripted path. The scene window is MuJoCo's passive viewer, so the
  picking and the drag are inherited rather than re-implemented.

Not imported by :mod:`patchworks`; import it directly, as with
:mod:`patchworks.sandbox`.
"""

from patchworks.surface.dome_panel import (
    DEFAULT_PITCH,
    BandLayout,
    DomePanel,
    Slot,
    colormap,
    measured_persistence,
)
from patchworks.surface.gestures import (
    IMPULSE_PER_METRE,
    MINIMUM_DRAG,
    OUT_OF_PLANE_TOLERANCE,
    TOP_DOWN_ELEVATION,
    Drag,
    Gestures,
    Pointer,
    Referent,
    ReferentKind,
    drive,
    hold_top_down,
)
from patchworks.surface.onset import Hands, OnsetCounter
from patchworks.surface.private_component import (
    PANEL_HEIGHT,
    PANEL_WIDTH,
    PrivateComponentPanel,
    Scatter,
    hop_distance,
)
from patchworks.surface.record import (
    CAPTURE_EVERY,
    CAPTURE_HZ,
    Event,
    EventKind,
    Recorder,
    TickRecord,
    Trace,
)
from patchworks.surface.renderer import Renderer

__all__ = [
    "CAPTURE_EVERY",
    "CAPTURE_HZ",
    "DEFAULT_PITCH",
    "IMPULSE_PER_METRE",
    "MINIMUM_DRAG",
    "OUT_OF_PLANE_TOLERANCE",
    "PANEL_HEIGHT",
    "PANEL_WIDTH",
    "TOP_DOWN_ELEVATION",
    "BandLayout",
    "DomePanel",
    "Drag",
    "Event",
    "EventKind",
    "Gestures",
    "Hands",
    "OnsetCounter",
    "Pointer",
    "PrivateComponentPanel",
    "Recorder",
    "Referent",
    "ReferentKind",
    "Renderer",
    "Scatter",
    "Slot",
    "TickRecord",
    "Trace",
    "colormap",
    "drive",
    "hold_top_down",
    "hop_distance",
    "measured_persistence",
]
