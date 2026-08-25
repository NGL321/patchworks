"""The demo surface: the tick record, and the renderer that reads it.

`docs/spec/10-the-demo-surface.md` is the whole of what this package
implements, and its opening constraint governs every module in it:

    **Nothing in this file is part of the architecture.** The panel reads
    privileged state -- prediction error, private components, edge
    disagreement -- on exactly the footing `03-the-sandbox.md` gives `info`:
    for looking at, never fed back. No cell reads anything the surface
    computes, and switching the whole surface off must change no trajectory.

Two modules, and the seam between them is the record:

* :mod:`patchworks.surface.record` -- the tick record. The snapshot/restore
  contract, unchanged, plus per-cell prediction error and `‖Δ private‖`, plus
  the markers the human's hands drop. Not a new format.
* :mod:`patchworks.surface.renderer` -- the one renderer. It consumes an
  iterable of records; a live run and a file off disk are two iterables, not
  two code paths.
* :mod:`patchworks.surface.dome_panel` -- the second window's dome: bands
  stacked with the sensorimotor boundary at the bottom, prediction error
  normalised per cell as colour, and a trail decaying at each cell's own
  measured persistence. It reads a record and draws marks; it holds no agent,
  no sheaf and no world, so closing it changes nothing but the view.

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
    "BandLayout",
    "DomePanel",
    "Event",
    "EventKind",
    "Recorder",
    "Renderer",
    "Slot",
    "TickRecord",
    "Trace",
    "colormap",
    "measured_persistence",
]
