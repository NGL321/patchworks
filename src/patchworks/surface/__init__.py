"""The demo surface: the tick record, and the renderer that reads it.

`docs/spec/10-the-demo-surface.md` is the whole of what this package
implements, and its opening constraint governs every module in it:

    **Nothing in this file is part of the architecture.** The panel reads
    privileged state -- residuals, private components, edge disagreement -- on
    exactly the footing `03-the-sandbox.md` gives `info`: for looking at, never
    fed back. No cell reads anything the surface computes, and switching the
    whole surface off must change no trajectory.

Two modules, and the seam between them is the record:

* :mod:`patchworks.surface.record` -- the tick record. The snapshot/restore
  contract, unchanged, plus per-cell prediction error and `‖Δ private‖`, plus
  the markers the human's hands drop. Not a new format.
* :mod:`patchworks.surface.renderer` -- the one renderer. It consumes an
  iterable of records; a live run and a file off disk are two iterables, not
  two code paths.

Not imported by :mod:`patchworks`; import it directly, as with
:mod:`patchworks.sandbox`.
"""

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
    "Event",
    "EventKind",
    "Recorder",
    "Renderer",
    "TickRecord",
    "Trace",
]
