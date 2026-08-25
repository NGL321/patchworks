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
* :mod:`patchworks.surface.renderer` -- the one renderer. It consumes an
  iterable of records; a live run and a file off disk are two iterables, not
  two code paths.
* :mod:`patchworks.surface.private_component` -- the private-component panel:
  `‖Δ(private component)‖` per cell against hop distance from the sensorimotor
  rim, drawn as its own panel so that it can disagree with the claim it tests.
* :mod:`patchworks.surface.onset` -- what the surface owes onset latency: the
  hands, bound so that firing one drops its marker, and the tick counter the
  motor strip runs from the most recent one.

Not imported by :mod:`patchworks`; import it directly, as with
:mod:`patchworks.sandbox`.
"""

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
    "PANEL_HEIGHT",
    "PANEL_WIDTH",
    "Event",
    "EventKind",
    "Hands",
    "OnsetCounter",
    "PrivateComponentPanel",
    "Recorder",
    "Renderer",
    "Scatter",
    "TickRecord",
    "Trace",
    "hop_distance",
]
