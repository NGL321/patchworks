"""The private-component panel: `‖Δ(private component)‖` against hop distance.

`docs/spec/10-the-demo-surface.md`, *The private-component panel*, is the whole
of what this module implements, over `docs/spec/05-timescales.md`'s
*Demonstrating it*. It is displayed during each of the three events.

**This scatter is the picture, and it is no longer the criterion.** `08`'s
depth criterion is the conduction ratio `tau_hat_c / world_loop(c) >= 1`
(`docs/adr/0026-...`, `docs/adr/0027-...`), because the amplitude reading below
cannot fail: the channel's own attenuation supplies a falling scatter whether
or not anything is retained. The panel keeps `‖Δ private‖` because it is what a
viewer can read, and `10` now also owes a `tau_hat` scatter with its bar drawn
at `world_loop(c)`. That scatter is not built here yet. The divisor was
`|loop(c)|` until #383 moved it, and the bar this panel has to draw is
per-cell rather than one step per depth column, because `world_loop(c)`'s
per-level ranges overlap where `|loop(c)|` was one value per level. The loop
lengths themselves are no longer #99's to build: `benchmarks/loop_length.py`
enumerates both off the mask and `detectability.py` imports the divisor (#398).

**A second panel, below the dome.** One mark per predicting cell: how far that
cell's private component moved this tick, against the cell's hop distance from
the sensorimotor rim.

**It stays off the dome's marks deliberately.** Folding it in as brightness or
dot size beside prediction-error-as-hue puts two quantities on one mark and
makes neither readable -- and readability is the whole point, because this
panel has to be able to *disagree*: a `tau_hat` scatter flat across depth sits
under its bar at every depth, and `08` counts that a failure even when every
recovery looks perfect. So this panel reads exactly one of a record's arrays --
:attr:`~patchworks.surface.record.TickRecord.private_delta` -- and the dome's
channel (#93) is not reachable from anything here.

**The readout is a measured trace.** What a mark carries is what the run
actually did to `H⁰` between two consecutive ticks: the recorder differences
the node stalk's private component and takes a norm
(:meth:`~patchworks.surface.record.Recorder.observe`), and this panel places
that number and applies no factor to it. There is no eigenvalue in it and no
stored rate -- **it was never an eigenvalue, and it must not become one**
(`05-timescales.md`, *Demonstrating it*). That is also why being a distribution
costs the readout nothing: a measured trace of how far private content moved is
already an average over whatever activation regions the cell passed through.

**Nothing here is part of the architecture.** A panel is built on a
:class:`~patchworks.graph.Dome` and reads records; it holds no agent, no sheaf
and no world, so there is nothing for a cell to read back and switching it off
changes no trajectory (:mod:`patchworks.surface`).

**How the marks are drawn is a display choice, not a spec commitment.**
`10-the-demo-surface.md`'s *Known exposure* keeps the drawing library a
build-time choice, so :meth:`PrivateComponentPanel.draw` rasterises the scatter
into a plain `uint8` array with numpy and nothing else, the way
:class:`~patchworks.surface.renderer.Renderer` hands back the scene. Whoever
composes the two windows at capture time decides what to do with them.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable, Iterator

import numpy as np

from patchworks.graph import CellKind, Dome

from .record import TickRecord

__all__ = [
    "PANEL_HEIGHT",
    "PANEL_WIDTH",
    "PrivateComponentPanel",
    "Scatter",
    "hop_distance",
]

#: The panel's default size in pixels. A display default and only that -- a
#: panel is built at whatever size the window or the capture wants, the way a
#: :class:`~patchworks.surface.renderer.Renderer` is, and nothing about the
#: readout changes with it.
PANEL_WIDTH = 480
PANEL_HEIGHT = 240

#: The panel's ink, in RGB. Display defaults too: the encoding is the mark's
#: *position*, so no quantity is carried by any of these.
BACKGROUND = (16, 17, 21)
GUIDE = (44, 47, 56)
AXIS = (96, 102, 116)
INK = (232, 234, 240)


def hop_distance(dome: Dome) -> np.ndarray:
    """`[predicting cells]` int: hop distance from the sensorimotor rim.

    Rows are indexed by :attr:`~patchworks.graph.Dome.predicting`, matching a
    record's arrays.

    **Measured on the graph, not read off the index.** The construction layout
    is an index with no runtime role (`docs/spec/06-graph-topology.md`), and
    while its level and this distance agree -- `tests/test_private_component.py`
    asserts they do, over the whole population -- what the spec plots against is
    *hop distance*, so hop distance is what is counted: a breadth-first sweep
    outward from the rim over the edges the dome actually has.

    The rim is the sensory and actuator boundary cells together (`CONTEXT.md`,
    *Sensorimotor rim*). The internal rim is excluded, and that exclusion is the
    whole of what makes this axis abstraction rather than proximity to something
    outside: a drive attaches at the **apex**, so counting from it too would
    make the deepest cells the shallowest.

    **Excluded from the walk, not only from its start.** A drive boundary cell
    is adjacent to every apex cell it attaches to, so left traversable it is a
    two-hop bridge between them and a depth counted through it is a distance
    from the *internal* rim wearing the other rim's name. On the default spec
    the two agree -- every apex cell is the same distance out, so the bridge
    shortens nothing -- which is exactly why it is barred here rather than left
    to be noticed on the first spec where it does not.
    """
    internal_rim = {
        cell.id for cell in dome.cells if cell.kind is CellKind.DRIVE
    }
    rim = [
        cell.id
        for cell in dome.cells
        if cell.is_boundary and cell.id not in internal_rim
    ]
    if not rim:
        raise ValueError(
            "this dome has no sensorimotor rim to count hops from: no sensory or "
            "actuator boundary cell. Abstraction is distance from the rim "
            "(CONTEXT.md, Sensorimotor rim), so without one there is no axis."
        )
    distance = [-1] * len(dome.cells)
    frontier: deque[int] = deque()
    for cell_id in rim:
        distance[cell_id] = 0
        frontier.append(cell_id)
    while frontier:
        cell_id = frontier.popleft()
        for neighbour in dome.neighbours(cell_id):
            if neighbour in internal_rim or distance[neighbour] >= 0:
                continue
            distance[neighbour] = distance[cell_id] + 1
            frontier.append(neighbour)
    unreachable = [cell_id for cell_id in dome.predicting if distance[cell_id] < 0]
    if unreachable:
        raise ValueError(
            f"{len(unreachable)} predicting cells are not reachable from the "
            f"sensorimotor rim (first: {unreachable[0]}), so they have no hop "
            "distance to be plotted against. A dome not connected to its own rim "
            "is a construction fault, not a display one."
        )
    return np.array([distance[cell_id] for cell_id in dome.predicting], dtype=np.int64)


@dataclass(frozen=True, eq=False)
class Scatter:
    """One tick's marks: what moved, and how deep it sits.

    The numbers behind :meth:`PrivateComponentPanel.draw`, handed out because
    the drawing is not the only consumer -- a falsification sweep reads them
    and renders no frame at all. `08`'s depth *criterion* is not read off
    these: it is the conduction ratio, on a per-cell decay time these arrays do
    not carry.

    `eq=False` for :class:`~patchworks.surface.record.TickRecord`'s reason: the
    fields are arrays, so a generated `__eq__` would raise rather than compare.
    """

    tick: int
    """The tick the record was captured on."""

    depth: np.ndarray
    """`[cells]` int: hop distance from the sensorimotor rim. Fixed at
    construction -- the same array every tick."""

    moved: np.ndarray
    """`[cells]`: `‖Δ(private component)‖` on this tick, exactly as the record
    carries it. No normalisation, no per-cell factor, nothing stored."""


class PrivateComponentPanel:
    """The panel: a scatter of `‖Δ private‖` against hop distance, per tick.

    ::

        panel = PrivateComponentPanel(dome)
        for record in recorder.watch(ticks=600, seed=0):
            image = panel.draw(record)

    Built on the dome, because both halves of a mark's position come from
    there: the depth axis is a construction quantity, and so is the private
    component the record's array was taken through -- the node-stalk directions
    masked out on every incident edge, known at construction, a **fixed
    projection computed per tick**
    (:attr:`~patchworks.graph.Dome.private_projection`).

    Live and replay are one code path here as they are for the renderer: a
    record is a record.
    """

    def __init__(
        self,
        dome: Dome,
        *,
        width: int = PANEL_WIDTH,
        height: int = PANEL_HEIGHT,
    ) -> None:
        for name, size in (("width", width), ("height", height)):
            if isinstance(size, bool) or not isinstance(size, int) or size < 16:
                raise ValueError(
                    f"a panel is at least 16 pixels of {name}; got {size!r}"
                )
        self.width = width
        self.height = height
        #: `[cells]`: the depth axis, counted once. It is a property of the
        #: graph, so no record can move a mark sideways. Sealed for the reason
        #: :func:`~patchworks.sandbox.state.snapshot` seals a state and the
        #: dome hands its masks out as copies: every :class:`Scatter` shares
        #: this array, and one consumer writing into it would move another's
        #: marks.
        self.depth = hop_distance(dome)
        self.depth.flags.writeable = False
        self._cells = len(dome.predicting)
        pad = max(3, min(width, height) // 12)
        self._top = pad
        self._baseline = height - pad - 1
        self._left = pad
        self._right = width - pad - 1
        self._x, self._columns = _columns(self.depth, self._left, self._right)

    def __repr__(self) -> str:
        return (
            f"PrivateComponentPanel({self._cells} cells, "
            f"depth {self.depth.min()}-{self.depth.max()}, "
            f"{self.width}x{self.height})"
        )

    def scatter(self, record: TickRecord) -> Scatter:
        """This tick's marks. Reads one array of the record and places it.

        `‖Δ private‖` arrives measured -- the recorder differenced two
        consecutive ticks of the node stalk's private component and took the
        norm -- and is passed through untouched. A panel that scaled it per
        cell would be doing to this readout what the dome does to prediction
        error, and for the opposite reason: the dome normalises per cell so
        that *which level lit up* means something, while here the comparison
        **across** depth is the measurement, and a per-cell scale would flatten
        exactly the slope the demo is judged on.
        """
        moved = np.asarray(record.private_delta)
        if moved.shape != (self._cells,):
            raise ValueError(
                f"this panel was built on a dome of {self._cells} predicting cells "
                f"and this record carries {moved.shape}. A record and a panel from "
                "two different domes have no common row order."
            )
        finite = np.isfinite(moved)
        if not finite.all():
            rows = np.flatnonzero(~finite)
            raise ValueError(
                f"tick {record.tick} carries a non-finite `‖Δ private‖` at "
                f"{rows.size} of {self._cells} cells (first: row {rows[0]}, "
                f"{moved[rows[0]]}). The run diverged, and this readout is refused "
                "rather than drawn: every mark is placed against this tick's own "
                "peak, and a peak that is NaN puts the whole population back on "
                "the baseline -- a graph perfectly at rest, which is the most "
                "reassuring picture this panel can draw and the exact opposite of "
                "what happened. The panel exists to be able to disagree "
                "(docs/spec/10-the-demo-surface.md, The private-component panel), "
                "so it says so instead. The raw array is on the record for "
                "whoever is diagnosing the divergence."
            )
        return Scatter(tick=record.tick, depth=self.depth, moved=moved)

    def draw(self, record: TickRecord) -> np.ndarray:
        """`[height, width, 3]` uint8: the scatter, drawn.

        **Its own image.** What comes back is a panel beside the dome's and not
        inside it -- the deliberate separation of *The private-component panel*,
        made structural: this method is handed no prediction error and no dome
        mark, so there is nothing here to fold one into.

        **A diverged run is refused, not drawn.** A non-finite `‖Δ private‖`
        would place every mark on the baseline through the scale below -- a
        graph perfectly at rest -- so :meth:`scatter` refuses it first. A
        display that cannot draw what happened says so; it does not draw the
        calmest thing it has.

        **The vertical scale is the tick's own.** Marks are placed against the
        largest `‖Δ private‖` on screen this tick rather than against a
        constant, for the reason `10-the-demo-surface.md` gives for the edge
        threshold -- a hand-set constant would make the picture an artifact of
        the constant, and `‖Δ private‖` is a raw norm with no bound to pick one
        from. The panel's claim survives it: a common factor moves every mark
        together, so a scatter that slopes still slopes and a flat one is still
        flat. What a shared scale does cost is the absolute magnitude, which is
        in :meth:`scatter` for anyone measuring rather than watching.
        """
        marks = self.scatter(record)
        image = np.empty((self.height, self.width, 3), dtype=np.uint8)
        image[:, :] = BACKGROUND
        for x in self._columns:
            image[self._top : self._baseline, x] = GUIDE
        image[self._baseline, self._left : self._right + 1] = AXIS

        span = self._baseline - self._top
        peak = float(marks.moved.max()) if marks.moved.size else 0.0
        # A tick on which nothing moved is drawn flat on the baseline rather
        # than scaled by a zero peak: a graph at rest is a picture this panel
        # is allowed to show, and it is not the same picture as a flat scatter
        # sitting high.
        raised = (
            np.round(marks.moved / peak * span).astype(np.int64)
            if peak > 0
            else np.zeros(marks.moved.shape, dtype=np.int64)
        )
        rows = self._baseline - np.clip(raised, 0, span)
        for x, row in zip(self._x, rows):
            image[max(row - 1, 0) : row + 2, max(x - 1, 0) : x + 2] = INK
        return image

    def frames(self, feed: Iterable[TickRecord]) -> Iterator[np.ndarray]:
        """One panel per record of `feed`, in order.

        `feed` is a live recorder's
        :meth:`~patchworks.surface.record.Recorder.watch` or a
        :class:`~patchworks.surface.record.Trace` off disk, and this method
        cannot tell which -- the same seam
        :meth:`~patchworks.surface.renderer.Renderer.frames` has, so the scene
        and the panel decimate together and are composed at capture time.
        """
        for record in feed:
            yield self.draw(record)


def _columns(
    depth: np.ndarray, left: int, right: int
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Where each cell's mark sits horizontally, and where each depth's does.

    One column per hop distance present, evenly spaced, the rim at the left and
    the apex at the right. Cells sharing a depth are spread across their column
    in row order rather than stacked on one abscissa: a column holds 256 cells
    at L1 and 8 at the apex, and a scatter that overplots 256 marks onto one
    pixel column would report the taper's shape instead of the readout.
    """
    levels = sorted({int(d) for d in depth})
    reach = right - left
    if len(levels) == 1:
        centres = {levels[0]: (left + right) // 2}
        half = reach // 2
    else:
        centres = {
            level: left + (reach * i) // (len(levels) - 1)
            for i, level in enumerate(levels)
        }
        half = int(reach / (len(levels) - 1) * 0.4)
    x = np.empty(depth.shape, dtype=np.int64)
    for level in levels:
        rows = np.flatnonzero(depth == level)
        offsets = (
            np.zeros(1) if rows.size == 1 else np.linspace(-half, half, rows.size)
        )
        x[rows] = np.clip(
            np.round(centres[level] + offsets).astype(np.int64), left, right
        )
    return x, tuple(centres[level] for level in levels)
