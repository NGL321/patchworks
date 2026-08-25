"""The dome panel: stacked bands, prediction error as colour, and the trail.

`docs/spec/10-the-demo-surface.md`, *The dome panel*, is the whole of what this
module implements, over the record :mod:`patchworks.surface.record` already
defines. Three things, and the first is the frame the other two are drawn in:

* **Stacked bands, the sensorimotor boundary at the bottom and the apex at the
  top**, one band per level, each at its own lattice shape, every position
  taken from the construction index. Depth reads as height, which is the whole
  reason for the layout -- *recovered at the appropriate level* becomes
  something a bystander watches rather than something a caption asserts.
* **Colour is prediction error, normalised per cell** against that cell's own
  running statistics. Raw norms are not comparable across the dome -- the
  sensory funnel carries 12,288 numbers at the base and eight core cells carry
  32 dimensions each -- so a raw map would show the taper's shape and nothing
  else. The raw map stays available behind :attr:`DomePanel.raw`, which is the
  debug flag that section keeps.
* **The trail**: a cell's glow decays at that cell's own **measured
  persistence** -- `05-timescales.md`'s estimate, taken from the run's own body
  and biases by :func:`measured_persistence`, and never a second definition of
  timescale. It is deliberately **not** driven by `‖Δ private‖`: that would
  make the display's decay and the claim the display tests the same number, so
  the panel could never contradict the thesis. Nothing in this module reads
  :attr:`~patchworks.surface.record.TickRecord.private_delta`, and
  `tests/test_dome_panel.py` asserts that both ways.

**Nothing here is part of the architecture**, and for this module that is
structural rather than careful. A panel holds a :class:`~patchworks.graph.Dome`
-- read-only, fixed at construction -- an array of persistences and its own
running statistics. It holds no agent, no sheaf, no env and no recorder, so
there is no route from anything a cell's computation is handed to anything
computed here, and closing the panel is the view ending and nothing else.

**What this ticket draws and what it leaves.** The bands are the frame for
every mark the panel will ever carry, and #93 fills in one channel of it: the
predicting cells' prediction error. A boundary cell's slot is laid out and left
empty on purpose -- boundary cells run no body and make no prediction
(ADR-0006), so colouring them on this map would be a fabrication in the
largest, most eye-catching band on screen. The marks they do get -- the tiled
render in the boundary band, the somatomotor strip, the drive mark, and the
thresholded edges -- are drawn from edge disagreement and from the world, and
they are a later ticket's.

**The panel is not the toolkit.** `10-the-demo-surface.md` names no drawing
library and this module needs none: a frame is a `[height, width, 3]` uint8
array, the same thing :meth:`patchworks.surface.renderer.Renderer.frame` hands
back for the scene, and what puts one on screen or into a capture is the
caller's business. The bitmap font below exists for the same reason -- the
warm-up notice has to reach the screen, and the dependency set has nothing that
draws text.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence

import numpy as np
import torch

from patchworks.bias_selection import DEFAULT_BURN_IN, DEFAULT_TICKS, measure
from patchworks.graph import Dome
from patchworks.tick import Sheaf

from .record import TickRecord

__all__ = [
    "BandLayout",
    "DEFAULT_PITCH",
    "DomePanel",
    "Slot",
    "colormap",
    "measured_persistence",
]

#: Pixels per lattice slot, and the panel's only size knob. A band's shape is
#: fixed by the construction index, so what a display chooses is how big a cell
#: is drawn -- not how many there are, and not where they go.
DEFAULT_PITCH = 8

#: Empty slots between two columns of one band, and between two bands. The
#: somatomotor column reads as beside the vision lattice rather than part of
#: it, and a band reads as a level rather than as more of the level below.
_COLUMN_GAP = 1
_BAND_GAP = 1

#: The panel's ground, and the slot of a cell this ticket draws no mark for.
#: Distinct from the ground so that the frame is visible -- an empty slot is a
#: cell whose mark is a later ticket's, not an absent cell.
_BACKGROUND = (8, 8, 10)
_EMPTY = (26, 26, 30)
_NOTICE_INK = (216, 216, 220)

#: A cell whose prediction error this tick is not a number. Deliberately a
#: colour the colormap cannot produce -- no colour on that ramp carries more
#: green than red -- so that *no reading* is never mistaken for a reading. A
#: diverging cell is the event this display exists to make visible, and the
#: arithmetic that would otherwise absorb it (`NaN > 0` is False, so it would
#: draw at the calmest stop on the ramp) would render it as the quietest cell
#: in the dome.
_NO_READING = (0, 230, 230)


# -- colour -----------------------------------------------------------------

#: The colormap's anchors, dark to bright. Monotone in luminance, which is the
#: property the encoding actually rests on: calm reads dark and lit reads
#: bright, at a glance, in a capture that may end up greyscale. `10`'s other
#: marks are specified to be drawn on *the same colormap*, so it is one
#: function here rather than a constant inside the draw.
_ANCHORS = (
    (0.00, (0, 0, 4)),
    (0.25, (87, 16, 110)),
    (0.50, (188, 55, 84)),
    (0.75, (249, 142, 9)),
    (1.00, (252, 255, 164)),
)


def colormap(values: np.ndarray | float) -> np.ndarray:
    """`[..., 3]` uint8 for values in `[0, 1]`, clipped outside it."""
    v = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    stops = np.array([stop for stop, _ in _ANCHORS])
    channels = np.array([list(colour) for _, colour in _ANCHORS], dtype=np.float64)
    out = np.empty(v.shape + (3,), dtype=np.float64)
    for channel in range(3):
        out[..., channel] = np.interp(v, stops, channels[:, channel])
    return np.rint(out).astype(np.uint8)


# -- the frame --------------------------------------------------------------


@dataclass(frozen=True)
class Slot:
    """Where one cell's mark goes, in lattice slots rather than pixels.

    `row` counts from the top of the panel, so the apex band holds row 0 and
    the sensorimotor boundary holds the last one. Nothing here is a distance:
    a slot is the construction index, laid out.
    """

    cell: int
    level: int
    column_name: str
    row: int
    column: int


class BandLayout:
    """One band per level, stacked, each at its own lattice shape.

    L0 at the bottom and the apex at the top (`docs/spec/10-the-demo-surface.md`,
    *The dome panel*), because **depth reads as height**: the vertical axis is
    already the axis `05-timescales.md` asks its readout to be plotted against,
    hop distance from the sensorimotor rim.

    Every position is read off :class:`~patchworks.graph.CellIndex` and nothing
    else, which is what keeps this a drawing of the **construction layout** --
    an index, never an embedding. A force-directed drawing is ruled out
    permanently for that reason, and the rule here is the opposite of one: a
    two-dimensional index is drawn as the grid it is, and a one-dimensional one
    is drawn along the band. Concentric rings are the recorded fallback if the
    dome is ever abandoned, and they are not this class.

    Within a band the columns of the construction layout sit side by side in
    index order, so the somatomotor column is drawn **beside** the vision
    lattice, which is where the cluster actually attaches. A one-dimensional
    column fills the band's height before it takes a second slot column, so it
    stands beside a lattice that has height and lies flat along a core level
    that does not -- the core is not a lattice (`CONTEXT.md`, *Core*) and is
    drawn as the row of indices it is.
    """

    def __init__(self, dome: Dome, *, pitch: int = DEFAULT_PITCH) -> None:
        if isinstance(pitch, bool) or not isinstance(pitch, int) or pitch < 2:
            raise ValueError(
                f"a slot is a whole number of pixels across, at least 2; got {pitch!r}"
            )
        self.dome = dome
        self.pitch = pitch

        by_level: dict[int, list[int]] = {}
        for cell in dome.cells:
            by_level.setdefault(cell.index.level, []).append(cell.id)
        # Deepest first: the apex band is drawn at the top of the panel.
        bands = [
            (level, *_place(dome, by_level[level])) for level in sorted(by_level, reverse=True)
        ]

        self.columns = max(width for _, _, _, width in bands)
        slots: list[Slot] = []
        placement: list[tuple[int, int, int]] = []
        top = 0
        for level, placed, rows, width in bands:
            # Bands are centred on each other, so the taper is the shape the
            # panel shows rather than an artifact of which edge they align to.
            left = (self.columns - width) // 2
            for cell_id, (row, column) in placed.items():
                slots.append(
                    Slot(
                        cell=cell_id,
                        level=level,
                        column_name=dome.cells[cell_id].index.column,
                        row=top + row,
                        column=left + column,
                    )
                )
            placement.append((level, top, rows))
            top += rows + _BAND_GAP

        self.rows = top - _BAND_GAP
        self.slots = tuple(sorted(slots, key=lambda slot: slot.cell))
        #: `(level, first slot row, rows)` per band, apex first.
        self.bands = tuple(placement)
        self._by_cell = {slot.cell: slot for slot in self.slots}

    def __repr__(self) -> str:
        return (
            f"BandLayout({len(self.bands)} bands, {self.rows}x{self.columns} slots, "
            f"pitch={self.pitch})"
        )

    @property
    def width(self) -> int:
        """The lattice's width in pixels."""
        return self.columns * self.pitch

    @property
    def height(self) -> int:
        """The lattice's height in pixels."""
        return self.rows * self.pitch

    @property
    def mark(self) -> int:
        """A mark's size in pixels: the slot, less the gap that separates it."""
        return max(1, self.pitch - 1)

    def slot(self, cell_id: int) -> Slot:
        """Where this cell's mark goes. Every cell of the dome has exactly one."""
        return self._by_cell[cell_id]

    def rect(self, cell_id: int) -> tuple[int, int, int]:
        """`(top, left, size)` of one cell's mark, in lattice pixels."""
        slot = self._by_cell[cell_id]
        return slot.row * self.pitch, slot.column * self.pitch, self.mark


def _place(dome: Dome, cell_ids: Sequence[int]) -> tuple[dict[int, tuple[int, int]], int, int]:
    """One band: `{cell id: (row, column)}` in band-local slots, its rows, its width."""
    columns: dict[str, list[int]] = {}
    for cell_id in cell_ids:
        columns.setdefault(dome.cells[cell_id].index.column, []).append(cell_id)

    shapes: dict[str, int] = {}
    rows = 1
    for name, ids in columns.items():
        dimensions = {len(dome.cells[cell_id].index.position) for cell_id in ids}
        if len(dimensions) != 1 or not dimensions <= {1, 2}:
            raise ValueError(
                f"the construction layout's {name!r} column mixes index shapes "
                f"{sorted(dimensions)}; a band draws a lattice as a grid and an "
                "ordinal along the band, and cannot draw both at once"
            )
        (shapes[name],) = dimensions
        if shapes[name] == 2:
            rows = max(rows, max(dome.cells[i].index.position[0] for i in ids) + 1)

    placed: dict[int, tuple[int, int]] = {}
    left = 0
    for name, ids in columns.items():
        ids = sorted(ids, key=lambda cell_id: dome.cells[cell_id].index.position)
        if shapes[name] == 2:
            positions = [dome.cells[cell_id].index.position for cell_id in ids]
            width = max(position[1] for position in positions) + 1
            for cell_id, position in zip(ids, positions):
                placed[cell_id] = (position[0], left + position[1])
        else:
            # Down the band and then across, so an ordinal column stands beside
            # a lattice and lies flat where there is no lattice to stand beside.
            width = -(-len(ids) // rows)
            for ordinal, cell_id in enumerate(ids):
                placed[cell_id] = (ordinal % rows, left + ordinal // rows)
        left += width + _COLUMN_GAP
    return placed, rows, left - _COLUMN_GAP


# -- the measured persistence the trail decays at ---------------------------


def measured_persistence(
    sheaf: Sheaf,
    *,
    ticks: int = DEFAULT_TICKS,
    burn_in: int = DEFAULT_BURN_IN,
    generator: torch.Generator | None = None,
) -> np.ndarray:
    """`[predicting cells]` ticks: each cell's own measured persistence.

    `docs/spec/10-the-demo-surface.md`, *The trail*: **the estimate
    `05-timescales.md` already defines, not a second definition of timescale.**
    So this is not a new measurement -- it is
    :func:`patchworks.bias_selection.measure`'s, run over the sheaf's own frozen
    body and its own biases rather than over drawn candidates, and read at
    :attr:`~patchworks.bias_selection.Measurement.effective_timescale`: the
    median of the per-tick regional `tau = -1/ln rho` a driven trajectory
    visits. The median because `tau` diverges as `rho -> 1` and a mean is
    dominated by whichever region came closest to the boundary.

    **Measured, and measured from outside.** Nothing in the architecture reads a
    cell's timescale (ADR-0005), and nothing here changes that: this is a
    display asking a question about the graph, the same act as a sweep, and what
    it returns is a plain array of numbers no cell can reach. It reads the body
    and the biases and writes neither -- `measure` runs the whole trajectory
    under `torch.no_grad`, on a chart of its own.

    **It is a number about the biases, so it is taken when the panel is built**
    rather than per tick. The biases adapt, so a cell's persistence drifts over
    a long run; re-measuring it every frame would cost a 64-tick trajectory per
    frame to chase a quantity that moves on the timescale of learning. A caller
    who wants the drift can build a second panel.

    The trajectory is the rig's plausible one rather than the run's own -- there
    is no message stream to drive the body with from out here -- which is what
    `05-timescales.md` specifies its estimator over, and `generator` is what it
    is drawn with.

    **It never draws from the global RNG**, which is #77's standing constraint
    rather than a nicety: *switching the surface off must change no trajectory*.
    `measure` draws `(burn_in + ticks + 1)` batches of normals, so taking the
    default generator would advance the process-wide stream and a later
    `Sheaf(dome)` or `Agent(env, dome=dome)` built without one of its own would
    get **different parameters** because a panel was opened. So `generator=None`
    means a private :class:`torch.Generator`, not the global one. A fresh
    generator's seed is fixed, so the default is repeatable as well as
    inert -- pass one only to draw a different trajectory.

    **A cell with no persistence at all is refused rather than clamped.** An
    expansive region has no `tau` -- `tau = -1/ln rho` does not exist at `rho >=
    1` -- so :func:`~patchworks.bias_selection.measure` clamps `rho` for the
    ordering's sake and reports the expansive fraction separately; the clamp is
    a reporting guard and *not a timescale*, which is why
    :meth:`~patchworks.bias_selection.Measurement.contained` reads `rho_median`
    rather than the clamped value, and this reads it the same way. Handed to a
    trail, the clamp would be a cell lit forever at a rate nothing measured, and
    a cell whose trajectory overflowed would come back the **fastest** in the
    dome, because a NaN pre-activation reads downstream as no unit active. A
    body with either in it has failed `05-timescales.md`'s go/no-go, and that is
    a thing to see rather than to draw around.

    A cell just *inside* the boundary is not refused, and its trail does not
    visibly decay: `rho_median` a hair under one is a genuine `tau` of order a
    million ticks, and a cell whose content does not decay over any run the demo
    lasts should be drawn as a cell whose glow does not fade. It will also sit
    in :attr:`DomePanel.warming_up` for the whole run, because statistics over a
    thousand ticks are not a baseline for a million-tick cell -- which is the
    notice doing its job rather than failing to clear.
    """
    measurement = measure(
        sheaf.body,
        sheaf.biases,
        ticks=ticks,
        burn_in=burn_in,
        generator=torch.Generator() if generator is None else generator,
    )
    usable = measurement.finite & (measurement.rho_median < 1.0)
    if not bool(usable.all()):
        diverged = int((~measurement.finite).sum())
        expansive = int((measurement.finite & ~usable).sum())
        raise ValueError(
            f"{diverged + expansive} of {len(usable)} cells have no measured "
            f"persistence for a trail to decay at: {expansive} sit in expansive "
            f"regions, where `tau = -1/ln rho` does not exist, and {diverged} "
            "overflowed the driven trajectory. `bias_selection` clamps both for "
            "the ordering's sake and reports them separately, and the clamp is "
            "not a timescale. This body has failed the go/no-go of "
            "docs/spec/05-timescales.md, whose selection caps the slow end by "
            "measured contraction; the panel is not the place to hide that."
        )
    return measurement.effective_timescale.detach().cpu().numpy().astype(np.float64)


# -- the panel --------------------------------------------------------------


class DomePanel:
    """The dome panel: bands, prediction error, and the trail.

    Built on the graph's shape and the run's measured persistences, and fed one
    tick record at a time::

        panel = DomePanel(agent.dome, measured_persistence(agent.sheaf))
        for _ in range(ticks):
            agent.tick()
            record = recorder.observe()
            if record is not None:
                show(panel.frame(record))

    or over a feed, live or off disk, exactly as
    :meth:`patchworks.surface.renderer.Renderer.frames` takes one::

        panel.frames(Trace.load(path))

    **A frame is a `[height, width, 3]` uint8 array.** What shows it is the
    caller's; see the module docstring.

    **Closable, and closing it changes nothing but the view.** The panel holds
    no agent, no sheaf and no world, so there is nothing for :meth:`close` to
    change -- it stops the drawing and nothing else. :meth:`frames` keeps
    draining a closed feed rather than stopping it, because a live feed is a
    run: a display that stopped consuming would stop the run it is watching,
    which is the one thing the surface must never do.

    **Fed in order, every capture.** The trail is a decay across the ticks
    between two records, so a record older than the last one it saw is refused
    rather than absorbed -- see :meth:`frame`.
    """

    def __init__(
        self,
        dome: Dome,
        persistence: Sequence[float] | np.ndarray,
        *,
        pitch: int = DEFAULT_PITCH,
        raw: bool = False,
    ) -> None:
        self.dome = dome
        self.layout = BandLayout(dome, pitch=pitch)
        cells = len(dome.predicting)
        values = np.array(persistence, dtype=np.float64).reshape(-1)
        if values.shape != (cells,):
            raise ValueError(
                f"the trail decays at one measured persistence per predicting cell, "
                f"{cells} of them in `dome.predicting` order; got "
                f"{tuple(np.shape(persistence))}. `measured_persistence(sheaf)` is "
                "where it comes from."
            )
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError(
                "a measured persistence is a positive, finite number of ticks; got "
                f"min {values.min()}, max {values.max()}"
            )
        #: `[predicting cells]` ticks, in `dome.predicting` order. A copy, and
        #: read-only: it is validated once here, and a caller who could still
        #: reach the array afterwards could put a negative number in it and turn
        #: the trail's `exp(-elapsed / tau)` into growth without decay.
        values.flags.writeable = False
        self.persistence = values
        self._raw = bool(raw)

        self._row = {cell_id: row for row, cell_id in enumerate(dome.predicting)}
        # The running statistics: Welford, so the baseline is the whole run so
        # far rather than a window whose length would be a hand-set constant.
        # Counted per cell rather than once, because a cell whose prediction error is
        # not a number this tick contributes nothing to its own statistics and the
        # rest of the dome carries on.
        self._seen = np.zeros(cells, dtype=np.int64)
        self._mean = np.zeros(cells)
        self._m2 = np.zeros(cells)
        self._glow = np.zeros(cells)
        # The ticks a cell's *own* readings span, so a cell that was not a
        # number for a long stretch does not inherit the run's span. -1 is "no
        # reading yet"; both move only on a tick this cell was readable.
        self._first_read = np.full(cells, -1, dtype=np.int64)
        self._last_read = np.full(cells, -1, dtype=np.int64)
        self._no_reading = np.zeros(cells, dtype=bool)
        self._raw_scale = 0.0
        self._last_tick: int | None = None
        self._closed = False

        self._notice_scale = max(1, pitch // DEFAULT_PITCH)
        self._notice_height = _FONT_HEIGHT * self._notice_scale + 2 * self._notice_scale
        # Fixed at construction from the longest notice this panel can ever
        # show, so the frame is one size for a whole run: a capture whose frames
        # change shape halfway through is not a capture.
        self._width = max(
            self.layout.width, _text_width(self._notice(cells), self._notice_scale)
        )

    def __repr__(self) -> str:
        state = "closed" if self._closed else f"{self.warming_up} warming up"
        return f"DomePanel({self.width}x{self.height}, raw={self.raw}, {state})"

    def __enter__(self) -> "DomePanel":
        return self

    def __exit__(self, *exception) -> None:
        self.close()

    # -- the window --------------------------------------------------------

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._notice_height + self.layout.height

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def raw(self) -> bool:
        """The debug flag `10-the-demo-surface.md` keeps the raw map behind.

        Switchable mid-run, and switching it **clears the trail**: a glow is
        what a value decayed from, and the two maps' values are not the same
        quantity -- one is a deviation from a cell's own baseline and the other
        is a raw norm against the dome's largest. Carrying one map's trail into
        the other would draw a decay of a number that map never showed. The
        running statistics are untouched, so the map that comes back is the one
        the whole run so far paid for.
        """
        return self._raw

    @raw.setter
    def raw(self, raw: bool) -> None:
        raw = bool(raw)
        if raw != self._raw:
            self._glow = np.zeros_like(self._glow)
        self._raw = raw

    def close(self) -> None:
        """Close the panel. Idempotent, and it changes nothing but the view."""
        self._closed = True

    # -- what it knows about the run ---------------------------------------

    @property
    def baseline(self) -> np.ndarray:
        """`[predicting cells]` bool: which cells' statistics are a baseline yet.

        A cell has one when its statistics hold a spread at all -- two
        observations -- **and** span at least its own measured persistence. The
        second half is why this is per cell rather than a count: a cell whose
        content turns over in a tick has a baseline almost at once, and a core
        cell whose content stands for hundreds of ticks does not have one until
        the statistics have watched it for that long. Anything shorter is a
        baseline over a stretch in which the cell barely moved, and calling that
        warmed up is the pretending `10-the-demo-surface.md` rules out.

        The condition is read off the record's own tick counter, so a decimated
        capture warms up in the same number of *ticks* as an undecimated one,
        with fewer samples in it.

        **The span is the cell's own readings, not the panel's ticks.** A cell
        whose prediction error was not a number for most of the run has not been
        watched for that stretch, and counting it would hand a slow cell a
        baseline built from the two readings either side of a long silence --
        the pretending this notice exists to prevent, in exactly the
        recovering-from-divergence case the panel is built to show.
        """
        spanned = np.where(
            self._first_read >= 0, self._last_read - self._first_read, -1
        ).astype(np.float64)
        return (self._seen >= 2) & (spanned >= self.persistence)

    @property
    def warming_up(self) -> int:
        """How many predicting cells have no baseline yet."""
        return int((~self.baseline).sum())

    def rect(self, cell_id: int) -> tuple[int, int, int]:
        """`(top, left, size)` of one cell's mark **in the frame**.

        :meth:`BandLayout.rect` is in the lattice's own pixels; a frame also
        carries the notice above the bands and centres the lattice in whatever
        width the notice needs. This is the one a caller indexes a frame with --
        to label a cell, to crop one band, or to find what was clicked.
        """
        top, left, size = self.layout.rect(cell_id)
        return top + self._notice_height, left + self._lattice_left, size

    @property
    def _lattice_left(self) -> int:
        """Where the lattice starts across the frame; it is centred in it."""
        return (self.width - self.layout.width) // 2

    @property
    def mean(self) -> np.ndarray:
        """`[predicting cells]`: each cell's own running mean, as a copy."""
        return self._mean.copy()

    @property
    def spread(self) -> np.ndarray:
        """`[predicting cells]`: each cell's own running standard deviation.

        Zero for a cell with fewer than two readings to take one over, which is
        the first half of :attr:`baseline`.
        """
        return np.where(
            self._seen >= 2,
            np.sqrt(self._m2 / np.maximum(self._seen - 1, 1)),
            0.0,
        )

    @property
    def no_reading(self) -> np.ndarray:
        """`[predicting cells]` bool: whose last prediction error was not a number.

        Drawn in their own colour rather than on the colormap; see
        :data:`_NO_READING`.
        """
        return self._no_reading.copy()

    @property
    def glow(self) -> np.ndarray:
        """`[predicting cells]` in `[0, 1]`: the trail, as a copy.

        What :meth:`frame` puts on the colormap. Handed out because it is the
        quantity the trail's claim is about -- one exponential per cell, at that
        cell's own persistence -- and reading it off pixels is reading it
        through a colormap.
        """
        return self._glow.copy()

    # -- drawing -----------------------------------------------------------

    def frame(self, record: TickRecord) -> np.ndarray:
        """`[height, width, 3]` uint8: this record, drawn. Advances the trail.

        **Colour is prediction error, normalised per cell.** Each cell's error
        is read against that cell's own running mean and spread, and what is
        drawn is how far above its own baseline it is -- one standard deviation
        above is a fixed, readable place on the ramp, and the scale comes from
        the cell's statistics rather than from a constant chosen here. Two
        consequences, both accepted rather than fixed: a **chronically wrong**
        cell renders calm, because a cell always this wrong is not surprised;
        and the statistics need a **warm-up**, which the panel says on screen
        (:attr:`baseline`) rather than pretending.

        **With :attr:`raw` set** the map is the un-normalised one that section
        keeps behind a debug flag: every cell against one shared scale, the
        largest prediction error the panel has seen anywhere in the dome. That map shows
        the taper's shape -- which is exactly why it is not the primary channel
        -- and it is the map a falsification sweep should read, because chronic
        failure is visible on it.

        **The trail.** Whatever a cell reaches, it decays from at that cell's
        own measured persistence: `glow <- max(value, glow * exp(-Δticks/tau))`,
        over the ticks between this record and the last one, so a decimated
        capture decays by the ticks that passed rather than by the frames that
        were kept. One exponential per cell, and it renders the multi-timescale
        claim for free -- rim cells go dark almost immediately, core cells stay
        lit for hundreds of ticks, and a wave of glow climbing the bands is
        message passing, watched.

        **A cell whose prediction error is not a number has no reading**, and is
        drawn in its own colour (:data:`_NO_READING`) rather than anywhere on
        the colormap. It is kept out of its own statistics, so one NaN does not
        leave a cell's baseline undefined for the rest of the run, and out of
        the raw map's scale, so one infinity does not black out the dome. A
        diverging body is the event this display exists to make visible, and
        both of those would have rendered it as the quietest cell on screen.

        Records must arrive in order and each at most once: the decay is taken
        over the gap between two of them, so a record at or behind the last tick
        seen has no gap to decay over and is refused rather than absorbed.
        """
        if self._closed:
            raise ValueError(
                "this panel is closed; a closed panel draws nothing. Closing "
                "changes nothing but the view, so the run it was watching is "
                "still going."
            )
        error = np.asarray(record.prediction_error, dtype=np.float64).reshape(-1)
        cells = len(self.persistence)
        if error.shape != (cells,):
            raise ValueError(
                f"this dome has {cells} predicting cells and the record carries "
                f"{error.shape[0]} prediction errors; the panel and the record are "
                "on different graphs"
            )
        if self._last_tick is None:
            elapsed = 0.0
        elif record.tick <= self._last_tick:
            raise ValueError(
                f"this panel last drew tick {self._last_tick} and was handed "
                f"{record.tick}. The trail decays over the ticks between two "
                "records, so they arrive in order and each one once. Replaying a "
                "stretch of a trace means a fresh panel."
            )
        else:
            elapsed = float(record.tick - self._last_tick)
        self._last_tick = record.tick

        # A cell whose prediction error is not a number has no reading this tick: it is
        # kept out of its own statistics, out of the maps' scales and out of the
        # glow, and it is drawn in its own colour instead of on the colormap.
        self._no_reading = ~np.isfinite(error)
        # The ticks this cell's own readings span, which is what its baseline is
        # measured against: a stretch it was not a number for is a stretch the
        # statistics did not watch it.
        read = ~self._no_reading
        self._first_read = np.where(
            read & (self._first_read < 0), record.tick, self._first_read
        )
        self._last_read = np.where(read, record.tick, self._last_read)
        readable = self._observe(error)
        value = self._raw_value(readable) if self.raw else self._normalised(readable)
        self._glow = np.maximum(self._glow * np.exp(-elapsed / self.persistence), value)
        return self._draw()

    def frames(self, feed: Iterable[TickRecord]) -> Iterator[np.ndarray]:
        """One frame per record of `feed`, in order, while the panel is open.

        `feed` is a live recorder's
        :meth:`~patchworks.surface.record.Recorder.watch` or a
        :class:`~patchworks.surface.record.Trace` off disk, and this method
        cannot tell which -- one renderer over a tick record, as the scene's
        renderer is.

        **A closed panel drains its feed and yields nothing.** A live feed is a
        run being driven, so a display that stopped consuming would stop the run
        rather than close a window.
        """
        for record in feed:
            if self._closed:
                continue
            yield self.frame(record)

    # -- the two maps ------------------------------------------------------

    def _observe(self, error: np.ndarray) -> np.ndarray:
        """This tick, into the statistics: Welford's, and the raw map's scale.

        Returns the readable prediction error -- the array with a cell that has
        no reading standing at its own mean, so that everything downstream treats
        it as *nothing happened here* rather than propagating a NaN into a colour.

        **A trail already drawn on the raw map is rescaled with it.** The glow
        holds values that were divided by the scale as it stood when they were
        drawn, so a new largest prediction error anywhere in the dome would
        otherwise leave a decaying cell brighter than a cell reaching the same
        raw norm now -- and comparing cells is the whole of what the raw map is for.
        """
        readable = np.where(self._no_reading, self._mean, error)
        self._seen += ~self._no_reading
        # Welford's, per cell: where there is no reading the delta is zero, so
        # the mean and the sum of squares stand and the count does not move.
        delta = readable - self._mean
        self._mean += delta / np.maximum(self._seen, 1)
        self._m2 += delta * (readable - self._mean)
        largest = float(readable.max(initial=0.0))
        if largest > self._raw_scale:
            if self._raw and self._raw_scale > 0.0:
                self._glow *= self._raw_scale / largest
            self._raw_scale = largest
        return readable

    def _normalised(self, error: np.ndarray) -> np.ndarray:
        """`[cells]` in `[0, 1]`: how far above its own baseline each cell is.

        `1 - exp(-z)` over the positive part of the cell's own z-score. Above
        the baseline only: prediction error is a magnitude, so a cell quieter
        than usual is calm rather than negatively surprised. Saturating, and its
        scale is the cell's own standard deviation -- there is no threshold
        here to hand-set, which is the objection `10-the-demo-surface.md` raises
        against a hand-set constant on the edge threshold and it applies to a
        colour scale for the same reason.
        """
        above = np.maximum(error - self._mean, 0.0)
        spread = self.spread
        lit = spread > 0.0
        # No spread yet is no baseline yet, and this is the noise a cold panel
        # shows: anything above a mean it has barely measured reads as
        # everything. The notice on screen is what keeps that honest.
        return np.where(
            lit,
            1.0 - np.exp(-above / np.where(lit, spread, 1.0)),
            np.where(above > 0.0, 1.0, 0.0),
        )

    def _raw_value(self, error: np.ndarray) -> np.ndarray:
        """`[cells]` in `[0, 1]`: the raw norms, on one scale shared by the dome.

        One scale, because comparing cells is the whole of what the raw map is
        for. It is the largest prediction error seen anywhere so far rather than this
        tick's largest, so a quiet tick does not rescale the dome under the
        viewer; :meth:`_observe` keeps it, and keeps the trail on it.

        **A cell with no reading reaches zero here**, so that it is kept out of
        the glow as the normalised map already keeps it. :meth:`_observe` hands
        back such a cell standing at its own mean, which is *nothing happened
        here* on a map read against that mean -- but on this one a mean is a
        positive raw norm, and feeding it to the trail would pin a diverged
        cell's glow at a constant forever instead of letting it decay. A
        permanently diverged cell would then be drawn, on recovery, from a
        brightness it never produced.
        """
        if self._raw_scale <= 0.0:
            return np.zeros_like(error)
        return np.where(self._no_reading, 0.0, error / self._raw_scale)

    # -- pixels ------------------------------------------------------------

    def _draw(self) -> np.ndarray:
        canvas = np.empty((self.height, self.width, 3), dtype=np.uint8)
        canvas[:, :] = _BACKGROUND
        colours = colormap(self._glow)
        for slot in self.layout.slots:
            y, x, size = self.rect(slot.cell)
            row = self._row.get(slot.cell)
            if row is None:
                colour = _EMPTY
            elif self._no_reading[row]:
                colour = _NO_READING
            else:
                colour = colours[row]
            canvas[y : y + size, x : x + size] = colour
        if not self.raw:
            warming = self.warming_up
            if warming:
                _draw_text(
                    canvas,
                    self._notice(warming),
                    top=self._notice_scale,
                    left=self._notice_scale,
                    scale=self._notice_scale,
                    ink=_NOTICE_INK,
                )
        return canvas

    def _notice(self, warming: int) -> str:
        """What the panel says while the statistics are warming up.

        On screen, in the frame, rather than in a log the capture does not
        carry: a viewer who cannot see that the map is still noise has been
        misled by the picture.
        """
        return f"WARMING UP {warming}/{len(self.persistence)} CELLS"


# -- text -------------------------------------------------------------------

_FONT_WIDTH = 4
_FONT_HEIGHT = 5
_FONT_GAP = 1

#: A 4x5 bitmap font, upper case and digits, because the notice has to be *on
#: screen* and nothing in the dependency set draws text. Rows top to bottom,
#: `#` set. An unknown character draws as a filled box, which is visible rather
#: than silent.
_GLYPHS = {
    " ": "..../..../..../..../....",
    "A": ".##./#..#/####/#..#/#..#",
    "B": "###./#..#/###./#..#/###.",
    "C": ".###/#.../#.../#.../.###",
    "D": "###./#..#/#..#/#..#/###.",
    "E": "####/#.../###./#.../####",
    "F": "####/#.../###./#.../#...",
    "G": ".###/#.../#.##/#..#/.###",
    "H": "#..#/#..#/####/#..#/#..#",
    "I": "###./.#../.#../.#../###.",
    "J": "..##/...#/...#/#..#/.##.",
    "K": "#..#/#.#./##../#.#./#..#",
    "L": "#.../#.../#.../#.../####",
    "M": "#..#/####/####/#..#/#..#",
    "N": "#..#/##.#/####/#.##/#..#",
    "O": ".##./#..#/#..#/#..#/.##.",
    "P": "###./#..#/###./#.../#...",
    "Q": ".##./#..#/#..#/#.##/.###",
    "R": "###./#..#/###./#.#./#..#",
    "S": ".###/#.../.##./...#/###.",
    "T": "####/.#../.#../.#../.#..",
    "U": "#..#/#..#/#..#/#..#/.##.",
    "V": "#..#/#..#/#..#/#.#./.#..",
    "W": "#..#/#..#/####/####/#..#",
    "X": "#..#/#..#/.##./#..#/#..#",
    "Y": "#..#/#..#/.##./.#../.#..",
    "Z": "####/...#/.##./#.../####",
    "0": ".##./#..#/#..#/#..#/.##.",
    "1": "..#./.##./..#./..#./.###",
    "2": "###./...#/.##./#.../####",
    "3": "###./...#/.##./...#/###.",
    "4": "#..#/#..#/####/...#/...#",
    "5": "####/#.../###./...#/###.",
    "6": ".##./#.../###./#..#/.##.",
    "7": "####/...#/..#./.#../.#..",
    "8": ".##./#..#/.##./#..#/.##.",
    "9": ".##./#..#/.###/...#/.##.",
    "/": "...#/..#./.#../#.../#...",
    ":": "..../.#../..../.#../....",
    "-": "..../..../.##./..../....",
    ".": "..../..../..../..../.#..",
}
_UNKNOWN = "####/####/####/####/####"


def _text_width(text: str, scale: int) -> int:
    """The pixels `text` takes at `scale`, including a margin either side."""
    if not text:
        return 0
    glyphs = len(text) * (_FONT_WIDTH + _FONT_GAP) - _FONT_GAP
    return (glyphs + 2) * scale


def _draw_text(
    canvas: np.ndarray, text: str, *, top: int, left: int, scale: int, ink: tuple[int, int, int]
) -> None:
    """Draw `text` into `canvas`, clipped at its edges."""
    height, width = canvas.shape[:2]
    x = left
    for character in text:
        rows = _GLYPHS.get(character.upper(), _UNKNOWN).split("/")
        for r, row in enumerate(rows):
            for c, pixel in enumerate(row):
                if pixel != "#":
                    continue
                y0, x0 = top + r * scale, x + c * scale
                if y0 >= height or x0 >= width:
                    continue
                canvas[y0 : y0 + scale, x0 : x0 + scale] = ink
        x += (_FONT_WIDTH + _FONT_GAP) * scale
