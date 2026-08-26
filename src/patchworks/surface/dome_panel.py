"""The dome panel: the bands, and every mark drawn in them.

`docs/spec/10-the-demo-surface.md`, *The dome panel*, is the whole of what this
module implements, over the record :mod:`patchworks.surface.record` already
defines. Three things make the **predicting** cells' picture, and the first is
the frame every other mark is drawn in too:

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

**Every mark draws a quantity that is honestly earned.** A boundary cell runs
no body and makes no prediction (ADR-0006), so it has **no prediction error**,
and colouring one on that map would be a fabrication in the largest, most
eye-catching band on screen. What a boundary cell has instead is an **edge**,
and edge disagreement is drawn on the same colormap, honestly earned. That
single distinction is what the rest of this module is: #93 drew the bands and
the predicting cells' channel, and #94 draws the marks the boundary cells get.

* **The boundary band.** L0 draws the agent's own render, tiled into the patch
  lattice, and carries no prediction-error colour. The render costs nothing --
  it already exists every tick -- and it ties the abstract stack to the world.
  It is handed in rather than taken: this module owns no world (see below), and
  a record holds state rather than frames, so a caller re-renders one from a
  record's snapshot (:class:`~patchworks.surface.renderer.Renderer`) or hands
  over the observation the agent was given.
* **The somatomotor strip**, beside the tiled render, because that is where the
  cluster attaches -- 3 proprioceptive, 3 touch, 1 actuator -- coloured by edge
  disagreement. The actuator additionally draws **decomposed**: three paired
  bars, commanded as an outline and applied as a fill, which is
  `04-action-and-the-boundary.md`'s efference copy made visible, beside a
  standing bar for its own motor-side disagreement.
* **The drive mark** on the apex band, drawn like the strip. It is what makes
  `08`'s **task-invariant behaviour** near-miss distinguishable on screen from
  the demo working, and it falsifies the drive rather than the demo (ADR-0009).
* **The edge overlay**: only the edges carrying the most disagreement this
  tick, thresholded from **the tick's own scale**, and off by default.

None of those is on the prediction-error map, and none of them reaches for it:
`DomePanel` keeps the two populations in two arrays with no index in common,
and `tests/test_dome_panel.py` asserts that no boundary cell is ever assigned a
prediction-error value.

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
from typing import Callable, Iterable, Iterator, Sequence

import numpy as np
import torch

from patchworks.bias_selection import DEFAULT_BURN_IN, DEFAULT_TICKS, measure
from patchworks.graph import CellKind, Dome
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

#: Slots between the lattice and the motor strip, which stands beside the
#: boundary band rather than in it: the actuator's decomposition is three
#: paired bars and a slot is one square.
_GUTTER_GAP = 1

#: The panel's ground, and the slot of a cell with no mark to draw. Distinct
#: from the ground so that the frame is visible -- an empty slot is a cell
#: whose quantity this record did not carry, not an absent cell. A patch cell
#: sits at this colour until a render is handed in: it has no prediction error
#: to fall back on, and drawing one would be the fabrication this panel is
#: built to refuse.
_BACKGROUND = (8, 8, 10)
_EMPTY = (26, 26, 30)
_NOTICE_INK = (216, 216, 220)

#: The motor strip's inks. Deliberately off the colormap: a torque is not a
#: disagreement, and putting the two on one ramp would invite the bars to be
#: read against the marks around them. The outline is what was **commanded**
#: and the fill what was **applied**, so saturation reads as the fill falling
#: short of its own outline.
_ZERO_LINE = (72, 74, 84)
_COMMANDED = (150, 200, 236)
_APPLIED = (64, 132, 190)

#: What a full-height torque bar means, and it is **the environment's own
#: contract rather than anything chosen here**: the sandbox declares
#: `spaces.Box(-1.0, 1.0, (nu,))` (`patchworks.sandbox.env`) and
#: :meth:`patchworks.agent.Agent.act` clips the command to it before the arm
#: reads it, so the *applied* row is bounded by this on every tick of every run.
#:
#: Drawing both rows against it is what makes `04-action-and-the-boundary.md`'s
#: *near-zero commanded torque* an absolute claim: near zero means near zero,
#: on every tick, with no dependence on what the run has happened to see. A
#: running maximum here -- what #94 first shipped -- is a ratchet that one
#: unclipped command raises for the rest of the run, after which a command at
#: the limit draws at the zero line and the stall signature is fabricated out
#: of a full-strength swing. Ruled on in #94.
_ACTION_BOUND = 1.0

#: The onset counter the motor strip is sized to fit. Six digits is a run of
#: hours at the sandbox's 50 Hz control rate. The count is always drawn in
#: full: one that outgrew this runs off the strip and is clipped at the frame,
#: which is visibly wrong, where a saturated count would be a plausible number
#: that is not the reading.
_ONSET_WIDEST = "T+999999"

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


# -- the statistics a mark is normalised against ----------------------------


class _RunningScale:
    """One running mean and spread per mark, and the raw map's shared scale.

    **Normalisation is per mark, against that mark's own running statistics**
    (`docs/spec/10-the-demo-surface.md`, *Colour is prediction error*). The
    argument is made there for prediction error across the taper -- 12,288
    numbers at the base against 32 at the core -- and it holds just as hard for
    the marks drawn from edge disagreement, which are read off stalks the world
    writes at whatever width it has and off a **scalar** drive edge. A raw map
    over those would show the boundary's shape and nothing else.

    One class for both populations, because they are normalised the same way
    and a second copy of Welford's would be a second place for the arithmetic
    to drift. What differs between them is what happens to the value
    afterwards -- the predicting cells' feeds a trail, the boundary marks' does
    not -- and that stays in :class:`DomePanel`.

    **A mark whose value is not a number has no reading.** It is kept out of
    its own statistics, out of the raw map's scale, and drawn in its own
    colour; :meth:`observe` hands it back standing at its own mean, which is
    *nothing happened here* on a map read against that mean.
    """

    def __init__(self, marks: int) -> None:
        self.seen = np.zeros(marks, dtype=np.int64)
        self.mean = np.zeros(marks)
        self.m2 = np.zeros(marks)
        self.no_reading = np.zeros(marks, dtype=bool)
        # The ticks a mark's *own* readings span, so a mark that was not a
        # number for a long stretch does not inherit the run's span. -1 is "no
        # reading yet"; both move only on a tick this mark was readable.
        self.first_read = np.full(marks, -1, dtype=np.int64)
        self.last_read = np.full(marks, -1, dtype=np.int64)
        self.scale = 0.0

    def observe(self, values: np.ndarray, tick: int) -> tuple[np.ndarray, float]:
        """This tick into the statistics. Returns the readable values, and a factor.

        The factor is what a trail already drawn on the raw map must be
        multiplied by, and it is 1.0 unless this tick raised the shared scale:
        a glow holds values that were divided by the scale as it stood when
        they were drawn, so a new largest reading anywhere would otherwise
        leave a decaying mark brighter than one reaching the same raw value
        now -- and comparing marks is the whole of what the raw map is for.
        """
        self.no_reading = ~np.isfinite(values)
        read = ~self.no_reading
        self.first_read = np.where(read & (self.first_read < 0), tick, self.first_read)
        self.last_read = np.where(read, tick, self.last_read)
        readable = np.where(self.no_reading, self.mean, values)
        self.seen += read
        # Welford's, per mark: where there is no reading the delta is zero, so
        # the mean and the sum of squares stand and the count does not move.
        delta = readable - self.mean
        self.mean += delta / np.maximum(self.seen, 1)
        self.m2 += delta * (readable - self.mean)
        largest = float(readable.max(initial=0.0))
        rescale = 1.0
        if largest > self.scale:
            if self.scale > 0.0:
                rescale = self.scale / largest
            self.scale = largest
        return readable, rescale

    @property
    def spread(self) -> np.ndarray:
        """`[marks]`: each mark's own running standard deviation.

        Zero for a mark with fewer than two readings to take one over.
        """
        return np.where(
            self.seen >= 2, np.sqrt(self.m2 / np.maximum(self.seen - 1, 1)), 0.0
        )

    @property
    def spanned(self) -> np.ndarray:
        """`[marks]`: the ticks each mark's own readings span, or -1 for none."""
        return np.where(
            self.first_read >= 0, self.last_read - self.first_read, -1
        ).astype(np.float64)

    def normalised(self, readable: np.ndarray) -> np.ndarray:
        """`[marks]` in `[0, 1]`: how far above its own baseline each mark is.

        `1 - exp(-z)` over the positive part of the mark's own z-score. Above
        the baseline only: both quantities drawn through this are magnitudes,
        so a mark quieter than usual is calm rather than negatively surprised.
        Saturating, and its scale is the mark's own standard deviation -- there
        is no threshold here to hand-set, which is the objection
        `10-the-demo-surface.md` raises against a hand-set constant on the edge
        threshold and it applies to a colour scale for the same reason.
        """
        above = np.maximum(readable - self.mean, 0.0)
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

    def raw(self, readable: np.ndarray) -> np.ndarray:
        """`[marks]` in `[0, 1]`: the raw values, on one scale shared by all of them.

        One scale, because comparing marks is the whole of what the raw map is
        for. It is the largest reading seen anywhere so far rather than this
        tick's largest, so a quiet tick does not rescale the panel under the
        viewer.

        **A mark with no reading reaches zero here**, so that it is kept out of
        a trail as the normalised map already keeps it. :meth:`observe` hands
        back such a mark standing at its own mean, which is *nothing happened
        here* on a map read against that mean -- but on this one a mean is a
        positive raw value, and feeding it to a trail would pin a diverged
        mark's glow at a constant forever.
        """
        if self.scale <= 0.0:
            return np.zeros_like(readable)
        return np.where(self.no_reading, 0.0, readable / self.scale)


# -- the panel --------------------------------------------------------------


class DomePanel:
    """The dome panel: the bands, and every mark drawn in them.

    Built on the graph's shape and the run's measured persistences, and fed one
    tick record at a time::

        panel = DomePanel(agent.dome, measured_persistence(agent.sheaf))
        for _ in range(ticks):
            agent.tick()
            record = recorder.observe()
            if record is not None:
                show(panel.frame(record, render=observation["image"]))

    or over a feed, live or off disk, exactly as
    :meth:`patchworks.surface.renderer.Renderer.frames` takes one -- with the
    scene's own renderer supplying the boundary band's picture from each
    record's state::

        with Renderer(size=64) as scene:
            panel.frames(Trace.load(path), renderer=scene.frame)

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

    **Two populations, two arrays, no index in common.** The predicting cells'
    marks are drawn from :attr:`~patchworks.surface.record.TickRecord.prediction_error`
    and the boundary cells' from
    :attr:`~patchworks.surface.record.TickRecord.disagreement`; a cell is in
    exactly one of :attr:`~patchworks.graph.Dome.predicting` and
    :attr:`~patchworks.graph.Dome.boundary`, and the rows of the first array are
    unreachable from any boundary cell's slot. That is what makes *a boundary
    cell is never assigned a prediction error* structural here rather than
    careful.
    """

    def __init__(
        self,
        dome: Dome,
        persistence: Sequence[float] | np.ndarray,
        *,
        pitch: int = DEFAULT_PITCH,
        raw: bool = False,
        edges: bool = False,
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
        self._edges = bool(edges)

        self._row = {cell_id: row for row, cell_id in enumerate(dome.predicting)}
        # The running statistics: Welford, so the baseline is the whole run so
        # far rather than a window whose length would be a hand-set constant.
        # Counted per cell rather than once, because a cell whose prediction error is
        # not a number this tick contributes nothing to its own statistics and the
        # rest of the dome carries on.
        self._cell_scale = _RunningScale(cells)
        self._glow = np.zeros(cells)
        self._last_tick: int | None = None
        self._closed = False

        # -- the boundary cells' half -------------------------------------
        # The marks drawn from edge disagreement: every boundary cell that is
        # not a patch cell. A patch cell's mark is its piece of the render, so
        # it is not on this map at all -- and leaving it out is also what keeps
        # the warm-up notice about marks a viewer can see, rather than about
        # 256 statistics nothing draws.
        self._marks = tuple(
            cell_id
            for cell_id in dome.boundary
            if dome.cells[cell_id].kind is not CellKind.PATCH
        )
        self._mark_row = {cell_id: row for row, cell_id in enumerate(self._marks)}
        self._mark_scale = _RunningScale(len(self._marks))
        self._mark_value = np.zeros(len(self._marks))
        self._mark_raw = np.zeros(len(self._marks))
        #: `[marks]` in `[0, 1]`: each mark against **this tick's** largest
        #: boundary mark, which is what the strip's standing bar is drawn from.
        #: See :meth:`_observe_disagreement`.
        self._mark_standing = np.zeros(len(self._marks))
        self._last_disagreement = np.zeros(len(dome.edges))
        #: Whether the **last** record carried disagreement at all. What the
        #: marks are drawn from is a fact about that record, not about the run:
        #: a capture that left the array out is nothing to draw, not agreement.
        self._carried = False
        # `[marks, widest degree]`: the incident edges of each mark, padded with
        # the one slot past the end of a record's array -- a zero, which
        # contributes nothing to the sum of squares below. The whole gather is
        # then one indexing operation rather than a loop over cells and edges.
        self._mark_edges = np.full(
            (len(self._marks), max((dome.degrees[c] for c in self._marks), default=1)),
            len(dome.edges),
            dtype=np.int64,
        )
        for row, cell_id in enumerate(self._marks):
            incident = dome.incident[cell_id]
            self._mark_edges[row, : len(incident)] = incident
        self._patches = tuple(
            cell for cell in dome.cells if cell.kind is CellKind.PATCH
        )
        actuator = [cell for cell in dome.cells if cell.kind is CellKind.ACTUATOR]
        self._actuator = actuator[0].id if actuator else None
        # One commanded and one efference component per joint (ADR-0006), so
        # the number of paired bars is a construction fact and the strip is
        # sized for it before any record arrives.
        self._joints = actuator[0].stalk // 2 if actuator else 0
        self._edge_ends = np.array(
            [(edge.u, edge.v) for edge in dome.edges], dtype=np.int64
        ).reshape(len(dome.edges), 2)
        self._drawn_edges: tuple[int, ...] = ()
        self._unread_edges: tuple[int, ...] = ()
        self._threshold = 0.0
        self._torque = np.zeros((2, 0))

        self._notice_scale = max(1, pitch // DEFAULT_PITCH)
        self._notice_height = _FONT_HEIGHT * self._notice_scale + 2 * self._notice_scale
        # The motor strip stands beside the boundary band rather than in it,
        # and its width is what four bars and the onset counter need. Fixed
        # here, for the reason the notice's width is fixed here.
        # At least three pixels across, whatever the lattice's own mark is: the
        # applied fill is drawn one pixel inside the commanded outline, so a
        # narrower bar has no interior for it to fall short *in* -- at two it
        # overwrites the outline's far column and at one it lands wholly
        # outside, and "the fill falls short of its outline" is then not what is
        # drawn. The strip stands beside the band rather than in it
        # (:data:`_GUTTER_GAP`), so its bars owe the lattice no width.
        self._bar = max(3, self.layout.mark)
        self._bar_gap = max(1, pitch // 3)
        self._bar_margin = max(2, pitch // 2)
        bars = self._joints + 1  # one per joint, and the disagreement bar
        self._gutter = max(
            bars * self._bar + (bars - 1) * self._bar_gap + 2 * self._bar_margin,
            _text_width(_ONSET_WIDEST, self._notice_scale),
        )
        self._content = self.layout.width + _GUTTER_GAP * pitch + self._gutter
        # Fixed at construction from the longest notice this panel can ever
        # show, so the frame is one size for a whole run: a capture whose frames
        # change shape halfway through is not a capture.
        self._width = max(
            self._content,
            _text_width(self._notice(cells, len(self._marks)), self._notice_scale),
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

    @property
    def edges(self) -> bool:
        """The edge overlay: **thresholded, and off by default**.

        `10-the-demo-surface.md`, *Edges: thresholded, and off by default*.
        Drawing all of them is a hairball over a sparse mask that would grey out
        the bands underneath; drawing none gives up the live **route** through
        the graph, which is the thing the trail cannot show -- it says influence
        propagated, not which cells were carrying it during a reconciliation
        round.

        Default off, so a README capture stays clean and the live demo can turn
        it on. A toggle, and switchable mid-run: what it draws is this tick's
        disagreement and nothing accumulated, so there is no trail here to clear
        and no state to carry across the switch.
        """
        return self._edges

    @edges.setter
    def edges(self, edges: bool) -> None:
        self._edges = bool(edges)

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
        return (self._cell_scale.seen >= 2) & (
            self._cell_scale.spanned >= self.persistence
        )

    @property
    def warming_up(self) -> int:
        """How many predicting cells have no baseline yet."""
        return int((~self.baseline).sum())

    @property
    def boundary_marks(self) -> tuple[int, ...]:
        """The cell ids drawn from edge disagreement, in row order.

        Every boundary cell except a patch cell, whose mark is its piece of the
        render instead: the somatomotor strip's seven, and the drive.
        """
        return self._marks

    @property
    def boundary_baseline(self) -> np.ndarray:
        """`[boundary marks]` bool: which marks' statistics are a baseline yet.

        **Two readings, and no span condition** -- which is not a weaker test
        than :attr:`baseline`'s but the same one applied to a different object.
        A cell's span has to reach its own measured persistence because a
        baseline over a stretch the cell barely moved in is not one; a boundary
        cell runs no body, holds no chart and has the world write its whole node
        stalk every tick (ADR-0006), so its content turns over in a tick and
        there is no slow content for a short span to have missed. There is also
        no measured persistence for one -- `05-timescales.md`'s estimate is
        taken over the body a boundary cell does not run -- so a span condition
        here would have to invent the number it compared against.

        A mark that has never been read at all is **not warming up**: it is not
        being drawn. That is a record carrying no disagreement, not a statistic
        that has yet to settle.
        """
        return self._mark_scale.seen >= 2

    @property
    def boundary_warming_up(self) -> int:
        """How many boundary marks are being drawn without a baseline yet.

        Only the marks actually on the colormap: a record that carried no
        disagreement draws none of them, and a mark with no reading this tick
        is drawn in its own colour rather than against its statistics. Neither
        can mislead a viewer, so neither is what the notice is about.
        """
        if not self._carried:
            return 0
        drawn = ~self._mark_scale.no_reading
        return int((drawn & ~self.boundary_baseline).sum())

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
    def motor_strip(self) -> tuple[int, int, int, int]:
        """`(top, left, height, width)` of the actuator's decomposition, in the frame.

        The strip stands beside the boundary band, on the same footing
        :meth:`rect` puts a cell's mark on: the one a caller indexes a frame
        with to crop the bars out of it or to find what was clicked.
        """
        band = min(self.layout.bands, key=lambda entry: entry[0])
        _level, top, rows = band
        height = min(self.layout.height, max(rows * self.layout.pitch, 6 * self.layout.pitch))
        bottom = (top + rows) * self.layout.pitch
        return (
            bottom - height + self._notice_height,
            self._lattice_left + self.layout.width + _GUTTER_GAP * self.layout.pitch,
            height,
            self._gutter,
        )

    @property
    def _lattice_left(self) -> int:
        """Where the lattice starts across the frame.

        The lattice and the motor strip beside it are centred together, so the
        taper stays the shape the panel shows rather than shifting by whatever
        the strip needs.
        """
        return (self.width - self._content) // 2

    @property
    def mean(self) -> np.ndarray:
        """`[predicting cells]`: each cell's own running mean, as a copy."""
        return self._cell_scale.mean.copy()

    @property
    def spread(self) -> np.ndarray:
        """`[predicting cells]`: each cell's own running standard deviation.

        Zero for a cell with fewer than two readings to take one over, which is
        the first half of :attr:`baseline`.
        """
        return self._cell_scale.spread

    @property
    def no_reading(self) -> np.ndarray:
        """`[predicting cells]` bool: whose last prediction error was not a number.

        Drawn in their own colour rather than on the colormap; see
        :data:`_NO_READING`.
        """
        return self._cell_scale.no_reading.copy()

    @property
    def glow(self) -> np.ndarray:
        """`[predicting cells]` in `[0, 1]`: the trail, as a copy.

        What :meth:`frame` puts on the colormap. Handed out because it is the
        quantity the trail's claim is about -- one exponential per cell, at that
        cell's own persistence -- and reading it off pixels is reading it
        through a colormap.
        """
        return self._glow.copy()

    @property
    def boundary_lit(self) -> np.ndarray:
        """`[boundary marks]` in `[0, 1]`: what the last record lit each mark to.

        The counterpart of :attr:`glow` for the marks drawn from edge
        disagreement, and handed out for the same reason -- reading it off
        pixels is reading it through a colormap.

        **There is no trail on it.** A trail decays at the cell's own measured
        persistence, and a boundary cell has none to decay at: it runs no body,
        so `05-timescales.md`'s estimate is not defined for it, and a glow
        fading at a rate nothing measured would be exactly the fabrication these
        marks exist to avoid. What a strip mark shows is this tick.

        **Not a number where there is nothing to draw** -- a record that carried
        no disagreement, or a mark whose own reading was not one. Zero is a
        reading here, and a sweep that read one off a replay of a trace saved
        before this array existed would read the graph as agreeing on every
        edge. The pixels say the same thing in their own way: an empty slot and
        :data:`_NO_READING`.
        """
        if not self._carried:
            return np.full(len(self._marks), np.nan)
        return np.where(self._mark_scale.no_reading, np.nan, self._mark_value)

    @property
    def torque(self) -> np.ndarray:
        """`[2, joints]`: the commanded and applied rows the strip last drew.

        A copy of what the record carried, held so that the falsification test
        the bars render can be read as numbers as well as watched:
        `04-action-and-the-boundary.md`'s route-selection signature is an
        outline near zero beside a standing disagreement bar, and a sweep should
        not have to measure pixels to see it.
        """
        return self._torque.copy()

    @property
    def edge_threshold(self) -> float:
        """The disagreement an edge had to carry to be drawn on the last record.

        **Derived from that tick's own scale**, never hand-set: this tick's mean
        plus one standard deviation over the edges that had a reading. Multiply
        every edge's disagreement by any positive constant and exactly the same
        edges are drawn, which is what makes the picture of the route a property
        of the graph rather than of a number chosen here
        (`10-the-demo-surface.md`, *Edges: thresholded, and off by default*).
        """
        return self._threshold

    @property
    def drawn_edges(self) -> tuple[int, ...]:
        """The edge ids that cleared :attr:`edge_threshold` on the last record.

        The route. :attr:`unread_edges` is drawn beside it and is not part of
        it: an edge with no reading cleared no threshold.
        """
        return self._drawn_edges

    @property
    def unread_edges(self) -> tuple[int, ...]:
        """The edge ids whose disagreement was not a number on the last record.

        Drawn in :data:`_NO_READING` whatever the threshold says, and kept out
        of the statistics the threshold is taken over. An edge that has left the
        numbers is the loudest thing on the graph, and an overlay that dropped
        it would draw the route *around* a divergence.
        """
        return self._unread_edges

    # -- drawing -----------------------------------------------------------

    def frame(
        self,
        record: TickRecord,
        *,
        render: np.ndarray | None = None,
        since: int | None = None,
    ) -> np.ndarray:
        """`[height, width, 3]` uint8: this record, drawn. Advances the trail.

        `render` is the agent's own render for this tick, and it is what the
        **boundary band** draws: the picture is cut along the patch lattice and
        each cell's block goes in that cell's slot, so the bottom of the panel
        is the thing the arm is doing in the other window. It is handed in
        because a record holds state rather than frames -- re-render one from
        the record's snapshot with
        :meth:`patchworks.surface.renderer.Renderer.frame`, or hand over the
        observation the agent was given, which is the same camera. Without one
        the band is drawn empty: a patch cell has no prediction error to fall
        back on, and inventing a colour for the largest band on screen is the
        fabrication this panel is built to refuse.

        `since` is the onset counter's ticks-since-the-last-marker
        (:class:`~patchworks.surface.onset.OnsetCounter`), drawn on the motor
        strip beside the bars so that **onset is read off the strip rather than
        reconstructed afterward** (`10-the-demo-surface.md`, *Onset, and the
        near-misses*): the bars carry the moment of the first corrective torque
        and this carries the count. `None` before any hand has fired, and
        nothing is drawn for it -- a zero there would read as an event that just
        happened.

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

        **A boundary cell is never on any of that.** It runs no body and makes
        no prediction, so the record carries none for it and this method has
        none to draw: what its mark shows is the disagreement on its own edges,
        normalised the same way against its own statistics
        (:attr:`boundary_lit`), and the actuator's is drawn decomposed on the
        motor strip beside the band.

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
        # Everything this record carries is checked before anything is
        # observed. A record the panel refuses part-way through would already
        # have folded this tick into every cell's statistics and advanced the
        # trail, and the corrected record is then refused as out of order --
        # so the panel would be permanently one tick of statistics off, with
        # no recovery short of a fresh one.
        disagreement = self._checked(
            record.disagreement,
            (len(self.dome.edges),),
            f"this dome has {len(self.dome.edges)} edges and the record carries "
            "{shape} disagreements; the panel and the record are on different graphs",
        )
        actuator = self._checked(
            record.actuator,
            (2, self._joints),
            f"this dome's actuator cell holds {self._joints} commanded and "
            f"{self._joints} efference components and the record carries "
            "{shape}; the rows are `[2, joints]`, commanded then applied, as "
            "the boundary cell holds them",
        )
        tiles = None if render is None else self._tiles(render)
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

        # A cell whose prediction error is not a number has no reading this
        # tick: `_RunningScale` keeps it out of its own statistics and out of
        # the maps' scales, and it is drawn in its own colour instead of on the
        # colormap. The ticks a cell's own readings span are counted there too,
        # which is what its baseline is measured against: a stretch it was not a
        # number for is a stretch the statistics did not watch it.
        readable, rescale = self._cell_scale.observe(error, record.tick)
        if self.raw and rescale != 1.0:
            # A trail already drawn on the raw map is rescaled with it, or a new
            # largest prediction error anywhere in the dome would leave a
            # decaying cell brighter than a cell reaching the same raw norm now.
            self._glow *= rescale
        value = (
            self._cell_scale.raw(readable)
            if self.raw
            else self._cell_scale.normalised(readable)
        )
        self._glow = np.maximum(self._glow * np.exp(-elapsed / self.persistence), value)
        self._observe_disagreement(disagreement, record.tick)
        self._observe_torque(actuator)
        return self._draw(tiles=tiles, since=since)

    def frames(
        self,
        feed: Iterable[TickRecord],
        *,
        renderer: Callable[[TickRecord], np.ndarray] | None = None,
        since: Callable[[TickRecord], int | None] | None = None,
    ) -> Iterator[np.ndarray]:
        """One frame per record of `feed`, in order, while the panel is open.

        `feed` is a live recorder's
        :meth:`~patchworks.surface.record.Recorder.watch` or a
        :class:`~patchworks.surface.record.Trace` off disk, and this method
        cannot tell which -- one renderer over a tick record, as the scene's
        renderer is.

        `renderer` is what the boundary band's picture comes from: any callable
        taking a record and returning that tick's render, which is exactly
        :meth:`patchworks.surface.renderer.Renderer.frame`'s shape. Passing one
        is what fills L0 in; without one the band stays empty, because a record
        holds state rather than frames and this panel owns no world to re-render
        one from.

        `since` is the onset counter, on the same footing:
        :meth:`patchworks.surface.onset.OnsetCounter.count` is exactly its
        shape. It is here so that **a replay draws the counter the live panel
        drew** -- onset is read off the strip rather than reconstructed
        afterward, and a README capture composed from a trace would otherwise be
        the one picture without it::

            panel.frames(Trace.load(path), renderer=scene.frame, since=counter.count)

        **A closed panel drains its feed and yields nothing.** A live feed is a
        run being driven, so a display that stopped consuming would stop the run
        rather than close a window.
        """
        for record in feed:
            if self._closed:
                continue
            yield self.frame(
                record,
                render=None if renderer is None else renderer(record),
                since=None if since is None else since(record),
            )

    # -- what a boundary cell has instead ----------------------------------

    @staticmethod
    def _checked(
        values: np.ndarray, shape: tuple[int, ...], complaint: str
    ) -> np.ndarray | None:
        """One of a record's optional arrays, as floats. `None` if it carries none.

        *Not captured* and *the wrong graph* are different things: the empty
        one-dimensional array :func:`~patchworks.surface.record._nothing`
        defaults to is a record built without this quantity, which the marks
        drawn from it answer by drawing nothing, and anything else has to be the
        shape this dome gives it. The test is that exact shape rather than
        `size == 0`, so a record carrying `[2, 0]` for a dome whose actuator has
        joints is refused as the wrong graph instead of passing for a record
        that never captured the rows.

        **A copy, never the caller's array.** What comes back is held until the
        next record, and :attr:`torque` hands it out; a harness reusing one
        scratch buffer per tick would otherwise have the panel reporting a
        number no frame ever drew. `asarray` copies only when the dtype differs,
        which is exactly the case that is not worth relying on.
        """
        array = np.array(values, dtype=np.float64)
        if array.ndim == 1 and array.size == 0:
            return None
        if array.shape != shape:
            raise ValueError(complaint.format(shape=array.shape))
        return array

    def _observe_disagreement(
        self, disagreement: np.ndarray | None, tick: int
    ) -> None:
        """This tick's edge disagreement, into the marks that are drawn from it.

        **A boundary cell's mark is the disagreement on its own edges**, taken
        as `sqrt(Σ_e ‖d_e‖²)` over the edges incident on it -- the cell's own
        share of the sheaf's Dirichlet energy, which is the one quantity that
        exists for a cell of any degree with edge stalks of any width. A mean
        would say a cell with one loud edge and five quiet ones is calm, and a
        sum would make degree the brightest thing on the strip.

        From there it is the predicting cells' channel exactly: per-mark running
        statistics, the same `1 - exp(-z)` ramp, the same raw map behind the
        same debug flag. The two populations never meet -- this reads
        :attr:`~patchworks.surface.record.TickRecord.disagreement` and nothing
        else, and the marks it fills are indexed by boundary cell.

        A record carrying no disagreement leaves every mark unread and unlit,
        which draws as an empty slot. *Not captured* is not *zero*, and it is a
        fact about **this** record: a mark drawn calm because the last capture
        happened to leave the array out would be a graph reported as agreeing
        on every edge by nobody.
        """
        self._carried = disagreement is not None
        if disagreement is None:
            self._mark_value = np.zeros(len(self._marks))
            self._mark_raw = np.zeros(len(self._marks))
            self._mark_standing = np.zeros(len(self._marks))
            self._last_disagreement = np.zeros(len(self.dome.edges))
            self._drawn_edges, self._unread_edges, self._threshold = (), (), 0.0
            return
        # The pad slot the mark index points spare degrees at: zero, so a cell
        # of below-maximum degree sums only its own edges.
        padded = np.append(disagreement, 0.0)
        per_mark = np.sqrt((padded[self._mark_edges] ** 2).sum(axis=1))
        readable, _rescale = self._mark_scale.observe(per_mark, tick)
        # Both maps, every tick: the colour channel takes whichever the debug
        # flag asks for, and the motor strip's standing bar takes the raw one
        # whatever the flag says -- see :meth:`_draw_motor_strip`.
        self._mark_raw = self._mark_scale.raw(readable)
        self._mark_value = (
            self._mark_raw if self.raw else self._mark_scale.normalised(readable)
        )
        # And the strip's standing bar: every mark against **this tick's**
        # largest, shared across the marks and recomputed each tick. Shared,
        # because comparison between the marks is what that bar is for; per
        # tick, because an all-time maximum is a ratchet -- one spike on the
        # drive permanently flattened the actuator's bar beside it, which is
        # the signature `04` is read off. A mark with no reading is kept out of
        # the divisor as it is kept out of everything else here.
        read = ~self._mark_scale.no_reading
        loudest = float(readable[read].max(initial=0.0)) if read.any() else 0.0
        self._mark_standing = (
            np.where(read, readable / loudest, 0.0)
            if loudest > 0.0
            else np.zeros(len(self._marks))
        )
        self._last_disagreement = disagreement
        (
            self._threshold,
            self._drawn_edges,
            self._unread_edges,
        ) = _above_the_ticks_own_scale(disagreement)

    def _observe_torque(self, actuator: np.ndarray | None) -> None:
        """The actuator's commanded and applied rows, held for the strip to draw.

        There is no scale to accumulate: both rows are drawn against
        :data:`_ACTION_BOUND`, which is the environment's declared action space
        and not a quantity this panel measures. The six numbers share it because
        they are components of one action in the contract's own units -- **not**
        because they are torques in the same physical units, which they are not:
        the arm multiplies each by its own limit (3 / 2 / 1 N·m), so equal bar
        heights on two joints are equal *fractions of each joint's own limit*
        and not equal torques.
        """
        self._torque = np.zeros((2, 0)) if actuator is None else actuator

    # -- pixels ------------------------------------------------------------

    def _draw(
        self,
        *,
        tiles: tuple[np.ndarray, int] | None = None,
        since: int | None = None,
    ) -> np.ndarray:
        canvas = np.empty((self.height, self.width, 3), dtype=np.uint8)
        canvas[:, :] = _BACKGROUND
        colours = colormap(self._glow)
        marks = colormap(self._mark_value)
        for slot in self.layout.slots:
            y, x, size = self.rect(slot.cell)
            row = self._row.get(slot.cell)
            mark = self._mark_row.get(slot.cell)
            if row is not None:
                colour = _NO_READING if self._cell_scale.no_reading[row] else colours[row]
            elif mark is not None and self._carried:
                colour = (
                    _NO_READING if self._mark_scale.no_reading[mark] else marks[mark]
                )
            else:
                # A patch cell, or a mark this record carried nothing for.
                colour = _EMPTY
            canvas[y : y + size, x : x + size] = colour
        if tiles is not None:
            self._draw_render(canvas, *tiles)
        if self._edges:
            self._draw_edges(canvas)
        self._draw_motor_strip(canvas, since=since)
        if not self.raw:
            warming, marks_warming = self.warming_up, self.boundary_warming_up
            if warming or marks_warming:
                _draw_text(
                    canvas,
                    self._notice(warming, marks_warming),
                    top=self._notice_scale,
                    left=self._notice_scale,
                    scale=self._notice_scale,
                    ink=_NOTICE_INK,
                )
        return canvas

    def _tiles(self, render: np.ndarray) -> tuple[np.ndarray, int]:
        """The render, and the side of one cell's block of it.

        The tiling is **the one the world writes through**
        (:mod:`patchworks.agent`): patch cell `(r, c)` owns the block at `(r,
        c)` of the lattice, so a render the cells were never cut from is
        refused rather than resampled into one.
        """
        image = np.asarray(render)
        if image.ndim != 3 or image.shape[2] != 3 or image.shape[0] != image.shape[1]:
            raise ValueError(
                "the boundary band draws the agent's own render, which is a "
                f"square `[side, side, 3]` image; got {image.shape}"
            )
        if image.dtype != np.uint8:
            raise ValueError(
                f"the render is `uint8` -- the world's own, as the env's "
                f"observation space declares it -- and this one is {image.dtype}. "
                "Refused rather than converted: a normalised image assigned into "
                "the frame draws a black arena, which is a picture rather than a "
                "refusal, and nothing here can tell a scaled render from a dark "
                "one. Convert it where the convention that scaled it is known."
            )
        grid = self.dome.spec.patch_grid
        side, remainder = divmod(image.shape[0], grid)
        if remainder or side < 1:
            raise ValueError(
                f"a {grid}x{grid} patch lattice does not tile a "
                f"{image.shape[0]}-pixel render. The tiling is the one the world "
                "writes through (`patchworks.agent`), so a render the cells were "
                "never cut from is refused rather than resampled."
            )
        return image, side

    def _draw_render(self, canvas: np.ndarray, image: np.ndarray, side: int) -> None:
        """The boundary band: the agent's own render, tiled into the patch lattice.

        Each patch cell's slot holds **that cell's own block of the render**,
        cut the way the world writes it into the cell's node stalk, so the
        picture at the bottom of the panel is the thing the arm is doing in the
        other window and each square of it is what one cell is looking at. The
        block is scaled to the slot by repetition -- nearest neighbour, no
        interpolation -- because a smoothed patch would be a picture of pixels
        that were never written anywhere.
        """
        size = self.layout.mark
        # Which source row and column each drawn pixel comes from: computed
        # once, since every patch is the same shape.
        take = np.minimum((np.arange(size) * side) // size, side - 1)
        for cell in self._patches:
            r, c = cell.index.position
            block = image[
                r * side : (r + 1) * side, c * side : (c + 1) * side
            ]
            y, x, _size = self.rect(cell.id)
            canvas[y : y + size, x : x + size] = block[take][:, take]

    def _draw_edges(self, canvas: np.ndarray) -> None:
        """The route: the edges carrying the most disagreement this tick.

        A line between the two slots, on the same colormap as the marks, drawn
        over them rather than under: the overlay is off by default and turned on
        to answer *which cells were carrying it*, and a route hidden behind the
        marks it runs between would not answer it.
        """
        for edge_id in self._unread_edges:
            u, v = self._edge_ends[edge_id]
            _draw_line(canvas, self._centre(int(u)), self._centre(int(v)), _NO_READING)
        if not self._drawn_edges:
            return
        drawn = np.array(self._drawn_edges, dtype=np.int64)
        disagreement = self._last_disagreement[drawn]
        # Against this tick's largest, so the brightest line is the loudest edge
        # now. The threshold below it is the same tick's, so the ramp is read
        # over the range that survived it rather than over a constant.
        peak = float(disagreement.max(initial=0.0))
        colours = colormap(disagreement / peak if peak > 0 else disagreement)
        for edge_id, colour in zip(self._drawn_edges, colours):
            u, v = self._edge_ends[edge_id]
            _draw_line(canvas, self._centre(int(u)), self._centre(int(v)), colour)

    def _centre(self, cell_id: int) -> tuple[int, int]:
        """The middle of one cell's mark, in frame pixels."""
        top, left, size = self.rect(cell_id)
        return top + size // 2, left + size // 2

    def _draw_motor_strip(self, canvas: np.ndarray, *, since: int | None) -> None:
        """The actuator, decomposed: three paired bars and its disagreement.

        **Commanded as an outline, applied as a fill** -- efference copy made
        visible. Both rows are drawn against :data:`_ACTION_BOUND`, the
        environment's declared action space, so a height here is an absolute
        reading: *near-zero commanded torque* means near zero on every tick of
        every run. Since the applied row is the command clipped to that same
        bound, the two rows can only differ once the command has passed it --
        so **saturation is an outline running off the top of the strip, left
        open, beside a fill standing at the bound**, which is the shortfall the
        pair exists to show.

        The fourth bar is the actuator's own motor-side disagreement, and it is
        drawn **relative to the other boundary marks on this tick**: each mark
        against the largest of them, recomputed every tick.
        `04-action-and-the-boundary.md`'s route-selection signature is
        *near-zero commanded torque with standing motor-side disagreement*, and
        on the normalised colour map that bar would habituate -- a chronic stall
        would settle to its own baseline and render calm, which is the
        consequence `10-the-demo-surface.md` accepts for the colour channel and
        cannot be accepted for the one bar a falsification test is read off.

        **That height is a comparison, never a quantity**, and the difference
        matters. Disagreement has no natural unit, so there is no absolute
        reference to draw it against -- an all-time maximum looks like one and
        is a ratchet, where one spike anywhere on the boundary flattens every
        later tick, and a mark's *own* peak is worse still, since a mark sits at
        its own peak whenever it is quiet and the bar would stand through a
        silent run. What is honest is which boundary mark is loudest now, so
        some mark is at full height on every tick and that is the reading rather
        than a defect. Ruled on in #94.

        **A bar with no reading spans the whole strip in its own colour**
        (:data:`_NO_READING`) rather than standing at zero. A bar encodes its
        quantity in a height, and there is no height that means *not a number*:
        drawing one at the zero line would put *near-zero commanded torque* --
        half of `04`'s stall signature -- on screen out of no reading at all, on
        the one mark a falsification test is read off. A pair is one mark, so a
        non-finite half takes both with it, and a full-height column crosses the
        zero line, which no torque bar ever does.

        **The bars and the bar beside them are one tick apart**, and are meant
        to be. The rows are what the world read and wrote on this tick; the
        disagreement is what the actuator's edges carried when the cells
        broadcast, which is while the cell's stalk still held the *previous*
        tick's efference copy
        (:attr:`~patchworks.surface.record.TickRecord.disagreement`). That is
        the unit delay the whole graph runs on rather than a reading taken late,
        and it costs the signature nothing: a stall is a swing's worth of ticks,
        not one.

        The onset counter goes below the bars, so the ticks since the last
        marker and the moment of the first corrective torque are read off one
        strip.
        """
        top, left, height, width = self.motor_strip
        # `self._notice_height`, not the formula again: the strip's zero line
        # and every bar height are measured down from it, so a second copy is
        # two places for the strip's geometry to drift apart.
        bottom = top + height - self._notice_height
        zero = (top + bottom) // 2
        half = max(1, (bottom - top) // 2 - 1)
        # :data:`_ACTION_BOUND` draws to one row short of the bar's full reach,
        # and the row that leaves is entered **only** by a command that passed
        # the bound. So a saturating command stands one row above the fill that
        # met the bound beside it, and *the fill falls short of its outline* is
        # what the strip literally draws -- no second ink, and no rescaling of
        # the strip around an overrun.
        reach = max(1, half - 1)
        canvas[zero, left + self._bar_margin : left + width - self._bar_margin] = _ZERO_LINE

        for joint in range(self._torque.shape[1]):
            commanded, applied = self._torque[0, joint], self._torque[1, joint]
            if not (np.isfinite(commanded) and np.isfinite(applied)):
                x = self._bar_left(left, joint)
                canvas[top:bottom, x : x + self._bar] = _NO_READING
                continue
            _draw_bar(
                canvas,
                zero=zero,
                left=self._bar_left(left, joint),
                width=self._bar,
                height=self._scaled(commanded, reach, half),
                colour=_COMMANDED,
                fill=False,
            )
            _draw_bar(
                canvas,
                zero=zero,
                left=self._bar_left(left, joint) + 1,
                width=max(1, self._bar - 2),
                height=self._scaled(applied, reach, half),
                colour=_APPLIED,
                fill=True,
            )
        # Always the last bar, whether or not the record carried the rows
        # beside it: a mark that moved along the strip with what was captured
        # would be a different mark from one tick to the next.
        if self._actuator is not None and self._carried:
            row = self._mark_row[self._actuator]
            x = self._bar_left(left, self._joints)
            if self._mark_scale.no_reading[row]:
                canvas[top:bottom, x : x + self._bar] = _NO_READING
            else:
                standing = float(self._mark_standing[row])
                _draw_bar(
                    canvas,
                    zero=zero,
                    left=x,
                    width=self._bar,
                    height=int(round(standing * half)),
                    colour=tuple(int(channel) for channel in colormap(standing)),
                    fill=True,
                )
        if since is not None:
            # In full, and clipped at the frame if it outgrows the strip: a
            # count that ran off the edge is visibly wrong, where a saturated
            # one is a plausible number that is not the reading. A harness that
            # misses an `OnsetCounter.restart()` is the way to get there, and
            # that class documents a restore as invisible to it.
            _draw_text(
                canvas,
                f"T+{since}",
                top=bottom + self._notice_scale,
                left=left,
                scale=self._notice_scale,
                ink=_NOTICE_INK,
            )

    def _bar_left(self, left: int, index: int) -> int:
        """Where one of the strip's bars starts, counting from the strip's edge."""
        return left + self._bar_margin + index * (self._bar + self._bar_gap)

    def _scaled(self, torque: float, reach: int, limit: int) -> int:
        """One torque, in pixels above the strip's zero line.

        Against :data:`_ACTION_BOUND`, absolutely -- see there. `reach` is the
        height that bound draws to and `limit` the height the bar has, and they
        differ by the row reserved for an overrun: a command is deliberately
        unclipped, so its outline can ask for more than the bound, and that
        asks for the reserved row rather than rescaling the strip around it
        (:meth:`_draw_motor_strip`).

        Only ever called on a reading: a non-finite one is drawn as no reading
        rather than reduced to a height, and its caller has already left.
        """
        return int(np.clip(round(torque / _ACTION_BOUND * reach), -limit, limit))

    def _notice(self, warming: int, marks: int = 0) -> str:
        """What the panel says while the statistics are warming up.

        On screen, in the frame, rather than in a log the capture does not
        carry: a viewer who cannot see that the map is still noise has been
        misled by the picture. The marks drawn from edge disagreement are
        counted beside the cells rather than folded in with them, because they
        are a different population warming to a different condition
        (:attr:`boundary_baseline`), and one number over both would be a count
        of nothing in particular.
        """
        notice = f"WARMING UP {warming}/{len(self.persistence)} CELLS"
        if marks:
            notice += f" {marks}/{len(self._marks)} MARKS"
        return notice


# -- marks that are not squares ---------------------------------------------


def _above_the_ticks_own_scale(
    disagreement: np.ndarray,
) -> tuple[float, tuple[int, ...], tuple[int, ...]]:
    """The edges carrying the most disagreement this tick, and the bar they cleared.

    **The threshold is derived from the tick's own scale, never hand-set**
    (`docs/spec/10-the-demo-surface.md`, *Edges: thresholded, and off by
    default*): this tick's mean plus one standard deviation, over the edges that
    have a reading. A fixed magnitude would make the panel's picture of the
    route an artifact of the constant -- the same objection `05-timescales.md`
    raises against hand-set thresholds on the change gate, and it applies to a
    display for the same reason.

    What makes this rule *scale-free* rather than merely computed is that both
    terms are homogeneous in the disagreement: multiply every edge by any
    positive constant and the same edges clear the bar. Nothing here carries
    units, so nothing here can be tuned to make a route appear.

    An edge with no reading is left out of the **statistics** -- a NaN would
    make the threshold NaN and empty the overlay, which is the quietest possible
    picture of a graph that has just diverged -- and is returned separately
    rather than dropped. It is drawn whatever the threshold says, in its own
    colour: an edge whose disagreement has left the numbers is the loudest thing
    on the graph, and a route drawn *around* the divergence would be the same
    quiet picture arrived at more slowly.
    """
    finite = np.isfinite(disagreement)
    unread = tuple(int(edge) for edge in np.flatnonzero(~finite))
    if not finite.any():
        return 0.0, (), unread
    readings = disagreement[finite]
    threshold = float(readings.mean() + readings.std())
    drawn = np.flatnonzero(finite & (disagreement > threshold))
    return threshold, tuple(int(edge) for edge in drawn), unread


def _draw_bar(
    canvas: np.ndarray,
    *,
    zero: int,
    left: int,
    width: int,
    height: int,
    colour,
    fill: bool,
) -> None:
    """One bar of the motor strip, up or down from the zero line.

    `fill=False` draws the outline alone -- what was **commanded** -- so that
    the fill drawn inside it can fall short and be seen to. A bar of no height
    is still drawn, one pixel at the zero line, because *zero torque* is a
    reading and an absent bar is not.
    """
    top = zero - height if height >= 0 else zero
    bottom = zero if height >= 0 else zero - height
    top, bottom = max(0, min(top, bottom)), max(top, bottom)
    if bottom == top:
        bottom = top + 1
    if fill:
        canvas[top:bottom, left : left + width] = colour
        return
    canvas[top:bottom, left] = colour
    canvas[top:bottom, left + width - 1] = colour
    edge = top if height >= 0 else bottom - 1
    canvas[edge, left : left + width] = colour


def _draw_line(
    canvas: np.ndarray, start: tuple[int, int], end: tuple[int, int], colour
) -> None:
    """A one-pixel line between two marks' centres, clipped at the frame."""
    height, width = canvas.shape[:2]
    (y0, x0), (y1, x1) = start, end
    steps = max(abs(y1 - y0), abs(x1 - x0))
    if steps == 0:
        if 0 <= y0 < height and 0 <= x0 < width:
            canvas[y0, x0] = colour
        return
    for step in range(steps + 1):
        y = y0 + round((y1 - y0) * step / steps)
        x = x0 + round((x1 - x0) * step / steps)
        if 0 <= y < height and 0 <= x < width:
            canvas[y, x] = colour


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
    "+": "..../..#./.###/..#./....",
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
