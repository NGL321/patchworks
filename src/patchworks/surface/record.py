"""The tick record: the snapshot/restore contract, plus what the panel draws.

`docs/spec/10-the-demo-surface.md`, *The trace*, is the whole of what this
module implements, over `docs/spec/03-the-sandbox.md`'s *The Gymnasium
contract, made continual*.

**Nothing here is part of the architecture.** What a record holds is privileged
state -- what each cell's prediction missed by, how far its private component
moved, where the human's hands fired -- on exactly the footing
`03-the-sandbox.md` gives `info`: for looking at, never fed back. No cell reads
anything computed here, and the guarantee is structural rather than careful:

* **A recorder is driven from outside and nothing holds a reference to one.**
  :class:`Recorder` holds the agent; the agent holds no recorder, the sheaf
  holds no recorder, and nothing a cell's computation is handed can reach one.
  It is the shape :mod:`patchworks.timescale` gives the clock divisor, for the
  same reason.
* **A recorder only ever reads.** It gathers node stalks, subtracts, takes
  norms, and asks the sandbox for the state it already knows how to hand out.
  Nothing it does writes to a chart, a node stalk, an edge buffer or the
  world, so switching it off changes no trajectory --
  `tests/test_surface.py` asserts that bit for bit rather than leaving it to
  inspection.

**A trace is state, not frames.** A record holds no image. The scene
re-renders from MuJoCo offscreen at capture time
(:mod:`patchworks.surface.renderer`), so the capture resolution is chosen when
rendering rather than baked into the recording -- which is what lets the README
capture, a falsification sweep and a debugging pass all read the same file. It
also keeps the live budget near zero: the arrays are `~150 cells x 2 floats`
plus `~700 edges x 1` plus the actuator's six, and the state is the vector
snapshot/restore already takes.

**Why an edge array as well as the two cell ones.** `10-the-demo-surface.md`
names three privileged quantities the panel reads -- prediction error, private
components, **edge disagreement** -- and the marks that draw the third are the
boundary band's: the somatomotor strip, the drive mark and the thresholded edge
overlay (#94). A boundary cell runs no body and makes no prediction, so what it
has is an edge, and a record without that array could feed those marks live and
never off disk -- which is exactly the live/replay split *The trace* exists to
refuse. So it rides in the record like the other two, at the same order of size,
and is the same kind of thing: privileged state, read, never fed back.

**Not a new format.** A record is `03-the-sandbox.md`'s
:class:`~patchworks.sandbox.state.Snapshot` -- `mjSTATE_INTEGRATION`, the task
and the sampler's RNG -- plus those arrays and any markers that fired. What
:meth:`Trace.save` writes is that and nothing else, so a record read back off
disk restores a world the same way the one held in memory does.

**What a trace cannot hold: a record no tick produced.** Every record is one
tick's reading, and #116 ruled that it stays that way. A trailing record
fabricated after a run -- to give a marker fired on the last tick somewhere to
land -- would have to carry *something* in the four arrays, and neither answer
is honest. **Zeros** read back as a graph that agreed on every edge, predicted
perfectly, and moved no private component: the reading :func:`_nothing` exists
to be distinct from, and the one :meth:`Trace.save` already declines to pad
with. ***Not captured*** is
a shape the rest of the trace disagrees with, so :meth:`Trace.save` would
refuse the very trace such a record was added to complete.

So the marker is what gives instead, and the loss is stated rather than
patched: see :meth:`Recorder.mark`, *What this cannot see*, and
:func:`~patchworks.surface.gestures.drive`. A run that must not lose its last
event **declares one tick more than it measures**.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

from patchworks.agent import Agent, run
from patchworks.sandbox.env import CONTROL_HZ, Task
from patchworks.sandbox.state import Snapshot, snapshot

__all__ = [
    "CAPTURE_EVERY",
    "CAPTURE_HZ",
    "Event",
    "EventKind",
    "Recorder",
    "TickRecord",
    "Trace",
]

#: How often a run is captured, in hertz. `docs/spec/10-the-demo-surface.md`,
#: *The trace*: live mode feeds the renderer at ~10 Hz. The number is a display
#: rate and nothing reads it but the recorder -- the env still ticks at
#: `CONTROL_HZ`, and what a capture decimates is how much of that run is kept,
#: never how much of it happens.
CAPTURE_HZ = 10.0

#: One capture every this many ticks, from the two rates above. At the
#: sandbox's 50 Hz control that is every fifth tick.
CAPTURE_EVERY = round(CONTROL_HZ / CAPTURE_HZ)


def _nothing() -> np.ndarray:
    """An array that carries no reading. *Not captured*, distinct from zero.

    A default for a record's optional arrays: a test builds one by hand to
    exercise a mark it is not about, and a hand-built record that defaulted to
    zeros would say the graph agreed on every edge -- the calmest thing the
    panel can draw, asserted by nobody.
    """
    return np.zeros(0)


class EventKind(str, Enum):
    """The three hands, and nothing else.

    `docs/spec/03-the-sandbox.md`, *The human's hand*, names exactly three
    entry points, and `10-the-demo-surface.md` binds each to a gesture. A
    marker names which one fired; #96 is where the gestures call them.
    """

    DISTURB_ARM = "disturb_arm"
    PERTURB = "perturb"
    RETARGET = "retarget"


@dataclass(frozen=True)
class Event:
    """One marker: which hand fired, on which tick, with what arguments.

    The tick is the one the hand fired on, **not** the one the marker was
    captured with. A capture is decimated and an event is not, so a marker
    that carried only its record's tick would put the demo's temporal measure
    -- onset latency, counted in ticks (`docs/spec/08-the-acceptance-demo.md`)
    -- at the mercy of the display's rate. It carries its own.

    `detail` is the hand's own arguments, in the order the hand takes them, so
    that a debugging pass can read back what was done rather than only that
    something was. It is coerced to plain floats **here**, because the hands
    are bound to gestures (#96) and a gesture hands over whatever it picked off
    a numpy array or a MuJoCo pick. Left uncoerced, a `np.float32` is accepted
    silently and then refused by `json.dumps` at :meth:`Trace.save` -- at the
    end of the run, taking the whole trace with it. Coercing at construction
    turns that into a `TypeError` on the marker that caused it.
    """

    kind: EventKind
    tick: int
    detail: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "detail", tuple(float(value) for value in self.detail))


# eq=False for `Snapshot`'s reason: the fields are arrays, so a generated
# __eq__ would raise rather than compare.
@dataclass(frozen=True, eq=False)
class TickRecord:
    """One captured tick. The contract, plus the arrays, plus what fired.

    Rows of the two **cell** arrays are indexed by
    :attr:`~patchworks.graph.Dome.predicting` -- the predicting population, in
    the order the biases are indexed by. Boundary cells are absent from both on
    purpose: they run no body and make no prediction (ADR-0006), so a prediction
    error for one would be a fabrication, and the marks the panel gives them are
    drawn from :attr:`disagreement` instead
    (`docs/spec/10-the-demo-surface.md`, *The boundary band*). That array is
    indexed by edge, not by cell, and is the whole of what a boundary cell has.
    """

    tick: int
    """The sheaf's tick counter when this was captured."""

    state: Snapshot
    """`03-the-sandbox.md`'s snapshot/restore contract, unchanged: the physics
    under `mjSTATE_INTEGRATION`, the task, and the sampler's RNG. This is what
    makes a trace replayable rather than merely readable -- a record restores
    the world it was taken from."""

    prediction_error: np.ndarray
    """`[predicting cells]`: the magnitude of each cell's prediction error, and
    the panel's primary channel (`docs/spec/10-the-demo-surface.md`, *Colour is
    prediction error*). Raw, not normalised: normalising against a cell's own
    running statistics is the panel's job (#93), and a record that had already
    done it could not also serve the raw map that section keeps behind a debug
    flag."""

    private_delta: np.ndarray
    """`[predicting cells]`: `‖Δ(private component)‖`, tick to tick. The
    private component is the node-stalk directions masked out on every incident
    edge, known at construction, so this is a fixed projection computed per
    tick (`docs/spec/05-timescales.md`, *Demonstrating it*)."""

    disagreement: np.ndarray = field(default_factory=_nothing, kw_only=True)
    """`[edges]`: the magnitude of each edge's disagreement, indexed by
    :attr:`~patchworks.graph.Dome.edges` rather than by cell -- disagreement is
    edge-owned and spatial where prediction error is cell-owned and temporal
    (`CONTEXT.md`), and an edge has two ends with no reason to prefer one.

    This is what every mark a **boundary cell** gets is drawn from
    (`docs/spec/10-the-demo-surface.md`, *The somatomotor strip*, *The drive
    mark*, *Edges: thresholded, and off by default*). Raw, like the prediction
    error beside it, and for the same reason: the panel normalises against a
    mark's own running statistics, and a record that had already done it could
    not also serve the raw map.

    **It is the tick's disagreement as the sheaf defines it**, derived from what
    each end *broadcast* during this tick's message-passing phase
    (:meth:`~patchworks.tick.Sheaf.disagreement`) -- which is before the
    external write lands, since that write is the tick's last word
    (`docs/spec/02-tick-semantics.md`). So a motor edge's disagreement is
    measured against the previous tick's efference copy, and a sensory edge's
    against the previous tick's observation. That is not a reading taken late:
    it is where the boundary's content sits when the graph speaks, because the
    external write is the tick's last word.

    It is **not** what a cell reconciled against, which is the partner's
    broadcast from `t − 1` -- :meth:`~patchworks.tick.Sheaf.disagreement` takes
    both terms from the same tick, so the unit delay does not show up inside it.
    It is the sheaf's coboundary on this tick, which is the quantity `CONTEXT.md`
    defines and the one an edge is judged by. Read the strip's disagreement bar
    as belonging to the swing rather than to the single tick beside it.

    Empty when a record was built without one -- **no disagreement was
    captured**, and the panel then draws no mark that would need it rather than
    a calm one. A record from :meth:`Recorder.observe` always carries it."""

    actuator: np.ndarray = field(default_factory=_nothing, kw_only=True)
    """`[2, joints]`: the actuator boundary cell's own node stalk, decomposed --
    row 0 the **commanded** components the world read, row 1 the **applied**
    ones it wrote back, post-clip and post-saturation.

    `04-action-and-the-boundary.md`'s efference copy, as the cell holds it: not
    a second reading of the torque taken from the engine, but the six numbers
    that make the motor edge carry ordinary disagreement rather than being the
    one edge with none. The panel draws them paired, commanded as an outline and
    applied as a fill (`10-the-demo-surface.md`, *The somatomotor strip*), so a
    saturating command reads as the fill falling short of its outline.

    Empty when a record was built without one, exactly as
    :attr:`disagreement` is."""

    events: tuple[Event, ...] = ()
    """The markers that fired since the previous capture, in the order they
    fired."""


class Trace(Sequence[TickRecord]):
    """A run's records, in order, and the file they go to.

    A sequence rather than a stream, because every consumer the record has
    wants a different pass over the same one: the README capture takes two
    short stretches of it, a falsification sweep reads the arrays and never
    renders a frame, and a debugging pass restores from the middle of it.
    """

    def __init__(self, records: Sequence[TickRecord] = ()) -> None:
        self._records: list[TickRecord] = list(records)

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index):  # type: ignore[override]
        # A slice of a trace is a trace: the README's two short loops are two
        # stretches of one run, and each of them is the same thing to render
        # as the whole.
        if isinstance(index, slice):
            return Trace(self._records[index])
        return self._records[index]

    def __iter__(self) -> Iterator[TickRecord]:
        return iter(self._records)

    def __repr__(self) -> str:
        return f"Trace({len(self._records)} records)"

    def append(self, record: TickRecord) -> None:
        self._records.append(record)

    # -- the file ----------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        """Write the trace to one `.npz`. Returns the path written.

        The file holds the contract and the arrays and nothing else -- no
        frames, no resolution, no colormap, nothing a display chose. What is
        stored is stacked by field rather than by record, because every
        consumer reads one field across the whole run.

        A trace whose records disagree about whether they carry one of the
        optional arrays is refused rather than padded: a zero written in for a
        record that never captured one would read back as a graph agreeing on
        every edge.

        Two fields go through JSON, and both for the same reason: `npz` stores
        arrays. The sampler's RNG state carries PCG64's 128-bit integers, which
        no numpy integer dtype holds, and a lossy round trip there is a replay
        that diverges rather than an error. The markers are a ragged list of
        heterogeneous tuples. JSON round-trips both exactly.
        """
        path = Path(path)
        records = self._records
        np.savez(
            path,
            tick=np.array([r.tick for r in records], dtype=np.int64),
            physics=_stack([r.state.physics for r in records], "physics"),
            puck_xy=_stack([r.state.task.puck_xy for r in records], "puck_xy"),
            puck_theta=_stack([r.state.task.puck_theta for r in records], "puck_theta"),
            goal=np.array(
                [(r.state.task.goal_puck, r.state.task.goal_zone) for r in records],
                dtype=np.int64,
            ).reshape(len(records), 2),
            prediction_error=_stack([r.prediction_error for r in records], "prediction_error"),
            private_delta=_stack([r.private_delta for r in records], "private_delta"),
            disagreement=_stack([r.disagreement for r in records], "disagreement"),
            actuator=_stack([r.actuator for r in records], "actuator"),
            rng=np.asarray(json.dumps([r.state.rng for r in records])),
            events=np.asarray(
                json.dumps(
                    [
                        [[e.kind.value, e.tick, list(e.detail)] for e in r.events]
                        for r in records
                    ]
                )
            ),
        )
        # `np.savez` appends the extension unless it is already there, and the
        # caller is owed the name that was actually written.
        if path.name.endswith(".npz"):
            return path
        return path.with_name(f"{path.name}.npz")

    @classmethod
    def load(cls, path: str | Path) -> "Trace":
        """Read a trace back. What comes out restores a world; see :meth:`save`.

        Every field is pulled out of the archive **once**, before the loop.
        `np.load` is lazy and re-reads the whole member on each subscript, so
        indexing one inside the loop would decompress the entire run per record.

        **A file written before one of the optional arrays existed still
        reads.** What comes back for it is *not captured*, which is exactly
        what happened, and the marks drawn from it then draw nothing rather
        than a zero. A trace is meant to outlive the run that made it, and a
        capture from last month is not made wrong by a mark added since.
        """
        with np.load(Path(path)) as stored:
            rng = json.loads(str(stored["rng"].item()))
            events = json.loads(str(stored["events"].item()))
            ticks = stored["tick"]
            physics = stored["physics"]
            puck_xy = stored["puck_xy"]
            puck_theta = stored["puck_theta"]
            goal = stored["goal"]
            prediction_error = stored["prediction_error"]
            private_delta = stored["private_delta"]
            disagreement = _optional(stored, "disagreement", len(ticks))
            actuator = _optional(stored, "actuator", len(ticks))
        records = []
        for i, tick in enumerate(ticks):
            state = np.array(physics[i])
            # Sealed on the way out for the reason `snapshot()` seals it: a
            # state that drifts while it is being held is a restore that lands
            # somewhere nobody chose.
            state.flags.writeable = False
            records.append(
                TickRecord(
                    tick=int(tick),
                    state=Snapshot(
                        physics=state,
                        task=Task(
                            puck_xy=np.array(puck_xy[i]),
                            puck_theta=np.array(puck_theta[i]),
                            goal_puck=int(goal[i][0]),
                            goal_zone=int(goal[i][1]),
                        ),
                        rng=rng[i],
                    ),
                    prediction_error=np.array(prediction_error[i]),
                    private_delta=np.array(private_delta[i]),
                    disagreement=np.array(disagreement[i]),
                    actuator=np.array(actuator[i]),
                    events=tuple(
                        Event(EventKind(kind), int(at), tuple(detail))
                        for kind, at, detail in events[i]
                    ),
                )
            )
        return cls(records)


def _optional(stored, name: str, records: int) -> list[np.ndarray]:
    """One of the archive's optional members, or *not captured* per record.

    See :meth:`Trace.load`: a file written before a field existed carries no
    member for it, and what that means is that nothing was captured -- not that
    the file is unreadable, and never that the quantity was zero.
    """
    if name not in stored.files:
        return [_nothing() for _ in range(records)]
    return stored[name]


def _stack(arrays: list[np.ndarray], what: str = "field") -> np.ndarray:
    """`np.stack`, with an empty trace surviving it and a ragged one explained.

    The optional arrays default to *not captured*, so a trace assembled by hand
    can hold records that carry one and records that do not. `np.stack` refuses
    that with a shape complaint naming neither the field nor the record, and
    the fix is never to pad -- see :meth:`Trace.save`.
    """
    if not arrays:
        return np.zeros((0,))
    shapes = {array.shape for array in arrays}
    if len(shapes) > 1:
        odd = next(
            i for i, array in enumerate(arrays) if array.shape != arrays[0].shape
        )
        raise ValueError(
            f"this trace's records disagree about `{what}`: record 0 carries "
            f"{arrays[0].shape} and record {odd} carries {arrays[odd].shape}. A "
            "trace is one run through one graph, and a record that carried none "
            "of a quantity cannot be written beside records that did -- the zero "
            "it would be padded with is a reading nobody took."
        )
    return np.stack(arrays)


class Recorder:
    """Watches a run from outside and leaves a :class:`Trace`.

    Built on an agent, driven by whoever drives the ticks::

        recorder = Recorder(agent)
        for _ in recorder.watch(ticks=600, seed=0):
            ...

    or, when something else owns the loop -- the live viewer's event loop, a
    clock divisor's run, the acceptance demo's harness -- one call after each
    whole tick::

        agent.tick()
        recorder.observe()

    **Every tick, not every capture.** `‖Δ(private component)‖` is a difference
    between consecutive ticks, so :meth:`observe` has to see all of them even
    though it keeps one in :data:`CAPTURE_EVERY`. A caller who skipped ticks
    would silently redefine the readout as a difference over whatever interval
    it happened to call at, which is why skipping is refused rather than
    absorbed.
    """

    def __init__(self, agent: Agent, *, every: int = CAPTURE_EVERY) -> None:
        if isinstance(every, bool) or not isinstance(every, int) or every < 1:
            raise ValueError(
                f"a capture keeps one tick in `every` >= 1; got {every!r}"
            )
        self.agent = agent
        self.every = every
        self.trace = Trace()
        # The private component as an index, built once. It is a fixed
        # projection known at construction (`docs/spec/05-timescales.md`,
        # *Demonstrating it*), so the positions it keeps are a fixed array of
        # positions in the flat node stalk buffer -- the same act the agent's
        # write tables and `StalkLayout`'s index arrays are. Reading it that
        # way rather than as a mask over the whole population is what keeps
        # the per-tick cost of the surface where the spec claims it is: this
        # runs on every tick, because a difference between consecutive ticks
        # has to.
        mask = agent.sheaf.dome.private_mask
        self._cells = mask.shape[0]
        self._private_positions = agent.sheaf.layout.predicting_positions[mask]
        self._private_cell = torch.nonzero(mask)[:, 0].contiguous()
        self._previous_private: torch.Tensor | None = None
        self._observed: int | None = None
        self._pending: list[Event] = []

    def __repr__(self) -> str:
        return f"Recorder(every={self.every}, {len(self.trace)} records)"

    # -- the markers -------------------------------------------------------

    def mark(self, kind: EventKind | str, *detail: float) -> Event:
        """Drop a marker for a hand that just fired.

        Called by whoever fires the hand -- the bound gesture in the live
        viewer (#96), or a scripted demo. It is not called by the sandbox: a
        marker is a thing the surface records, and an env that dropped one
        would be the world knowing about the display.

        The marker rides on the next capture, carrying the tick it fired on.
        A tick that fires one is always captured, whatever the decimation says,
        because a marker decimated away is a marker lost.

        **What this cannot see: the end of the run.** There is one tick in
        which a marker exists and no record holds it -- the one between firing
        and the next :meth:`observe` -- and a run that stops in it never takes
        that next observation. The marker is then left in :attr:`pending`,
        where a caller can still read it, rather than dropped somewhere nothing
        reports; but it does not reach the trace, and this class cannot tell
        that the loop above it has ended. The tick it carries is right either
        way, so nothing recorded is wrong: what is lost is the last event of
        the run, in the trace only.

        The remedy is the caller's, because the missing thing is a tick and
        only the caller declares those. **A run that must not lose its last
        event declares one tick more than it measures** -- one iteration for
        the marker to be yielded on, at the cost of one tick and no semantics.
        That is a falsification sweep's obligation, next to the per-trial
        :meth:`~patchworks.surface.onset.OnsetCounter.restart` #95 handed
        forward, and this recorder can check neither. Fabricating the missing
        iteration here instead was ruled out in #116; see this module's
        docstring, *What a trace cannot hold*.
        """
        event = Event(EventKind(kind), self.agent.sheaf.ticks, tuple(detail))
        self._pending.append(event)
        return event

    @property
    def pending(self) -> tuple[Event, ...]:
        """The markers that have fired and not yet been captured, as a copy."""
        return tuple(self._pending)

    # -- what it reads -----------------------------------------------------

    def reprime(self) -> None:
        """Take the gap and carry on: forget the previous tick, keep the trace.

        The refusal in :meth:`observe` is terminal on purpose -- a recorder
        that quietly absorbed a missed tick would go on reporting a difference
        over an interval nobody chose, which is the one thing the refusal
        exists to prevent. But the remedy cannot be a new recorder either: that
        throws away everything captured so far, and a live viewer that dropped
        one call in an exception handler would lose the run rather than a tick.

        So the gap is taken deliberately. The next :meth:`observe` primes
        against the tick it sees and captures nothing, exactly as the first one
        does; from the one after, the readout is a tick-to-tick difference
        again. The markers still waiting are untouched.
        """
        self._previous_private, self._observed = None, None

    def observe(self) -> TickRecord | None:
        """Read the tick that just finished. Returns its record, or `None`.

        `None` on the ticks a capture skips, and on the first tick it sees:
        `‖Δ private‖` is a difference, and there is nothing yet to difference
        against, so the first observation primes it rather than fabricating a
        zero.

        Reads and nothing else. Both quantities come off the flat node stalk
        buffer through an advanced index, which returns a fresh tensor rather
        than a view -- so the tick's next in-place edit cannot reach back into
        one of these, and nothing here can reach into the tick. The world's
        half is :func:`~patchworks.sandbox.state.snapshot`, which the sandbox
        already offers an experimenter, and which takes the state without
        touching it.

        **What runs every tick and what runs on a capture** is the whole of
        the live budget. `‖Δ private‖` is a difference between consecutive
        ticks and has to be taken on all of them; prediction error, edge
        disagreement and the actuator's two rows are quantities of the tick they
        are read on, so they wait until there is a record to put them in.

        Prediction error is **read**, not recomputed. The bias rule's
        :func:`~patchworks.learning.prediction_error` re-runs the cell's
        forward path so the quantity is live in the biases and can be descended
        on; what a display wants is the opposite -- the dead number the tick
        left behind, off the tape, with no gradient in anything.
        """
        sheaf = self.agent.sheaf
        ticks = sheaf.ticks
        with torch.no_grad():
            private = sheaf.stalks[self._private_positions]

            if self._previous_private is None:
                self._previous_private, self._observed = private, ticks
                return None
            if ticks - self._observed != 1:
                raise ValueError(
                    f"the last tick this recorder saw was {self._observed} and this "
                    f"one is {ticks}. `‖Δ private‖` is a difference between "
                    "consecutive ticks, so observe() has to be called on every one "
                    "of them -- it is the capture that is decimated, not the "
                    "reading. See docs/spec/10-the-demo-surface.md, The trace. "
                    "To go on from here having lost some, call reprime(): it "
                    "keeps the trace and skips the one delta it cannot state."
                )
            moved = private - self._previous_private
            delta = (
                torch.zeros(self._cells)
                .index_add_(0, self._private_cell, moved * moved)
                .sqrt()
            )
            self._previous_private, self._observed = private, ticks

            if ticks % self.every and not self._pending:
                return None
            error = (sheaf.prediction - sheaf.evidence()).norm(dim=-1)
            # Derived from what both ends broadcast this tick, which is what
            # `Sheaf.disagreement` is for: the panel's boundary marks read the
            # magnitude per edge, so the `m_max` axis goes here rather than in
            # the record. Rows past an edge's own `m` are zero and contribute
            # nothing to the norm.
            disagreement = sheaf.disagreement().norm(dim=-1)
            # The actuator's own stalk, decomposed. Read *after* the whole tick,
            # so the commanded half is what the world read this tick and the
            # efference half is what it wrote back having clipped it -- the two
            # halves of one tick's efference copy rather than a command paired
            # with the previous tick's answer. `stack` allocates, so neither row
            # is a view on the flat buffer for the next tick to move.
            commanded = self.agent.commanded
            stalk = sheaf.stalk(self.agent.actuator_cell)
            actuator = torch.stack((stalk[:commanded], stalk[commanded:]))

        record = TickRecord(
            tick=ticks,
            state=snapshot(self.agent.env),
            prediction_error=error.numpy(),
            private_delta=delta.numpy(),
            disagreement=disagreement.numpy(),
            actuator=actuator.numpy(),
            events=tuple(self._pending),
        )
        self._pending.clear()
        self.trace.append(record)
        return record

    # -- driving the loop --------------------------------------------------

    def watch(
        self, ticks: int, *, seed: int | None = None
    ) -> Iterator[TickRecord]:
        """:func:`patchworks.agent.run`, watched. Yields each captured record.

        The run is :func:`~patchworks.agent.run`'s, called rather than
        reimplemented, so a watched run cannot drift from an unwatched one by
        so much as an ordering. The world is arranged when this is called, for
        the reason that function gives.

        This is the **live feed**: what it yields is what
        :meth:`patchworks.surface.renderer.Renderer.frames` consumes, and a
        trace off disk is the same thing from the other end. Live and replay
        are one code path with two feeds.

        A hand fired between two of these ticks is captured by the next one, so
        a hand fired after the last of them is not: see :meth:`mark`, *What
        this cannot see*. Declare one tick more than the measurement needs.
        """
        outcomes = run(self.agent, ticks, seed=seed)

        def watching() -> Iterator[TickRecord]:
            for _outcome in outcomes:
                record = self.observe()
                if record is not None:
                    yield record

        return watching()
