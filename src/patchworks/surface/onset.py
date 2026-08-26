"""Onset: the markers the hands drop, and the counter that runs from them.

`docs/spec/10-the-demo-surface.md`, *Onset, and the near-misses*: `08` measures
**onset latency** -- ticks from an event to the first corrective torque -- and
the surface owes two things for it.

* **An event marker the hands drop into the record when they fire.**
  :class:`Hands` is where a hand and its marker are the same call, so a fired
  hand cannot reach the world without the record hearing about it.
* **A tick counter on the motor strip, running from the most recent marker.**
  :class:`OnsetCounter` is that counter. **Onset is read off the strip rather
  than reconstructed afterward**, which is the point of counting live: a number
  a viewer watches climb, and stops watching when the first corrective torque
  appears. One thing it cannot do for its caller is notice a restore -- see
  :class:`OnsetCounter`, *What this cannot see*.

**No new quantity.** The commanded/applied bars on the somatomotor strip
already carry "first corrective torque" (`10-the-demo-surface.md`, *The
somatomotor strip*), so the counter supplies the ticks and the strip supplies
the moment. Drawing the strip, and reading the latency off it at that moment,
is #94's; what is here is the number it draws.

**Nothing here is part of the architecture.** A marker is a thing the *surface*
records: the sandbox drops none, because an env that dropped one would be the
world knowing about the display (:mod:`patchworks.surface.record`). :class:`Hands`
sits on the other side of that line -- it is a caller of the env's hands, on the
footing any experimenter is, and the world it calls into cannot see it.
"""

from __future__ import annotations

from typing import Any, Iterable, Iterator

from .record import Event, EventKind, Recorder, TickRecord

__all__ = ["Hands", "OnsetCounter"]


class Hands:
    """The three hands of `08`, bound so that firing one leaves a marker.

    ::

        hands = Hands(recorder)
        hands.perturb(puck=1, xy=(0.1, -0.2))

    The alternative -- calling the env's hand and then remembering to call
    :meth:`~patchworks.surface.record.Recorder.mark` -- is the one that loses
    the demo's temporal measure to a missed line, and it loses it *silently*:
    the run looks fine, the recovery looks fine, and onset is measured from
    nothing. So the two are one call. Every gesture in the live viewer (#96)
    and every scripted event in the demo harness fires a hand through here.

    **Fire first, mark second.** A hand refuses a puck or a joint that does not
    exist, and a marker for a hand that never landed would put an event in the
    record that the world never saw.

    The hands live on the unwrapped env: `perturb`, `disturb_arm` and
    `retarget` are the experimenter's entry points, not the Gymnasium
    contract's (`docs/spec/03-the-sandbox.md`, *The human's hand*), so a
    wrapped env is unwrapped once, here.
    """

    def __init__(self, recorder: Recorder) -> None:
        self.recorder = recorder

    def __repr__(self) -> str:
        return f"Hands({self.recorder!r})"

    @property
    def world(self) -> Any:
        """The env the hands reach, unwrapped."""
        env = self.recorder.agent.env
        return getattr(env, "unwrapped", env)

    def disturb_arm(self, joint: int, impulse: float) -> Event:
        """`08` event 1: an impulse to one joint. The world moves the body."""
        self.world.disturb_arm(joint, impulse)
        return self.recorder.mark(EventKind.DISTURB_ARM, joint, impulse)

    def perturb(self, puck: int, xy, theta: float | None = None) -> Event:
        """`08` event 2: teleport a puck. The world changes; what is wanted does not.

        The marker carries the hand's own arguments in the order it takes them,
        with `xy` flattened into them -- a marker's detail is a tuple of floats
        (:class:`~patchworks.surface.record.Event`), and an orientation is
        carried only when one was asked for, because a marker that invented a
        `theta` would read back as a hand that turned the puck.
        """
        x, y = (float(value) for value in xy)
        self.world.perturb(puck, (x, y), theta)
        detail = (puck, x, y) if theta is None else (puck, x, y, theta)
        return self.recorder.mark(EventKind.PERTURB, *detail)

    def retarget(
        self, goal_puck: int | None = None, goal_zone: int | None = None
    ) -> Event:
        """`08` event 3: change what is wanted. The world does not move.

        The marker carries the goal that now stands, not the arguments as
        passed: either may be `None`, meaning *leave that half alone*, and a
        marker holding a `None` would say less than the record beside it
        already does. What is written is what a viewer would read off the
        arena -- and the third event of the demo names the puck the second one
        displaced, so which puck it was is the whole of what makes the pair
        legible.
        """
        world = self.world
        world.retarget(goal_puck, goal_zone)
        task = world.task
        return self.recorder.mark(
            EventKind.RETARGET, task.goal_puck, task.goal_zone
        )


class OnsetCounter:
    """Ticks since the most recent event marker, read one record at a time.

    ::

        counter = OnsetCounter()
        for record in recorder.watch(ticks=600, seed=0):
            strip.draw(record, since=counter.count(record))

    **Counted in ticks, not in captures.** A capture is decimated and a marker
    is not -- an :class:`~patchworks.surface.record.Event` carries the tick its
    hand fired on rather than the tick it was captured with -- so the count is
    exact at 10 Hz and would stay exact at any other display rate. Onset
    latency is a property of the graph (`CONTEXT.md`, *Onset latency*), and a
    display rate is not allowed near it.

    `None` until a hand has fired: before the first marker there is nothing for
    a count to run from, and a zero there would read as an event that just
    happened.

    **What this cannot see: a restore.** A trial is *a restore to a snapshot
    tick, one event fired, one onset latency recorded*
    (`docs/spec/08-the-acceptance-demo.md`, *The repeated runs*) -- and a
    restore rewinds the world, the clock and the adapting surface without
    rewinding :attr:`~patchworks.tick.Sheaf.ticks`, which is a counter of ticks
    that happened rather than a property of the state. So a restore is
    invisible here for the same reason it is invisible from inside the graph
    (`CONTEXT.md`, *Restore*): nothing in a record reports one. A harness that
    restores and does not call :meth:`restart` goes on counting from the
    **previous** trial's marker, and what it gets is a large but perfectly
    plausible latency rather than an error. Calling :meth:`restart` per trial
    is the harness's to get right; this counter cannot check it, and says so
    rather than implying otherwise.

    **The harness owes one other thing, for the same reason.** A marker fired
    on a run's last tick never reaches the trace, because the capture that
    would have carried it is the next iteration's and there is no next one
    (:meth:`~patchworks.surface.record.Recorder.mark`, *What this cannot see*;
    #116). A trial that fires its event and stops on the tick it measures to
    therefore hands this counter a feed with no marker in it, and gets `None`
    where it expected a latency. So a trial **declares one tick more than it
    measures**, as well as calling :meth:`restart` before its event. Both are
    the harness's, and neither is checkable from here.
    """

    def __init__(self) -> None:
        self.since: Event | None = None
        """The marker the count is running from, or `None`."""

    def __repr__(self) -> str:
        return f"OnsetCounter(since={self.since!r})"

    def restart(self) -> None:
        """Forget the marker. What the demo's harness calls per trial.

        Once before each trial's event, because the counter cannot tell that a
        trial ended (*What this cannot see*, above). Carrying a marker across a
        restore counts from an event the restored run never had, and the number
        that comes out looks like a latency.
        """
        self.since = None

    def count(self, record: TickRecord) -> int | None:
        """Ticks from the most recent marker to this record. `None` before any.

        Fed the run's records in order. A record that carries markers restarts
        the count on the last of them: the counter runs from the **most
        recent** marker, and `08`'s three events arrive seconds apart in one
        unbroken run.
        """
        if record.events:
            self.since = max(record.events, key=lambda event: event.tick)
        if self.since is None:
            return None
        elapsed = record.tick - self.since.tick
        if elapsed < 0:
            raise ValueError(
                f"this record is tick {record.tick} and the marker being counted "
                f"from fired on {self.since.tick}, which is later. A counter is fed "
                "one run's records in order, so the clock it counts on only goes "
                "forward: this is a record from a rebuilt sheaf, or one fed out of "
                "order. Call restart() to count from the current trial's own event "
                "(docs/spec/08-the-acceptance-demo.md, The repeated runs). Note "
                "that a plain restore does **not** land here -- it rewinds the "
                "world and not the tick counter, so it is invisible to this class "
                "and restart() is the harness's to call."
            )
        return elapsed

    def counts(
        self, feed: Iterable[TickRecord]
    ) -> Iterator[tuple[TickRecord, int | None]]:
        """Each record of `feed` with its count, in order.

        The seam the strip draws from, and the same one
        :meth:`~patchworks.surface.renderer.Renderer.frames` and
        :meth:`~patchworks.surface.private_component.PrivateComponentPanel.frames`
        take: a live recorder's watch and a trace off disk are two iterables,
        not two code paths.
        """
        for record in feed:
            yield record, self.count(record)
