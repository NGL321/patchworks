"""`patchworks run`: a long headless run that shows it is progressing (#121).

The sibling of the panel, and deliberately not a small version of it. The panel
answers *watch it and understand it*; this answers **leave it running and come
back** — no display, no GPU, no window server, sane over SSH, and cheap enough
that the reporting is never the reason a run is slow.

**What it prints is the substance.** Not a progress bar: numbers that
distinguish a run that is learning from one that is stuck. #120 is the
cautionary case — an untrained agent whose command locks constant to one part
in 10⁵ with every joint pinned at its limit and 682 of 682 edges still
disagreeing — and a readout that would not have made that obvious is not worth
printing. So each report carries the tick and the rate, the disagreement mean
and max, the effective rank beside it, the torque asked and applied, the
per-joint spread of the command, the arm's travel since the last report, and
whether anything has gone non-finite. The travel and the spread are the two
columns #120 shows as exact zeros; :func:`format_summary` says so in words when
the run's own numbers say it.

**The paired instrument is #91's, not a second one.**
:class:`~patchworks.diagnostics.Diagnostics` already computes per-edge Dirichlet
energy alongside per-edge effective rank, never one without the other, and the
pairing is the whole of what makes either half readable: energy falling with
rank sliding toward 1 is collapse, energy falling at rest with rank steady is
the lag floor draining, and one reading cannot tell those apart. Everything
printed here about the graph comes out of a
:class:`~patchworks.diagnostics.Reading`, both halves of it are printed on every
line, and nothing in this module computes an energy or a rank of its own.

**The expensive half stays off the fast path.**
:meth:`~patchworks.diagnostics.Diagnostics.whole_graph` is a `3764 × 3764`
symmetric eigendecomposition, so it is **off by default** here and opt-in on a
cadence of its own that must be a multiple of the reporting cadence.

**What it all costs, measured.** On the real dome in the supported container
(#131), 1500 ticks, default cadences:

* the per-tick half — the accumulators and the finite check — **15 µs a tick**,
  against a ~10 ms tick;
* one report — the paired reading, the observation sweep, the stalk sweep and
  the line — **4.5 ms**, once every 500 ticks;
* the two together, timed in situ by the run itself, **0.60 % of wall clock**
  (0.59, 0.72, 0.60 over three runs), against the ~1–2 % #92 measured for live
  capture and the 10 % bound;
* building the instrument, **~2 s**, once, before the first tick.

Turning the whole-graph reading on is what changes that picture and is why it is
off: at `--whole-graph-every 1000` over 2000 ticks the same run reported
**35.4 %**, two readings at ~5.4 s each.

None of those is a number this docstring asks to be believed. :class:`Progress`
times its own work and :func:`format_summary` prints it as a share of the run's
wall clock, so every run states its own overhead on its own machine — the figures
above are simply what that said here.

**Interruption is clean.** Ctrl-C sets a flag, the loop finishes the tick it is
on, and the final report prints — see :func:`stopping_on_interrupt`. A second
Ctrl-C is handed back to Python, so an unresponsive run is still killable.

**Nothing here is part of the architecture** (#77, and `CONTEXT.md`'s *Demo
surface*). No cell reads anything this computes, every seed is passed
explicitly, nothing draws from a global RNG, and running with reporting off
produces a bit-identical trajectory to running with it on — asserted in
`tests/test_progress.py` rather than intended, because that is the one property
that makes the reporting free to leave on.

Checkpointing, resuming and writing trained artifacts are deliberately absent.
That is where this leads, and there is nothing to save yet.
"""

from __future__ import annotations

import signal
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TextIO

import numpy as np
import torch

from patchworks.agent import Agent, TickOutcome, run
from patchworks.diagnostics import Condition, Diagnostics, Reading, WholeGraphReading
from patchworks.graph import Dome
from patchworks.tick import Sheaf

#: How often a report line is printed, in ticks.
#:
#: 500 ticks is 10 seconds of world at the sandbox's 50 Hz control rate, and on
#: the real dome rather more than that in wall clock — about the longest a human
#: should have to wait to see that a run they walked away from is still alive.
#: It is *not* #91's :data:`~patchworks.diagnostics.DEFAULT_EVERY` of 10: that
#: cadence is set by how finely the fall in energy is worth resolving into a
#: record, and this one by how often a line is worth putting on someone's
#: terminal for an hour.
DEFAULT_REPORT_EVERY = 500

#: `--whole-graph-every`'s default: never.
#:
#: The expensive half of the instrument, off unless asked for, for the reason
#: this module's docstring gives. Turned on it must be a multiple of the
#: reporting cadence, which is #91's own multiple-of rule — every whole-graph
#: reading then lands on a tick that also carries the paired per-edge reading of
#: the same configuration.
WHOLE_GRAPH_OFF = 0


# ---------------------------------------------------------------------------
# what one report holds
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Report:
    """One cadence report's numbers, before anything is formatted.

    Numbers rather than a line of text, so that a test can ask whether the arm
    moved without parsing a column out of a string, and so that a caller driving
    :class:`Progress` itself can do something other than print.
    """

    tick: int
    """The sheaf's tick counter when this was taken. The cadence is counted on
    it rather than on calls, for the reason
    :meth:`~patchworks.diagnostics.Diagnostics.observe` gives."""

    since: int
    """Ticks since the previous report — the window every *since the last
    report* number below is over."""

    seconds: float
    """Wall seconds the window took."""

    rate: float
    """Ticks per wall second over the window. Not the control rate: this is how
    fast the machine is going, and the control rate is a property of the world."""

    elapsed: float
    """Wall seconds since the run began."""

    energy_mean: float
    """Mean per-edge Dirichlet energy, from #91's paired reading. This is
    **squared** disagreement — one edge's own term of `xᵀLx` — so it is not on
    the scale of the norms :meth:`~patchworks.tick.Sheaf.disagreement` reports."""

    energy_max: float
    """The worst edge's energy. Printed beside the mean because a mean that has
    fallen while one edge holds all the error is a different run from one where
    the fall is everywhere."""

    disagreeing_edges: int
    """Edges whose energy is not exactly zero. #120's signature quotes `682 of
    682`; this is that count, and :attr:`edges` is that denominator."""

    edges: int
    """How many edges the reading covered."""

    effective_rank_mean: float
    """Mean per-edge effective rank, over both ends of every edge — the other
    half of #91's pair, and never printed without the energy above it. Sliding
    toward 1 while the energy falls is collapse."""

    effective_rank_min: float
    """The most concentrated map in the population. One end collapsing while the
    fleet's mean holds is exactly the case the pair exists to catch, which is why
    the minimum is printed and not only the mean."""

    commanded_mean_abs: float
    """Mean `|torque|` the graph asked for over the window, **before** the arm's
    limits saw it. Pre-clip, so it can exceed the action space's 1."""

    applied_mean_abs: float
    """Mean `|torque|` the arm applied — post-clip, so 1 is saturation. A gap
    between this and the figure above is the graph asking for more than the arm
    will give, which is half of #120's fixed point."""

    commanded_sd: tuple[float, ...]
    """Per-joint standard deviation of the commanded torque across the window.
    **The column #120 turns on**: a command constant to one part in 10⁵ reads
    here as `7e-06`, and no other number in the report says it."""

    arm_travel: tuple[float, ...]
    """Per joint, radians travelled since the last report — the sum of `|Δq|`
    across the window's ticks, cumulative rather than net. The other column #120
    turns on, where it is exactly zero."""

    non_finite: tuple[str, ...]
    """Where anything non-finite has been seen so far, named rather than
    counted. Empty is the good case."""

    whole_graph: WholeGraphReading | None = None
    """#91's expensive half, on the reports its own cadence lands on and `None`
    on the rest — which is all of them unless it was asked for."""

    @property
    def travel(self) -> float:
        """Total radians the arm travelled over the window, across all joints."""
        return float(sum(self.arm_travel))

    @property
    def command_spread(self) -> float:
        """The widest joint's command spread — the single number for the column."""
        return max(self.commanded_sd) if self.commanded_sd else 0.0


@dataclass(frozen=True)
class Summary:
    """What the whole run did, for the exit code and the closing report."""

    ticks: int
    """Ticks actually run, which is fewer than asked for if it was interrupted."""

    asked_for: int | None
    """Ticks asked for, or `None` for a run that was going to go until stopped."""

    seed: int
    split: str
    control_hz: float
    elapsed: float

    interrupted: bool
    """Whether the run stopped because someone pressed Ctrl-C. **Not a failure**:
    the exit code hangs off :attr:`non_finite`, not off this."""

    reporting_seconds: float
    """Wall seconds spent inside the reporting, measured rather than estimated.
    Zero on a run with reporting off."""

    setup_seconds: float
    """Wall seconds spent **building** the instrument, before the first tick —
    the topology-only `H¹` decomposition :class:`Progress` pays for eagerly.

    Stated separately rather than folded into :attr:`reporting_seconds` or left
    out. It is not part of the per-tick cost and pooling it with one would
    misreport both; leaving it out would let a run with a four-second startup
    claim its reporting was free."""

    reports: tuple[Report, ...]

    non_finite: tuple[str, ...]
    """Everywhere anything non-finite was seen, pooled across the run."""


# ---------------------------------------------------------------------------
# taking the readings
# ---------------------------------------------------------------------------


class Progress:
    """The reporting cadence, its accumulators, and the clock on its own cost.

    Built on a :class:`~patchworks.tick.Sheaf` and fed
    :class:`~patchworks.agent.TickOutcome`\\ s, so it holds no world and no
    agent — the shape :class:`~patchworks.diagnostics.Diagnostics` takes, for the
    same reason. Nothing a cell's computation is handed can reach one.

    Constructing it constructs a :class:`~patchworks.diagnostics.Diagnostics`,
    and that **costs a decomposition** — about 4 seconds and a transient
    `3764 × 4800` matrix on the real dome — because the topology-only `H¹`
    baseline is computed eagerly there. That is a one-off at the start of a run
    measured in hours, and it is named here so nobody looks for it in the
    per-tick path.
    """

    def __init__(
        self,
        sheaf: Sheaf,
        *,
        every: int = DEFAULT_REPORT_EVERY,
        whole_graph_every: int = WHOLE_GRAPH_OFF,
        out: TextIO | None = None,
        generator: torch.Generator | None = None,
    ) -> None:
        if isinstance(every, bool) or not isinstance(every, int) or every < 1:
            raise ValueError(f"every is a cadence in ticks, >= 1; got {every!r}")
        if (
            isinstance(whole_graph_every, bool)
            or not isinstance(whole_graph_every, int)
            or whole_graph_every < 0
        ):
            raise ValueError(
                "whole_graph_every is a cadence in ticks, >= 0 with 0 meaning "
                f"never; got {whole_graph_every!r}"
            )
        if whole_graph_every and whole_graph_every % every:
            raise ValueError(
                f"whole_graph_every ({whole_graph_every}) must be a multiple of "
                f"every ({every}), so that every whole-graph reading lands on a "
                "report that also carries its paired per-edge reading of the "
                "same configuration. See patchworks.diagnostics, Two cadences."
            )
        self.every = every
        self.whole_graph_every = whole_graph_every
        self.out = out

        # `every` is handed to the instrument as well as kept here, so the two
        # cadences agree in its own record; `whole_graph_every` is given a valid
        # stand-in when the expensive half is off, because `read` below is
        # always told explicitly which reading to take and the instrument's
        # cadence therefore decides nothing for this module.
        self.diagnostics = Diagnostics(
            sheaf,
            every=every,
            whole_graph_every=whole_graph_every or every,
            generator=generator,
        )

        #: Every report taken so far, in order.
        self.reports: list[Report] = []
        #: Wall seconds spent reporting. What :func:`format_summary` states.
        self.reporting_seconds = 0.0

        self._started = time.perf_counter()
        self._window_started = self._started
        self._window_tick = sheaf.ticks

        self._count = 0
        # All four are per-joint sums over the window, allocated on the first
        # tick when the arm's width is known and reduced only at report time.
        self._commanded_abs: np.ndarray | None = None
        self._applied_abs: np.ndarray | None = None
        self._command_sum: np.ndarray | None = None
        self._command_square_sum: np.ndarray | None = None
        self._travel: np.ndarray | None = None
        # Kept *across* windows, so the step from the last tick of one window to
        # the first of the next is counted rather than dropped. Only the
        # accumulator resets.
        self._previous_arm: np.ndarray | None = None
        self._seen: dict[str, int] = {}
        self._first_seen: dict[str, int] = {}

    def __repr__(self) -> str:
        return (
            f"Progress(every={self.every}, "
            f"whole_graph_every={self.whole_graph_every}, "
            f"{len(self.reports)} reports)"
        )

    @property
    def sheaf(self) -> Sheaf:
        return self.diagnostics.sheaf

    # -- the per-tick half, which has to stay cheap -------------------------

    def after(self, outcome: TickOutcome) -> Report | None:
        """Fold one finished tick in, and return a report on the cadence.

        `None` on the ticks the cadence skips, which is all but one in
        :attr:`every`. What happens on every tick is three-vector arithmetic and
        one `isfinite` over six floats; the reading, the observation sweep and
        the stalk sweep all wait for the report.

        **The per-tick non-finite check is of the command and the efference copy
        only**, and that is a complete net rather than a partial one for the
        reason a fuller sweep would be: a non-finite observation or stalk reaches
        the actuator boundary cell on the next tick, so it shows up in this check
        one tick later rather than not at all. The observation and the sheaf's
        stalks are swept at report time as well, so the report names *where*
        rather than only saying that.
        """
        started = time.perf_counter()
        command = outcome.command
        applied = outcome.applied

        # One check for the common case and two only when it fails. Every one of
        # these is a numpy call on a three-vector, where the call costs more than
        # the arithmetic — so the way to keep this path cheap is to make fewer of
        # them, not smaller ones.
        if not (np.isfinite(command).all() and np.isfinite(applied).all()):
            for label, values in (("command", command), ("applied", applied)):
                if not np.all(np.isfinite(values)):
                    self._saw(label)

        if self._command_sum is None or self._command_square_sum is None:
            shape = np.shape(command)
            self._command_sum = np.zeros(shape, dtype=float)
            self._command_square_sum = np.zeros(shape, dtype=float)
            self._commanded_abs = np.zeros(shape, dtype=float)
            self._applied_abs = np.zeros(shape, dtype=float)
        self._command_sum += command
        self._command_square_sum += np.square(command)
        # Summed per joint and averaged at report time rather than reduced to a
        # scalar here: the reduction is the same arithmetic either way, and doing
        # it once per report instead of once per tick is two fewer numpy calls on
        # the path every tick runs down.
        self._commanded_abs += np.abs(command)
        self._applied_abs += np.abs(applied)

        arm = outcome.observation["qpos"]
        if self._travel is None:
            self._travel = np.zeros(np.shape(arm), dtype=float)
        if self._previous_arm is None:
            self._previous_arm = np.array(arm, dtype=float)
        else:
            self._travel += np.abs(arm - self._previous_arm)
            # **Copied, not referenced.** The observation's array belongs to the
            # world, which is free to write the next tick's values into the same
            # buffer; holding a reference to it would compare a tick against
            # itself and report a moving arm as motionless — which is the one
            # reading this whole readout must not produce by accident.
            np.copyto(self._previous_arm, arm)
        self._count += 1

        self.reporting_seconds += time.perf_counter() - started
        if self.sheaf.ticks % self.every:
            return None
        return self.report(outcome)

    # -- the report itself --------------------------------------------------

    def report(self, outcome: TickOutcome) -> Report:
        """Take a report now, whatever the cadence says, and record it.

        Separate from :meth:`after` for the reason
        :meth:`~patchworks.diagnostics.Diagnostics.read` is separate from
        `observe`: the closing report of an interrupted run lands on whatever
        tick the interruption landed on, and an instrument that could only speak
        on the grid would be silent at exactly the moment someone stopped the run
        to look at it.
        """
        started = time.perf_counter()
        tick = self.sheaf.ticks
        reading = self.diagnostics.read(
            Condition.DRIVEN,
            whole_graph=bool(self.whole_graph_every)
            and tick % self.whole_graph_every == 0,
        )
        report = self._assemble(tick, reading, outcome)
        self.reports.append(report)
        self._reset(tick)
        self.reporting_seconds += time.perf_counter() - started
        if self.out is not None:
            print(format_report(report), file=self.out, flush=True)
            if report.whole_graph is not None:
                print(
                    format_whole_graph(report.whole_graph, self.diagnostics),
                    file=self.out,
                    flush=True,
                )
        return report

    def _assemble(self, tick: int, reading: Reading, outcome: TickOutcome) -> Report:
        """Every number in one report, from the window's accumulators and the reading."""
        now = time.perf_counter()
        seconds = now - self._window_started
        count = max(self._count, 1)

        # The report-time half of the non-finite sweep: the world's answer as it
        # was written in, and the graph's own state. Named per key and per place,
        # because a bug report that says where is worth more than one that says
        # whether.
        for key, value in outcome.observation.items():
            array = np.asarray(value)
            if array.dtype.kind == "f" and not np.all(np.isfinite(array)):
                self._saw(f"observation[{key!r}]")
        with torch.no_grad():
            if not bool(torch.isfinite(self.sheaf.stalks).all()):
                self._saw("the sheaf's stalks")
        if not bool(torch.isfinite(reading.edges.energy).all()):
            self._saw("per-edge energy")

        energy = reading.edges.energy
        rank = reading.edges.effective_rank
        if self._command_sum is None or self._command_square_sum is None:
            mean = variance = np.zeros(0)
        else:
            mean = self._command_sum / count
            # Population variance over the window, floored at zero: the
            # sum-of-squares form can go a few ulp negative on a command that
            # never moved, which is precisely the case this column is read for.
            variance = np.maximum(
                self._command_square_sum / count - mean * mean, 0.0
            )
        travel = self._travel if self._travel is not None else np.zeros(0)

        return Report(
            tick=tick,
            since=tick - self._window_tick,
            seconds=seconds,
            rate=self._count / seconds if seconds > 0 else float("inf"),
            elapsed=now - self._started,
            energy_mean=float(energy.mean()),
            energy_max=float(energy.max()),
            disagreeing_edges=int(torch.count_nonzero(energy)),
            edges=len(reading.edges),
            effective_rank_mean=float(rank.mean()),
            effective_rank_min=float(rank.min()),
            commanded_mean_abs=float(self._commanded_abs.mean()) / count
            if self._commanded_abs is not None
            else 0.0,
            applied_mean_abs=float(self._applied_abs.mean()) / count
            if self._applied_abs is not None
            else 0.0,
            commanded_sd=tuple(float(value) for value in np.sqrt(variance)),
            arm_travel=tuple(float(value) for value in travel),
            non_finite=self.non_finite,
            whole_graph=reading.whole_graph,
        )

    def _reset(self, tick: int) -> None:
        """Start a fresh window. `_previous_arm` survives; see its note."""
        self._window_started = time.perf_counter()
        self._window_tick = tick
        self._count = 0
        for accumulator in (
            self._command_sum,
            self._command_square_sum,
            self._commanded_abs,
            self._applied_abs,
            self._travel,
        ):
            if accumulator is not None:
                accumulator[:] = 0.0

    def _saw(self, label: str) -> None:
        if label not in self._first_seen:
            self._first_seen[label] = self.sheaf.ticks
        self._seen[label] = self._seen.get(label, 0) + 1

    @property
    def non_finite(self) -> tuple[str, ...]:
        """Everywhere anything non-finite has been seen, with when and how often."""
        return tuple(
            f"{label}, first at tick {self._first_seen[label]}, seen {count}x"
            for label, count in self._seen.items()
        )


# ---------------------------------------------------------------------------
# the loop, and stopping it
# ---------------------------------------------------------------------------


@contextmanager
def stopping_on_interrupt(
    out: TextIO | None = None,
) -> Iterator[Callable[[], bool]]:
    """Ctrl-C asks the loop to stop; it does not tear it down.

    Yields a predicate the loop checks between ticks. The first Ctrl-C sets it
    and prints a line saying what will happen — the tick in flight finishes and
    the final report prints — rather than raising a `KeyboardInterrupt` through
    whatever the run happened to be inside. That is the ticket's requirement, and
    it also means a run is never stopped halfway through a tick, in a world whose
    physics state would then not correspond to any tick the report described.

    **A second Ctrl-C is handed straight back to Python.** The handler restores
    what it replaced and raises, so a run that has stopped responding is still
    killable by the gesture everybody already knows — an interruption guard that
    can itself hang is worse than none.

    Signal handlers can only be installed from the main thread, so off it this
    installs nothing and yields a predicate that is always false; a caller there
    sees the ordinary `KeyboardInterrupt`. Whatever was installed before is
    restored on the way out, including the default — nothing here leaves a
    process's interrupt handling changed behind it.
    """
    asked = False

    def stop() -> bool:
        return asked

    def handler(signum: int, frame: object) -> None:
        nonlocal asked
        if asked:
            signal.signal(signal.SIGINT, previous)
            raise KeyboardInterrupt
        asked = True
        print(
            "\n  Ctrl-C: stopping at the end of this tick, then the final "
            "report. Ctrl-C again to stop now.",
            file=out if out is not None else sys.stdout,
            flush=True,
        )

    try:
        previous = signal.signal(signal.SIGINT, handler)
    except ValueError:
        # Not the main thread. Nothing was installed, so nothing is restored.
        yield lambda: False
        return
    try:
        yield stop
    finally:
        signal.signal(signal.SIGINT, previous)


def ticking(
    agent: Agent,
    ticks: int | None,
    *,
    seed: int | None = None,
    progress: Progress | None = None,
    stopping: Callable[[], bool] | None = None,
) -> Iterator[TickOutcome]:
    """Every tick of a headless run, yielded, with the reporting folded in.

    `ticks` of `None` is a run that goes until something stops it, which is what
    *leave it running and come back* means. It is spelled as `sys.maxsize` ticks
    through :func:`~patchworks.agent.run` rather than as a second loop, so that
    the world is arranged by exactly the code every other run is arranged by;
    `sys.maxsize` ticks at the sandbox's 50 Hz is some `10¹¹` years of world.

    `progress` of `None` is reporting off, and then nothing in this function
    reads the sheaf, allocates, or takes a clock reading between ticks — which is
    what makes the bit-identical claim in this module's docstring a property of
    the code rather than a hope.

    Yields, for the reason :func:`~patchworks.agent.run` does: a caller may want
    to watch a run, disturb it, or stop it. :func:`drive` is the caller that
    only wants it to happen.
    """
    for outcome in run(agent, sys.maxsize if ticks is None else ticks, seed=seed):
        if progress is not None:
            progress.after(outcome)
        yield outcome
        if stopping is not None and stopping():
            return


def drive(
    *,
    ticks: int | None = None,
    seed: int = 0,
    split: str = "train",
    report_every: int = DEFAULT_REPORT_EVERY,
    whole_graph_every: int = WHOLE_GRAPH_OFF,
    reporting: bool = True,
    dome: Dome | None = None,
    image_size: int | None = None,
    out: TextIO | None = None,
) -> Summary:
    """Build the world and the agent, run headless, report on the cadence.

    **Every draw is seeded explicitly and none of them is global**, the same way
    :func:`patchworks.cli.measure_liveness` does it and for the same reason: the
    sheaf's parameters come from a `torch.Generator` built here and the world's
    layout from `reset(seed=...)`, so a run changes no trajectory of anything run
    after it in the same process (#77).

    `reporting=False` runs the same loop with no :class:`Progress` at all — not a
    silenced one — which is the arm of the comparison
    `tests/test_progress.py` holds the bit-identical trajectory against.

    `dome` and `image_size` exist so the suite can run this on the small dome
    against a 16×16 render rather than the full one at 64×64; the CLI passes
    neither and gets the defaults.

    The imports are inside the function, as `measure_liveness` does it: what it
    defers is :mod:`patchworks.sandbox`, and MuJoCo with it, so `patchworks
    --help` never loads the world.
    """
    from patchworks.graph import build_graph
    from patchworks.sandbox import PlanarPushSandbox

    stream = sys.stdout if out is None else out
    if dome is None:
        dome = build_graph()
    world = PlanarPushSandbox(
        split=split, **({} if image_size is None else {"image_size": image_size})
    )
    try:
        agent = Agent(world, dome=dome, generator=torch.Generator().manual_seed(seed))
        inner = world.unwrapped
        control_hz = 1.0 / (inner.model.opt.timestep * inner.frame_skip)
        building = time.perf_counter()
        progress = (
            Progress(
                agent.sheaf,
                every=report_every,
                whole_graph_every=whole_graph_every,
                out=stream,
            )
            if reporting
            else None
        )
        setup_seconds = time.perf_counter() - building if reporting else 0.0
        if progress is not None:
            print(
                format_preamble(
                    ticks=ticks,
                    seed=seed,
                    split=split,
                    control_hz=control_hz,
                    progress=progress,
                ),
                file=stream,
                flush=True,
            )

        started = time.perf_counter()
        ran = 0
        last: TickOutcome | None = None
        with stopping_on_interrupt(stream) as stopping:
            for outcome in ticking(
                agent, ticks, seed=seed, progress=progress, stopping=stopping
            ):
                ran += 1
                last = outcome
            interrupted = stopping()
        elapsed = time.perf_counter() - started

        # The closing report lands wherever the run stopped, which is why
        # `Progress.report` exists beside the cadence. Skipped when the cadence
        # already reported this exact tick, so an interrupted run does not
        # print the same tick twice, and when nothing ran at all.
        if progress is not None and last is not None:
            if not progress.reports or progress.reports[-1].tick != agent.sheaf.ticks:
                progress.report(last)

        summary = Summary(
            ticks=ran,
            asked_for=ticks,
            seed=seed,
            split=split,
            control_hz=control_hz,
            elapsed=elapsed,
            interrupted=interrupted,
            reporting_seconds=progress.reporting_seconds if progress else 0.0,
            setup_seconds=setup_seconds,
            reports=tuple(progress.reports) if progress else (),
            non_finite=progress.non_finite if progress else (),
        )
        if progress is not None:
            print(format_summary(summary), file=stream, flush=True)
        return summary
    finally:
        world.close()


# ---------------------------------------------------------------------------
# the readout
# ---------------------------------------------------------------------------

#: The report table's columns: heading, width, and how the value is rendered.
#:
#: One table rather than a header string beside a format string, because the two
#: drift the moment a column is added and a readout whose headings have slipped a
#: column is worse than one with no headings at all.
_COLUMNS: tuple[tuple[str, int, Callable[[Report], str]], ...] = (
    # Wide enough for a tick count no run will reach: at the sandbox's 50 Hz a
    # twelve-digit tick is six hundred years of world. A value wider than its
    # column pushes every column after it along, which is the one way a table
    # laid out from a single spec can still come out misaligned.
    ("tick", 12, lambda r: f"{r.tick:d}"),
    ("t/s", 6, lambda r: f"{r.rate:.1f}"),
    ("energy mean", 11, lambda r: f"{r.energy_mean:.4g}"),
    ("energy max", 10, lambda r: f"{r.energy_max:.4g}"),
    ("rank mean", 9, lambda r: f"{r.effective_rank_mean:.3f}"),
    ("rank min", 8, lambda r: f"{r.effective_rank_min:.3f}"),
    ("edges", 9, lambda r: f"{r.disagreeing_edges}/{r.edges}"),
    ("torque", 7, lambda r: f"{r.commanded_mean_abs:.3f}"),
    ("applied", 7, lambda r: f"{r.applied_mean_abs:.3f}"),
    ("cmd sd", 9, lambda r: f"{r.command_spread:.1e}"),
    ("travel", 8, lambda r: f"{r.travel:.3f}"),
    ("non-finite", 10, lambda r: "SEE BELOW" if r.non_finite else "-"),
)

LEGEND = """\
  tick        the sheaf's tick counter, and t/s the wall-clock rate over the window
  energy      per-edge Dirichlet energy: squared disagreement, from the paired
              instrument (patchworks.diagnostics). edges = how many of them are
              not exactly zero
  rank        per-edge effective rank, the other half of that pair, and never read
              without it: energy falling with rank sliding toward 1 is collapse,
              energy falling with rank steady is the lag floor draining
  torque      mean |torque| asked for (pre-clip) and applied (post-clip, 1 =
              saturated); cmd sd is the widest joint's spread across the window
  travel      radians the arm moved since the last line, summed over joints and
              cumulative rather than net\
"""


def header() -> str:
    """The column headings, aligned with :func:`format_report` by construction."""
    return " ".join(f"{name:>{width}}" for name, width, _ in _COLUMNS)


def format_report(report: Report) -> str:
    """One report, as one line of the table."""
    return " ".join(
        f"{render(report):>{width}}" for _name, width, render in _COLUMNS
    )


def format_preamble(
    *,
    ticks: int | None,
    seed: int,
    split: str,
    control_hz: float,
    progress: Progress,
) -> str:
    """What is printed once, above the table: what is running and what the columns are."""
    from patchworks.cli import _environment_line

    how_many = "until interrupted" if ticks is None else f"{ticks} ticks"
    world = (
        ""
        if ticks is None
        else f" = {ticks / control_hz:.0f} s of world at {control_hz:.0f} Hz"
    )
    whole = (
        "off"
        if not progress.whole_graph_every
        else f"every {progress.whole_graph_every} ticks"
    )
    lines = [
        f"patchworks run -- {how_many}{world}, headless, seed {seed}, split {split}",
        _environment_line(),
        "",
        f"reporting every {progress.every} ticks; whole-graph readings {whole}; "
        "Ctrl-C stops cleanly",
        "",
        LEGEND,
        "",
        header(),
    ]
    return "\n".join(lines)


def format_whole_graph(reading: WholeGraphReading, diagnostics: Diagnostics) -> str:
    """The expensive half, indented under the line it belongs to.

    Against the baselines rather than against zero, which is #91's own reading of
    them: `dim H⁰` against the construction bound, `dim H¹` against the
    topology-only baseline a lattice's cycles guarantee whatever the maps do.
    """
    bound = int(diagnostics.sheaf.dome.private_dimensions.sum())
    excess = reading.dim_h1 - diagnostics.h1_baseline
    return (
        f"    whole graph: dim H^0 = {reading.dim_h0} (construction bound "
        f"{bound}), dim H^1 = {reading.dim_h1} (topology-only baseline "
        f"{diagnostics.h1_baseline}, {excess:+d} from the maps), minimum "
        f"achievable energy {reading.minimum_energy:.6g}"
    )


def stationary_reports(reports: tuple[Report, ...] | list[Report]) -> tuple[Report, ...]:
    """The run of reports at the end whose arm travel was **exactly** zero.

    Exactly zero, and no tolerance beside it. That is what #120 measured — a
    joint resting on its stop returns a bit-identical `qpos` tick after tick, so
    the differences are zeros rather than small numbers — and a threshold here
    would be a number invented to make a verdict come out, which is the one thing
    a readout like this must not contain. An arm jittering at `1e-9` is not this
    case and is not reported as it.
    """
    trailing: list[Report] = []
    for report in reversed(list(reports)):
        if report.travel != 0.0:
            break
        trailing.append(report)
    return tuple(reversed(trailing))


def format_summary(summary: Summary) -> str:
    """The closing report: what moved across the run, what it cost, and what stopped it.

    The paired instrument is quoted first and last **as a pair**, because the
    ticket's whole point is that neither half is a verdict alone. The overhead
    line is this run's measured figure. The stationary paragraph appears only
    when the run's own numbers put it there.
    """
    reports = summary.reports
    stopped = "interrupted" if summary.interrupted else "finished"
    lines = [
        "",
        f"patchworks run -- {stopped} after {summary.ticks} ticks "
        f"= {summary.ticks / summary.control_hz:.0f} s of world, "
        f"seed {summary.seed}, split {summary.split}",
        f"wall clock           : {summary.elapsed:.1f} s"
        f"   ({summary.ticks / summary.elapsed:.1f} ticks/s)"
        if summary.elapsed > 0
        else f"wall clock           : {summary.elapsed:.1f} s",
    ]
    share = (
        100.0 * summary.reporting_seconds / summary.elapsed
        if summary.elapsed > 0
        else 0.0
    )
    lines.append(
        f"reporting cost       : {summary.reporting_seconds:.2f} s, "
        f"{share:.2f}% of the run   (measured here, not estimated)"
    )
    lines.append(
        f"  + building it      : {summary.setup_seconds:.2f} s, once, before the "
        "first tick   (the topology-only H^1 decomposition)"
    )

    if not reports:
        lines.append("")
        lines.append("No reports were taken, so nothing above was measured.")
        return "\n".join(lines)

    first, last = reports[0], reports[-1]
    lines += [
        "",
        f"the paired instrument, driven, ticks {first.tick} -> {last.tick}",
        f"  per-edge energy, mean      : {first.energy_mean:.6g} -> "
        f"{last.energy_mean:.6g}",
        f"  per-edge energy, max       : {first.energy_max:.6g} -> "
        f"{last.energy_max:.6g}",
        f"  effective rank, mean       : {first.effective_rank_mean:.6g} -> "
        f"{last.effective_rank_mean:.6g}",
        f"  effective rank, min        : {first.effective_rank_min:.6g} -> "
        f"{last.effective_rank_min:.6g}",
        f"  edges not exactly agreeing : {first.disagreeing_edges}/{first.edges}"
        f" -> {last.disagreeing_edges}/{last.edges}",
        "  Read the two together: energy falling with the rank sliding toward 1",
        "  is collapse; energy falling with the rank steady is not.",
    ]
    if last.whole_graph is not None:
        lines.append(
            f"  minimum achievable energy  : {last.whole_graph.minimum_energy:.6g}"
            f"   (measured total {last.energy_mean * last.edges:.6g})"
        )

    travel = ", ".join(f"{value:.2f}" for value in last.arm_travel)
    spread = ", ".join(f"{value:.2e}" for value in last.commanded_sd)
    lines += [
        "",
        f"the last {last.since} ticks",
        f"  arm travelled (rad)  : [{travel}]   cumulative, per joint",
        f"  command sd per joint : [{spread}]   0 = a frozen command",
        f"  torque |mean|        : {last.commanded_mean_abs:.3f} asked, "
        f"{last.applied_mean_abs:.3f} applied",
    ]

    if summary.non_finite:
        lines += [
            "",
            f"NON-FINITE: {', '.join(summary.non_finite)}",
            "",
            "Something went non-finite. That is a bug, not an untrained agent:",
            "send this whole output.",
        ]
        return "\n".join(lines)

    lines.append("  non-finite           : none")

    stationary = stationary_reports(reports)
    if stationary:
        span = sum(report.since for report in stationary)
        worst = max(report.command_spread for report in stationary)
        energy = stationary[-1].energy_mean
        lines += [
            "",
            f"STUCK: the last {len(stationary)} report(s), {span} ticks, moved the "
            f"arm exactly 0 radians,",
            f"held the command constant to {worst:.1e}, and left "
            f"{stationary[-1].disagreeing_edges} of {stationary[-1].edges} edges "
            f"disagreeing (mean energy {energy:.4g}).",
            "A constant command against a saturated arm with the error signal "
            "still large is",
            "#120's locked loop: the run is alive and it is not going anywhere.",
        ]
    return "\n".join(lines)
