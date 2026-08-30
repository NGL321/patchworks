"""What `patchworks run`'s readout costs, and what it shows on #120 (ticket #121).

#121 asks for two things this file is the provenance of. **The reporting
overhead is measured and stated** — the figures in :mod:`patchworks.progress`'
docstring came from `cost` below. And **the readout is checked against #120's
locked-loop signature** — `locked` below is that check, run on the real dome
rather than described, because a readout that would not have made #120 obvious
is the one thing the ticket says is not worth printing.

Like every other script here it **asserts nothing**. `cost` is a wall-clock
measurement and belongs on a machine rather than in a suite, and `locked` shows
what #120's case looks like through the readout — which is a thing for a human
to read, not a number to pin. What stands in the suite is one smoke test that
this file still runs (`tests/test_progress.py::TestBenchmark`), on the same
footing as `benchmarks/agent_tick.py` and `benchmarks/untrained_fixed_point.py`.

    python benchmarks/run_reporting.py cost
    python benchmarks/run_reporting.py locked

**`cost`** times the two halves separately and then in situ. The two halves are
timed on their own because they are charged differently: the per-tick half is
paid every tick and the report is paid once a cadence, so one number covering
both would hide which of them a change had made worse. The in-situ figure is the
run's own — :class:`~patchworks.progress.Progress` times itself and the closing
report prints the share — and it is the one to quote, because the micro-timings
run hot in a loop while the real thing runs once every 10 ms against a cold
cache.

It also runs the same seed with the reporting **off**, which is the arm the
"reporting is free" claim would be falsified by. Do not expect that comparison
to resolve anything on its own: run-to-run variance here is around 10%, an order
of magnitude larger than the effect, and three paired runs was not enough to see
through it. The in-situ self-timing is what carries the claim; this is only a
check that the two do not disagree wildly.

**`locked`** drives the untrained agent on the full dome at seed 0 for long
enough to reach the fixed point #120 characterised, and prints the readout. What
to look for is in the ticket: travel falling to exactly zero, the command's
spread collapsing, and 682 of 682 edges still disagreeing while it does — the
combination that says the loop has locked with the disagreement still large,
rather than settled.

Run it on an otherwise idle machine, for the reason `benchmarks/agent_tick.py`
gives: the tick spreads over every torch thread and the reporting does not, so
under load the ratio between them measures the machine.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time

import numpy as np
import torch

from patchworks.agent import TickOutcome
from patchworks.graph import DomeSpec, build_graph
from patchworks.progress import DEFAULT_REPORT_EVERY, Progress, drive
from patchworks.tick import Sheaf

#: Long enough to reach #120's fixed point. The ticket measured it settled well
#: before 1500 ticks at seed 0, and the last few reports are the ones that show
#: it, so the default leaves several past the point of arrival.
LOCKED_TICKS = 2000

#: Reports over that run. 250 rather than the CLI's 500 so that the arrival is
#: resolved into more than a couple of lines.
LOCKED_EVERY = 250

COST_TICKS = 1500
TRIALS = 3


class Sink:
    """Somewhere for the readout to go when the subject is the clock, not the text."""

    def write(self, text: str) -> int:
        return len(text)

    def flush(self) -> None:
        pass


def _a_sheaf(spec: DomeSpec | None = None) -> Sheaf:
    """A sheaf with content in its stalks, for the micro-timings.

    The same stand-in the suite uses: a fresh sheaf's stalks are all zero, every
    edge agrees exactly, and timing the instrument on that would time it on a
    configuration no run is ever in.
    """
    dome = build_graph() if spec is None else build_graph(spec)
    sheaf = Sheaf(dome, generator=torch.Generator().manual_seed(0))
    torch.manual_seed(1)
    sheaf.stalks = torch.randn_like(sheaf.stalks)
    sheaf.stalks[sheaf.layout.pad] = 0.0
    return sheaf


def _an_outcome(width: int = 3, image: int = 64) -> TickOutcome:
    """One tick's worth of what the readout reads, at the real dome's sizes."""
    command = np.zeros(width, dtype=np.float32)
    return TickOutcome(
        command=command,
        applied=command,
        observation={
            "qpos": np.zeros(width),
            "image": np.zeros((image, image, 3), dtype=np.float32),
        },
        info={},
    )


def halves(sheaf: Sheaf, *, ticks: int = 5000) -> tuple[float, float]:
    """The per-tick half and one report, timed apart.

    The per-tick half is timed with the sheaf's counter parked on a tick the
    cadence does **not** land on. Tick 0 is a multiple of everything, so a naive
    loop on a fresh sheaf reports on every call and times the wrong thing — by a
    factor of two hundred, which is how this was found.
    """
    outcome = _an_outcome()
    # Reporting to a sink rather than to `None`, so that formatting the line is
    # inside the report's figure. It is a real part of what a report costs and
    # the only part that grows when the whole-graph line is turned on; timing a
    # report that never renders would understate it against the in-situ number
    # below, which does render.
    progress = Progress(sheaf, every=10**9, out=Sink())

    sheaf.ticks = 7
    for _ in range(min(500, ticks)):
        progress.after(outcome)
    started = time.perf_counter()
    for _ in range(ticks):
        progress.after(outcome)
    per_tick = (time.perf_counter() - started) / ticks

    sheaf.ticks = 10**9
    started = time.perf_counter()
    progress.report(outcome)
    one_report = time.perf_counter() - started
    return per_tick, one_report


def cost(
    ticks: int = COST_TICKS,
    trials: int = TRIALS,
    *,
    spec: DomeSpec | None = None,
    image_size: int | None = None,
) -> None:
    """Both halves, then the whole thing in situ, then the same run unreported.

    `spec` and `image_size` exist so the suite can ask both measurements to
    *complete* on the small dome; the figures on record are the defaults', on the
    real one.
    """
    per_tick, one_report = halves(_a_sheaf(spec))
    print(
        f"per-tick half   : {per_tick * 1e6:7.1f} us a tick"
        f"   (paid every tick)"
    )
    print(
        f"one report      : {one_report * 1e3:7.2f} ms"
        f"        (paid once every {DEFAULT_REPORT_EVERY} ticks by default)"
    )
    amortised = per_tick + one_report / DEFAULT_REPORT_EVERY
    print(
        f"the two together: {amortised * 1e6:7.1f} us a tick amortised at that "
        "cadence"
    )
    print()

    on, off, shares, setup = [], [], [], []
    for trial in range(trials):
        # Alternating, because whichever runs first is measurably slower and
        # three trials is not enough to average that out -- taking them always
        # in the same order once reported a 5.6% difference that reversed when
        # the order did.
        first_off = trial % 2 == 0
        world = {"dome": None if spec is None else build_graph(spec),
                 "image_size": image_size}
        if first_off:
            quiet = drive(ticks=ticks, seed=0, reporting=False, out=Sink(), **world)
        reported = drive(ticks=ticks, seed=0, out=Sink(), **world)
        if not first_off:
            quiet = drive(ticks=ticks, seed=0, reporting=False, out=Sink(), **world)
        on.append(reported.elapsed)
        off.append(quiet.elapsed)
        setup.append(reported.setup_seconds)
        shares.append(100 * reported.reporting_seconds / reported.elapsed)
        print(
            f"  trial {trial}: reported {reported.elapsed:6.2f} s, "
            f"unreported {quiet.elapsed:6.2f} s, "
            f"self-timed {shares[-1]:.2f}%, setup {setup[-1]:.2f} s"
        )
    print()
    print(
        f"in situ, self-timed : {statistics.median(shares):.2f}% of wall clock "
        f"(median of {trials})"
    )
    print(
        f"building it         : {statistics.median(setup):.2f} s, once, before "
        "the first tick"
    )
    difference = 100 * (statistics.median(on) - statistics.median(off))
    print(
        f"wall clock on vs off: {difference / statistics.median(off):+.2f}% "
        "-- noise at this sample size; see this file's docstring"
    )


def locked(
    ticks: int = LOCKED_TICKS,
    every: int = LOCKED_EVERY,
    seed: int = 0,
    *,
    spec: DomeSpec | None = None,
    image_size: int | None = None,
) -> None:
    """#120's case, through the readout, on the real dome.

    `spec` and `image_size` are the suite's, as in :func:`cost`. #120's fixed
    point is the *full* dome's at seed 0, so a small-dome run of this is a check
    that the code runs and not a sighting of the thing.
    """
    summary = drive(
        ticks=ticks,
        seed=seed,
        report_every=every,
        dome=None if spec is None else build_graph(spec),
        image_size=image_size,
    )
    settled = summary.reports[-1] if summary.reports else None
    print()
    if settled is None:
        print("no reports were taken; nothing to read")
        return
    print(
        "what #120 measured, beside what the readout above shows at "
        f"tick {settled.tick}:"
    )
    for label, shown, recorded in (
        (f"travel over the last {settled.since} ticks",
         f"{settled.travel:.3g} rad", "exactly 0"),
        ("widest command spread",
         f"{settled.command_spread:.1e}", "constant to one part in 10^5"),
        ("edges disagreeing",
         f"{settled.disagreeing_edges} of {settled.edges}", "682 of 682"),
        ("torque applied",
         f"{settled.applied_mean_abs:.3f}", "|torque| ~ 0.37"),
    ):
        print(f"  {label:<34} {shown:>16}   (#120: {recorded})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="measurement", required=True)

    costing = sub.add_parser("cost", help="what the reporting costs")
    costing.add_argument("--ticks", type=int, default=COST_TICKS)
    costing.add_argument("--trials", type=int, default=TRIALS)

    showing = sub.add_parser("locked", help="the readout on #120's locked loop")
    showing.add_argument("--ticks", type=int, default=LOCKED_TICKS)
    showing.add_argument("--report-every", type=int, default=LOCKED_EVERY)
    showing.add_argument("--seed", type=int, default=0)

    arguments = parser.parse_args(argv)
    if arguments.measurement == "cost":
        cost(ticks=arguments.ticks, trials=arguments.trials)
    else:
        locked(
            ticks=arguments.ticks, every=arguments.report_every, seed=arguments.seed
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
