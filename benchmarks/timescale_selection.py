"""The bias-selection construction run: reachability, the fold margin, the go/no-go.

The falsification condition for
`docs/adr/0005-timescale-is-persistence-not-a-schedule.md`, run before anything
is trained (`docs/spec/05-timescales.md`, *A cheap go/no-go before anything is
built*). It draws candidate bias vectors for the shared frozen body, measures the
timescale each produces over a **driven trajectory**, and reports acceptance rate
per band of the target range, region dwell, realised contraction `lambda`, and
`02-tick-semantics.md`'s `gamma x floor <` fold margin check across the taper.

    .venv/bin/python benchmarks/timescale_selection.py

**The target range is an input, not a constant.** `05-timescales.md` fixes the
derivation and deliberately not the number: the fastest band must resolve the
fastest perturbation the acceptance demo applies, and the slowest must outlast
the longest one. The two readings of that the record supports are both run here,
because they differ by two orders of magnitude and the verdict should be seen to
survive both:

* **onset** — the demo's own tick-valued ladder. The arm nudge's somatomotor
  reflex loop is **3 ticks** and a drive asserted at the apex is seven levels out
  and back, so **~14** (`06-graph-topology.md`, `08-the-acceptance-demo.md`).
* **duration** — how long each perturbation is *in force*. The arm nudge is the
  same 3 ticks; the retarget holds until the task is done, and the only measured
  task duration in the record is `benchmarks/achievability.py`'s **~15 s of sim**
  at 50 Hz, so **~750 ticks**.

The sweep over the drive stand-in is printed with them. Nothing is trained, so
the node stalk sequence a cell is driven with is plausible rather than real, and
the go/no-go is a shape check: it establishes that the mechanism is available. It
does not produce the body's number.
"""

import time

import torch

from patchworks.body import CellBody
from patchworks.graph import build_graph
from patchworks.bias_selection import (
    DEFAULT_DRIVE_CORRELATION,
    DemoHorizons,
    go_no_go,
    measure,
    sweep,
)

#: Draws per run. Larger than the reachability arm needs and small enough to stay
#: an afternoon: #42's rig drew 20,000 at a frozen operating point, where each
#: draw cost one eigendecomposition rather than a whole trajectory.
DRAWS = 8192

#: Fixed here so the run reproduces. Snapshot/restore is what reproduces a
#: trained agent; a construction sweep only needs a seed.
SEED = 42

#: The two readings of `05-timescales.md`'s derivation, both run.
READINGS = {
    "onset": DemoHorizons(fastest=3.0, longest=14.0),
    "duration": DemoHorizons(fastest=3.0, longest=750.0),
}

#: `(correlation time, scale)` of the node stalk sequence cells are driven with.
#: The first is the rig's default; the rest walk toward a frozen operating
#: point, which is the most favourable case the mechanism could ask for and not
#: a plausible one. Reported because the dwell arm is the arm this stand-in
#: could distort.
DRIVES = ((DEFAULT_DRIVE_CORRELATION, 1.0), (64.0, 0.1), (1000.0, 0.01))


def main() -> None:
    start = time.perf_counter()
    dome = build_graph()
    body = CellBody(dome.shape, generator=torch.Generator().manual_seed(SEED))

    # One sweep, put to both readings. What differs between them is the target
    # range; the measurement does not depend on it, and re-drawing would only
    # spend the afternoon twice on the same numbers.
    drawn = sweep(body, draws=DRAWS, generator=torch.Generator().manual_seed(SEED))
    for name, horizons in READINGS.items():
        print(f"### the {name} reading of the demo's perturbation horizons\n")
        print(go_no_go(dome, body, horizons=horizons, drawn=drawn).report())
        print()

    # The verdict above is read off one drive stand-in. This is whether it
    # survives the others -- a cell whose operating point barely moves has the
    # longest dwell and the best chance of a slow effective timescale, so if the
    # slow tail is empty there too, the rig's stand-in is not what emptied it.
    print("### the drive stand-in, and whether the verdict turns on it\n")
    candidates = drawn.candidates
    thresholds = (3, 10, 100)
    print(
        f"  {'correlation':>11} {'scale':>6} | {'dwell med':>9} | {'tau med':>8} "
        f"{'tau p95':>8} {'tau max':>8} | "
        + "  ".join(f"tau>={t:g}" for t in thresholds)
    )
    for correlation, scale in DRIVES:
        m = measure(
            body,
            candidates,
            drive_correlation=correlation,
            drive_scale=scale,
            generator=torch.Generator().manual_seed(SEED + 1),
        )
        tau = m.effective_timescale
        reach = "  ".join(
            f"{100 * float((tau >= t).to(tau.dtype).mean()):7.3f}%" for t in thresholds
        )
        print(
            f"  {correlation:11g} {scale:6g} | {float(torch.quantile(m.dwell, 0.5)):9.2f} | "
            f"{float(torch.quantile(tau, 0.5)):8.3f} {float(torch.quantile(tau, 0.95)):8.3f} "
            f"{float(tau.max()):8.2f} | {reach}"
        )
    print(f"\n({time.perf_counter() - start:.0f} s wall)")


if __name__ == "__main__":
    main()
