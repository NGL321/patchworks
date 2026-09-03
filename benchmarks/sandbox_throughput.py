"""Wall-clock throughput of the sandbox, with and without its render (#113).

`docs/spec/03-the-sandbox.md`, last line, records roughly **400 ticks/s with
rendering on a single laptop CPU core**. This is where that number comes from,
and it lives here rather than in the suite for the reason
`benchmarks/body_forward.py` and `benchmarks/agent_tick.py` do: a wall-clock
number belongs to a machine, and a threshold on one in CI measures the runner.
What the suite keeps is in `tests/test_sandbox_throughput.py` -- an
order-of-magnitude floor and an operation count -- and what it does not keep is
the number itself, which is reproduced by running this.

    python benchmarks/sandbox_throughput.py

The two rows are the same physics; they differ only in whether the observation's
64x64 camera pass runs, so the gap between them is the render's whole cost.

The render is the GL stack's, not MuJoCo's alone: on the development laptop it
is CGL, and in CI `MUJOCO_GL=osmesa` makes it software. Those are different
machines by the same argument as above, so the two rows are reported side by
side rather than reduced to a ratio anything is compared against.

**It carries the cutoff hook** (#284). After the rows, it states for every open
problem whose `@cutoff measurement` names this rig whether the bar was crossed,
and records the run on the problem — `tools/cutoff_report.py`, and
`docs/agents/registers.md`, *Cutoffs*. That is the only way a measurement cutoff
can fire, because a rig is the only thing that takes the reading and rigs do not
run in CI. It changes nothing about the paragraph above: the report is printed
and filed, nothing is asserted, and a crossing does not make this script exit
non-zero. `--no-file` prints the same report and touches the tracker not at all.

The metrics a bar may name are :func:`readings`' keys.
"""

from __future__ import annotations

import argparse
import math
import os
import pathlib
import statistics
import sys
import time

import numpy as np

# Borrowed rather than copied, as `agent_tick.py` borrows it: two copies of the
# `sysctl` fallback would drift, and numbers from different runs are comparable
# only because the machine is named the same way in each.
from body_forward import cpu_name
from patchworks.sandbox import PlanarPushSandbox

# The cutoff hook lives in `tools/` and not here, because it shells `gh` and a
# network tool belongs on the far side of the line `tests/test_cli.py` defends.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "tools"))
from cutoff_report import report as report_cutoffs  # noqa: E402

TICKS = 300
WARMUP = 20


def time_steps(render_obs: bool, ticks: int = TICKS) -> list[float]:
    """Milliseconds per `env.step`, one sample per tick."""
    env = PlanarPushSandbox(split="any", render_obs=render_obs)
    action = np.zeros(3, np.float32)
    samples: list[float] = []
    try:
        env.reset(seed=0, options={"reset_arm": True})
        for i in range(WARMUP + ticks):
            start = time.perf_counter()
            env.step(action)
            elapsed = time.perf_counter() - start
            if i >= WARMUP:
                samples.append(elapsed * 1e3)
    finally:
        env.close()
    return samples


def report(label: str, samples: list[float]) -> None:
    # Nearest rank, as in benchmarks/body_forward.py: the smallest sample at or
    # above the 95th percentile.
    p95 = sorted(samples)[math.ceil(0.95 * len(samples)) - 1]
    median = statistics.median(samples)
    print(
        f"{label:<28} median {median:6.3f} ms"
        f"   mean {statistics.fmean(samples):6.3f} ms"
        f"   p95 {p95:6.3f} ms"
        f"   ({1e3 / median:5.0f} ticks/s)"
    )


def readings(drawn: list[float], plain: list[float]) -> dict[str, float]:
    """What this run has to offer a `measurement` cutoff, by name.

    The keys are the vocabulary a `@cutoff measurement sandbox_throughput …`
    may write a bar against, and they are stated here rather than derived from
    the printed lines: a cutoff naming a metric this rig does not report is a
    thing the report says out loud (`tools/cutoff_report.py`), and it can only
    say it against a named set.
    """
    rendered = statistics.median(drawn)
    blanked = statistics.median(plain)
    return {
        "ticks_per_second": 1e3 / rendered,
        "ticks_per_second_blanked": 1e3 / blanked,
        "step_ms": rendered,
        "step_ms_blanked": blanked,
        "camera_ms": rendered - blanked,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--no-file",
        dest="file",
        action="store_false",
        help="print the cutoff report and file nothing on the tracker",
    )
    arguments = parser.parse_args(argv)

    print(f"{cpu_name()}, MUJOCO_GL={os.environ.get('MUJOCO_GL', '(default)')}\n")
    drawn = time_steps(True)
    plain = time_steps(False)
    report("step, camera rendered", drawn)
    report("step, camera blanked", plain)
    # The camera pass is the *difference* between the rows, not their quotient:
    # the drawn row is physics and render together.
    camera = statistics.median(drawn) - statistics.median(plain)
    print(
        f"\nthe camera pass is {camera:.3f} ms, "
        f"{camera / statistics.median(plain):.1f}x the physics beside it."
    )
    # The measurement half of the cutoff mechanism (#284). It states a verdict
    # per open problem cutting on this rig, files the run on the problem, and
    # asserts nothing -- a crossing is a report and a label, and this function
    # still returns the same way it did before the bar was crossed.
    report_cutoffs("sandbox_throughput", readings(drawn, plain), file=arguments.file)


if __name__ == "__main__":
    main()
