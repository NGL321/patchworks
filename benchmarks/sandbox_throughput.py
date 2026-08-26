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
"""

from __future__ import annotations

import math
import platform
import statistics
import subprocess
import time

import numpy as np

from patchworks.sandbox import PlanarPushSandbox

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


def cpu_name() -> str:
    """The reference machine, so a number can be compared with the ones on record."""
    if platform.system() == "Darwin":
        try:
            return subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    return platform.processor() or platform.machine()


def main() -> None:
    import os

    print(f"{cpu_name()}, MUJOCO_GL={os.environ.get('MUJOCO_GL', '(default)')}\n")
    drawn = time_steps(True)
    plain = time_steps(False)
    report("step, camera rendered", drawn)
    report("step, camera blanked", plain)
    print(
        f"\nthe render is {statistics.median(drawn) / statistics.median(plain):.1f}x "
        "a step's physics on this machine."
    )


if __name__ == "__main__":
    main()
