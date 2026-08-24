"""Forward-pass wall time of the shared frozen cell body (ticket #84).

`docs/spec/09-the-build-stack.md`, *Measured, not assumed*, carried a numpy
stand-in at body width 128 because the body's hidden width was not yet
specified. It is now, so this measures the real thing: the whole population's
inference-phase forward path at the sizes `06-graph-topology.md` fixes --
~150 predicting cells, `n = 32`, `k = 12` -- as one batched evaluation.

The tick carries no tape (`09-the-build-stack.md`, *The locality guard*), so
the measured path runs under `torch.no_grad()`, which is how it runs for real.

    python benchmarks/body_forward.py
"""

from __future__ import annotations

import platform
import statistics
import subprocess
import time

import torch

from patchworks.body import BodyShape, CellBiases, CellBody

CELLS = 150
REPEATS = 200
WARMUP = 20


def time_forward(cells: int, shape: BodyShape, repeats: int = REPEATS) -> list[float]:
    """Milliseconds per whole-population forward pass, one sample per repeat."""
    generator = torch.Generator().manual_seed(0)
    body = CellBody(shape, generator=generator)
    biases = CellBiases(shape, cells, generator=generator)
    chart = torch.randn(cells, shape.k, generator=generator)
    node_stalk = torch.randn(cells, shape.n, generator=generator)

    samples: list[float] = []
    with torch.no_grad():
        for i in range(WARMUP + repeats):
            start = time.perf_counter()
            chart, _prediction = body(chart, node_stalk, biases)
            elapsed = time.perf_counter() - start
            if i >= WARMUP:
                samples.append(elapsed * 1e3)
            # The chart persists across ticks, and a frozen body contracts, so
            # re-seed rather than let the population decay into one region.
            if not torch.isfinite(chart).all() or chart.abs().max() < 1e-6:
                chart = torch.randn(cells, shape.k, generator=generator)
    return samples


def report(label: str, samples: list[float]) -> None:
    print(
        f"{label:<28} median {statistics.median(samples):6.3f} ms"
        f"   mean {statistics.fmean(samples):6.3f} ms"
        f"   p95 {sorted(samples)[int(0.95 * len(samples))]:6.3f} ms"
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
    shape = BodyShape(n=32, k=12)
    print(
        f"{cpu_name()}, "
        f"torch {torch.__version__}, {torch.get_num_threads()} threads, CPU"
    )
    print(
        f"n={shape.n}, k={shape.k}, widths="
        f"({shape.encode_width}, {shape.step_width}, {shape.decode_width}), "
        f"one hidden layer per map, float32\n"
    )

    report(f"{CELLS} cells (the graph)", time_forward(CELLS, shape))

    print("\nbatching, for scale:")
    for cells in (1, 10, 150, 1500):
        report(f"  {cells} cells", time_forward(cells, shape, repeats=100))


if __name__ == "__main__":
    main()
