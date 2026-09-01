"""Per-tick wall time of the whole tick, against `env.step()` (ticket #86).

`docs/spec/09-the-build-stack.md`, *Measured, not assumed*, carried a numpy
stand-in for a whole agent tick, taken at a body width of 128 before the body's
width was specified. This measures the real thing — both phases and both halves
of the world seam, over the real dome, 150 predicting cells and 682 edges — and
the environment beside it, in the same run on the same machine, because the
ratio is the number that matters and the rows on record were taken on a laptop
this may not be.

The split is reported because the four parts are different objects: the
inference phase is one batched forward pass over the population, the
message-passing phase is two `bmm`s over 1364 edge endpoints plus a scatter, and
the world's read and the external write are index traffic with no arithmetic in
them at all.

Run it on an otherwise idle machine. The tick spreads over every torch thread
and `env.step()` does not, so under load the tick degrades by a factor of two
while the environment beside it barely moves -- and the ratio, which is the
output, then measures the machine instead.

    python benchmarks/agent_tick.py
"""

from __future__ import annotations

import math
import statistics
import time

import numpy as np
import torch

from body_forward import cpu_name
from patchworks.agent import Agent
from patchworks.graph import build_graph
from patchworks.sandbox import PlanarPushSandbox

#: The figure on record in `docs/spec/09-the-build-stack.md`, measured on the
#: development laptop (i7-8750H, CPU only).
RECORDED_ENV_STEP_MS = 3.18

TICKS = 200
WARMUP = 20


def report(label: str, samples: list[float], reference: float | None = None) -> None:
    p95 = sorted(samples)[math.ceil(0.95 * len(samples)) - 1]
    line = (
        f"{label:<34} median {statistics.median(samples):6.3f} ms"
        f"   mean {statistics.fmean(samples):6.3f} ms"
        f"   p95 {p95:6.3f} ms"
    )
    if reference:
        line += f"   {statistics.median(samples) / reference * 100:5.1f}% of env.step()"
    print(line)


def measure(ticks: int = TICKS) -> dict[str, list[float]]:
    """One sustained run, timing each part of every tick separately."""
    env = PlanarPushSandbox(split="any")
    agent = Agent(env, dome=build_graph(), generator=torch.Generator().manual_seed(0))
    observation, _info = env.reset(seed=0)
    agent.observe(observation)

    samples: dict[str, list[float]] = {
        "inference phase": [],
        "message-passing phase": [],
        "the world's read": [],
        "external write": [],
        "env.step()": [],
    }
    for i in range(WARMUP + ticks):
        start = time.perf_counter()
        agent.sheaf.inference_phase()
        inference = time.perf_counter()
        agent.sheaf.message_passing_phase()
        passing = time.perf_counter()
        # The world's read and the world's write are the two halves of the
        # ordering's seam. Both are billed to the tick rather than to the
        # environment -- the whole output of this script is a ratio, and a
        # misattribution across that seam would flatter one side -- and both get
        # their own row rather than being folded into a phase, which would
        # inflate the row the doc leans on.
        command = agent.command()
        applied = np.clip(command, agent.action_low, agent.action_high)
        read = time.perf_counter()
        observation, _r, _t, _tr, _info = env.step(applied)
        stepped = time.perf_counter()
        agent.write(observation, applied)
        written = time.perf_counter()
        if i >= WARMUP:
            samples["inference phase"].append((inference - start) * 1e3)
            samples["message-passing phase"].append((passing - inference) * 1e3)
            samples["the world's read"].append((read - passing) * 1e3)
            samples["env.step()"].append((stepped - read) * 1e3)
            samples["external write"].append((written - stepped) * 1e3)
    env.close()
    return samples


def main() -> None:
    print(
        f"{cpu_name()}, torch {torch.__version__}, "
        f"{torch.get_num_threads()} threads, CPU"
    )
    dome = build_graph()
    print(
        f"{len(dome.predicting)} predicting cells, {len(dome.edges)} edges, "
        f"n={dome.shape.n}, k={dome.shape.k}, float32, no_grad\n"
    )

    samples = measure()
    env_step = statistics.median(samples["env.step()"])
    phases = [
        a + b
        for a, b in zip(samples["inference phase"], samples["message-passing phase"])
    ]
    # The seam is part of a tick -- the world's read, and the write that is the
    # ordering's last word -- so the tick's own cost includes both. What it does
    # not include is the world's own step, which is the row it is compared
    # against.
    seam = [
        r + w
        for r, w in zip(samples["the world's read"], samples["external write"])
    ]
    agent = [p + s for p, s in zip(phases, seam)]

    for label in (
        "inference phase",
        "message-passing phase",
        "the world's read",
        "external write",
    ):
        report(f"  {label}", samples[label], env_step)
    report("the two phases", phases, env_step)
    report("the tick, without the world", agent, env_step)
    report("env.step()", samples["env.step()"])
    print()
    report(
        "a whole tick, world included",
        [a + s for a, s in zip(agent, samples["env.step()"])],
    )
    ratio = statistics.median(agent) / env_step
    print(
        f"\nenv.step() is {env_step:.3f} ms here against {RECORDED_ENV_STEP_MS} ms on "
        f"record; the tick costs {ratio:.2f}x the environment, so at the recorded "
        f"figure it would be {ratio * RECORDED_ENV_STEP_MS:.2f} ms."
    )


if __name__ == "__main__":
    main()
