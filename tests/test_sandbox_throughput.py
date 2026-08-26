"""How fast the world runs, and what a test of that can honestly hold (#113).

The spec records roughly **400 ticks/s with rendering on a single laptop CPU
core** (`docs/spec/03-the-sandbox.md`, last line); the reference machine
measures 471 ticks/s. **That number belongs to a machine, and this file no
longer asserts it.** Its provenance is `benchmarks/sandbox_throughput.py`, off
the suite's path for the reason `benchmarks/body_forward.py` and
`benchmarks/agent_tick.py` are: a wall-clock threshold in CI measures the
runner, not the sandbox -- and this laptop runs three build agents at once, so
"the runner" is whatever the other two happened to be doing.

What stays here is the constraint that number was standing in for -- *the
render path has not regressed by an order* -- held two ways, neither of which
is a benchmark:

* **An operation count**, which is the one that bites. A step draws the camera
  exactly once, off one renderer built once. That is what "the render path has
  started doing more work" looks like in the code, and counting calls says the
  same thing on a quiet laptop, on a laptop running three builds, and on CI's
  software GL.
* **A floor on the median step**, which is the one that does not, much. Median
  rather than total: a busy machine stalls a step now and then, a total over
  300 steps adds every stall up, and a median discards them.

**Be clear about what the floor is worth.** The reference machine reads 471
ticks/s and a quiet development laptop 530-640, of which the camera pass is
most of it -- 1.6 ms of a 1.9 ms step, against 0.3 ms of physics. A render that got
ten times more expensive would still leave 60-80 ticks/s and sail past a 50 Hz
floor -- measured, with a mutant that renders ten times per frame, which the
operation count caught and the floor did not. The
floor cannot be tightened to catch that, because CI renders through
`MUJOCO_GL=osmesa` in software and the development laptop renders through CGL,
and a bound loose enough for both bounds nothing. So it is kept for what it
does do -- catch a sandbox that has become *unusable*, a renderer rebuilt per
frame at 23 ticks/s, which it does catch -- and the constraint it used to be
asked to carry is carried by the count above it.

The other constraint this file used to be read as holding is a **substep count
drifting**, and that is `tests/test_sandbox_env.py`'s
`test_control_runs_at_fifty_hertz_over_ten_substeps`: `frame_skip ==
FRAME_SKIP`, `timestep * FRAME_SKIP == 1 / CONTROL_HZ`, and one step advancing
`data.time` by exactly `1 / CONTROL_HZ`. Verified against a 100x-substep mutant,
which that test fails and no clock here needs to notice.
"""

import statistics
import time

import numpy as np

from patchworks.sandbox import PlanarPushSandbox

#: Enough steps for a median to be a median, and few enough that this stays a
#: fraction of a second on the reference machine.
TICKS = 200
WARMUP = 20

#: The floor, in ticks per second, on the **median** step with the camera
#: drawn. The reference machine reads 471, so this is an order below it.
#: Deliberately loose, and unchanged from before this file was reworked: the
#: headroom is what lets it mean the same thing on CGL and on CI's osmesa, and
#: what it costs is spelt out in the module docstring.
FLOOR_HZ = 50.0


def test_a_step_draws_the_camera_exactly_once(monkeypatch):
    """The operation count, which is what the budget is really about.

    Load-free by construction: it counts calls rather than timing them, so it
    says the same thing on a quiet laptop, on a laptop running three builds,
    and on CI's software GL.
    """
    env = PlanarPushSandbox(split="any")
    action = np.zeros(3, np.float32)
    try:
        env.reset(seed=0, options={"reset_arm": True})
        # The renderer is built lazily, on the first frame anyone asks for.
        env.step(action)
        renderer = env._renderer
        assert renderer is not None, "the observation never drew the camera"

        drawn, scened = [], []

        def counting(calls, real):
            def counted(*args, **kwargs):
                calls.append(1)
                return real(*args, **kwargs)

            return counted

        monkeypatch.setattr(renderer, "render", counting(drawn, renderer.render))
        monkeypatch.setattr(
            renderer, "update_scene", counting(scened, renderer.update_scene)
        )
        for _ in range(5):
            env.step(action)
        assert (len(drawn), len(scened)) == (5, 5)
        assert env._renderer is renderer, "the renderer is rebuilt per step"
    finally:
        env.close()


def test_the_median_step_stays_an_order_above_unusable(capsys):
    """A smoke check against a regression of *order*, not a benchmark.

    The blanked-camera env beside it is not a second assertion -- it is the
    same physics with the render taken out, timed in the same interleaved loop,
    so the printed line says how much of a step the render is on *this*
    machine without anything being compared against a number from another one.
    """
    drawn_env = PlanarPushSandbox(split="any")
    plain_env = PlanarPushSandbox(split="any", render_obs=False)
    action = np.zeros(3, np.float32)
    try:
        for world in (drawn_env, plain_env):
            world.reset(seed=0, options={"reset_arm": True})
        for _ in range(WARMUP):  # warm the renderer up
            drawn_env.step(action)
            plain_env.step(action)

        # Interleaved, one step of each per pass: a stall lands in whichever
        # half happened to be running, so it cannot bias the pair.
        drawn, plain = [], []
        for _ in range(TICKS):
            start = time.perf_counter()
            drawn_env.step(action)
            middle = time.perf_counter()
            plain_env.step(action)
            drawn.append(middle - start)
            plain.append(time.perf_counter() - middle)
    finally:
        drawn_env.close()
        plain_env.close()

    step = statistics.median(drawn)
    physics = statistics.median(plain)
    with capsys.disabled():
        print(
            f"\n  sandbox median step {1e3 * step:.2f} ms ({1 / step:.0f} ticks/s); "
            f"physics alone {1e3 * physics:.2f} ms, "
            f"so the render is {step / physics:.1f}x the physics"
        )
    assert 1.0 / step > FLOOR_HZ


class TestBenchmark:
    """`benchmarks/sandbox_throughput.py` is the reported ticks/s' provenance.

    Timings are not asserted -- a wall-clock threshold in CI measures the
    runner, not the sandbox. What is asserted is that the script still runs
    against the current API, so the number in `03-the-sandbox.md` keeps a
    reproduction.
    """

    def test_the_benchmark_still_runs(self):
        import sandbox_throughput

        samples = sandbox_throughput.time_steps(True, ticks=3)
        assert len(samples) == 3
        assert all(sample > 0 for sample in samples)
