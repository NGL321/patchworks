"""`benchmarks/untrained_fixed_point.py` still runs against the API (#120).

**Nothing about the fixed point is asserted here, and that is the point.** #120
is a diagnosis, and its last acceptance criterion forbids changing the learning
rules or the architecture on its strength alone; a test pinning today's poses,
commands or sensitivities would answer the ticket's question — *is today's
behaviour correct* — by assuming it, and would then have to be deleted by
whoever acts on the answer.

What is worth holding is that the measurement keeps a reproduction. The numbers
in #120's verdict came out of that script, and a script that stopped running
against the current API would leave them unreproducible while every test stayed
green. So this asks all five measurements to complete on the small dome at tick
counts far too short to settle anything -- and the two that take `--learn` on
that path too -- in the same shape and for the same reason
`tests/test_agent.py::TestBenchmark` asks it of `benchmarks/agent_tick.py`.

The one thing it does assert about the code under test is a shape invariant of
`sensitivity`'s protocol rather than of its result: restoring the tick state and
re-running has to leave the sheaf where it found it, or the six variants are
measured against six different starting points and the table means nothing.
"""

import re

import numpy as np
import pytest
import torch

import untrained_fixed_point as fixed_point
from patchworks import tick
from patchworks.agent import Agent, run
from patchworks.graph import build_graph

from conftest import SMALL

#: Far too few to settle anything, which is deliberate: this file is asking
#: whether the script runs, and #120's own runs are 1500 and 100000.
TICKS = 6


def test_characterise_runs(capsys):
    fixed_point.characterise(["small"], ["train"], [0], ticks=TICKS)
    out = capsys.readouterr().out
    assert "joint range" in out


def test_sensitivity_runs(capsys):
    fixed_point.sensitivity("small", "train", 0, ticks=TICKS, hold_ticks=2)
    out = capsys.readouterr().out
    for variant in ("render blanked", "efference + 0.5", "drive 1.0 -> 0.0"):
        assert variant in out
    assert "untrained" in out


def test_sensitivity_runs_on_a_taught_surface(capsys):
    """`--learn` is the same table off an adapting surface the rules have moved,
    and the header has to say which of the two a reader is looking at."""
    fixed_point.sensitivity("small", "train", 0, ticks=0, hold_ticks=2, learn=TICKS)
    out = capsys.readouterr().out
    assert "with both rules on" in out
    assert "render blanked" in out


def test_attenuation_runs(capsys):
    fixed_point.attenuation("small", "train", 0, ticks=TICKS, epsilon=1e-3)
    out = capsys.readouterr().out
    assert "one hop, one tick" in out
    assert "untrained" in out


def test_attenuation_runs_on_a_taught_surface(capsys):
    fixed_point.attenuation("small", "train", 0, ticks=0, epsilon=1e-3, learn=TICKS)
    out = capsys.readouterr().out
    assert "with both rules on" in out
    assert "one hop, one tick" in out


@pytest.mark.parametrize(
    "measurement, extra",
    (
        (fixed_point.characterise, None),
        (fixed_point.sensitivity, {"hold_ticks": 2}),
        (fixed_point.attenuation, {"epsilon": 1e-3}),
    ),
)
def test_nothing_to_measure_is_refused_in_words(measurement, extra):
    """The two `--learn` tests above pass `ticks=0`, which is the shape a reader
    copies. Forget the `learn` and there is nothing to settle a fixed point --
    which used to arrive as `IndexError` on an empty list, or an `AttributeError`
    on a `None` nobody named. Neither says what was wrong with the call."""
    with pytest.raises(ValueError, match="ticks"):
        if extra is None:
            measurement(["small"], ["train"], [0], ticks=0)
        else:
            measurement("small", "train", 0, ticks=0, **extra)


def test_drive_runs(capsys):
    fixed_point.drive("small", "train", [0], ticks=TICKS, assertions=[0.0])
    out = capsys.readouterr().out
    assert "drive   0.0" in out


def test_learning_runs(capsys):
    fixed_point.learning("small", "train", 0, ticks=TICKS, every=3)
    out = capsys.readouterr().out
    assert out.count("\n") > 4


def test_learning_never_reports_a_nan(capsys):
    """At `every == 1` a window holds one row, and a window of one row has no
    difference to take: `|d cmd|` printed `nan` out of an empty slice, in the
    column that says whether the command is still moving. Every window after
    the first now opens on the previous one's last row, and the first prints a
    dash -- there being nothing to print is worth saying, and a fabricated
    number is not."""
    fixed_point.learning("small", "train", 0, ticks=3, every=1)
    out = capsys.readouterr().out
    assert "nan" not in out
    # The `|d cmd|` column sits between the command vector and the pose vector,
    # which is what makes it findable without parsing the whole row: a lone dash
    # on the first, a number on every one after it.
    rows = [line for line in out.splitlines() if re.search(r"\]\s+\S+\s+\[", line)]
    assert len(rows) == 3
    assert re.search(r"\]\s+-\s+\[", rows[0])
    assert all(re.search(r"\]\s+\d\.\d+e[-+]\d+\s+\[", row) for row in rows[1:])


@pytest.fixture
def settled():
    """A sheaf that has ticked a few times, so its state is not the zeros."""
    env, agent = fixed_point.build("small", "train", 0)
    for _ in run(agent, TICKS, seed=0):
        pass
    yield agent
    env.close()


def test_a_restored_sheaf_is_where_it_was(settled):
    """`sensitivity` measures six variants against one reference, so the
    restore between them has to be exact. It is not enough for the node stalks
    to match: the chart, the delay buffer and the pair the bias rule would
    re-run all feed the next tick, so a restore that missed one would send the
    next variant down a different trajectory from the same visible state."""
    state = fixed_point.snapshot(settled.sheaf)
    settled.sheaf.tick()
    fixed_point.restore(settled.sheaf, state)
    for name in fixed_point._TICK_STATE:
        assert torch.equal(getattr(settled.sheaf, name), state[name]), name
    assert settled.sheaf.ticks == state["ticks"]


def test_the_restored_state_is_everything_the_tick_produced(monkeypatch, settled):
    """The check above iterates `_TICK_STATE`, so it cannot see a tensor missing
    from it -- which is the failure that matters. `incoming` was added to `Sheaf`
    for the transport rule after the tick already had six buffers; the next such
    addition, unnoticed here, would start `sensitivity`'s six variants from six
    different values of it and the table would silently mean nothing.

    So the list is held against one the tick maintains for its own reasons.
    `Sheaf.assert_no_tape` names every quantity a tick produces, it runs on
    every tick, and a new buffer that escaped it would be a hole in the locality
    guard rather than a quiet one here."""
    guarded: dict[str, torch.Tensor] = {}
    monkeypatch.setattr(tick, "assert_no_tape", lambda **named: guarded.update(named))
    settled.sheaf.assert_no_tape()
    assert set(guarded) == set(fixed_point._TICK_STATE)


def test_the_small_dome_is_the_suite_s_own(settled):
    """The script measures the dome `tests/conftest.py` defines, not a copy of
    it. A benchmark aimed at a drifted copy does not fail; it reports someone
    else's numbers under this dome's name."""
    spec, image_size = fixed_point.dome_named("small")
    assert spec is SMALL
    assert image_size == SMALL.patch_grid * 4


def test_the_hold_takes_the_world_out_of_the_loop(monkeypatch, settled):
    """`sensitivity`'s hold never steps the environment: the same observation is
    written every tick, which is what makes the settled difference a reading of
    the graph's own transfer rather than of the physics."""
    stepped = []
    monkeypatch.setattr(
        settled.env, "step", lambda action: stepped.append(action) or pytest.fail()
    )
    observation, _ = settled.env.reset(seed=0)
    applied = np.zeros(settled.joints, dtype=np.float32)
    fixed_point.hold(settled, observation, applied, None, 3)
    assert stepped == []


def test_the_drive_override_is_the_last_word(settled):
    """The hold's `drive` argument stands where `DRIVE_ASSERTION` would, after
    the external write rather than before it. Written before, reconciliation
    would not have eroded it but `Agent.write` would have overwritten it, and
    the drive variants would have measured nothing at all."""
    observation, _ = settled.env.reset(seed=0)
    applied = np.zeros(settled.joints, dtype=np.float32)
    stalks, _ = fixed_point.hold(settled, observation, applied, 7.0, 2)
    assert float(stalks[settled._drive_slice][0]) == 7.0


def test_a_render_is_sized_by_the_dome_that_tiles_it():
    """`Agent` refuses a render its patch grid does not tile, so the two sizes
    the script offers are facts about the two domes rather than settings."""
    for name in ("small", "full"):
        spec, image_size = fixed_point.dome_named(name)
        side, remainder = divmod(image_size, spec.patch_grid)
        assert not remainder
        assert side * side * 3 == spec.patch_stalk


def test_it_builds_the_real_dome_too():
    """`dome_named('full')` is the graph #120's headline numbers were taken on,
    and it is built here rather than run: construction is what could break
    under a spec change, and 682 edges of it is not a thing to tick in CI."""
    spec, image_size = fixed_point.dome_named("full")
    assert image_size == 64
    assert len(build_graph(spec).edges) > len(build_graph(SMALL).edges)


def test_the_agent_it_builds_is_seeded():
    """Two builds at one seed are the same draw, or nothing in #120's tables is
    reproducible."""
    env_a, agent_a = fixed_point.build("small", "train", 3)
    env_b, agent_b = fixed_point.build("small", "train", 3)
    try:
        assert torch.equal(agent_a.sheaf.maps.maps, agent_b.sheaf.maps.maps)
        assert isinstance(agent_a, Agent)
    finally:
        env_a.close()
        env_b.close()
