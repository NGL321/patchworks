"""The scripted lower bound on the world, and the run that reports it.

`benchmarks/achievability.py` is the provenance of `docs/spec/03-the-sandbox.md`'s
**14 of 72** (*Achievability*). The count itself is not asserted here: the full
run is 72 tasks of up to 60 s of sim each and takes minutes, and a solve count
in the suite would be measuring the controller rather than the world. What is
asserted is the shape of the run -- 72 tasks, three splits, 60 s each -- and
that the controller still drives the current env, so the spec's number keeps a
reproduction.
"""

import numpy as np
import pytest

import achievability
from patchworks.sandbox import (
    CONTROL_HZ,
    N_PUCKS,
    BlockedAnnulusError,
    PlanarPushSandbox,
)


def test_the_run_is_seventy_two_tasks_across_all_three_splits():
    """`any` is excluded: it ignores the split distinction and draws from the
    whole space, so its tasks would be attributable to no split."""
    assert achievability.MEASURED_SPLITS == ("train", "heldout_pair", "heldout_sector")
    assert len(achievability.MEASURED_SPLITS) * achievability.TASKS_PER_SPLIT == 72
    assert achievability.CAP_TICKS / CONTROL_HZ == 60.0


def test_the_controller_stays_inside_the_action_space():
    env = PlanarPushSandbox(split="any", render_obs=False)
    try:
        _, info = env.reset(seed=0, options={"reset_arm": True})
        policy = achievability.ScriptedPusher(env)
        for _ in range(50):
            action = policy(info)
            assert env.action_space.contains(np.asarray(action, np.float32))
            _, _, _, _, info = env.step(action)
    finally:
        env.close()


def test_the_controller_carries_nothing_between_ticks():
    """No model of the puck, no memory, no plan: the same situation gets the
    same torque, whatever happened in between. That is what makes the count a
    lower bound rather than a baseline -- what the controller takes off the
    task set is what the geometry alone gives away."""
    env = PlanarPushSandbox(split="any", render_obs=False)
    try:
        _, info = env.reset(seed=2, options={"reset_arm": True})
        policy = achievability.ScriptedPusher(env)
        state = env.snapshot()
        first = policy(info)

        later = info
        for _ in range(20):
            _, _, _, _, later = env.step(policy(later))
        env.restore(state)

        assert policy(info) == pytest.approx(first)
    finally:
        env.close()


def test_the_run_reports_a_solve_count_per_split_and_per_puck():
    """A cap of 25 ticks leaves no time to solve anything, which is fine: what
    is asserted is the shape of what comes back, not the count."""
    results = achievability.measure(tasks_per_split=2, cap=25, splits=("train", "heldout_pair"))

    assert set(results) == {"train", "heldout_pair"}
    for result in results.values():
        assert result.tasks == 2
        assert 0 <= result.solved <= result.tasks
        assert sum(n for _, n in result.by_puck.values()) == result.tasks
        assert all(puck in range(N_PUCKS) for puck in result.by_puck)
        assert len(result.ticks_to_solve) == result.solved


class _ScriptedWorld:
    """A stand-in for the env: `run_task` only ever calls `step()`."""

    def __init__(self, satisfied):
        self.satisfied = iter(satisfied)

    def step(self, action):
        return None, 0.0, False, False, {"goal_satisfied": next(self.satisfied)}


def test_a_solve_holds_the_goal_rather_than_skidding_through_the_zone():
    """`goal_satisfied` is instantaneous, so a puck crossing a zone at speed
    would count as a solve if the run took the first satisfied tick."""
    world = _ScriptedWorld([True] * 5 + [False] + [True] * 40)
    solved, ticks = achievability.run_task(world, lambda info: None, {}, cap=100, hold=10)

    assert solved
    assert ticks == 16  # the run of five is broken, so the hold restarts

    stopped = _ScriptedWorld([True] * 9 + [False] * 91)
    assert achievability.run_task(stopped, lambda info: None, {}, cap=100, hold=10) == (False, 100)


def test_an_annulus_the_arm_is_standing_in_is_reported_not_absorbed(monkeypatch):
    """The arm is not reset between tasks, so it can end a task standing where
    the next layout has to go, and reset() refuses to place one inside it.
    Moving it starts a task from a pose the rest of the run does not draw from,
    so the run says how often that happened rather than quietly averaging it
    in."""
    real_reset = PlanarPushSandbox.reset
    calls = []

    def blocked_once(self, *, seed=None, options=None):
        options = options or {}
        calls.append(options)
        if len(calls) == 2 and not options.get("reset_arm"):
            raise BlockedAnnulusError("No layout clear of the arm in 64 draws.")
        return real_reset(self, seed=seed, options=options)

    monkeypatch.setattr(PlanarPushSandbox, "reset", blocked_once)
    results = achievability.measure(tasks_per_split=2, cap=5, splits=("train",))

    assert results["train"].arm_in_the_way == 1
    assert results["train"].tasks == 2
    assert calls[2] == {"reset_arm": True}


def test_the_report_names_the_count_and_calls_it_a_lower_bound(capsys):
    """And reports the budget it was actually run with: the cap travels with
    the result, so the line cannot advertise 60 s over a run given less."""
    results = achievability.measure(tasks_per_split=1, cap=500, splits=("train",))
    achievability.report(results)

    out = capsys.readouterr().out
    assert f"{results['train'].solved}/1" in out
    assert "within 10 s of sim each" in out
    assert "lower bound" in out
