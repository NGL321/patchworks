"""What the three deviations cost against stock Gymnasium tooling.

`docs/spec/03-the-sandbox.md`, *What the deviations cost in the ecosystem*.
The deviations are honoured against Gymnasium rather than by it, and every
place they bite fails *quietly* -- a run corrupted quietly is indistinguishable
from the thing the agent is supposed to experience when `reset()` fires. So the
cost is asserted rather than tolerated: `check_env` must fail exactly one check,
by name, and a second failure breaks the build instead of being lost in a
known-fails suite.

**Which check is named differs from the spec, and the difference is measured.**
The spec names `check_reset_seed_determinism`, reasoning that it "calls
`reset(seed=123)` twice and requires the observations to match; here they do
not, because `reset()` does not reset the arm". Against Gymnasium 1.x that
check never steps the env between its two seeded resets, so the arm is in the
*same* configuration for both and the deviation it is supposed to detect cannot
fire there -- it passes. The check that does step between its seeded resets is
`check_step_determinism`, and that is the one that fails, for exactly the
reason the spec gives. The deviation detected is still the first one; only the
name of the check that catches it is different. Recorded here because the
spec's paragraph needs the same correction.
"""

import gymnasium
import pytest
from gymnasium.envs.registration import EnvSpec
from gymnasium.utils import env_checker

from patchworks.sandbox import ENTRY_POINT, ENV_ID, PlanarPushSandbox, SpecLimitError

#: The one check `check_env` is expected to fail, and the whole of it.
EXPECTED_FAILURE = "check_step_determinism"

#: Everything `check_env` calls, so that a check silently disappearing from
#: Gymnasium is visible here rather than quietly reducing what is asserted.
CHECKS = (
    "check_action_space",
    "check_space_limit",
    "check_observation_space",
    "check_seed_deprecation",
    "check_reset_return_info_deprecation",
    "check_reset_return_type",
    "check_reset_seed_determinism",
    "check_reset_options",
    "env_reset_passive_checker",
    "env_step_passive_checker",
    "check_step_determinism",
    "env_render_passive_checker",
)


def _run_check_env_collecting_failures(env, monkeypatch):
    """Run every `check_env` check, rather than stopping at the first failure.

    `check_env` raises on the first assertion, which would hide whatever comes
    after it. Each check is wrapped so a failure is recorded and the run
    continues, which is what lets the assertion below be about the whole set.
    """
    failures = {}

    def wrap(name):
        original = getattr(env_checker, name)

        def wrapped(*args, **kwargs):
            try:
                return original(*args, **kwargs)
            except AssertionError as exc:
                failures.setdefault(name, str(exc))

        return wrapped

    for name in CHECKS:
        assert hasattr(env_checker, name), f"gymnasium no longer defines {name}"
        monkeypatch.setattr(env_checker, name, wrap(name))

    env_checker.check_env(env)
    return failures


def test_check_env_fails_exactly_one_check_and_it_is_named(monkeypatch):
    env = gymnasium.make(ENV_ID, render_mode="rgb_array")
    try:
        failures = _run_check_env_collecting_failures(env.unwrapped, monkeypatch)
    finally:
        env.close()

    assert set(failures) == {EXPECTED_FAILURE}, (
        "check_env's failures are part of the contract: exactly one, and it is "
        f"{EXPECTED_FAILURE}. Got {sorted(failures)}."
    )
    # It fails for the stated reason -- the arm is not reset, so the same seed
    # and the same action do not reproduce the same observation.
    assert "observations are not equivalent" in failures[EXPECTED_FAILURE]


def test_the_registration_carries_no_step_limit():
    """`make()` wraps in TimeLimit whenever max_episode_steps is passed or
    carried on the registered spec, and TimeLimit sets truncated=True -- which
    every standard loop resets on."""
    env = gymnasium.make(ENV_ID)
    try:
        assert env.spec.max_episode_steps is None
        assert not any(isinstance(e, gymnasium.wrappers.TimeLimit) for e in _wrappers(env))
    finally:
        env.close()


def test_nondeterministic_is_not_set():
    """The flag means the observation cannot be repeated from the same initial
    state, RNG state and actions, which is false of this env and is precisely
    what snapshot/restore delivers. Setting it would assert something untrue
    about the physics in order to quiet a check about `reset()`."""
    env = gymnasium.make(ENV_ID)
    try:
        assert env.spec.nondeterministic is False
    finally:
        env.close()


def test_the_env_refuses_a_registration_carrying_a_step_limit():
    """Refused, not merely undocumented: prefer the constraint that cannot
    drift to the sentence nobody rereads.

    `make()` rebuilds the unwrapped env's spec with `max_episode_steps=None`
    whatever the registration said, and wraps TimeLimit outside it, so the
    limit is invisible from inside the env once it exists. The registry is
    where it is still visible, and construction is when it is still early
    enough to matter.
    """
    limited = "Patchworks/PlanarPushSandbox-limited-v0"
    gymnasium.register(id=limited, entry_point=ENTRY_POINT, max_episode_steps=500)
    try:
        with pytest.raises(SpecLimitError, match="max_episode_steps"):
            gymnasium.make(limited)
        with pytest.raises(SpecLimitError, match="max_episode_steps"):
            PlanarPushSandbox(render_obs=False)
    finally:
        del gymnasium.registry[limited]


def test_the_env_refuses_a_spec_carrying_a_step_limit_assigned_to_it():
    env = PlanarPushSandbox(render_obs=False)
    try:
        assert env.spec is None
        env.spec = None  # the ordinary case: no limit, no complaint
        with pytest.raises(SpecLimitError, match="max_episode_steps"):
            env.spec = EnvSpec(id=ENV_ID, entry_point=ENTRY_POINT, max_episode_steps=500)
    finally:
        env.close()


def test_record_episode_statistics_goes_inert():
    """It fills its info key only under `if terminated or truncated:`, so
    across an entire run it emits nothing at all. Not a defect to fix -- this
    is what "no episodes" means downstream, and it is asserted so that reaching
    for the wrapper cannot look like it is working."""
    env = gymnasium.wrappers.RecordEpisodeStatistics(
        gymnasium.make(ENV_ID, render_obs=False)
    )
    try:
        env.reset(seed=0, options={"reset_arm": True})
        action = env.action_space.sample()
        for _ in range(50):
            _, _, _, _, info = env.step(action)
            assert "episode" not in info
    finally:
        env.close()


def _wrappers(env):
    while isinstance(env, gymnasium.Wrapper):
        yield env
        env = env.env
