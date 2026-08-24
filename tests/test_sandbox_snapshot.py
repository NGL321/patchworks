"""Snapshot and restore, bit-exact.

`docs/spec/03-the-sandbox.md`, *The Gymnasium contract, made continual*, and
[ADR-0001](../docs/adr/0001-continual-learning-applies-to-the-adapting-surface.md).

There is no episode boundary to restart from, so this is where reproducibility
comes from, and the acceptance demo's repeated runs are built on it: a trial
starts from a restore, not a reset.
"""

import gymnasium as gym
import mujoco
import numpy as np
import pytest

from patchworks.sandbox import (
    ENV_ID,
    N_PUCKS,
    STATE_SPEC,
    PlanarPushSandbox,
    Snapshot,
    Task,
    friction_scale,
    restore,
    snapshot,
)

ZERO = np.zeros(3, np.float32)


@pytest.fixture
def env():
    e = PlanarPushSandbox(split="train")
    e.reset(seed=7, options={"reset_arm": True})
    yield e
    e.close()


@pytest.fixture
def fast_env():
    """Physics only, for the checks that never look at the render."""
    e = PlanarPushSandbox(split="any", render_obs=False)
    yield e
    e.close()


def mid_flight(env, ticks=40, seed=3):
    """Step the env into a state nothing was initialised to.

    Every restore here is a non-initial-state load -- there is no episode
    boundary, so the load-bearing case is the only case -- and a snapshot taken
    straight off `reset()` would not exercise it. The arm is driven at random
    to a pose nothing chose, at speed, with the solver warm: `qacc_warmstart`
    is the field an enumeration of the state would have dropped, and it is
    still zero at the initial state that is always cold-started.
    """
    rng = np.random.default_rng(seed)
    for _ in range(ticks):
        env.step(rng.uniform(-1, 1, 3).astype(np.float32))
    assert env.data.qacc_warmstart.any(), "not warm yet: the interesting field is still zero"


def physics_state(env) -> np.ndarray:
    v = np.empty(mujoco.mj_stateSize(env.model, STATE_SPEC))
    mujoco.mj_getState(env.model, env.data, v, STATE_SPEC)
    return v


# -- what the state is ----------------------------------------------------------


def test_the_snapshot_is_the_engine_constant_not_an_enumeration(env):
    """MuJoCo defines `mjSTATE_INTEGRATION` as the entire set of inputs to the
    forward dynamics, so naming it tracks the model. An enumeration drifts the
    moment the arena gains a feature, and drifts *silently* -- as a trajectory
    that diverges rather than an error."""
    mid_flight(env)
    state = snapshot(env)

    assert STATE_SPEC is mujoco.mjtState.mjSTATE_INTEGRATION
    assert state.physics.size == mujoco.mj_stateSize(env.model, STATE_SPEC)
    # The three things there are, and no fourth: the friction field in
    # particular is derived rather than carried.
    assert isinstance(state, Snapshot)
    assert [f.name for f in state.__dataclass_fields__.values()] == ["physics", "task", "rng"]
    assert state.task is env.task
    assert state.rng == env.np_random.bit_generator.state


def test_the_state_carries_the_warmstart_an_enumeration_would_drop(env):
    """The field the obvious enumeration omits is `qacc_warmstart`, and
    warmstarts matter precisely when loading a non-initial state, since the
    initial state is always cold-started.

    Sized here rather than argued: `mjSTATE_PHYSICS` is what an enumeration of
    qpos/qvel/act comes to, and the constant this env names is larger by, among
    other things, one warmstart per degree of freedom.
    """
    mid_flight(env)
    state = snapshot(env)
    warm = env.data.qacc_warmstart.copy()

    enumerated = mujoco.mj_stateSize(env.model, mujoco.mjtState.mjSTATE_PHYSICS)
    assert state.physics.size > enumerated
    assert mujoco.mj_stateSize(env.model, mujoco.mjtState.mjSTATE_WARMSTART) == env.model.nv

    for _ in range(5):
        env.step(np.full(3, 0.8, np.float32))
    assert not np.array_equal(env.data.qacc_warmstart, warm)

    restore(env, state)
    assert np.array_equal(env.data.qacc_warmstart, warm)


def test_a_snapshot_of_an_unarranged_world_is_refused():
    """A snapshot of a world nobody has arranged has no task to return to, and
    the restore that could not light a goal zone is a worse place to say so."""
    env = PlanarPushSandbox(render_obs=False)
    try:
        with pytest.raises(RuntimeError, match="call reset\\(\\) before"):
            snapshot(env)
    finally:
        env.close()


def test_a_restore_into_a_world_nobody_has_arranged_is_refused(env):
    """A restore rewinds a run that is already going, and two things can
    disagree about whether one has started. The env knows whether a layout was
    ever drawn. The wrapper `gymnasium.make()` returns knows whether `reset()`
    came through it -- it refuses a `step()` until it has, and it sees neither
    a restore going around it nor a `reset()` on `env.unwrapped`. Both are
    asked, because an env that takes the state and then declines to tick is
    the worst of the outcomes: the remedy for it throws the state away."""
    mid_flight(env, ticks=5)
    state = snapshot(env)

    bare = PlanarPushSandbox(render_obs=False)
    try:
        with pytest.raises(RuntimeError, match="call reset\\(\\) before"):
            restore(bare, state)
    finally:
        bare.close()

    fresh = gym.make(ENV_ID)
    try:
        with pytest.raises(RuntimeError, match="has not seen a reset"):
            restore(fresh, state)

        # Resetting the base env is the near miss: the world is arranged, so
        # the env is satisfied, and the wrapper still has not seen a reset.
        # Every hand is reached as env.unwrapped.<method>(), so it is an easy
        # slip, and the remedy afterwards -- another reset() -- would throw the
        # restored state away.
        fresh.unwrapped.reset(seed=0, options={"reset_arm": True})
        with pytest.raises(RuntimeError, match="has not seen a reset"):
            restore(fresh, state)

        # and the remedy costs nothing: the layout reset() draws is overwritten
        # by the restore a moment later.
        fresh.reset(seed=0, options={"reset_arm": True})
        restore(fresh, state)
        assert np.array_equal(physics_state(fresh.unwrapped), physics_state(env))
        fresh.step(ZERO)
    finally:
        fresh.close()


def test_a_held_snapshot_cannot_drift(env):
    """The demo holds forty of these down one continual run. A snapshot that
    changed while it was being held would be a restore landing somewhere nobody
    chose, so the copy is sealed rather than shared with `MjData`."""
    mid_flight(env, ticks=5)
    state = snapshot(env)
    held = np.array(state.physics)
    with pytest.raises(ValueError):
        state.physics[0] = 0.0
    for _ in range(5):
        env.step(np.full(3, 0.7, np.float32))
    assert np.array_equal(state.physics, held)


# -- the replayed tail ----------------------------------------------------------


def test_a_replayed_hundred_tick_tail_shows_zero_divergence(env):
    """Not "close": equal. Divergence is compared on the whole state and on the
    whole observation, the 64x64 render included, at every one of the hundred
    ticks -- a float32 observation would hide a divergence the state carries.
    """
    mid_flight(env)
    assert env.data.ncon > 0, "the solver wants real contacts to warm-start from"
    rng = np.random.default_rng(0)
    actions = rng.uniform(-1, 1, (100, 3)).astype(np.float32)

    state = snapshot(env)
    first = [(env.step(a)[0], physics_state(env)) for a in actions]
    restore(env, state)
    second = [(env.step(a)[0], physics_state(env)) for a in actions]

    for tick, ((obs_a, state_a), (obs_b, state_b)) in enumerate(zip(first, second)):
        assert np.array_equal(state_a, state_b), f"state diverged at tick {tick}"
        for key in obs_a:
            assert np.array_equal(obs_a[key], obs_b[key]), f"{key} diverged at tick {tick}"

    # and the tail is a real one: the world moved over those hundred ticks.
    assert not np.array_equal(first[0][1], first[-1][1])


def test_a_restore_rewinds_the_clock_where_a_reset_never_does(env):
    """The sharpest way the two differ in the data. Physics time is monotonic
    across an entire run of `reset()`s -- the agent lives through those. A
    restore rewinds the whole universe, so time goes back with it."""
    mid_flight(env, ticks=10)
    state = snapshot(env)
    at_snapshot = env.data.time

    for _ in range(10):
        env.step(ZERO)
    env.reset()
    assert env.data.time > at_snapshot

    restore(env, state)
    assert env.data.time == at_snapshot


def test_a_restore_rewinds_the_sampler_too(env):
    """The sampler's RNG is one of the two things MuJoCo does not know about,
    so a restored world draws the same next layout it would have drawn."""
    state = snapshot(env)
    first = env.sample_task()
    restore(env, state)
    second = env.sample_task()
    assert np.array_equal(first.puck_xy, second.puck_xy)
    assert np.array_equal(first.puck_theta, second.puck_theta)
    assert first.pair == second.pair


def test_a_restore_rewinds_what_is_wanted(env):
    """The task is the other. A retarget after a snapshot does not survive it,
    and the goal light follows the restored task rather than the live one."""
    state = snapshot(env)
    other = next(z for z in range(3) if z != env.task.goal_zone)
    env.retarget(goal_zone=other)

    restore(env, state)
    assert env.task is state.task
    lit = [z for z, sid in enumerate(env._zone_sid) if env.model.site_rgba[sid][3] > 0.5]
    assert lit == [state.task.goal_zone]


# -- the friction field is reconstructed, not stored -----------------------------


def test_the_friction_field_is_rebuilt_from_the_restored_positions(fast_env):
    """The field is deliberately not part of the state: it is a pure function
    of puck position, so restoring the state restores it.

    Storage and reconstruction are told apart by writing a value into the model
    that the field function can never produce, and *then* snapshotting. A
    stored field would bring that value back. A reconstructed one comes out of
    where the pucks now are, and the scribble is gone.
    """
    env = fast_env
    xy = np.array([[0.0, -0.25], [-0.22, 0.10], [0.10, 0.30]])
    env.reset(seed=1, options={"reset_arm": True, "task": Task(xy, np.zeros(3), 0, 0)})

    impossible = 9.0  # far outside nominal * [0.75, 1.25]
    for dof in env._puck_dofadr:
        env.model.dof_frictionloss[dof : dof + 3] = impossible
    state = snapshot(env)

    env.perturb(0, [0.30, -0.20])
    env.step(ZERO)

    restore(env, state)
    for i, dof in enumerate(env._puck_dofadr):
        live = env.model.dof_frictionloss[dof : dof + 3]
        assert live == pytest.approx(env._friction_nominal[i] * friction_scale(env.puck_pose(i)))
        assert not np.any(live == impossible), "the field came back stored, not rebuilt"


def test_no_friction_number_is_carried_in_the_state(fast_env):
    """The other half of the same claim, from the snapshot's side: resampling
    frictionloss per `reset()` was rejected because it would put a number into
    the model that `mjSTATE_INTEGRATION` does not cover, and restore would then
    diverge silently. Nothing in the state is that number."""
    env = fast_env
    xy = np.array([[0.20, 0.10], [-0.18, -0.22], [0.05, 0.30]])
    env.reset(seed=2, options={"reset_arm": True, "task": Task(xy, np.zeros(3), 0, 0)})
    env.step(ZERO)
    state = snapshot(env)

    for i, dof in enumerate(env._puck_dofadr):
        for value in env.model.dof_frictionloss[dof : dof + 3]:
            assert not np.any(state.physics == value)
        # and the field is genuinely scaled, so the search above was not for 1.0
        assert friction_scale(xy[i]) != pytest.approx(1.0, abs=1e-3)


# -- an experimenter's tool, off the env's surface -------------------------------


def test_restore_is_not_an_operation_the_env_offers():
    """A restore rewinds the agent's own adapting surface, so there is no tick
    at which a cell could observe one. `reset()`, `perturb()` and
    `disturb_arm()` are all in-band by contrast -- the world moves and the arm
    feels it -- which is why they are methods and these two are not."""
    for name in ("snapshot", "restore"):
        assert not hasattr(PlanarPushSandbox, name)


def test_the_experimenter_reaches_them_through_whatever_make_hands_back():
    """`gymnasium.make()` returns an `OrderEnforcing` wrapper, which forwards
    no attribute the `Env` contract names. A method on the env would therefore
    be unreachable from the standard construction path; a function that unwraps
    what it is given is not."""
    wrapped = gym.make(ENV_ID)
    try:
        assert wrapped is not wrapped.unwrapped
        # `perturb` is the demonstration, not `snapshot`: it is a method the
        # env really does define, and the wrapper still does not carry it.
        assert hasattr(wrapped.unwrapped, "perturb")
        assert not hasattr(wrapped, "perturb")

        wrapped.reset(seed=11, options={"reset_arm": True})
        for _ in range(5):
            wrapped.step(ZERO)
        state = snapshot(wrapped)
        before = physics_state(wrapped.unwrapped)

        for _ in range(5):
            wrapped.step(np.full(3, 0.5, np.float32))
        restore(wrapped, state)
        assert np.array_equal(physics_state(wrapped.unwrapped), before)
    finally:
        wrapped.close()


def test_a_restore_rewinds_nothing_above_the_env_itself():
    """Rewinding the whole universe stops at `env.unwrapped`: a wrapper's own
    bookkeeping is not state a restore can reach.

    The one step limit the env cannot refuse counts in exactly that place.
    `gymnasium.make(id, max_episode_steps=n)` wraps `TimeLimit` outside, where
    nothing the env can read mentions the limit -- pinned in
    `tests/test_sandbox_conformance.py` -- and the count runs on through a
    restore, so `truncated=True` lands `n` ticks after the *run* began rather
    than after the trial did. Every standard loop resets on that, which here
    rearranges the world mid-trial. Asserted so the exposure is a recorded fact
    rather than an assumption.
    """
    wrapped = gym.make(ENV_ID, max_episode_steps=3, render_obs=False)
    try:
        wrapped.reset(seed=0, options={"reset_arm": True})
        assert wrapped.step(ZERO)[3] is False
        state = snapshot(wrapped)
        assert wrapped.step(ZERO)[3] is False

        restore(wrapped, state)
        # the trial's second tick, and the run's third
        assert wrapped.step(ZERO)[3] is True
    finally:
        wrapped.close()


def test_a_restore_is_invisible_from_inside(env):
    """Nothing in the observation announces one, which is the whole difference
    from `reset()` being in-band: the agent that would have to notice is itself
    rewound. Three restores of one snapshot give three identical next ticks --
    which is what pairing the demo's three events at a snapshot rests on."""
    mid_flight(env, ticks=12)
    state = snapshot(env)

    seen = []
    for _ in range(3):
        restore(env, state)
        obs, reward, terminated, truncated, info = env.step(ZERO)
        assert (reward, terminated, truncated) == (0.0, False, False)
        seen.append((obs, info))

    for obs, info in seen[1:]:
        for key in obs:
            assert np.array_equal(obs[key], seen[0][0][key])
        assert np.array_equal(info["puck_pose"], seen[0][1]["puck_pose"])


def test_three_restores_of_one_snapshot_carry_three_different_events(env):
    """The demo restores each snapshot three times, once per hand, so that both
    hands see an identical agent at an identical moment. What differs after the
    restore is the event, and nothing else."""
    mid_flight(env, ticks=12)
    state = snapshot(env)

    restore(env, state)
    plain = env.step(ZERO)[0]  # the control: the same restore, no event

    restore(env, state)
    env.disturb_arm(0, 0.05)
    nudged = env.step(ZERO)[0]

    restore(env, state)
    env.perturb(0, [0.30, -0.20])
    moved = env.step(ZERO)[0]

    restore(env, state)
    env.retarget(goal_zone=(env.task.goal_zone + 1) % 3)
    retargeted = env.step(ZERO)[0]

    # An impulse is the world moving, so proprioception reports it.
    assert not np.array_equal(nudged["qvel"], plain["qvel"])
    # A retarget touches nothing physical: it is a change in appearance, and
    # the render is the only place it shows up.
    assert np.array_equal(retargeted["qvel"], plain["qvel"])
    assert not np.array_equal(retargeted["image"], plain["image"])
    # A teleport moves a puck, which the render sees whether or not the arm does.
    assert not np.array_equal(moved["image"], plain["image"])


def test_the_task_survives_a_restore_as_the_same_object(env):
    """A `Task` is frozen and replaced rather than mutated, so a snapshot can
    hold the reference: forty of them down one run cost forty pointers."""
    state = snapshot(env)
    env.reset()
    assert env.task is not state.task
    restore(env, state)
    assert env.task is state.task
    assert len(state.task.puck_xy) == N_PUCKS
