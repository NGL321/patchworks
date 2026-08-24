"""The sandbox's contract: what a tick gives, what it takes, and what it keeps.

`docs/spec/03-the-sandbox.md`, *Sensory surface*, *Motor surface*, and *The
Gymnasium contract, made continual*.
"""

import mujoco
import numpy as np
import pytest

from patchworks.sandbox import (
    CONTROL_HZ,
    FRAME_SKIP,
    IMAGE_SIZE,
    HELDOUT_PAIRS,
    N_PUCKS,
    N_ZONES,
    SPAWN_R,
    SPLITS,
    ZONE_RADIUS,
    ZONE_XY,
    BlockedAnnulusError,
    PlanarPushSandbox,
    Task,
    friction_scale,
    in_heldout_sector,
    restore,
    snapshot,
)

TORQUE_LIMIT = np.array([3.0, 2.0, 1.0])
ZERO = np.zeros(3, np.float32)


@pytest.fixture
def env():
    e = PlanarPushSandbox(split="train")
    e.reset(seed=7, options={"reset_arm": True})
    yield e
    e.close()


@pytest.fixture
def fast_env():
    """Physics only. The agent needs the render; a 60 s probe does not."""
    e = PlanarPushSandbox(split="any", render_obs=False)
    yield e
    e.close()


# -- the sensory surface --------------------------------------------------------


def test_the_observation_contract(env):
    obs, *_ = env.step(ZERO)

    # Nothing here names a plane: these name joints and a camera. Going 3D adds
    # joints and turns gravity on; it does not rename anything in this dict.
    assert set(obs) == {"qpos", "qvel", "touch", "image"}

    assert obs["qpos"].shape == (3,) and obs["qpos"].dtype == np.float32
    assert obs["qvel"].shape == (3,) and obs["qvel"].dtype == np.float32
    assert obs["touch"].shape == (3,) and obs["touch"].dtype == np.float32
    assert np.array_equal(obs["touch"], env.data.sensordata[env._touch_adr])
    assert obs["image"].shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
    assert obs["image"].dtype == np.uint8
    assert obs in env.observation_space


def test_every_observation_stays_inside_the_declared_space(env):
    """A MuJoCo joint limit is a soft constraint, so the arm overshoots the
    arena's joint ranges; a qpos bound taken from them would be a promise the
    physics does not keep. Hammer the arm against its limits and check."""
    env.action_space.seed(0)
    for _ in range(400):
        obs, *_ = env.step(env.action_space.sample())
        assert obs in env.observation_space
    for _ in range(400):
        obs, *_ = env.step(np.ones(3, np.float32))
        assert obs in env.observation_space


def test_render_is_not_blanked_by_the_headless_flag():
    """`render_obs=False` is about the observation. A recorder asking for
    frames should get frames, not an all-black video."""
    env = PlanarPushSandbox(render_mode="rgb_array", render_obs=False)
    try:
        obs, _ = env.reset(seed=0, options={"reset_arm": True})
        assert not obs["image"].any()
        frame = env.render()
        assert frame.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
        assert frame.any()
    finally:
        env.close()


def test_render_returns_nothing_without_a_render_mode(env):
    assert env.render() is None


def test_no_object_pose_reaches_the_agent(env):
    """Everything about the world arrives through the render, unlabelled: the
    agent must learn that the coloured blobs are objects.

    Move a puck somewhere the arm cannot feel, and the image is the only thing
    in the observation that knows.
    """
    xy = np.array([[0.0, -0.25], [-0.22, 0.10], [0.10, 0.30]])
    env.reset(seed=4, options={"reset_arm": True, "task": Task(xy, np.zeros(3), 0, 0)})
    state = snapshot(env)
    untouched, *_ = env.step(ZERO)

    restore(env, state)
    env.perturb(0, [0.32, -0.30])  # well clear of the arm, which lies along +x
    moved, *_ = env.step(ZERO)

    for key in ("qpos", "qvel", "touch"):
        assert np.array_equal(untouched[key], moved[key])
    assert not np.array_equal(untouched["image"], moved["image"])


def test_the_goal_reaches_the_agent_as_perception(env):
    """The target zone lights up in the render. There is no goal vector, no
    task id, and no reward, so retargeting is just a change in appearance."""
    before = env.step(ZERO)[0]["image"].astype(int)
    other = next(z for z in range(3) if z != env.task.goal_zone)
    env.retarget(goal_zone=other)
    after = env.step(ZERO)[0]["image"].astype(int)
    assert np.abs(before - after).sum() > 0


def test_the_render_is_the_only_place_the_goal_appears(env):
    """A retarget changes the image and nothing else in the observation."""
    env.step(ZERO)
    before, *_ = env.step(ZERO)
    state = snapshot(env)
    other = next(z for z in range(3) if z != env.task.goal_zone)
    env.retarget(goal_zone=other)
    restore(env, state)
    env.retarget(goal_zone=other)
    after, *_ = env.step(ZERO)
    for key in ("qpos", "qvel", "touch"):
        assert np.array_equal(before[key], after[key])


# -- the motor surface ----------------------------------------------------------


def test_the_action_contract(env):
    assert env.action_space.shape == (3,)
    assert np.all(env.action_space.low == -1.0)
    assert np.all(env.action_space.high == 1.0)


def test_actions_are_normalised_to_the_per_joint_torque_limits(env):
    env.step(np.ones(3, np.float32))
    assert env.data.ctrl == pytest.approx(TORQUE_LIMIT)
    env.step(-np.ones(3, np.float32))
    assert env.data.ctrl == pytest.approx(-TORQUE_LIMIT)


def test_actions_outside_the_box_are_clipped(env):
    env.step(np.array([5.0, -5.0, 5.0], np.float32))
    assert env.data.ctrl == pytest.approx(TORQUE_LIMIT * [1, -1, 1])


def test_control_runs_at_fifty_hertz_over_ten_substeps(env):
    assert env.frame_skip == FRAME_SKIP
    before = env.data.time
    env.step(ZERO)
    assert env.data.time - before == pytest.approx(1.0 / CONTROL_HZ)
    assert env.model.opt.timestep * FRAME_SKIP == pytest.approx(1.0 / CONTROL_HZ)


# -- the three deviations -------------------------------------------------------


def test_there_is_no_reward_channel_and_no_episode(env):
    for _ in range(20):
        _, reward, terminated, truncated, _ = env.step(env.action_space.sample())
        assert reward == 0.0
        assert terminated is False
        assert truncated is False


def test_reset_rearranges_the_world_and_never_resets_the_agent(env):
    for _ in range(30):
        env.step(np.array([0.4, -0.3, 0.2], np.float32))
    qpos = env.data.qpos[env._arm_qadr].copy()
    qvel = env.data.qvel[env._arm_dofadr].copy()
    time_before = env.data.time
    poses = np.stack([env.puck_pose(i) for i in range(N_PUCKS)])

    env.reset()

    assert np.array_equal(env.data.qpos[env._arm_qadr], qpos)
    assert np.array_equal(env.data.qvel[env._arm_dofadr], qvel)
    assert env.data.time == time_before, "physics time is monotonic across the run"
    assert not np.allclose(np.stack([env.puck_pose(i) for i in range(N_PUCKS)]), poses)


def test_nothing_in_the_observation_announces_a_reset(env):
    """The agent finds out the world changed the way it finds out anything
    else: its predictions stop working."""
    before, *_ = env.step(ZERO)
    after, _ = env.reset()
    assert set(before) == set(after)
    for key in ("qpos", "qvel", "touch"):
        assert np.array_equal(before[key], after[key])


def test_reset_arm_is_available_for_setup(env):
    for _ in range(30):
        env.step(np.array([0.5, 0.5, -0.5], np.float32))
    assert np.any(env.data.qpos[env._arm_qadr] != 0.0)
    env.reset(options={"reset_arm": True})
    assert np.all(env.data.qpos[env._arm_qadr] == 0.0)
    assert np.all(env.data.qvel[env._arm_dofadr] == 0.0)


def test_info_is_privileged_and_carries_exactly_the_spec_s_four_things(env):
    _, _, _, _, info = env.step(ZERO)
    assert set(info) == {
        "puck_pose",
        "goal_puck",
        "goal_zone",
        "goal_distance",
        "goal_satisfied",
    }
    assert info["puck_pose"].shape == (N_PUCKS, 3)
    assert info["goal_puck"] in range(N_PUCKS)
    assert info["goal_zone"] in range(3)
    expected = np.linalg.norm(
        info["puck_pose"][info["goal_puck"], :2] - ZONE_XY[info["goal_zone"]]
    )
    assert info["goal_distance"] == pytest.approx(expected)
    assert isinstance(info["goal_satisfied"], bool)


def test_goal_satisfaction_is_a_gate_on_zone_radius(env):
    env.perturb(env.task.goal_puck, ZONE_XY[env.task.goal_zone])
    _, _, _, _, info = env.step(ZERO)
    assert info["goal_distance"] < ZONE_RADIUS
    assert info["goal_satisfied"] is True


# -- the sampler ----------------------------------------------------------------


def test_layouts_land_in_the_spawn_annulus_clear_of_each_other_and_of_the_zones(env):
    """Clear of each other subtracts both radii; clear of the zones means
    **centres**, with the puck's own radius ignored. The two are deliberately
    inconsistent -- see `docs/spec/03-the-sandbox.md`, *Two limitations of the
    sampler, on the record* -- so a puck may spawn with its rim already
    overlapping a zone's disc. Asserted as built, because it is recorded as
    built."""
    for _ in range(20):
        _, info = env.reset()
        xy = info["puck_pose"][:, :2]
        r = np.linalg.norm(xy, axis=1)
        assert np.all((r >= SPAWN_R[0] - 1e-9) & (r <= SPAWN_R[1] + 1e-9))
        for i in range(N_PUCKS):
            for j in range(i + 1, N_PUCKS):
                gap = np.linalg.norm(xy[i] - xy[j]) - env.puck_radius[i] - env.puck_radius[j]
                assert gap > 0.03
            for z in range(N_ZONES):
                assert np.linalg.norm(xy[i] - ZONE_XY[z]) > ZONE_RADIUS + 0.04


def _split_signatures(split: str, draws: int, seed: int = 3) -> set[tuple[bool, bool]]:
    """Which `(held-out pair, held-out sector)` combinations `split` draws."""
    e = PlanarPushSandbox(split=split, render_obs=False)
    try:
        e.reset(seed=seed, options={"reset_arm": True})
        tasks = [e.sample_task() for _ in range(draws)]
    finally:
        e.close()
    return {
        (task.pair in HELDOUT_PAIRS, in_heldout_sector(task.puck_xy[task.goal_puck]))
        for task in tasks
    }


def test_the_held_out_slice_is_two_axes_that_never_merge():
    assert set(SPLITS) == {"train", "heldout_pair", "heldout_sector", "any"}
    for split, expected in (
        ("train", {(False, False)}),
        ("heldout_pair", {(True, False)}),
        ("heldout_sector", {(False, True)}),
    ):
        assert _split_signatures(split, draws=8) == expected, split


def test_no_split_value_returns_the_union_of_the_two_axes():
    """A union value would draw tasks held out on *either* axis, mixed, and the
    number it produced would be attributable to neither: withholding two pairs
    leaves every puck, zone and region seen and withholds only compounds, while
    withholding a sector removes a target-puck position outright.

    So each of the three named splits withholds on **at most one** axis over
    all its draws -- a value that withheld on both, whether by mixing the two
    signatures or by drawing tasks held out on both at once, is what does not
    exist. `any` is the whole space rather than the union: it draws tasks held
    out on neither axis, which no union value could. Naming the axes separately
    is what makes the confound unconstructible; this is that, asserted."""
    for split in SPLITS:
        signatures = _split_signatures(split, draws=40)
        if split == "any":
            assert (False, False) in signatures
            continue
        axes = {
            name
            for flags in signatures
            for name, held in zip(("pair", "sector"), flags)
            if held
        }
        assert len(axes) <= 1, f"{split} withholds on both axes: {axes}"


def test_reset_raises_rather_than_hand_back_a_penetrating_layout(fast_env):
    """Falling through the rejection loop would deliver the exact failure the
    loop exists to prevent, silently. The arm is not reset by reset(), so a
    pose blocking the annulus is the caller's to move."""
    env = fast_env
    env.reset(seed=0, options={"reset_arm": True})
    # a layout that must intersect the arm: every draw is rejected
    poses = np.stack([env.puck_pose(i) for i in range(N_PUCKS)])
    task = env.task

    env._pucks_touching_arm = lambda: True
    # Named, not merely a RuntimeError: a caller looping over tasks has to tell
    # "move the arm" apart from "the env is broken" without matching a message.
    with pytest.raises(BlockedAnnulusError, match="clear of the arm"):
        env.reset()

    # the world is as it was found, not standing in the layout that was refused
    assert np.array_equal(np.stack([env.puck_pose(i) for i in range(N_PUCKS)]), poses)
    assert env.task is task


def test_calls_before_the_first_reset_say_what_is_missing():
    env = PlanarPushSandbox(render_obs=False)
    try:
        with pytest.raises(RuntimeError, match="call reset\\(\\) before"):
            env.step(ZERO)
        # and the refused tick did not happen: reset() cannot take one back,
        # since it resets neither the arm nor the clock
        assert env.data.time == 0.0
        assert np.all(env.data.ctrl == 0.0)
        assert np.all(env.data.qvel == 0.0)
        with pytest.raises(RuntimeError, match="call reset\\(\\) before"):
            env.retarget(goal_zone=1)
        with pytest.raises(RuntimeError, match="call reset\\(\\) before"):
            snapshot(env)
    finally:
        env.close()


def test_the_advertised_frame_rate_follows_frame_skip():
    """frame_skip is a knob; a recorder trusting stale metadata encodes the run
    at the wrong speed."""
    env = PlanarPushSandbox(render_obs=False)
    fast = PlanarPushSandbox(frame_skip=5, render_obs=False)
    try:
        assert env.metadata["render_fps"] == CONTROL_HZ
        assert fast.metadata["render_fps"] == 2 * CONTROL_HZ
        assert fast.metadata["render_modes"] == env.metadata["render_modes"]
    finally:
        env.close()
        fast.close()


def test_a_headless_env_never_builds_a_renderer():
    """The renderer needs a GL context a physics-only probe may not have."""
    env = PlanarPushSandbox(render_obs=False)
    try:
        env.reset(seed=0, options={"reset_arm": True})
        for _ in range(5):
            env.step(ZERO)
        assert env._renderer is None
    finally:
        env.close()


def test_a_layout_never_starts_inside_the_arm(env):
    """The arm is never reset, so a layout that intersects its pose starts the
    world inside a penetration and the solver launches a puck across the arena."""
    for _ in range(20):
        env.step(np.array([0.6, 0.4, -0.4], np.float32))
    for _ in range(10):
        env.reset()
        assert not env._pucks_touching_arm()


# -- the human's hand -----------------------------------------------------------


def test_the_arm_is_disturbed_by_an_impulse_never_by_a_teleport(env):
    """Displacing qpos would have the world rewrite the arm's configuration,
    which is the one thing this env never does. What lands instead is an
    impulse: the change in momentum is the impulse, at the named joint and
    nowhere else, so proprioception reports it the way it reports everything
    else and no new observation path exists."""
    qpos = env.data.qpos[env._arm_qadr].copy()
    qvel = env.data.qvel.copy()

    env.disturb_arm(1, 0.05)

    assert np.array_equal(env.data.qpos[env._arm_qadr], qpos)
    momentum = np.zeros(env.model.nv)
    mujoco.mj_mulM(env.model, env.data, momentum, env.data.qvel - qvel)
    expected = np.zeros(env.model.nv)
    expected[env._arm_dofadr[1]] = 0.05
    assert momentum == pytest.approx(expected, abs=1e-9)


def test_the_three_hands_are_callable_headlessly(fast_env):
    """The live viewer binds them to ctrl-drags, but the acceptance demo's
    instrumentation drives them from a script and may have no GL context."""
    env = fast_env
    env.reset(seed=4, options={"reset_arm": True})

    env.disturb_arm(0, 0.02)
    env.perturb(1, [0.18, -0.22])
    env.retarget(goal_puck=1, goal_zone=2)
    env.step(ZERO)

    assert env._renderer is None
    assert env.puck_pose(1)[:2] == pytest.approx([0.18, -0.22], abs=1e-3)
    assert (env.task.goal_puck, env.task.goal_zone) == (1, 2)


def test_the_hands_refuse_an_index_that_is_not_a_puck_or_a_joint(env):
    """A negative index wrapped silently: retarget(goal_zone=-1) dimmed every
    zone -- taking the goal out of the only channel the agent has -- while
    info went on measuring the distance to the last one."""
    for bad in (-1, N_PUCKS):
        with pytest.raises(ValueError, match="range"):
            env.retarget(goal_zone=bad)
        with pytest.raises(ValueError, match="range"):
            env.retarget(goal_puck=bad)
        with pytest.raises(ValueError, match="range"):
            env.perturb(bad, [0.2, 0.2])
        with pytest.raises(ValueError, match="range"):
            env.disturb_arm(bad, 0.01)


def test_perturb_teleports_a_puck_by_writing_qpos(env):
    """And leaves the applied-force fields zero, which is not decoration.

    `qfrc_applied` and `xfrc_applied` are inert in this arena *only* because
    this hand teleports. Implemented as an applied force instead, the force
    would become state -- and a snapshot that enumerated fields rather than
    naming `mjSTATE_INTEGRATION` would drop it, silently, in a file nobody
    would think to reread."""
    target = np.array([0.10, -0.28])
    env.perturb(1, target)
    assert env.puck_pose(1)[:2] == pytest.approx(target)
    dof = env._puck_dofadr[1]
    assert np.all(env.data.qvel[dof : dof + 3] == 0.0)
    assert np.all(env.data.qfrc_applied == 0.0)
    assert np.all(env.data.xfrc_applied == 0.0)


def test_perturb_brings_the_friction_field_with_the_puck(fast_env):
    """The third path that moves a puck, and the field is a function of where
    a puck is -- a read taken before the next tick must not be of where it was."""
    env = fast_env
    xy = np.array([[0.0, -0.25], [-0.22, 0.10], [0.10, 0.30]])
    env.reset(seed=1, options={"reset_arm": True, "task": Task(xy, np.zeros(3), 0, 0)})
    dof = env._puck_dofadr[0]

    env.perturb(0, [0.40, -0.30])
    nominal = env._friction_nominal[0]
    assert env.model.dof_frictionloss[dof : dof + 3] == pytest.approx(
        nominal * friction_scale([0.40, -0.30])
    )


def test_retarget_changes_what_is_wanted_without_touching_the_world(env):
    poses = np.stack([env.puck_pose(i) for i in range(N_PUCKS)])
    puck = (env.task.goal_puck + 1) % N_PUCKS
    zone = (env.task.goal_zone + 1) % 3
    env.retarget(goal_puck=puck, goal_zone=zone)
    assert (env.task.goal_puck, env.task.goal_zone) == (puck, zone)
    assert np.array_equal(np.stack([env.puck_pose(i) for i in range(N_PUCKS)]), poses)


# -- the friction field ---------------------------------------------------------


def test_the_friction_field_has_mean_one_and_range_three_quarters_to_five_fourths():
    grid = np.linspace(-0.52, 0.52, 301)
    xs, ys = np.meshgrid(grid, grid)
    inside = xs**2 + ys**2 <= 0.52**2
    values = np.array([friction_scale((x, y)) for x, y in zip(xs[inside], ys[inside])])
    assert values.mean() == pytest.approx(1.0, abs=0.01)
    assert values.min() >= 0.75
    assert values.max() <= 1.25
    # and it uses the range it is given, rather than hugging the mean
    assert values.min() < 0.80 and values.max() > 1.20


def test_the_friction_field_is_a_pure_function_of_puck_position(fast_env):
    """Not a random draw: resampling per reset() would put a number into the
    model that mjSTATE_INTEGRATION does not cover, so restore would diverge."""
    env = fast_env
    xy = np.array([[0.20, 0.10], [-0.18, -0.22], [0.05, 0.30]])
    env.reset(seed=2, options={"reset_arm": True, "task": Task(xy, np.zeros(3), 0, 0)})
    env.step(ZERO)
    seen = [env.model.dof_frictionloss[d : d + 3].copy() for d in env._puck_dofadr]
    for i, values in enumerate(seen):
        assert values == pytest.approx(env._friction_nominal[i] * friction_scale(xy[i]))

    # the same position gives the same scale, anywhere in the run
    env.perturb(0, [0.40, -0.30])
    env.step(ZERO)
    env.perturb(0, xy[0])
    env.step(ZERO)
    assert env.model.dof_frictionloss[env._puck_dofadr[0] : env._puck_dofadr[0] + 3] == (
        pytest.approx(seen[0])
    )


def test_the_same_push_at_two_places_gives_two_outcomes(fast_env):
    """Repeated identical pushes in a rigid-body simulator are bit-identical;
    the field is what makes this world's not be."""
    env = fast_env
    travelled = []
    for centre in ([0.18, 0.06], [-0.06, -0.30]):
        xy = np.array([centre, [0.30, 0.25], [-0.30, 0.20]])
        env.reset(seed=5, options={"reset_arm": True, "task": Task(xy, np.zeros(3), 0, 0)})
        dof = env._puck_dofadr[0]
        env.data.qfrc_applied[dof] = 0.15
        start = env.puck_pose(0)[:2].copy()
        for _ in range(50):
            env.step(ZERO)
        env.data.qfrc_applied[dof] = 0.0
        travelled.append(float(np.linalg.norm(env.puck_pose(0)[:2] - start)))
    assert abs(travelled[0] - travelled[1]) > 1e-3


# -- static equilibrium below threshold -----------------------------------------


@pytest.mark.parametrize("puck", range(N_PUCKS))
def test_a_sixty_second_sub_threshold_hold_drifts_under_a_millimetre(fast_env, puck):
    """A frictionloss constraint takes its impedance from solimp[0]; at the
    default 0.9 a puck held at 90% of its break-away threshold crept 159 mm
    over 60 s. solimpfriction at 0.9999 is what makes static equilibrium below
    threshold a claim this sandbox can make."""
    env = fast_env
    # keep every puck off the +x ray the reset arm lies along
    xy = np.array([[0.0, -0.25], [-0.22, 0.10], [0.10, 0.30]])
    env.reset(seed=1, options={"reset_arm": True, "task": Task(xy, np.zeros(3), puck, 0)})
    assert env.data.ncon == 0

    env._apply_friction_field()
    dof = env._puck_dofadr[puck]
    env.data.qfrc_applied[dof] = 0.9 * env.model.dof_frictionloss[dof]

    start = env.puck_pose(puck)[:2].copy()
    for _ in range(int(60.0 * CONTROL_HZ)):
        env.step(ZERO)
    drift = float(np.linalg.norm(env.puck_pose(puck)[:2] - start))
    assert drift < 1e-3, f"puck {puck} drifted {drift * 1000:.3f} mm"


@pytest.mark.parametrize("puck", range(N_PUCKS))
def test_the_break_away_threshold_at_the_joint_is_the_frictionloss_value(fast_env, puck):
    """Above it the puck moves, which is what makes the hold above a claim
    about the threshold rather than about a stuck solver."""
    env = fast_env
    xy = np.array([[0.0, -0.25], [-0.22, 0.10], [0.10, 0.30]])
    env.reset(seed=1, options={"reset_arm": True, "task": Task(xy, np.zeros(3), puck, 0)})
    env._apply_friction_field()
    dof = env._puck_dofadr[puck]
    env.data.qfrc_applied[dof] = 1.5 * env.model.dof_frictionloss[dof]
    start = env.puck_pose(puck)[:2].copy()
    for _ in range(int(1.0 * CONTROL_HZ)):
        env.step(ZERO)
    assert float(np.linalg.norm(env.puck_pose(puck)[:2] - start)) > 0.05
