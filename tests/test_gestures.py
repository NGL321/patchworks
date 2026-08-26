"""The hands, bound in the live viewer (ticket #96).

Most of this ticket is a human at a mouse, so the seam is where the tests are.
:class:`~patchworks.surface.gestures.Gestures` is *gesture in, hand out* and
:class:`~patchworks.surface.gestures.Pointer` is *perturb struct in, one whole
gesture out*; both are plain numbers at the door and both are driven here
without a window. What is left over -- that MuJoCo reports a pick and a drag
the way this reads them -- is checked against a stand-in for the passive
viewer's handle, and rests in the end on the window being driven by hand.
"""

import contextlib
import warnings

import mujoco
import numpy as np
import pytest
import torch

from patchworks.agent import Agent
from patchworks.graph import build_graph
from patchworks.sandbox import (
    N_PUCKS,
    N_ZONES,
    ZONE_RADIUS,
    ZONE_XY,
    BlockedAnnulusError,
    PlanarPushSandbox,
)
from patchworks.sandbox.env import ARM_JOINTS
from patchworks.surface import (
    Drag,
    EventKind,
    Gestures,
    Hands,
    Pointer,
    Recorder,
    Referent,
    ReferentKind,
    drive,
)
from patchworks.surface.gestures import (
    IMPULSE_PER_METRE,
    MINIMUM_DRAG,
    OUT_OF_PLANE_TOLERANCE,
    TOP_DOWN_ELEVATION,
    hold_top_down,
)


@pytest.fixture(scope="module")
def dome():
    return build_graph()


@pytest.fixture(scope="module")
def env():
    world = PlanarPushSandbox(split="any")
    yield world
    world.close()


@pytest.fixture(scope="module")
def agent(env, dome):
    """One agent for the file: nothing here reads a cell, so nothing here needs
    a fresh one, and building a dome's worth of tables per test would cost more
    than every gesture in it."""
    return Agent(env, dome=dome, generator=torch.Generator().manual_seed(0))


@pytest.fixture
def recorder(env, agent):
    """A fresh recorder, over a world put back the way each test finds it.

    `reset_arm` is the one place this asks for something the run-time env never
    does: a test that reasons about which way a joint swings needs the arm at a
    pose it can name, and the previous test may have shoved it.
    """
    observation, _info = env.reset(seed=0, options={"reset_arm": True})
    agent.observe(observation)
    return Recorder(agent, every=1)


@pytest.fixture
def gestures(recorder):
    return Gestures(Hands(recorder))


def body(env, name):
    return mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, name)


def arm_qpos(env):
    """The arm's three joint angles, by name rather than by position."""
    joints = mujoco.mjtObj.mjOBJ_JOINT
    return np.array(
        [
            env.data.qpos[env.model.jnt_qposadr[mujoco.mj_name2id(env.model, joints, name)]]
            for name in ARM_JOINTS
        ]
    )


def pucks(env):
    return np.stack([env.puck_pose(i) for i in range(N_PUCKS)])


def dragged(env, name, moved, grip=None):
    """A completed drag on `name`, pulled `moved` from wherever it is now."""
    grabbed = body(env, name)
    origin = np.array(env.data.xpos[grabbed])
    return Drag(
        body=grabbed,
        grip=np.array(origin if grip is None else grip, dtype=float),
        origin=origin,
        moved=np.array(moved, dtype=float),
    )


def tick(recorder, times=1):
    """One whole tick, watched, the way the viewer's loop drives it."""
    records = []
    for _ in range(times):
        recorder.agent.tick()
        record = recorder.observe()
        if record is not None:
            records.append(record)
    return records


class TestTheReferentAloneDecidesWhichHand:
    """`08`'s first two events, which need no new binding to tell apart."""

    def test_a_ctrl_drag_on_a_link_disturbs_the_arm(self, gestures, env):
        before = np.array(env.data.qvel)
        event = gestures.drag(dragged(env, "link1", (0.0, 0.05, 0.0), grip=(0.28, 0.0, 0.0)))
        assert event.kind is EventKind.DISTURB_ARM
        assert event.detail[0] == 1
        assert not np.array_equal(env.data.qvel, before)

    def test_a_ctrl_drag_on_a_puck_perturbs_it(self, gestures, env):
        was = np.array(env.data.xpos[body(env, "puck_2")][:2])
        event = gestures.drag(dragged(env, "puck_2", (0.04, 0.03, 0.0)))
        assert event.kind is EventKind.PERTURB
        assert env.puck_pose(2)[:2] == pytest.approx(was + [0.04, 0.03], abs=1e-9)

    def test_the_same_drag_on_two_referents_is_two_different_hands(self, gestures, env):
        """The whole of the criterion: only *what was grabbed* differs."""
        pull = (0.0, 0.05, 0.0)
        link = gestures.drag(dragged(env, "link0", pull, grip=(0.1, 0.0, 0.0)))
        puck = gestures.drag(dragged(env, "puck_0", pull))
        assert (link.kind, puck.kind) == (EventKind.DISTURB_ARM, EventKind.PERTURB)

    def test_a_drag_on_neither_fires_no_hand_and_leaves_no_marker(
        self, gestures, recorder, env
    ):
        assert gestures.drag(dragged(env, "base", (0.05, 0.0, 0.0))) is None
        assert recorder.pending == ()

    def test_a_puck_is_placed_by_the_drag_and_not_at_the_grip(self, gestures, env):
        """Grabbing a puck by its rim and dropping it does not put the rim there."""
        centre = np.array(env.data.xpos[body(env, "puck_0")])
        rim = centre + [env.puck_radius[0], 0.0, 0.0]
        gestures.drag(dragged(env, "puck_0", (0.06, 0.0, 0.0), grip=rim))
        assert env.puck_pose(0)[:2] == pytest.approx(centre[:2] + [0.06, 0.0], abs=1e-9)


class TestWhatADragOnALinkDelivers:
    """An impulse, never a teleport: the world moves the body."""

    def test_the_impulse_is_the_drag_along_the_way_that_joint_swings(self, gestures, env):
        # Nothing has moved the arm, so link0 lies along +x and joint 0 swings a
        # point at (0.1, 0, 0) straight along +y.
        assert arm_qpos(env) == pytest.approx([0.0, 0.0, 0.0])
        event = gestures.drag(dragged(env, "link0", (0.0, 0.06, 0.0), grip=(0.1, 0.0, 0.0)))
        assert event.detail[1] == pytest.approx(0.06 * IMPULSE_PER_METRE)

    def test_a_drag_the_other_way_reverses_it(self, gestures, env):
        event = gestures.drag(dragged(env, "link0", (0.0, -0.06, 0.0), grip=(0.1, 0.0, 0.0)))
        assert event.detail[1] == pytest.approx(-0.06 * IMPULSE_PER_METRE)

    def test_the_drag_is_scaled_by_the_feel_constant(self, recorder, env):
        """The constant is 1.0, so every other assertion here would hold with
        the scaling dropped entirely. This is the one that would not: a hand
        built with a different feel delivers a proportionally different shove
        from the very same pull."""
        pull = dragged(env, "link0", (0.0, 0.06, 0.0), grip=(0.1, 0.0, 0.0))
        default = Gestures(Hands(recorder)).drag(pull)
        firmer = Gestures(Hands(recorder), impulse_per_metre=4.0).drag(pull)
        assert default.detail[1] == pytest.approx(0.06 * IMPULSE_PER_METRE)
        assert firmer.detail[1] == pytest.approx(0.06 * 4.0)
        assert firmer.detail[1] == pytest.approx(4.0 * default.detail[1])

    def test_a_pull_along_the_link_delivers_nothing(self, gestures, env):
        """A hinge has no answer to a pull with no moment arm, and says zero."""
        event = gestures.drag(dragged(env, "link0", (0.06, 0.0, 0.0), grip=(0.1, 0.0, 0.0)))
        assert event.detail[1] == pytest.approx(0.0, abs=1e-12)

    def test_a_grip_on_the_joints_own_axis_delivers_zero_and_does_not_raise(
        self, gestures, env
    ):
        """No moment arm, so no impulse -- and a number rather than an
        exception. `drive` reads this under the viewer's lock, so a raised
        `ZeroDivisionError` would come out through the generator and end the
        session over one badly-aimed click."""
        joint = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, ARM_JOINTS[0])
        on_the_axis = np.array(env.data.xanchor[joint])
        event = gestures.drag(
            dragged(env, "link0", (0.0, 0.06, 0.0), grip=on_the_axis)
        )
        assert event.kind is EventKind.DISTURB_ARM
        assert event.detail[1] == pytest.approx(0.0, abs=1e-12)

    def test_a_drag_too_short_to_be_one_fires_nothing(self, gestures, recorder, env):
        short = MINIMUM_DRAG / 2
        assert gestures.drag(dragged(env, "link0", (0.0, short, 0.0), grip=(0.1, 0.0, 0.0))) is None
        assert gestures.drag(dragged(env, "puck_0", (short, 0.0, 0.0))) is None
        assert recorder.pending == ()


class TestADragIsReadInThePlane:
    """#123's first lever: an out-of-plane drag is refused, not projected.

    The world has zero z degrees of freedom (`src/patchworks/sandbox/arena.xml`),
    so a drag with a z in it names nothing this world can do. #96 said so in a
    docstring and gated on the *planar* magnitude instead, which is a different
    set: a drag that was mostly z with a millimetre of planar residue passed
    that gate and fired a hand with the residue as its argument. That is the
    bug this class pins -- the gate is on the out-of-plane component now.

    The second lever, the camera held top-down, is
    :class:`TestTheCameraIsHeldTopDown`. Neither of them can say whether the
    gesture *feels* right, which is a human at the window.
    """

    def test_a_drag_in_the_plane_fires_its_hand(self, gestures, env):
        """The gesture the demo is made of, unchanged: z is exactly zero when
        the view is top-down, and that drag still fires."""
        event = gestures.drag(dragged(env, "puck_0", (0.05, 0.02, 0.0)))
        assert event.kind is EventKind.PERTURB

    def test_a_mostly_out_of_plane_drag_fires_nothing(self, gestures, recorder, env):
        """**The reported bug.** Half a metre of z, two millimetres of planar
        residue: enough planar travel to clear :data:`MINIMUM_DRAG`, and the
        residue is the "random direction" the human saw the puck take."""
        before = np.array(env.puck_pose(0)[:2])
        with pytest.warns(UserWarning, match="out of the plane"):
            assert gestures.drag(dragged(env, "puck_0", (0.002, 0.0, 0.5))) is None
        assert env.puck_pose(0)[:2] == pytest.approx(before, abs=1e-12)

        before = np.array(env.data.qvel)
        with pytest.warns(UserWarning, match="out of the plane"):
            pull = dragged(env, "link0", (0.0, 0.002, 0.4), grip=(0.1, 0.0, 0.0))
            assert gestures.drag(pull) is None
        assert env.data.qvel == pytest.approx(before)
        assert recorder.pending == ()

    def test_a_purely_out_of_plane_drag_fires_nothing(self, gestures, recorder, env):
        """The case #96's docstring already claimed, now enforced rather than
        reached by way of the planar magnitude being zero."""
        with pytest.warns(UserWarning, match="out of the plane"):
            assert gestures.drag(dragged(env, "puck_0", (0.0, 0.0, 0.5))) is None
        with pytest.warns(UserWarning, match="out of the plane"):
            pull = dragged(env, "link0", (0.0, 0.0, 0.5), grip=(0.1, 0.0, 0.0))
            assert gestures.drag(pull) is None
        assert recorder.pending == ()

    def test_the_refusal_says_why_and_names_both_components(self, gestures, env):
        """A human who is told nothing is left wondering, which is the
        complaint. The numbers are in the message because they are what makes
        one refusal different from the next -- and, incidentally, what stops
        the default warning filter from showing only the first."""
        with pytest.warns(UserWarning) as refusals:
            gestures.drag(dragged(env, "puck_0", (0.002, 0.0, 0.5)))
            gestures.drag(dragged(env, "puck_0", (0.004, 0.0, 0.3)))
        said = [str(refusal.message) for refusal in refusals]
        assert len(said) == 2 and said[0] != said[1]
        for message in said:
            assert "no hand fired" in message
            # And what to do instead, which is the shifted drag: the plain one
            # is MuJoCo's vertical-plane translate, and no view makes it planar.
            assert "shift" in message

    def test_a_drag_that_moved_in_neither_is_short_rather_than_refused(
        self, gestures, recorder, env
    ):
        """A press that went nowhere is not an out-of-plane gesture: it is not
        a gesture. It fires nothing and says nothing -- including MuJoCo's
        rotating drag, which arrives here as a displacement of zero."""
        short = MINIMUM_DRAG / 2
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert gestures.drag(dragged(env, "puck_0", (0.0, 0.0, short))) is None
            assert gestures.drag(dragged(env, "puck_0", (0.0, 0.0, 0.0))) is None
        assert recorder.pending == ()

    def test_a_hand_never_fires_on_a_planar_residue_below_the_minimum(
        self, gestures, recorder, env
    ):
        """Whatever the z, what reaches a hand cleared :data:`MINIMUM_DRAG` in
        the plane. The two gates leave no gap between them for the residue that
        started this ticket to come through."""
        residue = MINIMUM_DRAG / 2
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            for z in (0.0, MINIMUM_DRAG, 0.05, 0.5, 5.0):
                assert gestures.drag(dragged(env, "puck_0", (residue, 0.0, z))) is None
        assert recorder.pending == ()

    def test_the_tolerance_is_an_angle_off_the_plane(self, gestures, env):
        """A ratio, not a length: a drag is refused for the *direction* it went
        in, so the same tilt of the view refuses a long drag and a short one
        alike."""
        just_inside = OUT_OF_PLANE_TOLERANCE * 0.9
        for planar in (0.01, 0.5):
            inside = gestures.drag(
                dragged(env, "puck_0", (planar, 0.0, planar * just_inside))
            )
            assert inside is not None and inside.kind is EventKind.PERTURB
            with pytest.warns(UserWarning, match="out of the plane"):
                outside = gestures.drag(
                    dragged(env, "puck_0", (planar, 0.0, planar * 2.0))
                )
            assert outside is None

    def test_a_drag_on_nothing_is_a_miss_and_not_a_refusal(
        self, gestures, recorder, env
    ):
        """No hand takes the table, whatever plane the pull was in, so there is
        nothing to refuse and nobody to tell: a message saying "no hand fired"
        would blame the pull for what was an aim."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert gestures.drag(dragged(env, "base", (0.002, 0.0, 0.5))) is None
        assert recorder.pending == ()

    def test_the_refusal_lifts_in_one_place_for_a_world_with_a_third_dimension(
        self, recorder, env
    ):
        """Relaxable, not welded shut (#123). A world whose pucks could move in
        z would hand its own tolerance in, and the drag that is refused here
        fires there -- with its planar part, which is all either hand takes."""
        pull = dragged(env, "puck_0", (0.002, 0.0, 0.5))
        relaxed = Gestures(Hands(recorder), out_of_plane_tolerance=float("inf"))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            event = relaxed.drag(pull)
        assert event is not None and event.kind is EventKind.PERTURB

    def test_a_lifted_refusal_still_hands_no_hand_a_drag_of_nothing(
        self, recorder, env
    ):
        """The minimum survives the relaxation. `inf * 0.0` is `nan` and every
        comparison with one is False, so a straight-up drag would otherwise
        fall through a lifted refusal and teleport a puck to where it already
        is -- with a marker onset latency would then be measured from, which is
        the whole reason :data:`MINIMUM_DRAG` exists."""
        was = np.array(env.puck_pose(0)[:2])
        relaxed = Gestures(Hands(recorder), out_of_plane_tolerance=float("inf"))
        assert relaxed.drag(dragged(env, "puck_0", (0.0, 0.0, 0.5))) is None
        assert recorder.pending == ()
        assert env.puck_pose(0)[:2] == pytest.approx(was, abs=1e-12)


class TestTheDragMujocoHandsOver:
    """Where a drag's z actually comes from, asked of MuJoCo (#123).

    #123 was filed on the reading that a drag carries a z when the camera is
    off top-down, and that a top-down view therefore has none. **It is not the
    view.** MuJoCo's plain ctrl-drag is `mjMOUSE_MOVE_V`, which translates the
    grabbed point in the *vertical* plane -- the mouse's own vertical axis is
    world z -- at every elevation, top-down included; the **shifted** one is
    `mjMOUSE_MOVE_H`, which is planar at every elevation. The window's own help
    table says as much in one line: *Object Translate: Ctrl [Shift] right drag*.

    So the human's up-the-screen pull is the z, and no camera constraint can
    take it out of one. That is why :meth:`Gestures.drag`'s refusal is the lever
    that does the work here, and why its message names the shift rather than the
    view. Pinned in the suite because the whole refusal rests on it and because
    it is a fact about MuJoCo 3.10 that a version bump could move.

    Nothing here is a stand-in for MuJoCo's picking: this calls MuJoCo's own
    `mjv_movePerturb` with MuJoCo's own scene, which is what the viewer does
    with the mouse on its UI thread.
    """

    def translated(self, env, action, mouse, elevation=TOP_DOWN_ELEVATION):
        """`refselpos`'s travel for one mouse move, as MuJoCo computes it."""
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = [0.0, 0.0, 0.0]
        camera.distance = 1.6
        camera.azimuth = 90.0
        camera.elevation = elevation
        perturb = mujoco.MjvPerturb()
        perturb.select = body(env, "puck_0")
        perturb.active = mujoco.mjtPertBit.mjPERT_TRANSLATE
        perturb.localpos[:] = [0.0, 0.0, 0.0]
        scene = mujoco.MjvScene(env.model, maxgeom=1000)
        mujoco.mjv_updateScene(
            env.model,
            env.data,
            mujoco.MjvOption(),
            perturb,
            camera,
            mujoco.mjtCatBit.mjCAT_ALL,
            scene,
        )
        mujoco.mjv_initPerturb(env.model, env.data, scene, perturb)
        before = np.array(perturb.refselpos)
        mujoco.mjv_movePerturb(env.model, env.data, action, *mouse, scene, perturb)
        return np.array(perturb.refselpos) - before

    def test_a_plain_drag_up_the_screen_is_pure_z_even_top_down(self, env):
        """The reported bug, at its source: the view is straight down and the
        pull still names a place this world cannot put anything."""
        moved = self.translated(env, mujoco.mjtMouse.mjMOUSE_MOVE_V, (0.0, 0.1))
        assert abs(moved[2]) > 0.1
        assert np.linalg.norm(moved[:2]) == pytest.approx(0.0, abs=1e-12)

    def test_a_plain_drag_across_the_screen_is_planar(self, env):
        """Which is why the gesture works at all, and why what the human sees
        is a puck that follows sideways and refuses to follow upwards."""
        moved = self.translated(env, mujoco.mjtMouse.mjMOUSE_MOVE_V, (0.1, 0.0))
        assert np.linalg.norm(moved[:2]) > 0.1
        assert moved[2] == pytest.approx(0.0, abs=1e-12)

    def test_the_view_is_not_what_puts_the_z_in(self, env):
        """Tilt the camera as far as you like: the plain drag's z is the
        mouse's vertical axis, and it is world z at every elevation."""
        for elevation in (TOP_DOWN_ELEVATION, -80.0, -45.0, 0.0):
            moved = self.translated(
                env, mujoco.mjtMouse.mjMOUSE_MOVE_V, (0.07, 0.07), elevation
            )
            planar = float(np.linalg.norm(moved[:2]))
            assert abs(moved[2]) == pytest.approx(planar, rel=1e-6), elevation

    def test_the_shifted_drag_is_planar_whatever_the_view(self, env):
        """`mjMOUSE_MOVE_H`: the one MuJoCo's help puts the shift in brackets
        for, and the one this world's gesture actually wants."""
        for elevation in (TOP_DOWN_ELEVATION, -80.0, -45.0, 0.0):
            for mouse in ((0.1, 0.0), (0.0, 0.1), (0.07, 0.07)):
                moved = self.translated(
                    env, mujoco.mjtMouse.mjMOUSE_MOVE_H, mouse, elevation
                )
                assert np.linalg.norm(moved[:2]) > 0.09, (elevation, mouse)
                assert moved[2] == pytest.approx(0.0, abs=1e-12), (elevation, mouse)

    def test_the_gesture_layer_fires_a_hand_on_what_a_shifted_drag_hands_it(
        self, gestures, env
    ):
        """End to end on MuJoCo's own numbers: the shifted drag clears the
        refusal and moves the puck by what the mouse asked for."""
        moved = self.translated(env, mujoco.mjtMouse.mjMOUSE_MOVE_H, (0.07, 0.07))
        was = np.array(env.puck_pose(0)[:2])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            event = gestures.drag(dragged(env, "puck_0", moved))
        assert event.kind is EventKind.PERTURB
        assert env.puck_pose(0)[:2] == pytest.approx(was + moved[:2], abs=1e-9)


def zone_point(zone, offset=0.0):
    return np.array([ZONE_XY[zone][0] + offset, ZONE_XY[zone][1], 0.0])


class TestClickThenClick:
    """`08` event 3: point at the thing, then point at where you want it."""

    def test_a_puck_then_a_zone_retargets(self, gestures, env):
        puck = body(env, "puck_1")
        assert gestures.pick(puck, env.data.xpos[puck]) is None
        assert gestures.pointed_at == 1
        event = gestures.pick(0, zone_point(2))
        assert event.kind is EventKind.RETARGET
        assert event.detail == (1.0, 2.0)
        assert (env.task.goal_puck, env.task.goal_zone) == (1, 2)
        assert gestures.pointed_at is None

    def test_a_zone_on_its_own_is_not_half_a_gesture(self, gestures, recorder):
        assert gestures.pick(0, zone_point(0)) is None
        assert recorder.pending == ()

    def test_a_click_on_nothing_forgets_the_puck(self, gestures, recorder, env):
        gestures.pick(body(env, "puck_0"), env.data.xpos[body(env, "puck_0")])
        assert gestures.pick(0, (0.45, 0.45, 0.0)) is None
        assert gestures.pointed_at is None
        assert gestures.pick(0, zone_point(1)) is None
        assert recorder.pending == ()

    def test_a_click_on_a_link_forgets_the_puck(self, gestures, env):
        gestures.pick(body(env, "puck_0"), env.data.xpos[body(env, "puck_0")])
        gestures.pick(body(env, "link2"), (0.4, 0.0, 0.0))
        assert gestures.pointed_at is None

    def test_the_zone_is_whichever_one_the_click_landed_in(self, gestures):
        assert gestures.referent(0, zone_point(1, ZONE_RADIUS * 0.9)) == Referent(
            ReferentKind.ZONE, 1
        )
        assert (
            gestures.referent(0, zone_point(1, ZONE_RADIUS * 1.1)).kind
            is ReferentKind.NOTHING
        )

    def test_the_last_puck_pointed_at_is_the_one_that_counts(self, gestures, env):
        for puck in (0, 2):
            named = body(env, f"puck_{puck}")
            gestures.pick(named, env.data.xpos[named])
        event = gestures.pick(0, zone_point(0))
        assert event.detail == (2.0, 0.0)


class TestTheKeys:
    def test_r_rearranges_the_world_and_leaves_the_arm_where_it_is(self, gestures, env):
        env.data.qpos[:3] = [0.4, -0.3, 0.2]
        env.data.qvel[:3] = [0.1, 0.2, 0.3]
        mujoco.mj_forward(env.model, env.data)
        moved, before = arm_qpos(env), pucks(env)
        gestures.key(ord("R"))
        assert not np.allclose(before, pucks(env))
        assert arm_qpos(env) == pytest.approx(moved)
        assert env.data.qvel[:3] == pytest.approx([0.1, 0.2, 0.3])

    def test_rearranging_is_setup_and_leaves_no_marker(self, gestures, recorder):
        assert gestures.key(ord("r")) is None
        assert recorder.pending == ()

    def test_the_number_keys_cycle_the_three_by_three_pairs(self, gestures, env):
        seen = []
        for key in range(ord("1"), ord("1") + N_PUCKS * N_ZONES):
            event = gestures.key(key)
            assert event.kind is EventKind.RETARGET
            seen.append((env.task.goal_puck, env.task.goal_zone))
        # Spelled out rather than rebuilt from `N_PUCKS`/`N_ZONES`: they are
        # both 3, so a comprehension over them restates `divmod` and would
        # agree with a column-major reading of the very same grid.
        assert seen == [
            (0, 0), (0, 1), (0, 2),
            (1, 0), (1, 1), (1, 2),
            (2, 0), (2, 1), (2, 2),
        ]

    def test_a_pair_off_the_grid_is_refused(self, gestures):
        with pytest.raises(ValueError, match="0 .. 8"):
            gestures.pair(N_PUCKS * N_ZONES)

    def test_a_key_nothing_is_bound_to_does_nothing(self, gestures, recorder, env):
        before = pucks(env)
        assert gestures.key(ord("Q")) is None
        assert gestures.key(ord("0")) is None
        assert recorder.pending == ()
        assert np.array_equal(before, pucks(env))


class TestEveryHandWritesItsMarker:
    def test_each_gesture_lands_in_the_record_at_the_tick_it_fired(
        self, gestures, recorder, env
    ):
        tick(recorder, 2)  # primes the delta and gets a capture out of the way
        at = recorder.agent.sheaf.ticks
        gestures.drag(dragged(env, "link0", (0.0, 0.05, 0.0), grip=(0.1, 0.0, 0.0)))
        gestures.drag(dragged(env, "puck_0", (0.03, 0.0, 0.0)))
        gestures.pick(body(env, "puck_0"), env.data.xpos[body(env, "puck_0")])
        gestures.pick(0, zone_point(0))
        (record,) = tick(recorder)
        assert [event.kind for event in record.events] == [
            EventKind.DISTURB_ARM,
            EventKind.PERTURB,
            EventKind.RETARGET,
        ]
        assert {event.tick for event in record.events} == {at}


class TestThePointerReadsWholeGestures:
    """The perturb struct is a state; a gesture is an act. This is the seam."""

    def grip(self, env, name):
        return np.array(env.data.xpos[body(env, name)])

    def test_a_drag_fires_once_on_release_and_not_once_a_tick(
        self, gestures, recorder, env
    ):
        pointer = Pointer(gestures)
        puck, start = body(env, "puck_0"), self.grip(env, "puck_0")
        reach = start
        for step in range(6):
            assert pointer.sample(1, puck, (0.0, 0.0, 0.0), reach) is None
            reach = start + [0.01 * step, 0.0, 0.0]
        event = pointer.sample(0, puck, (0.0, 0.0, 0.0), reach)
        assert event.kind is EventKind.PERTURB
        assert len(recorder.pending) == 1

    def test_the_release_uses_where_the_drag_ended(self, gestures, env):
        pointer = Pointer(gestures)
        puck, start = body(env, "puck_1"), self.grip(env, "puck_1")
        for reach in (0.0, 0.02, 0.09):
            pointer.sample(1, puck, (0.0, 0.0, 0.0), start + [reach, 0.0, 0.0])
        pointer.sample(0, puck, (0.0, 0.0, 0.0), start + [0.09, 0.0, 0.0])
        assert env.puck_pose(1)[:2] == pytest.approx(start[:2] + [0.09, 0.0], abs=1e-9)

    def test_a_drag_the_mouse_never_made_fires_nothing(self, gestures, recorder, env):
        """The body moves under a held drag. That is the agent, not the human."""
        pointer = Pointer(gestures)
        puck, start = body(env, "puck_2"), self.grip(env, "puck_2")
        pointer.sample(1, puck, (0.0, 0.0, 0.0), start)
        env.perturb(2, start[:2] + [0.10, 0.0])  # the world moves it, mid-drag
        assert pointer.sample(0, puck, (0.0, 0.0, 0.0), start) is None
        assert recorder.pending == ()

    def test_a_drag_over_a_swinging_link_is_still_the_mouse_that_pulled(
        self, gestures, recorder, env
    ):
        """The arm moves under a held mouse, because the agent is driving it.

        Including between the press and the first sample that sees it: MuJoCo
        froze `refselpos` at the press, and a tick of the world happens before
        anything here looks.
        """
        pointer = Pointer(gestures)
        link, localpos = body(env, "link0"), np.array([0.1, 0.0, 0.0])
        pressed = np.array(env.data.xpos[link] + env.data.xmat[link].reshape(3, 3) @ localpos)
        env.data.ctrl[:] = 0.0
        env.data.qvel[:3] = [2.0, 0.0, 0.0]
        # A whole control tick passes between MuJoCo's press and the first
        # sample of it, and another between every sample after that.
        for _ in range(10):
            for _ in range(env.frame_skip):
                mujoco.mj_step(env.model, env.data)
            assert pointer.sample(1, link, localpos, pressed) is None
        assert arm_qpos(env)[0] > 0.05, "the arm did not swing far enough to matter"
        assert pointer.sample(0, link, localpos, pressed) is None
        assert recorder.pending == ()

    def test_a_drag_ends_where_it_itself_reached(self, gestures, recorder, env):
        """`refselpos` is one field successive drags take turns owning."""
        pointer = Pointer(gestures)
        first, second = body(env, "puck_0"), body(env, "puck_2")
        held = self.grip(env, "puck_0")
        pointer.sample(1, first, (0.0, 0.0, 0.0), held)
        # Let go and grab the far puck, both between one sample and the next.
        pointer.sample(1, second, (0.0, 0.0, 0.0), self.grip(env, "puck_2"))
        assert recorder.pending == ()
        assert env.puck_pose(0)[:2] == pytest.approx(held[:2], abs=1e-9)

    def test_a_puck_is_placed_from_where_the_drag_began(self, gestures, env):
        """Not from wherever the puck drifted to while the drag was held."""
        pointer = Pointer(gestures)
        puck, start = body(env, "puck_2"), self.grip(env, "puck_2")
        pointer.sample(1, puck, (0.0, 0.0, 0.0), start)
        env.perturb(2, start[:2] + [0.10, 0.0])
        pointer.sample(0, puck, (0.0, 0.0, 0.0), start + [0.0, 0.05, 0.0])
        assert env.puck_pose(2)[:2] == pytest.approx(start[:2] + [0.0, 0.05], abs=1e-9)

    def test_a_rotating_ctrl_drag_names_no_place_and_fires_no_hand(
        self, gestures, recorder, env
    ):
        """#116, and the whole of why retarget is not a drag.

        MuJoCo tells its two ctrl-drags apart in one field -- `mjPERT_TRANSLATE`
        for ctrl + the right button and `mjPERT_ROTATE` for ctrl + the left --
        so a rotating drag *is* deterministically distinguishable from the drag
        that already means `perturb`, and would have been the retarget-by-drag
        the spec would rather have had. What it does not carry is a
        destination: `mjv_movePerturb` turns `refquat` for it and leaves
        `refselpos` exactly where the press froze it, so there is no zone for it
        to name. Nothing fires, and nothing is special-cased to make that true.
        """
        pointer = Pointer(gestures)
        puck, start = body(env, "puck_0"), self.grip(env, "puck_0")
        rotating = int(mujoco.mjtPertBit.mjPERT_ROTATE)
        for _ in range(5):
            # `refselpos` stands still for the whole of it, which is the point.
            assert pointer.sample(rotating, puck, (0.0, 0.0, 0.0), start) is None
        assert pointer.sample(0, puck, (0.0, 0.0, 0.0), start) is None
        assert recorder.pending == ()
        assert env.puck_pose(0)[:2] == pytest.approx(start[:2], abs=1e-9)

    def test_the_struct_as_found_is_not_a_gesture(self, gestures, env):
        """A viewer opened with something already selected has clicked nothing."""
        pointer = Pointer(gestures)
        assert pointer.sample(0, body(env, "puck_2"), (0.0, 0.0, 0.0), (0, 0, 0)) is None
        assert gestures.pointed_at is None

    def test_a_release_is_the_one_gesture_a_sample_can_carry(
        self, gestures, recorder, env
    ):
        """A selection changing under a held drag is a drag ending, not a click."""
        pointer = Pointer(gestures)
        puck, start = body(env, "puck_0"), self.grip(env, "puck_0")
        pointer.sample(0, 0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        pointer.sample(0, puck, (0.0, 0.0, 0.0), start)  # the pick: points at it
        pointer.sample(1, puck, (0.0, 0.0, 0.0), start)  # ctrl down: the grab
        pointer.sample(1, puck, (0.0, 0.0, 0.0), start + [0.05, 0.0, 0.0])
        event = pointer.sample(0, 0, zone_point(1), start + [0.05, 0.0, 0.0])
        assert event.kind is EventKind.PERTURB
        assert [marker.kind for marker in recorder.pending] == [EventKind.PERTURB]

    def test_a_pick_the_release_elbowed_aside_is_deferred_and_not_dropped(
        self, gestures, recorder, env
    ):
        """The wrong-puck retarget: a selection that changes on the very sample
        a drag ends on is still a pointing, and must survive to the next one.

        Latching it would leave `pointed_at` on the puck named *before* the
        drag, so the human's next zone click reassigns a puck they stopped
        pointing at -- and the marker in the record would say they meant it.
        """
        pointer = Pointer(gestures)
        first, second = body(env, "puck_0"), body(env, "puck_2")
        pointer.sample(0, 0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        pointer.sample(0, first, (0.0, 0.0, 0.0), self.grip(env, "puck_0"))
        assert gestures.pointed_at == 0
        held = self.grip(env, "puck_0")
        pointer.sample(1, first, (0.0, 0.0, 0.0), held)
        # Pulled while still on puck 0, so the drag has travel of its own to end
        # with once the selection moves off it.
        pointer.sample(1, first, (0.0, 0.0, 0.0), held + [0.05, 0.0, 0.0])
        # One sample: the drag on puck 0 ends and puck 2 becomes the selection.
        ended = pointer.sample(0, second, (0.0, 0.0, 0.0), self.grip(env, "puck_2"))
        assert ended.kind is EventKind.PERTURB
        # The next sample is where that pointing gets read.
        pointer.sample(0, second, (0.0, 0.0, 0.0), self.grip(env, "puck_2"))
        assert gestures.pointed_at == 2, "the human's last pointing was lost"
        event = pointer.sample(0, 0, zone_point(1), zone_point(1))
        assert event.detail == (2.0, 1.0)
        assert (env.task.goal_puck, env.task.goal_zone) == (2, 1)

    def test_two_retargets_running_do_not_poison_each_other(self, gestures, env):
        """Point, assign, point again, assign again. The second gesture must
        start from nothing the first left behind."""
        pointer = Pointer(gestures)
        pointer.sample(0, 0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        for puck, zone in ((0, 0), (2, 1)):
            named = body(env, f"puck_{puck}")
            pointer.sample(0, named, (0.0, 0.0, 0.0), self.grip(env, f"puck_{puck}"))
            pointer.sample(0, 0, zone_point(zone), zone_point(zone))
            assert (env.task.goal_puck, env.task.goal_zone) == (puck, zone)

    def test_the_struct_is_one_buffer_the_viewer_edits_in_place(self, gestures, env):
        """`MjvPerturb.refselpos` is not a new array per tick -- it is one field
        MuJoCo overwrites. A pointer that kept a reference to it rather than a
        copy would have the start of the drag follow the mouse, so `moved`
        would be zero however far the human pulled, and no drag would fire."""
        pointer = Pointer(gestures)
        puck, start = body(env, "puck_1"), self.grip(env, "puck_1")
        refselpos = np.array(start)  # the one buffer, as the struct holds it
        pointer.sample(1, puck, (0.0, 0.0, 0.0), refselpos)
        for reach in (0.03, 0.06, 0.09):
            refselpos[:] = start + [reach, 0.0, 0.0]
            pointer.sample(1, puck, (0.0, 0.0, 0.0), refselpos)
        event = pointer.sample(0, puck, (0.0, 0.0, 0.0), refselpos)
        assert event is not None, "the drag was read as going nowhere"
        assert env.puck_pose(1)[:2] == pytest.approx(start[:2] + [0.09, 0.0], abs=1e-9)

    def test_one_stream_carries_a_pick_and_a_drag_and_a_pick(
        self, gestures, recorder, env
    ):
        """`08`'s events 2 and 3, as one motion continued: shove it, reassign it."""
        pointer = Pointer(gestures)
        puck, here = body(env, "puck_0"), self.grip(env, "puck_0")
        pointer.sample(0, 0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        pointer.sample(0, puck, (0.0, 0.0, 0.0), here)  # double-click: the pick
        pointer.sample(1, puck, (0.0, 0.0, 0.0), here)  # ctrl down: the grab
        for _ in range(3):
            pointer.sample(1, puck, (0.0, 0.0, 0.0), here + [0.05, 0.0, 0.0])
        pointer.sample(0, puck, (0.0, 0.0, 0.0), here + [0.05, 0.0, 0.0])
        pointer.sample(0, 0, zone_point(1), zone_point(1))
        assert [marker.kind for marker in recorder.pending] == [
            EventKind.PERTURB,
            EventKind.RETARGET,
        ]
        assert recorder.pending[1].detail == (0.0, 1.0)


class FakeViewer:
    """Enough of `mujoco.viewer`'s handle to drive the loop with no display.

    Not a stand-in for MuJoCo's picking, which nothing here re-implements: it
    is the same :class:`~mujoco.MjvPerturb` struct the real handle hands out,
    written by a script where a hand would write it with a mouse.
    """

    def __init__(self, model, data, key_callback, script):
        self.model, self.data, self.key_callback = model, data, key_callback
        self.script = script
        self.perturb = mujoco.MjvPerturb()
        self.cam = mujoco.MjvCamera()
        self.syncs = 0
        self.locked = 0
        self.running = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def is_running(self):
        return self.running

    @contextlib.contextmanager
    def lock(self):
        self.locked += 1
        try:
            yield
        finally:
            self.locked -= 1

    def sync(self):
        self.syncs += 1
        step = self.script.get(self.syncs)
        if step is not None:
            step(self)


class Window:
    """A script of `sync` -> what the human did, and the handles it drove."""

    def __init__(self):
        self.script = {}
        self.handles = []
        self.qpos_at_open = None
        """`qpos` as it stood the instant `launch_passive` was called."""


@pytest.fixture
def clock(monkeypatch):
    """The loop's clock, supplied rather than read (#113).

    `realtime=True` is the only thing in this file that reads a wall clock, and
    what it computes off one is exact arithmetic: sleep out `period` minus
    however long the iteration took. Timed against a real clock that exact
    statement becomes a race with whatever else the machine is doing -- and
    this laptop runs three build agents at once, so the loop routinely overruns
    its period, never sleeps, and the assertion fails for a reason that is not
    about the code. Held against a supplied clock the statement is exact, the
    race is gone, and the overrun becomes a case with an assertion of its own.

    Time moves here only where a test says it does: by the work a scripted
    iteration declares, and by whatever the loop itself sleeps.

    Substituted for the `time` module `drive` looked up, rather than for
    `time.sleep` on the stdlib module itself. The two are the same object, so
    patching the attribute would hold the whole process still for the length of
    the test, and any other sleep in it -- a library's retry, a thread -- would
    land in :attr:`slept` and fail an exact assertion for a reason that is not
    about the code. Which is the shape of failure this fixture exists to remove.
    """

    class Clock:
        def __init__(self):
            self.now = 0.0
            self.slept: list[float] = []

        def time(self) -> float:
            return self.now

        def sleep(self, seconds: float) -> None:
            self.slept.append(seconds)
            self.now += seconds

        def costing(self, seconds: float):
            """What one iteration costs, as a step for the window's script."""

            def spend(_handle):
                self.now += seconds

            return spend

    held = Clock()
    monkeypatch.setattr("patchworks.surface.gestures.time", held)
    return held


@pytest.fixture
def window(monkeypatch):
    import mujoco.viewer

    made = Window()

    def launch_passive(model, data, *, key_callback=None, **kwargs):
        made.qpos_at_open = np.array(data.qpos)
        made.handles.append(FakeViewer(model, data, key_callback, made.script))
        return made.handles[-1]

    monkeypatch.setattr(mujoco.viewer, "launch_passive", launch_passive)
    return made


class TestTheSceneWindowIsMujocosPassiveViewer:
    def test_the_window_is_the_viewer_over_the_running_world(self, recorder, window, env):
        list(drive(recorder, ticks=2, realtime=False))
        (opened,) = window.handles
        assert opened.model is env.model and opened.data is env.data

    def test_the_world_is_arranged_before_the_window_opens(self, recorder, window, env):
        """`run()` resets when it is *called*, and a reset rewrites every puck's
        `qpos` and runs `mj_forward`.

        Left inside the viewer block, that rewrite would run while the render
        thread was already reading the same `MjData` -- the wholesale rewrite
        the `r` key deliberately takes the lock for, done with no lock at all.
        So it has to have happened by the time the window opens.
        """
        before = np.array(env.data.qpos)
        list(drive(recorder, ticks=2, realtime=False))
        assert window.qpos_at_open is not None, "the window never opened"
        assert not np.allclose(before, window.qpos_at_open), (
            "the world was still un-arranged when the render thread started"
        )

    def test_a_ctrl_drag_through_the_struct_fires_a_hand(self, recorder, window, env):
        puck = body(env, "puck_0")
        dropped = []

        def grab(handle):
            handle.perturb.select = puck
            handle.perturb.localpos[:] = [0.0, 0.0, 0.0]
            handle.perturb.refselpos[:] = env.data.xpos[puck]
            handle.perturb.active = 1

        def pull(handle):
            handle.perturb.refselpos[:] += [0.05, 0.0, 0.0]
            dropped.append(np.array(handle.perturb.refselpos[:2]))

        def release(handle):
            handle.perturb.active = 0

        window.script.update({2: grab, 3: pull, 4: release})
        fired = [
            event
            for record in drive(recorder, ticks=8, realtime=False)
            for event in record.events
        ]
        assert [event.kind for event in fired] == [EventKind.PERTURB]
        assert fired[0].detail[1:] == pytest.approx(dropped[0], abs=1e-9)

    def test_a_key_is_acted_on_in_the_loop_and_not_in_the_callback(
        self, recorder, window, env
    ):
        """`r` reaches `MjData`, and MuJoCo calls the callback on its own thread."""
        seen = []

        def press(handle):
            before = pucks(env)
            handle.key_callback(ord("R"))
            seen.append((before, pucks(env)))

        window.script[2] = press
        list(drive(recorder, ticks=4, realtime=False))
        ((before, during),) = seen
        assert np.array_equal(before, during), "the callback rearranged the world itself"
        assert not np.allclose(before, pucks(env)), "the loop never drained the key"

    def test_the_loop_clears_any_applied_force_before_each_tick(
        self, recorder, window, env
    ):
        """A drag reaches the world once, through the hand, and not as a force.

        Named for what this checks rather than for what it would like to prove:
        the force written here is the *fake's*, so what is shown is that the
        loop clears `xfrc_applied` every tick -- not that MuJoCo's passive
        viewer ever writes it. It does not, having no physics thread; see
        :func:`drive`. This pins the clearing line, which is the insurance.
        """
        puck = body(env, "puck_1")

        def shove(handle):
            handle.data.xfrc_applied[puck, :3] = [50.0, 50.0, 0.0]

        window.script.update(dict.fromkeys(range(1, 9), shove))
        start = None
        for _record in drive(recorder, ticks=6, realtime=False):
            if start is None:
                start = np.array(env.puck_pose(1)[:2])
        assert env.puck_pose(1)[:2] == pytest.approx(start, abs=1e-6)

    def test_the_world_is_only_touched_under_the_viewers_lock(
        self, recorder, window, monkeypatch
    ):
        """The render thread reads the same model and data a hand writes to."""
        gestures = Gestures(Hands(recorder))
        under = []
        rearrange = gestures.rearrange

        def watched():
            under.append(window.handles[0].locked)
            rearrange()

        monkeypatch.setattr(gestures, "rearrange", watched)
        window.script[2] = lambda handle: handle.key_callback(ord("R"))
        list(drive(recorder, ticks=4, realtime=False, gestures=gestures))
        assert under == [1]

    def test_a_rearrangement_the_world_refuses_does_not_end_the_run(
        self, recorder, window, monkeypatch
    ):
        """`r` is setup. A refused one owes a message, not the end of a session."""
        gestures = Gestures(Hands(recorder))

        def refuse():
            raise BlockedAnnulusError("no layout clear of the arm")

        monkeypatch.setattr(gestures, "rearrange", refuse)
        window.script[2] = lambda handle: handle.key_callback(ord("R"))
        with pytest.warns(UserWarning, match="clear of the arm"):
            records = list(drive(recorder, ticks=6, realtime=False, gestures=gestures))
        assert len(records) == 5

    def test_realtime_sleeps_out_the_rest_of_the_worlds_own_tick(
        self, recorder, window, env, clock
    ):
        """The default, and the one the demo runs -- every other test here
        turns it off, so nothing otherwise executes the sleep at all.

        An iteration that cost nothing owes the whole period, and the period is
        the world's own two numbers.
        """
        list(drive(recorder, ticks=4, realtime=True))
        period = env.model.opt.timestep * env.frame_skip
        assert clock.slept == [pytest.approx(period)] * 4

    def test_an_iteration_that_took_time_sleeps_out_only_the_remainder(
        self, recorder, window, env, clock
    ):
        """Pacing, rather than a fixed delay bolted onto the end of a tick.

        A quarter of the period spent inside the loop leaves three quarters to
        sleep, which is the whole of what `realtime` means: the demo runs at
        the world's speed however long the graph took to think.
        """
        period = env.model.opt.timestep * env.frame_skip
        window.script = {sync: clock.costing(0.25 * period) for sync in range(1, 6)}
        list(drive(recorder, ticks=4, realtime=True))
        assert clock.slept == [pytest.approx(0.75 * period)] * 4

    def test_an_iteration_that_overran_its_period_does_not_sleep_backwards(
        self, recorder, window, env, clock
    ):
        """The case a busy machine is always in, asserted rather than suffered.

        This is what used to make the pacing test flake (#113): three build
        agents on one laptop push a tick past its period, the loop rightly
        declines to sleep, and a test that demanded a sleep failed for a reason
        that was never about the code. Overrunning is *correct* behaviour --
        there is no time left to give back -- so it is a case with an
        assertion of its own, and the run carries on unpaced.
        """
        period = env.model.opt.timestep * env.frame_skip
        window.script = {sync: clock.costing(2.0 * period) for sync in range(1, 6)}
        records = list(drive(recorder, ticks=4, realtime=True))
        assert clock.slept == []
        assert records, "the run stopped rather than falling behind"

    def test_the_pace_is_the_worlds_two_numbers_not_the_control_rate(
        self, recorder, window, env, clock, monkeypatch
    ):
        """`frame_skip` is a knob, and a pace taken from the nominal control
        rate would run the demo at the wrong speed the moment anyone turned it.

        Halving it halves the period, and nothing else here changes.
        """
        monkeypatch.setattr(env, "frame_skip", env.frame_skip // 2)
        list(drive(recorder, ticks=2, realtime=True))
        assert clock.slept == [
            pytest.approx(env.model.opt.timestep * env.frame_skip)
        ] * 2

    def test_no_realtime_never_sleeps(self, recorder, window, clock):
        list(drive(recorder, ticks=4, realtime=False))
        assert clock.slept == []

    def test_closing_the_window_ends_the_run(self, recorder, window):
        def shut(handle):
            handle.running = False

        window.script[4] = shut
        assert len(list(drive(recorder, ticks=50, realtime=False))) == 3


class TestTheCameraIsHeldTopDown:
    """#123's second lever, and the reason the first one is not enough alone.

    A drag is read against the camera plane, so the tilt of the view decides
    whether a gesture has a z in it at all. Set once at startup -- which is
    what #96 did -- it is not a lock: MuJoCo lets the human rotate the view
    freely, and the gestures then quietly change meaning. Here it is re-asserted
    every tick, which is the most the passive viewer allows (see
    :func:`~patchworks.surface.gestures.hold_top_down`).

    What no test here can settle is whether a view that will not tilt is
    pleasant to work at. That is the human at the window.
    """

    def opens(self, window):
        (handle,) = window.handles
        return handle

    def test_the_view_opens_looking_straight_down(self, recorder, window):
        """Read on the first `sync`, which is before the loop has held
        anything: the opening pose is the window's own and the hold is what
        keeps it, so lifting the hold does not open the view edge-on."""
        seen = []
        window.script[1] = lambda handle: seen.append(handle.cam.elevation)
        list(drive(recorder, ticks=4, realtime=False, hold_camera=None))
        assert seen == [TOP_DOWN_ELEVATION]

    def test_a_rotated_view_is_put_back_on_the_next_tick(self, recorder, window):
        """The bug the once-only assignment left: the human rotates, and every
        drag after that carries a z nobody asked for."""
        seen = []

        def rotate(handle):
            handle.cam.elevation = -20.0
            seen.append(handle.cam.elevation)

        def read(handle):
            seen.append(handle.cam.elevation)

        window.script.update({2: rotate, 3: read, 4: read})
        list(drive(recorder, ticks=6, realtime=False))
        assert seen == [-20.0, TOP_DOWN_ELEVATION, TOP_DOWN_ELEVATION]
        assert self.opens(window).cam.elevation == TOP_DOWN_ELEVATION

    def test_the_hold_leaves_panning_and_zooming_alone(self, recorder, window):
        """Only the tilt decides whether screen and plane coincide. Pan, zoom
        and spin cannot put a z on the screen, so they are not taken away."""

        def look_elsewhere(handle):
            handle.cam.lookat[:] = [0.2, -0.1, 0.0]
            handle.cam.distance = 0.7
            handle.cam.azimuth = 15.0

        window.script[2] = look_elsewhere
        list(drive(recorder, ticks=6, realtime=False))
        camera = self.opens(window).cam
        assert list(camera.lookat) == [0.2, -0.1, 0.0]
        assert (camera.distance, camera.azimuth) == (0.7, 15.0)
        assert camera.elevation == TOP_DOWN_ELEVATION

    def test_the_hold_lifts_in_one_place_for_a_world_worth_tilting(
        self, recorder, window
    ):
        """Relaxable, not welded shut (#123): a world with a third dimension
        passes `hold_camera=None` and the view is the human's again."""

        def rotate(handle):
            handle.cam.elevation = -20.0

        window.script[2] = rotate
        list(drive(recorder, ticks=6, realtime=False, hold_camera=None))
        assert self.opens(window).cam.elevation == -20.0

    def test_the_hold_is_one_call_the_loop_makes_and_not_a_pose_of_its_own(
        self, recorder, window
    ):
        """What is held is whatever :func:`hold_top_down` says, so a different
        constraint is that function's business and not the loop's."""
        held = []

        def hold(cam):
            held.append(cam)
            hold_top_down(cam)

        list(drive(recorder, ticks=4, realtime=False, hold_camera=hold))
        # Once a tick, on the viewer's own camera. The opening pose is the
        # window's and not one of these calls.
        assert held == [self.opens(window).cam] * 4

    def test_the_camera_is_only_touched_under_the_viewers_lock(self, recorder, window):
        """The render thread reads the camera every frame it draws."""
        under = []

        def hold(cam):
            under.append(window.handles[0].locked)
            hold_top_down(cam)

        list(drive(recorder, ticks=4, realtime=False, hold_camera=hold))
        assert under == [1] * 4


def retargets_on_iteration(window, env, at, puck=1, zone=2):
    """Script a retarget whose second pointing lands in iteration `at`.

    The loop's order within one iteration is *tick, capture, gesture*, and the
    fake viewer runs its script from `sync()`, which falls between the capture
    and the gesture. So a step filed under `at` is read by the pointer during
    iteration `at` and by nothing before it.

    **Three samples is the floor**, hence the refusal below: the pointer latches
    what it finds without firing (*the struct as found is not a gesture*), so
    iteration 1 is spent priming and the two pointings need the two after it.
    Filed any earlier, the puck pointing lands on the priming sample and
    `Gestures.pick` is never called -- and a test asserting that nothing reached
    the trace would pass with nothing ever fired, which is the shape of a test
    that cannot fail.
    """
    if at < 3:
        raise ValueError(
            f"the earliest iteration a retarget can land on is 3, not {at!r}: "
            "one sample primes the pointer and the two pointings follow it."
        )
    named = body(env, f"puck_{puck}")

    def point_at_the_puck(handle):
        handle.perturb.select = named
        handle.perturb.localpos[:] = [0.0, 0.0, 0.0]

    def point_at_the_zone(handle):
        handle.perturb.select = 0  # the table under the zone: a zone is a site
        handle.perturb.localpos[:] = zone_point(zone)

    window.script.update({at - 1: point_at_the_puck, at: point_at_the_zone})


class TestAMarkerFiredOnTheRunsLastTick:
    """#116's second ruling: a trace holds no record that no tick produced.

    A marker rides on the *next* capture, so the last tick of a run has no
    capture left to carry one. The loss is real, bounded and stated -- see
    :meth:`~patchworks.surface.record.Recorder.mark`, *What this cannot see* --
    and the remedy belongs to whoever declares the ticks.
    """

    def test_it_stays_pending_and_never_reaches_the_trace(self, recorder, window, env):
        retargets_on_iteration(window, env, at=4)
        fired = [
            event
            for record in drive(recorder, ticks=4, realtime=False)
            for event in record.events
        ]
        assert fired == [], "a marker fired after the last capture reached the trace"
        # Nothing is mis-recorded: the hand landed, the marker exists, it says
        # what it did, and it carries the tick it fired on -- the last one.
        assert [marker.kind for marker in recorder.pending] == [EventKind.RETARGET]
        assert recorder.pending[0].detail == (1.0, 2.0)
        assert recorder.pending[0].tick == recorder.agent.sheaf.ticks
        assert (env.task.goal_puck, env.task.goal_zone) == (1, 2)

    def test_one_more_declared_tick_is_the_whole_of_the_remedy(
        self, recorder, window, env
    ):
        """The same gesture on the same iteration, with one tick more declared."""
        retargets_on_iteration(window, env, at=4)
        records = list(drive(recorder, ticks=5, realtime=False))
        carrying = [record for record in records if record.events]
        assert len(carrying) == 1, [len(r.events) for r in records]
        (marker,) = carrying[0].events
        assert marker.kind is EventKind.RETARGET
        assert marker.detail == (1.0, 2.0)
        assert marker.tick == carrying[0].tick - 1, (
            "the marker was restamped to the capture that carried it"
        )
        assert recorder.pending == ()
