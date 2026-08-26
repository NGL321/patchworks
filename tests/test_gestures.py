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
from patchworks.surface.gestures import IMPULSE_PER_METRE, MINIMUM_DRAG


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

    def test_a_drag_too_short_to_be_one_fires_nothing(self, gestures, recorder, env):
        short = MINIMUM_DRAG / 2
        assert gestures.drag(dragged(env, "link0", (0.0, short, 0.0), grip=(0.1, 0.0, 0.0))) is None
        assert gestures.drag(dragged(env, "puck_0", (short, 0.0, 0.0))) is None
        assert recorder.pending == ()

    def test_a_drag_out_of_the_plane_is_no_gesture_at_all(self, gestures, recorder, env):
        """This world is planar, and both hands take a planar argument."""
        assert gestures.drag(dragged(env, "puck_0", (0.0, 0.0, 0.5))) is None
        assert gestures.drag(dragged(env, "link0", (0.0, 0.0, 0.5), grip=(0.1, 0.0, 0.0))) is None
        assert recorder.pending == ()


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
        assert seen == [(puck, zone) for puck in range(N_PUCKS) for zone in range(N_ZONES)]

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


@pytest.fixture
def window(monkeypatch):
    import mujoco.viewer

    made = Window()

    def launch_passive(model, data, *, key_callback=None, **kwargs):
        made.handles.append(FakeViewer(model, data, key_callback, made.script))
        return made.handles[-1]

    monkeypatch.setattr(mujoco.viewer, "launch_passive", launch_passive)
    return made


class TestTheSceneWindowIsMujocosPassiveViewer:
    def test_the_window_is_the_viewer_over_the_running_world(self, recorder, window, env):
        list(drive(recorder, ticks=2, realtime=False))
        (opened,) = window.handles
        assert opened.model is env.model and opened.data is env.data

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

    def test_mujocos_own_perturbation_never_reaches_the_world(self, recorder, window, env):
        """A drag reaches the world once, through the hand, and not as a force."""
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

    def test_closing_the_window_ends_the_run(self, recorder, window):
        def shut(handle):
            handle.running = False

        window.script[4] = shut
        assert len(list(drive(recorder, ticks=50, realtime=False))) == 3
