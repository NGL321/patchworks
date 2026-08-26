"""The gestures bound to the hands, and the live viewer they are bound in.

`docs/spec/10-the-demo-surface.md`, *The hands*:

| gesture | call | |
|---|---|---|
| ctrl-drag a **link** | `disturb_arm(joint, impulse)` | `08` event 1 |
| ctrl-drag a **puck** | `perturb(puck, xy)` | `08` event 2 |
| **left-double-click a puck, then left-double-click a zone** | `retarget(goal_puck, goal_zone)` | `08` event 3 |
| `r` | rearrange without resetting the arm | setup |

The third row is what the spec used to call a right-click. #116 ruled on the
button after #96 found it unbindable, and :func:`drive`, *What the viewer
reports*, is where the whole of that reading lives.

**The scene window is MuJoCo's passive viewer**, as `prototypes/sandbox/watch.py`
already runs it -- its camera, its picking, its drag. :func:`drive` opens that
viewer and writes one field of it: the camera's tilt, held straight down
(:func:`hold_top_down`). It does nothing else to the picture. A single composed
window was considered and rejected because it pays for one tidy capture frame with the
exact interaction that already works, so nothing here selects a body, casts a
ray, or tracks a mouse: MuJoCo does all three, into
:class:`~mujoco.MjvPerturb`, and what this module reads is that struct.

**Two seams, and the first is the one worth testing.** :class:`Gestures` is
*gesture in, hand out*: it takes a referent and a displacement -- plain numbers
-- and decides which hand fires with which arguments. It never touches a
window, so `tests/test_gestures.py` drives every binding in the table above
without one. :class:`Pointer` is the small state machine above it, turning the
per-tick samples of MuJoCo's perturb struct into *one* completed gesture; it is
plain numbers in as well. :func:`drive` is the only part that needs a display,
and it is the part that decides nothing.

**The referent alone decides which hand fires.** A ctrl-drag on a link is
`disturb_arm` and a ctrl-drag on a puck is `perturb` because of what was
grabbed and nothing else, which is why `08`'s first two events need no new
binding and read as one motion with two targets.

**A gesture is planar, and an out-of-plane drag fires nothing** (#123). The
world has no third dimension -- `arena.xml` has hinges about z and slides in x
and y, and no z slide anywhere -- so a drag with a z in it names nothing the
world can do, and :meth:`Gestures.drag` refuses it and says so.

**Where the z comes from is the mouse, not the camera.** MuJoCo's plain
ctrl-drag is `mjMOUSE_MOVE_V`, which translates the grabbed point in the
*vertical* plane: pull across the screen and the point moves in the world's
xy, pull up the screen and it moves in world z -- **at every camera elevation,
top-down included**. The shifted drag is `mjMOUSE_MOVE_H` and is planar
whichever way it goes. `tests/test_gestures.py` pins both against MuJoCo's own
`mjv_movePerturb`, because #123 was filed on the other reading and the code
should not quietly keep it. :func:`hold_top_down` therefore holds the *picture*
in the plane and not the gesture, and the refusal is what enforces the plane.
Whether the gesture then *feels* right is a human at the window.

**Nothing here is part of the architecture.** These are the *human's* hands
(`docs/spec/03-the-sandbox.md`, *The human's hand*), fired from outside on the
footing any experimenter is on: no cell reads anything this module computes,
and a run with the window shut is the same run. What it does write to is the
world -- that is what a hand is -- and every firing leaves a marker in the tick
record through :class:`~patchworks.surface.onset.Hands`, so the demo's temporal
measure cannot be lost to a gesture nobody recorded.
"""

from __future__ import annotations

import time
import warnings
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Iterator

import mujoco
import numpy as np

from patchworks.agent import Agent, run
from patchworks.sandbox.env import (
    ARM_JOINTS,
    BlockedAnnulusError,
    N_PUCKS,
    N_ZONES,
    ZONE_RADIUS,
    ZONE_XY,
)

from .onset import Hands
from .record import Event, Recorder, TickRecord

__all__ = [
    "IMPULSE_PER_METRE",
    "MINIMUM_DRAG",
    "OUT_OF_PLANE_TOLERANCE",
    "TOP_DOWN_ELEVATION",
    "Drag",
    "Gestures",
    "Pointer",
    "Referent",
    "ReferentKind",
    "drive",
    "hold_top_down",
]

#: Newton-metre-seconds of joint impulse per metre the drag pulls a link along
#: the direction that joint moves it in. A **feel** constant, and the surface's
#: rather than the world's: `disturb_arm` takes an impulse, a drag has a
#: length, and something has to relate the two. Nothing in the record fixes it,
#: nothing downstream reads it, and turning it up or down changes how hard a
#: hand shoves rather than what any of this means. At 1.0 a 10 cm drag on the
#: shoulder is about 0.1 N m s, which against that joint's inertia is a visible
#: nudge rather than a launch.
IMPULSE_PER_METRE = 1.0

#: Metres of drag below which nothing fires, read on the larger of the drag's
#: planar and out-of-plane components -- a press that went nowhere went nowhere
#: in either. A ctrl-press without a pull is not a gesture, and firing on one
#: would teleport a puck to where it already is --
#: leaving a marker that onset latency would then be measured from
#: (`docs/spec/10-the-demo-surface.md`, *Onset, and the near-misses*). It also
#: happens to be what leaves MuJoCo's *rotate* drag alone: ctrl + the left
#: button is `mjPERT_ROTATE`, which turns the reference orientation and leaves
#: the reference position where it was, so it arrives here as a displacement of
#: zero and is passed over rather than special-cased. The drag that fires a hand
#: is the *translating* one, `mjPERT_TRANSLATE` -- ctrl + the right button (see
#: :func:`drive`, *What the viewer reports*).
MINIMUM_DRAG = 1e-3

#: How far a ctrl-drag may stray out of the plane before it is no gesture at
#: all, **as a fraction of how far it went in the plane**. A ratio rather than a
#: length, because what makes a drag out-of-plane is the direction it went in
#: and not how long it was: at 0.1 a pull more than about six degrees off the
#: table is refused, and one along it is not.
#:
#: **This is #123's enforcing lever**, and the one that does the work: MuJoCo's
#: plain translate drag carries a z whatever the camera is doing, so this is
#: what stops one reaching a hand -- see :meth:`Gestures.drag`, and
#: :func:`hold_top_down` for the display constraint kept beside it.
#: :class:`Gestures` takes its own, so a world with a third dimension in it can
#: lift the refusal by handing in `float("inf")` without this line moving.
OUT_OF_PLANE_TOLERANCE = 0.1

#: The camera elevation at which the arena's own plane fills the screen:
#: straight down. #123's display constraint -- what the human sees, rather than
#: what a gesture carries -- applied by :func:`hold_top_down` and re-applied
#: every tick by :func:`drive`.
TOP_DOWN_ELEVATION = -90.0


def hold_top_down(cam) -> None:
    """Hold the scene camera's tilt at straight down. #123's second lever.

    Takes MuJoCo's own :class:`~mujoco.MjvCamera` -- `viewer.cam` -- and writes
    one field. :func:`drive` calls it every tick, under the viewer's lock,
    which is what makes it a hold rather than the startup pose #96 set and #123
    found had never held anything: MuJoCo lets the human rotate the view freely
    and nothing said the picture had stopped agreeing with the world.

    **What this does and does not do, plainly.** It holds the *picture* in the
    plane: straight down, the arena's own xy fills the screen, so the planar
    half of a drag is the motion the human watches themselves making and the
    perturbation ghost's own travel reads as what it is. **It does not make a
    gesture planar.** MuJoCo's plain ctrl-drag translates in the vertical plane
    -- its up-the-screen axis is world z at every elevation, top-down included
    (`tests/test_gestures.py`, *TestTheDragMujocoHandsOver*) -- so no camera
    constraint can take a z out of one. :meth:`Gestures.drag`'s refusal is what
    enforces the plane; this is kept beside it, as #123 ruled, because the two
    fail in different ways and the ruling wanted the redundancy: **neither is
    dead weight, and neither is to be removed as redundant with the other.**

    **Re-assertion, not prevention.** The passive viewer's mouse handling lives
    inside MuJoCo's own `Simulate` on its UI thread, and `launch_passive` hands
    back a camera and no way to turn any of it off -- so nothing here can stop
    the human rotating the view. What it can do is put the view back, and it
    does, every tick: at this world's 50 Hz a rotation survives at most 20 ms,
    which is a flicker rather than a regime.

    **Only the tilt is held.** Panning, zooming and spinning the view leave the
    arena's plane square to the screen, so they are left to the human; and the
    arena's one fixed camera (`topdown` in `src/patchworks/sandbox/arena.xml`)
    looks straight down too, so the viewer's own camera keys land somewhere
    this already agrees with.

    **Relaxable in one place**: `drive(..., hold_camera=None)` opens the window
    where it always did and then leaves the view to the human, and a world that
    wants a different constraint passes its own callable rather than editing
    that loop.
    """
    cam.elevation = TOP_DOWN_ELEVATION


class ReferentKind(str, Enum):
    """What a gesture was pointed at. **This is what decides which hand.**"""

    LINK = "link"
    PUCK = "puck"
    ZONE = "zone"
    NOTHING = "nothing"


@dataclass(frozen=True)
class Referent:
    """One thing in the arena, named the way a hand names it.

    :attr:`index` is the argument the hand takes: the joint for a link, the
    puck for a puck, the zone for a zone. It is `-1` for
    :attr:`ReferentKind.NOTHING`, which no hand takes.
    """

    kind: ReferentKind
    index: int = -1


_NOTHING = Referent(ReferentKind.NOTHING)


@dataclass(frozen=True)
class Drag:
    """One completed ctrl-drag, in the terms MuJoCo's perturb struct gives it.

    **`moved` is the mouse's own displacement**, from where the drag was
    grabbed to where it was let go -- not the gap between the mouse and the
    body. Those two are the same thing only while the body stands still, and
    the body here is an arm the agent is driving: hold the mouse dead still for
    a second over a swinging link and the gap grows to something that would
    fire a hand nobody pulled. The gap is what MuJoCo draws its perturbation
    spring across, which is the right thing for a *force* and the wrong thing
    for a gesture that ends in one impulse.

    :attr:`grip` is where the body has the grabbed point **now**, which is what
    a Jacobian can be taken at, and :attr:`origin` is where the body itself was
    when the drag began, which is what a teleport is measured from.
    """

    body: int
    """The MuJoCo body that was grabbed. **This is what decides the hand.**"""

    grip: np.ndarray
    """The grabbed point in world coordinates, as the body holds it now."""

    origin: np.ndarray
    """Where the body was when the drag began."""

    moved: np.ndarray
    """How far the drag has pulled since it began."""


class Gestures:
    """Gesture in, hand out. The whole of the table in this module's docstring.

    ::

        gestures = Gestures(Hands(recorder))
        gestures.drag(body, grip=(0.2, 0.0, 0.0), to=(0.2, 0.1, 0.0))

    Holds a :class:`~patchworks.surface.onset.Hands`, so a gesture that fires
    leaves its marker in the tick record on the tick it fired -- the hands and
    the markers are one call there, and this class is one of the two callers
    that section names (`docs/spec/10-the-demo-surface.md`, *Onset, and the
    near-misses*).

    **The bodies are read once, here.** Which MuJoCo body is a link and which
    is a puck is a fact about the arena, fixed at construction, so resolving a
    referent at gesture time is a dictionary lookup rather than a search --
    the same act the recorder's private-component index is.
    """

    def __init__(
        self,
        hands: Hands,
        *,
        impulse_per_metre: float = IMPULSE_PER_METRE,
        out_of_plane_tolerance: float = OUT_OF_PLANE_TOLERANCE,
    ) -> None:
        self.hands = hands
        self.impulse_per_metre = float(impulse_per_metre)
        self.out_of_plane_tolerance = float(out_of_plane_tolerance)
        model = self.world.model
        name2id = mujoco.mj_name2id
        joints = mujoco.mjtObj.mjOBJ_JOINT
        # A link's body carries exactly one of the arm's joints, so the joint
        # the hand wants is the body's own. Read off the env's joint ids rather
        # than off body names: the arm is what `ARM_JOINTS` says it is, in one
        # place.
        arm = [name2id(model, joints, name) for name in ARM_JOINTS]
        self._body_of = tuple(int(model.jnt_bodyid[jid]) for jid in arm)
        self._dofadr = tuple(int(model.jnt_dofadr[jid]) for jid in arm)
        self._link = {body: joint for joint, body in enumerate(self._body_of)}
        self._puck = {
            int(model.jnt_bodyid[name2id(model, joints, f"p{puck}_x")]): puck
            for puck in range(N_PUCKS)
        }
        self._pointed_at: int | None = None

    def __repr__(self) -> str:
        return f"Gestures(pointed_at={self._pointed_at!r})"

    @property
    def world(self) -> Any:
        """The env the hands reach, unwrapped."""
        return self.hands.world

    @property
    def pointed_at(self) -> int | None:
        """The puck a retarget is half-way through naming, or `None`.

        See :meth:`pick`: the gesture is two pointings, and this is what the
        first one left behind.
        """
        return self._pointed_at

    # -- what was pointed at ---------------------------------------------------

    def referent(self, body: int, point) -> Referent:
        """What a pick at `point` on MuJoCo body `body` was pointed at.

        The body settles a link and a puck, because MuJoCo's own picking
        already returns one and a link and a puck are bodies. A **zone** is
        not: zones are sites, so no ray hits one and the pick comes back as the
        table -- the world body, with the click point on it. So a pick that
        named no body of ours is resolved by *where* it landed, against the
        zones' own fixed positions and radius. That is the same test
        `goal_satisfied` applies to a puck, so "inside the zone" means on
        screen what it means in the task.
        """
        body = int(body)
        if body in self._link:
            return Referent(ReferentKind.LINK, self._link[body])
        if body in self._puck:
            return Referent(ReferentKind.PUCK, self._puck[body])
        xy = np.asarray(point, dtype=np.float64)[:2]
        distance = np.linalg.norm(ZONE_XY - xy, axis=1)
        zone = int(np.argmin(distance))
        if distance[zone] <= ZONE_RADIUS:
            return Referent(ReferentKind.ZONE, zone)
        return _NOTHING

    # -- ctrl-drag: `08` events 1 and 2 ----------------------------------------

    def drag(self, drag: Drag) -> Event | None:
        """One completed ctrl-drag. The referent alone decides which hand.

        What comes back is the marker the hand dropped, or `None` for a drag on
        nothing, for a drag too short to be one (:data:`MINIMUM_DRAG`), and for
        one out of the plane (:data:`OUT_OF_PLANE_TOLERANCE`).

        **A link is nudged, never placed.** `disturb_arm` takes an impulse
        because displacing `qpos` would have the world rewrite the arm's
        configuration, which is the one thing this env never does
        (`docs/spec/03-the-sandbox.md`, *The human's hand*). So the drag is
        turned into an impulse about the grabbed link's own joint: the joint
        moves the grabbed point along a direction that MuJoCo's Jacobian
        already knows, and the impulse is how far the drag went **along that
        direction**, times :data:`IMPULSE_PER_METRE`. Pull the link the way the
        joint swings and it swings; pull it along its own length and nothing
        happens, which is the truth about a hinge rather than a shortcut.

        **A puck is placed, never nudged.** `perturb` teleports, so the
        argument is where the puck ends up: where it was when the drag began,
        plus how far the drag went -- which is where the viewer has been
        drawing the ghost of it all the while.

        **An out-of-plane drag is refused, and said so.** This world is planar
        by construction (`src/patchworks/sandbox/arena.xml`: every joint is a
        hinge about z or a slide in x/y, and there is no z slide anywhere), and
        both hands take a planar argument -- so an out-of-plane pull is not a
        small gesture but no gesture at all. The gate is on the out-of-plane
        component, measured against the planar one
        (:data:`OUT_OF_PLANE_TOLERANCE`), and **that is #123's enforcing
        lever**: #96 wrote the sentence above and gated on the planar magnitude
        instead, which is a different set. A drag that was mostly z with a
        millimetre of planar residue cleared that gate and fired a hand with
        the residue as its argument -- a puck teleported a millimetre in a
        direction the human never expressed, which is the "random directions"
        the report describes. Nothing arbitrary reaches the world now: the
        refusal warns, because a human who is given nothing and told nothing is
        left wondering which of the two happened.

        **Where the z comes from is the mouse, not the camera.** #123 was
        filed on the reading that a drag carries a z only when the view is off
        top-down. MuJoCo says otherwise: the plain ctrl-drag is
        `mjMOUSE_MOVE_V`, a translate in the *vertical* plane, so pulling
        across the screen moves the point in the world's xy and pulling up the
        screen moves it in world z -- at **every** elevation, straight down
        included. The shifted drag, `mjMOUSE_MOVE_H`, is planar whichever way
        it goes. `tests/test_gestures.py`, *TestTheDragMujocoHandsOver*, pins
        both against `mjv_movePerturb` itself. So this refusal, and not the
        camera, is what keeps the plane -- and the warning names the shift,
        because "look from above" is advice that would not have helped.

        **Kept beside the camera hold, on purpose.** :func:`hold_top_down`
        holds the picture in the plane; this holds the gesture. They fail
        differently and #123 ruled for the redundancy, so **neither is dead
        weight and neither is to be removed as redundant with the other**. Both
        are relaxable in one place: a world with a third dimension in it lifts
        this one by constructing :class:`Gestures` with a larger
        `out_of_plane_tolerance`, and lifts the other with `hold_camera=None`.

        **The perturbation ghost stays 3D.** MuJoCo draws the drag as a spring
        in three dimensions and nothing here can make it do otherwise. Held
        top-down that is mostly harmless -- the planar half of the drag is
        drawn where the puck would go -- but an out-of-plane pull is drawn
        going *somewhere*, towards or away from the camera, and then fires
        nothing at all. That gap between the picture and the effect is
        MuJoCo's, it cannot be closed here, and the warning is what stands in
        for closing it: it is why this refusal speaks rather than passing over
        the drag in silence.
        """
        moved = np.asarray(drag.moved, dtype=np.float64)
        planar = float(np.linalg.norm(moved[:2]))
        out_of_plane = abs(float(moved[2]))
        # Whether the mouse moved at all, asked of the whole displacement: a
        # press that went nowhere is not a gesture in any dimension, and this is
        # where MuJoCo's rotating drag -- a displacement of exactly zero -- is
        # passed over. Only then is it worth saying which way it went.
        if max(planar, out_of_plane) < MINIMUM_DRAG:
            return None
        referent = self.referent(drag.body, drag.grip)
        if referent.kind not in (ReferentKind.LINK, ReferentKind.PUCK):
            # A drag on the table, a wall, or nothing. No hand takes it whatever
            # plane it was in, so there is nothing here to refuse and nobody to
            # tell: saying "no hand fired" over a miss would blame the pull for
            # what was an aim.
            return None
        if out_of_plane > self.out_of_plane_tolerance * planar:
            warnings.warn(
                f"a ctrl-drag {out_of_plane:.3g} m out of the plane against "
                f"{planar:.3g} m in it is no gesture at all: this world has no "
                "third dimension for it to mean anything in, so no hand fired. "
                "MuJoCo's plain ctrl-drag translates in the vertical plane, "
                "so drag across the screen rather than up it -- or hold shift "
                "as well, which is MuJoCo's horizontal-plane translate and is "
                "planar whichever way it goes (#123).",
                stacklevel=2,
            )
            return None
        if planar < MINIMUM_DRAG:
            # Unreachable at any tolerance of 1 or less, the default included:
            # a planar component under the minimum has an out-of-plane one at
            # or over it beside it -- or the first gate would have taken the
            # drag -- and the second gate then refuses it. What this is here for
            # is a world that lifted the tolerance: such a world can arrive
            # with a drag it considers perfectly in-plane and nothing in the
            # plane to hand over, and both hands take xy. Note `inf * 0.0` is
            # `nan` and every comparison with one is False, so a straight-up
            # drag under a lifted tolerance arrives here rather than being
            # refused above. Firing would teleport a puck to where it already
            # is, and leave a marker onset latency would then be measured from.
            return None
        if referent.kind is ReferentKind.LINK:
            return self.hands.disturb_arm(
                referent.index, self._impulse(referent.index, drag.grip, moved)
            )
        return self.hands.perturb(referent.index, (drag.origin + moved)[:2])

    def _impulse(self, joint: int, grip: np.ndarray, moved: np.ndarray) -> float:
        """The drag, resolved along the direction that joint moves the grip in.

        `mj_jac` is the same map `04-action-and-the-boundary.md`'s torques go
        through and the same one the prototype's scripted pusher reads: one
        column of it is how the grabbed point travels per radian of this joint,
        which is the only direction a hinge can answer a pull in.
        """
        world = self.world
        model, data = world.model, world.data
        jacp = np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jacp, None, grip, self._body_of[joint])
        # In the plane, like both hands' arguments. What reaches here has
        # already been refused if it was out of the plane, so the z dropped is
        # at most `OUT_OF_PLANE_TOLERANCE` of the pull -- the residue of a view
        # a few degrees off top-down, not a gesture. Every arm joint in
        # `arena.xml` is a z-hinge, so this column's own z entry is zero today
        # and the projection changes nothing either; it is here so that the
        # impulse stays planar the first time a joint is not a z-hinge.
        moved = np.asarray(moved, dtype=np.float64).copy()
        moved[2] = 0.0
        direction = jacp[:, self._dofadr[joint]]
        length = float(np.linalg.norm(direction))
        if length == 0.0:
            # The grip sits on the joint's own axis, where no pull has a moment
            # arm. A hand still fired -- the human dragged a link -- and what it
            # delivers is nothing, which is what the physics says.
            return 0.0
        return float(direction @ moved) / length * self.impulse_per_metre

    # -- click, then click: `08` event 3 ---------------------------------------

    def pick(self, body: int, point) -> Event | None:
        """One pick of MuJoCo's own selection. Two of them are a retarget.

        **Point at the thing, then point at where you now want it.** Retarget
        is the hand that needed designing, because unlike the other two it has
        no world-side handle to grab; click-then-click was chosen over a
        keypress because it reads to a bystander with no caption, and the third
        event of the demo is precisely the one a viewer must understand without
        help (`docs/spec/10-the-demo-surface.md`, *The hands*).

        A pick on a **puck** points at the thing. A pick on a **zone**
        completes the gesture and fires `retarget`, returning its marker. A
        pick on anything else -- a link, the table, the wall -- clears what was
        pointed at: an aimless click is not half of a gesture, and leaving it
        armed would let a click into a zone minutes later fire an event nobody
        asked for.

        The first pointing is MuJoCo's **selection**, which is the pointing the
        viewer already makes a human do -- it is what arms a ctrl-drag, and the
        viewer highlights it. That is why `08`'s events 2 and 3 read as one
        motion continued: the double-click that lets you shove the puck is the
        same one that names it, and the zone click after it says where it now
        belongs. Note the button: a pointing is a **left double-click**,
        because that is the only mouse act the passive viewer turns into a
        selection. See :func:`drive`, *What the viewer reports*, for the whole
        of why, and #116 for the ruling that fixed it there.
        """
        referent = self.referent(body, point)
        if referent.kind is ReferentKind.PUCK:
            self._pointed_at = referent.index
            return None
        if referent.kind is ReferentKind.ZONE and self._pointed_at is not None:
            puck, self._pointed_at = self._pointed_at, None
            return self.hands.retarget(goal_puck=puck, goal_zone=referent.index)
        if referent.kind is not ReferentKind.ZONE:
            self._pointed_at = None
        return None

    # -- the keys: the headless and scripted path ------------------------------

    def pair(self, index: int) -> Event:
        """Retarget to one of the 3x3 (puck, zone) pairs, by index. Row-major.

        **The headless and scripted path**, which is what the repeated runs use
        (`docs/spec/10-the-demo-surface.md`, *The hands*). A script that cannot
        click needs event 3 as much as a human can, and 40 snapshots times
        three events is not a thing anybody points at
        (`docs/spec/08-the-acceptance-demo.md`, *The repeated runs*).

        Nine pairs, `0 .. 8`: puck `index // 3`, zone `index % 3`. Pressing the
        number keys in order therefore cycles the grid a row at a time, which
        is the gesture :func:`drive` binds them to.
        """
        if not 0 <= int(index) < N_PUCKS * N_ZONES:
            raise ValueError(
                f"there are {N_PUCKS} pucks and {N_ZONES} zones, so a pair is "
                f"0 .. {N_PUCKS * N_ZONES - 1}; got {index!r}"
            )
        puck, zone = divmod(int(index), N_ZONES)
        return self.hands.retarget(goal_puck=puck, goal_zone=zone)

    def rearrange(self) -> None:
        """`r`: rearrange the world, and leave the arm where it is.

        `reset()` is that already -- it rearranges the world and does not touch
        the arm or the clock (`docs/spec/03-the-sandbox.md`, *The Gymnasium
        contract, made continual*) -- so this is one call and no options.

        **No marker.** A rearrangement is setup rather than one of `08`'s
        events: the record names three kinds and only three
        (:class:`~patchworks.surface.record.EventKind`), and onset latency
        counts from a disturbance the agent was mid-task through.

        The observation the reset hands back is dropped on purpose. A reset is
        in-band and the agent lives through it, so the agent meets the
        rearranged world on the next tick the way it meets everything else --
        through the external write at the end of it. Writing it here would be a
        second external write, in the middle of nothing.

        It can refuse, and the refusal is left to travel: `reset()` will not
        move the arm to make room for a layout, so an arm parked across the
        spawn annulus raises
        :class:`~patchworks.sandbox.env.BlockedAnnulusError` and says so.
        Swallowing it here would leave a human pressing `r` at a world that
        quietly declines to rearrange. :func:`drive` turns it into a warning at
        the one place that knows the difference between a refused key and a
        session worth ending.
        """
        self.world.reset()

    def key(self, code: int) -> Event | None:
        """One key, as the viewer's key callback reports it. `r`, or `1` - `9`.

        Anything else is ignored: this is one window's worth of bindings, not a
        keymap.
        """
        code = int(code)
        if code in (ord("r"), ord("R")):
            self.rearrange()
            return None
        if ord("1") <= code <= ord("9"):
            return self.pair(code - ord("1"))
        return None


@dataclass
class _Grab:
    """A ctrl-drag in flight, as :class:`Pointer` is watching it happen."""

    body: int
    origin: np.ndarray
    """Where the body was when this drag was first seen."""
    grabbed: np.ndarray
    """`refselpos` then: MuJoCo froze it at the press and the mouse moves it."""
    localpos: np.ndarray
    """The grabbed point, in the body's own frame."""
    reached: np.ndarray
    """`refselpos` as of the last sample of **this** drag."""


class Pointer:
    """MuJoCo's perturb struct, sampled per tick, turned into whole gestures.

    ::

        pointer = Pointer(gestures)
        pointer.sample(active=1, select=body, localpos=..., refselpos=...)

    The four arguments are the four fields of :class:`~mujoco.MjvPerturb` that
    say what the mouse is doing, passed on unread: what was picked, where on it,
    whether a ctrl-drag is held, and where that drag has pulled the picked point
    to. Everything else -- where the body is, what the grabbed point is in the
    world -- is looked up here from the world the hands already reach, because
    it is a fact about the world rather than about the mouse.

    Two state machines, and both exist because a gesture happens over many
    ticks while a hand fires on one.

    **A drag fires on release.** A hand is a discrete act -- one impulse, one
    teleport -- and a drag is fifty ticks of the mouse being somewhere. Firing
    per tick would put fifty markers in the record and leave onset latency
    counting from the last of them, which is to say from nothing
    (`docs/spec/10-the-demo-surface.md`, *Onset, and the near-misses*). So the
    drag is watched while it is held, and the hand fires once, when it ends,
    with how far the mouse took it. What the human watches in the meantime is
    MuJoCo's own perturbation spring, which is drawn whether or not anything
    reads it.

    **Every drag is measured against its own start, and ends where it itself
    reached.** `refselpos` is one field that successive drags take turns
    owning, so a drag whose end and a next drag's start both land between two
    samples would otherwise be read as one enormous pull from the first body to
    the second -- a teleport of a puck onto whatever was grabbed next, and a
    marker for it in the record. Each drag keeps its own two numbers instead.

    **A pick is an edge.** MuJoCo reports a selection as a state, not an event:
    the struct simply holds whichever body was picked last. A new pick is
    therefore a *change* in that state, read off the body-local click point --
    the world point of a grip on a moving puck changes every tick, and every
    tick would be a new pick.
    """

    def __init__(self, gestures: Gestures) -> None:
        self.gestures = gestures
        self._grab: _Grab | None = None
        self._picked: tuple[int, tuple[float, ...]] | None = None

    def __repr__(self) -> str:
        held = None if self._grab is None else self._grab.body
        return f"Pointer(holding={held!r}, picked={self._picked!r})"

    def sample(self, active: int, select: int, localpos, refselpos) -> Event | None:
        """One tick's worth of the perturb struct. At most one hand fires.

        `active` is `mjvPerturb.active`, non-zero while a ctrl-drag is held;
        `select` the picked body, which MuJoCo documents as **non-positive for
        none** -- so nothing but a real body is grabbed, read or indexed with.
        Nothing outside can tell the difference, since a referent no hand takes
        fires no hand either way; what the guard stops is a `-1` wrapping round
        `xpos` into a grip point on the last body in the arena, which would be
        an answer rather than a refusal. Untested for exactly that reason.
        `localpos` is the click point in that body's own frame, and `refselpos`
        where the drag has pulled it.

        A release wins over a pick. The one sample that can look like both is a
        selection changing under a held drag, which is not two gestures one
        mouse can make: it is a drag ending.
        """
        select = int(select)
        localpos = np.asarray(localpos, dtype=np.float64)
        # Copied, not referenced: `np.asarray` does not copy, and `_Grab` holds
        # `grabbed` for the whole life of the drag. A caller who hands over
        # `viewer.perturb.refselpos` itself -- which is what the struct offers,
        # and what this class's own example reads like -- would otherwise have
        # the start of the drag follow the mouse, leaving `moved` identically
        # zero and no drag ever firing.
        refselpos = np.array(refselpos, dtype=np.float64)
        holding = active and select > 0

        fired: Event | None = None
        held = self._grab
        if held is not None and not (holding and held.body == select):
            if held.body == select:
                # Its own last word: the drag ended where the mouse was when it
                # was let go, which is this sample if it is still this body's.
                held.reached = refselpos
            fired = self._release()
        if holding:
            if self._grab is None:
                self._grab = _Grab(
                    body=select,
                    origin=np.array(self.gestures.world.data.xpos[select]),
                    grabbed=refselpos,
                    localpos=localpos,
                    reached=refselpos,
                )
            else:
                self._grab.reached = refselpos

        picked = (select, tuple(float(value) for value in localpos))
        if fired is None:
            if self._picked is not None and picked != self._picked:
                fired = self.gestures.pick(select, self._point(select, localpos))
            self._picked = picked
        # A pick the release elbowed aside is **deferred, not dropped**. What
        # was selected is left unlatched, so the next sample still reads it as
        # a change and the pointing lands one tick late. Latching it here would
        # lose it for good: the human would have named a puck the struct agrees
        # is selected and this would never have seen it, so the zone click
        # after it would retarget whichever puck was named before -- the wrong
        # one, with a marker in the record saying otherwise.
        return fired

    def _release(self) -> Event | None:
        """The drag in flight, if there is one, as one completed gesture.

        The mouse's travel is `refselpos` now against `refselpos` when this
        drag was first seen -- **both mouse-side**, so the body's own motion
        under a held mouse cannot get into it. Anchoring on where the body had
        the grabbed point instead would put an arm the agent is driving into
        the measurement, and a mouse held perfectly still over a swinging link
        would fire a hand nobody pulled.

        The one thing lost is whatever the mouse did between MuJoCo's press and
        the first sample that saw it, which is under a tick and comes off the
        near end of the human's own pull.
        """
        grab, self._grab = self._grab, None
        if grab is None:
            return None
        return self.gestures.drag(
            Drag(
                body=grab.body,
                grip=self._point(grab.body, grab.localpos),
                origin=grab.origin,
                moved=grab.reached - grab.grabbed,
            )
        )

    def _point(self, body: int, localpos: np.ndarray) -> np.ndarray:
        """A body-local click point, in the world. The world body's frame is
        the world's, so a pick that named no body reads as the point itself."""
        if body <= 0:
            return localpos
        data = self.gestures.world.data
        return data.xpos[body] + data.xmat[body].reshape(3, 3) @ localpos


def drive(
    recorder: Recorder,
    ticks: int,
    *,
    seed: int | None = None,
    realtime: bool = True,
    gestures: Gestures | None = None,
    hold_camera: Callable[[Any], None] | None = hold_top_down,
) -> Iterator[TickRecord]:
    """Run the agent in MuJoCo's passive viewer, with the hands bound to it.

    ::

        for record in drive(Recorder(agent), ticks=100_000):
            ...

    Yields each captured :class:`~patchworks.surface.record.TickRecord`, which
    is the same thing :meth:`~patchworks.surface.record.Recorder.watch` yields
    and the same thing a trace off disk holds -- so the dome panel plugs in
    here without this function knowing that it exists
    (`docs/spec/10-the-demo-surface.md`, *The trace*). It ends when the run
    does or when the window is closed, whichever is first.

    On macOS the passive viewer must be run under `mjpython` rather than
    `python`; that is MuJoCo's requirement and its error message says so.

    **What the viewer reports.** `launch_passive` offers exactly one event
    hook, `key_callback`, and one piece of readable mouse state: the
    :class:`~mujoco.MjvPerturb` struct. So the picking, the ray and the drag
    are inherited entire -- this function casts no ray and tracks no cursor --
    and the price is that a *button* is not a field anything here can read.
    What the viewer will deterministically say, read off MuJoCo 3.10's
    `simulate/simulate.cc` and `engine/engine_vis_interact.c`, is exactly three
    acts:

    * **A selection**, in `pert.select` and `pert.localpos`. Written on a
      *double-click*, and only when the button is the **left** one: MuJoCo's
      selection mode is 1 for `mjBUTTON_LEFT` and 2 or 3 for anything else,
      and those two move the camera's `lookat` or start it tracking instead.
    * **A translating drag**, `pert.active == mjPERT_TRANSLATE`, which MuJoCo
      sets on ctrl + the **right** button over something already selected. It
      is the one act that moves `pert.refselpos`, which is to say the one that
      names a *place*.
    * **A rotating drag**, `pert.active == mjPERT_ROTATE`, ctrl + the **left**
      button. It turns `pert.refquat` and moves no point at all.

    `pert.active2` is never written by the viewer, `key_callback` is handed a
    key code and nothing else -- presses, not releases, and no modifier state
    -- so those three acts, and the `r`/`1`-`9` keys, are the whole palette.
    It is the palette `prototypes/sandbox/watch.py` already describes from the
    other end: "double-click a puck and ctrl-drag it".

    **The retarget gesture is two selections: left-double-click the puck, then
    left-double-click the zone.** `10-the-demo-surface.md` used to call it a
    right-click; #116 ruled on the button after #96 found that one unbindable,
    and this is where that ruling landed. Its priority order was: take a legal
    button pairing if one is deterministic, prefer a *drag* from puck to zone
    if one is achievable, and re-implement picking only if neither is. **The
    drag is closed by the list above**, not by preference: a drag that means
    *put it there* has to report where *there* is, the only drag that reports a
    place is the translating one, and a translating ctrl-drag on a puck is
    already `perturb` -- two hands cannot fire from one gesture on one
    referent. The rotating drag is genuinely distinguishable from it, and
    carries an orientation rather than a position, so it can name no zone. So
    the first option stands, picking stays inherited, and the ctrl-drag that
    fires `08`'s events 1 and 2 is untouched. The second click lands on the
    table geom under the zone, since a zone is a site and no ray hits one, and
    :meth:`Gestures.referent` resolves it by where it landed.

    The spec's *point at the thing, then point at where you want it* is met
    exactly; its button is not, and cannot be. **That the gesture reads
    correctly to a bystander is the one thing here a human at the window has to
    confirm**; nothing in the test suite can, and which physical button a given
    platform delivers as MuJoCo's "left" or "right" is a question for the same
    human.

    **What this cannot see: a marker fired on the last tick of a run.** The
    order within an iteration is *tick, capture, then gesture* -- a hand fires
    after this tick's record has already been yielded, so its marker rides on
    the **next** iteration's capture
    (:meth:`~patchworks.surface.record.Recorder.mark`). On the last iteration
    there is no next one, and the marker stays in
    :attr:`~patchworks.surface.record.Recorder.pending` rather than reaching
    the trace. Nothing is mis-recorded: the marker carries the tick it fired
    on, and `pending` is where a caller can still read it. What is lost is the
    last event of the run, in the trace only.

    It is left that way on purpose (#116): the alternative is a trailing record
    that no tick produced, and there is nothing honest to put in its arrays
    -- see :mod:`~patchworks.surface.record`, *What a trace cannot hold*. **A
    run that must not lose its last event declares one tick more than it
    measures**, which costs one tick and no semantics; that is the falsification
    sweep's obligation, alongside the per-trial
    :meth:`~patchworks.surface.onset.OnsetCounter.restart` that #95 handed
    forward, and neither is a thing this loop can check for its caller.

    **The camera is held top-down, every tick** (#123). #96 set the pose once,
    here, before the loop -- which is not a hold, since MuJoCo lets the human
    rotate the view freely and nothing said the picture had stopped agreeing
    with the arena's plane. So the tilt is re-asserted on every iteration, under
    the same lock as everything else the render thread reads. It is
    **re-assertion and not prevention**: the passive viewer exposes no way to
    disable its own mouse camera, so a rotation lasts until the next tick --
    20 ms here -- and then goes back.

    What it buys is the *picture*: the arena's xy fills the screen, so a drag's
    planar half is the motion the human watches themselves make. It does not
    make a gesture planar, and :func:`hold_top_down` says why at length -- the
    plain translate drag carries a z at every elevation, and
    :meth:`Gestures.drag`'s refusal is what keeps that z out of a hand.

    The opening pose is unchanged and unconditional -- the window still opens
    looking straight down at the arena. `hold_camera=None` lifts the *hold* in
    one place, for a world worth looking at from an angle: the view opens where
    it always did and is then the human's. A different constraint is a different
    callable, and this loop stays one line either way.

    **The keys are drained here, not handled in the callback.** MuJoCo calls
    `key_callback` on its own UI thread, and `r` rearranges the world -- so
    handling it there would have two threads inside one `MjData`. The callback
    appends a code and returns; this loop is where a key does anything.

    **The hand is the only thing that moves the world.** `xfrc_applied` is
    cleared every tick, so a drag reaches the world once -- through the hand,
    on the tick it was released -- and never also as a force nobody recorded
    and no marker accounts for. This is **insurance rather than a correction**:
    `launch_passive` runs with no physics thread of its own, and MuJoCo applies
    its perturbation force from that thread, so in this mode nothing upstream
    writes that field. What the viewer does draw is the perturbation spring,
    which is a picture and not a force. The arena uses `xfrc_applied` for
    nothing else (`tests/test_sandbox_env.py`), so clearing it costs nothing
    and stops the day a caller wraps this loop around a viewer that does.
    """
    import mujoco.viewer

    agent = recorder.agent
    world = getattr(agent.env, "unwrapped", agent.env)
    gestures = gestures if gestures is not None else Gestures(Hands(recorder))
    pointer = Pointer(gestures)
    keys: deque[int] = deque()
    # How long a tick actually lasts, from the world's own two numbers rather
    # than from the nominal rate: `frame_skip` is a knob, and the env already
    # adjusts its advertised frame rate for it.
    period = world.model.opt.timestep * world.frame_skip

    # **The world is arranged before the render thread exists.** `run()` resets
    # when it is *called*, not on the first `next()` (:func:`patchworks.agent.run`),
    # and a reset rewrites every puck's `qpos` and runs `mj_forward` -- the same
    # wholesale rewrite the `r` key takes the lock for. Building the iterator
    # inside the viewer block would do all of that while the render thread was
    # already reading `MjData`, so the window would open on a torn frame at
    # best. Built here, it opens on the world the run starts from.
    ticking = run(agent, ticks, seed=seed)

    with mujoco.viewer.launch_passive(
        world.model, world.data, key_callback=keys.append
    ) as viewer:
        # Under the lock like every other write to state the render thread
        # reads: the camera is read every frame it draws.
        with viewer.lock():
            viewer.cam.lookat[:] = [0.0, 0.0, 0.0]
            viewer.cam.distance = 1.6
            viewer.cam.azimuth = 90.0
            viewer.cam.elevation = TOP_DOWN_ELEVATION
        last = time.time()
        for _outcome in ticking:
            record = recorder.observe()
            if record is not None:
                yield record
            viewer.sync()
            if not viewer.is_running():
                break

            # All of it under the lock, which is the mutex the render thread
            # holds while it reads the model and the data: a hand that fires
            # writes to both, and `r` rewrites the whole world.
            with viewer.lock():
                # The tilt, put back before anything is read off it. A drag is
                # measured against the camera plane, so this line and the
                # refusal in `Gestures.drag` are the same guarantee twice over
                # (#123): the human is free to rotate, and both the picture and
                # the gesture are back in the plane a tick later.
                if hold_camera is not None:
                    hold_camera(viewer.cam)
                while keys:
                    try:
                        gestures.key(keys.popleft())
                    except BlockedAnnulusError as refused:
                        # The world declining to rearrange is a refusal, not the
                        # end of the session: `r` is setup, and a human who
                        # pressed it with the arm across the annulus is owed the
                        # message and the run they were in the middle of.
                        warnings.warn(str(refused), stacklevel=2)
                perturb = viewer.perturb
                pointer.sample(
                    int(perturb.active),
                    int(perturb.select),
                    np.array(perturb.localpos),
                    np.array(perturb.refselpos),
                )
                world.data.xfrc_applied[:] = 0.0

            if realtime:
                slack = period - (time.time() - last)
                if slack > 0:
                    time.sleep(slack)
                last = time.time()


def main(argv: list[str] | None = None) -> None:
    """`mjpython -m patchworks.surface.gestures` -- the demo, drivable by hand.

    An agent, a world, and the window: enough to sit in front of and perturb.
    Ctrl-drag a link or a puck, left-double-click a puck and then a zone, `r`
    to rearrange, `1` - `9` for the pairs. On macOS the passive viewer needs
    `mjpython`.

    The dome panel is the other window and is not opened here. It reads the
    records this yields (`docs/spec/10-the-demo-surface.md`, *Two windows*),
    which is the whole of the seam between them.
    """
    import argparse

    from patchworks.graph import build_graph
    from patchworks.sandbox import PlanarPushSandbox

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--ticks", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", default="train")
    arguments = parser.parse_args(argv)

    world = PlanarPushSandbox(split=arguments.split)
    agent = Agent(world, dome=build_graph())
    for _record in drive(Recorder(agent), arguments.ticks, seed=arguments.seed):
        pass
    world.close()


if __name__ == "__main__":
    main()
