"""The gestures bound to the hands, and the live viewer they are bound in.

`docs/spec/10-the-demo-surface.md`, *The hands*:

| gesture | call | |
|---|---|---|
| ctrl-drag a **link** | `disturb_arm(joint, impulse)` | `08` event 1 |
| ctrl-drag a **puck** | `perturb(puck, xy)` | `08` event 2 |
| **right-click a puck, then click a zone** | `retarget(goal_puck, goal_zone)` | `08` event 3 |
| `r` | rearrange without resetting the arm | setup |

**The scene window is MuJoCo's passive viewer**, as `prototypes/sandbox/watch.py`
already runs it -- its camera, its picking, its drag. :func:`drive` opens that
viewer and does nothing to the picture in it. A single composed window was
considered and rejected because it pays for one tidy capture frame with the
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
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterator

import mujoco
import numpy as np

from patchworks.agent import run
from patchworks.sandbox.env import (
    ARM_JOINTS,
    CONTROL_HZ,
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
    "Gestures",
    "Pointer",
    "Referent",
    "ReferentKind",
    "drive",
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

#: Metres of drag below which nothing fires. A ctrl-press without a pull is not
#: a gesture, and firing on one would teleport a puck to where it already is --
#: leaving a marker that onset latency would then be measured from
#: (`docs/spec/10-the-demo-surface.md`, *Onset, and the near-misses*). It also
#: happens to be what leaves MuJoCo's *rotate* drag alone: ctrl-left-drag turns
#: the reference orientation and leaves the reference position where it was, so
#: it arrives here as a displacement of zero and is passed over rather than
#: special-cased.
MINIMUM_DRAG = 1e-3


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
        self, hands: Hands, *, impulse_per_metre: float = IMPULSE_PER_METRE
    ) -> None:
        self.hands = hands
        self.impulse_per_metre = float(impulse_per_metre)
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

    def drag(self, body: int, grip, to) -> Event | None:
        """One completed ctrl-drag. The referent alone decides which hand.

        `grip` is the point that was grabbed, in world coordinates, and `to` is
        where the drag has pulled it -- MuJoCo's own two numbers, the ones the
        spring in the viewer is drawn between. What comes back is the marker
        the hand dropped, or `None` for a drag on nothing and for a drag too
        short to be one (:data:`MINIMUM_DRAG`).

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
        argument is where the puck ends up: its centre, moved by the same
        displacement the grip was, which is exactly where the viewer has been
        drawing it while the drag was held.
        """
        grip = np.asarray(grip, dtype=np.float64)
        moved = np.asarray(to, dtype=np.float64) - grip
        if float(np.linalg.norm(moved)) < MINIMUM_DRAG:
            return None
        referent = self.referent(body, grip)
        if referent.kind is ReferentKind.LINK:
            return self.hands.disturb_arm(
                referent.index, self._impulse(referent.index, grip, moved)
            )
        if referent.kind is ReferentKind.PUCK:
            centre = self.world.data.xpos[int(body)][:2] + moved[:2]
            return self.hands.perturb(referent.index, centre)
        return None

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
        belongs. Note the button: MuJoCo's passive viewer reports picks and
        drags and no other mouse traffic, and the pick it reports is its
        selection. See :func:`drive`, *What the viewer reports*.
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
        Swallowing that here would leave a human pressing `r` at a world that
        quietly declines to rearrange.
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


class Pointer:
    """MuJoCo's perturb struct, sampled per tick, turned into whole gestures.

    ::

        pointer = Pointer(gestures)
        pointer.sample(active=1, select=puck_body, localpos=..., grip=..., to=...)

    Two state machines, and both exist because a gesture happens over many
    ticks while a hand fires on one.

    **A drag fires on release.** A hand is a discrete act -- one impulse, one
    teleport -- and a drag is fifty ticks of the mouse being somewhere. Firing
    per tick would put fifty markers in the record and leave onset latency
    counting from the last of them, which is to say from nothing
    (`docs/spec/10-the-demo-surface.md`, *Onset, and the near-misses*). So the
    drag is watched while it is held, and the hand fires once, when it ends,
    with where it ended up. What the human watches in the meantime is MuJoCo's
    own perturbation spring, which is drawn whether or not anything reads it.

    **A pick is an edge.** MuJoCo reports a selection as a state, not an event:
    the struct simply holds whichever body was picked last. A new pick is
    therefore a *change* in that state, and the change has to be read off the
    body-local click point rather than the world one -- the world point of a
    grip on a moving puck changes every tick, and every tick would be a new
    pick.
    """

    def __init__(self, gestures: Gestures) -> None:
        self.gestures = gestures
        self._held: tuple[int, np.ndarray, np.ndarray] | None = None
        self._picked: tuple[int, tuple[float, ...]] | None = None

    def __repr__(self) -> str:
        held = None if self._held is None else self._held[0]
        return f"Pointer(holding={held!r}, picked={self._picked!r})"

    def sample(self, active: int, select: int, localpos, grip, to) -> Event | None:
        """One tick's worth of the perturb struct. At most one hand fires.

        `active` is `mjvPerturb.active`, non-zero while a ctrl-drag is held;
        `select` the picked body; `localpos` the click point in that body's own
        frame; `grip` that same point in the world, and `to` where the drag has
        pulled it (`mjvPerturb.refselpos`).

        A pick and a drag-release cannot both be new on one tick -- a release
        is not a click -- so the order they are checked in decides nothing.
        """
        select = int(select)
        picked = (select, tuple(float(v) for v in localpos))
        fired: Event | None = None

        if self._picked is not None and picked != self._picked:
            fired = self.gestures.pick(select, grip)
        self._picked = picked

        if active:
            held = self._held
            if held is not None and held[0] != select:
                # The selection moved while a drag was held, which is not a
                # gesture anybody can make with one mouse -- but if it happens,
                # the drag that was in flight is the one that has ended.
                fired = self._release() or fired
            self._held = (select, np.asarray(grip, dtype=np.float64), np.asarray(to, dtype=np.float64))
        elif self._held is not None:
            fired = self._release() or fired
        return fired

    def _release(self) -> Event | None:
        held, self._held = self._held, None
        if held is None:
            return None
        body, grip, to = held
        return self.gestures.drag(body, grip, to)


def drive(
    recorder: Recorder,
    ticks: int,
    *,
    seed: int | None = None,
    realtime: bool = True,
    gestures: Gestures | None = None,
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
    :class:`~mujoco.MjvPerturb` struct, which holds what was picked and what is
    being dragged. So the picking, the ray and the drag are inherited entire --
    this function casts no ray and tracks no cursor -- and the price is that a
    *button* is not a thing that can be read. The retarget gesture's first
    pointing is therefore MuJoCo's selection, which is the pick the viewer
    already makes a human perform before it will let them drag anything
    (`prototypes/sandbox/watch.py`: "double-click a puck and ctrl-drag it").
    A raw right-click, as `10-the-demo-surface.md` names it, is not observable
    through this API at all; the pointing it asks for is.

    **The keys are drained here, not handled in the callback.** MuJoCo calls
    `key_callback` on its own UI thread, and `r` rearranges the world -- so
    handling it there would have two threads inside one `MjData`. The callback
    appends a code and returns; this loop is where a key does anything.

    **The hand is the only thing that moves the world.** MuJoCo's own
    perturbation force is cleared out of `xfrc_applied` before each tick, so a
    drag reaches the world once, through the hand, on the tick it was released
    -- rather than also as a force nobody recorded and no marker accounts for.
    The arena uses `xfrc_applied` for nothing else
    (`tests/test_sandbox_env.py`), so there is nothing else to clear away.
    """
    import mujoco.viewer

    agent = recorder.agent
    world = getattr(agent.env, "unwrapped", agent.env)
    gestures = gestures if gestures is not None else Gestures(Hands(recorder))
    pointer = Pointer(gestures)
    keys: deque[int] = deque()
    period = 1.0 / CONTROL_HZ

    with mujoco.viewer.launch_passive(
        world.model, world.data, key_callback=keys.append
    ) as viewer:
        viewer.cam.lookat[:] = [0.0, 0.0, 0.0]
        viewer.cam.distance = 1.6
        viewer.cam.elevation = -90.0
        viewer.cam.azimuth = 90.0
        last = time.time()
        for _outcome in run(agent, ticks, seed=seed):
            record = recorder.observe()
            if record is not None:
                yield record
            viewer.sync()
            if not viewer.is_running():
                break

            while keys:
                gestures.key(keys.popleft())
            with viewer.lock():
                perturb = viewer.perturb
                select = int(perturb.select)
                localpos = np.array(perturb.localpos)
                to = np.array(perturb.refselpos)
                active = int(perturb.active)
            grip = world.data.xpos[select] + world.data.xmat[select].reshape(3, 3) @ localpos
            pointer.sample(active, select, localpos, grip, to)
            world.data.xfrc_applied[:] = 0.0

            if realtime:
                slack = period - (time.time() - last)
                if slack > 0:
                    time.sleep(slack)
                last = time.time()
