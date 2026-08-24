"""The sandbox: a 3-link planar arm pushing pucks into zones, as a `gymnasium.Env`.

The world and the surfaces through which the agent meets it are specified in
`docs/spec/03-the-sandbox.md`. Every number there was chosen by building the
thing and watching it, so the numbers here are transcribed, not re-derived.

Three deviations from Gymnasium are the contract, not placeholders:

1. `reset()` rearranges the world; it never resets the agent. The arm keeps
   whatever configuration it was left in, physics time is monotonic across the
   entire run, and no observation component announces that anything happened.
   The agent finds out the world changed the way it finds out anything else:
   its predictions stop working. `reset(options={"reset_arm": True})` exists
   for setup and is not used in normal operation.
2. `reward` is always `0.0`; `terminated` and `truncated` are always `False`.
   There is no reward channel and there are no episodes. The goal reaches the
   agent as perception -- the target zone lights up in the render.
3. Privileged truth lives in `info` -- puck poses, goal identity, goal distance,
   whether the goal is satisfied. It is for logging and for the acceptance
   demo's instrumentation only. Feeding it to the agent defeats the sandbox.

Because there is no episode boundary to restart from, reproducibility comes
from snapshot and restore of `mjSTATE_INTEGRATION` plus the task and the
sampler's RNG. That lives in `patchworks.sandbox.state`, off this class on
purpose: a restore is an experimenter's tool and is invisible from inside,
where everything here is in-band and the agent lives through it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from patchworks.sandbox.state import STATE_SPEC

ARENA_XML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "arena.xml")

ENV_ID = "Patchworks/PlanarPushSandbox-v0"
ENTRY_POINT = "patchworks.sandbox.env:PlanarPushSandbox"

N_PUCKS = 3
N_ZONES = 3
ARM_JOINTS = ("j0", "j1", "j2")

# 500 Hz physics, 10 substeps per tick, so control runs at 50 Hz.
FRAME_SKIP = 10
CONTROL_HZ = 50.0

IMAGE_SIZE = 64

#: How many layouts reset() will draw before giving up on finding one clear of
#: the arm, whose pose it is not allowed to change.
PLACEMENT_ATTEMPTS = 64

# --- the sampler's space -------------------------------------------------------
# The held-out slice is defined along two axes at once, and the two are kept
# separate: there is deliberately no split value returning their union, because
# a number drawn from it would be attributable to neither axis.
HELDOUT_PAIRS = frozenset({(0, 2), (2, 0)})
HELDOUT_SECTOR = (np.deg2rad(30.0), np.deg2rad(75.0))
HELDOUT_SECTOR_MIN_R = 0.22
SPLITS = ("train", "heldout_pair", "heldout_sector", "any")

SPAWN_R = (0.15, 0.36)  # annulus: pedestal at 0.08, ring wall at 0.52
ZONE_XY = np.array([[0.0, 0.30], [-0.26, -0.15], [0.26, -0.15]])
ZONE_RADIUS = 0.075

ZONE_DIM_RGBA = np.array([0.35, 0.35, 0.35, 0.35])
ZONE_LIT_RGBA = np.array([1.00, 0.85, 0.10, 0.85])

# --- the friction field --------------------------------------------------------
# The table is not uniform. Repeated identical pushes in a rigid-body simulator
# are bit-identical; real ones are not, and a world whose disagreement can be
# fully cleared is a weak proving ground. Each puck's frictionloss is therefore
# scaled by a smooth, fixed function of where that puck is -- mean 1, range
# 0.75-1.25. Resampling per `reset()` was rejected: it would put a number into
# the model that `mjSTATE_INTEGRATION` does not cover, so restore would diverge
# silently. A field is a pure function of state, so snapshot and restore stay
# bit-exact, and where a puck *is* now changes what a push does.
FRICTION_FIELD_AMP = (0.15, 0.10)  # sums to the +/- 25% about nominal
FRICTION_FIELD_WAVELENGTH = (0.31, 0.37, 0.53)
FRICTION_FIELD_PHASE = 0.7


def friction_scale(xy) -> float:
    """Multiplier on a puck's frictionloss at position `xy`. Mean 1, range 0.75-1.25."""
    x, y = float(xy[0]), float(xy[1])
    a, b = FRICTION_FIELD_AMP
    lx, ly, ld = FRICTION_FIELD_WAVELENGTH
    return (
        1.0
        + a * np.sin(2 * np.pi * x / lx + FRICTION_FIELD_PHASE) * np.cos(2 * np.pi * y / ly)
        + b * np.sin(2 * np.pi * (x + y) / ld)
    )


def in_heldout_sector(xy) -> bool:
    """Is `xy` in the wedge training never places the target puck in?"""
    r = float(np.linalg.norm(xy))
    if r < HELDOUT_SECTOR_MIN_R:
        return False
    angle = float(np.arctan2(xy[1], xy[0])) % (2 * np.pi)
    return HELDOUT_SECTOR[0] <= angle <= HELDOUT_SECTOR[1]


# eq=False because two fields are arrays: the generated __eq__ would raise
# rather than compare, and the generated __hash__ would raise rather than hash.
@dataclass(frozen=True, eq=False)
class Task:
    """One sampled arrangement of the world plus what is wanted of it."""

    puck_xy: np.ndarray  # (N_PUCKS, 2)
    puck_theta: np.ndarray  # (N_PUCKS,)
    goal_puck: int
    goal_zone: int

    @property
    def pair(self) -> tuple[int, int]:
        return (self.goal_puck, self.goal_zone)


class SpecLimitError(RuntimeError):
    """Raised when a spec-level `max_episode_steps` is attached to this env.

    `TimeLimit` sets `truncated=True`, which every standard loop resets on,
    and a reset here is not a restart but an unannounced rearrangement of the
    world mid-trajectory. A limit is therefore refused rather than merely
    undocumented: prefer the constraint that cannot drift to the sentence
    nobody rereads.

    What is refused is a limit on a *spec*: one carried by a registration this
    env is the entry point of, or one on an `EnvSpec` assigned to the env.
    `gymnasium.make(id, max_episode_steps=n)` is a fourth path and cannot be
    reached from in here -- `make()` hands the unwrapped env a spec with
    `max_episode_steps=None` whatever it was asked for, and wraps `TimeLimit`
    around the outside afterwards, so nothing the env can read ever mentions
    the limit. `tests/test_sandbox_conformance.py` records that exposure.
    """


def _check_index(value: int, limit: int, what: str) -> int:
    if not 0 <= value < limit:
        raise ValueError(f"{what} must be in range(0, {limit}), got {value}")
    return value


def _refuse_step_limit(env_spec) -> None:
    if env_spec is not None and getattr(env_spec, "max_episode_steps", None) is not None:
        raise SpecLimitError(
            f"{getattr(env_spec, 'id', env_spec)} carries max_episode_steps="
            f"{env_spec.max_episode_steps}. gymnasium.make() would wrap this env in "
            "TimeLimit, whose truncated=True every standard loop resets on -- and a "
            "reset here rearranges the world mid-trajectory rather than restarting "
            "anything. Register the sandbox with max_episode_steps=None."
        )


class PlanarPushSandbox(gym.Env):
    """The sandbox, as a literal `gymnasium.Env`.

    Observation: `qpos (3,)`, `qvel (3,)`, `touch (3,)`, `image (64, 64, 3)`
    uint8. Action: three torques normalised to `[-1, 1]` and scaled to the
    per-joint limits. No object pose is ever given to the agent: everything
    about the world arrives through the render, unlabelled.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": int(CONTROL_HZ)}

    def __init__(
        self,
        split: str = "train",
        image_size: int = IMAGE_SIZE,
        frame_skip: int = FRAME_SKIP,
        render_mode: str | None = None,
        render_obs: bool = True,
    ):
        """Build the world.

        `render_obs=False` blanks the observation's image for headless probes
        that only want the physics; the agent needs it, so it defaults on.
        `render()` is unaffected by it.
        """
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"unsupported render_mode {render_mode!r}")

        self._refuse_registered_step_limit()

        self._spec = None
        self.split = split
        self.image_size = image_size
        self.frame_skip = frame_skip
        self.render_mode = render_mode
        self.render_obs = render_obs

        self.model = mujoco.MjModel.from_xml_path(ARENA_XML)
        self.data = mujoco.MjData(self.model)
        # The renderer needs a GL context, which a headless probe may not have,
        # so it is built on the first frame anyone actually asks for.
        self._renderer = None
        self._closed = False
        # frame_skip is a knob, so the advertised frame rate follows it rather
        # than the default -- a recorder would otherwise encode the run at the
        # wrong speed.
        self.metadata = {
            **self.metadata,
            "render_fps": round(1.0 / (self.model.opt.timestep * frame_skip)),
        }

        self.task: Task | None = None

        name2id = mujoco.mj_name2id
        obj = mujoco.mjtObj
        self._arm_jid = [name2id(self.model, obj.mjOBJ_JOINT, n) for n in ARM_JOINTS]
        self._arm_qadr = [self.model.jnt_qposadr[j] for j in self._arm_jid]
        self._arm_dofadr = [self.model.jnt_dofadr[j] for j in self._arm_jid]

        puck_jid = [name2id(self.model, obj.mjOBJ_JOINT, f"p{i}_x") for i in range(N_PUCKS)]
        # Each puck's three joints (slide x, slide y, hinge r) are contiguous
        # from the x joint, so one address indexes the whole pose.
        self._puck_qadr = [self.model.jnt_qposadr[j] for j in puck_jid]
        self._puck_dofadr = [self.model.jnt_dofadr[j] for j in puck_jid]
        self._friction_nominal = np.array(
            [self.model.dof_frictionloss[a : a + 3].copy() for a in self._puck_dofadr]
        )

        self._touch_adr = [
            self.model.sensor_adr[name2id(self.model, obj.mjOBJ_SENSOR, f"t{i}")]
            for i in range(len(ARM_JOINTS))
        ]
        self._zone_sid = [name2id(self.model, obj.mjOBJ_SITE, f"zone_{i}") for i in range(N_ZONES)]
        self._arm_gid = [
            name2id(self.model, obj.mjOBJ_GEOM, n)
            for n in ("g_link0", "g_link1", "g_link2", "g_tip")
        ]
        self._puck_gid = [name2id(self.model, obj.mjOBJ_GEOM, f"g_puck_{i}") for i in range(N_PUCKS)]
        self.puck_radius = self.model.geom_size[self._puck_gid, 0].copy()

        # qpos is unbounded rather than held to the arena's joint ranges. A
        # MuJoCo joint limit is a soft constraint, so the arm overshoots it --
        # measured at up to 0.028 rad under a uniform-random policy -- and
        # disturb_arm() takes an impulse of any size, so no finite bound is one
        # the physics honours. A bound a consumer would clip against and be
        # wrong is worse than no bound; the spec's joint ranges are asserted
        # against the arena in tests/test_sandbox_world.py, where they are a
        # fact about the body rather than a promise about an observation.
        self.observation_space = spaces.Dict(
            {
                "qpos": spaces.Box(-np.inf, np.inf, (len(ARM_JOINTS),), np.float32),
                "qvel": spaces.Box(-np.inf, np.inf, (len(ARM_JOINTS),), np.float32),
                "touch": spaces.Box(0.0, np.inf, (len(ARM_JOINTS),), np.float32),
                "image": spaces.Box(0, 255, (image_size, image_size, 3), np.uint8),
            }
        )
        self.action_space = spaces.Box(-1.0, 1.0, (self.model.nu,), np.float32)
        self._torque_limit = self.model.actuator_ctrlrange[:, 1].copy()

        mujoco.mj_forward(self.model, self.data)

    # -- the refused step limit -------------------------------------------------

    @staticmethod
    def _refuse_registered_step_limit() -> None:
        """Refuse at construction, before a TimeLimit can be wrapped around us.

        `gymnasium.make()` rebuilds the unwrapped env's spec with
        `max_episode_steps=None` whatever the registration said, and applies
        TimeLimit outside, so the limit is invisible from in here once the env
        exists. The registry is where it is still visible.

        Every registration of this entry point is checked, not only the one
        being constructed, because construction cannot tell which one it is
        for. That is blunt -- one limited registration refuses every sandbox in
        the process, including a correctly registered one -- and deliberately
        so: the alternative is a limit that fires only sometimes, which is the
        drift this refusal exists to rule out. The error names the registration
        at fault.
        """
        for env_spec in gym.registry.values():
            if env_spec.entry_point in (ENTRY_POINT, PlanarPushSandbox):
                _refuse_step_limit(env_spec)

    @property
    def spec(self):
        return self._spec

    @spec.setter
    def spec(self, value) -> None:
        _refuse_step_limit(value)
        self._spec = value

    # -- observation ------------------------------------------------------------

    def _camera_image(self) -> np.ndarray:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, self.image_size, self.image_size)
        self._renderer.update_scene(self.data, camera="topdown")
        return self._renderer.render()

    def _obs(self) -> dict:
        image = (
            self._camera_image()
            if self.render_obs
            else np.zeros((self.image_size, self.image_size, 3), np.uint8)
        )
        return {
            "qpos": self.data.qpos[self._arm_qadr].astype(np.float32),
            "qvel": self.data.qvel[self._arm_dofadr].astype(np.float32),
            "touch": self.data.sensordata[self._touch_adr].astype(np.float32),
            "image": image,
        }

    def puck_pose(self, i: int) -> np.ndarray:
        """`(x, y, theta)` of puck `i`. Privileged: for `info` and the demo only."""
        a = self._puck_qadr[i]
        return np.array(self.data.qpos[a : a + 3])

    def _require_task(self) -> Task:
        if self.task is None:
            raise RuntimeError(
                "The world has not been arranged yet: call reset() before step(), "
                "retarget(), a snapshot, or a restore."
            )
        return self.task

    def _info(self) -> dict:
        """Privileged truth. For logging and the acceptance demo, never for the agent."""
        poses = np.stack([self.puck_pose(i) for i in range(N_PUCKS)])
        task = self._require_task()
        distance = float(np.linalg.norm(poses[task.goal_puck, :2] - ZONE_XY[task.goal_zone]))
        return {
            "puck_pose": poses,
            "goal_puck": task.goal_puck,
            "goal_zone": task.goal_zone,
            "goal_distance": distance,
            # A gate on whether a trial is valid, never a score.
            "goal_satisfied": bool(distance < ZONE_RADIUS),
        }

    # -- the sampler ------------------------------------------------------------

    def sample_task(self) -> Task:
        """Draw a (layout, target puck, target zone) triple from this env's split.

        Known limitation, deliberate: a target that fails its split condition
        costs the *whole* layout, so the condition selects over whole layouts
        and reaches the distractors through the clearance constraint -- the
        held-out wedge is systematically emptier of them (0.034 against 0.104).
        Recorded in `docs/spec/03-the-sandbox.md`, *Two limitations of the
        sampler, on the record*, and left as built there: resampling the target
        alone would move the spawn distribution the 14/72 achievability figure
        was measured on.
        """
        while True:
            xy, theta = self._sample_layout()
            puck = int(self.np_random.integers(N_PUCKS))
            zone = int(self.np_random.integers(N_ZONES))
            held_pair = (puck, zone) in HELDOUT_PAIRS
            held_sector = in_heldout_sector(xy[puck])
            if self.split == "any":
                return Task(xy, theta, puck, zone)
            if self.split == "train" and not (held_pair or held_sector):
                return Task(xy, theta, puck, zone)
            if self.split == "heldout_pair" and held_pair and not held_sector:
                return Task(xy, theta, puck, zone)
            if self.split == "heldout_sector" and held_sector and not held_pair:
                return Task(xy, theta, puck, zone)

    def _sample_layout(self) -> tuple[np.ndarray, np.ndarray]:
        """Three puck poses in the spawn annulus, clear of each other and of the zones.

        "Clear of the zones" means **centres**: the zone test below ignores the
        puck's radius, which the puck/puck test does not. Deliberately
        inconsistent and left as built, for the same reason as above;
        `goal_satisfied` is a centre test too, so it is unaffected. See
        `docs/spec/03-the-sandbox.md`, *Two limitations of the sampler, on the
        record*.
        """
        radius = self.puck_radius
        while True:
            angle = self.np_random.uniform(0, 2 * np.pi, N_PUCKS)
            r = self.np_random.uniform(*SPAWN_R, N_PUCKS)
            xy = np.stack([r * np.cos(angle), r * np.sin(angle)], axis=1)
            gaps = [
                np.linalg.norm(xy[i] - xy[j]) - radius[i] - radius[j]
                for i in range(N_PUCKS)
                for j in range(i + 1, N_PUCKS)
            ]
            to_zones = min(
                np.linalg.norm(xy[i] - ZONE_XY[z]) for i in range(N_PUCKS) for z in range(N_ZONES)
            )
            if min(gaps) > 0.03 and to_zones > ZONE_RADIUS + 0.04:
                return xy, self.np_random.uniform(-np.pi, np.pi, N_PUCKS)

    # -- the continual reset ----------------------------------------------------

    def reset(self, *, seed=None, options=None):
        """Rearrange the world. The agent is not reset, and neither is the clock."""
        super().reset(seed=seed)
        options = options or {}

        if options.get("reset_arm", False):  # off by default: continual means continual
            self.data.qpos[self._arm_qadr] = 0.0
            self.data.qvel[self._arm_dofadr] = 0.0
            self.data.ctrl[:] = 0.0

        # The arm is not reset, so its pose is wherever the agent left it. A
        # layout that intersects that pose starts the world inside a
        # penetration and the solver launches a puck across the arena. Place,
        # then check, then re-place.
        # Not a restore: this is the same engine constant, taken and put back
        # inside one call to undo a rejected placement. `state.restore()` would
        # be wrong here in both directions -- there may be no task yet to light
        # a zone from, and nothing has happened for an experimenter to rewind.
        before = np.empty(mujoco.mj_stateSize(self.model, STATE_SPEC))
        mujoco.mj_getState(self.model, self.data, before, STATE_SPEC)
        previous_task = self.task

        given = options.get("task")
        for _ in range(PLACEMENT_ATTEMPTS):
            self.task = given or self.sample_task()
            self._place(self.task)
            if given is not None or not self._pucks_touching_arm():
                break
        else:
            # Falling through would hand back the last, penetrating layout --
            # exactly the failure the loop exists to prevent, delivered
            # silently. Put the world back the way it was found first, so that
            # a caller who catches this and moves the arm is not left standing
            # in the rejected layout.
            mujoco.mj_setState(self.model, self.data, before, STATE_SPEC)
            self.task = previous_task
            mujoco.mj_forward(self.model, self.data)
            raise RuntimeError(
                f"No layout clear of the arm in {PLACEMENT_ATTEMPTS} draws. The arm is "
                "not reset by reset(), so its current pose is blocking the spawn "
                "annulus; move it, or pass options={'reset_arm': True}."
            )

        self._rederive_from_state()
        return self._obs(), self._info()

    def _place(self, task: Task) -> None:
        for i in range(N_PUCKS):
            a = self._puck_qadr[i]
            self.data.qpos[a : a + 2] = task.puck_xy[i]
            self.data.qpos[a + 2] = task.puck_theta[i]
            dof = self._puck_dofadr[i]
            self.data.qvel[dof : dof + 3] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _pucks_touching_arm(self) -> bool:
        arm, pucks = set(self._arm_gid), set(self._puck_gid)
        return any(
            {c.geom1, c.geom2} & arm and {c.geom1, c.geom2} & pucks
            for c in self.data.contact[: self.data.ncon]
        )

    def _light_goal_zone(self) -> None:
        """The goal is perception, not a scalar: the target zone changes colour."""
        goal_zone = self._require_task().goal_zone
        for z, sid in enumerate(self._zone_sid):
            self.model.site_rgba[sid] = ZONE_LIT_RGBA if z == goal_zone else ZONE_DIM_RGBA

    # -- the tick ---------------------------------------------------------------

    def _apply_friction_field(self) -> None:
        """Scale each puck's frictionloss by the table's local roughness."""
        for i, dof in enumerate(self._puck_dofadr):
            a = self._puck_qadr[i]
            scale = friction_scale(self.data.qpos[a : a + 2])
            self.model.dof_frictionloss[dof : dof + 3] = self._friction_nominal[i] * scale

    def step(self, action):
        # Before anything moves: reset() never resets the arm or the clock, so
        # a tick taken by mistake cannot be taken back.
        self._require_task()
        self._apply_friction_field()
        torque = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0) * self._torque_limit
        self.data.ctrl[:] = torque
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        # reward 0.0 and both flags False are the contract, not placeholders.
        return self._obs(), 0.0, False, False, self._info()

    def render(self):
        """The top-down camera, independent of whether the observation carries it."""
        if self.render_mode is None:
            gym.logger.warn("render() was called with no render_mode set; returning None.")
            return None
        return self._camera_image()

    # -- the human's hand -------------------------------------------------------

    def disturb_arm(self, joint: int, impulse: float) -> None:
        """An impulse to one joint mid-task.

        The arm is disturbed by an impulse, never by a teleport: displacing
        `qpos` would have the world rewrite the arm's configuration, which is
        the one thing this env never does. An impulse is the world moving, so
        proprioception reports it the way it reports everything else and no new
        observation path exists.
        """
        _check_index(joint, len(ARM_JOINTS), "joint")
        mujoco.mj_forward(self.model, self.data)
        applied = np.zeros((1, self.model.nv))
        applied[0, self._arm_dofadr[joint]] = impulse
        delta_qvel = np.zeros((1, self.model.nv))
        mujoco.mj_solveM(self.model, self.data, delta_qvel, applied)
        self.data.qvel += delta_qvel[0]
        mujoco.mj_forward(self.model, self.data)

    def perturb(self, puck: int, xy, theta: float | None = None) -> None:
        """Teleport a puck mid-task."""
        _check_index(puck, N_PUCKS, "puck")
        a = self._puck_qadr[puck]
        self.data.qpos[a : a + 2] = np.asarray(xy, dtype=np.float64)
        if theta is not None:
            self.data.qpos[a + 2] = theta
        dof = self._puck_dofadr[puck]
        self.data.qvel[dof : dof + 3] = 0.0
        self._apply_friction_field()
        mujoco.mj_forward(self.model, self.data)

    def retarget(self, goal_puck: int | None = None, goal_zone: int | None = None) -> None:
        """Change what is wanted without touching the world."""
        task = self._require_task()
        self.task = Task(
            task.puck_xy,
            task.puck_theta,
            task.goal_puck if goal_puck is None else _check_index(goal_puck, N_PUCKS, "goal_puck"),
            task.goal_zone if goal_zone is None else _check_index(goal_zone, N_ZONES, "goal_zone"),
        )
        self._light_goal_zone()

    # -- what is derived from the state rather than part of it -------------------

    def _rederive_from_state(self) -> None:
        """Recompute the two things a write to the state leaves stale.

        The goal light follows the task, and the friction field follows where
        the pucks are. Neither is in `mjSTATE_INTEGRATION` and neither needs to
        be -- a pure function of the state is restored along with it -- but
        both have to be read back out of the state once something has written
        to it. `step()` does that anyway; doing it here as well means a forward
        taken before the next tick sees the world as it now is rather than as
        it had wandered to.

        The two writers are `reset()`, just below the placement loop, and
        `patchworks.sandbox.state.restore()`.
        """
        self._light_goal_zone()
        self._apply_friction_field()
        mujoco.mj_forward(self.model, self.data)

    def close(self):
        if not self._closed:
            if self._renderer is not None:
                self._renderer.close()
            self._closed = True
