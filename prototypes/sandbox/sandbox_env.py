"""PROTOTYPE — throwaway. Answers issue #3.

A continual Gymnasium environment: a 3-link planar arm pushing pucks into zones,
seen through proprioception, touch, and a top-down render.

Three things here are the actual proposal, and everything else is scaffolding:

1. `reset()` rearranges the world; it never resets the agent. Physics time is
   monotonic across the whole run, the arm keeps the configuration it was left in,
   and nothing in the observation says "an episode began". The agent finds out the
   world changed the way it finds out anything else: its predictions stop working.
2. There is no reward channel. `reward` is hard-wired to 0.0 and `terminated` to
   False. The goal reaches the agent as *perception* — the target zone lights up in
   the render — never as a scalar.
3. Privileged truth (object poses, goal identity, satisfaction) is in `info`, for
   logging and evaluation only. Feeding it to the agent defeats the sandbox.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

import gymnasium as gym
import mujoco
from gymnasium import spaces

XML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "arena.xml")

N_PUCKS = 3
N_ZONES = 3

# --- the sampler's space -------------------------------------------------------
# (puck, zone) pairs, plus where the target puck may start.
HELDOUT_PAIRS = {(0, 2), (2, 0)}          # red -> east zone, blue -> north zone
HELDOUT_SECTOR = (np.deg2rad(30.0), np.deg2rad(75.0))   # a wedge of the workspace
HELDOUT_SECTOR_MIN_R = 0.22

SPAWN_R = (0.15, 0.36)                     # annulus: pedestal at 0.08, ring wall at 0.52
ZONE_XY = np.array([[0.0, 0.30], [-0.26, -0.15], [0.26, -0.15]])
ZONE_RADIUS = 0.075

ZONE_DIM_RGBA = np.array([0.35, 0.35, 0.35, 0.35])
ZONE_LIT_RGBA = np.array([1.00, 0.85, 0.10, 0.85])


def _in_heldout_sector(xy: np.ndarray) -> bool:
    r = float(np.linalg.norm(xy))
    if r < HELDOUT_SECTOR_MIN_R:
        return False
    a = float(np.arctan2(xy[1], xy[0])) % (2 * np.pi)
    return HELDOUT_SECTOR[0] <= a <= HELDOUT_SECTOR[1]


@dataclass
class Task:
    """One sampled arrangement of the world plus what is wanted of it."""
    puck_xy: np.ndarray      # (N_PUCKS, 2)
    puck_theta: np.ndarray   # (N_PUCKS,)
    goal_puck: int
    goal_zone: int

    @property
    def pair(self) -> tuple[int, int]:
        return (self.goal_puck, self.goal_zone)


class PlanarPushSandbox(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 50}

    def __init__(
        self,
        split: str = "train",
        image_size: int = 64,
        frame_skip: int = 10,          # 0.002 * 10 -> 50 Hz control
        seed: int | None = None,
        render_obs: bool = True,       # off only for headless evaluation; the agent needs it
    ):
        assert split in ("train", "heldout_pair", "heldout_sector", "any")
        self.split = split
        self.image_size = image_size
        self.frame_skip = frame_skip
        self.render_obs = render_obs

        self.model = mujoco.MjModel.from_xml_path(XML)
        self.data = mujoco.MjData(self.model)
        self._renderer = mujoco.Renderer(self.model, image_size, image_size)

        self._rng = np.random.default_rng(seed)
        self.rearrangements = 0        # how many times the world changed under the agent
        self.task: Task | None = None

        self._jid = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
                     for n in ("j0", "j1", "j2")]
        self._puck_bid = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"puck_{i}")
                          for i in range(N_PUCKS)]
        self._puck_qadr = [self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"p{i}_x")]
            for i in range(N_PUCKS)]
        self._zone_sid = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, f"zone_{i}")
                          for i in range(N_ZONES)]
        self._tip_sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tip")
        gid = lambda n: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, n)
        self._link_gid = [gid(n) for n in ("g_link0", "g_link1", "g_link2", "g_tip")]
        self._puck_gid = [gid(f"g_puck_{i}") for i in range(N_PUCKS)]

        n_j = len(self._jid)
        self.observation_space = spaces.Dict({
            "qpos":  spaces.Box(-np.pi, np.pi, (n_j,), np.float32),
            "qvel":  spaces.Box(-np.inf, np.inf, (n_j,), np.float32),
            "touch": spaces.Box(0.0, np.inf, (3,), np.float32),
            "image": spaces.Box(0, 255, (image_size, image_size, 3), np.uint8),
        })
        self.action_space = spaces.Box(-1.0, 1.0, (self.model.nu,), np.float32)
        self._ctrl_scale = self.model.actuator_ctrlrange[:, 1].copy()

        mujoco.mj_forward(self.model, self.data)

    # -- observation ------------------------------------------------------------

    def _obs(self) -> dict:
        qpos = np.array([self.data.qpos[self.model.jnt_qposadr[j]] for j in self._jid])
        qvel = np.array([self.data.qvel[self.model.jnt_dofadr[j]] for j in self._jid])
        if self.render_obs:
            self._renderer.update_scene(self.data, camera="topdown")
            image = self._renderer.render()
        else:
            image = np.zeros((self.image_size, self.image_size, 3), np.uint8)
        return {
            "qpos": qpos.astype(np.float32),
            "qvel": qvel.astype(np.float32),
            "touch": np.asarray(self.data.sensordata[:3], dtype=np.float32),
            "image": image,
        }

    def _puck_pose(self, i: int) -> np.ndarray:
        a = self._puck_qadr[i]
        return np.array(self.data.qpos[a:a + 3])      # x, y, theta

    def _info(self) -> dict:
        """Privileged. For logging and evaluation, never for the agent."""
        poses = np.stack([self._puck_pose(i) for i in range(N_PUCKS)])
        t = self.task
        d = float(np.linalg.norm(poses[t.goal_puck, :2] - ZONE_XY[t.goal_zone]))
        return {
            "puck_pose": poses,
            "tip_xy": np.array(self.data.site_xpos[self._tip_sid][:2]),
            "goal_puck": t.goal_puck,
            "goal_zone": t.goal_zone,
            "goal_distance": d,
            "goal_satisfied": bool(d < ZONE_RADIUS),
            "split": self.split,
            "heldout_pair": t.pair in HELDOUT_PAIRS,
            "rearrangements": self.rearrangements,
            "sim_time": float(self.data.time),
        }

    # -- the sampler ------------------------------------------------------------

    def sample_task(self) -> Task:
        while True:
            xy, th = self._sample_layout()
            puck = int(self._rng.integers(N_PUCKS))
            zone = int(self._rng.integers(N_ZONES))
            held_pair = (puck, zone) in HELDOUT_PAIRS
            held_sector = _in_heldout_sector(xy[puck])
            if self.split == "any":
                return Task(xy, th, puck, zone)
            # the two axes stay separate: there is no value returning their union,
            # because a draw from it is attributable to neither axis.
            if self.split == "train" and not (held_pair or held_sector):
                return Task(xy, th, puck, zone)
            if self.split == "heldout_pair" and held_pair and not held_sector:
                return Task(xy, th, puck, zone)
            if self.split == "heldout_sector" and held_sector and not held_pair:
                return Task(xy, th, puck, zone)

    def _sample_layout(self) -> tuple[np.ndarray, np.ndarray]:
        radii = self.model.geom_size[self._puck_gid, 0]
        while True:
            a = self._rng.uniform(0, 2 * np.pi, N_PUCKS)
            r = self._rng.uniform(*SPAWN_R, N_PUCKS)
            xy = np.stack([r * np.cos(a), r * np.sin(a)], axis=1)
            gaps = [np.linalg.norm(xy[i] - xy[j]) - radii[i] - radii[j]
                    for i in range(N_PUCKS) for j in range(i + 1, N_PUCKS)]
            far_from_zones = min(
                np.linalg.norm(xy[i] - ZONE_XY[z]) for i in range(N_PUCKS) for z in range(N_ZONES))
            if min(gaps) > 0.03 and far_from_zones > ZONE_RADIUS + 0.04:
                return xy, self._rng.uniform(-np.pi, np.pi, N_PUCKS)

    # -- the continual reset ----------------------------------------------------

    def reset(self, *, seed=None, options=None):
        """Rearrange the world. The agent is not reset, and neither is the clock."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        options = options or {}

        if options.get("reset_arm", False):      # off by default: continual means continual
            for j in self._jid:
                self.data.qpos[self.model.jnt_qposadr[j]] = 0.0
                self.data.qvel[self.model.jnt_dofadr[j]] = 0.0

        # The arm is not reset, so its pose is wherever the agent left it. A layout
        # that intersects that pose starts the world inside a penetration, and the
        # solver launches the puck across the arena. Place, then check, then re-place.
        task = options.get("task")
        for _ in range(64):
            self.task = task or self.sample_task()
            self._place(self.task)
            if task is not None or not self._pucks_touching_arm():
                break

        self._light_goal_zone()
        self.rearrangements += 1
        mujoco.mj_forward(self.model, self.data)
        return self._obs(), self._info()

    def _place(self, task: Task) -> None:
        for i in range(N_PUCKS):
            a = self._puck_qadr[i]
            self.data.qpos[a:a + 2] = task.puck_xy[i]
            self.data.qpos[a + 2] = task.puck_theta[i]
            dof = self.model.jnt_dofadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"p{i}_x")]
            self.data.qvel[dof:dof + 3] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _pucks_touching_arm(self) -> bool:
        arm = set(self._link_gid)
        pucks = set(self._puck_gid)
        return any({c.geom1, c.geom2} & arm and {c.geom1, c.geom2} & pucks
                   for c in self.data.contact[:self.data.ncon])

    def _light_goal_zone(self) -> None:
        """The goal is perception, not a scalar: the target zone changes colour."""
        for z, sid in enumerate(self._zone_sid):
            self.model.site_rgba[sid] = (
                ZONE_LIT_RGBA if z == self.task.goal_zone else ZONE_DIM_RGBA)

    # -- the tick ---------------------------------------------------------------

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self.data.ctrl[:] = a * self._ctrl_scale
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        # reward 0.0 and terminated False are the contract, not placeholders.
        return self._obs(), 0.0, False, False, self._info()

    # -- the human's hand -------------------------------------------------------

    def perturb(self, puck: int, xy, theta: float | None = None) -> None:
        """Teleport a puck mid-task. This is the acceptance demo's interface."""
        a = self._puck_qadr[puck]
        self.data.qpos[a:a + 2] = np.asarray(xy, dtype=np.float64)
        if theta is not None:
            self.data.qpos[a + 2] = theta
        dof = self.model.jnt_dofadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"p{puck}_x")]
        self.data.qvel[dof:dof + 3] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def retarget(self, goal_puck: int | None = None, goal_zone: int | None = None) -> None:
        """Change the goal without touching the world — the other half of the demo."""
        t = self.task
        self.task = Task(t.puck_xy, t.puck_theta,
                         t.goal_puck if goal_puck is None else goal_puck,
                         t.goal_zone if goal_zone is None else goal_zone)
        self._light_goal_zone()

    # -- snapshots (continual learning has no episode boundary to restart from) --

    def snapshot(self) -> dict:
        return {
            "qpos": self.data.qpos.copy(), "qvel": self.data.qvel.copy(),
            "ctrl": self.data.ctrl.copy(), "time": float(self.data.time),
            "task": self.task, "rng": self._rng.bit_generator.state,
            "rearrangements": self.rearrangements,
        }

    def restore(self, s: dict) -> None:
        self.data.qpos[:] = s["qpos"]
        self.data.qvel[:] = s["qvel"]
        self.data.ctrl[:] = s["ctrl"]
        self.data.time = s["time"]
        self.task = s["task"]
        self._rng.bit_generator.state = s["rng"]
        self.rearrangements = s["rearrangements"]
        self._light_goal_zone()
        mujoco.mj_forward(self.model, self.data)

    def render(self):
        return self._obs()["image"]

    def close(self):
        self._renderer.close()
