"""The sandbox: the world the agent lives in, and the surfaces it meets it through.

See `docs/spec/03-the-sandbox.md`. Importing this module registers the env with
Gymnasium under `ENV_ID`, with `max_episode_steps=None` -- the env refuses a
spec-level limit, and `EnvSpec.nondeterministic` is deliberately not set,
because it would assert something untrue about the physics in order to quiet a
check about `reset()`.

`snapshot()` and `restore()` are re-exported from `patchworks.sandbox.state`,
where they live off the env class on purpose: a restore is an experimenter's
tool rather than an operation the env offers, and nothing on the env's contract
surface would survive `gymnasium.make()`'s wrapper anyway.
"""

from gymnasium.envs.registration import register, registry

from patchworks.sandbox.env import (
    ARENA_XML,
    CONTROL_HZ,
    ENTRY_POINT,
    ENV_ID,
    FRAME_SKIP,
    HELDOUT_PAIRS,
    HELDOUT_SECTOR,
    HELDOUT_SECTOR_MIN_R,
    IMAGE_SIZE,
    N_PUCKS,
    N_ZONES,
    PATCH_PX,
    RENDER_CHANNELS,
    SPAWN_R,
    SPLITS,
    ZONE_RADIUS,
    ZONE_XY,
    BlockedAnnulusError,
    PlanarPushSandbox,
    SpecLimitError,
    Task,
    friction_scale,
    in_heldout_sector,
)
from patchworks.sandbox.state import (
    STATE_SPEC,
    Snapshot,
    restore,
    snapshot,
)

if ENV_ID not in registry:
    register(id=ENV_ID, entry_point=ENTRY_POINT, max_episode_steps=None)

__all__ = [
    "ARENA_XML",
    "CONTROL_HZ",
    "ENTRY_POINT",
    "ENV_ID",
    "FRAME_SKIP",
    "HELDOUT_PAIRS",
    "HELDOUT_SECTOR",
    "HELDOUT_SECTOR_MIN_R",
    "IMAGE_SIZE",
    "N_PUCKS",
    "N_ZONES",
    "PATCH_PX",
    "RENDER_CHANNELS",
    "SPAWN_R",
    "SPLITS",
    "STATE_SPEC",
    "ZONE_RADIUS",
    "ZONE_XY",
    "BlockedAnnulusError",
    "PlanarPushSandbox",
    "Snapshot",
    "SpecLimitError",
    "Task",
    "friction_scale",
    "in_heldout_sector",
    "restore",
    "snapshot",
]
