"""The full state: taking it, and putting it back.

`docs/spec/03-the-sandbox.md`, *The Gymnasium contract, made continual*, and
[ADR-0001](../../../docs/adr/0001-continual-learning-applies-to-the-adapting-surface.md).

There is no episode boundary to restart from, so this is where reproducibility
comes from, and it is what the acceptance demo's repeated runs are built on: a
trial starts from a restore, not a reset.

**A restore is not a reset**, and this module is where that shows up as
structure rather than as a paragraph. `reset()` is in-band -- it rearranges the
world, the agent lives through it, and it finds out by being wrong -- and so
are `perturb()` and `disturb_arm()`: the world moves and the arm feels it. A
restore rewinds the whole universe, the agent's own adapting surface included,
so there is no tick at which a cell could observe one. It is the experimenter
standing outside the run, not the world acting inside it, and it is therefore a
function taking an env rather than another method sitting next to `step()`.

Keeping it off the env is also the only form that survives the way an env is
normally built. `gymnasium.make(ENV_ID)` hands back an `OrderEnforcing`
wrapper, which forwards no attribute the `Env` contract does not name, so an
`env.snapshot()` would raise `AttributeError` on everything but a bare
`PlanarPushSandbox`. These take the wrapper and unwrap it themselves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import gymnasium as gym
import mujoco
import numpy as np

if TYPE_CHECKING:  # importing env at runtime would close a cycle: env imports this
    from patchworks.sandbox.env import PlanarPushSandbox, Task

#: The physics half of the state, named by the engine constant rather than by
#: an enumeration of fields. MuJoCo defines `mjSTATE_INTEGRATION` as the entire
#: set of inputs to the forward dynamics, so it tracks the model: an
#: enumeration drifts the moment the arena gains a feature, and it drifts
#: *silently*, as a trajectory that diverges rather than an error. The field
#: the obvious enumeration omits is `qacc_warmstart`, and MuJoCo's docs flag
#: exactly this case -- warmstarts matter when loading a non-initial state,
#: since the initial state is always cold-started. **Every restore here is a
#: non-initial-state load**, because there is no episode boundary; the
#: load-bearing case is the only case.
#:
#: Three of the fields the constant covers are inert in *this* arena and are
#: the reason to name it anyway: `act` is empty (the actuators are direct-drive
#: `motor`), and `qfrc_applied` / `xfrc_applied` are zero only because
#: `perturb()` teleports a puck by writing `qpos`. Implement that hand as an
#: applied force instead and the force becomes state -- a change to the human's
#: hand would silently break restore, in a file nobody would think to reread.
STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION


# eq=False for `Task`'s reason: `physics` is an array, so a generated __eq__
# would raise rather than compare.
@dataclass(frozen=True, eq=False)
class Snapshot:
    """Everything needed to rewind, and nothing that can be derived from it.

    `physics` is `STATE_SPEC`'s vector, whatever the arena makes that. `task`
    and `rng` are the two things MuJoCo does not know about: what is wanted of
    the world, and where the sampler had got to.

    The friction field is deliberately absent. It is a pure function of puck
    position, so restoring the state restores the field with it -- and holding
    a copy would be worse than redundant, because a stored number and a
    recomputed one can disagree.
    """

    physics: np.ndarray
    task: Task
    rng: dict


def snapshot(env: PlanarPushSandbox | gym.Wrapper) -> Snapshot:
    """Take the full state of `env`. Wrapped or bare, either is fine."""
    env = env.unwrapped
    # Refuse here rather than at the restore that cannot light a goal zone: a
    # snapshot of a world nobody has arranged is not a state worth returning to.
    task = env._require_task()
    physics = np.empty(mujoco.mj_stateSize(env.model, STATE_SPEC))
    mujoco.mj_getState(env.model, env.data, physics, STATE_SPEC)
    # The demo holds forty of these down one continual run, so the copy is
    # sealed: a snapshot that drifts while it is being held is a restore that
    # lands somewhere nobody chose.
    physics.flags.writeable = False
    return Snapshot(physics=physics, task=task, rng=env.np_random.bit_generator.state)


def _refuse_a_wrapper_that_has_not_seen_a_reset(env: PlanarPushSandbox | gym.Wrapper) -> None:
    """Refuse a restore the env would accept and then decline to tick on.

    Two things can disagree about whether a run has started. The env knows
    whether a layout was ever drawn -- that is `task`, and `_require_task()`
    asks it. The wrappers around it know whether `reset()` came *through them*:
    `OrderEnforcing` refuses a `step()` until it has seen one, and it can see
    neither a restore going around it nor a `reset()` called on
    `env.unwrapped`. Since every hand -- `perturb`, `disturb_arm`, `retarget`
    -- is reachable only as `env.unwrapped.<method>()`, resetting the same way
    is an easy slip to make.

    Taking the state and then declining to tick is the worst of the outcomes,
    because the remedy for it, another `reset()`, throws the restored state
    away. So the chain is walked and asked first, by `OrderEnforcing`'s public
    `has_reset` and then by the private flag behind it, falling through to
    "started" for the wrappers that answer neither.

    **What this cannot see**: `restore(env.unwrapped, state)` hands over a bare
    env, so there is no chain to walk and no refusal to make. Gymnasium gives
    an env no way back to its wrappers, so nothing in here can detect that --
    and it is the same slip, made one call earlier. Pass the env you are
    holding.
    """
    node = env
    while isinstance(node, gym.Wrapper):
        started = getattr(node, "has_reset", getattr(node, "_has_reset", True))
        if started is False:
            raise RuntimeError(
                f"{type(node).__name__} has not seen a reset() and would refuse the "
                "next step(), whatever this restore writes. Call reset() on the env "
                "you are holding rather than on env.unwrapped, then restore: the "
                "layout it draws is overwritten a moment later."
            )
        node = node.env


def restore(env: PlanarPushSandbox | gym.Wrapper, state: Snapshot) -> None:
    """Rewind `env` to `state`. Invisible from inside: no tick observes one.

    A restore rewinds a run that is already going, so a run has to have been
    started -- through the env being held, wrappers and all. Both halves of
    that are refused up front rather than half-rewound; see
    `_refuse_a_wrapper_that_has_not_seen_a_reset`. Reset a fresh env first: the
    layout that draws is overwritten by the restore a moment later, and reset
    never touches the arm anyway.

    "Rewinds the whole universe" means everything at `env.unwrapped` and
    nothing above it. A wrapper's own bookkeeping is not state this can reach,
    so it runs on. That is worth knowing for the one step limit the env cannot
    refuse -- `gymnasium.make(id, max_episode_steps=n)` wraps `TimeLimit`
    outside, where nothing the env can read mentions the limit, and its count
    survives a restore, so `truncated=True` lands `n` ticks after the *run*
    began rather than after the trial did. Every standard loop resets on that,
    which here rearranges the world mid-trial. Register and construct without
    one, as `patchworks.sandbox` does.
    """
    _refuse_a_wrapper_that_has_not_seen_a_reset(env)
    env = env.unwrapped
    env._require_task()
    mujoco.mj_setState(env.model, env.data, state.physics, STATE_SPEC)
    env.task = state.task
    env.np_random.bit_generator.state = state.rng
    env._rederive_from_state()
