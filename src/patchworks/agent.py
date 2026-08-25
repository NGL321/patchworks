"""The agent: the graph, the world, and the ordering between them.

This is the tracer bullet — world to graph to torque to world. The graph's own
half of a tick is :mod:`patchworks.tick`; what this module adds is the third
thing `docs/spec/02-tick-semantics.md` insists is **not** a phase:

    Whatever is outside the sheaf writes its boundary cells after the
    message-passing phase, as the last word in a tick.

Nothing computes during it and no cell reads another; it is a statement about
ordering, and the ordering is what matters. Two things fall out of it here
without any machinery:

* **The drive's standing assertion actually stands.** Reconciliation moves the
  drive boundary cell's node stalk like any other, and the write below restores
  it before it next speaks. So disagreement on a drive edge is never reduced
  from the drive's side; only the core cell can move, which is the motor-edge
  signature falling out of tick order rather than being stipulated.
* **The motor pathway is untouched.** The actuator boundary cell's three
  *commanded* components appear nowhere in :meth:`Agent.write` — nobody outside
  writes them — so reconciliation fills them and the world reads them, which is
  how a command reaches the arm at all. A blanket "boundary stalks are not
  reconciled" rule would have severed that; the ordering rule needs no
  exemption and no per-component bookkeeping.

**Nothing here compresses across slices.** The membership rule
(`docs/spec/04-action-and-the-boundary.md`, *The membership rule*) bans an
out-of-graph module that does, because that specific act is what cells exist to
do. What this module does instead is layout and unit conversion: a 4×4 patch of
the render laid into a patch cell's 48 components raw, and radians, radians per
second, a contact scalar and a normalised torque laid into their own stalks. The
compression is the patch cell's own 48 → 8 restriction, performed by a cell,
inside the graph, costing a tick.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np
import torch

from .graph import Cell, CellKind, Dome, build_graph
from .tick import DEFAULT_GAMMA, Sheaf

__all__ = ["DRIVE_ASSERTION", "PIXEL_SCALE", "Agent", "TickOutcome", "run"]

#: What the drive boundary cell asserts, every tick, forever. **Chosen here,
#: not recorded**: the record fixes the stalk's width and its content — one
#: dimension carrying *valence, not specification* — but not the number. It has
#: to be a non-zero constant. Constant because a drive is a standing assertion
#: and never a schedule; non-zero because a restriction of zero is a belief no
#: apex cell can ever disagree with, which would make the drive inert and the
#: dark room problem unanswered. One, for the same reason a satisfied assertion
#: is naturally read as true.
DRIVE_ASSERTION = 1.0

#: The render arrives as `uint8`. Dividing by 255 is unit conversion at the
#: environment contract's seam, which `docs/spec/01-cell-and-sheaf.md` puts
#: outside the sheaf's job — not a compressor, which is what the membership rule
#: bans: no patch is mixed with any other and every pixel survives.
PIXEL_SCALE = 1.0 / 255.0


@dataclass(frozen=True)
class TickOutcome:
    """What one whole tick did. Logging and instrumentation only; no cell reads it."""

    command: np.ndarray
    """The three components the world read off the actuator boundary cell,
    before the arm's limits had anything to say about them."""

    applied: np.ndarray
    """What the arm actually applied — post-clip. The efference copy, written
    back so that a motor edge carries ordinary disagreement rather than being
    the one edge in the graph with none. Derived from the action space rather
    than read out of the physics, which keeps this seam a Gymnasium contract
    rather than a MuJoCo one; that the two agree is asserted in
    `tests/test_agent.py`, at the saturating command where they would part."""

    observation: dict
    """The world's answer, as it was written into the sensory boundary cells."""

    info: dict
    """Privileged truth. For logging and the acceptance demo, never for the agent."""


class Agent:
    """One dome, one body, and the world it is wired to.

    Construction reads each boundary cell's **construction layout** exactly
    once, to work out which patch cell covers which patch of the render and
    which proprioceptive cell reports which joint. That is what the layout is
    for — retinotopy falls out of the index rather than being designed — and it
    is a construction act: after this constructor returns, every tick reads a
    precomputed index array and no cell's :class:`~patchworks.graph.CellIndex`
    is touched again.
    """

    def __init__(
        self,
        env: Any,
        *,
        dome: Dome | None = None,
        gamma: float = DEFAULT_GAMMA,
        generator: torch.Generator | None = None,
        sheaf: Sheaf | None = None,
    ) -> None:
        self.dome = dome if dome is not None else build_graph()
        self.sheaf = (
            sheaf
            if sheaf is not None
            else Sheaf(self.dome, gamma=gamma, generator=generator)
        )
        # Refused for the same reason Sheaf refuses mismatched maps: the write
        # tables below are built from this dome's cell ids against the sheaf's
        # layout, so a sheaf on another graph would write the render into the
        # wrong components rather than fail.
        if self.sheaf.dome is not self.dome:
            raise ValueError("the sheaf was built on a different dome")
        self.env = env
        spec = self.dome.spec

        image_space = env.observation_space["image"]
        height, width = image_space.shape[0], image_space.shape[1]
        channels = image_space.shape[2]
        side, remainder = divmod(width, spec.patch_grid)
        if height != width or remainder or side * spec.patch_grid != height:
            raise ValueError(
                f"the sensory tiling is {spec.patch_grid}x{spec.patch_grid} cells over "
                f"a square render; a {height}x{width} one does not tile"
            )
        if side * side * channels != spec.patch_stalk:
            raise ValueError(
                f"a {side}x{side}x{channels} patch is {side * side * channels} numbers "
                f"and the patch cell's node stalk is {spec.patch_stalk}. The world "
                "writes the stalk raw, so the two are one number."
            )
        self.patch_side = side

        joints = env.action_space.shape[0]
        if joints != spec.joints:
            raise ValueError(
                f"the arena has {joints} joints and the dome was built for {spec.joints}"
            )
        self.joints = joints
        #: How many of the actuator's components the world reads. The rest are
        #: the efference copy it writes back.
        self.commanded = spec.actuator_stalk - joints
        if self.commanded != joints:
            raise ValueError(
                f"the actuator's node stalk is {spec.actuator_stalk} for {joints} "
                "joints; it holds one commanded and one efference component each"
            )
        # A boundary cell's node stalk is the world's shape, so these widths
        # are facts about what writes them: angle and velocity, one contact
        # scalar, one valence. A different width would be a different world,
        # and the write tables below lay out components by position.
        for built, expected, what in (
            (spec.proprioceptive_stalk, 2, "proprioceptive (angle, velocity)"),
            (spec.touch_stalk, 1, "touch (one contact scalar)"),
            (spec.drive_stalk, 1, "drive (valence, not specification)"),
        ):
            if built != expected:
                raise ValueError(
                    f"the {what} node stalk is {expected} wide in this world, and "
                    f"the dome was built for {built}"
                )

        # float32, like the command they clip, so the efference copy and the
        # command are commensurable down to the last bit.
        self.action_low = np.asarray(env.action_space.low, dtype=np.float32)
        self.action_high = np.asarray(env.action_space.high, dtype=np.float32)

        self._build_write_tables(width, channels)

    # -- construction: who is where --------------------------------------------

    def _build_write_tables(self, image_width: int, channels: int) -> None:
        """The index arrays the external write runs on, built once from the layout."""
        layout = self.sheaf.layout
        patches: list[Cell] = []
        proprioceptive: list[Cell] = []
        touch: list[Cell] = []
        actuator: list[Cell] = []
        drives: list[Cell] = []
        for cell_id in self.dome.boundary:
            cell = self.dome.cells[cell_id]
            {
                CellKind.PATCH: patches,
                CellKind.PROPRIOCEPTIVE: proprioceptive,
                CellKind.TOUCH: touch,
                CellKind.ACTUATOR: actuator,
                CellKind.DRIVE: drives,
            }[cell.kind].append(cell)

        if len(actuator) != 1 or len(drives) != 1:
            raise ValueError(
                "this proof of concept has exactly one actuator boundary cell and "
                f"one drive boundary cell; got {len(actuator)} and {len(drives)}"
            )
        self.actuator_cell = actuator[0].id
        self.drive_cell = drives[0].id
        self._actuator_slice = layout.slice(self.actuator_cell)
        self._commanded_slice = slice(
            self._actuator_slice.start, self._actuator_slice.start + self.commanded
        )
        self._efference_slice = slice(
            self._commanded_slice.stop, self._actuator_slice.stop
        )
        self._drive_slice = layout.slice(self.drive_cell)

        # Retinotopy: patch cell (r, c) owns the block of the render at
        # (r, c) of the tiling, read row-major over rows, columns and channels.
        # The order is arbitrary and only has to be *fixed*: the cell learns
        # what its components mean, and the one thing it cannot survive is them
        # meaning something different from one tick to the next.
        side = self.patch_side
        pixels = []
        stalk_positions = []
        for cell in patches:
            r, c = cell.index.position
            block = [
                ((r * side + dr) * image_width + (c * side + dc)) * channels + ch
                for dr in range(side)
                for dc in range(side)
                for ch in range(channels)
            ]
            pixels.append(block)
            base = layout.offsets[cell.id]
            stalk_positions.append(list(range(base, base + cell.stalk)))
        self._patch_pixels = torch.tensor(pixels, dtype=torch.long)
        self._patch_stalks = torch.tensor(stalk_positions, dtype=torch.long)

        # The somatomotor sensors tile the column's index two to a joint, so a
        # cell's ordinal is which joint it reports on.
        self._proprioceptive = torch.tensor(
            [
                [layout.offsets[cell.id], layout.offsets[cell.id] + 1]
                for cell in sorted(proprioceptive, key=lambda c: c.index.position)
            ],
            dtype=torch.long,
        )
        self._touch = torch.tensor(
            [
                layout.offsets[cell.id]
                for cell in sorted(touch, key=lambda c: c.index.position)
            ],
            dtype=torch.long,
        )
        if len(self._proprioceptive) != self.joints or len(self._touch) != self.joints:
            raise ValueError(
                f"{len(self._proprioceptive)} proprioceptive and {len(self._touch)} "
                f"touch boundary cells for {self.joints} joints"
            )

    # -- the world reads -------------------------------------------------------

    def command(self) -> np.ndarray:
        """The actuator boundary cell's commanded components, as an action.

        There is no read-out map here and no decode path to torque that bypasses
        a stalk: this is a slice of a node stalk that reconciliation filled, and
        the arm reads it the way the world reads any motor edge's far endpoint.
        It is deliberately not clipped — :meth:`act` clips on the way to the
        arm, and the difference between what was asked and what the body would
        do is exactly the disagreement the efference copy carries back.
        """
        commanded = self.sheaf.stalks[self._commanded_slice]
        return commanded.detach().clone().numpy().astype(np.float32, copy=False)

    # -- the world writes ------------------------------------------------------

    def write(self, observation: dict, applied: np.ndarray | None = None) -> None:
        """The external write: the last word in a tick.

        Sensory patch cells, the proprioceptive and touch cells, the actuator's
        efference components, and the drive's standing assertion. The actuator's
        *commanded* components are conspicuously absent — nobody outside writes
        them — and so is anything to do with a chart: the world only ever
        touches node stalks.
        """
        stalks = self.sheaf.stalks
        with torch.no_grad():
            image = torch.as_tensor(
                np.ascontiguousarray(observation["image"])
            ).reshape(-1)
            stalks[self._patch_stalks] = (
                image[self._patch_pixels].to(stalks.dtype) * PIXEL_SCALE
            )
            stalks[self._proprioceptive] = torch.stack(
                (
                    torch.as_tensor(observation["qpos"], dtype=stalks.dtype),
                    torch.as_tensor(observation["qvel"], dtype=stalks.dtype),
                ),
                dim=-1,
            )
            stalks[self._touch] = torch.as_tensor(observation["touch"], dtype=stalks.dtype)
            if applied is not None:
                stalks[self._efference_slice] = torch.as_tensor(
                    applied, dtype=stalks.dtype
                )
            stalks[self._drive_slice] = DRIVE_ASSERTION

    def observe(self, observation: dict) -> None:
        """The world's last word before the first tick: write, with nothing applied yet.

        A reset is in-band and the agent lives through it, so this is not a
        special case of anything — it is the same external write, on a tick
        whose two phases have not happened.
        """
        self.write(observation)

    # -- a whole tick ----------------------------------------------------------

    def tick(self) -> TickOutcome:
        """One tick, entire: two phases, then the world, in that order.

        The phases run under `no_grad` and the external write is plain array
        traffic, so the guard is checked once more at the end — after the last
        thing that touches a node stalk, which is where "nothing leaving the
        tick carries a `grad_fn`" is actually a statement about the tick.
        """
        self.sheaf.tick()
        outcome = self.act(self.command())
        self.sheaf.assert_no_tape()
        return outcome

    def act(self, command: np.ndarray) -> TickOutcome:
        """The world's half of the ordering: it reads, it moves, and it answers.

        Split out from :meth:`tick` because it is the half with no graph in it
        at all — the arm's limits clip what was asked, the physics runs, and
        what comes back is written to the boundary cells as the tick's last
        word. Nothing here computes anything a cell reads.

        The clip is against the environment's declared action space, so the
        efference copy is a statement in the contract's own terms rather than a
        reading of the engine's registers. The same clipped value is what the
        env is stepped with, so nothing leaves here outside the declared space
        and a wrapper that enforces the contract sees a legal action. The
        sandbox clips identically on its own, so this decides no trajectory —
        only whose statement the action is. See :class:`TickOutcome`.
        """
        applied = np.clip(command, self.action_low, self.action_high)
        # A copy, because the same array is the efference copy a moment later
        # and a wrapper that normalised its action in place would otherwise
        # rewrite what the actuator cell reports having done.
        observation, _reward, _terminated, _truncated, info = self.env.step(
            applied.copy()
        )
        self.write(observation, applied)
        return TickOutcome(
            command=command, applied=applied, observation=observation, info=info
        )


def run(agent: Agent, ticks: int, *, seed: int | None = None) -> Iterator[TickOutcome]:
    """Arrange the world once, then run `ticks` whole ticks, yielding each.

    Yields, because a run has no episodes to batch into and a caller may want to
    watch it, disturb it, or stop it. The `reset()` is the world being arranged,
    not the agent being restarted: the agent is not reset by anything, ever.

    The arranging happens **when this is called**, not on the first `next()`, so
    that a caller who builds the iterator and then looks at the world sees the
    world the run will start from. A bare generator would have deferred it, and
    two interleaved runs would then have rearranged the world at surprising
    moments -- which in a sandbox with no episode boundary is a rearrangement
    mid-trajectory rather than a restart.
    """
    observation, _info = agent.env.reset(seed=seed)
    agent.observe(observation)

    def ticking() -> Iterator[TickOutcome]:
        for _ in range(ticks):
            yield agent.tick()

    return ticking()
