"""Koopman lift: does a cell's piece linearise? The go/no-go for ticket #126.

`#126` asks one question and holds nothing else:

    Does a `k`-dimensional linear operator, on top of a frozen random lift,
    model one cell's piece of a dynamical system well enough to be worth the
    conversion?

    python benchmarks/koopman_lift.py

Offline, against the sandbox that already runs. No graph, no sheaf, no learning
rule, no reconciliation -- one cell's worth of world, in isolation. Nothing in
the spec changes because this script ran; it exists so that a decision has
numbers under it.

**The piece.** A 4x4 tile of the 64x64 render, which is a sensory boundary
cell's stalk exactly as `graph.py` sizes it: `patch_stalk = 48`. The 16x16
tiling is the dome's own (`06-graph-topology.md`), so a "tile" here is not an
analysis convenience -- it is the piece a cell actually gets.

**The models**, all predicting the same thing (the tile's next stalk), scored
on the same held-out ticks, so that their errors are comparable. The ticket
names the first three; the last two are here because without them the first
three cannot be read:

* **(a) raw DMD** -- an affine linear operator on the 48-dimensional stalk, no
  lift. 2352 parameters. If (b) does not beat this, the lift earns nothing and
  the whole idea is decoration.
* **(b) the variant** -- `encode` (frozen, random, at the body's own widths)
  lifts the stalk to `k`; an affine `K` advances it; an affine readout `C`
  brings it back to the stalk. `K` is fit by least squares on one-step pairs in
  lifted coordinates, which is EDMD with a fixed dictionary.
* **(c) a small nonlinear step** -- the identical lift and the identical
  readout, with `K` replaced by the body's own `step` shape (`k -> step_width
  -> k`, ReLU) trained by Adam. The upper bound of what a predictor of this
  size achieves through this bottleneck. **The gap between (b) and (c) is the
  price of linearity**, and it is a clean price precisely because only one
  component differs.
* **(d) the same shape with no lift** -- one hidden layer on the raw stalk, no
  bottleneck and no linear readout. It separates *the lift is too narrow* from
  *the task is persistence-dominated*, which (a) to (c) cannot tell apart.
* **(e) the lift's floor** -- `C` applied to the **true** next chart. No
  operator and no prediction: what is lost by going through the lift and back
  at all. (b) cannot be read without it, because (b)'s error is reconstruction
  error plus operator error and only their sum is otherwise visible.

`C` is fit once per (tile, lift) and shared by (b), (c) and (e). (b) and (c)
are both scored in stalk coordinates through that same `C`, so the comparison
is symmetric in what each is allowed to spend and in what each is asked for.

**(c) and (d) are started at the linear fit they bound** (:func:`_start_from_linear`),
so each is an upper bound on its linear counterpart by construction. Randomly
initialised they are not: measured here, a random (c) finished five times worse
than the (b) it exists to bound, which measures Adam and not linearity.

**Every fit is selected on its own blocks.** Ridge terms and stopping points
come off a validation set that is never scored, and the fit/validation/score
blocks are interleaved rather than split head-and-tail -- see :func:`_split`,
where the head-and-tail split that came first is recorded along with how badly
it misread.

**Three input modes, not one:**

* `plain` -- the instantaneous stalk, lifted. The sweep the ticket's body asks
  for.
* `delay` -- comment 11's item. A Hankel stack of `DELAYS` consecutive stalks,
  lifted. HAVOK pays the lift's dimension **in time rather than in width**; if
  the knee moves left here, the most expensive move in the design (widening
  `k`, last on #14's ladder) is avoided by machinery already built.

  **Read this mode on `b/e`, never on the raw ratio.** It compresses
  `48 * DELAYS` inputs into the same `k` that `plain` compresses 48 into, and
  is then scored on reconstructing the current stalk only -- so it carries a
  four-times harsher bottleneck and its raw error is not comparable with
  `plain`'s. The raw column says delay is worse; that is the bottleneck
  talking, not the operator. `b/e` divides the bottleneck out.
* `recurrent` -- the same claim in the architecture's own shape. `encode` runs
  *recurrently*, its output chart persisting into the next tick's fuse, which
  is what a cell actually does. A recurrent chart is an implicit delay
  embedding, so this is the delay claim tested without a Hankel stack.

**Two tile classes, never averaged together** (step 5). A tile is a *contact*
tile if an arm-puck contact projected into it on at least `CONTACT_TICKS`
ticks, and a *free* tile otherwise. Contact is the case most likely to fail and
the one the sandbox exists to contain. Tiles below `VARIANCE_FLOOR` are dropped
before either class is formed: a tile of unchanging background linearises
perfectly and says nothing.

**The eccentric puck separately** (the last readout). Puck 1's orientation
enters its own equations of motion and is invisible in the render, so if any
tile's dynamics resist a linear lift that is the one.

**Batched over tiles**, in this repository's `[cells, ...]` idiom: every tile
is a cell of one population, every lift is one `CellBody.encode` call over all
of them, and every least-squares fit is one batched solve. The alternative --
a loop over tiles -- runs the same arithmetic an order of magnitude slower.

**What is not here.** The cylinder wake (step 7, comment 12) does not run: see
:func:`wake_status`. It was never a pass condition -- "the wake does not gate
anything" -- but its absence means every readout below is self-referential, and
that is a real limit on what this script can settle.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass, field

import mujoco
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from patchworks.body import (  # noqa: E402
    BodyShape,
    CellBiases,
    CellBody,
    hidden_width,
)
from patchworks.sandbox import (  # noqa: E402
    CONTROL_HZ,
    BlockedAnnulusError,
    PlanarPushSandbox,
)

# --- the world, and how it is driven -------------------------------------------

#: The dome's tiling of the 64x64 render (`06-graph-topology.md`).
TILE = 4
GRID = 64 // TILE
#: `graph.py`'s `patch_stalk`. 4 x 4 x 3 channels, and not a coincidence.
PATCH_STALK = TILE * TILE * 3

#: Ticks per collected segment, and how many segments each driver contributes.
#: `reset()` rearranges the world, so a segment boundary is a discontinuity and
#: no one-step pair is ever formed across one.
SEGMENT_TICKS = 900
SEGMENTS_PER_DRIVER = 4

#: Smooth babble, not white noise -- `prototypes/sandbox/probe.py`'s constant.
BABBLE_SMOOTHING = 0.9

# --- the sweep ------------------------------------------------------------------

K_SWEEP = (8, 12, 16, 24, 32)
#: How many stalks the Hankel stack holds. Four ticks at 50 Hz is 80 ms, just
#: past the demo's fastest perturbation horizon (3 ticks): a delay embedding
#: shorter than the horizon it must resolve would be measuring the wrong thing.
DELAYS = 4
INPUT_MODES = ("plain", "delay", "recurrent")

HORIZONS = (1, 5, 25, 50)

#: Ticks per block, and how many are dropped at each block's end. Every segment
#: is chopped into blocks and the blocks are dealt out to fit, validation and
#: score sets in rotation -- see :func:`_split` for why the obvious alternative
#: (fit on the head of a segment, score on its tail) measures the wrong thing.
BLOCK_TICKS = 100
BLOCK_GAP = 10
MINIMUM_BLOCK = 20

#: Tikhonov terms tried on every least-squares fit, relative to the mean
#: diagonal of the design's Gram matrix. **Selected per tile on the validation
#: blocks**, never on the scored ones: a 48-dimensional stalk on a 4x4 tile is
#: collinear enough that the choice moves the answer by two orders of magnitude,
#: so fixing it by hand would be choosing the result.
RIDGE_GRID = (1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)

#: Adam steps and learning rate for the trained models. Full-batch, and jointly
#: over tiles: Adam is elementwise, so one optimiser over a `[tiles, ...]`
#: parameter block is the same fit as one optimiser per tile. The iterate that
#: scores best on the validation blocks is the one kept, which is what makes
#: these an upper bound rather than a record of when the loop stopped.
NONLINEAR_STEPS = 1500
NONLINEAR_LR = 3e-3
NONLINEAR_VALIDATE_EVERY = 50

# --- tile selection -------------------------------------------------------------

#: A tile whose stalk barely moves is background. Its dynamics are trivially
#: linear and including it would inflate every number here. The floor is on the
#: mean per-channel temporal variance of the tile's stalk, in [0, 1] units.
VARIANCE_FLOOR = 1e-4
#: A floor on the tile's mean squared one-step change, which is the persistence
#: baseline every error here is divided by. A tile can clear the variance floor
#: on a single slow drift and still hold a scoring block over which nothing
#: moves at all, and that tile's ratios are a division by nearly zero.
MOTION_FLOOR = 1e-5
#: How many ticks of contact inside a tile make it a contact tile. One frame of
#: grazing contact does not make a tile's dynamics a contact regime.
CONTACT_TICKS = 20
TILES_PER_CLASS = 12

# --- the pre-registered pass conditions -----------------------------------------
#
# From the ticket, unchanged. Constants so that the verdict is arithmetic rather
# than judgement, and so a reader can see the conditions were fixed before the
# curve was.

#: Go: (b) recovers at least this share of (c)'s error reduction over (a).
RECOVERY_TARGET = 0.80
#: Go: at a lift no wider than this. `k = 24` or `32` is a qualified go.
GO_K_CEILING = 16
QUALIFIED_K_CEILING = 32
#: Go: contact tiles no worse than this multiple of free-motion tiles.
CONTACT_TOLERANCE = 2.0

#: The demo's perturbation horizons, in ticks, which `bias_selection.py` derives
#: the target timescale range from (`benchmarks/timescale_selection.py`). The
#: `K` spectrum is held against both readings.
TARGET_TAU = {"onset": (3.0, 14.0), "duration": (3.0, 750.0)}

SEED = 126
DTYPE = torch.float64


# =============================================================================
# collection
# =============================================================================


class Babble:
    """Smooth motor babble. Holds one state and no opinions."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.action = np.zeros(3)

    def __call__(self, info: dict) -> np.ndarray:
        self.action = BABBLE_SMOOTHING * self.action + (
            1 - BABBLE_SMOOTHING
        ) * self.rng.uniform(-1, 1, 3)
        return self.action


def _scripted_pusher(env):
    """`benchmarks/achievability.py`'s dumb hand, imported rather than re-written."""
    from achievability import ScriptedPusher

    return ScriptedPusher(env)


@dataclass
class Collected:
    """Everything the sweep is fit on."""

    #: `[ticks, 64, 64, 3]` uint8. Kept as the render rather than as stalks: the
    #: full `[ticks, GRID, GRID, 48]` float array is 4x the memory and only 24
    #: of its 256 tiles are ever used.
    images: np.ndarray
    #: `[segments, 2]` half-open `[start, stop)` tick ranges. No pair crosses one.
    segments: np.ndarray
    #: `[GRID, GRID]` mean per-channel temporal variance of each tile's stalk.
    variance: np.ndarray
    #: `[GRID, GRID]` mean squared one-step change: the persistence baseline.
    motion: np.ndarray
    #: `[GRID, GRID]` ticks on which an arm-puck contact projected into each tile.
    contact_ticks: np.ndarray
    #: `[GRID, GRID]` ticks on which puck 1 -- the eccentric one -- covered each tile.
    eccentric_ticks: np.ndarray
    driver_of_segment: list[str] = field(default_factory=list)


def tile_series(images: np.ndarray, row: int, col: int) -> np.ndarray:
    """One tile's stalk trajectory, `[ticks, 48]` in [0, 1]."""
    patch = images[:, row * TILE : (row + 1) * TILE, col * TILE : (col + 1) * TILE, :]
    return patch.reshape(images.shape[0], PATCH_STALK).astype(np.float64) / 255.0


def _as_stalks(images: np.ndarray) -> np.ndarray:
    """`[ticks, 64, 64, 3]` uint8 to `[ticks, GRID, GRID, 48]` float in [0, 1]."""
    block = images.astype(np.float64) / 255.0
    block = block.reshape(-1, GRID, TILE, GRID, TILE, 3).transpose(0, 1, 3, 2, 4, 5)
    return block.reshape(-1, GRID, GRID, PATCH_STALK)


def tile_activity(
    images: np.ndarray, segments, *, chunk: int = 512
) -> tuple[np.ndarray, np.ndarray]:
    """Per tile: `(temporal variance, mean squared one-step change)`.

    Both `[GRID, GRID]`, both averaged over the tile's 48 channels. The first
    ranks tiles by how much of the world passes through them; the second is the
    persistence baseline every ratio in this script divides by, and is floored
    so that no reported ratio is a division by nearly zero. Accumulated in
    chunks: the whole run as one float array of stalks is four times the memory
    of the render it came from, and only two moments are wanted off it.
    """
    total = np.zeros((GRID, GRID, PATCH_STALK), dtype=np.float64)
    total_square = np.zeros_like(total)
    for start in range(0, images.shape[0], chunk):
        block = _as_stalks(images[start : start + chunk])
        total += block.sum(axis=0)
        total_square += (block**2).sum(axis=0)
    ticks = images.shape[0]
    mean = total / ticks
    variance = (total_square / ticks - mean**2).mean(axis=-1)

    change = np.zeros((GRID, GRID), dtype=np.float64)
    pairs = 0
    for a, b in segments:
        for start in range(int(a), int(b), chunk):
            # One tick of overlap, so the pair straddling a chunk boundary is
            # counted once rather than dropped.
            stop = min(start + chunk + 1, int(b))
            if stop - start < 2:
                continue
            block = _as_stalks(images[start:stop])
            change += (np.diff(block, axis=0) ** 2).sum(axis=(0, -1))
            pairs += stop - start - 1
    return variance, change / max(pairs * PATCH_STALK, 1)


class Projection:
    """World point to render pixel, for the topdown camera in `arena.xml`.

    A camera at `(0, 0, 1.4)`, `euler="0 0 0"`, `fovy=42`, on a square image: it
    looks down `-z` with `+y` up and `+x` right, so the map is one similarity
    per depth. Validated against the pucks -- see :meth:`check`, which runs
    before any data is kept.
    """

    def __init__(self, env: PlanarPushSandbox):
        cam = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "topdown")
        self.height = float(env.model.cam_pos[cam][2])
        self.half_angle = math.radians(float(env.model.cam_fovy[cam]) / 2.0)
        self.size = env.image_size

    def pixel(self, point) -> tuple[int, int] | None:
        """`(row, col)` of a world point, or `None` if it falls off the render."""
        half = (self.height - float(point[2])) * math.tan(self.half_angle)
        if half <= 0:
            return None
        half_px = self.size / 2
        col = half_px + half_px * float(point[0]) / half
        row = half_px - half_px * float(point[1]) / half
        r, c = int(round(row)), int(round(col))
        return (r, c) if 0 <= r < self.size and 0 <= c < self.size else None

    def tile(self, point) -> tuple[int, int] | None:
        px = self.pixel(point)
        return None if px is None else (px[0] // TILE, px[1] // TILE)

    def check(self, env: PlanarPushSandbox, image: np.ndarray, info: dict) -> bool:
        """Does each puck's centre land on a pixel whose dominant channel is its own?

        Cheap, and it is the only thing standing between a projection sign error
        and a contact/free split that is quietly random.
        """
        for i, pose in enumerate(info["puck_pose"]):
            gid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, f"g_puck_{i}")
            px = self.pixel((pose[0], pose[1], 0.022))
            if px is None:
                return False
            seen = image[px[0], px[1]].astype(np.float32)
            if int(np.argmax(seen)) != int(np.argmax(env.model.geom_rgba[gid][:3])):
                return False
        return True


def collect(*, seed: int = SEED, verbose: bool = True) -> Collected:
    """Drive the sandbox under babble and under the scripted pusher, and record."""
    env = PlanarPushSandbox()
    rng = np.random.default_rng(seed)
    projection = Projection(env)

    images: list[np.ndarray] = []
    segments: list[tuple[int, int]] = []
    drivers: list[str] = []
    contact_ticks = np.zeros((GRID, GRID), dtype=np.int64)
    eccentric_ticks = np.zeros((GRID, GRID), dtype=np.int64)
    arm_gid, puck_gid = set(env._arm_gid), set(env._puck_gid)
    checked = False

    plan = [("babble", i) for i in range(SEGMENTS_PER_DRIVER)]
    plan += [("pusher", i) for i in range(SEGMENTS_PER_DRIVER)]

    for driver_name, index in plan:
        obs, info = env.reset(
            seed=seed + 17 * index + (0 if driver_name == "babble" else 1000),
            options={"reset_arm": True},
        )
        driver = Babble(rng) if driver_name == "babble" else _scripted_pusher(env)

        if not checked:
            if not projection.check(env, obs["image"], info):
                raise RuntimeError(
                    "the topdown projection does not land pucks on their own colour; "
                    "the contact/free split would be noise"
                )
            checked = True

        start = len(images)
        for _ in range(SEGMENT_TICKS):
            obs, _, _, _, info = env.step(driver(info))
            images.append(obs["image"].copy())
            touched = set()
            for contact in env.data.contact[: env.data.ncon]:
                pair = {contact.geom1, contact.geom2}
                if pair & arm_gid and pair & puck_gid:
                    tile = projection.tile(contact.pos)
                    if tile is not None:
                        touched.add(tile)
            for tile in touched:
                contact_ticks[tile] += 1
            puck = info["puck_pose"][1]
            eccentric = projection.tile((puck[0], puck[1], 0.022))
            if eccentric is not None:
                eccentric_ticks[eccentric] += 1
        segments.append((start, len(images)))
        drivers.append(driver_name)
        if verbose:
            print(f"    {driver_name} segment {index}: {len(images) - start} ticks")

    env.close()
    stack = np.stack(images)
    variance, motion = tile_activity(stack, segments)

    return Collected(
        images=stack,
        segments=np.array(segments),
        variance=variance,
        motion=motion,
        contact_ticks=contact_ticks,
        eccentric_ticks=eccentric_ticks,
        driver_of_segment=drivers,
    )


# =============================================================================
# tiles
# =============================================================================


@dataclass(frozen=True)
class Tile:
    row: int
    col: int
    kind: str  # "contact" or "free"
    variance: float
    eccentric: bool


def choose_tiles(data: Collected, *, per_class: int = TILES_PER_CLASS) -> list[Tile]:
    """The tiles the sweep runs on: the most active of each class.

    Sorted by temporal variance and taken from the top, because a tile that
    moves is a tile whose dynamics exist to be modelled. Anything under
    :data:`VARIANCE_FLOOR` is background and is dropped from both classes.
    """
    candidates: list[Tile] = []
    for r in range(GRID):
        for c in range(GRID):
            if data.variance[r, c] < VARIANCE_FLOOR or data.motion[r, c] < MOTION_FLOOR:
                continue
            kind = "contact" if data.contact_ticks[r, c] >= CONTACT_TICKS else "free"
            candidates.append(
                Tile(
                    row=r,
                    col=c,
                    kind=kind,
                    variance=float(data.variance[r, c]),
                    eccentric=bool(data.eccentric_ticks[r, c] >= CONTACT_TICKS),
                )
            )
    chosen: list[Tile] = []
    for kind in ("free", "contact"):
        same = sorted((t for t in candidates if t.kind == kind), key=lambda t: -t.variance)
        chosen.extend(same[:per_class])
    return chosen


# =============================================================================
# the lift
# =============================================================================


def _body(n: int, k: int, cells: int, seed: int) -> tuple[CellBody, CellBiases]:
    """A frozen random body at the widths `body.py`'s own rule gives for `n, k`.

    Not a re-implementation of the lift: this *is* the lift, constructed the way
    `CellBody` constructs it and evaluated through `CellBody.encode`, so a body
    swapped in under those buffers would be measured here rather than beside
    here. Every tile is one cell of the population and gets its own bias draw,
    which is what makes one shared body many different cells.
    """
    shape = BodyShape(n=n, k=k)
    generator = torch.Generator().manual_seed(seed)
    body = CellBody(shape, generator=generator, dtype=DTYPE)
    biases = CellBiases(shape, cells, generator=generator, dtype=DTYPE)
    return body, biases


@dataclass
class Lift:
    """Every tile's trajectory in lifted coordinates, with its segment structure.

    `z` is `[tiles, ticks, k]` and `x` is `[tiles, ticks, 48]` -- the stalk each
    `z` was taken at. `segments` index the tick axis and are what a fit is
    allowed to form pairs inside.
    """

    z: torch.Tensor
    x: torch.Tensor
    segments: list[tuple[int, int]]
    k: int


def lift(series: torch.Tensor, segments, *, mode: str, k: int, seed: int) -> Lift:
    """Push every tile's stalk trajectory through a frozen `encode`.

    `series` is `[tiles, ticks, 48]`, already centred. The modes differ only in
    what `encode` is given:

    * `plain` -- a zero chart and the stalk. The instantaneous lift.
    * `delay` -- a zero chart and a Hankel stack of `DELAYS` stalks. The lift's
      dimension paid in time (comment 11).
    * `recurrent` -- the stalk and *the chart `encode` produced last tick*. The
      delay embedding the architecture already has, rather than one bolted on.

    `delay` and `recurrent` both drop the first `DELAYS - 1` ticks of every
    segment, so that no lifted coordinate carries evidence from before a reset,
    and both leave `x` aligned to what `z` was taken at.
    """
    tiles, ticks, _ = series.shape
    ranges = [(int(a), int(b)) for a, b in segments]
    # The biases are `torch.nn.Parameter`s, so an un-guarded lift would hand
    # back tensors carrying a graph back into the frozen body -- and the body
    # is frozen. Nothing downstream of the lift is differentiated except (c)'s
    # own parameters.
    guard = torch.no_grad()
    guard.__enter__()
    try:
        return _lift(series, ranges, tiles, ticks, mode=mode, k=k, seed=seed)
    finally:
        guard.__exit__(None, None, None)


def _lift(series, ranges, tiles, ticks, *, mode, k, seed) -> "Lift":

    if mode == "plain":
        body, biases = _body(PATCH_STALK, k, tiles, seed)
        chart = torch.zeros(tiles, k, dtype=DTYPE)
        z = torch.stack(
            [body.encode(chart, series[:, t, :], biases) for t in range(ticks)], dim=1
        )
        return Lift(z=z, x=series, segments=ranges, k=k)

    if mode == "delay":
        body, biases = _body(PATCH_STALK * DELAYS, k, tiles, seed)
        chart = torch.zeros(tiles, k, dtype=DTYPE)
        kept, columns, out = [], [], []
        for a, b in ranges:
            if b - a <= DELAYS:
                continue
            start = len(kept)
            kept.extend(range(a + DELAYS - 1, b))
            out.append((start, len(kept)))
        for t in kept:
            stacked = series[:, t - DELAYS + 1 : t + 1, :].reshape(tiles, -1)
            columns.append(body.encode(chart, stacked, biases))
        return Lift(
            z=torch.stack(columns, dim=1), x=series[:, kept, :], segments=out, k=k
        )

    if mode == "recurrent":
        body, biases = _body(PATCH_STALK, k, tiles, seed)
        kept, columns, out = [], [], []
        for a, b in ranges:
            if b - a <= DELAYS:
                continue
            chart = torch.zeros(tiles, k, dtype=DTYPE)
            start = len(kept)
            for t in range(a, b):
                chart = body.encode(chart, series[:, t, :], biases)
                # The first charts of a segment still carry the zero it started
                # from; they are burn-in, not evidence.
                if t - a >= DELAYS - 1:
                    kept.append(t)
                    columns.append(chart)
            out.append((start, len(kept)))
        return Lift(
            z=torch.stack(columns, dim=1), x=series[:, kept, :], segments=out, k=k
        )

    raise ValueError(f"unknown input mode {mode!r}")


# =============================================================================
# fits -- batched over tiles
# =============================================================================


def _pairs(segments, horizon: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Indices `(t, t + horizon)` that stay inside one segment."""
    t, t_next = [], []
    for a, b in segments:
        if b - a <= horizon:
            continue
        t.extend(range(a, b - horizon))
        t_next.extend(range(a + horizon, b))
    return np.array(t, dtype=np.int64), np.array(t_next, dtype=np.int64)


def _split(segments) -> tuple[list, list, list]:
    """Partition every segment into interleaved fit, validation and score blocks.

    **Why not the leading fraction of each segment.** That was the first
    protocol here and it was wrong: a segment's tail is a different part of the
    world from its head -- the arm has swept somewhere else, the pucks have
    moved -- so scoring on the tail measures extrapolation to a shifted
    distribution rather than whether the dynamics linearise. It shows up
    immediately: raw DMD scored **0.83x persistence on the ticks it was fit on
    and 429x on the tail**, and every number downstream of that is about the
    shift and not about the question.

    Interleaving `BLOCK_TICKS`-long blocks in a fixed 3:1:1 rotation keeps the
    three sets drawn from the same distribution. `BLOCK_GAP` ticks are dropped
    at the end of every block so that no pair straddles a boundary and no
    scored tick is the immediate successor of a fitted one -- adjacent ticks at
    50 Hz are nearly identical, and a high-capacity model would otherwise be
    scored partly on what it memorised.
    """
    parts: tuple[list, list, list] = ([], [], [])
    rotation = (0, 0, 0, 1, 2)
    index = 0
    for a, b in segments:
        for start in range(a, b, BLOCK_TICKS):
            stop = min(start + BLOCK_TICKS, b) - BLOCK_GAP
            if stop - start < MINIMUM_BLOCK:
                continue
            parts[rotation[index % len(rotation)]].append((start, stop))
            index += 1
    return parts


def _affine_fit(
    x: torch.Tensor, y: torch.Tensor, ridge: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched ridge least squares `y ~ A x + b`.

    `x` is `[tiles, rows, d]`, `y` is `[tiles, rows, p]`; returns `A` as
    `[tiles, p, d]` and `b` as `[tiles, p]`.
    """
    ones = torch.ones(*x.shape[:-1], 1, dtype=x.dtype)
    design = torch.cat((x, ones), dim=-1)
    gram = design.transpose(-1, -2) @ design
    scale = ridge * torch.diagonal(gram, dim1=-2, dim2=-1).mean(-1).clamp(min=1e-12)
    eye = torch.eye(gram.shape[-1], dtype=x.dtype)
    solution = torch.linalg.solve(gram + scale[:, None, None] * eye, design.transpose(-1, -2) @ y)
    return solution[:, :-1, :].transpose(-1, -2).contiguous(), solution[:, -1, :].contiguous()


def _apply(a: torch.Tensor, bias: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """`A x + b` for `x` of `[tiles, rows, d]`."""
    return x @ a.transpose(-1, -2) + bias[:, None, :]


def _per_tile_mse(prediction: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    """`[tiles]` mean squared error, kept per tile so classes are never mixed."""
    return ((prediction - truth) ** 2).mean(dim=(1, 2))


def _selected_affine(source: torch.Tensor, target: torch.Tensor, score) -> tuple:
    """`_affine_fit` with the ridge chosen **per tile** on the validation blocks.

    `score(A, b)` returns the `[tiles]` validation error of one candidate, in
    whatever space the fit will finally be reported in -- so a lifted `K` is
    selected on the stalk error it will be judged by rather than on the
    observable error it happens to minimise. Each tile keeps its own winner:
    tiles differ by orders of magnitude in how collinear their stalks are, and
    one ridge for all of them would be a compromise nobody asked for.
    """
    candidates = [_affine_fit(source, target, ridge) for ridge in RIDGE_GRID]
    errors = torch.stack([score(a, b) for a, b in candidates])
    best = errors.argmin(dim=0)
    a = torch.stack([candidates[int(i)][0][t] for t, i in enumerate(best)])
    b = torch.stack([candidates[int(i)][1][t] for t, i in enumerate(best)])
    return a, b, best


class BatchedMLP:
    """One hidden-layer ReLU network per tile, trained together.

    Adam is elementwise, so a single optimiser over a `[tiles, ...]` parameter
    block is the same fit as one optimiser per tile -- and about two orders of
    magnitude faster than the loop over tiles it replaces.
    """

    def __init__(self, tiles: int, d_in: int, width: int, d_out: int, generator):
        def draw(*size: int, fan_in: int) -> torch.Tensor:
            bound = 1.0 / math.sqrt(fan_in)
            return torch.empty(*size, dtype=DTYPE).uniform_(-bound, bound, generator=generator)

        self.hidden_weight = draw(tiles, width, d_in, fan_in=d_in).requires_grad_(True)
        self.hidden_bias = draw(tiles, width, fan_in=d_in).requires_grad_(True)
        self.output_weight = draw(tiles, d_out, width, fan_in=width).requires_grad_(True)
        self.output_bias = draw(tiles, d_out, fan_in=width).requires_grad_(True)

    def parameters(self):
        return [self.hidden_weight, self.hidden_bias, self.output_weight, self.output_bias]

    def state(self):
        return [p.detach().clone() for p in self.parameters()]

    def load(self, state) -> None:
        with torch.no_grad():
            for parameter, saved in zip(self.parameters(), state):
                parameter.copy_(saved)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(_apply(self.hidden_weight, self.hidden_bias, x))
        return _apply(self.output_weight, self.output_bias, hidden)


def _start_from_linear(
    net: BatchedMLP, a: torch.Tensor, bias: torch.Tensor, source: torch.Tensor, generator
) -> None:
    """Set a network's weights so that it computes exactly `a x + b` to begin with.

    A one-hidden-layer ReLU network reproduces an affine map on any region where
    every hidden pre-activation is positive: put a scaled identity in the first
    `d_in` rows of the hidden layer with an offset large enough to clear the
    data, and the ReLU is transparent there. The offset is undone in the output
    bias, so the network starts *at* the linear fit and gradient descent can only
    improve on it.

    This is what makes the trained models upper bounds rather than a race. Left
    to a standard random initialisation they are not: measured here, a randomly
    started (c) finished five times **worse** than the linear (b) it is supposed
    to bound, because it has to discover, through a frozen readout, a solution
    that (b) is handed in closed form.

    The spare hidden units are given a small random draw rather than zeros, so
    they are not all the same unit.
    """
    tiles, width, d_in = net.hidden_weight.shape
    if width < d_in:
        raise ValueError(f"a linear start needs width >= d_in, got {width} < {d_in}")
    offset = (-source.amin(dim=(1, 2)) + 1.0).clamp(min=1.0)  # [tiles]
    with torch.no_grad():
        net.hidden_weight.zero_()
        net.hidden_weight[:, :d_in, :] = torch.eye(d_in, dtype=DTYPE)
        net.hidden_bias.fill_(1.0)
        net.hidden_bias[:, :d_in] = offset[:, None]
        net.output_weight.zero_()
        net.output_weight[:, :, :d_in] = a
        net.output_bias.copy_(bias - offset[:, None] * a.sum(dim=-1))
        if width > d_in:
            spare = torch.empty(tiles, width - d_in, d_in, dtype=DTYPE)
            net.hidden_weight[:, d_in:, :] = spare.normal_(0.0, 0.01, generator=generator)


def _train(net: BatchedMLP, source, target, validate, through=lambda y: y) -> None:
    """Adam to convergence, keeping the iterate that scores best **per tile**.

    `validate(net)` returns the `[tiles]` validation error, and `through` maps
    the network's output into the space `target` lives in -- so a model behind a
    frozen readout is trained on the error it will be judged by rather than on
    the one it happens to sit closest to.

    The best iterate is tracked per tile and reassembled at the end, because
    tiles converge at different rates and stopping all of them together would
    under-train the slow ones -- and an under-trained upper bound is not an
    upper bound, it is a record of when the loop happened to stop.
    """
    optimiser = torch.optim.Adam(net.parameters(), lr=NONLINEAR_LR)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=NONLINEAR_STEPS)
    with torch.no_grad():
        best_error = validate(net)
    best_state = net.state()
    for step in range(NONLINEAR_STEPS):
        optimiser.zero_grad()
        ((through(net(source)) - target) ** 2).mean(dim=(1, 2)).sum().backward()
        optimiser.step()
        schedule.step()
        if (step + 1) % NONLINEAR_VALIDATE_EVERY:
            continue
        with torch.no_grad():
            error = validate(net)
        improved = error < best_error
        if not bool(improved.any()):
            continue
        best_error = torch.where(improved, error, best_error)
        for kept, current in zip(best_state, net.state()):
            mask = improved.reshape(-1, *([1] * (current.dim() - 1)))
            kept.copy_(torch.where(mask, current, kept))
    net.load(best_state)


@dataclass
class Fit:
    """Every number one `(mode, k)` produces, per tile."""

    persistence: np.ndarray
    dmd: np.ndarray
    linear: np.ndarray
    nonlinear: np.ndarray
    unbottlenecked: np.ndarray
    floor: np.ndarray
    horizon_error: dict[int, np.ndarray]
    horizon_persistence: dict[int, np.ndarray]
    tau: np.ndarray  # [tiles, k], effective timescale of each eigenvalue, in ticks
    spectral_radius: np.ndarray


def unbottlenecked(lifted: Lift) -> torch.Tensor:
    """(d): one hidden layer on the raw stalk, no lift and no linear readout.

    Not one of the ticket's three. It answers what the other three cannot: when
    every model lands above persistence, is that the *lift* being too narrow or
    the *task* being persistence-dominated? (d) has neither a bottleneck nor a
    linear readout, so if it is above persistence too, the answer is the task.

    Depends on the input mode only through which ticks survive it, never on `k`,
    so :func:`sweep` fits it once per mode rather than once per `(mode, k)`.
    """
    fitting, validation, scoring = _split(lifted.segments)
    t, t_next = _pairs(fitting)
    v, v_next = _pairs(validation)
    e, e_next = _pairs(scoring)
    x = lifted.x
    generator = torch.Generator().manual_seed(SEED)
    net = BatchedMLP(
        x.shape[0],
        PATCH_STALK,
        hidden_width(PATCH_STALK, PATCH_STALK),
        PATCH_STALK,
        generator,
    )
    with torch.no_grad():
        A, a_bias, _ = _selected_affine(
            x[:, t, :],
            x[:, t_next, :],
            lambda a, b: _per_tile_mse(_apply(a, b, x[:, v, :]), x[:, v_next, :]),
        )
    # Started from (a), so (d) is an upper bound on it by construction: the
    # question (d) exists to answer is whether *anything* of this size beats
    # persistence, and a (d) that lost to (a) would answer nothing.
    _start_from_linear(net, A, a_bias, x[:, t, :], generator)
    _train(
        net,
        x[:, t, :],
        x[:, t_next, :],
        lambda trained: _per_tile_mse(trained(x[:, v, :]), x[:, v_next, :]),
    )
    with torch.no_grad():
        return _per_tile_mse(net(x[:, e, :]), x[:, e_next, :])


def fit(lifted: Lift, reference: torch.Tensor) -> Fit:
    """Fit every model and score them all on the same held-out ticks.

    Three disjoint, interleaved sets of blocks: one to fit on, one to select
    every ridge and every stopping point on, and one that is only ever scored.
    `reference` is (d), fit once per mode by the caller.
    """
    fitting, validation, scoring = _split(lifted.segments)
    t, t_next = _pairs(fitting)
    v, v_next = _pairs(validation)
    e, e_next = _pairs(scoring)
    z, x = lifted.z, lifted.x
    tiles = z.shape[0]

    with torch.no_grad():
        # --- (a) raw DMD, in stalk coordinates, no lift ------------------------
        A, a_bias, _ = _selected_affine(
            x[:, t, :],
            x[:, t_next, :],
            lambda a, b: _per_tile_mse(_apply(a, b, x[:, v, :]), x[:, v_next, :]),
        )
        dmd = _per_tile_mse(_apply(A, a_bias, x[:, e, :]), x[:, e_next, :])
        persistence = _per_tile_mse(x[:, e, :], x[:, e_next, :])

        # --- the readout, shared by (b) and (c) --------------------------------
        C, c_bias, _ = _selected_affine(
            z[:, t_next, :],
            x[:, t_next, :],
            lambda a, b: _per_tile_mse(_apply(a, b, z[:, v_next, :]), x[:, v_next, :]),
        )

        # --- (b) the variant: K by least squares in lifted coordinates ---------
        K, k_bias, _ = _selected_affine(
            z[:, t, :],
            z[:, t_next, :],
            lambda a, b: _per_tile_mse(
                _apply(C, c_bias, _apply(a, b, z[:, v, :])), x[:, v_next, :]
            ),
        )
        linear = _per_tile_mse(_apply(C, c_bias, _apply(K, k_bias, z[:, e, :])), x[:, e_next, :])

        # --- (e) the lift's own floor ------------------------------------------
        # `C` applied to the **true** next chart. No operator, no prediction: the
        # error left over from going through the lift and back at all. Without it
        # (b) cannot be read, because (b)'s error is reconstruction error plus
        # operator error and only their sum is otherwise visible. (b) sitting at
        # this floor means the operator is doing as well as any operator could.
        floor = _per_tile_mse(_apply(C, c_bias, z[:, e_next, :]), x[:, e_next, :])

    # --- (c) the same lift and the same readout, with a nonlinear step ---------
    generator = torch.Generator().manual_seed(SEED + lifted.k)
    step_net = BatchedMLP(
        tiles, lifted.k, BodyShape(PATCH_STALK, lifted.k).step_width, lifted.k, generator
    )
    _start_from_linear(step_net, K, k_bias, z[:, t, :], generator)
    # Trained on the **stalk** error through the frozen readout, which is what it
    # is scored on. Training it on the observable error instead -- the obvious
    # symmetry with (b)'s least squares -- optimises a different quantity from
    # the one reported, and measured here as leaving (c) an order of magnitude
    # worse than (b): an upper bound that loses to what it bounds is not one.
    _train(
        step_net,
        z[:, t, :],
        x[:, t_next, :],
        lambda net: _per_tile_mse(_apply(C, c_bias, net(z[:, v, :])), x[:, v_next, :]),
        through=lambda advanced: _apply(C, c_bias, advanced),
    )

    with torch.no_grad():
        nonlinear = _per_tile_mse(_apply(C, c_bias, step_net(z[:, e, :])), x[:, e_next, :])

        # --- h-step error for (b) ----------------------------------------------
        horizon_error: dict[int, np.ndarray] = {}
        horizon_persistence: dict[int, np.ndarray] = {}
        for h in HORIZONS:
            eh, eh_next = _pairs(scoring, horizon=h)
            if eh.size == 0:
                horizon_error[h] = np.full(tiles, np.nan)
                horizon_persistence[h] = np.full(tiles, np.nan)
                continue
            rolled = z[:, eh, :]
            for _ in range(h):
                rolled = _apply(K, k_bias, rolled)
            horizon_error[h] = _per_tile_mse(_apply(C, c_bias, rolled), x[:, eh_next, :]).numpy()
            horizon_persistence[h] = _per_tile_mse(x[:, eh, :], x[:, eh_next, :]).numpy()

        # --- the spectrum ------------------------------------------------------
        magnitude = torch.linalg.eigvals(K).abs().numpy().astype(np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        tau = np.where(
            magnitude < 1.0,
            -1.0 / np.log(np.clip(magnitude, 1e-12, 1 - 1e-12)),
            np.inf,
        )

    return Fit(
        persistence=persistence.numpy(),
        dmd=dmd.numpy(),
        linear=linear.numpy(),
        nonlinear=nonlinear.numpy(),
        unbottlenecked=reference.numpy(),
        floor=floor.numpy(),
        horizon_error=horizon_error,
        horizon_persistence=horizon_persistence,
        tau=tau,
        spectral_radius=magnitude.max(axis=-1),
    )


# =============================================================================
# the sweep, and the verdict
# =============================================================================


@dataclass
class Cell:
    """The aggregate of one `(mode, k, class)` -- what the pass condition reads."""

    mode: str
    k: int
    kind: str
    #: Per-tile error as a multiple of that tile's own persistence. **Medians,
    #: not means**: the ratios are heavy-tailed across tiles -- one tile the
    #: models handle badly moved a pooled mean by 4x here -- and a class summary
    #: that one tile can set is not a summary of the class.
    dmd: np.ndarray
    linear: np.ndarray
    nonlinear: np.ndarray
    unbottlenecked: np.ndarray
    floor: np.ndarray
    #: Raw per-tile MSEs, kept for the recovery ratio, which is a ratio of
    #: differences and cannot be formed from normalised numbers.
    raw: dict[str, np.ndarray]
    horizon: dict[int, np.ndarray]
    tau: np.ndarray
    spectral_radius: float

    @property
    def recovery(self) -> float:
        """The share of (c)'s reduction over (a) that (b) recovers, per tile.

        Formed per tile and then taken as a median, so it is the typical tile's
        recovery rather than the pooled one. A tile where (c) fails to beat (a)
        contributes no ratio: the lift earns nothing there and the quotient
        would be a number about noise. `nan` if no tile in the class has any
        headroom at all.
        """
        headroom = self.raw["dmd"] - self.raw["nonlinear"]
        usable = headroom > 0
        if not usable.any():
            return float("nan")
        return float(
            np.median((self.raw["dmd"][usable] - self.raw["linear"][usable]) / headroom[usable])
        )

    @property
    def headroom_share(self) -> float:
        """Share of tiles where (c) beats (a) -- where a recovery ratio exists."""
        return float(np.mean(self.raw["dmd"] > self.raw["nonlinear"]))

    @property
    def overhead(self) -> float:
        """(b) against its own floor (e): how much of (b)'s error is the operator.

        The one column here that separates the ticket's question from its
        confound. (b)'s error is reconstruction error plus operator error, and
        the reconstruction half is a property of the *lift*, not of linearity.
        Dividing it out leaves what the fitted `K` costs over the best any
        operator could do through that same lift. 1.00 means the operator is
        perfect and every remaining error is the bottleneck.

        It is also the only fair way to compare the input modes. `delay`
        compresses `48 * DELAYS` inputs into the same `k` that `plain`
        compresses 48 into, so its floor is worse by construction and its raw
        error is not comparable with `plain`'s. This ratio is.
        """
        return float(np.median(self.linear) / np.median(self.floor))


def sweep(data: Collected, tiles: list[Tile], *, verbose: bool = True):
    """Every `(mode, k)`, and every tile's own numbers underneath."""
    stack = np.stack([tile_series(data.images, t.row, t.col) for t in tiles])
    series = torch.from_numpy(stack).to(DTYPE)
    # Centred on the fitting blocks only. A tile's mean colour is not dynamics
    # and every model here would otherwise spend capacity on carrying it.
    fitting, _, _ = _split([(int(a), int(b)) for a, b in data.segments])
    rows = np.concatenate([np.arange(a, b) for a, b in fitting])
    series = series - series[:, rows, :].mean(dim=1, keepdim=True)

    kinds = np.array([t.kind for t in tiles])
    cells: list[Cell] = []
    per_tile: dict[tuple[str, int], Fit] = {}
    for mode in INPUT_MODES:
        # (d) depends on the mode only through which ticks survive it, so it is
        # fit once here rather than once per k. `k = 8` is an arbitrary choice
        # of lift to take the surviving ticks from; every k in a mode keeps the
        # same ones.
        started = time.time()
        reference = unbottlenecked(lift(series, data.segments, mode=mode, k=8, seed=SEED + 8))
        if verbose:
            print(f"    {mode:>9} (d)    {time.time() - started:5.1f}s")
        for k in K_SWEEP:
            started = time.time()
            result = fit(lift(series, data.segments, mode=mode, k=k, seed=SEED + k), reference)
            per_tile[(mode, k)] = result
            for kind in ("free", "contact"):
                where = kinds == kind
                if not where.any():
                    continue
                baseline = result.persistence[where]
                cells.append(
                    Cell(
                        mode=mode,
                        k=k,
                        kind=kind,
                        dmd=result.dmd[where] / baseline,
                        linear=result.linear[where] / baseline,
                        nonlinear=result.nonlinear[where] / baseline,
                        unbottlenecked=result.unbottlenecked[where] / baseline,
                        floor=result.floor[where] / baseline,
                        raw={
                            "dmd": result.dmd[where],
                            "linear": result.linear[where],
                            "nonlinear": result.nonlinear[where],
                        },
                        horizon={
                            h: result.horizon_error[h][where] / result.horizon_persistence[h][where]
                            for h in HORIZONS
                        },
                        tau=result.tau[where].reshape(-1),
                        spectral_radius=float(np.median(result.spectral_radius[where])),
                    )
                )
            if verbose:
                print(f"    {mode:>9} k={k:<3} {time.time() - started:5.1f}s")
    return cells, per_tile


def verdict(cells: list[Cell]) -> tuple[str, list[str]]:
    """The pre-registered pass conditions, applied to the sweep. Arithmetic only.

    The ticket registers **three** ways to fail, and they are not the same
    failure. Both of the first two are checked here explicitly:

    1. (b) never approaches (c) at any `k_lift <= 32` -- linearity is too
       expensive. This is the recovery target.
    2. **(b) is no better than (a)** -- the lift earns nothing, whatever
       linearity costs. A sweep can pass (1) and fail (2) at the same time, and
       it means something quite different from failing (1): it says the cost is
       in the *lift*, not in the *linear operator*.
    3. Contact tiles fail at every width, which is a third outcome rather than a
       full no-go (comment 11: a switched operator is the mature answer to
       exactly that regime pair).
    """
    index = {(c.mode, c.k, c.kind): c for c in cells}
    lines: list[str] = []
    passed: list[tuple[str, str, int]] = []
    contact_only = False

    # Clause 2, checked across the whole sweep before anything else: does the
    # lift ever buy anything at all, at any mode and any width?
    earning = [
        (c.mode, c.k, float(np.median(c.dmd)), float(np.median(c.linear)))
        for c in cells
        if c.kind == "free"
    ]
    best = min(earning, key=lambda row: row[3] / row[2]) if earning else None
    if best is not None:
        mode, k, dmd, linear = best
        if linear >= dmd:
            lines.append(
                f"THE LIFT EARNS NOTHING: at its best (mode={mode}, k={k}) the lifted "
                f"linear model is {linear / dmd:.2f}x the error of the same linear model "
                "fit on the raw stalk. (b) never beats (a) anywhere in the sweep, which "
                "is the ticket's second no-go clause on its own terms."
            )
        else:
            lines.append(
                f"the lift earns its keep at mode={mode}, k={k}: {linear / dmd:.2f}x (a)."
            )

    for mode in INPUT_MODES:
        for k in K_SWEEP:
            free, contact = index.get((mode, k, "free")), index.get((mode, k, "contact"))
            if free is None:
                continue
            recovery = free.recovery
            if not (recovery == recovery and recovery >= RECOVERY_TARGET):
                continue
            free_linear = float(np.median(free.linear))
            ratio = (
                float(np.median(contact.linear)) / free_linear
                if contact is not None and free_linear > 0
                else float("inf")
            )
            label = "go" if k <= GO_K_CEILING else "qualified go"
            if k > QUALIFIED_K_CEILING:
                continue
            if ratio <= CONTACT_TOLERANCE:
                passed.append((label, mode, k))
                lines.append(
                    f"{label.upper()} at mode={mode}, k={k}: free-tile recovery "
                    f"{recovery:.2f} >= {RECOVERY_TARGET:.2f}, contact/free "
                    f"{ratio:.2f} <= {CONTACT_TOLERANCE:.1f}"
                )
            else:
                contact_only = True
                lines.append(
                    f"contact-only shortfall at mode={mode}, k={k}: free-tile recovery "
                    f"{recovery:.2f} passes, contact/free {ratio:.2f} > {CONTACT_TOLERANCE:.1f}"
                )

    lift_earns = best is not None and best[3] < best[2]
    if not passed and not contact_only:
        lines.append(
            "no (mode, k) in the sweep meets the free-tile recovery target at any width"
        )

    if not lift_earns:
        # Clause 2 is not a tie-break on clause 1: a lift that earns nothing is
        # a no-go however cheap linearity turns out to be, because what would be
        # converted is the lift and the operator together.
        outcome = "no-go"
    elif any(label == "go" for label, _, _ in passed):
        outcome = "go"
    elif passed:
        outcome = "qualified go"
    elif contact_only:
        outcome = "contact-only no-go"
    else:
        outcome = "no-go"
    return outcome, lines


def wake_status() -> str:
    """Step 7 and comment 12: has the cylinder wake been run, and if not, why not."""
    try:  # pragma: no cover - the point of the check is that it usually fails
        import hydrogym  # noqa: F401
    except Exception as error:
        return (
            "NOT RUN. HydroGym is unavailable here "
            f"({type(error).__name__}: {error}). It is not in this project's "
            "dependency set and pulls a Firedrake solver stack, which is the cost "
            "comment 12 flagged as unverified. The wake gates nothing -- a sandbox "
            "result stands or falls on the sandbox -- but its absence means every "
            "number above is self-referential and nothing here has been checked "
            "against a known answer."
        )
    return "AVAILABLE but not implemented in this pass; the sweep above is sandbox-only."


# =============================================================================
# the report
# =============================================================================


def report(data: Collected, tiles: list[Tile], cells: list[Cell], per_tile) -> str:
    out: list[str] = []
    add = out.append
    index = {(c.mode, c.k, c.kind): c for c in cells}

    add("# Koopman lift: does a cell's piece linearise? (#126)\n")
    ticks = int(data.segments[:, 1].max())
    add(
        f"{ticks:,} ticks ({ticks / CONTROL_HZ:.0f} s of sim) over {len(data.segments)} "
        f"segments, {data.driver_of_segment.count('babble')} babble and "
        f"{data.driver_of_segment.count('pusher')} scripted."
    )
    free = [t for t in tiles if t.kind == "free"]
    contact = [t for t in tiles if t.kind == "contact"]
    add(
        f"{int((data.variance >= VARIANCE_FLOOR).sum())} of {GRID * GRID} tiles clear the "
        f"variance floor; the sweep runs on {len(tiles)} of them -- {len(free)} free-motion, "
        f"{len(contact)} contact, {sum(t.eccentric for t in tiles)} carrying the eccentric puck."
    )
    add(
        "\nError is held-out one-step MSE on a 48-dimensional stalk in [0, 1] units, shown "
        "as a multiple of persistence (predict no change). Below 1.00 beats persistence; "
        "above 1.00 is worse than doing nothing.\n"
    )

    for mode in INPUT_MODES:
        add(f"## {mode}\n")
        add(
            f"  {'k':>3} | {'class':>7} | {'(a) DMD':>9} {'(b) linear':>10} "
            f"{'(c) nonlin':>10} {'(d) no lift':>11} {'(e) floor':>10} | {'b/e':>5} "
            f"{'recovery':>8} | {'rho(K)':>7} {'tau med':>8}"
        )
        add("  " + "-" * 107)
        for k in K_SWEEP:
            for kind in ("free", "contact"):
                cell = index.get((mode, k, kind))
                if cell is None:
                    continue
                finite = cell.tau[np.isfinite(cell.tau)]
                recovery = cell.recovery
                add(
                    f"  {k:>3} | {kind:>7} | {np.median(cell.dmd):9.3f} "
                    f"{np.median(cell.linear):10.3f} {np.median(cell.nonlinear):10.3f} "
                    f"{np.median(cell.unbottlenecked):11.3f} "
                    f"{np.median(cell.floor):10.3f} | {cell.overhead:5.2f} "
                    f"{f'{recovery:8.2f}' if recovery == recovery else '     n/a'} | "
                    f"{cell.spectral_radius:7.3f} "
                    f"{float(np.median(finite)) if finite.size else float('nan'):8.2f}"
                )
        add("")

    add("## h-step error for (b), as a multiple of persistence at the same horizon\n")
    add(
        "  Normalised per horizon, so 1.00 always means 'no better than holding the\n"
        "  stalk still for h ticks'. Persistence weakens as h grows; if the operator\n"
        "  is modelling anything, this is where it shows.\n"
    )
    add(f"  {'mode':>9} {'k':>3} {'class':>7} | " + "  ".join(f"h={h:<5}" for h in HORIZONS))
    add("  " + "-" * 62)
    for mode in INPUT_MODES:
        for k in K_SWEEP:
            for kind in ("free", "contact"):
                cell = index.get((mode, k, kind))
                if cell is None:
                    continue
                values = "  ".join(f"{np.median(cell.horizon[h]):7.3f}" for h in HORIZONS)
                add(f"  {mode:>9} {k:>3} {kind:>7} | {values}")
    add("")

    add("## the spectrum of the fitted K, against the demo's target range\n")
    add(
        "  tau = -1/ln|lambda|, in ticks. The construction aims at "
        f"{TARGET_TAU['onset'][0]:g}-{TARGET_TAU['onset'][1]:g} ticks on the onset reading "
        f"and {TARGET_TAU['duration'][0]:g}-{TARGET_TAU['duration'][1]:g} on the duration "
        "reading (`benchmarks/timescale_selection.py`)."
    )
    add(
        f"\n  {'mode':>9} {'k':>3} {'class':>7} | {'in onset':>9} {'in duration':>12} "
        f"{'|lambda|>=1':>12}"
    )
    add("  " + "-" * 60)
    for mode in INPUT_MODES:
        for k in K_SWEEP:
            for kind in ("free", "contact"):
                cell = index.get((mode, k, kind))
                if cell is None:
                    continue
                tau = cell.tau
                expansive = float(np.mean(~np.isfinite(tau)))
                finite = tau[np.isfinite(tau)]
                shares = []
                for reading in ("onset", "duration"):
                    lo, hi = TARGET_TAU[reading]
                    shares.append(
                        float(np.mean((finite >= lo) & (finite <= hi))) if finite.size else 0.0
                    )
                add(
                    f"  {mode:>9} {k:>3} {kind:>7} | {100 * shares[0]:8.1f}% "
                    f"{100 * shares[1]:11.1f}% {100 * expansive:11.1f}%"
                )
    add("")

    add("## the eccentric puck, separately\n")
    mine = np.array([t.eccentric for t in tiles])
    if not mine.any() or mine.all():
        add("  No separable set of tiles carried puck 1. Not measured.")
    else:
        add(f"  {'mode':>9} {'k':>3} | {'(b) eccentric':>14} {'(b) rest':>10} {'ratio':>7}")
        add("  " + "-" * 48)
        for mode in INPUT_MODES:
            for k in K_SWEEP:
                result = per_tile[(mode, k)]
                scaled = result.linear / result.persistence
                a, b = float(np.median(scaled[mine])), float(np.median(scaled[~mine]))
                add(f"  {mode:>9} {k:>3} | {a:14.3f} {b:10.3f} {a / b:7.2f}")
    add("")

    outcome, reasons = verdict(cells)
    add("## the pre-registered verdict\n")
    add(f"  **{outcome.upper()}**\n")
    for reason in reasons:
        add(f"  - {reason}")
    add("\n## the second data source (step 7, comment 12)\n")
    add(f"  {wake_status()}")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--quiet", action="store_true")
    arguments = parser.parse_args()

    print("collecting...")
    data = collect(seed=arguments.seed, verbose=not arguments.quiet)
    tiles = choose_tiles(data)
    print(f"  {len(tiles)} tiles chosen")
    print("sweeping...")
    cells, per_tile = sweep(data, tiles, verbose=not arguments.quiet)
    print()
    print(report(data, tiles, cells, per_tile))


if __name__ == "__main__":
    main()
