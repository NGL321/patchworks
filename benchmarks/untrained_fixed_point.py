"""The untrained agent's saturated fixed point, characterised (ticket #120).

Noah saw the arm contract into itself and stay there, and #120 asked which of
two things that is: the baseline an untrained agent is *expected* to start
from, or a broken feedback path. The two call for opposite responses, and
nothing in the build distinguished them — #105 asserts only that the assembled
loop runs and stays finite, which a locked loop does.

This script is the measurement that distinguishes them. It **asserts nothing**
and is deliberately not a test: the whole question #120 asks is whether today's
behaviour is correct, and a test pinning today's numbers would answer it by
assumption. What stands in the suite is one smoke test that this file still
runs (`tests/test_untrained_fixed_point.py`), on the same footing as every
other benchmark here.

Five measurements, five subcommands, in the order the ticket asks for them::

    python benchmarks/untrained_fixed_point.py characterise
    python benchmarks/untrained_fixed_point.py sensitivity
    python benchmarks/untrained_fixed_point.py attenuation
    python benchmarks/untrained_fixed_point.py drive
    python benchmarks/untrained_fixed_point.py learning --ticks 100000

**`characterise`** runs an untrained agent across three seeds, two task splits
and both domes, and records the pose, the command, the travel and the
disagreement it settles at. Identical numbers across seeds would point at
wiring; numbers that move with the seed point at initialisation.

**`sensitivity`** is the one that answers *does the sensory path reach the core
cells, and does the efference copy land*. It runs to the fixed point, then
holds the world still and re-runs the same graph from the same state under six
altered external writes — the render blanked, proprioception shifted, touch
raised, the efference copy shifted, and the drive's standing assertion turned
off and turned up tenfold. The difference between the settled node stalks is
the influence that input has, per level, in the units the stalk is in. Holding
the world still is what makes it a clean read: the world is out of the loop, so
what is left is the graph's own transfer.

`--learn N` takes the same reading on an adapting surface that has had N ticks
of both rules first, which is the only way to tell an attenuation the draw
produced from one learning cannot lift. The rules stay off during the hold
either way, so the variants differ in the external write and in nothing else.

The drive variants are ADR-0009's own instrument, used for the one job that ADR
permits it — "pinning a stalk to debug, and to isolate whether the drive edge
is doing the work". Nothing here changes :data:`~patchworks.agent.DRIVE_ASSERTION`.

**`attenuation`** decomposes the per-hop transfer `sensitivity` measures into
its two factors: the shared body's evidence-to-prediction gain, and one
message-passing step's transfer of a neighbour's belief into a node stalk. The
second is the one with a ceiling in the record — the reconciliation gain is
`γ / max(Σ_e m_e, ρ²·deg)` and ADR-0010 caps every map at `‖F‖_F ≤ ρ` — so the
run reports what the transport rule could grow it to as well as what it is. It
takes `--learn N` too, which is how to see *which* factor learning moves: the
gauge bounds the edge one and nothing in the record bounds the body's.

**`drive`** asks the same question from the other end: the world back in the
loop, the run entire, and the drive's assertion held at one value against
another for the whole of it. `sensitivity` reads what the assertion is worth to
the *graph*; this reads what it is worth to the *arm*, which is what ADR-0009
built it for.

**`learning`** turns both halves of the local learning rule on and runs long,
reporting the command, the pose, the travel, and the paired per-edge instrument
of `patchworks.diagnostics` on a cadence. The pairing is the point and is
reported as a pair: energy falling with effective rank sliding toward 1 is
collapse, energy falling at rest with rank steady is the lag floor draining,
and neither number alone tells them apart. `whole_graph()` is left off — it
measured ~9 s a reading on the real dome, which is not a thing to put on a
100k-tick run.

Runtimes on the development laptop, for budgeting: `characterise` is ~40 s on
the small dome and ~1 min on the real one, `sensitivity` ~1 min and
`attenuation` ~20 s on the real dome, `drive` ~1 min a seed, and `learning` runs
at ~105 ticks/s on the small dome with both rules on and ~50 ticks/s on the real
one. `attenuation` is the one whose cost is not ticks: it runs one
message-passing phase per edge endpoint whose far end runs a body, 1091 of
them on the real dome and 80 on the small one.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

from patchworks.agent import DRIVE_ASSERTION, Agent, run
from patchworks.diagnostics import Condition, Diagnostics
from patchworks.graph import DEFAULT_SPEC, Dome, DomeSpec, build_graph
from patchworks.learning import BiasRule, SparsityAnneal, TransportRule
from patchworks.restriction import pair_index
from patchworks.sandbox import PlanarPushSandbox
from patchworks.sandbox.env import ARM_JOINTS
from patchworks.tick import Sheaf

# The small dome the suite shares, from the one place it is defined
# (`tests/conftest.py`). Imported rather than copied for that file's own
# reason: a second copy drifting from it would leave this script measuring a
# different graph than the one every test names, and a benchmark aimed at the
# wrong dome does not fail -- it reports someone else's numbers. Under pytest
# `tests/` is already on the path; run as a script it is not, so it is put
# there relative to this file rather than to a working directory.
_TESTS = str(Path(__file__).resolve().parent.parent / "tests")
if _TESTS not in sys.path:
    sys.path.append(_TESTS)
from conftest import SMALL  # noqa: E402

#: The render each dome tiles. A patch cell's node stalk is 48 numbers written
#: raw, laid out `patch_grid` x `patch_grid` over the render, so this follows
#: from the spec rather than being a free choice -- `Agent` refuses any other.
IMAGE_SIZE = {SMALL.patch_grid: 16, DEFAULT_SPEC.patch_grid: 64}

#: The state one tick moves, and therefore the whole of what has to be put back
#: to re-run a graph from the same place. The body, the biases and the maps are
#: not in it: no learning runs in `sensitivity`, so they do not move.
_TICK_STATE = (
    "stalks",
    "charts",
    "broadcast",
    "incoming",
    "prediction",
    "prior_charts",
    "prior_evidence",
)

#: How far the arm has to be from a joint's stop to count as off it, in radians.
#: A hundredth of a radian, 0.57 degrees. Loose enough for MuJoCo's limits to be
#: the soft constraints they are -- a joint held against its stop by sustained
#: torque settles a few thousandths of a radian past it, which at a tenth of
#: this would read as off the stop -- and tight enough that the nearest thing
#: #120 measured genuinely off a stop, a link resting on a puck 0.04 rad short
#: of one, still reads as off it.
AT_LIMIT = 1e-2


def arm_limits(env: PlanarPushSandbox) -> np.ndarray:
    """`[joints, 2]`: each arm joint's stops, read off the arena.

    Off the model rather than off a constant, because a stop is a fact about
    the body and `tests/test_sandbox_world.py` is where the arena is held to
    the recorded ranges. The env exposes no accessor for them -- deliberately,
    per its `observation_space`, which declines to bound `qpos` because
    `disturb_arm()` takes an impulse of any size -- so they are looked up by
    the joints' names, the way that test does.
    """
    ids = [
        mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in ARM_JOINTS
    ]
    return env.model.jnt_range[ids].copy()


def dome_named(name: str) -> tuple[DomeSpec, int]:
    spec = SMALL if name == "small" else DEFAULT_SPEC
    return spec, IMAGE_SIZE[spec.patch_grid]


def build(name: str, split: str, seed: int) -> tuple[PlanarPushSandbox, Agent]:
    spec, image_size = dome_named(name)
    env = PlanarPushSandbox(split=split, image_size=image_size)
    agent = Agent(
        env, dome=build_graph(spec), generator=torch.Generator().manual_seed(seed)
    )
    return env, agent


def snapshot(sheaf: Sheaf) -> dict:
    """Everything a tick moves, cloned, so a run can be re-run from here."""
    state = {name: getattr(sheaf, name).clone() for name in _TICK_STATE}
    state["ticks"] = sheaf.ticks
    return state


def restore(sheaf: Sheaf, state: dict) -> None:
    for name in _TICK_STATE:
        setattr(sheaf, name, state[name].clone())
    sheaf.ticks = state["ticks"]


def levels(dome: Dome) -> list[tuple[int, str]]:
    """Every `(level, column)` group in the dome, deepest last."""
    return sorted({(cell.index.level, cell.index.column) for cell in dome.cells})


def by_level(agent: Agent, per_component: torch.Tensor) -> list[float]:
    """The largest per-component value in each `(level, column)` group."""
    layout = agent.sheaf.layout
    worst = []
    for group in levels(agent.dome):
        worst.append(
            max(
                float(per_component[layout.slice(cell.id)].max())
                for cell in agent.dome.cells
                if (cell.index.level, cell.index.column) == group
            )
        )
    return worst


# -- characterise ----------------------------------------------------------


#: How many ticks at the end of a run the fixed point is read off. Long enough
#: that a command still moving reads as still moving -- #120's own window, and
#: the one its numbers are quoted over.
WINDOW = 300


def settle(agent: Agent, ticks: int, seed: int, window: int = WINDOW):
    """Run, and report what the last `window` ticks did."""
    if ticks < 1:
        raise ValueError(
            f"a fixed point is read off ticks that happened; got ticks={ticks}"
        )
    poses, commands = [], []
    for outcome in run(agent, ticks, seed=seed):
        poses.append(outcome.observation["qpos"].copy())
        commands.append(outcome.command.copy())
    pose = np.array(poses[-window:])
    command = np.array(commands[-window:])
    disagreement = agent.sheaf.disagreement().norm(dim=-1).numpy()
    return {
        "pose": pose[-1],
        "travel": np.abs(np.diff(pose, axis=0)).sum(0),
        "command": command.mean(0),
        "command_sd": command.std(0),
        "disagreement": disagreement,
    }


def characterise(domes, splits, seeds, ticks: int) -> None:
    """The fixed point across seeds, splits and domes."""
    for name in domes:
        limits = None
        print(f"\n{name} dome, {ticks} ticks, last {min(ticks, WINDOW)} summarised")
        header = (
            f"  {'split':14s} {'seed':>4s}  {'pose':>26s}  {'at a stop':>9s}  "
            f"{'command':>28s}  {'command sd':>10s}  {'travel':>8s}  {'disagreement':>22s}"
        )
        print(header)
        for split in splits:
            for seed in seeds:
                env, agent = build(name, split, seed)
                try:
                    if limits is None:
                        limits = arm_limits(env)
                    r = settle(agent, ticks, seed)
                finally:
                    env.close()
                on_stop = np.abs(
                    np.stack([r["pose"] - limits[:, 0], r["pose"] - limits[:, 1]])
                ).min(0) < AT_LIMIT
                d = r["disagreement"]
                print(
                    f"  {split:14s} {seed:4d}  {np.round(r['pose'], 3)!s:>26s}  "
                    f"{''.join('X' if s else '.' for s in on_stop):>9s}  "
                    f"{np.round(r['command'], 4)!s:>28s}  "
                    f"{r['command_sd'].max():10.1e}  "
                    f"{np.abs(r['travel']).sum():8.4f}  "
                    f"mean {d.mean():6.3f} nz {int((d > 0).sum()):4d}/{len(d):4d}"
                )
        print(f"  joint range: {np.round(limits, 3).tolist()}")


# -- sensitivity -----------------------------------------------------------


def hold(agent: Agent, observation: dict, applied: np.ndarray, drive, ticks: int):
    """Hold the world still for `ticks` ticks, writing the same thing every tick.

    The external write is the tick's last word, so writing the same observation
    every tick is exactly "the world did not change" as the graph sees it. The
    world is not stepped at all, which is what takes the physics out of the
    measurement: what is left is the graph settling on its own.

    `drive`, when given, overrides the drive boundary cell's standing assertion
    after :meth:`Agent.write` has laid down :data:`DRIVE_ASSERTION`. That is
    ADR-0009's debugging instrument, and it is written here rather than in the
    library for the reason that ADR gives: instrument, never mechanism.
    """

    def external() -> None:
        agent.write(observation, applied)
        if drive is not None:
            with torch.no_grad():
                agent.sheaf.stalks[agent._drive_slice] = drive

    external()
    command = agent.command()
    for _ in range(ticks):
        agent.sheaf.tick()
        command = agent.command()
        external()
    return agent.sheaf.stalks.clone(), command


def teaching(agent: Agent, ticks: int, seed: int):
    """Run with both rules on, yielding each tick's outcome.

    Written once because two measurements want the same loop: `learning`
    watches it go by, and `sensitivity --learn` only wants the adapting surface
    it leaves behind. The shape is the one `tests/test_assembled_loop.py` fixes
    -- `agent.tick()`, then the rules, never the reverse.

    **The two rules join on different ticks, and the rules themselves say so.**
    The bias rule joins on the first: prediction error is a cell's own quantity
    and crosses no edge, and `BiasRule.gradient` refuses only at `ticks == 0`.
    The transport rule joins on the second, because the first tick reconciles
    against the constructor's zeros and leaves `incoming` zero -- the unit
    delay -- and `TransportRule.gradient` refuses below `ticks < 2` in those
    words. Holding both back to the second tick would drop a legitimate bias
    step for a reason that belongs to the other rule.
    """
    bias = BiasRule(agent.sheaf)
    transport = TransportRule(agent.sheaf, anneal=SparsityAnneal())
    for outcome in run(agent, ticks, seed=seed):
        bias.step()
        if agent.sheaf.ticks > 1:
            transport.step()
        yield outcome


def taught(agent: Agent, ticks: int, seed: int):
    """`teaching`, run to the end; the last tick comes back."""
    outcome = None
    for outcome in teaching(agent, ticks, seed):
        pass
    return outcome


def sensitivity(
    name: str, split: str, seed: int, ticks: int, hold_ticks: int, learn: int = 0
) -> None:
    """What each external input is worth at the fixed point, per level.

    `learn` runs the same measurement on an adapting surface that has had that
    many ticks of both rules, which is the only way to tell an attenuation the
    draw produced from one learning cannot lift. Nothing is learned during the
    hold itself: the rules are off there, so the six variants differ in the
    external write and in nothing else.
    """
    env, agent = build(name, split, seed)
    try:
        if learn:
            outcome = taught(agent, learn, seed)
        else:
            if ticks < 1:
                raise ValueError(
                    "there is no fixed point to be sensitive at without ticks to "
                    f"settle it; got ticks={ticks} and learn={learn}"
                )
            outcome = None
            for outcome in run(agent, ticks, seed=seed):
                pass
        observation, applied = outcome.observation, outcome.applied
        base = snapshot(agent.sheaf)

        def variant(label, obs, ap, drive):
            restore(agent.sheaf, base)
            stalks, command = hold(agent, obs, ap, drive, hold_ticks)
            return label, stalks, command

        blanked = dict(observation, image=np.zeros_like(observation["image"]))
        shifted = dict(
            observation,
            qpos=observation["qpos"] + 0.5,
            qvel=observation["qvel"] + 0.5,
        )
        touched = dict(observation, touch=observation["touch"] + 1.0)

        reference = variant("reference", observation, applied, None)
        variants = [
            variant("render blanked", blanked, applied, None),
            variant("qpos, qvel + 0.5", shifted, applied, None),
            variant("touch + 1.0", touched, applied, None),
            variant("efference + 0.5", observation, applied + 0.5, None),
            variant("drive 1.0 -> 0.0", observation, applied, 0.0),
            variant("drive 1.0 -> 10.0", observation, applied, 10.0),
        ]

        how = (
            f"{learn} ticks with both rules on"
            if learn
            else f"{ticks} ticks, untrained"
        )
        print(
            f"\n{name} dome, split {split!r}, seed {seed}: {how}, then the "
            f"world held still for {hold_ticks}"
        )
        print(f"  settled pose {np.round(observation['qpos'], 4)}")
        print(f"  settled command {np.round(reference[2], 6)}")
        print(
            "\n  the largest change each altered write makes to a node stalk, by "
            "level, and to the command"
        )
        # Eight wide and two apart, which is the `{v:8.2e}` the rows below
        # print in: a narrower header drifts left of its column, and on the real
        # dome's thirteen groups the drift passes a whole column, so a reader
        # lands on the wrong level in the one table that says where a signal
        # dies.
        groups = "  ".join(
            f"{f'L{level}{column[:4]}':>8s}" for level, column in levels(agent.dome)
        )
        print(f"  {'variant':18s} {'command':>9s}  {groups}")
        for label, stalks, command in variants:
            moved = (stalks - reference[1]).abs()
            print(
                f"  {label:18s} {float(np.abs(command - reference[2]).max()):9.2e}  "
                + "  ".join(f"{v:8.2e}" for v in by_level(agent, moved))
            )
        print(
            "\n  float32 carries ~7 significant digits, so against node stalks of "
            "order 1-10\n  anything at 1e-6 is the representation's own floor and "
            "not a signal."
        )
    finally:
        env.close()


# -- attenuation -----------------------------------------------------------


def attenuation(
    name: str, split: str, seed: int, ticks: int, epsilon: float, learn: int = 0
) -> None:
    """The two factors of one hop: the body's gain, and one edge's.

    `learn` takes the reading after that many ticks of both rules, which is what
    says *which* of the two factors learning can move. The gauge bounds the edge
    factor and nothing in the record bounds the body's, so the answer is not
    derivable and has to be measured.
    """
    env, agent = build(name, split, seed)
    try:
        if learn:
            taught(agent, learn, seed)
        else:
            if ticks < 1:
                raise ValueError(
                    "the gains are read at a settled configuration, and a sheaf "
                    f"that has not ticked has none; got ticks={ticks} and "
                    f"learn={learn}"
                )
            for _ in run(agent, ticks, seed=seed):
                pass
        sheaf = agent.sheaf
        base = snapshot(sheaf)
        generator = torch.Generator().manual_seed(seed)

        def nudge(shape: tuple[int, ...]) -> torch.Tensor:
            direction = torch.randn(shape, generator=generator)
            return direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-12) * epsilon

        # The body: how much of a change in the evidence a cell reads survives
        # `encode`/`step`/`decode` into the prediction it puts on its stalk.
        positions = sheaf.layout.predicting_positions
        sheaf.inference_phase()
        before = sheaf.prediction.clone()
        restore(sheaf, base)
        with torch.no_grad():
            sheaf.stalks[positions] += nudge(tuple(positions.shape))
        sheaf.inference_phase()
        body = ((sheaf.prediction - before).norm(dim=-1) / epsilon).numpy()

        # The edge: how much of a change in what *one* neighbour broadcast
        # survives one message-passing step into the node stalk that reconciles
        # with it. Perturbing `broadcast[2e + side]` moves the stalk at the
        # edge's *other* end, because the phase reads its incoming through the
        # partner flip.
        #
        # One endpoint at a time, and only the rows that endpoint has. Both
        # matter and both are easy to get wrong. The phase sums over every
        # incident edge, so perturbing all of them at once and dividing by one
        # `epsilon` attributes a cell's whole response to a single edge -- and a
        # predicting cell here has mean degree 5 to 7. And `broadcast` is padded
        # to the widest edge stalk in the graph while rows past an edge's own
        # `m` are structurally zero in the maps, so a nudge spread over all of
        # them spends most of itself on directions `spread()` discards: half of
        # it on an `m = 4` interior edge, seven eighths on a drive edge.
        restore(sheaf, base)
        sheaf.message_passing_phase()
        quiet = sheaf.stalks.clone()
        norms = sheaf.maps.norms()
        transfers, ceilings = [], []
        for e in agent.dome.edges:
            for side in (0, 1):
                receiver = (e.v, e.u)[side]
                # A boundary cell's stalk is the world's next word anyway, so
                # what a neighbour's belief did to it is not a hop in the taper.
                if agent.dome.cells[receiver].is_boundary:
                    continue
                restore(sheaf, base)
                with torch.no_grad():
                    sheaf.broadcast[pair_index(e.id, side), : e.m] += nudge((e.m,))
                sheaf.message_passing_phase()
                where = sheaf.layout.slice(receiver)
                moved = float((sheaf.stalks[where] - quiet[where]).norm() / epsilon)
                transfers.append(moved)
                # What ADR-0010's gauge leaves the transport rule room to grow
                # *this* transfer to. The step is linear in the receiver's own
                # map -- `gain_r · F_{p^1}ᵀ δ` -- so the ceiling is this
                # endpoint's transfer scaled by its own map's room to `ρ`, and
                # the fleet's ceiling is the mean of those. A ratio of means
                # would be a different number, and on a taught surface a wrong
                # one: the maps do not grow together.
                #
                # The receiver runs a body, so the map governing this transfer
                # is an interior one and genuinely has room. Averaging over
                # every map in the graph would fold in the boundary cells'
                # ~20-26% that are pinned at exactly 1 by the exact gauge and
                # can never move, and would report headroom that does not exist
                # for anything measured here.
                partner = pair_index(e.id, side) ^ 1
                ceilings.append(moved * sheaf.maps.rho / float(norms[partner]))
        edge = np.array(transfers)
        ceiling = np.array(ceilings)
        interior = norms[~sheaf.maps.pinned]

        after = (
            f"{learn} ticks with both rules on" if learn else f"{ticks} ticks, untrained"
        )
        print(f"\n{name} dome, split {split!r}, seed {seed}, after {after}")
        print(
            f"  nudge {epsilon:g}, over {len(body)} predicting cells and "
            f"{len(edge)} edge endpoints whose far end runs a body"
        )
        for label, values in (
            ("body   d|prediction| / d|evidence|", body),
            ("edge   d|node stalk| / d|belief| ", edge),
        ):
            print(
                f"  {label}  mean {values.mean():8.4f}  median "
                f"{np.median(values):8.4f}  max {values.max():8.4f}"
            )
        print(f"  one hop, one tick: mean {body.mean() * edge.mean():.4g}")
        print(
            f"  interior map norms mean {float(interior.mean()):.4g} against rho = "
            f"{sheaf.maps.rho:g} ({int((~sheaf.maps.pinned).sum())} of "
            f"{sheaf.maps.pairs} maps;\n  the rest are boundary cells' and pinned at 1 "
            f"by the exact gauge). Grown to rho, the edge\n  gain would reach "
            f"{ceiling.mean():8.4f} and the hop {body.mean() * ceiling.mean():.4g}, "
            f"on the body's present gain."
        )
        restore(sheaf, base)
    finally:
        env.close()


# -- drive -----------------------------------------------------------------


def driven_run(name: str, split: str, seed: int, ticks: int, assertion: float):
    """A whole run with the drive boundary cell held at `assertion`.

    `Agent.tick` is unrolled here rather than called, for the one thing that
    has to sit between the world's write and the next tick: the drive's stalk
    put back to the value under test. Writing it before
    :meth:`~patchworks.agent.Agent.write` would have it overwritten by
    :data:`~patchworks.agent.DRIVE_ASSERTION`, and the run would measure
    nothing; the ordering is the same one `02-tick-semantics.md` gives the
    external write, and this is a second external writer standing beside it.
    """
    env, agent = build(name, split, seed)
    try:
        observation, _info = env.reset(seed=seed)
        agent.observe(observation)
        with torch.no_grad():
            agent.sheaf.stalks[agent._drive_slice] = assertion
        poses, commands = [], []
        for _ in range(ticks):
            agent.sheaf.tick()
            outcome = agent.act(agent.command())
            with torch.no_grad():
                agent.sheaf.stalks[agent._drive_slice] = assertion
            poses.append(outcome.observation["qpos"].copy())
            commands.append(outcome.command.copy())
        return np.array(poses), np.array(commands)
    finally:
        env.close()


def drive(name: str, split: str, seeds, ticks: int, assertions) -> None:
    """What the standing assertion is worth to a whole trajectory.

    `sensitivity` reads the drive with the world held still, which is the clean
    read of the graph's transfer and says nothing about what the *arm* does.
    This is the other end of the same question: the world in the loop, the run
    entire, one assertion against another. ADR-0009's drive exists so that an
    unmet task is uncomfortable enough to be acted on, and the thing that would
    show it doing that is the arm going somewhere else.
    """
    print(f"\n{name} dome, split {split!r}, {ticks} ticks, world in the loop")
    print(
        f"  the whole trajectory under one assertion against the "
        f"{DRIVE_ASSERTION:g} `Agent` writes"
    )
    for seed in seeds:
        reference_pose, reference_command = driven_run(
            name, split, seed, ticks, DRIVE_ASSERTION
        )
        for assertion in assertions:
            pose, command = driven_run(name, split, seed, ticks, assertion)
            print(
                f"  seed {seed}  drive {assertion:5.1f}:  "
                f"max |Δ pose| {np.abs(pose - reference_pose).max():.2e}  "
                f"max |Δ command| {np.abs(command - reference_command).max():.2e}  "
                f"final pose {np.round(pose[-1], 4)} against "
                f"{np.round(reference_pose[-1], 4)}",
                flush=True,
            )


# -- learning --------------------------------------------------------------


def learning(name: str, split: str, seed: int, ticks: int, every: int) -> None:
    """Both rules on, long, with the paired per-edge instrument on a cadence."""
    env, agent = build(name, split, seed)
    try:
        # Every reading below passes `whole_graph=False`, which is the override
        # `read` offers and is what keeps the expensive half off this run
        # entirely: one whole-graph reading is a `3764 x 3764` eigendecomposition
        # on the real dome, ~9 s, and this run is 100k ticks. The cadence still
        # has to satisfy the multiple-of rule, so it is set to `every` and never
        # consulted.
        diagnostics = Diagnostics(agent.sheaf, every=every, whole_graph_every=every)
        print(f"\n{name} dome, split {split!r}, seed {seed}, both rules on, {ticks} ticks")
        print(
            f"  {'tick':>8s}  {'command':>28s}  {'|d cmd|':>9s}  {'pose':>24s}  "
            f"{'travel':>8s}  {'energy':>9s}  {'eff rank':>8s}  {'|F|':>6s}  {'t/s':>5s}"
        )
        started = time.time()
        commands, poses = [], []
        for i, outcome in enumerate(teaching(agent, ticks, seed), start=1):
            commands.append(outcome.command.copy())
            poses.append(outcome.observation["qpos"].copy())
            if i % every:
                continue
            reading = diagnostics.read(Condition.DRIVEN, whole_graph=False)
            command, pose = np.array(commands), np.array(poses)
            # A window of one row has no difference in it. That is only the
            # first window, since the carry-over below opens every later one on
            # the previous row -- and only at `every == 1` -- but the honest
            # thing to print is that there is nothing to print, not a `nan` out
            # of an empty slice in the column that says whether the command is
            # still moving.
            rate = (
                f"{np.abs(np.diff(command, axis=0)).mean():9.2e}"
                if len(command) > 1
                else f"{'-':>9s}"
            )
            print(
                f"  {i:8d}  {np.round(command[-1], 5)!s:>28s}  "
                f"{rate}  "
                f"{np.round(pose[-1], 3)!s:>24s}  "
                f"{np.abs(np.diff(pose, axis=0)).sum():8.4f}  "
                f"{float(reading.edges.energy.mean()):9.4g}  "
                f"{float(reading.edges.effective_rank.mean()):8.4f}  "
                f"{float(agent.sheaf.maps.norms().mean()):6.3f}  "
                f"{i / (time.time() - started):5.0f}",
                flush=True,
            )
            # The last row of the window opens the next one rather than being
            # dropped, which is what leaves every window after the first with a
            # difference to take however small `every` is. It also fixes a
            # quieter thing: the pose step *between* two windows used to belong
            # to neither, so `travel` was not a partition of the run's travel.
            commands, poses = commands[-1:], poses[-1:]
        print(
            "\n  Read energy and effective rank together, never apart: energy falling\n"
            "  with rank sliding toward 1 is collapse, energy falling with the world\n"
            "  at rest and rank steady is the lag floor draining (CONTEXT.md,\n"
            "  Effective rank). The condition is declared DRIVEN because the world is\n"
            "  in the loop and free to move; whether it *did* is the travel column."
        )
    finally:
        env.close()


# -- the entry point -------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="measurement", required=True)

    one = argparse.ArgumentParser(add_help=False)
    one.add_argument("--dome", default="full", choices=("small", "full"))
    one.add_argument("--split", default="train")
    one.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("characterise", help="the fixed point across seeds and splits")
    # Constrained for the reason the small dome is imported rather than
    # copied: `dome_named` reads anything that is not "small" as the real dome,
    # so `--domes smal` would measure 682 edges and head the table `smal`. A
    # benchmark aimed at the wrong dome does not fail; it reports someone
    # else's numbers.
    p.add_argument("--domes", nargs="+", default=["small", "full"], choices=("small", "full"))
    p.add_argument("--splits", nargs="+", default=["train", "heldout_pair"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--ticks", type=int, default=1500)

    p = sub.add_parser("sensitivity", parents=[one], help="what each input is worth")
    p.add_argument("--ticks", type=int, default=1500)
    p.add_argument("--hold", type=int, default=400)
    p.add_argument(
        "--learn",
        type=int,
        default=0,
        help="run this many ticks with both rules on first, instead of --ticks untrained",
    )

    p = sub.add_parser("attenuation", parents=[one], help="the two factors of one hop")
    p.add_argument("--ticks", type=int, default=1500)
    p.add_argument("--epsilon", type=float, default=1e-3)
    p.add_argument(
        "--learn",
        type=int,
        default=0,
        help="run this many ticks with both rules on first, instead of --ticks untrained",
    )

    p = sub.add_parser("drive", parents=[one], help="the assertion, end to end")
    p.add_argument("--ticks", type=int, default=1500)
    # `None` rather than a list, so that `--seed 5` -- inherited from the
    # shared parent and otherwise silently ignored here -- means what it says.
    # Three ~1-minute runs under seed numbers the caller did not ask for is a
    # quiet way to waste three minutes.
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    p.add_argument("--assertions", nargs="+", type=float, default=[0.0, 10.0])

    p = sub.add_parser("learning", parents=[one], help="both rules on, long")
    p.add_argument("--ticks", type=int, default=100000)
    p.add_argument("--every", type=int, default=2000)

    args = parser.parse_args(argv)
    if args.measurement == "characterise":
        characterise(args.domes, args.splits, args.seeds, args.ticks)
    elif args.measurement == "sensitivity":
        sensitivity(
            args.dome, args.split, args.seed, args.ticks, args.hold, args.learn
        )
    elif args.measurement == "attenuation":
        attenuation(
            args.dome, args.split, args.seed, args.ticks, args.epsilon, args.learn
        )
    elif args.measurement == "drive":
        drive(
            args.dome,
            args.split,
            args.seeds if args.seeds is not None else [args.seed],
            args.ticks,
            args.assertions,
        )
    else:
        learning(args.dome, args.split, args.seed, args.ticks, args.every)


if __name__ == "__main__":
    main()
