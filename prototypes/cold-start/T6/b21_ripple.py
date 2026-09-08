"""T6 / [B21](#570): the ripple test — how far does a perturbation reach, and does it stay distinguishable?

[B17](#565) found that `composed_reads` multiplies **learned restriction maps only**
and never touches `agent.sheaf.stalks`, so every reading on [#532](#532) to date is a
property of the transport *operator*. This one is the complement: it perturbs the
**state** and watches what happens, which is the causal property the user's own
statement of the architecture asks for — *"when you drop a rock into a puddle, the
ripples have to reach across. They might be distorted along the way, they might no
longer look anything like it, but the impact is global."*

## The protocol

**A yoked-world impulse.** The confound this rig has to kill is the world: the
agent's motor command is a slice of a node stalk (`Agent.command`), so a perturbed
agent acts differently, the sandbox answers differently, and the divergence that
follows is the *world's* doing rather than the graph's. So the world is put on a
tape:

1. Run the agent live for `T` ticks from a snapshot, recording every
   `(observation, applied)` the world produced. That is the **control**
   trajectory.
2. Restore the snapshot. Add `delta` to one rim cell's node stalk. Replay the
   same `T` ticks by hand — `sheaf.tick()` then `agent.write(obs, applied)` off
   the tape — so the exogenous input is bit-identical to the control's and the
   only difference in the whole system is the rock.

The control is replayed through the same path with a zero perturbation, and
:func:`check_replay` asserts it reproduces the live trajectory **exactly**. If it
does not, every number below is measuring the replay and not the ripple.

**It is an impulse, not a step.** `Agent.tick` is `sheaf.tick()` *then* the
world's write, and every sensorimotor rim stalk is world-written, so the injected
value survives exactly one message-passing phase before the world restores it.
That is the honest shape of dropping a rock: one splash, then watch.

**Learning is off during the reading.** No `PredictionRule`, no `TransportRule` —
the maps and the per-cell surface are frozen at the checkpoint, so what is
measured is propagation through the transport the checkpoint learned, not
propagation plus a moving target.

## What is measured

* **Reach.** `dev(c, t) = ||x_perturbed(c, t) − x_control(c, t)||`, per cell, per
  tick. Reported against two normalisers, because the raw norm is meaningless on
  its own: the cell's own **motion** (rms tick-to-tick change of the control),
  which is the scale at which a deviation is an event rather than a rounding
  artefact, and the **float32 resolution floor** `EPS32 * ||x_control(c)||`,
  below which no downstream computation in this graph could see the difference.
* **Distinguishability.** `K_DIRECTIONS` orthonormal perturbations of equal norm
  are dropped at the same cell, each its own replay. For every cell the `k`
  deviation *trajectories* are turned into a `k x k` Gram, and its participation
  ratio (`effective rank`), its mean pairwise `|cos|`, and its pairwise
  separations against that cell's own floor are read. **The distance at which
  magnitude dies and the distance at which distinguishability dies are different
  numbers, and the gap is the finding.**
* **Hierarchy.** Every cell is tagged with its hop distance from the rock, its
  dome level, and the modality of the nearest boundary cell, so the profile can
  be split up-down against lateral, and within-modality against across.
* **The motor command.** `Agent.command` is a slice of the actuator cell's stalk
  that the world never writes, so it is the one genuinely *causal* output the
  graph has. Its deviation is read every tick as the crispest statement of
  whether the impact was global.
* **Linearity.** The whole reading is repeated at `delta / 100` and the far field
  is checked for linear scaling. A far field that scales is signal; one that does
  not has hit the arithmetic floor and is not there.

Usage::

    PYTHONPATH=src python prototypes/cold-start/T6/b21_ripple.py check --arm shipped
    PYTHONPATH=src python prototypes/cold-start/T6/b21_ripple.py construction
    PYTHONPATH=src python prototypes/cold-start/T6/b21_ripple.py trained --arms shipped --ticks 20000
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
_T0, _T1, _T2, _T3, _T4 = (_HERE.parent / n for n in ("T0", "T1", "T2", "T3", "T4"))
for extra in (_ROOT / "prototypes" / "chart-double-duty-166", _ROOT / "benchmarks", _T0):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


t0 = _load("t0_run", _T0 / "run.py")
t1 = _load("t1_run", _T1 / "run.py")
t2 = _load("t2_run", _T2 / "run.py")
t3 = _load("t3_run", _T3 / "run.py")
arms_mod = _load("t6_arms", _HERE / "arms.py")

from patchworks.graph import CellKind, EdgeKind  # noqa: E402
from patchworks.learning import PredictionRule, TransportRule  # noqa: E402

#: float32's unit round-off. A deviation below `EPS32 * ||x||` is not a
#: difference any float32 arithmetic downstream of it could resolve.
EPS32 = float(np.finfo(np.float32).eps)

#: Ticks replayed after the impulse. The dome's eccentricity from any rim cell is
#: 9 and one hop costs one tick (`message_passing_phase` reconciles against the
#: previous tick's broadcast), so 9 ticks is the earliest the far side can hear
#: anything at all; the rest is time for the ripple to develop and decay.
HORIZON = 32

#: Distinct perturbations dropped at the same cell, for the distinguishability read.
K_DIRECTIONS = 8

#: The rock, as a multiple of the cell's own state norm at the moment it lands.
DELTA_REL = 1.0
#: The ladder's second rung. A far field that does not fall by the same factor
#: has hit the arithmetic floor.
DELTA_REL_SMALL = 0.01

#: The rock shrinks a hundredfold between the two rungs, so a deviation that is
#: the rock shrinks with it and reads ~100 here; one that is the arithmetic floor
#: reads ~1. Ten is a full order of margin on the noise and still two orders of
#: slack on a linear response — deliberately generous to reach.
LINEARITY_MIN = 10.0

#: A deviation is *an event at this cell* when it reaches this share of the
#: cell's own rms tick-to-tick motion. One part in a thousand is generous on
#: purpose: the claim being tested is reach, not fidelity.
EVENT_SHARE = 1e-3


# -- the graph ----------------------------------------------------------------


def adjacency(dome) -> dict[int, list[int]]:
    """Neighbours over every edge a signal can actually cross.

    Drive edges are excluded for `rim_chains`' own reason: the drive cell asserts
    a constant and is not a route between sensorimotor cells.
    """
    adj: dict[int, list[int]] = collections.defaultdict(list)
    for edge in dome.edges:
        if edge.kind is EdgeKind.DRIVE:
            continue
        adj[edge.u].append(edge.v)
        adj[edge.v].append(edge.u)
    return adj


def hop_distance(adj, source: int) -> dict[int, int]:
    """Breadth-first hop distance from one cell."""
    seen = {source: 0}
    queue = collections.deque([source])
    while queue:
        here = queue.popleft()
        for other in adj[here]:
            if other not in seen:
                seen[other] = seen[here] + 1
                queue.append(other)
    return seen


def cell_context(dome) -> dict:
    """Per-cell tags the profile is split by: level, modality, boundary-ness.

    A cell's **modality** is the kind of the nearest sensorimotor boundary cell.
    Interior cells have no modality of their own — the dome is one graph — so
    this is the only available reading of *"does a ripple cross between
    modalities"* that does not invent a partition.
    """
    adj = adjacency(dome)
    rim_kinds = {
        c.id: c.kind.value
        for c in dome.cells
        if c.is_boundary and c.kind is not CellKind.DRIVE
    }
    # Multi-source BFS, one source set per modality, keeping the nearest.
    best: dict[int, tuple[int, str]] = {}
    for kind in sorted(set(rim_kinds.values())):
        sources = [cid for cid, k in rim_kinds.items() if k == kind]
        seen = {s: 0 for s in sources}
        queue = collections.deque(sources)
        while queue:
            here = queue.popleft()
            for other in adj[here]:
                if other not in seen:
                    seen[other] = seen[here] + 1
                    queue.append(other)
        for cid, d in seen.items():
            if cid not in best or d < best[cid][0]:
                best[cid] = (d, kind)
    return {
        "level": {c.id: (None if c.is_boundary else int(c.index.level)) for c in dome.cells},
        "modality": {c.id: best.get(c.id, (None, None))[1] for c in dome.cells},
        "boundary": {c.id: bool(c.is_boundary) for c in dome.cells},
        "kind": {c.id: c.kind.value for c in dome.cells},
    }


# -- the state ----------------------------------------------------------------

_STATE = (
    "stalks",
    "charts",
    "broadcast",
    "incoming",
    "prediction",
    "prior_charts",
    "prior_evidence",
)


def snapshot(sheaf) -> dict:
    """Everything a tick reads and writes. Nothing else moves during a replay.

    The body, the biases, the operators and the maps are untouched by
    :meth:`Sheaf.tick` — only the learning rules move them, and the reading runs
    with the rules off — so they are deliberately not copied.
    """
    snap = {name: getattr(sheaf, name).clone() for name in _STATE}
    snap["ticks"] = int(sheaf.ticks)
    snap["fold_read"] = sheaf.fold_read.state()
    return snap


def restore(sheaf, snap: dict) -> None:
    for name in _STATE:
        getattr(sheaf, name).copy_(snap[name])
    sheaf.ticks = snap["ticks"]
    sheaf.fold_read.load(snap["fold_read"])


@torch.no_grad()
def record_tape(agent, ticks: int):
    """Run live, and keep the world's half of every tick plus the trajectory.

    Returns `(tape, states, commands)`: the `(observation, applied)` pairs the
    world produced, the flat node stalk buffer after each whole tick, and the
    motor command each tick asked for.
    """
    tape = []
    states = np.empty((ticks, agent.sheaf.stalks.numel()), dtype=np.float32)
    commands = np.empty((ticks, agent.command().shape[0]), dtype=np.float32)
    for i in range(ticks):
        outcome = agent.tick()
        tape.append((outcome.observation, outcome.applied))
        states[i] = agent.sheaf.stalks.numpy()
        commands[i] = outcome.command
    return tape, states, commands


@torch.no_grad()
def replay(agent, snap: dict, tape, delta: torch.Tensor | None, where: slice | None):
    """Restore, drop the rock, and re-run the recorded world by hand.

    `Agent.tick` is `sheaf.tick()` then `act()`, and `act` is the whole of the
    world: it clips, steps the sandbox and writes what came back. Replacing it
    with the taped write is exactly `Agent.tick` with the sandbox replaced by its
    own recording, which is what makes the two arms comparable.
    """
    restore(agent.sheaf, snap)
    if delta is not None:
        agent.sheaf.stalks[where] += delta
    states = np.empty((len(tape), agent.sheaf.stalks.numel()), dtype=np.float32)
    commands = np.empty((len(tape), agent.command().shape[0]), dtype=np.float32)
    for i, (observation, applied) in enumerate(tape):
        agent.sheaf.tick()
        commands[i] = agent.command()
        agent.write(observation, applied)
        states[i] = agent.sheaf.stalks.numpy()
    return states, commands


def check_replay(agent, snap, tape, live_states, live_commands) -> dict:
    """The control, put through the replay path. It must come back bit-identical."""
    states, commands = replay(agent, snap, tape, None, None)
    return {
        "state_max_abs_diff": float(np.abs(states - live_states).max()),
        "command_max_abs_diff": float(np.abs(commands - live_commands).max()),
        "bit_identical": bool(
            np.array_equal(states, live_states) and np.array_equal(commands, live_commands)
        ),
    }


# -- the reads ----------------------------------------------------------------


def effective_rank(values: np.ndarray) -> float:
    """`(sum s)^2 / sum s^2` on non-negative spectra — the participation ratio.

    The same statistic `T4/angles.effective_rank` reads on singular values, taken
    here on the eigenvalues of a Gram, which are those singular values squared.
    A Gram of `k` identical responses reads 1; of `k` orthogonal ones, `k`.
    """
    values = np.clip(np.asarray(values, dtype=np.float64), 0.0, None)
    total = values.sum()
    if total <= 0:
        return 0.0
    return float(total**2 / np.square(values).sum())


def cell_slices(dome) -> list[tuple[int, int]]:
    offsets, total = [], 0
    for cell in dome.cells:
        offsets.append((total, total + cell.stalk))
        total += cell.stalk
    return offsets


@torch.no_grad()
def ripple_read(agent, rim: int, seed: int, label: str, *, horizon: int = HORIZON) -> dict:
    """One rock, `K_DIRECTIONS` ways, at one cell, on the surface as it now stands."""
    dome = agent.dome
    bounds = cell_slices(dome)
    ctx = cell_context(dome)
    dist = hop_distance(adjacency(dome), rim)

    snap = snapshot(agent.sheaf)
    tape, control, control_cmd = record_tape(agent, horizon)
    check = check_replay(agent, snap, tape, control, control_cmd)

    lo, hi = bounds[rim]
    width = hi - lo
    scale = float(np.linalg.norm(control[0, lo:hi]))
    if scale <= 0:
        scale = 1.0
    rng = np.random.default_rng(seed)
    # A cell narrower than `K_DIRECTIONS` cannot be given that many distinct
    # rocks; the actuator boundary cell has a stalk of 6. Drop to what fits.
    k_dirs = min(K_DIRECTIONS, width)
    directions = np.linalg.qr(rng.standard_normal((width, k_dirs)))[0].T[:k_dirs]

    # The sham rock: same norm, in a direction the maps annihilate. Whatever it
    # produces downstream is round-off, and that is this reading's noise floor.
    sham = sham_direction(agent, rim)
    sham_states = sham_cmd = None
    if sham is not None:
        vec = torch.as_tensor(sham * DELTA_REL * scale, dtype=agent.sheaf.stalks.dtype)
        sham_states, sham_cmd = replay(agent, snap, tape, vec, slice(lo, hi))

    runs: dict[str, list[np.ndarray]] = {}
    cmds: dict[str, list[np.ndarray]] = {}
    for rel, tag in ((DELTA_REL, "delta"), (DELTA_REL_SMALL, "delta_small")):
        states_i, cmd_i = [], []
        for row in directions:
            vec = torch.as_tensor(row * rel * scale, dtype=agent.sheaf.stalks.dtype)
            s, c = replay(agent, snap, tape, vec, slice(lo, hi))
            states_i.append(s)
            cmd_i.append(c)
        runs[tag] = states_i
        cmds[tag] = cmd_i
    restore(agent.sheaf, snap)

    # -- per-cell norms and the two normalisers -------------------------------
    per_cell_scale = np.array(
        [np.sqrt(np.mean(np.sum(control[:, a:b] ** 2, axis=1))) for a, b in bounds]
    )
    motion = np.array(
        [
            np.sqrt(np.mean(np.sum(np.diff(control[:, a:b], axis=0) ** 2, axis=1)))
            for a, b in bounds
        ]
    )
    floor = EPS32 * per_cell_scale

    def deviations(states_list):
        """`[k, ticks, cells]` deviation norms, and the raw `[k]` deviation blocks."""
        dev = np.empty((len(states_list), len(states_list[0]), len(bounds)))
        for i, states in enumerate(states_list):
            diff = states - control
            for c, (a, b) in enumerate(bounds):
                dev[i, :, c] = np.linalg.norm(diff[:, a:b], axis=1)
        return dev

    dev = deviations(runs["delta"])
    dev_small = deviations(runs["delta_small"])
    if sham_states is not None:
        sham_peak = deviations([sham_states])[0].max(axis=0)   # [cells]
    else:
        sham_peak = np.zeros(len(bounds))

    peak = dev.max(axis=1)          # [k, cells] — the loudest the ripple ever got
    peak_small = dev_small.max(axis=1)
    peak_med = np.median(peak, axis=0)          # over the k directions
    peak_small_med = np.median(peak_small, axis=0)

    # -- the floor, measured ---------------------------------------------------
    # The sham rock turns out to change **nothing at all** — `ker(F)` is closed in
    # float32 too, so the broadcast comes back bit-identical and there is no
    # round-off to measure there. The floor therefore has to come from the ladder
    # instead, and `570-ripple-ladder.json` is what establishes it: from `delta`
    # down through four decades the *far* field does not move, sitting at a level
    # the rock's size has no influence over, while the *near* field tracks the
    # rock decade for decade. That level is round-off, amplified by the dynamics
    # to its own equilibrium, and the small rock's deviation measures it.
    #
    # So the criterion is **linearity, not magnitude**: a deviation at a cell is
    # the rock only if shrinking the rock a hundredfold shrinks it too. Signal
    # scales; an arithmetic floor does not.
    linearity = np.where(peak_small_med > 0, peak_med / np.maximum(peak_small_med, 1e-300), np.inf)
    noise_traj = np.array(
        [
            np.median(
                [np.linalg.norm((s - control)[:, a:b]) for s in runs["delta_small"]]
            )
            for a, b in bounds
        ]
    )
    measured_floor = np.maximum(peak_small_med, floor)

    # -- distinguishability, per cell ----------------------------------------
    # G[i, j] = sum_t <dev_i(c, t), dev_j(c, t)> over the whole trajectory, so the
    # read uses every tick rather than one arbitrarily chosen frame.
    diffs = [states - control for states in runs["delta"]]
    d_eff = np.zeros(len(bounds))
    mean_abs_cos = np.full(len(bounds), np.nan)
    min_sep = np.zeros(len(bounds))
    sep_over_floor = np.zeros(len(bounds))
    for c, (a, b) in enumerate(bounds):
        block = np.stack([d[:, a:b].reshape(-1) for d in diffs], axis=0).astype(np.float64)
        gram = block @ block.T
        vals = np.linalg.eigvalsh(gram)
        d_eff[c] = effective_rank(vals)
        norms = np.sqrt(np.clip(np.diag(gram), 0, None))
        if norms.min() > 0 and k_dirs >= 2:
            cos = gram / np.outer(norms, norms)
            off = cos[~np.eye(k_dirs, dtype=bool)]
            mean_abs_cos[c] = float(np.abs(off).mean())
            sep = np.array(
                [
                    np.linalg.norm(block[i] - block[j])
                    for i in range(k_dirs)
                    for j in range(i + 1, k_dirs)
                ]
            )
            min_sep[c] = float(sep.min())
            # The trajectory's floor is the per-tick floor over `horizon` ticks.
            # Two rocks are told apart at this cell only if what separates their
            # responses is bigger than what the sham rock — which carries nothing —
            # puts there on its own.
            traj_floor = max(noise_traj[c], floor[c] * np.sqrt(len(diffs[0])))
            sep_over_floor[c] = float(sep.min() / traj_floor) if traj_floor > 0 else np.inf

    # -- the profile, by hop distance ----------------------------------------
    interior = [c.id for c in dome.cells if not c.is_boundary]
    profile = []
    for d in sorted({dist[c] for c in interior if c in dist}):
        ids = [c for c in interior if dist.get(c) == d]
        m = np.array([motion[c] for c in ids])
        pk = np.array([peak_med[c] for c in ids])
        fl = np.array([measured_floor[c] for c in ids])
        snr = np.where(fl > 0, pk / np.where(fl > 0, fl, 1), np.inf)
        ratio = np.where(m > 0, pk / np.where(m > 0, m, 1), np.nan)
        lin = np.array([linearity[c] for c in ids])
        profile.append(
            {
                "distance": int(d),
                "cells": len(ids),
                "levels": sorted({ctx["level"][c] for c in ids if ctx["level"][c] is not None}),
                "peak_dev_median": float(np.median(pk)),
                "sham_peak_median": float(np.median([sham_peak[c] for c in ids])),
                "snr_median": float(np.median(snr)),
                "snr_max": float(np.max(snr)),
                #: The reach criterion: the share of cells at this hop whose
                #: deviation actually tracks the rock's size. `LINEARITY_MIN` of
                #: the hundredfold change is the bar; pure round-off reads 1.
                "signal_share": float(np.mean(lin >= LINEARITY_MIN)),
                "above_floor_share": float(np.mean(snr > 10.0)),
                "dev_over_motion_median": float(np.nanmedian(ratio)),
                "dev_over_motion_max": float(np.nanmax(ratio)),
                "over_floor_share": float(np.mean(pk > fl)),
                "event_share": float(np.nanmean(ratio > EVENT_SHARE)),
                "linearity_ratio_median": float(np.nanmedian(lin)),
                "d_eff_median": float(np.median(d_eff[ids])),
                "d_eff_max": float(np.max(d_eff[ids])),
                "mean_abs_cos_median": float(np.nanmedian(mean_abs_cos[ids])),
                "sep_over_floor_median": float(np.median(sep_over_floor[ids])),
                "sep_over_floor_min": float(np.min(sep_over_floor[ids])),
            }
        )

    # -- hierarchy: at matched distance, does level or modality matter? -------
    by_level = []
    for d, level in sorted(
        {(dist[c], ctx["level"][c]) for c in interior if c in dist}
    ):
        ids = [c for c in interior if dist.get(c) == d and ctx["level"][c] == level]
        m = np.array([motion[c] for c in ids])
        pk = np.array([peak_med[c] for c in ids])
        ratio = np.where(m > 0, pk / np.where(m > 0, m, 1), np.nan)
        by_level.append(
            {
                "distance": int(d),
                "level": int(level),
                "cells": len(ids),
                "dev_over_motion_median": float(np.nanmedian(ratio)),
                "d_eff_median": float(np.median(d_eff[ids])),
            }
        )

    by_modality = []
    for d, mod in sorted(
        {(dist[c], ctx["modality"][c]) for c in interior if c in dist},
        key=lambda kv: (kv[0], str(kv[1])),
    ):
        ids = [c for c in interior if dist.get(c) == d and ctx["modality"][c] == mod]
        m = np.array([motion[c] for c in ids])
        pk = np.array([peak_med[c] for c in ids])
        ratio = np.where(m > 0, pk / np.where(m > 0, m, 1), np.nan)
        by_modality.append(
            {
                "distance": int(d),
                "modality": mod,
                "cells": len(ids),
                "dev_over_motion_median": float(np.nanmedian(ratio)),
                "d_eff_median": float(np.median(d_eff[ids])),
            }
        )

    # -- the motor command: the one output the world never writes -------------
    cmd_dev = np.stack(
        [np.linalg.norm(c - control_cmd, axis=1) for c in cmds["delta"]], axis=0
    )
    cmd_scale = float(np.sqrt(np.mean(np.sum(control_cmd**2, axis=1))))
    cmd_motion = float(
        np.sqrt(np.mean(np.sum(np.diff(control_cmd, axis=0) ** 2, axis=1)))
    )
    cmd_block = np.stack(
        [(c - control_cmd).reshape(-1) for c in cmds["delta"]], axis=0
    ).astype(np.float64)
    cmd_gram = cmd_block @ cmd_block.T
    cmd_norms = np.sqrt(np.clip(np.diag(cmd_gram), 0, None))
    cmd_cos = (
        float(
            np.abs(
                (cmd_gram / np.outer(cmd_norms, cmd_norms))[
                    ~np.eye(k_dirs, dtype=bool)
                ]
            ).mean()
        )
        if cmd_norms.min() > 0 and k_dirs >= 2
        else None
    )
    first = np.argmax(cmd_dev.max(axis=0) > 0) if (cmd_dev > 0).any() else None
    # The command gets the same linearity test as every cell: a deviation that is
    # the rock shrinks with the rock, and one that is round-off does not.
    cmd_small = np.stack(
        [np.linalg.norm(c - control_cmd, axis=1) for c in cmds["delta_small"]], axis=0
    )
    cmd_peak = float(np.median(cmd_dev.max(axis=1)))
    cmd_peak_small = float(np.median(cmd_small.max(axis=1)))
    cmd_linearity = cmd_peak / cmd_peak_small if cmd_peak_small > 0 else float("inf")

    return {
        "label": label,
        "rim": int(rim),
        "rim_kind": ctx["kind"][rim],
        "horizon": int(horizon),
        "k_directions": int(k_dirs),
        "delta_rel": DELTA_REL,
        "delta_norm": float(DELTA_REL * scale),
        "rim_state_norm": scale,
        "replay_check": check,
        "sham": {
            "available": sham is not None,
            "interior_peak_median": float(
                np.median([sham_peak[c] for c in interior])
            ),
            "command_peak": (
                None
                if sham_cmd is None
                else float(np.linalg.norm(sham_cmd - control_cmd, axis=1).max())
            ),
        },
        #: The width of everything leaving this cell. A rock is dropped in a
        #: `stalk`-dimensional space and leaves through `sum m_e` dimensions, so
        #: this is the ceiling on how many distinct rocks could *ever* be told
        #: apart anywhere downstream — the structural bound `d_eff` is read against.
        "rim_lanes": {
            "stalk": int(dome.cells[rim].stalk),
            "edges": len(dome.incident[rim]),
            "sum_m": int(sum(dome.edges[e].m for e in dome.incident[rim])),
            "widths": [int(dome.edges[e].m) for e in dome.incident[rim]],
        },
        #: How far the other modalities are, so *"does it cross between
        #: modalities"* is answered against the reach rather than asserted.
        "modality_distance": {
            mod: int(
                min(
                    dist[c.id]
                    for c in dome.cells
                    if c.is_boundary and c.kind.value == mod and c.id in dist
                )
            )
            for mod in sorted(
                {
                    c.kind.value
                    for c in dome.cells
                    if c.is_boundary and c.kind is not CellKind.DRIVE
                }
            )
        },
        "eccentricity": int(max(dist.values())),
        "cells_reachable": len(dist),
        "cells_total": len(dome.cells),
        "boundary_dev_max": float(
            max(peak_med[c.id] for c in dome.cells if c.is_boundary)
        ),
        "profile": profile,
        "by_level": by_level,
        "by_modality": by_modality,
        "command": {
            "control_rms": cmd_scale,
            "control_motion_rms": cmd_motion,
            "peak_dev_median": cmd_peak,
            "peak_dev_small_median": cmd_peak_small,
            "linearity": cmd_linearity,
            "is_signal": bool(cmd_linearity >= LINEARITY_MIN),
            "peak_over_motion": float(
                cmd_peak / cmd_motion if cmd_motion > 0 else np.nan
            ),
            "first_nonzero_tick": None if first is None else int(first),
            "d_eff": effective_rank(np.linalg.eigvalsh(cmd_gram)),
            "mean_abs_cos": cmd_cos,
        },
    }


def _line(read: dict, prefix: str) -> str:
    rows = read["profile"]
    tail = rows[-1] if rows else {}
    alive = [r["distance"] for r in rows if r["signal_share"] >= 0.5]
    distinct = [r["distance"] for r in rows if r["sep_over_floor_median"] >= 10.0]
    return (
        f"{prefix} ecc {read['eccentricity']} | signal to hop "
        f"{max(alive) if alive else 'none'} | distinct to hop "
        f"{max(distinct) if distinct else 'none'} | far d/motion "
        f"{tail.get('dev_over_motion_median', float('nan')):.3g} d_eff "
        f"{tail.get('d_eff_median', float('nan')):.3f} cos "
        f"{tail.get('mean_abs_cos_median', float('nan')):.4f} | cmd d/motion "
        f"{read['command']['peak_over_motion']:.3g} lin {read['command']['linearity']:.3g} "
        f"d_eff {read['command']['d_eff']:.3f}"
    )


# -- drivers ------------------------------------------------------------------


def rim_of_kind(dome, kind: str) -> int:
    """The lowest-numbered rim cell of one modality, so the choice is not a knob."""
    for cell in dome.cells:
        if cell.is_boundary and cell.kind.value == kind:
            return int(cell.id)
    raise KeyError(kind)


@torch.no_grad()
def sham_direction(agent, rim: int) -> np.ndarray | None:
    """A rock the graph provably cannot transmit: a direction in `ker(F)`.

    **This is the noise control the whole reading rests on.** A perturbation of
    the rim cell's stalk reaches the rest of the graph through exactly one
    object — `maps.restrict`, which applies that cell's restriction map on each
    incident edge. A direction in the intersection of those maps' null spaces is
    therefore invisible to the graph *in exact arithmetic*: it changes the
    broadcast by identically zero, so every deviation it produces downstream is
    float32 round-off and nothing else.

    Dropping a sham rock of the **same norm** as the real one, on the same
    surface, through the same replay, gives a **measured** noise floor per cell
    rather than a chosen one — which is what `#570`'s *"where does it fall into
    the noise floor?"* actually asks for. Returns `None` where the maps leave no
    null space to draw from, in which case the floor falls back to `EPS32`.
    """
    dome, maps = agent.dome, agent.sheaf.maps
    from patchworks.restriction import pair_index

    width = dome.cells[rim].stalk
    blocks = []
    for edge_id in dome.incident[rim]:
        edge = dome.edges[edge_id]
        side = 0 if edge.u == rim else 1
        block = maps.maps[pair_index(edge_id, side)][: edge.m, :width]
        blocks.append(block.detach().double().numpy())
    if not blocks:
        return None
    stacked = np.concatenate(blocks, axis=0)
    # The null space, from the trailing right singular vectors.
    _, s, vh = np.linalg.svd(stacked)
    rank = int((s > max(stacked.shape) * np.finfo(np.float64).eps * s[0]).sum())
    if rank >= width:
        return None
    null = vh[rank:]
    weights = np.random.default_rng(0).standard_normal(null.shape[0])
    vec = weights @ null
    return vec / np.linalg.norm(vec)


#: The ladder `run_ladder` walks, in multiples of the perturbed cell's own state
#: norm. Six decades, because one decade cannot tell a saturating response from
#: an arithmetic floor and the difference decides what the far field *is*.
LADDER = (1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8)


@torch.no_grad()
def ladder_read(agent, rim: int, seed: int, *, horizon: int = HORIZON) -> dict:
    """Peak deviation against rock size, per hop, over six decades.

    The one diagnostic that separates the three things a rock-size-independent
    far field could be: a **saturating** response (peak grows then flattens at
    the top of the ladder), a **threshold** response (peak is flat all the way
    down, because what travels is a discrete flip whose size the state sets, not
    the rock), and an **arithmetic floor** (peak flat and at the level round-off
    would reach). It also reports the deviation's shape in *time*, which is what
    says whether the ripple decays or is being amplified.
    """
    dome = agent.dome
    bounds = cell_slices(dome)
    dist = hop_distance(adjacency(dome), rim)
    interior = [c.id for c in dome.cells if not c.is_boundary]

    snap = snapshot(agent.sheaf)
    tape, control, _ = record_tape(agent, horizon)
    lo, hi = bounds[rim]
    scale = float(np.linalg.norm(control[0, lo:hi])) or 1.0
    rng = np.random.default_rng(seed)
    direction = rng.standard_normal(hi - lo)
    direction /= np.linalg.norm(direction)

    per_cell_scale = np.array(
        [np.sqrt(np.mean(np.sum(control[:, a:b] ** 2, axis=1))) for a, b in bounds]
    )
    motion = np.array(
        [
            np.sqrt(np.mean(np.sum(np.diff(control[:, a:b], axis=0) ** 2, axis=1)))
            for a, b in bounds
        ]
    )

    hops = sorted({dist[c] for c in interior if c in dist})
    rungs = []
    for rel in LADDER:
        vec = torch.as_tensor(direction * rel * scale, dtype=agent.sheaf.stalks.dtype)
        states, _ = replay(agent, snap, tape, vec, slice(lo, hi))
        diff = states - control
        dev = np.stack(
            [np.linalg.norm(diff[:, a:b], axis=1) for a, b in bounds], axis=1
        )  # [ticks, cells]
        peak = dev.max(axis=0)
        rungs.append(
            {
                "delta_rel": rel,
                "delta_norm": float(rel * scale),
                "by_hop": [
                    {
                        "distance": int(d),
                        "peak_median": float(
                            np.median([peak[c] for c in interior if dist.get(c) == d])
                        ),
                    }
                    for d in hops
                ],
                # The whole-graph deviation per tick: does the ripple decay, hold,
                # or grow? A growing curve at a fixed rock size is amplification.
                "total_by_tick": [
                    float(np.linalg.norm(diff[t])) for t in range(diff.shape[0])
                ],
            }
        )
    restore(agent.sheaf, snap)
    return {
        "rim": int(rim),
        "rim_kind": {c.id: c.kind.value for c in dome.cells}[rim],
        "horizon": int(horizon),
        "rim_state_norm": scale,
        "eps32": EPS32,
        "hops": hops,
        "interior_scale_median": float(np.median([per_cell_scale[c] for c in interior])),
        "interior_motion_median": float(np.median([motion[c] for c in interior])),
        "rungs": rungs,
    }


def run_ladder(args) -> None:
    record = {"issue": 570, "reading": "peak deviation against rock size, per hop",
              "ladder": list(LADDER), "rows": []}
    for arm in args.arms:
        env, agent = arms_mod.build_arm(arm, args.seed)
        try:
            for _ in t0.run_ticks(agent, args.warm, seed=args.seed):
                pass
            read = ladder_read(agent, rim_of_kind(agent.dome, args.kind), args.seed,
                               horizon=args.horizon)
            record["rows"].append({"arm": arm, "warm": args.warm, "ladder": read})
            print(f"  {arm} {args.kind} rim {read['rim']} "
                  f"(state norm {read['rim_state_norm']:.4g}, interior motion median "
                  f"{read['interior_motion_median']:.4g})", flush=True)
            header = "  " + "".join(f"hop{d:<9}" for d in read["hops"])
            print(f"    {'delta':>10} {header}", flush=True)
            for rung in read["rungs"]:
                cells = "".join(f"{h['peak_median']:<12.3e}" for h in rung["by_hop"])
                print(f"    {rung['delta_rel']:>10.0e}  {cells}", flush=True)
            for rung in read["rungs"]:
                tot = rung["total_by_tick"]
                marks = [0, 1, 2, 4, 8, 16, len(tot) - 1]
                shown = " ".join(f"t{m}={tot[m]:.2e}" for m in marks if m < len(tot))
                print(f"    {rung['delta_rel']:>10.0e}  {shown}", flush=True)
        finally:
            env.close()
        args.out.write_text(json.dumps(record, indent=1))
    print(f"wrote {args.out.name}", flush=True)


def run_check(args) -> None:
    """Does the replay reproduce the live run, and does the impulse do anything at all?"""
    env, agent = arms_mod.build_arm(args.arm, args.seed)
    try:
        for _ in t0.run_ticks(agent, args.warm, seed=args.seed):
            pass
        read = ripple_read(
            agent, rim_of_kind(agent.dome, args.kind), args.seed,
            f"{args.arm} check", horizon=args.horizon,
        )
        print(json.dumps(read["replay_check"], indent=1), flush=True)
        print(_line(read, f"  {args.arm}:"), flush=True)
        for row in read["profile"]:
            print(
                f"    hop {row['distance']}: n {row['cells']:>3} "
                f"peak {row['peak_dev_median']:.3e} sham {row['sham_peak_median']:.3e} "
                f"snr {row['snr_median']:.3g} signal {row['signal_share']:.2f} "
                f"d/motion {row['dev_over_motion_median']:.3e} "
                f"lin {row['linearity_ratio_median']:.1f} "
                f"d_eff {row['d_eff_median']:.3f} cos {row['mean_abs_cos_median']:.4f} "
                f"sep/floor {row['sep_over_floor_median']:.3g}",
                flush=True,
            )
    finally:
        env.close()


def _record(issue_note: str) -> dict:
    return {"issue": 570, "reading": issue_note, "eps32": EPS32,
            "horizon": HORIZON, "k_directions": K_DIRECTIONS,
            "delta_rel": DELTA_REL, "delta_rel_small": DELTA_REL_SMALL,
            "event_share_threshold": EVENT_SHARE, "rows": []}


def run_construction(args) -> None:
    record = _record("the ripple at construction, across #560's p sweep")
    for arm in args.arms:
        for kind in args.kinds:
            env, agent = arms_mod.build_arm(arm, args.seed)
            try:
                # A construction surface has never ticked, so it has no state to
                # perturb and no motion to measure a deviation against. `warm`
                # ticks with the rules **off** give it an operating point without
                # giving it any learning — the construction reading stays a
                # construction reading. B11's warning stands either way: this is
                # not the number to rule on.
                for _ in t0.run_ticks(agent, args.warm, seed=args.seed):
                    pass
                rim = rim_of_kind(agent.dome, kind)
                read = ripple_read(agent, rim, args.seed, f"{arm} {kind}")
                _, reserve_p = arms_mod.ARMS[arm]
                record["rows"].append(
                    {"arm": arm, "reserve_p": reserve_p, "seed": args.seed,
                     "warm": args.warm, "trained_ticks": 0, "ripple": read}
                )
                print(_line(read, f"  {arm:>13} {kind:>14}:"), flush=True)
            finally:
                env.close()
            args.out.write_text(json.dumps(record, indent=1))
    print(f"wrote {args.out.name}", flush=True)


def train_one(arm: str, args) -> None:
    """One arm, trained, with the ripple taken at every checkpoint.

    Sequential and checkpointed, as #555's notes require: long runs get killed
    when parallelised, and the record is written as each checkpoint lands.
    """
    started = time.time()
    out = _HERE / f"570-ripple-{arm}-{args.condition}-seed{args.seed}-{args.ticks}.json"
    if out.exists():
        print(f"[B21] {out.name} already at the horizon, skipping", flush=True)
        return
    inflight = out.with_suffix(".inflight.json")
    cond = t3.CONDITIONS[args.condition]
    pin, c = cond["pin"], cond["c"]
    env, agent = arms_mod.build_arm(arm, args.seed)
    try:
        if pin:
            t1.pin_drive_edges(agent)
        dome = agent.dome
        chains = t2.rim_chains(dome)
        _, reserve_p = arms_mod.ARMS[arm]
        rims = {k: rim_of_kind(dome, k) for k in args.kinds}
        record = {
            "issue": 570,
            "reading": "the ripple under training",
            "arm": arm,
            "reserve_p": reserve_p,
            "condition": args.condition,
            "seed": args.seed,
            "ticks": args.ticks,
            "eps32": EPS32,
            "horizon": HORIZON,
            "k_directions": K_DIRECTIONS,
            "delta_rel": DELTA_REL,
            "delta_rel_small": DELTA_REL_SMALL,
            "event_share_threshold": EVENT_SHARE,
            "rims": rims,
            "privacy": arms_mod.privacy_read(dome, reserve_p),
            "widths": arms_mod.widths_read(dome, chains),
            "checkpoints": [],
        }
        recorder = t2.EdgeRecorder(agent)
        bias = PredictionRule(agent.sheaf, operator_rate_ratio=c)
        transport = TransportRule(agent.sheaf)

        ladder = [cp for cp in args.checkpoints if cp <= args.ticks]
        if args.ticks not in ladder:
            ladder.append(args.ticks)
        seen = 0
        for target in ladder:
            for _ in t0.teaching_read(
                agent, target - seen, args.seed + seen, recorder, bias, transport
            ):
                pass
            seen = target
            entry = {"ticks": target, "ripples": []}
            for kind, rim in rims.items():
                read = ripple_read(agent, rim, args.seed, f"{arm} {kind} @{target}")
                entry["ripples"].append(read)
                print(_line(read, f"  {arm} s{args.seed} @{target:>6} {kind:>14}:"), flush=True)
            entry["elapsed_minutes"] = (time.time() - started) / 60.0
            record["checkpoints"].append(entry)
            inflight.write_text(json.dumps(record, indent=1))
            print(f"    ({entry['elapsed_minutes']:.1f} min)", flush=True)
        inflight.replace(out)
        print(f"wrote {out.name}", flush=True)
    finally:
        env.close()


def run_trained(args) -> None:
    """The arms, **one at a time**: parallel long runs trip this box's memory guard."""
    for arm in args.arms:
        print(f"[B21] {arm} {args.condition} seed {args.seed}, {args.ticks} ticks", flush=True)
        train_one(arm, args)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)

    k = sub.add_parser("check", help="is the replay faithful and the impulse live?")
    k.add_argument("--arm", default="shipped")
    k.add_argument("--kind", default="patch")
    k.add_argument("--seed", type=int, default=42)
    k.add_argument("--warm", type=int, default=200)
    k.add_argument("--horizon", type=int, default=HORIZON)
    k.set_defaults(func=run_check)

    l = sub.add_parser("ladder", help="peak deviation against rock size, over six decades")
    l.add_argument("--arms", nargs="+", default=["shipped"])
    l.add_argument("--kind", default="patch")
    l.add_argument("--seed", type=int, default=42)
    l.add_argument("--warm", type=int, default=200)
    l.add_argument("--horizon", type=int, default=HORIZON)
    l.add_argument("--out", type=Path, default=_HERE / "570-ripple-ladder.json")
    l.set_defaults(func=run_ladder)

    c = sub.add_parser("construction", help="the ripple at construction, across the p sweep")
    c.add_argument("--arms", nargs="+", default=["shipped", "reserve_p8", "reserve_p16", "reserve_p24"])
    c.add_argument("--kinds", nargs="+", default=["patch", "proprioceptive", "touch", "actuator"])
    c.add_argument("--seed", type=int, default=42)
    c.add_argument("--warm", type=int, default=200)
    c.add_argument("--out", type=Path, default=_HERE / "570-ripple-construction.json")
    c.set_defaults(func=run_construction)

    t = sub.add_parser("trained", help="arms trained one at a time, ripple at every checkpoint")
    t.add_argument("--arms", nargs="+", default=["shipped"])
    t.add_argument("--kinds", nargs="+", default=["patch", "actuator"])
    t.add_argument("--condition", choices=sorted(t3.CONDITIONS), default="baseline")
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--ticks", type=int, default=20_000)
    t.add_argument("--checkpoints", type=int, nargs="+", default=[1000, 5000, 20000])
    t.set_defaults(func=run_trained)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
