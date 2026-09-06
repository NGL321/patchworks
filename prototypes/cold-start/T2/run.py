"""T2 (#522): the induced-activity sweep on the baseline build.

Which of reach, rank and structure is what the cold start is short of?
`#522 <https://github.com/NGL321/patchworks/issues/522>`_.

**Surface.** The shallow dome (`T1/run.py::SHALLOW`, three levels, `core_degree
= 7`; ledger row 2 on #520), the *baseline* build (`rho = 1` off, `c = 1`: T1's
winner is T2b's job), `map/cold-start` at HEAD, seeds 42/43/44. Each run is an
**induced phase** of `T_b in {5k, 20k}` ticks with the supply annealed as
`A(t) = A0 * max(0, 1 - t / T_b)`, then **30k post-phase ticks** with the world
arranged and the supply off -- both rules on throughout. Four conditions x two
`T_b` x three seeds = 24 runs.

**The family: induced activity** (#527's proposal; ADR-0009's *Bootstrapping*
exposure as amended by #519). Task-blind, seeded before the run, reads nothing,
an instrument under ADR-0009's extended instrument clause exactly as `exogenous
arm` and `exogenous drive` were (#481, #495). Three axes:

* **wall** -- *motor*: torque through `act()`, **added** to the graph's own
  command so that what is left at `A = 0` is the graph's own command and the
  efference copy stays honest (`act()` clips the sum and writes the clipped
  value); *sensory*: a synthetic observation written through `Agent.write()`
  onto the 256 vision patches with the world **not stepped** -- the held
  observation is the one the arranged world gave at the phase's start, the
  front is **added** to its render in stalk units, and the somatomotor rim is
  written exactly as the frozen world writes it (held `qpos`/`qvel`/`touch`,
  efference = the clipped command), so the sensory conditions differ from the
  frozen baseline on the vision patches and nowhere else.
* **amplitude** -- *large*: the boundary contract's bound (`agent.action_high`
  for the motor wall; the `PIXEL_SCALE` range, `1.0` in stalk units, for the
  sensory wall); *small*: **babble scale**, which for the sensory wall is
  *derived* rather than chosen: the RMS deviation of the render's patch stalks
  under motor babble at the bound (`--calibrate`, written to
  `calibration.json` and read by every sensory run), so condition D injects the
  energy the babbling arm would have put onto the patches and D-vs-A is a
  structure comparison at matched sensory energy.
* **structure** -- *ordered*: a smooth travelling front; *shuffled*: the same
  per-tick energy with the spatial order destroyed (the 256 patch blocks of the
  front permuted afresh every tick, so the energy is identical to the last
  bit and only the arrangement is gone).

| condition | wall | amplitude | structure | isolates |
|---|---|---|---|---|
| A | motor | small (= the bound; babble *is* the small motor member) | shuffled (three joints independent) | today's babble, **smooth** AR(1) rather than white |
| B | sensory | large | ordered | the user's wave: reach + structure |
| C | sensory | large | shuffled | amplitude without structure |
| D | sensory | small | ordered | structure without amplitude |

**Constants, derived where they can be.** The babble is `x_{t+1} = phi x_t +
sqrt(1 - phi^2) u_t`, `u ~ U(-1, 1)^3`, so its marginal is #496's uniform
babble's in RMS (`1/sqrt(3)` of the bound per joint) and it is smooth rather
than white; `phi = exp(-1 / tau)` with **`tau` = the median `world_loop(c)` of
the six L1 somatomotor cells** (`loop_length.world_loops`; 3 and 4 on this
dome, so 3.5), not `probe.py:51`'s 0.9. **The wave pattern is one stated
choice**: a plane front sweeping the tiling along `+x`, a raised-cosine bump of
half-width two patches (8 px, support four patches), advancing **one patch per
`world_loop` tick** with `world_loop` the median over the 64 L1 vision cells,
wrapping so it leaves the frame before it re-enters; uniform across rows and
channels. A second pattern is a branch of this ticket, not a fog item. Nothing
is clipped at the render's `uint8` ceiling: `Agent.write` writes floats, the
synthetic observation is an instrument, and clipping would have made the
shuffled control's energy differ from the ordered one's.

**Pre-registered reads, per column and per edge, never averaged across strata.**

* **Reach** (during the phase, at every in-phase checkpoint with `A > 0`):
  ADR-0026's paired counterfactual, rules off. Fork the run -- graph state by
  `untrained_fixed_point.snapshot`, world by `patchworks.sandbox.state`, the
  babble's own state -- and run `FORK` ticks twice, once with the supply and
  once without, everything else identical; difference the predicting cells'
  node stalks projected onto `H^0` (the reading site, #506) tick by tick, and
  read the peak against `EPS_F32 * ||state||` at that tick
  (`patchworks.tick.precision_floor`'s quantity). *Reach* at a cell is the
  peak clearing that floor. Then restore and carry on, so the fork costs the
  run nothing but wall clock.
* **Rank** (during the phase): per-edge excitation rank -- #154 s3's
  participation ratio of the edge's own stream, each end's restriction over
  the last 1,000 ticks read off the broadcast -- at the **deepest interior
  edges** (core-apex, `m_e = 3`) against `m_e`, uncentred and centred (ledger
  row 1: the uncentred form is DC-dominated, so both are published and T0's
  persistent-error reads sit beside them).
* **Priming** (post-phase, at +30k): composed rim-to-apex effective rank --
  #436/#497's instrument: one chain per sensorimotor rim cell, the graph's own
  shortest edge path to the apex, its hops composed and the composition's
  singular-value participation ratio taken -- against its shuffled control
  (B vs C, D vs A). Read at every checkpoint so its trajectory is on record.
* **Travel** (post-phase): arm travel per window under the graph's **own**
  command, T0's read.
* **Retention guard**: apex and soma `rho(K)` at +30k not below T1's frozen
  baseline at 30k beyond spread.

Usage::

    PYTHONPATH=src python prototypes/cold-start/T2/run.py --calibrate
    PYTHONPATH=src python prototypes/cold-start/T2/run.py --condition B --tb 5000 --seeds 42 43 44
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
_T0 = _HERE.parent / "T0"
_T1 = _HERE.parent / "T1"
for extra in (_ROOT / "prototypes" / "chart-double-duty-166", _ROOT / "benchmarks", _T0):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))


def _load(name: str, path: Path):
    """A sibling rig by path: every rig here is `run.py`, so a bare import is ambiguous."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


t0 = _load("t0_run", _T0 / "run.py")
t1 = _load("t1_run", _T1 / "run.py")

import construction_grading as cg  # noqa: E402
import loop_length  # noqa: E402
import untrained_fixed_point as ufp  # noqa: E402
from excitation import participation_ratio  # noqa: E402
from patchworks import agent as agent_module  # noqa: E402
from patchworks import tick as tick_module  # noqa: E402
from patchworks.agent import PIXEL_SCALE  # noqa: E402
from patchworks.graph import CellKind, EdgeKind, build_graph  # noqa: E402
from patchworks.learning import PredictionRule, TransportRule  # noqa: E402
from patchworks.restriction import pair_index  # noqa: E402
from patchworks.sandbox import PlanarPushSandbox  # noqa: E402
from patchworks.sandbox.state import restore as world_restore  # noqa: E402
from patchworks.sandbox.state import snapshot as world_snapshot  # noqa: E402

SHALLOW = t1.SHALLOW
CHECKPOINTS = t0.CHECKPOINTS
#: The post-phase horizon: the shallow dome's canonical horizon (#517's table).
POST = 30_000
#: The post-phase checkpoint ladder, relative to the phase's end.
POST_LADDER = (1_000, 2_000, 5_000, 10_000, 20_000, 30_000)
#: The paired-counterfactual fork's length, `detectability.WINDOW`: long enough
#: for a rim deviation to cross the three hops to the apex and settle.
FORK = 64
#: float32's granularity, from its one definition site (#381).
EPS_F32 = float(tick_module.EPS_F32)
#: The front's half-width in patches: the one shape choice in the wave pattern.
FRONT_HALF_WIDTH_PATCHES = 2

CONDITIONS = {
    "A": {"wall": "motor", "amplitude": "small", "structure": "shuffled"},
    "B": {"wall": "sensory", "amplitude": "large", "structure": "ordered"},
    "C": {"wall": "sensory", "amplitude": "large", "structure": "shuffled"},
    "D": {"wall": "sensory", "amplitude": "small", "structure": "ordered"},
    # **The zero-supply control, and it is not one of the ticket's four.** The
    # ticket's priming read is each ordered condition against its *shuffled*
    # control, which says whether structure bought anything but cannot separate
    # *nothing was laid down* from *all four laid down the same amount*. Z is
    # the motor wall at `A0 = 0`: T1's frozen baseline exactly -- the graph's own
    # command, the world stepped every tick, no supply at any point -- carrying
    # T2's extra reads (composed rank, per-edge rank), which T1 did not record.
    # It runs `T_b + 30k` ticks so its horizon matches the arm it controls.
    "Z": {"wall": "motor", "amplitude": "none", "structure": "none"},
}


# -- derived constants -------------------------------------------------------


def loop_constants(dome) -> dict:
    """`world_loop(c)` medians for the two L1 columns, and the cells they came from."""
    wl = loop_length.world_loops(dome).lengths
    soma = [c.id for c in dome.cells if not c.is_boundary and c.index.level == 1 and str(c.index.column) == "somatomotor"]
    vision = [c.id for c in dome.cells if not c.is_boundary and c.index.level == 1 and str(c.index.column) == "vision"]
    soma_loops = [wl[c] for c in soma]
    vision_loops = [wl[c] for c in vision]
    return {
        "tau_motor": float(np.median(soma_loops)),
        "soma_l1_world_loops": soma_loops,
        "tau_vision": float(np.median(vision_loops)),
        "vision_l1_world_loops_hist": dict(collections.Counter(vision_loops)),
        "world_tick": loop_length.WORLD_TICK,
    }


# -- the supply ----------------------------------------------------------------


class Babble:
    """Smooth torque babble: AR(1) per joint, marginal RMS `1/sqrt(3)` of the bound.

    `x_{t+1} = phi x_t + sqrt(1 - phi^2) u_t`, `u ~ U(-1, 1)`, so the stationary
    variance equals the innovation's -- #496's uniform babble in RMS, smoothed to
    a correlation time of `tau` ticks. Task-blind: drawn from a seed, reads
    nothing. `state()`/`load()` exist for the paired fork.
    """

    def __init__(self, joints: int, tau: float, seed: int) -> None:
        self.tau = float(tau)
        self.phi = float(np.exp(-1.0 / tau))
        self.innovation = float(np.sqrt(1.0 - self.phi**2))
        self.rng = np.random.default_rng([seed, 522, 1])
        self.x = np.zeros(joints, dtype=np.float64)

    def step(self) -> np.ndarray:
        self.x = self.phi * self.x + self.innovation * self.rng.uniform(-1.0, 1.0, self.x.shape)
        return self.x.copy()

    def state(self) -> tuple:
        return self.x.copy(), self.rng.bit_generator.state

    def load(self, state: tuple) -> None:
        self.x = state[0].copy()
        self.rng.bit_generator.state = state[1]

    def summary(self) -> dict:
        return {"tau": self.tau, "phi": self.phi, "innovation_scale": self.innovation, "marginal_rms_of_bound": float(1 / np.sqrt(3))}


class Front:
    """A plane front sweeping the render along `+x`, in stalk units at peak 1.

    A raised-cosine bump of half-width `half_width` px, moving `speed` px per
    tick (one patch per `tau` ticks), on a period of `size + 2 * half_width` so
    it is fully off-frame before it re-enters. Uniform over rows and channels.
    *Shuffled*: the 256 patch blocks permuted by a permutation keyed on `(seed,
    tick)`, so the per-tick energy is the ordered front's exactly and the
    read needs no state to fork.
    """

    def __init__(self, size: int, patch_side: int, tau: float, half_width_patches: int, seed: int, shuffled: bool) -> None:
        self.size, self.side = size, patch_side
        self.grid = size // patch_side
        self.tau = float(tau)
        self.speed = patch_side / tau
        self.half_width = half_width_patches * patch_side
        self.period = size + 2 * self.half_width
        self.seed, self.shuffled = seed, shuffled
        self.x = np.arange(size, dtype=np.float64)

    def profile(self, t: int) -> np.ndarray:
        pos = (t * self.speed) % self.period - self.half_width
        d = np.abs(self.x - pos)
        return np.where(d < self.half_width, 0.5 * (1.0 + np.cos(np.pi * d / self.half_width)), 0.0)

    def ordered(self, t: int) -> np.ndarray:
        g = self.profile(t)
        return np.broadcast_to(g[None, :, None], (self.size, self.size, 3)).astype(np.float32)

    def image(self, t: int) -> np.ndarray:
        img = self.ordered(t)
        if not self.shuffled:
            return img
        rng = np.random.default_rng([self.seed, 522, 2, int(t)])
        perm = rng.permutation(self.grid * self.grid)
        g, s = self.grid, self.side
        blocks = img.reshape(g, s, g, s, 3).transpose(0, 2, 1, 3, 4).reshape(g * g, s, s, 3)
        blocks = blocks[perm]
        return np.ascontiguousarray(blocks.reshape(g, g, s, s, 3).transpose(0, 2, 1, 3, 4).reshape(self.size, self.size, 3))

    def rms_at_peak_one(self) -> float:
        """Time-averaged RMS over all render components of the ordered front at peak 1."""
        ticks = int(np.ceil(self.period / self.speed))
        return float(np.sqrt(np.mean([np.mean(self.profile(t) ** 2) for t in range(ticks)])))

    def summary(self) -> dict:
        return {
            "pattern": "plane front along +x, raised cosine, uniform over rows and channels",
            "half_width_px": self.half_width,
            "speed_px_per_tick": self.speed,
            "tau_vision": self.tau,
            "period_px": self.period,
            "period_ticks": self.period / self.speed,
            "rms_at_peak_one": self.rms_at_peak_one(),
            "shuffled": self.shuffled,
        }


class Supply:
    """One condition's induced activity, with its anneal."""

    def __init__(self, condition: str, tb: int, agent, seed: int, loops: dict, calibration: dict | None) -> None:
        spec = CONDITIONS[condition]
        self.condition, self.tb = condition, tb
        self.wall, self.amplitude_class, self.structure = spec["wall"], spec["amplitude"], spec["structure"]
        self.babble = None
        self.front = None
        if self.wall == "motor":
            self.bound = float(np.max(agent.action_high))
            # The zero-supply control asks for no supply at all, not for a small
            # one: `A0 = 0` makes every `amplitude()` zero, so the command that
            # reaches `act()` is the graph's own and nothing is added to it.
            self.a0 = 0.0 if self.amplitude_class == "none" else self.bound
            self.babble = Babble(agent.joints, loops["tau_motor"], seed)
        else:
            if calibration is None:
                raise SystemExit("sensory conditions need calibration.json: run --calibrate first")
            self.bound = 1.0  # the PIXEL_SCALE range in stalk units
            self.front = Front(
                agent.env.image_size, agent.patch_side, loops["tau_vision"],
                FRONT_HALF_WIDTH_PATCHES, seed, shuffled=(self.structure == "shuffled"),
            )
            rms1 = self.front.rms_at_peak_one()
            if self.amplitude_class == "large":
                self.a0 = self.bound
            else:
                self.a0 = float(calibration["babble_render_rms"]) / rms1
            self.rms0 = self.a0 * rms1
        self.calibration = calibration

    def amplitude(self, t: int) -> float:
        if t < 0 or t >= self.tb:
            return 0.0
        return self.a0 * max(0.0, 1.0 - t / self.tb)

    def summary(self) -> dict:
        out = {
            "condition": self.condition,
            "wall": self.wall,
            "amplitude": self.amplitude_class,
            "structure": self.structure,
            "tb": self.tb,
            "anneal": "A(t) = A0 * max(0, 1 - t / T_b)",
            "bound": self.bound,
            "a0_peak": self.a0,
        }
        if self.babble is not None:
            out["babble"] = self.babble.summary()
            out["motor_command"] = "graph's own command + A(t) * babble, clipped by act(); efference = the clipped sum"
        if self.front is not None:
            out["front"] = self.front.summary()
            out["a0_rms_over_render"] = self.rms0
            out["sensory_write"] = "held render + A(t) * front (stalk units, unclipped); held qpos/qvel/touch; efference = clipped own command"
            out["calibration"] = self.calibration
        return out


# -- the recorder, extended with the broadcast ring ------------------------------


class EdgeRecorder(t0.Recorder):
    """T0's recorder plus the per-tick broadcast, so edge streams can be read."""

    def __init__(self, agent) -> None:
        super().__init__(agent)
        self.b = torch.zeros(t0.WINDOW, agent.sheaf.maps.pairs, agent.sheaf.maps.edge_width)

    @torch.no_grad()
    def observe(self) -> None:
        super().observe()
        self.b[(self.ptr - 1) % t0.WINDOW] = self.agent.sheaf.broadcast

    def broadcast_window(self) -> torch.Tensor:
        if self.filled < t0.WINDOW:
            return self.b[: self.filled]
        order = torch.arange(self.ptr, self.ptr + t0.WINDOW) % t0.WINDOW
        return self.b[order]


# -- the reads ---------------------------------------------------------------------


def edge_strata(dome) -> dict:
    """Interior edges by the levels they join, on three levels; and the drive edges."""
    names = {(1, 1): "L1-L1", (1, 2): "L1-core", (2, 2): "core-core", (2, 3): "core-apex"}
    out = {"edge": [], "m": [], "stratum": [], "kind": []}
    for e in dome.edges:
        lu, lv = dome.cells[e.u].index.level, dome.cells[e.v].index.level
        key = (min(lu, lv), max(lu, lv))
        if e.kind is EdgeKind.INTERIOR:
            stratum = names.get(key, f"L{key[0]}-L{key[1]}")
        elif e.kind is EdgeKind.DRIVE:
            stratum = "drive"
        else:
            stratum = f"{e.kind.value}-rim"
        out["edge"].append(int(e.id))
        out["m"].append(int(e.m))
        out["stratum"].append(stratum)
        out["kind"].append(e.kind.value)
    return out


@torch.no_grad()
def edge_reads(agent, recorder: EdgeRecorder, strata: dict) -> dict:
    """Per edge: excitation rank of each end's stream, disagreement energy, map effective rank."""
    b = recorder.broadcast_window().double()  # [T, pairs, w]
    ticks = b.shape[0]
    maps = agent.sheaf.maps
    edges = agent.dome.edges
    n = agent.dome.shape.n
    pairs = b.shape[1]
    # Streams as [pairs, T, w]; columns past an edge's m are already zero.
    stream = b.transpose(0, 1)
    pr_u = participation_ratio(stream).numpy()
    pr_c = participation_ratio(stream, centred=True).numpy()
    ends = b.reshape(ticks, -1, 2, b.shape[-1])
    dis = (ends[:, :, 0] - ends[:, :, 1]).pow(2).sum(-1).mean(0).numpy()  # [edges]
    energy = ends.pow(2).sum(-1).mean(0).numpy()  # [edges, 2]
    m_all = torch.tensor([e.m for e in edges])
    f = maps.maps.detach().double()[:, :, :n]  # [pairs, w, n]
    s = torch.linalg.svdvals(f)  # [pairs, w]
    map_er = (s.pow(2).sum(-1).pow(2) / s.pow(4).sum(-1).clamp(min=1e-300)).numpy()
    out = {
        "window_ticks": int(ticks),
        "pr_uncentred": [[float(pr_u[2 * e]), float(pr_u[2 * e + 1])] for e in range(len(edges))],
        "pr_centred": [[float(pr_c[2 * e]), float(pr_c[2 * e + 1])] for e in range(len(edges))],
        "disagreement_energy": [float(x) for x in dis],
        "stream_energy": [[float(x), float(y)] for x, y in energy],
        "map_effective_rank": [[float(map_er[2 * e]), float(map_er[2 * e + 1])] for e in range(len(edges))],
    }
    # Per-stratum medians, both ends pooled, so the printout has a number.
    by = collections.defaultdict(list)
    for e, st in enumerate(strata["stratum"]):
        by[st].append(e)
    summary = {}
    for st, idx in by.items():
        pu = np.array([out["pr_uncentred"][e] for e in idx]).ravel()
        pc = np.array([out["pr_centred"][e] for e in idx]).ravel()
        mer = np.array([out["map_effective_rank"][e] for e in idx]).ravel()
        summary[st] = {
            "edges": len(idx),
            "m": int(m_all[idx[0]]),
            "pr_uncentred_median": float(np.median(pu)),
            "pr_centred_median": float(np.median(pc)),
            "pr_centred_max": float(pc.max()),
            "disagreement_energy_median": float(np.median(dis[idx])),
            "map_effective_rank_median": float(np.median(mer)),
        }
    out["by_stratum"] = summary
    return out


def rim_chains(dome) -> list[dict]:
    """One chain per sensorimotor rim cell: the shortest edge path to an apex cell (#497's construction)."""
    apex = set(t1_apex(dome))
    rim = [c for c in dome.cells if c.is_boundary and c.kind is not CellKind.DRIVE]
    chains = []
    for cell in rim:
        # Breadth-first over edges, never through a drive edge.
        prev: dict[int, tuple[int, int] | None] = {cell.id: None}
        queue = collections.deque([cell.id])
        found = None
        while queue and found is None:
            here = queue.popleft()
            for edge_id in dome.incident[here]:
                edge = dome.edges[edge_id]
                if edge.kind is EdgeKind.DRIVE:
                    continue
                other = edge.other(here)
                if other in prev:
                    continue
                prev[other] = (here, edge_id)
                if other in apex:
                    found = other
                    break
                queue.append(other)
        if found is None:
            continue
        path, at = [], found
        while prev[at] is not None:
            here, edge_id = prev[at]
            path.append(edge_id)
            at = here
        path.reverse()
        chains.append({"rim": int(cell.id), "kind": cell.kind.value, "apex": int(found), "edges": [int(e) for e in path]})
    return chains


def t1_apex(dome) -> tuple[int, ...]:
    deepest = max(c.index.level for c in dome.cells if not c.is_boundary)
    return tuple(c.id for c in dome.cells if not c.is_boundary and c.index.level == deepest)


@torch.no_grad()
def hop_operator(dome, maps, key) -> torch.Tensor:
    """`F_out . F_in^T` in float64 (`holonomy_read.hop_operator`, verbatim)."""
    edge_in, cell, edge_out = key
    m_in, m_out = dome.edges[edge_in].m, dome.edges[edge_out].m
    f_in = maps.maps[pair_index(edge_in, cg.side_of(dome, edge_in, cell))][:m_in]
    f_out = maps.maps[pair_index(edge_out, cg.side_of(dome, edge_out, cell))][:m_out]
    return f_out.double() @ f_in.double().T


@torch.no_grad()
def composed_reads(agent, chains: list[dict]) -> dict:
    """Composed rim-to-apex transport per chain: effective rank and sigma_max."""
    dome, maps = agent.dome, agent.sheaf.maps
    er, smax, kinds = [], [], []
    for chain in chains:
        hops = cg.hops_of(dome, tuple(chain["edges"]))
        composed = hop_operator(dome, maps, hops[0])
        for key in hops[1:]:
            composed = hop_operator(dome, maps, key) @ composed
        s = torch.linalg.svdvals(composed)
        er.append(float(s.pow(2).sum() ** 2 / s.pow(4).sum().clamp(min=1e-300)))
        smax.append(float(s[0]))
        kinds.append(chain["kind"])
    er_a, smax_a = np.array(er), np.array(smax)
    by_kind = {}
    for kind in sorted(set(kinds)):
        idx = [i for i, k in enumerate(kinds) if k == kind]
        by_kind[kind] = {
            "chains": len(idx),
            "effective_rank_median": float(np.median(er_a[idx])),
            "effective_rank_p10": float(np.quantile(er_a[idx], 0.10)),
            "effective_rank_p90": float(np.quantile(er_a[idx], 0.90)),
            "sigma_max_median": float(np.median(smax_a[idx])),
        }
    return {"effective_rank": [float(x) for x in er_a], "sigma_max": [float(x) for x in smax_a], "by_kind": by_kind}


# -- the phase --------------------------------------------------------------------


def arranged(env, agent, seed: int) -> dict:
    """The world arranged (`run()`'s reset), retried past a blocked spawn annulus."""
    last = None
    for attempt in range(6):
        try:
            obs, _info = env.reset(seed=seed + 1000 * attempt)
            agent.observe(obs)
            return obs
        except Exception as exc:  # BlockedAnnulusError: the babbling arm sits in the annulus
            last = exc
    raise RuntimeError(f"could not arrange the world after 6 draws: {last}")


def _write_sensory(agent, held: dict, held_image: np.ndarray, front: Front, a: float, t: int) -> None:
    image = held_image if a == 0.0 else held_image + (a / PIXEL_SCALE) * front.image(t)
    applied = np.clip(agent.command(), agent.action_low, agent.action_high)
    agent.write({"image": image, "qpos": held["qpos"], "qvel": held["qvel"], "touch": held["touch"]}, applied)


def induced_ticks(agent, ticks: int, supply: Supply, held: dict, held_image: np.ndarray, recorder, bias, transport):
    """`ticks` whole ticks under the supply, both rules on; yields each tick's pose."""
    sheaf = agent.sheaf
    for _ in range(ticks):
        t = sheaf.ticks
        a = supply.amplitude(t)
        sheaf.tick()
        if supply.wall == "motor":
            x = supply.babble.step()
            outcome = agent.act((agent.command() + a * x).astype(np.float32))
            pose = outcome.observation["qpos"].copy()
        else:
            _write_sensory(agent, held, held_image, supply.front, a, t)
            pose = np.asarray(held["qpos"], dtype=np.float32).copy()
        sheaf.assert_no_tape()
        recorder.observe()
        bias.step()
        if sheaf.ticks > 1:
            transport.step()
        yield pose


@torch.no_grad()
def reach_fork(agent, env, supply: Supply, held: dict | None, held_image: np.ndarray | None, context: dict, groups: dict) -> dict:
    """ADR-0026's paired counterfactual on the live supply, rules off, then restored."""
    sheaf = agent.sheaf
    state = ufp.snapshot(sheaf)
    motor = supply.wall == "motor"
    world = world_snapshot(env) if motor else None
    babble = supply.babble.state() if motor else None
    t_start = sheaf.ticks
    a_start = supply.amplitude(t_start)

    def branch(perturbed: bool) -> torch.Tensor:
        ufp.restore(sheaf, state)
        if motor:
            world_restore(env, world)
            supply.babble.load(babble)
        kept = []
        for _ in range(FORK):
            t = sheaf.ticks
            a = supply.amplitude(t) if perturbed else 0.0
            sheaf.tick()
            if motor:
                x = supply.babble.step()
                agent.act((agent.command() + a * x).astype(np.float32))
            else:
                _write_sensory(agent, held, held_image, supply.front, a, t)
            kept.append(sheaf.evidence().clone())
        return torch.stack(kept).double()

    quiet = branch(False)
    moved = branch(True)
    ufp.restore(sheaf, state)
    if motor:
        world_restore(env, world)
        supply.babble.load(babble)
    if sheaf.ticks != t_start:
        raise RuntimeError("fork did not restore the tick count")

    proj = agent.dome.private_projection.double()
    dev = moved - quiet
    dev_priv = (dev * proj).norm(dim=-1)  # [FORK, cells]
    dev_full = dev.norm(dim=-1)
    floor = EPS_F32 * quiet.norm(dim=-1)  # [FORK, cells]
    cells = dev.shape[1]
    idx = torch.arange(cells)
    peak_at = dev_priv.argmax(0)
    peak_priv = dev_priv[peak_at, idx]
    floor_at_peak = floor[peak_at, idx]
    ratio_priv = peak_priv / floor_at_peak.clamp(min=1e-300)
    peak_full_at = dev_full.argmax(0)
    ratio_full = dev_full[peak_full_at, idx] / floor[peak_full_at, idx].clamp(min=1e-300)
    # Also the value at the end of the fork, so "stays above" has a late reading beside the peak.
    ratio_priv_end = dev_priv[-1] / floor[-1].clamp(min=1e-300)
    private_dim = agent.dome.private_dimensions
    per_class = {}
    for cls, rows in groups.items():
        r = ratio_priv[rows]
        per_class[cls] = {
            "cells": len(rows),
            "ratio_private_median": float(r.median()),
            "ratio_private_min": float(r.min()),
            "ratio_private_max": float(r.max()),
            "above_floor_fraction": float((r > 1.0).double().mean()),
            "ratio_full_median": float(ratio_full[rows].median()),
            "ratio_private_end_median": float(ratio_priv_end[rows].median()),
            "peak_private_median": float(peak_priv[rows].median()),
            "floor_median": float(floor_at_peak[rows].median()),
            "peak_tick_median": float(peak_at[rows].double().median()),
        }
    apex_rows = groups["apex"]
    return {
        "fork_ticks": FORK,
        "amplitude_at_fork": a_start,
        "tick_at_fork": int(t_start),
        "per_cell": {
            "peak_private": [float(x) for x in peak_priv],
            "peak_tick": [int(x) for x in peak_at],
            "floor_at_peak": [float(x) for x in floor_at_peak],
            "ratio_private": [float(x) for x in ratio_priv],
            "ratio_full": [float(x) for x in ratio_full],
            "ratio_private_end": [float(x) for x in ratio_priv_end],
            "private_dimension": [int(x) for x in private_dim],
        },
        "by_class": per_class,
        "apex_trajectory_median_ratio": [float(x) for x in (dev_priv[:, apex_rows] / floor[:, apex_rows].clamp(min=1e-300)).median(dim=1).values],
    }


# -- calibration --------------------------------------------------------------------


def calibrate(seed: int, ticks: int, out: Path) -> dict:
    """The render's deviation under motor babble at the bound: the sensory *babble scale*."""
    dome = build_graph(SHALLOW)
    loops = loop_constants(dome)
    env = PlanarPushSandbox(split="train", image_size=ufp.IMAGE_SIZE[SHALLOW.patch_grid])
    try:
        obs, _ = env.reset(seed=seed)
        babble = Babble(env.action_space.shape[0], loops["tau_motor"], seed)
        frames, poses, applied = [], [obs["qpos"].copy()], []
        for _ in range(ticks):
            a = np.clip(babble.step(), -1.0, 1.0)
            obs, _r, _t, _tr, _i = env.step(a.astype(np.float32))
            frames.append(obs["image"].astype(np.float32) * PIXEL_SCALE)
            poses.append(obs["qpos"].copy())
            applied.append(a)
        f = np.stack(frames)
        dev = f - f.mean(0, keepdims=True)
        change = np.diff(f, axis=0)
        poses = np.asarray(poses)
        front = Front(env.image_size, env.image_size // SHALLOW.patch_grid, loops["tau_vision"], FRONT_HALF_WIDTH_PATCHES, seed, shuffled=False)
        rms1 = front.rms_at_peak_one()
        record = {
            "issue": 522,
            "what": "RMS deviation of the render (stalk units, PIXEL_SCALE applied) from its time mean under smooth motor babble at the bound; the sensory wall's small amplitude is set so the front's time-averaged RMS over the render equals it",
            "seed": seed,
            "ticks": ticks,
            "loops": loops,
            "babble": babble.summary(),
            "babble_render_rms": float(np.sqrt(np.mean(dev**2))),
            "babble_render_change_rms": float(np.sqrt(np.mean(change**2))),
            "render_mean": float(f.mean()),
            "render_std_over_components": float(f.mean(0).std()),
            "fraction_components_moving": float((dev.std(0) > 1e-3).mean()),
            "arm_travel_per_tick": float(np.abs(np.diff(poses, axis=0)).sum() / ticks),
            "applied_rms": float(np.sqrt(np.mean(np.square(applied)))),
            "front_rms_at_peak_one": rms1,
            "small_peak_derived": float(np.sqrt(np.mean(dev**2)) / rms1),
            "surface": t0.surface(),
        }
        out.write_text(json.dumps(record, indent=1))
        return record
    finally:
        env.close()


# -- one run ---------------------------------------------------------------------------


def run_seed(condition: str, tb: int, seed: int, post: int, out: Path, calibration: dict | None, fork_every: bool = True) -> dict:
    started = time.time()
    inflight = t0.stage(out)
    npz_path = out.with_suffix(".npz")
    env, agent = t1.build_shallow(seed)
    try:
        loops = loop_constants(agent.dome)
        supply = Supply(condition, tb, agent, seed, loops, calibration)
        context = t0.cell_context(agent)
        grp = t1.groups(context)
        strata = edge_strata(agent.dome)
        chains = rim_chains(agent.dome)
        record = {
            "issue": 522,
            "condition": condition,
            "condition_spec": CONDITIONS[condition],
            "tb": tb,
            "post": post,
            "build": {"rho1_drive_edges": False, "c": 1.0, "note": "the baseline build; T1's winner is T2b's"},
            "dome": "shallow",
            "dome_spec": {
                "vision_sides": list(SHALLOW.vision_sides),
                "somatomotor_sizes": list(SHALLOW.somatomotor_sizes),
                "core_sizes": list(SHALLOW.core_sizes),
                "core_degree": SHALLOW.core_degree,
                "apex_degree": SHALLOW.apex_degree,
                "interior_m": SHALLOW.interior_m,
                "boundary_m": SHALLOW.boundary_m,
                "cells": len(agent.dome.cells),
                "edges": len(agent.dome.edges),
            },
            "split": "train",
            "seed": seed,
            "ticks": tb + post,
            "surface": t0.surface(),
            "cells": int(agent.sheaf.operators.cells),
            "k": int(agent.sheaf.operators.shape.k),
            "n": int(agent.dome.shape.n),
            "band": [1.0 / agent.sheaf.operators.rho_k, 1.0],
            "drive_assertion": float(agent_module.DRIVE_ASSERTION),
            "supply": supply.summary(),
            "loops": loops,
            "fork": FORK,
            "eps_f32": EPS_F32,
            "window": t0.WINDOW,
            "block": t0.BLOCK,
            "context": context,
            "groups": grp,
            "edge_strata": strata,
            "chains": chains,
            "checkpoints": [],
        }
        recorder = EdgeRecorder(agent)
        bias = PredictionRule(agent.sheaf)
        transport = TransportRule(agent.sheaf)
        record["operator_rate"] = float(bias.operator_rate)
        record["learning_rate"] = float(bias.learning_rate)
        arrays: dict[str, np.ndarray] = {}
        # The untrained surface's composed rank, the floor every later reading is above or not.
        record["composed_at_construction"] = composed_reads(agent, chains)["by_kind"]

        phase_ladder = [c for c in CHECKPOINTS if c < tb] + [tb]
        post_ladder = [tb + c for c in POST_LADDER if c < post] + [tb + post]
        seen = 0
        carry = None
        travel_total = 0.0
        held = None
        held_image = None
        for target in phase_ladder + post_ladder:
            window = target - seen
            in_phase = target <= tb
            poses = [] if carry is None else [carry]
            if in_phase:
                if supply.wall == "motor" or held is None:
                    held = arranged(env, agent, seed + seen)
                    held_image = held["image"].astype(np.float32)
                for pose in induced_ticks(agent, window, supply, held, held_image, recorder, bias, transport):
                    poses.append(pose)
            else:
                for outcome in t0.teaching_read(agent, window, seed + seen, recorder, bias, transport):
                    poses.append(outcome.observation["qpos"].copy())
            pose = np.asarray(poses)
            window_travel = float(np.abs(np.diff(pose, axis=0)).sum()) if len(pose) > 1 else 0.0
            travel_total += window_travel
            carry = poses[-1]
            seen = target
            if agent.sheaf.ticks != target:
                raise RuntimeError(f"tick count {agent.sheaf.ticks} != {target}")

            entry, arr = t0.checkpoint_read(agent, recorder, bias, context)
            entry["ticks"] = target
            entry["phase"] = "induced" if in_phase else "post"
            entry["since_phase_start"] = target if in_phase else target - tb
            entry["amplitude"] = supply.amplitude(target - 1)
            entry["travel_window"] = window_travel
            entry["travel_cumulative"] = travel_total
            entry["travel_per_tick"] = window_travel / max(1, window)
            entry["edges"] = edge_reads(agent, recorder, strata)
            entry["composed"] = composed_reads(agent, chains)
            if in_phase and fork_every and supply.amplitude(target) > 0.0:
                entry["reach"] = reach_fork(agent, env, supply, held, held_image, context, grp)
            else:
                entry["reach"] = None
            record["checkpoints"].append(entry)
            record["elapsed_minutes"] = (time.time() - started) / 60.0
            for key, val in arr.items():
                arrays[f"t{target}_{key}"] = val
            arrays[f"t{target}_composed_er"] = np.array(entry["composed"]["effective_rank"])
            record["blocks"] = [
                {"end_tick": b["end_tick"], **{key: [float(x) for x in b[key]] for key in b if key not in ("end_tick", "mean_h", "mean_e")}}
                for b in recorder.blocks
            ]
            if recorder.blocks:
                arrays["block_end_ticks"] = np.array([b["end_tick"] for b in recorder.blocks])
                arrays["block_mean_e"] = np.stack([b["mean_e"].numpy() for b in recorder.blocks])
                arrays["block_mean_h"] = np.stack([b["mean_h"].numpy() for b in recorder.blocks])
            inflight.write_text(json.dumps(record, indent=1))
            np.savez_compressed(npz_path, **arrays)

            pc = entry["per_cell"]
            rho = np.array(pc["rho_used"])
            ca = entry["edges"]["by_stratum"].get("core-apex", {})
            comp = entry["composed"]["by_kind"].get("patch", {})
            reach = entry["reach"]
            reach_s = f"reach apex {reach['by_class']['apex']['ratio_private_median']:.2g}x floor (A={reach['amplitude_at_fork']:.3g}) | " if reach else ""
            print(
                f"  [{condition} tb{tb}] seed {seed} @ {target:>6} ({entry['phase']}): "
                f"rho apex {np.median(rho[grp['apex']]):.3f} core {np.median(rho[grp['core']]):.3f} "
                f"vis {np.median(rho[grp['vision']]):.3f} soma {np.median(rho[grp['soma']]):.3f} | "
                f"|e| apex {np.median(np.array(pc['p3_ebar_norm'])[grp['apex']]):.2e} "
                f"stab {np.nanmedian(np.array(pc['p3_direction_stability'])[grp['apex']]):.2f} | "
                f"core-apex PR u {ca.get('pr_uncentred_median', float('nan')):.2f} c {ca.get('pr_centred_median', float('nan')):.2f} | "
                f"composed ER patch {comp.get('effective_rank_median', float('nan')):.3f} | "
                f"{reach_s}"
                f"travel/tick {entry['travel_per_tick']:.2e} | "
                f"dead {int((np.asarray(entry['used']['per_cell']['modes_retaining']) == 0).sum())} "
                f"({record['elapsed_minutes']:.1f} min, {target / max(1e-9, time.time() - started):.1f} t/s)",
                flush=True,
            )
        inflight.replace(out)
        return record
    finally:
        env.close()


def run_name(condition: str, tb: int, seed: int, post: int, tag: str = "") -> str:
    tag = f"-{tag}" if tag else ""
    return f"522-{condition}-tb{tb}{tag}-seed{seed}-post{post}.json"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--calibrate", action="store_true", help="derive the sensory small amplitude and write calibration.json")
    p.add_argument("--condition", choices=tuple(CONDITIONS), help="A, B, C or D")
    p.add_argument("--tb", type=int, default=5_000, help="induced phase length T_b")
    p.add_argument("--post", type=int, default=POST, help="post-phase ticks")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--calibration-ticks", type=int, default=2_000)
    p.add_argument("--tag", default="")
    p.add_argument("--no-fork", action="store_true", help="skip the reach fork (smoke tests)")
    args = p.parse_args()
    cal_path = _HERE / "calibration.json"
    if args.calibrate:
        rec = calibrate(args.seeds[0], args.calibration_ticks, cal_path)
        print(json.dumps({k: v for k, v in rec.items() if k not in ("surface",)}, indent=1))
        return
    if args.condition is None:
        raise SystemExit("--condition is required unless --calibrate")
    calibration = json.loads(cal_path.read_text()) if cal_path.exists() else None
    for seed in args.seeds:
        out = _HERE / run_name(args.condition, args.tb, seed, args.post, args.tag)
        print(f"[T2] condition {args.condition} T_b {args.tb} seed {seed}, {args.tb + args.post} ticks -> {out.name}", flush=True)
        run_seed(args.condition, args.tb, seed, args.post, out, calibration, fork_every=not args.no_fork)
        print(f"  wrote {out.name}", flush=True)


if __name__ == "__main__":
    main()
