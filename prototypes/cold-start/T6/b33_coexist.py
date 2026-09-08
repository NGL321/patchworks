"""B33 (#592): can a representation floor and a local holonomy term coexist?

**A cheap, rough prototype to react to — not an implementation.** #592 is a
`wayfinder:prototype` ticket on [#532](https://github.com/NGL321/patchworks/issues/532),
whose standing note is *plan, don't do*: what follows is built to make the
argument concrete enough to rule on, and every surrogate below is named as one.

## The question, and the trap it is built around

[B27](https://github.com/NGL321/patchworks/issues/576) widened #532 to cover
*transport and the rule that trains it*. Two terms follow and neither is
optional: a **floor on the representation** ([B19](https://github.com/NGL321/patchworks/issues/568):
learned maps read `corr(state, emitted) = -0.05` where a random map of identical
shape, mask and band reads **+0.97**) and a **local holonomy term**
([B22](https://github.com/NGL321/patchworks/issues/571): the mask closes at
construction, so nothing in the current rule can move coherent structure).

They are natural adversaries. **Flatness is free at width one** —
[B29](https://github.com/NGL321/patchworks/issues/585) found 250 of 260 basis
cycles rank-1 before a map is drawn, and on those free cycles the share
returning positive climbs from exactly 0.500 at construction to 0.69-0.81 by
2,000 ticks *unasked*. So the reading is **not** *does flatness rise*; it is
**does flatness rise on cycles carrying more than one direction, with the floor
holding**. `surface_read`'s `wide` subset (base width >= 2) is that population
and it is the one this module reports on.

## What B32 changed about the design, and why it is measured here

[B32](https://github.com/NGL321/patchworks/issues/591)'s literature pass landed
after #592 was written and indicted the ticket's own guard:

* **The variance floor alone is not the floor.** VICReg (Bardes, Ponce & LeCun,
  ICLR 2022) pairs a per-dimension variance term with a **decorrelation** term,
  and a variance floor by itself *"is satisfiable by duplicating one direction
  across all dimensions"* — which, given B29's 250-of-260 rank-1 cycles, is the
  live failure rather than a hypothetical. So the floor is run **both ways**
  here (`floor` and `floor_dec`) and the difference is the reading.
* **A path-independence penalty has a published degenerate optimum.** Chu,
  Zhmoginov & Sandler (NIPS 2017 Workshop) show CycleGAN satisfies cycle
  consistency by hiding a low-amplitude high-frequency side-channel. B32's
  words: *"a width floor does not close it; the steganographic channel has
  width."* `channel_return` on wide cycles beside the floor is what would catch
  it here, and `amplitude` is reported so a term satisfied at vanishing scale is
  visible rather than scored as a win.

## Three costs this prototype pays, stated rather than smuggled

1. **ADR-0031 — the transport rule holds no state.** Its words: *"no momentum,
   no running average, no baseline, no estimate of any edge's recent scale."*
   A variance floor is a statistic over samples; one tick is one sample and has
   no variance. `RepresentationFloor` therefore keeps a **detached window** of
   the last `WINDOW` emitted vectors per pair, with gradient flowing through the
   current sample only. That is a per-tick surrogate for VICReg's batch
   statistic (which B32 records the paper does not supply) and it is **state the
   rule is currently forbidden to hold**. Named, not hidden: it is the first
   thing the map has to price.
2. **ADR-0011 — a cell may only ask about its own neighbours.** #396 files the
   objection against itself: *"A cycle is not incident to one cell."*
   :func:`local_cycles` keeps only cycles lying wholly inside some cell's
   closed `RADIUS`-hop neighbourhood, so every retained cycle is visible to at
   least one cell. That is the satisfiable form and it is still weaker than
   *one cell computes it alone*.
3. **The training term is a surrogate for the measured column.**
   `identification` is read off an SVD polar factor; differentiating an SVD near
   a degenerate spectrum is not something a rough prototype should do. The term
   descended is the scale-normalised Frobenius form
   `|| sqrt(m) * H/||H||_F - I ||_F^2`, which is scale-invariant and has a stable
   gradient. The **measurement** is unchanged from B29's.

## What is read, and what is refused

Per #592: `benchmarks/detectability.py` (rim-to-core detectability, the bar) and
the conduction ratio on the **world loop**; holonomy via B29's instrument;
`corr(state, emitted)` per B19 against its Haar control. **Composed effective
rank is retired as a bar** ([B17](https://github.com/NGL321/patchworks/issues/565):
`composed_reads` never touches the stalks) and appears only as a diagnostic.

`p = 12` throughout, per B27.

**The honest horizon on this arm is ~150 ticks, not 2,000, and that is a
correction to the record.** #572 and #518 put the sandbox's stall between 1,000
and 2,000 ticks on `reserve`/`shipped`. `b33_motion.py` traced `reserve_p12`
seed 42 directly and it stalls **between ticks 100 and 150** — `std_max` 1.287 at
50 and 1.268 at 100, then 6.9e-04 at 150, and exactly zero moving components by
600. So the live rungs are 50, 100 and 150; everything past them is a reading
against a motionless world and is marked `past_stall` in the record.
:class:`Motion` stamps the world's own state at every checkpoint, so no number
here has to be taken on trust about which side of the stall it sits.

Usage::

    PYTHONPATH=src python prototypes/cold-start/T6/b33_coexist.py run
    PYTHONPATH=src python prototypes/cold-start/T6/b33_coexist.py run --arms baseline both
"""

from __future__ import annotations

import argparse
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
_T0, _T1, _T2, _T3, _T4, _T5 = (
    _HERE.parent / n for n in ("T0", "T1", "T2", "T3", "T4", "T5")
)
for extra in (
    _ROOT / "prototypes" / "chart-double-duty-166",
    _ROOT / "benchmarks",
    _T0,
):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


t0 = _load("t0_run", _T0 / "run.py")
t2 = _load("t2_run", _T2 / "run.py")
arms_mod = _load("t6_arms", _HERE / "arms.py")
b29 = _load("t6_b29", _HERE / "b29_holonomy.py")
b19 = _load("t6_b19", _HERE / "b19_state_emitted.py")

import construction_grading as cg  # noqa: E402
import holonomy_read as hr  # noqa: E402
from patchworks.learning import PredictionRule, TransportRule  # noqa: E402
from patchworks.restriction import pair_index  # noqa: E402

#: B27's ruling. `p = 16` and `p = 24` are off the live candidates.
ARM = "reserve_p12"

#: The horizon, and the rungs that are actually *live*.
#:
#: The record says the sandbox stalls between 1,000 and 2,000 ticks (#572, #518).
#: **On this arm it does not.** `b33_motion.py` traced `reserve_p12` seed 42 in
#: disjoint 50-tick windows: `std_max` runs 1.287 at tick 50 and 1.268 at 100,
#: then **6.9e-04 at 150** — a fall of ~1,800x — and reaches exactly zero moving
#: components by 600. So the honest horizon here is **~100-150 ticks**, and every
#: rung past it is a reading against a motionless world.
#:
#: The early rungs are therefore not a convenience, they are the only live ones;
#: the long rungs are kept because every earlier ticket on this map published at
#: them and dropping them would make this arm incomparable.
TICKS = 2000
CHECKPOINTS = (50, 100, 150, 250, 500, 1000, 2000)
#: Past this, the world is not moving on this arm. Stamped, not assumed.
LIVE_HORIZON = 150

#: The read window, in ticks, for the floor and the motion stamp.
#:
#: **Not B19's.** `b19_state_emitted.WINDOW` is 1,000 ticks — longer than this
#: arm's entire live period — so reading through it would average a moving world
#: with a frozen one and report the mixture as a trained result. 100 ticks is
#: short enough that the first rung is wholly live and long enough to estimate a
#: participation ratio over a stalk of this width. The realised length is
#: recorded per checkpoint rather than assumed.
READ_WINDOW = 100

#: ADR-0011's radius. Q2's conduction bound (20-50x per hop, reach 2-3 hops) and
#: the locality bound land on the same number, which is why this is coherent
#: rather than a convenient restriction.
RADIUS = 2

#: The floor's window. State ADR-0031 forbids — see the module docstring.
WINDOW = 32

#: VICReg's variance target, on the emitted stream. Not tuned; stated.
GAMMA = 1.0

#: Term weights, applied to **unit-normalised** gradients (see `extra_step`), so
#: these are a stated ratio rather than an accident of which term is louder.
#: Equal by choice: the ticket asks whether the two terms are adversaries, and
#: the fair form of that question gives them the same budget. Unswept.
LAMBDA_HOLO = 0.05
LAMBDA_VAR = 0.05
#: Weight of VICReg's decorrelation term *within* the absolute floor's loss.
LAMBDA_DEC = 0.05


# -- the local cycles ADR-0011 admits -----------------------------------------


def _cell_neighbourhood(dome, radius: int) -> dict[int, set[int]]:
    """Closed `radius`-hop neighbourhood of every interior cell, by cell id."""
    adj: dict[int, set[int]] = {}
    for edge in dome.edges:
        adj.setdefault(edge.u, set()).add(edge.v)
        adj.setdefault(edge.v, set()).add(edge.u)
    out = {}
    for cell in adj:
        seen = {cell}
        frontier = {cell}
        for _ in range(radius):
            nxt = set()
            for c in frontier:
                nxt |= adj.get(c, set())
            nxt -= seen
            seen |= nxt
            frontier = nxt
        out[cell] = seen
    return out


def local_cycles(dome, cycles, radius: int = RADIUS) -> list[tuple[int, ...]]:
    """Cycles lying wholly inside some cell's closed `radius`-hop neighbourhood.

    **This is what keeps ADR-0011 satisfiable**, and it is weaker than the ADR's
    own bar rather than equal to it: it establishes that *some* cell can see the
    whole cycle, not that the cell whose map is being moved can. #396's
    objection — *"a cycle is not incident to one cell"* — is answered to the
    extent a short local cycle can answer it, and no further. Said plainly
    because the whole holonomy term rests on it.
    """
    hoods = _cell_neighbourhood(dome, radius)
    kept = []
    for cycle in cycles:
        cells = set()
        for edge_id in cycle:
            cells.add(dome.edges[edge_id].u)
            cells.add(dome.edges[edge_id].v)
        if any(cells <= hood for hood in hoods.values()):
            kept.append(tuple(cycle))
    return kept


# -- the holonomy term --------------------------------------------------------


def holonomy_of(dome, tensor: torch.Tensor, cycle) -> torch.Tensor:
    """`hr.holonomy`, differentiable: the same composition with the grad kept.

    `hr.hop_operator` takes its blocks under `no_grad` because it is a reading.
    This is the identical product on the identical blocks with that guard
    removed, so the term descended and the column measured are the same object
    up to the surrogate named in the module docstring.
    """
    hops = hr.cycle_hops(dome, cycle)
    composed = None
    for edge_in, cell, edge_out in hops:
        m_in = dome.edges[edge_in].m
        m_out = dome.edges[edge_out].m
        f_in = tensor[pair_index(edge_in, cg.side_of(dome, edge_in, cell))][:m_in]
        f_out = tensor[pair_index(edge_out, cg.side_of(dome, edge_out, cell))][:m_out]
        hop = f_out @ f_in.T
        composed = hop if composed is None else hop @ composed
    return composed


def holonomy_loss(dome, tensor: torch.Tensor, cycles) -> torch.Tensor:
    """Mean scale-normalised departure from `I` over the admitted local cycles.

    `|| sqrt(m) * H/||H||_F - I ||_F^2 / (2m)`. Scale-invariant, so a cycle
    cannot buy the term by shrinking — which matters because ADR-0032's band
    forces near-isometries and shrinking to zero is meant to be unavailable.
    The normalisation makes that explicit rather than relying on the band.
    """
    total = tensor.new_zeros(())
    count = 0
    for cycle in cycles:
        h = holonomy_of(dome, tensor, cycle)
        m = h.shape[0]
        norm = torch.linalg.norm(h)
        if not bool(torch.isfinite(norm.detach())) or float(norm.detach()) <= 0.0:
            continue
        scaled = h * (np.sqrt(m) / norm)
        eye = torch.eye(m, dtype=h.dtype, device=h.device)
        total = total + (scaled - eye).pow(2).sum() / (2 * m)
        count += 1
    return total / max(count, 1)


# -- the representation floor -------------------------------------------------


class RepresentationFloor:
    """VICReg's two terms on the emitted stream, per edge endpoint.

    **The floor is on the representation, never on the map.** ADR-0032's band is
    a floor on each map and B19 is the demonstration that it does not buy this:
    a random map wearing that exact band reads `corr(state, emitted) = +0.97`
    where the learned maps read `-0.05`. What is floored here is `F h_v` — what
    the cell actually says — and that is the distinction #592 calls the whole
    point.

    Holds `WINDOW` detached emitted vectors per pair. That is the ADR-0031 cost
    in one sentence.
    """

    def __init__(self, sheaf, dome, *, decorrelate: bool, scale_free: bool = False) -> None:
        self.sheaf = sheaf
        self.decorrelate = decorrelate
        self.scale_free = scale_free
        self.buffer: list[torch.Tensor] = []
        # **The lane mask, and it is load-bearing.** `maps.maps` is
        # `[pairs, m_max, stalk]` with `m_max` the widest lane in the graph, so
        # an edge of width 1 carries 19 padded rows the structural mask holds at
        # zero. Floored naively, `relu(gamma - 0)**2 = 1` fires on every one of
        # them: the first run of this prototype spent ~98% of the term's budget
        # on slots that cannot move (loss pinned at 0.049, gradient 7e-05, three
        # orders below the holonomy term) and the floor arm came out bit-identical
        # to baseline. The floor is a statement about what a cell *says*, and a
        # padded row is not something it says.
        rows = torch.zeros(sheaf.maps.pairs, sheaf.maps.edge_width, dtype=torch.bool)
        for edge in dome.edges:
            for side in (0, 1):
                rows[pair_index(edge.id, side), : edge.m] = True
        self.lanes = rows
        self.live = int(rows.sum())

    def emitted(self, tensor: torch.Tensor) -> torch.Tensor:
        """`[pairs, m_max]` — each endpoint's own stalk restricted onto its lane."""
        sheaf = self.sheaf
        own = sheaf.stalks[sheaf.layout.pair_positions].detach()
        return torch.einsum("pms,ps->pm", tensor, own)

    def observe(self, tensor: torch.Tensor) -> None:
        with torch.no_grad():
            self.buffer.append(self.emitted(tensor).clone())
        if len(self.buffer) > WINDOW:
            self.buffer.pop(0)

    def _participation(self, centred: torch.Tensor) -> torch.Tensor:
        """The scale-free floor: how many directions the edge actually uses.

        **Why the absolute floor cannot be the floor, measured rather than
        argued.** On `reserve_p12` seed 42 the live-lane emitted per-dimension
        std is median **0.019** (p95 0.104) and only **0.32%** of live lanes
        reach `GAMMA = 1`, while ADR-0032's band pins the maps' singular values
        at median 0.302. The band is what forbids buying variance with gain, so
        the *only* move left is a rotation of `F` — and a rotation cannot raise
        an absolute variance whose scale is set by a stalk that barely
        fluctuates (B18: the mean carries it). The absolute term is therefore
        maximally violated and nearly inert at once: `floor_loss` pinned at
        ~0.048 with a gradient two orders below the holonomy term's.

        So the floor is stated on the **participation ratio** of the emitted
        covariance, `(tr C)^2 / ||C||_F^2` — the number of directions carrying
        comparable energy. It is invariant to the overall scale the band fixes,
        it is exactly what a rotation *can* move, and it subsumes VICReg's
        decorrelation term rather than needing it alongside: a duplicated
        direction and a correlated pair both read as participation 1. That is
        B32's correction to #592's guard, in the one form this architecture can
        actually enforce.
        """
        total = centred.new_zeros(())
        count = 0
        for pair in range(centred.shape[1]):
            keep = self.lanes[pair]
            m = int(keep.sum())
            if m < 2:
                continue  # a width-1 lane has participation 1 by construction
            x = centred[:, pair][:, keep]
            cov = (x.T @ x) / x.shape[0]
            trace = torch.diagonal(cov).sum()
            frob = cov.pow(2).sum()
            if float(frob.detach()) <= 1e-20:
                continue
            er = trace.pow(2) / (frob + 1e-20)
            # Floor at half the lane's directions. Rough, stated, unswept.
            total = total + torch.relu(0.5 * m - er).pow(2) / (m * m)
            count += 1
        return total / max(count, 1)

    def loss(self, tensor: torch.Tensor) -> torch.Tensor:
        """Variance floor, plus decorrelation when asked. Zero until the window fills.

        Gradient flows through the **current** sample only; the window is
        detached history. That is the per-tick surrogate for a batch statistic,
        and B32 records that VICReg supplies none.
        """
        if len(self.buffer) < WINDOW // 2:
            return tensor.new_zeros(())
        current = self.emitted(tensor)
        history = torch.stack(self.buffer, dim=0)
        window = torch.cat([history, current.unsqueeze(0)], dim=0)
        centred = window - window.mean(dim=0, keepdim=True)
        if self.scale_free:
            return LAMBDA_VAR * self._participation(centred)
        std = torch.sqrt(centred.pow(2).mean(dim=0) + 1e-8)
        # Live lanes only -- see the mask's note in __init__.
        variance = torch.relu(GAMMA - std)[self.lanes].pow(2).mean()
        loss = LAMBDA_VAR * variance
        if self.decorrelate:
            # VICReg's second term, the one #592's formulation did not have.
            # Off-diagonal covariance per pair: what stops the floor being
            # satisfied by duplicating one direction across dimensions. Masked
            # the same way -- a covariance with a padded row is not a
            # correlation between two things the cell said.
            keep = self.lanes.unsqueeze(-1) & self.lanes.unsqueeze(-2)
            cov = torch.einsum("tpm,tpn->pmn", centred, centred) / window.shape[0]
            eye = torch.eye(cov.shape[-1], dtype=torch.bool).expand_as(keep)
            off = cov * (keep & ~eye)
            loss = loss + LAMBDA_DEC * off.pow(2).sum() / max(int(keep.sum()), 1)
        return loss


# -- the arms -----------------------------------------------------------------

#: The five arms, and what each one is for.
#:
#: `baseline`  -- the transport rule as shipped. B19's degeneracy arriving.
#: `holo`      -- the local holonomy term alone. Does flatness move at all?
#: `floor`     -- the variance floor alone. B32 predicts this one is satisfiable
#:                by duplication and that is the point of running it.
#: `floor_dec` -- floor with VICReg's decorrelation term. B32's correction.
#: `both`      -- the two terms together. #592's actual question.
#: `floor_si`  -- the **scale-free** floor: participation ratio of the emitted
#:                covariance. The absolute floor is unreachable under ADR-0032's
#:                band (see `_participation`), so this is the corrected form and
#:                the one the coexistence question should actually be asked of.
#: `both_si`   -- holonomy + the scale-free floor. **#592's question, corrected.**
ARMS = {
    # (holonomy, floor, decorrelate, scale_free)
    "baseline": (False, False, False, False),
    "holo": (True, False, False, False),
    "floor": (False, True, False, False),
    "floor_dec": (False, True, True, False),
    "floor_si": (False, True, False, True),
    "both": (True, True, True, False),
    "both_si": (True, True, False, True),
}


class Motion:
    """Is the world moving over this window? Put beside every dynamical reading.

    The rig runs against a body that stalls between 1,000 and 2,000 ticks
    (#572's `world_std_*`, #518's own `travel_window`), and a reading taken on a
    motionless world is not a reading about learning. This watches the
    **boundary** cells — the rows the world writes — and reports the largest
    per-component standard deviation over the window, which is #572's quantity
    on the stalks this rig actually has. Cheap enough that there is no reason not
    to stamp every checkpoint.
    """

    def __init__(self, env) -> None:
        self.env = env
        self.buffer: list[np.ndarray] = []

    def reset(self) -> None:
        self.buffer = []

    def observe(self) -> None:
        """The world's own state, not the sheaf's view of it.

        Read straight off MuJoCo — arm `qvel` and every puck pose — because the
        boundary stalks are a *representation* of the world and a dead
        representation and a dead world are exactly what has to be told apart.
        """
        try:
            env = self.env
            parts = [np.asarray(env.data.qvel, dtype=np.float64).ravel()]
            parts.append(np.asarray(env.data.qpos, dtype=np.float64).ravel())
            self.buffer.append(np.concatenate(parts))
        except Exception:
            pass

    def read(self) -> dict:
        if len(self.buffer) < 2:
            return {"available": False}
        arr = np.stack(self.buffer, axis=0)
        std = arr.std(axis=0)
        return {
            "available": True,
            "window": int(arr.shape[0]),
            "std_max": float(std.max()),
            "moving_share": float((std > 1e-6).mean()),
        }


def tracking(cells: list[dict]) -> dict:
    """Does what a cell emits track what it holds? B19's pairing, across cells.

    `corr(state, emitted)` over the predicting cells, learned against the Haar
    control drawn on this arm's own shape, mask and band — plus the quartile
    table #592 quotes, which is where the degeneracy is legible as *a constant
    near 1.5 carrying no information about the cell* rather than as a
    correlation coefficient.

    **This, and not the emitted rank alone, is whether the floor holds.** An arm
    can raise emitted rank everywhere and still say nothing about which cell it
    is: that is exactly the failure B19 recorded, and it is invisible to a
    median.
    """
    state = np.array([c["pr_total_centred"] for c in cells], dtype=np.float64)
    out: dict = {"cells": len(cells)}
    for name, key in (
        ("learned", "pr_emitted_centred"),
        ("haar", "pr_emitted_haar_centred"),
        ("haar_scaled", "pr_emitted_haar_scaled_centred"),
    ):
        if key not in cells[0]:
            continue
        emitted = np.array([c[key] for c in cells], dtype=np.float64)
        ok = np.isfinite(state) & np.isfinite(emitted)
        if ok.sum() < 3 or state[ok].std() == 0 or emitted[ok].std() == 0:
            out[f"corr_{name}"] = float("nan")
        else:
            out[f"corr_{name}"] = float(np.corrcoef(state[ok], emitted[ok])[0, 1])
        # The quartile table, by the cell's own state rank.
        order = np.argsort(state)
        quarters = np.array_split(order, 4)
        out[f"quartiles_{name}"] = [
            {
                "state": float(np.median(state[q])),
                "emitted": float(np.median(emitted[q])),
            }
            for q in quarters
        ]
    return out


def run_arm(arm: str, seed: int, ticks: int, out: Path) -> dict:
    """One arm to the horizon, checkpointed, with both readings at every rung."""
    started = time.time()
    use_holo, use_floor, use_dec, use_si = ARMS[arm]
    env, agent = arms_mod.build_arm(ARM, seed)
    dome = agent.dome
    bases = b29.cycles_of(dome)
    wide = bases["wide"]
    admitted = local_cycles(dome, wide, RADIUS)
    # If the wide basis yields nothing local, the term has no multi-dimensional
    # object to act on and that is itself the finding -- fall back to the full
    # basis's local cycles and say so in the record.
    fallback = False
    if not admitted:
        admitted = local_cycles(dome, bases["full"], RADIUS)
        fallback = True

    floor = RepresentationFloor(agent.sheaf, dome, decorrelate=use_dec, scale_free=use_si) if use_floor else None

    record = {
        "issue": 592,
        "reading": "can a representation floor and a local holonomy term coexist",
        "arm": arm,
        "base_arm": ARM,
        "terms": {"holonomy": use_holo, "floor": use_floor, "decorrelate": use_dec, "scale_free": use_si},
        "seed": seed,
        "ticks": ticks,
        "radius": RADIUS,
        "window": WINDOW,
        "gamma": GAMMA,
        "weights": {
            "holonomy": LAMBDA_HOLO,
            "variance": LAMBDA_VAR,
            "decorrelation": LAMBDA_DEC,
        },
        "cycles": {k: len(v) for k, v in bases.items()},
        "local_cycles": len(admitted),
        "local_from_fallback_basis": fallback,
        # How much of the emitted array is a lane at all. The first run of this
        # prototype floored the padded remainder too and the arm came out
        # bit-identical to baseline; the ratio is recorded so that failure is
        # visible in the data rather than only in the commentary.
        "live_lane_slots": int(floor.live) if floor is not None else None,
        "total_lane_slots": int(agent.sheaf.maps.pairs * agent.sheaf.maps.edge_width),
        "costs": {
            "ADR-0031": "the floor keeps a detached window; the rule holds no state",
            "ADR-0011": "cycles are local to some cell, not to the cell being moved",
        },
        "checkpoints": [],
    }

    bias = PredictionRule(agent.sheaf)
    transport = TransportRule(agent.sheaf)

    # B19's pairing, unchanged: a cell's state rank against what it can say, on
    # the same cell over the same window, with the Haar control (`+0.97`) drawn
    # on this arm's own shape, mask and band. This is *the* floor reading —
    # whether the floor holds is not a property of the term, it is this number.
    emission = b19.Emission(dome)
    attrs = b19.cell_attrs(dome)
    b19_rec = b19.Recorder(agent, emission)
    motion = Motion(env)

    def extra_step() -> dict:
        """The two added terms, descended after the shipped rule's own step.

        **Each term's gradient is normalised to unit norm before it is
        weighted**, and that is a deliberate design call rather than an
        implementation detail. Descended raw, the two terms differ by two to
        three orders of magnitude — the absolute floor's gradient measured
        7e-05 against the holonomy term's 3.5e-02 — so an arm carrying both
        would answer *does the floor survive the holonomy term* with *the floor
        was never applied*. Normalising makes `LAMBDA_HOLO` and `LAMBDA_VAR` an
        actual stated ratio, so *adversaries* is a claim the run can test rather
        than an artifact of whichever term happened to be louder.
        """
        stats = {}
        tensor = agent.sheaf.maps.maps
        if floor is not None:
            floor.observe(tensor)
        if not (use_holo or use_floor):
            return stats

        def unit(loss, param, key):
            if float(loss.detach()) == 0.0:
                return None
            (g,) = torch.autograd.grad(loss, param)
            n = torch.linalg.norm(g)
            if not bool(torch.isfinite(n)) or float(n) <= 0.0:
                return None
            stats[f"{key}_grad_raw"] = float(n)
            return g / n

        step = None
        if use_holo:
            param = tensor.detach().clone().requires_grad_(True)
            hl = holonomy_loss(dome, param, admitted)
            stats["holonomy_loss"] = float(hl.detach())
            g = unit(hl, param, "holonomy")
            if g is not None:
                step = LAMBDA_HOLO * g
        if floor is not None:
            param = tensor.detach().clone().requires_grad_(True)
            fl = floor.loss(param)
            stats["floor_loss"] = float(fl.detach())
            g = unit(fl, param, "floor")
            if g is not None:
                step = LAMBDA_VAR * g if step is None else step + LAMBDA_VAR * g
        if step is None:
            return stats
        with torch.no_grad():
            tensor.sub_(transport.learning_rate * step)
        agent.sheaf.maps.project()
        stats["extra_grad_norm"] = float(torch.linalg.norm(step))
        return stats

    def checkpoint(target: int, stats: dict) -> None:
        entry = {"ticks": target, "elapsed_minutes": (time.time() - started) / 60.0}
        entry["term_stats"] = stats
        # Every reading past 2,000 ticks is taken against a motionless world.
        # The flag is not a substitute for the stamp; both are recorded.
        entry["past_stall"] = target > LIVE_HORIZON
        entry["motion"] = motion.read()
        entry["read_window"] = len(motion.buffer)
        entry["holonomy"] = b29.surface_read(
            dome, agent.sheaf.maps, wide, f"{arm} s{seed} @{target} [wide]"
        )
        entry["holonomy_local"] = b29.surface_read(
            dome, agent.sheaf.maps, admitted, f"{arm} s{seed} @{target} [local]"
        )
        try:
            floor_read = b19.read(agent, b19_rec, attrs, null_seed=seed)
            cells = floor_read["cells"]
            floor_read.update(b19.summarise(cells))
            # **B19's own headline, and the number #592 quotes.** The quantiles
            # above say how much each cell emits; they do not say whether what a
            # cell emits tracks what it *has*. That is a correlation across
            # cells, and it is the whole finding: learned maps read -0.05 where a
            # random map of identical shape, mask and band reads +0.97, with the
            # poorest quartile lifted 1.054 -> 1.603 and the richest pressed
            # 2.167 -> 1.555 to a constant near 1.5 carrying no information about
            # the cell. Computed here rather than downstream so the per-cell rows
            # do not have to be carried in the record.
            floor_read["tracking"] = tracking(cells)
            floor_read.pop("cells", None)
            entry["floor"] = floor_read
        except Exception as exc:  # a floor read may not be the reason a run dies
            entry["floor"] = {"error": type(exc).__name__, "detail": str(exc)[:300]}
        record["checkpoints"].append(entry)
        out.with_suffix(".inflight.json").write_text(json.dumps(record, indent=1))
        sub = entry["holonomy"]["subsets"]["wide"]
        q = entry["floor"].get("quantiles", {})

        def med(key):
            v = q.get(key, {}).get("median")
            return f"{v:.3f}" if isinstance(v, float) else "  -  "

        mo = entry["motion"]
        tr = entry["floor"].get("tracking", {})

        def corr(key):
            v = tr.get(key)
            return f"{v:+.3f}" if isinstance(v, float) and np.isfinite(v) else "  -   "

        print(
            f"  [{arm} s{seed}] {target:>5}{'*' if entry['past_stall'] else ' '} "
            f"ident {sub['identification']['median']:.4f} "
            f"chan {sub['channel_return']['median']:.4f} "
            f"sig {sub['sigma_max']['median']:.2e} | "
            f"corr {corr('corr_learned')}/haar {corr('corr_haar')} "
            f"emit {med('pr_emitted_centred')} | "
            f"world {mo.get('std_max', float('nan')):.2e} "
            f"{stats}",
            flush=True,
        )

    checkpoint(0, {})
    seen = 0
    last_stats: dict = {}
    for target in [c for c in CHECKPOINTS if c <= ticks]:
        span = target - seen
        b19_rec.reset()
        motion.reset()
        remaining = span
        for _ in t0.run_ticks(agent, span, seed=seed + seen):
            remaining -= 1
            # The last `b19.WINDOW` ticks of each span are the read window, so
            # the floor and the world's motion are read over the same ticks.
            if remaining < min(READ_WINDOW, span):
                b19_rec.observe()
                motion.observe()
            bias.step()
            if agent.sheaf.ticks > 1:
                transport.step()
                last_stats = extra_step()
        seen = target
        checkpoint(target, last_stats)

    record["minutes"] = (time.time() - started) / 60.0
    out.write_text(json.dumps(record, indent=1))
    out.with_suffix(".inflight.json").unlink(missing_ok=True)
    return record


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["run"])
    p.add_argument("--arms", nargs="+", default=list(ARMS))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ticks", type=int, default=TICKS)
    args = p.parse_args()

    for arm in args.arms:
        out = _HERE / f"592-coexist-{arm}-seed{args.seed}-{args.ticks}.json"
        if out.exists():
            print(f"[B33] {out.name} already at the horizon, skipping", flush=True)
            continue
        print(f"[B33] {arm} seed {args.seed} -> {out.name}", flush=True)
        run_arm(arm, args.seed, args.ticks, out)


if __name__ == "__main__":
    main()
