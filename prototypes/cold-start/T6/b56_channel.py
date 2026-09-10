"""B56 (#628): does the two-sided channel-return holonomy term hold agreement
and audience differentiation up together on a **trained** surface?

**A cheap, rough prototype to react to — not an implementation.** #628 is a
`wayfinder:prototype` ticket on [#532](https://github.com/NGL321/patchworks/issues/532),
whose standing note is *plan, don't do*. Every surrogate below is named as one.

## What is being trained, and why it is not B33's term

[B54 (#624)](https://github.com/NGL321/patchworks/issues/624) lifted holonomy's
suspension **narrowly**, onto a *different* term from the one that was suspended.
[B33 (#592)](https://github.com/NGL321/patchworks/issues/592)'s full-stalk term
stays suspended and is **not** run here as a candidate. What re-enters is
two-sided, on the transport operator ([B17](https://github.com/NGL321/patchworks/issues/565):
never touches a stalk):

1. **Drive `channel_return` toward 1** — `|<u1, v1>|` on `holonomy_read`'s
   holonomy `H`, the round trip asked only of the one direction the loop
   actually transmits.
2. **Hold `identification` away from 0** — `‖UVᵀ − I‖_F / sqrt(2m)` on the full
   operator, so the collapsed-lane optimum is the term's **worst** point rather
   than its best.

Clause 1 alone is the graded form. Clause 2 is what discharges
[B42 (#605)](https://github.com/NGL321/patchworks/issues/605)'s standing
constraint. Local cycle scoping is kept ([B33](https://github.com/NGL321/patchworks/issues/592)'s
`RADIUS = 2`) but is **no longer what is load-bearing**; the measured guard
against drift to global is that both cycle bases are reported
([B32](https://github.com/NGL321/patchworks/issues/591) §1).

## The null it must beat, and it is a hard one

**B42's flat bundle reads `channel_return` 1.0000 at audience differentiation
0.0000.** Clause 1 alone is satisfied *perfectly* by the null. That is the entire
reason clause 2 exists, and it is why `channel` — clause 1 alone — is run as an
arm rather than assumed. Per [B48 (#615)](https://github.com/NGL321/patchworks/issues/615)
an arm that raises `channel_return` while differentiation falls has scored
**nothing**, so the two are never reported apart, and the exposure the
differentiation cost is reported with them.

## The two arms, and the user's own answer on B54 Q8

*Run both, leaning to the hinge.*

* **`hinge`** — clause 2 penalises only below a floor `GAMMA` and is inert above
  it, so it can forbid collapse without ever competing with clause 1.
* **`unbounded`** — clause 2 maximised. The named risk: `identification = 1` is
  **chance**, so an unbounded clause optimises toward *be random*, and B33
  measured an unbounded term outcompeting a bounded one by ~950x at natural
  gradient scales. Run to find out whether that risk is real here.

## Four surrogates, stated rather than smuggled

1. **`channel_return`'s gradient does not go through an SVD.** `H v1 = sigma1 u1`,
   so `<u1, v1> = v1ᵀ H v1 / ||H v1||` exactly when `v1` is the top right singular
   vector. `v1` is taken under `no_grad` and **detached**; the gradient flows
   through `H` only. This is the measured column itself wherever the top singular
   direction is stable across a step, and a Rayleigh quotient where it is not.
   Differentiating an SVD near a degenerate spectrum is not something a rough
   prototype should do, and B33 made the same call.
2. **`identification`'s gradient goes through B33's scale-normalised form.**
   `D = ‖sqrt(m)·H/‖H‖_F − I‖_F² / (2m)`, and `ident_hat = sqrt(D)` so the
   surrogate lands on the **same [0, sqrt(2)] scale** as the measured column
   (both are `1 − <normalised trace>` under the square root). It normalises by
   the Frobenius norm where the measurement normalises per-direction through the
   polar factor, so the two part company as the spectrum flattens or spikes.
   **`calibration` records the pairing cycle-by-cycle at every checkpoint**, so
   `GAMMA` can be read on the surrogate's own scale rather than assumed onto it.
3. **ADR-0011.** `b33.local_cycles` keeps only cycles inside some cell's closed
   `RADIUS`-hop neighbourhood: *some* cell can see the whole cycle, not the cell
   whose map is moved. Unchanged from B33, and still weaker than the ADR's bar.
4. **Unit-normalised gradients.** Per B33: descended raw, the two clauses differ
   by orders and the quieter one is invisible rather than outcompeted. Each
   clause's gradient is normalised to unit norm *before* it is weighted, so
   `LAMBDA_CHAN : LAMBDA_IDENT` is a stated ratio.

## GAMMA is an anchor, not a pass mark

**Provisional `GAMMA = 0.6075`**, *read off* B42's stagger-1 row — the first
notch of genuine audience differentiation on the record (`identification` 0.6075
at `channel_return` 1.0000 and differentiation 0.3000) — **not invented**. But
every number B54 reasoned from is a **construction sweep**, `605-stagger-seed42.json`,
not a trained surface. Per [B49 (#616)](https://github.com/NGL321/patchworks/issues/616)
a reading states which object it is on: this rig is where the
construction-to-trained gap is paid, and `GAMMA` is re-anchored by
`baseline`'s own `identification` at the live rungs, which is reported for
exactly that purpose.

## What is reported at every rung, and why each one is there

* **`channel_return` and `identification` together** — the two clauses, never one
  alone.
* **Both cycle bases, `wide` and `full`** — B32 §1's measured guard against the
  term drifting global. Not satisfied by having scoped the term locally in the
  source, so it is measured rather than argued.
* **Audience differentiation `1 − edge_overlap`** — B42's `b42_stagger.distinctness`,
  unmodified, plus **the exposure it cost**: the rank-measured `k_v` B22 uses,
  because the mask is fixed at construction and the *spec* field cannot move.
* **[B19 (#568)](https://github.com/NGL321/patchworks/issues/568)'s
  `corr(state, emitted)`** against its Haar control. B54 Q4 refused *"ADR-0032's
  band limits a side-channel's capacity"* as untested rather than testing it, so
  the CycleGAN failure is closed by joint report and by nothing else.
* **B33's scale-free representation floor, as a diagnostic** (B54 Q9). It does
  **not** ship in the term. But the channel form constrains **one** direction and
  leaves the other `m−1` unconstrained, and a collapse there is the failure mode
  nothing currently watches. Under B49 the floor is on the **emitted
  representation** and `identification` is on the **transport operator**, so one
  does not cover the other.
* **`rho(used)` and `tau`** — because **B54 Q5 was ceded**: that this term does
  not inherit [#610](https://github.com/NGL321/patchworks/issues/610)'s standing
  constraint (*a term aimed at retention states its case against the prediction
  term as an adversary*), on the argument that retention is temporal on `K` while
  holonomy is spatial on the transport operator. **That is an argument, not a
  measurement, and the user yielded the call.** This rig trains a holonomy term,
  so it is the cheap place to check.
* **The world's own motion** — [B38 (#599)](https://github.com/NGL321/patchworks/issues/599):
  the stall horizon is **per-run, not per-arm** (13x between seeds of one arm).
  Nothing is inherited: every run stamps MuJoCo's own state at every rung and
  `past_stall` is set from **that run's** stamps, in `b56_analyse.py`, rather than
  from a constant.

Usage::

    PYTHONPATH=src python prototypes/cold-start/T6/b56_channel.py run
    PYTHONPATH=src python prototypes/cold-start/T6/b56_channel.py run --arms hinge
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
for extra in (_ROOT / "prototypes" / "chart-double-duty-166", _ROOT / "benchmarks"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


b33 = _load("b56_b33", _HERE / "b33_coexist.py")
b42 = _load("b56_b42", _HERE / "b42_stagger.py")
b44 = _load("b56_b44", _HERE / "b44_return.py")
b29 = b33.b29
b19 = b33.b19

import holonomy_read as hr  # noqa: E402
from patchworks.learning import PredictionRule, TransportRule  # noqa: E402
from patchworks.restriction import pair_index  # noqa: E402

#: B27's ruling, and since #597 this is what `DomeSpec()` builds with nothing
#: overridden. `p = 12` is an **initialisation**, not a constant (B42).
ARM = b33.ARM

#: ADR-0011's radius, unchanged from B33.
RADIUS = b33.RADIUS

TICKS = 2000
CHECKPOINTS = (50, 100, 150, 250, 500, 1000, 2000)

#: The read window for B19's floor and the motion stamp. B33's, not B19's own
#: 1,000 — that is longer than this arm's live period on the record.
READ_WINDOW = b33.READ_WINDOW

#: **Read off B42's stagger-1 row, not invented.** See the module docstring:
#: it is an anchor for where to look, on a *construction* sweep, and this rig
#: re-anchors it against `baseline`'s own trained `identification`.
GAMMA = 0.6075

#: Stated ratio on unit-normalised gradients, equal by choice and unswept. Equal
#: is the fair form of *do the two clauses fight*, which is the question.
LAMBDA_CHAN = 0.05
LAMBDA_IDENT = 0.05

#: (clause 1, clause 2 mode)
#:
#: `baseline`   -- the transport rule as shipped. The trained surface's own numbers,
#:                 and where `GAMMA` gets re-anchored.
#: `channel`    -- **clause 1 alone**, which B42's flat bundle satisfies perfectly at
#:                 differentiation 0.0000. The control that says whether clause 2
#:                 is doing anything at all; without it `hinge` cannot be read.
#: `hinge`      -- clause 1 + a floor on clause 2, inert above `GAMMA`. The user's
#:                 lean on B54 Q8.
#: `unbounded`  -- clause 1 + clause 2 maximised. `identification = 1` is chance,
#:                 so this arm optimises toward *be random*; run to find out
#:                 whether that risk is real here rather than assumed.
ARMS = {
    "baseline": (False, None),
    "channel": (True, None),
    "hinge": (True, "hinge"),
    "unbounded": (True, "unbounded"),
}


# -- the two clauses ----------------------------------------------------------


def channel_return_of(h: torch.Tensor) -> torch.Tensor:
    """`|<u1, v1>|`, differentiable through `h` with the singular vector detached.

    `H v1 = sigma1 u1`, so `<u1, v1> = v1ᵀ H v1 / ||H v1||` exactly at the true
    top right singular vector. `v1` comes from an SVD under `no_grad` and carries
    no gradient; surrogate 1 in the module docstring.
    """
    with torch.no_grad():
        _u, _s, vh = torch.linalg.svd(h.detach())
        v1 = vh[0]
    hv = h @ v1
    denom = torch.linalg.norm(hv)
    if not bool(torch.isfinite(denom.detach())) or float(denom.detach()) <= 0.0:
        return None
    return torch.abs(torch.dot(v1, hv)) / denom


def identification_of(h: torch.Tensor) -> torch.Tensor:
    """`sqrt(‖sqrt(m)·H/‖H‖_F − I‖_F² / (2m))` — B33's form, on the measured scale.

    Both this and `holonomy_read.departures`'s `identification` are
    `sqrt(1 − <normalised trace>)`; this one normalises by the Frobenius norm
    where the measurement normalises per-direction through the polar factor.
    Surrogate 2, and `calibration` measures the gap rather than assuming it away.
    """
    m = h.shape[0]
    norm = torch.linalg.norm(h)
    if not bool(torch.isfinite(norm.detach())) or float(norm.detach()) <= 0.0:
        return None
    scaled = h * (np.sqrt(m) / norm)
    eye = torch.eye(m, dtype=h.dtype, device=h.device)
    d = (scaled - eye).pow(2).sum() / (2 * m)
    return torch.sqrt(torch.clamp(d, min=1e-12))


def clause_losses(dome, tensor: torch.Tensor, cycles, mode: str | None):
    """Both clauses over the admitted local cycles, as a pair of scalars.

    Returned separately and **never summed here**: they are descended as two
    unit-normalised gradients at a stated ratio, which is the only way *do the
    clauses fight* is a question the run can answer.
    """
    chan_total = tensor.new_zeros(())
    ident_total = tensor.new_zeros(())
    chan_raw, ident_raw = [], []
    count = 0
    for cycle in cycles:
        h = b33.holonomy_of(dome, tensor, cycle)
        if h is None or h.shape[0] != h.shape[1]:
            continue
        chan = channel_return_of(h)
        ident = identification_of(h)
        if chan is None or ident is None:
            continue
        count += 1
        chan_raw.append(float(chan.detach()))
        ident_raw.append(float(ident.detach()))
        # Clause 1: drive the channel to 1.
        chan_total = chan_total + (1.0 - chan)
        if mode == "hinge":
            # Inert above the floor, so it never competes with clause 1.
            ident_total = ident_total + torch.relu(GAMMA - ident).pow(2)
        elif mode == "unbounded":
            # Maximised. `identification = 1` is chance: this is the arm that
            # finds out whether *be random* is where it goes.
            ident_total = ident_total - ident
    n = max(count, 1)
    return (
        chan_total / n,
        ident_total / n,
        {
            "cycles_scored": count,
            "chan_surrogate_median": float(np.median(chan_raw)) if chan_raw else float("nan"),
            "ident_surrogate_median": float(np.median(ident_raw)) if ident_raw else float("nan"),
        },
    )


# -- the readings the ticket names --------------------------------------------


def exposure(dome, maps) -> dict:
    """What the differentiation cost, in the map's own currency.

    B22's **rank-measured** `k_v`: the rank of the stack of a cell's incident
    maps, which is how many of its own stalk directions reach a neighbour at all.
    The *spec* field cannot move — the mask is fixed at construction (B22) — so
    the spec number would report the same thing on every arm and say nothing.
    Reported as both a hard numeric rank and a participation ratio, because a
    rank is a cliff and this quantity slides.
    """
    n = dome.shape.n
    ranks, participations, permitted = [], [], []
    for cell in dome.predicting:
        blocks = []
        for edge_id in dome.incident[cell]:
            try:
                idx = pair_index(edge_id, b42.cg.side_of(dome, edge_id, cell))
            except Exception:
                continue
            m_e = dome.edges[edge_id].m
            b = maps.maps[idx].detach().to(torch.float64)[:m_e, :n]
            if b.shape[0] == 0 or b.norm() < 1e-12:
                continue
            blocks.append(b)
        if not blocks:
            continue
        stack = torch.cat(blocks, dim=0)
        s = torch.linalg.svdvals(stack).to(torch.float64)
        top = float(s[0])
        if top <= 0:
            continue
        ranks.append(int((s > top * 1e-8).sum()))
        # `(sum s^2)^2 / sum s^4` — the graded companion to the rank.
        participations.append(float(s.pow(2).sum().pow(2) / s.pow(4).sum()))
        permitted.append(int(dome._permitted[cell]))
    return {
        "k_v_rank_median": float(np.median(ranks)) if ranks else float("nan"),
        "k_v_participation_median": float(np.median(participations)) if participations else float("nan"),
        "k_v_permitted_median": float(np.median(permitted)) if permitted else float("nan"),
        "n": int(n),
        "cells": len(ranks),
    }


def calibration(dome, maps, cycles) -> dict:
    """The surrogate against the measured column, cycle by cycle.

    **This is what makes `GAMMA` re-anchorable.** `GAMMA` is read off a
    *measured* `identification` on a *construction* sweep, and the hinge acts on
    a *surrogate* on a *trained* surface. Two gaps, and neither is assumed away:
    this reports the pairing so both can be read off the record.
    """
    tensor = maps.maps.detach()
    meas_i, meas_c, surr_i, surr_c = [], [], [], []
    for cycle in cycles:
        with torch.no_grad():
            h = b33.holonomy_of(dome, tensor, cycle)
            if h is None or h.shape[0] != h.shape[1]:
                continue
            d = hr.departures(h)
            si = identification_of(h)
            sc = channel_return_of(h)
        if si is None or sc is None or not np.isfinite(d.get("identification", np.nan)):
            continue
        meas_i.append(d["identification"])
        meas_c.append(d["channel_return"])
        surr_i.append(float(si))
        surr_c.append(float(sc))
    out: dict = {"cycles": len(meas_i)}
    if len(meas_i) < 3:
        return out
    mi, si_ = np.array(meas_i), np.array(surr_i)
    mc, sc_ = np.array(meas_c), np.array(surr_c)
    out["identification"] = {
        "measured_median": float(np.median(mi)),
        "surrogate_median": float(np.median(si_)),
        "corr": float(np.corrcoef(mi, si_)[0, 1]) if mi.std() and si_.std() else float("nan"),
        # `GAMMA` translated onto the surrogate's scale by the median ratio.
        "gamma_on_surrogate_scale": float(GAMMA * np.median(si_) / np.median(mi))
        if np.median(mi) > 0
        else float("nan"),
    }
    out["channel_return"] = {
        "measured_median": float(np.median(mc)),
        "surrogate_median": float(np.median(sc_)),
        "max_abs_gap": float(np.abs(mc - sc_).max()),
    }
    return out


def run_arm(arm: str, seed: int, ticks: int, out: Path) -> dict:
    """One arm to the horizon, with every column the ticket names at every rung."""
    started = time.time()
    use_chan, mode = ARMS[arm]
    env, agent = b33.arms_mod.build_arm(ARM, seed)
    dome = agent.dome
    bases = b29.cycles_of(dome)
    wide, full = bases["wide"], bases["full"]
    admitted = b33.local_cycles(dome, wide, RADIUS)
    fallback = False
    if not admitted:
        admitted = b33.local_cycles(dome, full, RADIUS)
        fallback = True

    # B33's scale-free floor, **as a diagnostic only**. Constructed with
    # `scale_free=True` so `_participation` is what it reports, and its `loss` is
    # never added to `step`: B54 Q9 says the floor does not ship in the term.
    floor_probe = b33.RepresentationFloor(
        agent.sheaf, dome, decorrelate=False, scale_free=True
    )

    record = {
        "issue": 628,
        "reading": "two-sided channel-return holonomy on a trained surface",
        "arm": arm,
        "base_arm": ARM,
        "terms": {"channel_return": use_chan, "identification_mode": mode},
        "seed": seed,
        "ticks": ticks,
        "radius": RADIUS,
        "gamma": GAMMA,
        "weights": {"channel": LAMBDA_CHAN, "identification": LAMBDA_IDENT},
        "cycles": {k: len(v) for k, v in bases.items()},
        "local_cycles": len(admitted),
        "local_from_fallback_basis": fallback,
        "surrogates": {
            "channel_return": "v1' H v1 / ||H v1||, v1 detached from a no_grad SVD",
            "identification": "sqrt(||sqrt(m) H/||H||_F - I||_F^2 / 2m) (B33's form)",
            "ADR-0011": "cycles local to some cell, not to the cell being moved",
            "floor": "B33's scale-free floor is reported, never descended (B54 Q9)",
        },
        "null_to_beat": {
            "source": "B42 605-stagger-seed42.json stagger 0",
            "channel_return": 1.0,
            "differentiation": 0.0,
            "note": "clause 1 alone is satisfied perfectly by the flat bundle",
        },
        "checkpoints": [],
    }

    bias = PredictionRule(agent.sheaf)
    transport = TransportRule(agent.sheaf)
    emission = b19.Emission(dome)
    attrs = b19.cell_attrs(dome)
    b19_rec = b19.Recorder(agent, emission)
    motion = b33.Motion(env)

    def extra_step() -> dict:
        """The two clauses, descended after the shipped rule's own step.

        Each clause's gradient is normalised to **unit norm before it is
        weighted** (B33): descended raw the two differ by orders, and an arm
        carrying both would answer *do the clauses fight* with *one was never
        applied*.
        """
        stats: dict = {}
        tensor = agent.sheaf.maps.maps
        floor_probe.observe(tensor)
        if not use_chan:
            return stats

        param = tensor.detach().clone().requires_grad_(True)
        chan_loss, ident_loss, raw = clause_losses(dome, param, admitted, mode)
        stats.update(raw)
        stats["chan_loss"] = float(chan_loss.detach())
        stats["ident_loss"] = float(ident_loss.detach())

        def unit(loss, key):
            if float(loss.detach()) == 0.0:
                return None
            (g,) = torch.autograd.grad(loss, param, retain_graph=True)
            nrm = torch.linalg.norm(g)
            if not bool(torch.isfinite(nrm)) or float(nrm) <= 0.0:
                return None
            stats[f"{key}_grad_raw"] = float(nrm)
            return g / nrm

        step = None
        g = unit(chan_loss, "chan")
        if g is not None:
            step = LAMBDA_CHAN * g
        if mode is not None:
            g = unit(ident_loss, "ident")
            if g is not None:
                step = LAMBDA_IDENT * g if step is None else step + LAMBDA_IDENT * g
        if step is None:
            return stats
        with torch.no_grad():
            tensor.sub_(transport.learning_rate * step)
        agent.sheaf.maps.project()
        stats["extra_grad_norm"] = float(torch.linalg.norm(step))
        return stats

    def checkpoint(target: int, stats: dict) -> None:
        maps = agent.sheaf.maps
        entry: dict = {
            "ticks": target,
            "elapsed_minutes": (time.time() - started) / 60.0,
            "term_stats": stats,
        }
        # **Stamped, never inherited.** B38: the stall horizon is per-run, and
        # `past_stall` is set from these stamps in `b56_analyse.py`.
        entry["motion"] = motion.read()
        entry["read_window"] = len(motion.buffer)
        # Both bases. B32 §1's measured guard against the term drifting global.
        entry["holonomy_wide"] = b29.surface_read(
            dome, maps, wide, f"{arm} s{seed} @{target} [wide]"
        )
        entry["holonomy_full"] = b29.surface_read(
            dome, maps, full, f"{arm} s{seed} @{target} [full]"
        )
        entry["holonomy_local"] = b29.surface_read(
            dome, maps, admitted, f"{arm} s{seed} @{target} [local]"
        )
        # B48's joint scoring rule: agreement is never reported without the
        # differentiation beside it and the exposure it cost.
        overlap = b42.distinctness(dome, maps)
        entry["differentiation"] = {
            "edge_overlap": overlap,
            "audience_differentiation": 1.0 - overlap,
        }
        entry["exposure"] = exposure(dome, maps)
        entry["calibration"] = calibration(dome, maps, admitted)
        # B54 Q5, ceded on argument and checked here.
        entry["retention"] = b44.spectra(agent.sheaf.operators)
        entry["retention"].pop("rho_raw_per_cell", None)
        # B54 Q9's diagnostic. Reported, never descended.
        try:
            with torch.no_grad():
                if len(floor_probe.buffer) >= b33.WINDOW // 2:
                    entry["representation_floor"] = {
                        "participation_loss": float(floor_probe.loss(maps.maps.detach())),
                        "window": len(floor_probe.buffer),
                    }
                else:
                    entry["representation_floor"] = {"window": len(floor_probe.buffer)}
        except Exception as exc:
            entry["representation_floor"] = {"error": type(exc).__name__, "detail": str(exc)[:200]}
        # B19's pairing. B54 Q4 refused the band defence as untested; this is the
        # test, and the CycleGAN failure is closed by it or by nothing.
        try:
            floor_read = b19.read(agent, b19_rec, attrs, null_seed=seed)
            cells = floor_read["cells"]
            floor_read.update(b19.summarise(cells))
            floor_read["tracking"] = b33.tracking(cells)
            floor_read.pop("cells", None)
            entry["floor"] = floor_read
        except Exception as exc:
            entry["floor"] = {"error": type(exc).__name__, "detail": str(exc)[:300]}
        record["checkpoints"].append(entry)
        out.with_suffix(".inflight.json").write_text(json.dumps(record, indent=1))

        w = entry["holonomy_wide"]["subsets"]["wide"]
        # `surface_read`'s subsets are always (wide, live) whatever basis it is
        # handed, so the full basis is read at the top level -- all 260 cycles,
        # the width-1 ones included, which is what "does it drift global" asks.
        f = entry["holonomy_full"]
        # ...and its own width->=2 subset, which is the apples-to-apples
        # comparator against the `wide` basis: `full`'s top level is dominated by
        # the width-1 cycles whose `channel_return` is identically 1 (b29's note),
        # so a drift-to-global claim has to be read on `fw`, not on `f`.
        fw = entry["holonomy_full"]["subsets"]["wide"]
        tr = entry["floor"].get("tracking", {})
        mo = entry["motion"]
        ret = entry["retention"]

        def corr(key):
            v = tr.get(key)
            return f"{v:+.3f}" if isinstance(v, float) and np.isfinite(v) else "  -   "

        print(
            f"  [{arm} s{seed}] {target:>5} "
            f"wide chan {w['channel_return']['median']:.4f} id {w['identification']['median']:.4f} | "
            f"full chan {f['channel_return']['median']:.4f} id {f['identification']['median']:.4f} "
            f"(w {fw['channel_return']['median']:.4f}/{fw['identification']['median']:.4f}) | "
            f"diff {entry['differentiation']['audience_differentiation']:.4f} "
            f"kv {entry['exposure']['k_v_rank_median']:.1f}/{entry['exposure']['k_v_participation_median']:.1f} | "
            f"corr {corr('corr_learned')}/haar {corr('corr_haar')} | "
            f"rho(used) {ret['rho_used']['median']:.4f} tau {ret['tau_from_rho_used']['median']:.4g} | "
            f"world {mo.get('std_max', float('nan')):.2e} {stats}",
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
        for _ in b33.t0.run_ticks(agent, span, seed=seed + seen):
            remaining -= 1
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
        out = _HERE / f"628-channel-{arm}-seed{args.seed}-{args.ticks}.json"
        if out.exists():
            print(f"[B56] {out.name} already at the horizon, skipping", flush=True)
            continue
        print(f"[B56] {arm} seed {args.seed} -> {out.name}", flush=True)
        run_arm(arm, args.seed, args.ticks, out)


if __name__ == "__main__":
    main()
