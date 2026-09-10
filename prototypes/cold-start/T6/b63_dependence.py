"""B63 (#637): the sensorimotor dependence instrument, and its first reading.

[B43 (#609)](https://github.com/NGL321/patchworks/issues/609) decided **what the
measure is** and fenced the instrument out of its own scope. This builds it.

    I(P; Delta)  --  traditional Shannon mutual information between `P`, a random
    variable over `k` distinguishable perturbation **patterns within one sensory
    stratum** at fixed norm `A0 = 1`, and `Delta`, the **paired-counterfactual
    response at the world-read boundary cell**, with the situation `C`
    **marginalised**. The predicate is `I(P; Delta) > 0` as an excess over a
    matched scramble null, reported **per stratum as a profile**.

**`traditional`, declared** (B43 §8): every information-theoretic quantity in
this file is ordinary Shannon on random variables. Nothing here is
cohomological, in either of the record's two cohomological senses.

**Name the surface** (#455): `reserve_p12` — `capacity_budget = 63`,
`private_reserve = 12` — which is `b33.ARM`, the surface B44 (#610), B56 (#628)
and B58 (#630) all read on.

**Which object** (B49, #616): the endpoint measure is on **node stalks** — the
actuator's `commanded` block, which is `detectability.reading_sites`' own
write-complement at that cell. The per-edge profile (`--lanes`) is on the
**lane**. No number is carried between them, and each is labelled in the record.

---

## The five things this ticket had to settle, and what was done

**1. The estimator, and its bias.** Traditional MI between a finite `P` and a
continuous `Delta` is `H(P) - H(P|Delta)`, and with `P` uniform and *chosen* by
the experimenter `H(P) = log k` exactly. So the estimate is a **recovery**
problem, not a density problem: decode `P` from `Delta`, form the `k x k`
confusion matrix `(P, Phat)`, and take the plug-in MI of that table. By the
data-processing inequality `Phat = f(Delta)` gives

    I(P; Phat) <= I(P; Delta)

so **every number this file reports is a lower bound on the measure**, and the
sign of its bias runs the safe way: an estimator that degrades with dimension
loses the decoder accuracy, which *lowers* the estimate. That is the opposite of
the positive-bias failure the ticket names — a plug-in density estimate on a
wide continuous stalk reads bias as signal, and a bounded recovery estimate
cannot. The decoder is **leave-one-out** (a within-sample decoder makes the
table diagonal by construction), and two decoders are run — nearest class
centroid and 1-NN, both on the cosine metric — reported separately, never
maximised over.

Residual plug-in bias on the `k x k` table is positive and of order
`(k-1)^2 / 2N`. It is **not corrected by a formula**: the **matched scramble
null** runs the identical pipeline — same decoder, same folds, same table — on
permuted labels, so whatever bias the pipeline carries is in the null too and
the reported quantity is the **excess**. Miller-Madow is reported alongside as a
cross-check on that claim, never as the headline.

**2. The alphabet `k`, and what a pattern is.** A pattern is a **unit vector
over the whole stratum's product space**, `dim = cells x stalk` — B43 §7's *two
different things happening in two places in the visual field*, which is a
*spatial* distinction and not a direction in one cell's stalk. `k = 8`, drawn as
an orthonormal set where `dim >= k` and as a maximally-spread random set where
it is not, with the minimum pairwise `|cos|` recorded. The strata are very
unequal in this: patch is `256 x 48 = 12288`-dimensional, proprioceptive `3 x 2
= 6`, touch `3 x 1 = 3` — so touch **cannot** carry 8 orthogonal patterns and
its crowding is reported rather than hidden. The labels are **nested**, so
restricting the same trials to the first `k'` labels gives a free `k`-ladder at
`k' = 2, 4, 8` and the ceiling `log k'` moves with it.

**3. Trial and configuration counts, which are the noise model.** B43 §3: the
sweep **is** the noise, so its size is not a convenience. One `env.reset(seed)`
is one situation `C`; every pattern of every stratum is injected at that
situation off **one shared quiet fork**, so the design is balanced by
construction — exactly one trial per `(C, stratum, pattern)` cell. The reading
is reported on a **configuration ladder** (`--ladder`), because too few
situations and the marginalised form saturates toward the vacuous conditioned
one, and a ladder is what makes that visible instead of assumed.

**The conditioned form is not estimated, it is *verified vacuous*.** B43 §3
argues `I(P; Delta | C) = log k` by construction because the sandbox is
deterministic. Estimating a quantity that is `log k` by construction would be
theatre; instead one `(C, pattern)` branch is **re-run** and `Delta` checked
**bit-identical**. That is the determinism claim itself, and it is checked
rather than assumed (`determinism` in the record).

**4. `#224`'s gate.** The read is taken on the float64 cast, as `tau-hat` is,
and the record carries the peak `||Delta||` per trial against both
`eps_f32 * ||state||` and `eps_f64 * ||state||`. Every reading is reported
**twice** — over all trials, and over the gate-clearing subset — because a
decoder fed vectors at the arithmetic-noise floor is decoding noise, and the
scramble null is what says whether it did.

**Presence/absence is degenerate here, and that is the finding.** B43 §5 asks
for it as the cheap floor case. Under the paired counterfactual the unperturbed
arm's `Delta` is **exactly zero**, so `presence` is decodable at any amplitude
and `I = log 2` trivially — the question collapses into #224's gate, which is
what is reported in its place.

**5. The per-edge diagnostic.** Implemented and **opt-in** (`--lanes`), on the
budget call the ticket left open: it is the same trials, so it costs no runs,
but its null is `edges x permutations` and that is what costs. Labelled `lane`
throughout.

---

## What is *not* here

No path quantifier, in any form: B43 §4 struck it and ruled no successor, so
there is no `max` over routes and no `min` along one. No amplitude predicate:
`P` varies **which pattern**, never how hard (B43 §7), and the response feature
is a **direction** for the same reason. `--no-file` is unconditional — B43 ruled
the old cutoff readings not re-filable, and this instrument's first reading is
not *the* read.

Usage::

    PYTHONPATH=src python prototypes/cold-start/T6/b63_dependence.py train
    PYTHONPATH=src python prototypes/cold-start/T6/b63_dependence.py read --arm trained
    PYTHONPATH=src python prototypes/cold-start/T6/b63_dependence.py read --arm untrained
    PYTHONPATH=src python prototypes/cold-start/T6/b63_dependence.py read --arm flat
    PYTHONPATH=src python prototypes/cold-start/T6/b63_dependence.py analyse
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
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


b33 = _load("b63_b33", _HERE / "b33_coexist.py")
b29 = _load("b63_b29", _HERE / "b29_holonomy.py")
b44 = _load("b63_b44", _HERE / "b44_return.py")

import detectability as det  # noqa: E402
import untrained_fixed_point as ufp  # noqa: E402
from patchworks.graph import CellKind  # noqa: E402

#: Training horizon. B33's and B44's, so this reading sits on the same surface
#: every other reading on this map's trained arm sits on.
LEARN = 2000

#: The alphabet. See docstring item 2; the labels are nested so 2 and 4 come free.
K = 8
K_LADDER = (2, 4, 8)

#: Situations. See docstring item 3 — this is the noise model, not a convenience.
CONFIGS = 24
CONFIG_LADDER = (4, 8, 16, 24)

#: Permutations of the matched scramble null. The null is permuted **within each
#: configuration**, which preserves the balanced design exactly and destroys only
#: the `P`-to-`Delta` correspondence — a global shuffle would unbalance the table
#: and put design noise into the null the observed reading does not carry.
PERMUTATIONS = 400

ARMS = ("untrained", "trained", "flat")


# -- the alphabet -------------------------------------------------------------


def stratum_alphabet(dome, cells, k: int, seed: int) -> dict:
    """`k` unit patterns over one stratum's product space, at `A0 = 1`.

    Orthonormal where the stratum is wide enough to carry `k` orthogonal
    directions, maximally-spread random where it is not. The minimum pairwise
    `|cos|` is returned with them, because on touch (`dim = 3`) eight patterns
    are *crowded* and a reading that does not say so is quoting a ceiling it
    cannot reach.
    """
    width = int(dome.cells[cells[0]].stalk)
    dim = width * len(cells)
    gen = torch.Generator().manual_seed(seed + 63_000)
    raw = torch.randn(dim, k, generator=gen, dtype=torch.float64)
    if k <= dim:
        q, r = torch.linalg.qr(raw)
        basis = q[:, :k] * torch.sign(torch.diagonal(r)[:k]).unsqueeze(0)
    else:
        basis = raw / raw.norm(dim=0, keepdim=True).clamp_min(1e-300)
    gram = (basis.T @ basis).abs().numpy()
    off = gram[~np.eye(k, dtype=bool)]
    patterns = []
    for label in range(k):
        vec = basis[:, label]
        nudge = tuple(
            (int(c), vec[i * width : (i + 1) * width].clone())
            for i, c in enumerate(cells)
        )
        total = float(
            torch.sqrt(sum((v * v).sum() for _, v in nudge))
        )
        patterns.append(nudge)
        assert abs(total - 1.0) < 1e-9, f"A0 = 1 violated: {total}"
    return {
        "patterns": patterns,
        "dim": dim,
        "width": width,
        "cells": len(cells),
        "orthonormal": bool(k <= dim),
        "max_abs_cos": float(off.max()) if off.size else 0.0,
    }


# -- the estimator ------------------------------------------------------------


def _plugin_mi(table: np.ndarray) -> tuple[float, float]:
    """Plug-in and Miller-Madow MI of a `k x k` count table, in **bits**."""
    n = table.sum()
    if n == 0:
        return 0.0, 0.0
    p = table / n
    px = p.sum(axis=1, keepdims=True)
    py = p.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0, p * np.log2(p / (px * py)), 0.0)
    mi = float(terms.sum())
    # Miller-Madow on the three entropies: (support - 1) / (2 N ln 2) each.
    def bias(counts: np.ndarray) -> float:
        return (np.count_nonzero(counts) - 1) / (2 * n * np.log(2))
    mm = mi - bias(table.sum(axis=1)) - bias(table.sum(axis=0)) + bias(table)
    return mi, float(mm)


def _decode(features: np.ndarray, labels: np.ndarray, k: int, how: str) -> np.ndarray:
    """Leave-one-out decode of `labels` from `features`. Cosine metric.

    Cosine, not Euclidean: B43 §7 holds the *injected* norm fixed because
    amplitude is buyable on this map, and the response side inherits the same
    discipline — `||Delta||` varies with the situation, so a metric that reads it
    would let the estimator recover `C` and call it content.
    """
    x = features / np.linalg.norm(features, axis=1, keepdims=True).clip(1e-300)
    n = x.shape[0]
    gram = x @ x.T
    out = np.empty(n, dtype=int)
    if how == "centroid":
        sums = np.zeros((k, x.shape[1]))
        counts = np.zeros(k)
        for i in range(n):
            sums[labels[i]] += x[i]
            counts[labels[i]] += 1
        for i in range(n):
            held = sums.copy()
            hc = counts.copy()
            held[labels[i]] -= x[i]
            hc[labels[i]] -= 1
            live = hc > 0
            cent = np.zeros_like(held)
            cent[live] = held[live] / hc[live, None]
            norms = np.linalg.norm(cent, axis=1)
            score = np.where(norms > 0, cent @ x[i] / np.maximum(norms, 1e-300), -np.inf)
            out[i] = int(np.argmax(score))
        return out
    # 1-NN, self excluded.
    np.fill_diagonal(gram, -np.inf)
    return labels[np.argmax(gram, axis=1)]


def estimate(features: np.ndarray, labels: np.ndarray, groups: np.ndarray, k: int,
             how: str, permutations: int, rng: np.random.Generator) -> dict:
    """`I(P; Phat)` in bits, with its matched scramble null. A **lower bound**.

    `groups` is the configuration index; the null permutes labels **within** a
    group, which keeps the design balanced and removes only the correspondence.
    """
    if features.shape[0] < 2 * k:
        return {"trials": int(features.shape[0]), "insufficient": True}
    pred = _decode(features, labels, k, how)
    table = np.zeros((k, k), dtype=float)
    for a, b in zip(labels, pred):
        table[a, b] += 1
    mi, mm = _plugin_mi(table)
    accuracy = float((pred == labels).mean())
    null = np.empty(permutations)
    for j in range(permutations):
        shuffled = labels.copy()
        for g in np.unique(groups):
            idx = np.flatnonzero(groups == g)
            shuffled[idx] = rng.permutation(labels[idx])
        p2 = _decode(features, shuffled, k, how)
        t2 = np.zeros((k, k), dtype=float)
        for a, b in zip(shuffled, p2):
            t2[a, b] += 1
        null[j] = _plugin_mi(t2)[0]
    return {
        "decoder": how,
        "trials": int(features.shape[0]),
        "features": int(features.shape[1]),
        "k": k,
        "ceiling_bits": float(np.log2(k)),
        "accuracy": accuracy,
        "chance_accuracy": 1.0 / k,
        "mi_bits": mi,
        "mi_bits_miller_madow": mm,
        "null_mean_bits": float(null.mean()),
        "null_q95_bits": float(np.quantile(null, 0.95)),
        "excess_bits": float(mi - null.mean()),
        "p_value": float((np.sum(null >= mi) + 1) / (permutations + 1)),
        "permutations": permutations,
    }


# -- the run ------------------------------------------------------------------


def actuator_of(dome) -> int:
    for cell in dome.cells:
        if cell.kind is CellKind.ACTUATOR:
            return int(cell.id)
    raise ValueError("no actuator cell: there is no world-read boundary to read at")


def response(quiet: torch.Tensor, moved: torch.Tensor, joints: int) -> dict:
    """`Delta` at the world-read boundary: the paired deviation, commanded block.

    `reading_sites`' write-complement at the actuator is the leading `joints`
    components — `Agent.write` sets the efference components every tick and the
    commanded ones never — so the deviation is read there and nowhere else.

    Two features come back. **`peak`** is ADR-0021's own per-trial reduction with
    the amplitude divided out: the deviation vector at the tick of largest norm,
    normalised. **`trace`** is the whole window, scaled by that same peak norm.
    `peak` is the headline because it is the reduction B43 retained verbatim;
    `trace` is reported beside it because a 3-wide terminus is a hard ceiling and
    the window is the only other axis content could survive on.
    """
    dev = (moved - quiet)[:, :joints].numpy()
    norms = np.linalg.norm(dev, axis=1)
    at = int(np.argmax(norms))
    peak = float(norms[at])
    scale = max(peak, 1e-300)
    return {
        "peak_vector": dev[at] / scale,
        "trace_vector": (dev / scale).ravel(),
        "peak_norm": peak,
        "peak_at": at,
    }


def run_arm(arm: str, seed: int, learn: int, configs: int, k: int, window: int,
            hold: int, out: Path, lanes: bool) -> dict:
    started = time.time()
    env, agent = b33.arms_mod.build_arm(b33.ARM, seed)
    dome = agent.dome
    blob = torch.load(_HERE / f"637-trained-seed{seed}-{learn}.pt", weights_only=False)

    if arm == "untrained":
        note = "construction: no training run, the shipped rule's starting point"
    else:
        agent.sheaf.operators.load_state_dict(blob["operators"])
        agent.sheaf.biases.load_state_dict(blob["biases"])
        note = f"the trained maps at {learn} ticks"
    trained_maps = blob["maps"].to(agent.sheaf.maps.maps.dtype)
    if arm == "trained":
        with torch.no_grad():
            agent.sheaf.maps.maps.copy_(trained_maps)
    elif arm == "flat":
        # B42's retained null, installed exactly as B44 (#610) installs it: one
        # orthogonal frame per cell built **inside** the reserve, every incident
        # edge taking its own first `m_e` rows.
        b44.install_frames(dome, agent.sheaf.maps, b44.reserved_frames(dome, seed))
        note = "B42's flat bundle on the trained operators"

    wide = b29.cycles_of(dome)["wide"]
    cycle_surface = b44.surface(dome, agent.sheaf.maps, wide, arm)
    cast = det.double_precision(agent.sheaf)
    joints = int(dome.spec.joints)
    act = actuator_of(dome)
    strata = det.rim_strata(dome)
    kinds = [dome.cells[s[0]].kind.value for s in strata]
    books = {
        kind: stratum_alphabet(dome, tuple(cells), k, seed)
        for kind, cells in zip(kinds, strata)
    }

    print(
        f"[B63] {arm} on {b33.ARM} seed {seed}: {len(dome.cells)} cells, "
        f"{len(dome.edges)} edges; terminus = actuator #{act}, commanded width "
        f"{joints}; strata " + ", ".join(
            f"{kd} dim {books[kd]['dim']}" for kd in kinds
        ),
        flush=True,
    )
    print(
        f"[B63] {arm}: ident {cycle_surface['identification']:.4f}  "
        f"chan {cycle_surface['channel_return']:.4f}  "
        f"sigma {cycle_surface['sigma_max']:.3e}",
        flush=True,
    )

    record = {
        "issue": 637,
        "measure": "I(P; Delta), traditional Shannon, situation marginalised",
        "traditional_or_cohomological": "traditional",
        "object": {"endpoint": "node stalks (actuator commanded block)",
                   "per_edge": "lane"},
        "arm": arm,
        "note": note,
        "surface": b33.ARM,
        "seed": seed,
        "learn": learn if arm != "untrained" else 0,
        "configs": configs,
        "k": k,
        "window": window,
        "hold": hold,
        "double_precision_tensors": cast,
        "terminus": {"actuator_cell": act, "commanded_width": joints,
                     "stalk": int(dome.cells[act].stalk)},
        "alphabet": {kd: {x: books[kd][x] for x in
                          ("dim", "width", "cells", "orthonormal", "max_abs_cos")}
                     for kd in kinds},
        "cycle_surface": cycle_surface,
        "no_file": True,
        "trials": [],
    }

    lane_rows: list[np.ndarray] = []
    determinism = None

    for ci in range(configs):
        observation, _ = env.reset(seed=seed * 1000 + ci)
        agent.observe(observation)
        applied = np.zeros(env.action_space.shape, dtype=np.float64)
        det.hold_still(agent, observation, applied, hold)
        state = ufp.snapshot(agent.sheaf)
        with torch.no_grad():
            settled = float(agent.sheaf.stalks.norm())
        quiet_dis, quiet_stalks = det.branch(
            agent, state, observation, applied, window, None, record=(act,)
        )
        qa = quiet_stalks[act]
        for kind in kinds:
            for label, nudge in enumerate(books[kind]["patterns"]):
                moved_dis, moved_stalks = det.branch(
                    agent, state, observation, applied, window, nudge, record=(act,)
                )
                r = response(qa, moved_stalks[act], joints)
                row = {
                    "config": ci,
                    "kind": kind,
                    "label": label,
                    "peak_norm": r["peak_norm"],
                    "peak_at": r["peak_at"],
                    "state_norm_after_hold": settled,
                    "peak": [float(x) for x in r["peak_vector"]],
                    "trace": [float(x) for x in r["trace_vector"]],
                }
                record["trials"].append(row)
                if lanes:
                    ld = (moved_dis - quiet_dis).norm(dim=-1).numpy()
                    at = int(np.argmax(ld.sum(axis=1)))
                    lane_rows.append(ld[at])
                if determinism is None and ci == 0 and label == 0:
                    again_dis, again_stalks = det.branch(
                        agent, state, observation, applied, window, nudge, record=(act,)
                    )
                    r2 = response(qa, again_stalks[act], joints)
                    determinism = {
                        "claim": "B43 §3: the sandbox is deterministic, so the "
                                 "conditioned form I(P;Delta|C) reads log k by "
                                 "construction and is vacuous",
                        "bit_identical": bool(
                            torch.equal(again_stalks[act], moved_stalks[act])
                        ),
                        "max_abs_delta": float(
                            (again_stalks[act] - moved_stalks[act]).abs().max()
                        ),
                        "peak_norm_repeat_delta": abs(
                            r2["peak_norm"] - r["peak_norm"]
                        ),
                    }
        ufp.restore(agent.sheaf, state)
        print(
            f"  config {ci + 1}/{configs}: {len(record['trials'])} trials, "
            f"|Delta| peak median "
            f"{np.median([t['peak_norm'] for t in record['trials']]):.3e}, "
            f"{(time.time() - started) / 60:.1f} min",
            flush=True,
        )
        record["determinism"] = determinism
        record["minutes"] = (time.time() - started) / 60.0
        out.write_text(json.dumps(record, indent=1), encoding="utf-8")

    # #224's gate, per trial, on the cast the read is taken on.
    eps32, eps64 = float(np.finfo(np.float32).eps), float(np.finfo(np.float64).eps)
    for row in record["trials"]:
        row["clears_f32_floor"] = bool(
            row["peak_norm"] > eps32 * row["state_norm_after_hold"]
        )
        row["clears_f64_floor"] = bool(
            row["peak_norm"] > eps64 * row["state_norm_after_hold"]
        )
    record["gate_224"] = {
        "eps_f32": eps32,
        "eps_f64": eps64,
        "clearing_f32": int(sum(r["clears_f32_floor"] for r in record["trials"])),
        "clearing_f64": int(sum(r["clears_f64_floor"] for r in record["trials"])),
        "trials": len(record["trials"]),
        "note": "presence/absence, B43 §5's floor case, collapses into this: the "
                "unperturbed arm's Delta is exactly zero under the pairing, so "
                "presence is decodable at any amplitude that clears the floor",
    }
    if lanes and lane_rows:
        np.save(out.with_suffix(".lanes.npy"), np.asarray(lane_rows))
        record["lane_rows"] = [len(lane_rows), int(lane_rows[0].size)]
    record["minutes"] = (time.time() - started) / 60.0
    out.write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(
        f"[B63] {arm}: {len(record['trials'])} trials in "
        f"{record['minutes']:.1f} min -> {out.name}",
        flush=True,
    )
    return record


def train(seed: int, learn: int, out: Path) -> dict:
    """One training run, with its **own** motion horizon stamped (B38, #599)."""
    started = time.time()
    env, agent = b33.arms_mod.build_arm(b33.ARM, seed)
    motion = b33.Motion(env)
    print(f"[B63] training {b33.ARM} seed {seed} for {learn} ticks", flush=True)
    for i, _ in enumerate(ufp.teaching(agent, learn, seed)):
        if i >= learn - b33.READ_WINDOW:
            motion.observe()
    record = {
        "issue": 637,
        "surface": b33.ARM,
        "seed": seed,
        "learn": learn,
        "motion": motion.read(),
        "read_window": len(motion.buffer),
        "horizon_note": "stamped per-RUN, not per-arm: B38 (#599) measured the "
                        "stall horizon varying 13x between seeds of one arm. "
                        "detectability holds the world during the read, so a "
                        "stalled body does not invalidate the reading; what it "
                        "bounds is what may be said about the *trained* surface.",
        "minutes": (time.time() - started) / 60.0,
    }
    out.write_text(json.dumps(record, indent=1), encoding="utf-8")
    torch.save(
        {
            "state": ufp.snapshot(agent.sheaf),
            "maps": agent.sheaf.maps.maps.detach().clone(),
            "operators": {k: v.detach().clone()
                          for k, v in agent.sheaf.operators.state_dict().items()},
            "biases": {k: v.detach().clone()
                       for k, v in agent.sheaf.biases.state_dict().items()},
        },
        _HERE / f"637-trained-seed{seed}-{learn}.pt",
    )
    print(f"[B63] trained in {record['minutes']:.1f} min -> {out.name}", flush=True)
    return record


# -- analysis -----------------------------------------------------------------


def profile(record: dict, permutations: int, seed: int) -> dict:
    """The reading: per stratum, per feature, per `k'`, on the configuration ladder."""
    rng = np.random.default_rng(seed + 777)
    trials = record["trials"]
    kinds = sorted({t["kind"] for t in trials})
    out: dict = {"arm": record["arm"], "strata": {}}
    for kind in kinds:
        rows = [t for t in trials if t["kind"] == kind]
        block: dict = {"alphabet": record["alphabet"][kind], "readings": {}}
        for feature in ("peak", "trace"):
            for kk in K_LADDER:
                sub = [t for t in rows if t["label"] < kk]
                for gate in (False, True):
                    use = [t for t in sub if (not gate) or t["clears_f32_floor"]]
                    if len(use) < 2 * kk:
                        continue
                    x = np.array([t[feature] for t in use], dtype=float)
                    y = np.array([t["label"] for t in use], dtype=int)
                    g = np.array([t["config"] for t in use], dtype=int)
                    for how in ("centroid", "nn1"):
                        key = f"{feature}|k{kk}|{'gated' if gate else 'all'}|{how}"
                        block["readings"][key] = estimate(
                            x, y, g, kk, how, permutations, rng
                        )
        # The configuration ladder, on the headline reading only.
        ladder = {}
        for c in CONFIG_LADDER:
            use = [t for t in rows if t["label"] < K and t["config"] < c]
            if len(use) < 2 * K:
                continue
            x = np.array([t["peak"] for t in use], dtype=float)
            y = np.array([t["label"] for t in use], dtype=int)
            g = np.array([t["config"] for t in use], dtype=int)
            ladder[str(c)] = estimate(x, y, g, K, "centroid", permutations, rng)
        block["config_ladder"] = ladder
        out["strata"][kind] = block
    return out


def analyse(seed: int, learn: int, permutations: int) -> dict:
    combined = {"issue": 637, "seed": seed, "learn": learn,
                "traditional_or_cohomological": "traditional", "arms": {}}
    for arm in ARMS:
        path = _HERE / f"637-{arm}-seed{seed}-{learn}.json"
        if not path.exists():
            print(f"  (no {path.name})")
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        combined["arms"][arm] = {
            "gate_224": record["gate_224"],
            "determinism": record["determinism"],
            "cycle_surface": record["cycle_surface"],
            "terminus": record["terminus"],
            "profile": profile(record, permutations, seed),
        }
        print(f"  read {path.name}")
    out = _HERE / "637-analysis.json"
    out.write_text(json.dumps(combined, indent=1), encoding="utf-8")
    print(f"[B63] -> {out.name}")
    table(combined)
    return combined


def table(combined: dict) -> None:
    print()
    print("I(P; Delta) lower bound, bits -- traditional Shannon, situation marginalised")
    print("peak feature, k=8, all trials, nearest-centroid LOO; ceiling log2 8 = 3.000")
    print()
    head = f"{'arm':<11}{'stratum':<16}{'dim':>7}{'acc':>7}{'MI':>8}{'null':>8}{'excess':>9}{'p':>8}"
    print(head)
    print("-" * len(head))
    for arm, blob in combined["arms"].items():
        for kind, block in blob["profile"]["strata"].items():
            r = block["readings"].get("peak|k8|all|centroid")
            if not r or r.get("insufficient"):
                continue
            print(
                f"{arm:<11}{kind:<16}{block['alphabet']['dim']:>7}"
                f"{r['accuracy']:>7.3f}{r['mi_bits']:>8.3f}"
                f"{r['null_mean_bits']:>8.3f}{r['excess_bits']:>9.3f}"
                f"{r['p_value']:>8.4f}"
            )
    print()


def main() -> None:
    p = argparse.ArgumentParser(description="B63 (#637): the dependence instrument")
    p.add_argument("command", choices=["train", "read", "analyse"])
    p.add_argument("--arm", default="trained", choices=list(ARMS))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--learn", type=int, default=LEARN)
    p.add_argument("--configs", type=int, default=CONFIGS)
    p.add_argument("--k", type=int, default=K)
    p.add_argument("--window", type=int, default=det.WINDOW)
    p.add_argument("--hold", type=int, default=det.HOLD)
    p.add_argument("--permutations", type=int, default=PERMUTATIONS)
    p.add_argument("--lanes", action="store_true")
    args = p.parse_args()

    if args.command == "train":
        train(args.seed, args.learn, _HERE / f"637-train-seed{args.seed}-{args.learn}.json")
    elif args.command == "read":
        run_arm(
            args.arm, args.seed, args.learn, args.configs, args.k, args.window,
            args.hold, _HERE / f"637-{args.arm}-seed{args.seed}-{args.learn}.json",
            args.lanes,
        )
    else:
        analyse(args.seed, args.learn, args.permutations)


if __name__ == "__main__":
    main()
