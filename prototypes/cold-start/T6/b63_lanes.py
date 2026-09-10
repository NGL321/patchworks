"""B63 (#637), ask 5: the **per-edge** dependence profile, and what it costs.

B43 (#609) §6 retained a per-edge reading as the diagnostic and ruled it a
**lane** object — [B49 (#616)](https://github.com/NGL321/patchworks/issues/616)'s
ban on carrying a number between lane and node-stalk objects applying in full.
The ticket left *whether it runs in the first reading* as a budget call. It runs.

**Which object.** Every number in this file is on the **lane**: the paired
deviation in `Sheaf.disagreement()`, which is the edge's own term of `xᵀLx` and
lives in lane coordinates of width `m_e`. It is **not** comparable with
`b63_dependence.py`'s endpoint reading, which is on **node stalks** — B21's two
instruments differ by roughly two orders per hop and no number crosses.

**Not an amplitude.** B43 §7 bars amplitude from carrying the content, so the
per-edge feature is the lane deviation **vector** at that edge's own peak tick,
normalised — the direction, exactly as the endpoint reading takes it. A first
pass here stored `‖·‖` per edge and that is an amplitude profile wearing the
measure's name; it is discarded rather than reported.

**The cost, which was the open question.** No extra *runs*: the trials are the
same trials, and `detectability.branch` already returns the whole `[ticks,
edges, m]` disagreement trace, so the marginal cost of collecting is one
subtraction and one argmax per trial. What costs is the **null**: a matched
scramble per edge would be `edges x permutations` decodes. It is made affordable
by the design being **balanced** — exactly one trial per `(C, pattern)` — which
makes the leave-one-out centroid decode a closed form that vectorises across all
`edges` at once, so one permutation costs one pass over `[E, N, m]` rather than
`E` passes over `[N, m]`.

    PYTHONPATH=src python prototypes/cold-start/T6/b63_lanes.py collect
    PYTHONPATH=src python prototypes/cold-start/T6/b63_lanes.py estimate
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)

_HERE = Path(__file__).resolve().parent
b63 = __import__("importlib").import_module("b63_dependence") if str(_HERE) in sys.path else None
if b63 is None:
    sys.path.insert(0, str(_HERE))
    import b63_dependence as b63  # noqa: E402

import detectability as det  # noqa: E402
import untrained_fixed_point as ufp  # noqa: E402

PERMUTATIONS = 50


def collect(arm: str, seed: int, learn: int, configs: int, k: int, window: int,
            hold: int) -> None:
    """The same trials, keeping each edge's lane deviation direction."""
    started = time.time()
    env, agent = b63.b33.arms_mod.build_arm(b63.b33.ARM, seed)
    dome = agent.dome
    blob = torch.load(_HERE / f"637-trained-seed{seed}-{learn}.pt", weights_only=False)
    if arm != "untrained":
        agent.sheaf.operators.load_state_dict(blob["operators"])
        agent.sheaf.biases.load_state_dict(blob["biases"])
    if arm == "trained":
        with torch.no_grad():
            agent.sheaf.maps.maps.copy_(blob["maps"].to(agent.sheaf.maps.maps.dtype))
    elif arm == "flat":
        b63.b44.install_frames(dome, agent.sheaf.maps,
                               b63.b44.reserved_frames(dome, seed))
    det.double_precision(agent.sheaf)

    strata = det.rim_strata(dome)
    kinds = [dome.cells[s[0]].kind.value for s in strata]
    books = {kd: b63.stratum_alphabet(dome, tuple(c), k, seed)
             for kd, c in zip(kinds, strata)}
    widths = np.array([int(e.m) for e in dome.edges])

    feats, labels, groups, kindcol = [], [], [], []
    for ci in range(configs):
        observation, _ = env.reset(seed=seed * 1000 + ci)
        agent.observe(observation)
        applied = np.zeros(env.action_space.shape, dtype=np.float64)
        det.hold_still(agent, observation, applied, hold)
        state = ufp.snapshot(agent.sheaf)
        quiet, _ = det.branch(agent, state, observation, applied, window, None)
        for kd in kinds:
            for label, nudge in enumerate(books[kd]["patterns"]):
                moved, _ = det.branch(agent, state, observation, applied, window, nudge)
                dev = (moved - quiet).numpy()               # [ticks, edges, m]
                norms = np.linalg.norm(dev, axis=-1)        # [ticks, edges]
                at = np.argmax(norms, axis=0)               # [edges]
                idx = np.arange(dev.shape[1])
                vec = dev[at, idx, :]                       # [edges, m]
                scale = np.linalg.norm(vec, axis=-1, keepdims=True)
                feats.append((vec / np.maximum(scale, 1e-300)).astype(np.float32))
                labels.append(label)
                groups.append(ci)
                kindcol.append(kd)
        ufp.restore(agent.sheaf, state)
        print(f"  config {ci + 1}/{configs}: {len(feats)} trials, "
              f"{(time.time() - started) / 60:.1f} min", flush=True)

    np.savez_compressed(
        _HERE / f"637-lanes-{arm}-seed{seed}-{learn}.npz",
        features=np.stack(feats), labels=np.array(labels),
        groups=np.array(groups), kinds=np.array(kindcol),
        edge_m=widths,
        edge_name=np.array([det.name_edge(dome, i) for i in range(len(dome.edges))]),
    )
    print(f"[B63/lane] {arm}: {len(feats)} trials, {len(dome.edges)} edges, "
          f"{(time.time() - started) / 60:.1f} min", flush=True)


def _mi_rows(table: np.ndarray) -> np.ndarray:
    """Plug-in MI in bits for a stack of `[E, k, k]` count tables."""
    n = table.sum(axis=(1, 2), keepdims=True)
    p = table / np.maximum(n, 1)
    px = p.sum(axis=2, keepdims=True)
    py = p.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(p > 0, p * np.log2(p / np.maximum(px * py, 1e-300)), 0.0)
    return t.sum(axis=(1, 2))


def _decode_all(x: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """LOO nearest-centroid across every edge at once. `x` is `[E, N, m]`.

    The design is balanced — exactly `N/k` trials per label — so the held-out
    centroid is a closed form and the whole edge axis moves together.
    """
    E, N, _ = x.shape
    per = N // k
    sums = np.zeros((E, k, x.shape[2]), dtype=np.float64)
    for j in range(k):
        sums[:, j] = x[:, labels == j].sum(axis=1)
    table = np.zeros((E, k, k))
    for i in range(N):
        li = labels[i]
        cent = sums / per
        cent[:, li] = (sums[:, li] - x[:, i]) / max(per - 1, 1)
        nrm = np.linalg.norm(cent, axis=2)
        score = np.einsum("ekm,em->ek", cent, x[:, i]) / np.maximum(nrm, 1e-300)
        pred = np.argmax(score, axis=1)
        table[np.arange(E), li, pred] += 1
    return table


def estimate(arm: str, seed: int, learn: int, permutations: int) -> None:
    blob = np.load(_HERE / f"637-lanes-{arm}-seed{seed}-{learn}.npz",
                   allow_pickle=False)
    feats, labels = blob["features"], blob["labels"]
    groups, kinds = blob["groups"], blob["kinds"].astype(str)
    edge_m, edge_name = blob["edge_m"], blob["edge_name"].astype(str)
    k = int(labels.max()) + 1
    rng = np.random.default_rng(seed + 63)
    out = {"issue": 637, "object": "lane", "arm": arm, "seed": seed,
           "traditional_or_cohomological": "traditional",
           "b49_note": "lane object; no number here is comparable with the "
                       "endpoint reading, which is on node stalks",
           "k": k, "permutations": permutations, "strata": {}}
    for kd in sorted(set(kinds)):
        sel = kinds == kd
        x = np.transpose(feats[sel].astype(np.float64), (1, 0, 2))  # [E, N, m]
        y, g = labels[sel], groups[sel]
        started = time.time()
        mi = _mi_rows(_decode_all(x, y, k))
        null = np.empty((permutations, x.shape[0]))
        for p in range(permutations):
            sh = y.copy()
            for gg in np.unique(g):
                idx = np.flatnonzero(g == gg)
                sh[idx] = rng.permutation(y[idx])
            null[p] = _mi_rows(_decode_all(x, sh, k))
        excess = mi - null.mean(axis=0)
        pval = (np.sum(null >= mi[None, :], axis=0) + 1) / (permutations + 1)
        live = edge_m > 0
        order = np.argsort(-excess)
        out["strata"][kd] = {
            "edges": int(x.shape[0]),
            "carried_edges": int(live.sum()),
            "ceiling_bits": float(np.log2(k)),
            "mi_median_bits": float(np.median(mi[live])),
            "mi_max_bits": float(mi[live].max()),
            "null_median_bits": float(np.median(null.mean(axis=0)[live])),
            "excess_median_bits": float(np.median(excess[live])),
            "edges_significant_p05": int((pval[live] <= 0.05).sum()),
            "top": [
                {"edge": edge_name[i], "m": int(edge_m[i]),
                 "mi_bits": float(mi[i]), "excess_bits": float(excess[i]),
                 "p": float(pval[i])}
                for i in order[:10]
            ],
            "minutes": (time.time() - started) / 60.0,
        }
        s = out["strata"][kd]
        print(f"  {kd:<16} median {s['mi_median_bits']:.3f} bits "
              f"(null {s['null_median_bits']:.3f}, excess "
              f"{s['excess_median_bits']:.3f}), max {s['mi_max_bits']:.3f}, "
              f"{s['edges_significant_p05']}/{s['carried_edges']} edges at p<=0.05, "
              f"{s['minutes']:.1f} min", flush=True)
    path = _HERE / f"637-lanes-{arm}.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"[B63/lane] -> {path.name}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["collect", "estimate"])
    p.add_argument("--arm", default="trained", choices=list(b63.ARMS))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--learn", type=int, default=b63.LEARN)
    p.add_argument("--configs", type=int, default=b63.CONFIGS)
    p.add_argument("--k", type=int, default=b63.K)
    p.add_argument("--window", type=int, default=det.WINDOW)
    p.add_argument("--hold", type=int, default=det.HOLD)
    p.add_argument("--permutations", type=int, default=PERMUTATIONS)
    a = p.parse_args()
    if a.command == "collect":
        collect(a.arm, a.seed, a.learn, a.configs, a.k, a.window, a.hold)
    else:
        estimate(a.arm, a.seed, a.learn, a.permutations)


if __name__ == "__main__":
    main()
