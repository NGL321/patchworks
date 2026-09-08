"""T6 (#564 / B16): the collapse is in the junctions between hops, not in the angles.

`b16_saturation.py` tried the obvious mechanism — training drives each hop's
leading principal cosine to 1 — and it **fails at `p = 8`**, predicting a ratio
of 0.970 against B13's measured 0.629. The existing 100k runs say why: across
training at `p = 8`, `cos_product_er` (the composed ER the per-hop cosine spectra
alone predict) is **flat at ~3.0** while `composed_er` falls 2.29 -> 1.42.
The angles barely move. `angles.py` names the residue exactly:

    "The gap between them is exactly what the `U`-rotations carry."

**So the drive lives in the `U`-rotations, and that has a mechanism.** A hop is
`U_out (V_outᵀ V_in) U_inᵀ`, mapping edge `e_k`'s lane to edge `e_{k+1}`'s. Take
two consecutive hops in a chain. Hop `k`'s `U_out` and hop `k+1`'s `U_in` are
orthogonal bases of **the same lane, `e_{k+1}`** — held by the two cells at that
edge's two ends. And the transport objective is, in its own words,
`‖F_v x_v − y_e‖ / (‖F_v x_v‖ + ‖y_e‖)`: *make what this end sends match what the
other end sent.* **Training's whole purpose is to bring those two bases into
agreement.**

That is the compounding. Each hop contracts along its lower principal
directions. If consecutive hops' retained directions are incoherent, seven
contractions point seven different ways and the product stays spread — which is
what the generic instrument computes. Bring them into agreement and the seven
contractions align, so they compound multiplicatively onto one direction. The
collapse is not a pathology the rule suffers; **it is the rule's objective being
met.**

Two readouts, per checkpoint:

* ``junction_lead`` — `|<u_out_1(k), u_in_1(k+1)>|`, the coherence of the
  retained direction across each junction. The prediction is that it rises with
  training at `p = 8` and has nothing to do at `p = 16`, where the cosine
  matrices are already near-identity (0.9994 / 0.9913) and each hop is close to
  an isometry that cannot concentrate anything.
* ``scrambled_er`` — **the counterfactual.** Recompose each chain with a random
  orthogonal `R_k` inserted at every junction, leaving every hop exactly as
  trained. That destroys junction coherence and changes nothing else. If the
  drive is the junctions, `scrambled_er` stays near `cos_product_er` while
  `composed_er` collapses beneath it.

Usage::

    PYTHONPATH=src python prototypes/cold-start/T6/b16_junction.py --arms reserve reserve_p16 --ticks 20000
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
_T0, _T2, _T4 = (_HERE.parent / n for n in ("T0", "T2", "T4"))
for extra in (_ROOT / "prototypes" / "chart-double-duty-166", _ROOT / "benchmarks", _T0):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


t0 = _load("t0_run", _T0 / "run.py")
t2 = _load("t2_run", _T2 / "run.py")
angles = _load("t4_angles", _T4 / "angles.py")
arms_mod = _load("t6_arms", _HERE / "arms.py")
departure = _load("t6_departure", _HERE / "departure.py")

import construction_grading as cg  # noqa: E402
from patchworks.learning import PredictionRule, TransportRule  # noqa: E402

CHECKPOINTS = [100, 300, 1_000, 3_000, 10_000, 20_000, 50_000, 100_000]


@torch.no_grad()
def chain_junction_read(dome, maps, chain: dict, rng: np.random.Generator, n_scramble: int) -> dict:
    """One chain: its true composed ER, the junction coherences, and the counterfactual."""
    hops = cg.hops_of(dome, tuple(chain["edges"]))
    parts = []
    for edge_in, cell, edge_out in hops:
        u_in, v_in, s_in = angles.carried(dome, maps, edge_in, cell)
        u_out, v_out, s_out = angles.carried(dome, maps, edge_out, cell)
        # hop = F_out F_inᵀ = U_out diag(s_out) V_outᵀ V_in diag(s_in) U_inᵀ
        mid = np.diag(s_out) @ (v_out.T @ v_in) @ np.diag(s_in)
        parts.append({"u_in": u_in, "u_out": u_out, "mid": mid})

    def compose(scramble: bool) -> np.ndarray:
        composed = None
        for i, pt in enumerate(parts):
            hop = pt["u_out"] @ pt["mid"] @ pt["u_in"].T
            if composed is None:
                composed = hop
                continue
            if scramble:
                # A random orthogonal at the junction: destroys the coherence
                # between hop i-1's retained direction and hop i's, and leaves
                # every hop itself exactly as trained.
                d = composed.shape[0]
                r = np.linalg.qr(rng.standard_normal((d, d)))[0]
                composed = hop @ r @ composed
            else:
                composed = hop @ composed
        return composed

    true_er = angles.effective_rank(np.linalg.svd(compose(False), compute_uv=False))
    scr = [
        angles.effective_rank(np.linalg.svd(compose(True), compute_uv=False))
        for _ in range(n_scramble)
    ]

    # Junction coherence, first attempt: the individual maps' own leading left
    # factors. **This is the wrong pair of vectors** and is kept only because the
    # readout quotes it as flat: what compounds is not `F`'s leading direction
    # but the *hop's*, and the hop's singular basis folds in the `V_outᵀ V_in`
    # Gram and the two `sigma` diagonals as well.
    lead = []
    for a, b in zip(parts, parts[1:]):
        lead.append(float(abs(np.dot(a["u_out"][:, 0], b["u_in"][:, 0]))))

    # Junction coherence, the quantity that actually compounds: hop `k` retains
    # its leading **output** direction `P_k[:, 0]`; hop `k+1` is most
    # transmissive along its leading **input** direction `Q_{k+1}[:, 0]`. Their
    # overlap is what decides whether seven contractions stack onto one
    # direction or point seven different ways.
    hop_lead, hop_lead2 = [], []
    svds = []
    for pt in parts:
        p_, s_, qt_ = np.linalg.svd(pt["u_out"] @ pt["mid"] @ pt["u_in"].T, full_matrices=False)
        svds.append((p_, s_, qt_.T))
    for (p_a, _, _), (_, _, q_b) in zip(svds, svds[1:]):
        hop_lead.append(float(abs(np.dot(p_a[:, 0], q_b[:, 0]))))
        if p_a.shape[1] > 1 and q_b.shape[1] > 1:
            hop_lead2.append(float(abs(np.dot(p_a[:, 1], q_b[:, 1]))))
    return {
        "composed_er": true_er,
        "scrambled_er": float(np.mean(scr)),
        "junction_lead": float(np.mean(lead)),
        "junction_lead_max": float(np.max(lead)),
        "hop_junction_lead": float(np.mean(hop_lead)),
        "hop_junction_lead2": float(np.mean(hop_lead2)) if hop_lead2 else 0.0,
    }


def _q(a) -> dict:
    a = np.asarray(a, dtype=np.float64)
    return {
        "median": float(np.median(a)),
        "p10": float(np.quantile(a, 0.10)),
        "p90": float(np.quantile(a, 0.90)),
        "mean": float(a.mean()),
    }


def surface_read(agent, chains, rng, n_scramble: int) -> dict:
    dome, maps = agent.dome, agent.sheaf.maps
    rows = [chain_junction_read(dome, maps, ch, rng, n_scramble) for ch in chains]
    return {
        "composed_er": _q([r["composed_er"] for r in rows]),
        "scrambled_er": _q([r["scrambled_er"] for r in rows]),
        "junction_lead": _q([r["junction_lead"] for r in rows]),
        "junction_lead_max": _q([r["junction_lead_max"] for r in rows]),
        "hop_junction_lead": _q([r["hop_junction_lead"] for r in rows]),
        "hop_junction_lead2": _q([r["hop_junction_lead2"] for r in rows]),
    }


def run_seed(arm: str, seed: int, ticks: int, n_scramble: int, out: Path) -> dict:
    started = time.time()
    inflight = out.with_suffix(".inflight.json")
    env, agent = arms_mod.build_arm(arm, seed)
    try:
        dome = agent.dome
        chains = t2.rim_chains(dome)
        rng = np.random.default_rng(0)
        _, reserve_p = arms_mod.ARMS[arm]
        record = {
            "issue": 564,
            "arm": arm,
            "reserve_p": reserve_p,
            "seed": seed,
            "ticks": ticks,
            "n_scramble": n_scramble,
            "claim": (
                "the collapse is carried by the coherence of consecutive hops' U "
                "bases on the lane they share, which is exactly what the transport "
                "objective (agreement between an edge's two ends) maximises"
            ),
            "generic": arms_mod.generic_at(dome, chains),
            "checkpoints": [],
        }

        recorder = t2.EdgeRecorder(agent)
        bias = PredictionRule(agent.sheaf)
        transport = TransportRule(agent.sheaf)

        def snapshot(tick: int) -> dict:
            surf = angles.read_surface(agent, chains, f"{arm} s{seed} @{tick}")
            entry = surface_read(agent, chains, rng, n_scramble)
            entry["ticks"] = tick
            entry["cos_product_er"] = surf["cos_product_er"]
            entry["cos_leading_per_hop"] = surf["cos_leading_per_hop"]
            entry["sigma"] = departure.sigma_read(agent)
            entry["elapsed_minutes"] = (time.time() - started) / 60.0
            return entry

        def show(e: dict) -> None:
            print(
                f"  {arm} s{seed} @{e['ticks']:>6}: ER {e['composed_er']['median']:.4f} "
                f"| scrambled {e['scrambled_er']['median']:.4f} "
                f"| cos_product {e['cos_product_er']['median']:.4f} "
                f"| hop_junction {e['hop_junction_lead']['median']:.4f} "
                f"(2nd {e['hop_junction_lead2']['median']:.4f}) "
                f"| junction_lead {e['junction_lead']['median']:.4f} "
                f"| sigma {e['sigma']['sigma_min_over_max_median']:.4f} "
                f"({e['elapsed_minutes']:.1f} min)",
                flush=True,
            )

        e0 = snapshot(0)
        record["checkpoints"].append(e0)
        print(f"  {arm} generic {record['generic']['median']:.4f}", flush=True)
        show(e0)

        ladder = [c for c in CHECKPOINTS if c <= ticks]
        if ticks not in ladder:
            ladder.append(ticks)
        seen = 0
        for target in ladder:
            for _ in t0.run_ticks(agent, target - seen, seed=seed + seen):
                recorder.observe()
                bias.step()
                if agent.sheaf.ticks > 1:
                    transport.step()
            seen = target
            entry = snapshot(target)
            record["checkpoints"].append(entry)
            inflight.write_text(json.dumps(record, indent=1))
            show(entry)
        inflight.replace(out)
        return record
    finally:
        env.close()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arms", nargs="+", default=["reserve", "reserve_p16"])
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--ticks", type=int, default=20_000)
    p.add_argument("--n-scramble", type=int, default=3)
    p.add_argument("--tag", default="", help="suffix, so a re-read with a new measure does not collide")
    args = p.parse_args()
    for arm in args.arms:
        for seed in args.seeds:
            out = _HERE / f"564-junction{args.tag}-{arm}-seed{seed}-{args.ticks}.json"
            if out.exists():
                print(f"[T6] {out.name} already at the horizon, skipping", flush=True)
                continue
            print(f"[T6] B16 junction probe: {arm} seed {seed}, {args.ticks} ticks -> {out.name}", flush=True)
            run_seed(arm, seed, args.ticks, args.n_scramble, out)
            print(f"  wrote {out.name}", flush=True)


if __name__ == "__main__":
    main()
