"""B62 (#635), items 1-2: does the *rule* spend the traffic's rank?

`#635 <https://github.com/NGL321/patchworks/issues/635>`_, under
`the map <https://github.com/NGL321/patchworks/issues/532>`_.

`B57 (#629) <https://github.com/NGL321/patchworks/issues/629>`_ measured the
traffic's effective rank falling from 2.766 at construction to 1.0045 at 20k on
the baseline arm -- **while the rule ran** -- and deliberately declined to
attribute it, because no arm was run with the rule off past construction. So
*training spends the rank* and *the rank decays on its own* are not separated.
This is that arm.

**The control is exact and costs nothing to argue.** `T0/run.py`'s
``teaching_read`` is ``run_ticks`` -- `patchworks.agent.run`, the world loop --
with ``recorder.observe()``, ``bias.step()`` and ``transport.step()`` after each
tick. Dropping the two rule steps leaves the same world, the same dome, the same
seed stream and the same stimulus, with nothing learning. Four modes:

* ``frozen`` -- neither rule. Item 1.
* ``bias`` -- :class:`patchworks.learning.PredictionRule` only. Item 2.
* ``transport`` -- :class:`patchworks.learning.TransportRule` only. Item 2.
* ``both`` -- the control, which is B57's own baseline arm; runnable here so the
  comparison can be made inside one file if the inherited record is doubted.

**The window is swept at the horizon, not only at construction.** B57's own
amendment found centred traffic ER growing monotonically with `T` (13.014 /
18.465 / 24.912 at `T` = 250 / 500 / 1000) while the uncentered form is stable
to the fourth decimal, so *a centred rank quoted at one window means nothing on
its own*. The buffer therefore holds 2,000 ticks and every checkpoint is read at
`T` = 250 / 500 / 1,000 / 2,000. **The headline row is `T = 1,000`** -- T0's own
window, and exactly what B57 read -- so the arms are comparable row for row.

**24.9 is not a target.** Per #635: the comparison is stated in terms that
survive the window sweep, which means *the uncentered form for magnitude* and
*the centred form for direction, at a stated window*.

Usage::

    PYTHONPATH=src python prototypes/cold-start/T6/b62_frozen.py --mode frozen
    PYTHONPATH=src python prototypes/cold-start/T6/b62_frozen.py --mode bias
    PYTHONPATH=src python prototypes/cold-start/T6/b62_frozen.py --mode transport
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import torch

torch.set_num_threads(1)

_HERE = Path(__file__).resolve().parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


b57 = _load("b57_agreement", _HERE / "b57_agreement.py")

from patchworks.learning import PredictionRule, TransportRule  # noqa: E402

#: The buffer, so the horizon can be read at four windows. The *headline* window
#: stays `b57.WINDOW` -- T0's own -- and every arm is compared there.
BUFFER = 2_000
#: The window sweep, at construction and at every checkpoint.
WINDOWS = (250, 500, 1_000, 2_000)


def step(agent, ticks: int, seed: int, recorder, bias, transport, traffic) -> None:
    """`teaching_read`'s loop with the rule steps under a switch.

    Written out rather than reusing :func:`b57_agreement.collect` because the
    point of this file is the two rules *separately*, and a boolean `learn`
    cannot say which one ran. The tick, the recorder and the seed stream are
    `T0/run.py`'s, untouched.
    """
    for _ in b57.t0.run_ticks(agent, ticks, seed=seed):
        recorder.observe()
        if bias is not None:
            bias.step()
        if transport is not None and agent.sheaf.ticks > 1:
            transport.step()
        traffic.observe()


def sweep(agent, layout, chains, traffic, label: str, seed: int) -> dict:
    """One checkpoint: the headline row at `T = 1,000`, plus the window sweep.

    The headline is a full :func:`b57_agreement.joint` row -- agreement,
    audience differentiation, exposure, and the acceptance scramble -- because
    B48's joint rule is not optional here either (#635 item 4). The sweep is the
    agreement statistic alone at the other three windows: differentiation and
    exposure read the maps, not the traffic, so they do not move with `T`.
    """
    maps = agent.sheaf.maps
    delta, gram = layout.delta(maps), layout.gram(maps)
    entry = b57.joint(agent, layout, chains, traffic.matrix(b57.WINDOW), label, seed=seed)
    entry["window_sweep"] = {
        str(w): b57.both(traffic.matrix(w), delta, gram)
        for w in WINDOWS
        if len(traffic.buffer) >= w
    }
    return entry


def _sweep_line(entry: dict) -> str:
    parts = []
    for w, reading in entry["window_sweep"].items():
        parts.append(
            f"T={w}: {reading['uncentred']['traffic_effective_rank']:7.4f}"
            f"/{reading['centred']['traffic_effective_rank']:8.4f}"
        )
    return "      window sweep (ER unc/cen)  " + "  ".join(parts)


def run(mode: str, condition: str, seed: int, ticks: int, out: Path, *, generic: bool = True) -> None:
    started = time.time()
    env, agent, arm, flat_info = b57.build_arm(condition, seed)
    try:
        layout = b57.Layout(agent.dome)
        chains = b57.t2.rim_chains(agent.dome)
        record = {
            "issue": 635,
            "item": 1 if mode == "frozen" else 2,
            "stage": "rules",
            "mode": mode,
            "condition": condition,
            "arm": {"rho1_drive_edges": bool(arm["pin"]), "c_learning_rate": float(arm["c"])},
            "flat_bundle": flat_info,
            "seed": seed,
            "ticks": ticks,
            "window": b57.WINDOW,
            "buffer": BUFFER,
            "windows": list(WINDOWS),
            "surface": b57.t0.surface(),
            "delta_check": b57.check_delta(agent, layout),
            "matched_generic_null": generic,
            "checkpoints": [],
        }
        print(f"[B62] mode={mode} delta check: {record['delta_check']}", flush=True)

        traffic = b57.Traffic(agent, BUFFER)
        recorder = b57.t2.EdgeRecorder(agent)
        bias = PredictionRule(agent.sheaf, operator_rate_ratio=arm["c"]) if mode in ("bias", "both") else None
        transport = TransportRule(agent.sheaf) if mode in ("transport", "both") else None

        # At construction: the buffer filled with no rule running in any mode,
        # so every arm starts from the same reading and B49's construction /
        # trained line is paid rather than inherited.
        step(agent, BUFFER, seed, recorder, None, None, traffic)
        entry = sweep(agent, layout, chains, traffic, f"{mode} s{seed} @construction", seed)
        if generic:
            entry["generic"] = b57.against_generic(
                agent, layout, traffic.matrix(b57.WINDOW), seed
            )
        entry["ticks"] = 0
        record["at_construction"] = entry
        print(b57._line(f"{mode} @0", entry), flush=True)
        print(_sweep_line(entry), flush=True)
        out.write_text(json.dumps(record, indent=1))

        ladder = [cp for cp in b57.t3.CHECKPOINTS if cp <= ticks]
        if ticks not in ladder:
            ladder.append(ticks)
        seen = 0
        for target in ladder:
            step(agent, target - seen, seed + seen, recorder, bias, transport, traffic)
            seen = target
            entry = sweep(agent, layout, chains, traffic, f"{mode} s{seed} @{target}", seed)
            if generic:
                entry["generic"] = b57.against_generic(
                    agent, layout, traffic.matrix(b57.WINDOW), seed
                )
            entry["ticks"] = target
            entry["elapsed_minutes"] = (time.time() - started) / 60.0
            record["checkpoints"].append(entry)
            out.write_text(json.dumps(record, indent=1))
            print(
                b57._line(f"{mode} @{target}", entry)
                + f" ({entry['elapsed_minutes']:.1f} min)",
                flush=True,
            )
            print(_sweep_line(entry), flush=True)
    finally:
        env.close()
    print(f"[B62] wrote {out.name}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("frozen", "bias", "transport", "both"), required=True)
    p.add_argument("--condition", default="baseline", choices=("baseline", "winner", "flat"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ticks", type=int, default=20_000)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument(
        "--no-generic",
        action="store_true",
        help="skip the matched-generic null. It re-assembles three full delta/G "
        "pairs per reading and is this file's dominant memory cost; B57 already "
        "established N_gen = 0.0000 at every level, arm, checkpoint and redraw, "
        "and items 1-2 ask about the rank and q_w rather than the excess.",
    )
    args = p.parse_args()
    out = args.out or _HERE / f"635-{args.mode}-{args.condition}-seed{args.seed}-{args.ticks}.json"
    run(args.mode, args.condition, args.seed, args.ticks, out, generic=not args.no_generic)


if __name__ == "__main__":
    main()
