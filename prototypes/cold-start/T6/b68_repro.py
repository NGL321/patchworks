"""B68 (#645): is the transport arm reproducible run to run?

Not a question this ticket went looking for. `#645`'s item 1 re-ran B62's
`transport` arm on B62's own rig with `--no-generic`, and it **did not
reproduce B62's stored record**: at 5,000 ticks the uncentered traffic ER reads
3.0024 here against **3.1347** in `635-transport-baseline-seed42-5000.json`,
and the two arms have already parted at tick 100 (2.8266 / 0.5324 / 8.37
against 2.8109 / 0.5328 / 8.26). `git log f51667d..84a97d9 -- src/
prototypes/cold-start/T0 prototypes/cold-start/T3` is **empty**, so the code
driving the trajectory is byte-identical across the two surfaces, and the seed,
the condition, the mode and the checkpoint ladder's prefix all match.

Two candidates, and they have different consequences:

* **The matched-generic null perturbs the run.** `against_generic` builds
  `RestrictionMaps(dome, generator=torch.Generator().manual_seed(...))` and
  calls `project()` on it. Both take an explicit generator and neither reads
  the global RNG on inspection -- but *constructing a `RestrictionMaps` against
  the live `dome`* is the one thing in that path that touches shared state, and
  inspection is not a measurement. If this is it, `--no-generic` changes the
  trajectory and no `--no-generic` arm may be quoted against a stored one.
* **The arm is simply not deterministic.** If it is this, B62's attribution --
  and every arm-to-arm comparison on this map that is not seed-paired within
  one process -- is resting on a difference that a re-run could produce by
  itself, and that is a far more consequential finding than #645's own question.

Three runs settle it, all short, because the arms part by tick 100:

* ``A`` and ``B`` -- identical invocations, no null. Any difference is
  non-determinism.
* ``C`` -- the same, with the null taken at construction. A difference from
  ``A``/``B`` when those two agree is the null perturbing the run.

Reported on the three columns the parting shows up in, at both rungs.

Usage::

    PYTHONPATH=src python prototypes/cold-start/T6/b68_repro.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import torch

torch.set_num_threads(1)

_HERE = Path(__file__).resolve().parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


b57 = _load("b68r_b57", _HERE / "b57_agreement.py")
b62 = _load("b68r_b62", _HERE / "b62_frozen.py")

from patchworks.learning import TransportRule  # noqa: E402

RUNGS = (100, 200)


def one(label: str, seed: int, ticks: int, *, generic: bool) -> dict:
    env, agent, arm, _flat = b57.build_arm("baseline", seed)
    try:
        layout = b57.Layout(agent.dome)
        chains = b57.t2.rim_chains(agent.dome)
        traffic = b57.Traffic(agent, b62.BUFFER)
        recorder = b57.t2.EdgeRecorder(agent)
        transport = TransportRule(agent.sheaf)

        b62.step(agent, b62.BUFFER, seed, recorder, None, None, traffic)
        rows = []
        if generic:
            # Exactly where `b62_frozen.run` takes it: at construction, before
            # the first learning tick, so a perturbation here reaches everything.
            b57.against_generic(agent, layout, traffic.matrix(b57.WINDOW), seed)
        seen = 0
        for target in RUNGS:
            if target > ticks:
                break
            b62.step(agent, target - seen, seed + seen, recorder, None, transport, traffic)
            seen = target
            entry = b57.joint(
                agent, layout, chains, traffic.matrix(b57.WINDOW), f"{label} @{target}", seed=seed
            )
            rows.append(
                {
                    "ticks": target,
                    "er": entry["agreement"]["traffic_effective_rank"],
                    "auddiff": entry["audience_differentiation"]["median"],
                    "exposure": entry["exposure"]["effective_median"],
                }
            )
            print(
                f"  {label:<4} @{target:<6} ER {rows[-1]['er']:.6f}  "
                f"aud-diff {rows[-1]['auddiff']:.6f}  exposure {rows[-1]['exposure']:.6f}",
                flush=True,
            )
    finally:
        env.close()
    return {"label": label, "generic": generic, "rows": rows}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ticks", type=int, default=200)
    p.add_argument("--out", type=Path, default=_HERE / "645-repro.json")
    args = p.parse_args()

    record = {
        "issue": 645,
        "reading": "is the transport arm reproducible run to run, and does the null perturb it",
        "seed": args.seed,
        "ticks": args.ticks,
        "surface": b57.t0.surface(),
        "runs": [],
    }
    for label, generic in (("A", False), ("B", False), ("C", True)):
        record["runs"].append(one(label, args.seed, args.ticks, generic=generic))
        args.out.write_text(json.dumps(record, indent=1))

    a, b, c = record["runs"]

    def at(run, t):
        return next(r for r in run["rows"] if r["ticks"] == t)

    print("\n=== verdict ===")
    for t in RUNGS:
        ra, rb, rc = at(a, t), at(b, t), at(c, t)
        same_ab = all(abs(ra[k] - rb[k]) < 1e-12 for k in ("er", "auddiff", "exposure"))
        same_ac = all(abs(ra[k] - rc[k]) < 1e-12 for k in ("er", "auddiff", "exposure"))
        print(
            f"  @{t}: A==B {same_ab}   A==C {same_ac}   "
            f"|dER| A-B {abs(ra['er'] - rb['er']):.3e}  A-C {abs(ra['er'] - rc['er']):.3e}"
        )
    print(f"\nwritten to {args.out.name}")


if __name__ == "__main__":
    main()
