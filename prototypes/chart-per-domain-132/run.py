"""#132's dome term: the chart read taken on the post-floor channel, at 100k.

Wraps `prototypes/chart-double-duty-166/read.py` -- #166's instrument, which is
what [#132](https://github.com/NGL321/patchworks/issues/132) §1 names -- and runs
it twice per seed:

* **post-floor**, the surface ADR-0032 decided and PR #434 builds, where
  `RestrictionMaps.project()` runs mask, band, `_flatten`, `_push_apart`, re-cap;
* **pre-floor**, the control, with `_flatten` monkeypatched to a no-op, which is
  byte-for-byte `main`'s ordering.

The control exists because #132 §2's whole objection is that a `K` spectrum read
through a near-rank-1 channel is *a read of the collapse, not of the domain*. That
objection is testable rather than assumable: if the two arms agree, the chart read
was never channel-limited and §2's gate, though correctly raised, does not bind
this particular statistic.

Horizon is 100k, not 30k: ADR-0032's second pre-registration, and #178's three
repetitions of the 30k mistake.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_RIG = Path(__file__).resolve().parents[1] / "chart-double-duty-166"
sys.path.insert(0, str(_RIG))

from patchworks.restriction import RestrictionMaps  # noqa: E402
import read as rig  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ticks", type=int, default=100_000)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--arms", nargs="+", default=["postfloor", "prefloor"])
    p.add_argument("--dome", default="real")
    p.add_argument("--split", default="train")
    args = p.parse_args()

    live = RestrictionMaps._flatten

    for arm in args.arms:
        RestrictionMaps._flatten = (
            live if arm == "postfloor" else (lambda self: None)
        )
        for seed in args.seeds:
            print(f"[{arm}] seed {seed}, {args.ticks} ticks", flush=True)
            record = rig.run_seed(args.dome, args.split, seed, args.ticks)
            record["arm"] = arm
            out = Path(__file__).parent / (
                f"132-{arm}-{args.dome}-{args.split}"
                f"-seed{seed}-{args.ticks}.json"
            )
            out.write_text(json.dumps(record, indent=1))
            print(f"  wrote {out.name}", flush=True)


if __name__ == "__main__":
    main()
