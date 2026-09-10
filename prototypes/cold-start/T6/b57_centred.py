"""B57 (#629): the centred half of the record, beside B52's uncentered form.

B53's advisory on #629: `C = (1/T) sum_t x_t x_t^T` is a second moment about
zero, and a standing baseline `mu` puts a rank-one `mu mu^T` into it. This
prints the two side by side on every row so neither is quoted alone. Nothing is
computed here -- `b57_agreement.py` writes both.

Usage::

    python prototypes/cold-start/T6/b57_centred.py 629-baseline-seed42-20000.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

THETAS = ("0.05", "0.1", "0.25")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("records", type=Path, nargs="+")
    args = p.parse_args()
    for path in args.records:
        r = json.loads(path.read_text())
        print(f"== {r['condition']} seed {r['seed']} -- uncentered | centred ==")
        rows = [("construction", r["at_construction"])] + [
            (str(c["ticks"]), c) for c in r["checkpoints"]
        ]
        print(
            f"{'ticks':<14} {'ER':>17}  {'q_w':>15}  {'N@0.25':>15}  "
            f"{'N_gen@0.25':>15}  {'excess@0.25':>15}"
        )
        for tag, row in rows:
            u, c = row["agreement"], row["agreement_centred"]
            gu = row["generic"]["uncentred"]["N_mean"]["0.25"]
            gc = row["generic"]["centred"]["N_mean"]["0.25"]
            nu = u["profile"]["0.25"]["N"]
            nc = c["profile"]["0.25"]["N"]
            print(
                f"{tag:<14} {u['traffic_effective_rank']:8.4f}|{c['traffic_effective_rank']:8.4f}  "
                f"{u['q_weighted_mean']:7.4f}|{c['q_weighted_mean']:7.4f}  "
                f"{nu:7.4f}|{nc:7.4f}  {gu:7.4f}|{gc:7.4f}  "
                f"{nu - gu:+7.4f}|{nc - gc:+7.4f}"
            )
        print()
        print("-- the acceptance check, centred --")
        for tag, row in rows:
            c = row["agreement_centred"]
            for name, sc in row.get("scrambles_centred", {}).items():
                print(
                    f"{tag:<14} {name:<10} ER {c['traffic_effective_rank']:8.3f}->"
                    f"{sc['traffic_effective_rank']:8.3f} | q_w "
                    f"{c['q_weighted_mean']:.4f}->{sc['q_weighted_mean']:.4f} | A@1 "
                    f"{c['profile']['1']['A']:.4f}->{sc['profile']['1']['A']:.4f}"
                )
        print()


if __name__ == "__main__":
    main()
