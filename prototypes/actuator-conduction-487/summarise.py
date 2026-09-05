"""Roll the #487 arm/seed/horizon JSONs into the table the ticket asks for.

    python prototypes/actuator-conduction-487/summarise.py

Prints one row per (seed, horizon) with both arms side by side and the ratio
between them, then the `M` spectra that explain it. Reads whatever JSONs are
present, so it is worth running while the sweep is still going.
"""

from __future__ import annotations

import json
import pathlib

HERE = pathlib.Path(__file__).parent


def load() -> dict[tuple[str, int, int], dict]:
    rows: dict[tuple[str, int, int], dict] = {}
    for path in sorted(HERE.glob("487-*.json")):
        blob = json.loads(path.read_text())
        for row in blob.get("checkpoints", []):
            rows[(row["arm"], row["seed"], row["ticks"])] = row
    return rows


def main() -> None:
    rows = load()
    if not rows:
        print("no readings yet")
        return
    seeds = sorted({k[1] for k in rows})
    horizons = sorted({k[2] for k in rows})
    divisor = next(iter(rows.values()))["world_loop"]

    print(f"world_loop(actuator) = {divisor}\n")
    head = (
        f"{'seed':>4} {'ticks':>7} | {'gain pre':>8} {'tau pre':>8} {'ratio pre':>9}"
        f" | {'gain post':>9} {'tau post':>9} {'ratio post':>10} | {'tau x':>6} {'verdict':>16}"
    )
    print(head)
    print("-" * len(head))
    for seed in seeds:
        for ticks in horizons:
            pre = rows.get(("preclamp", seed, ticks))
            post = rows.get(("clamped", seed, ticks))
            if not (pre and post):
                continue
            factor = post["tau_closed"] / pre["tau_closed"]
            both = min(pre["conduction_ratio_closed"], post["conduction_ratio_closed"])
            verdict = "both pass" if both >= 1 else "short"
            print(
                f"{seed:>4} {ticks:>7} | {pre['gain']:>8.4f} {pre['tau_closed']:>8.2f}"
                f" {pre['conduction_ratio_closed']:>9.3f} | {post['gain']:>9.4f}"
                f" {post['tau_closed']:>9.2f} {post['conduction_ratio_closed']:>10.3f}"
                f" | {factor:>6.3f} {verdict:>16}"
            )

    print("\nM = Sum_e F^T F at the actuator (trace is pinned to deg(v) = 3 by the exact gauge)")
    print(f"{'arm':>9} {'seed':>4} {'ticks':>7} {'trace':>7} {'lam_min':>9} {'lam_max':>9} {'rho_cmd':>8} {'identity':>10}")
    for key in sorted(rows):
        r = rows[key]
        print(
            f"{r['arm']:>9} {r['seed']:>4} {r['ticks']:>7} {r['M_trace']:>7.4f}"
            f" {r['M_lambda_min']:>9.5f} {r['M_lambda_max']:>9.5f}"
            f" {r['rho_A_commanded']:>8.5f}"
            f" {r['operator_identity']['relative_residual_max']:>10.1e}"
        )

    print("\nThe three L1 predicting neighbours: ADR-0026's own tau_hat, on their private block")
    print(f"{'arm':>9} {'seed':>4} {'ticks':>7} " + " ".join(f"{c:>10}" for c in ("327", "329", "331")))
    for key in sorted(rows):
        r = rows[key]
        cells = r.get("neighbours", {})
        vals = " ".join(
            f"{cells[c]['tau_hat_median']:>10.2f}" if c in cells else f"{'-':>10}"
            for c in ("327", "329", "331")
        )
        print(f"{r['arm']:>9} {r['seed']:>4} {r['ticks']:>7} " + vals)

    print("\ntau_stalk (whole-stalk read, arrival-dominated; NOT ADR-0026's tau_hat)")
    for key in sorted(rows):
        r = rows[key]
        print(
            f"{r['arm']:>9} {r['seed']:>4} {r['ticks']:>7} "
            f"tau_stalk={r['tau_stalk_median']:>6.2f}  tau_closed={r['tau_closed']:>7.2f}"
        )


if __name__ == "__main__":
    main()
