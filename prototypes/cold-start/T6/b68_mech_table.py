"""B68 (#645) item 2: the mechanism records, four modes side by side.

Prints three tables, because the reading has three separable parts and fusing
them would be exactly the move `B49 (#616)
<https://github.com/NGL321/patchworks/issues/616>`_ struck.

1. **The decomposition.** Global traffic ER beside the median per-cell block ER
   and the synchrony of the blocks' leading temporal profiles. A global ER of 1
   needs both the blocks at ~1 *and* the joint profile ER at ~1; which of the
   two moves is what "the loop samples one direction of it" would have to name.
2. **`K`'s own numbers**, per B44's rig: `rho(K)`, `sigma(K)`, `rho(used)`, the
   non-normality ratio and `tau`. Reported per mode so `PredictionRule`'s effect
   is read against two arms that do not write `K` at all.
3. **The correlations**, across cells, at every checkpoint, with `n` beside every
   coefficient. These are correlations *between* two objects -- the traffic is
   node stalks and `K` is the cell operator -- which is a question about whether
   one tracks the other and never a number carried from one to the other.

Usage::

    python prototypes/cold-start/T6/b68_mech_table.py
"""

from __future__ import annotations

import io
import json
from pathlib import Path

_HERE = Path(__file__).resolve().parent

MODES = ("frozen", "bias", "transport", "both")


def load(mode: str) -> dict | None:
    path = _HERE / f"645-mech-{mode}-baseline-seed42-2000.json"
    if not path.exists():
        return None
    return json.load(io.open(path, encoding="utf-8"))


def entries(record: dict) -> list[dict]:
    return [record["at_construction"], *record["checkpoints"]]


def fmt_rho(node: dict) -> str:
    return "   n/a" if node.get("rho") is None else f"{node['rho']:6.3f}"


def main() -> None:
    records = {m: r for m in MODES if (r := load(m)) is not None}
    if not records:
        print("[B68] no mechanism records present yet")
        return

    print("\n=== 1. the decomposition: is the collapse per cell, or across cells? ===")
    print(
        f"{'mode':>10}{'ticks':>7}{'global ER':>11}{'block ER med':>14}{'block ER p10':>14}"
        f"{'block ER p90':>14}{'sync |cos|':>12}{'joint ER':>10}"
    )
    for mode, record in records.items():
        for e in entries(record):
            b, s = e["per_cell_block_er"], e["synchrony"]
            print(
                f"{mode:>10}{e['ticks']:>7}{e['global']['traffic_effective_rank']:>11.4f}"
                f"{b['median']:>14.4f}{b['p10']:>14.4f}{b['p90']:>14.4f}"
                f"{s['abs_cos_median']:>12.4f}{s['joint_effective_rank']:>10.3f}"
            )

    print("\n=== 2. K's own numbers, per mode (B44's rig, per cell) ===")
    print(
        f"{'mode':>10}{'ticks':>7}{'rho(K) med':>12}{'sig(K) med':>12}{'rho(used)':>11}"
        f"{'non-norm':>10}{'tau med':>10}{'tau inf':>9}{'sig>band':>10}"
    )
    for mode, record in records.items():
        for e in entries(record):
            o = e["operators"]
            tau = o["tau"]
            tau_med = tau.get("median")
            print(
                f"{mode:>10}{e['ticks']:>7}{o['rho_raw']['median']:>12.4f}"
                f"{o['sigma_raw']['median']:>12.4f}{o['rho_used']['median']:>11.4f}"
                f"{o['normality']['median']:>10.4f}"
                f"{(f'{tau_med:.1f}' if tau_med is not None else 'n/a'):>10}"
                f"{o['cells_tau_infinite']:>9}{o['cells_sigma_used_above_band']:>10}"
            )

    print("\n=== 3. does the per-cell collapse track K? (Spearman across cells) ===")
    print(
        f"{'mode':>10}{'ticks':>7}{'r(ER,rho_used)':>16}{'n':>5}{'r(ER,tau)':>11}{'n':>5}"
        f"{'r(ER,non-norm)':>16}{'n':>5}{'r(ER,rho_K)':>13}{'n':>5}"
    )
    for mode, record in records.items():
        for e in entries(record):
            c = e["correlations"]
            print(
                f"{mode:>10}{e['ticks']:>7}"
                f"{fmt_rho(c['block_er_vs_rho_used']):>16}{c['block_er_vs_rho_used']['n']:>5}"
                f"{fmt_rho(c['block_er_vs_tau']):>11}{c['block_er_vs_tau']['n']:>5}"
                f"{fmt_rho(c['block_er_vs_normality']):>16}{c['block_er_vs_normality']['n']:>5}"
                f"{fmt_rho(c['block_er_vs_rho_raw']):>13}{c['block_er_vs_rho_raw']['n']:>5}"
            )


if __name__ == "__main__":
    main()
