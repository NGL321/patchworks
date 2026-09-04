"""The curve: what `λ` bought and what it cost, read off `sweep.py`'s records.

    python prototypes/sparsity-lambda-393/read.py

#393's deliverable is the curve rather than a recommended value, so this prints
the three quantities side by side at every `λ` and does not reduce them to a
verdict. Per-seed rows are printed under each `λ`; the `λ` row is the median
over seeds, which is a spread over *runs* and not the graph-wide average #127's
standing rule forbids -- the within-run reductions are already per-map and
per-direction, and they are the ones that rule is about.
"""

import json
import pathlib
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent


def load(directory: pathlib.Path) -> dict[float, list[dict]]:
    by_lam: dict[float, list[dict]] = {}
    for path in sorted(directory.glob("393-lam*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        by_lam.setdefault(float(record["lam"]), []).append(record)
    for records in by_lam.values():
        records.sort(key=lambda r: r["seed"])
    return by_lam


def median(records: list[dict], *path: str) -> float:
    values = []
    for record in records:
        node = record
        for key in path:
            node = node[key]
        values.append(float(node))
    return statistics.median(values)


def main(argv: list[str] | None = None) -> int:
    directory = pathlib.Path(argv[0]) if argv else HERE
    by_lam = load(directory)
    if not by_lam:
        print(f"no records in {directory}; run sweep.py first")
        return 1

    any_record = next(iter(by_lam.values()))[0]
    print(
        f"\n== what the sparsity pressure buys, and what it costs the channel ==\n"
        f"   real dome, {any_record['ticks']} ticks, split {any_record['split']}, "
        f"chi = {any_record['cohomology']['chi']} fixed at construction\n"
        f"   effective rank reads 1 for a rank-1 map and up to "
        f"m = {any_record['edge_width']}\n"
    )
    header = (
        f"{'lam':>5}  {'eff rank':>9} {'(opening)':>10} {'drain':>6}  "
        f"{'dim H0':>7} {'rank d':>7} {'dim H1':>7}  "
        f"{'cond in':>9} {'cond out':>9}  {'txmit':>7}"
    )
    print(header)
    print("   " + "-" * (len(header) - 3))
    for lam in sorted(by_lam):
        records = by_lam[lam]
        print(
            f"{lam:>5g}  "
            f"{median(records, 'rank', 'draining_effective_rank'):>9.4g} "
            f"{median(records, 'rank', 'draining_effective_rank_opening'):>10.4g} "
            f"{median(records, 'rank', 'draining_edge_share'):>6.2f}  "
            f"{median(records, 'cohomology', 'dim_h0'):>7.0f} "
            f"{median(records, 'cohomology', 'rank_delta'):>7.0f} "
            f"{median(records, 'cohomology', 'dim_h1'):>7.0f}  "
            f"{median(records, 'channel', 'rim-to-apex', 'conduction', 'median'):>9.3g} "
            f"{median(records, 'channel', 'apex-to-rim', 'conduction', 'median'):>9.3g}  "
            f"{median(records, 'rank', 'transmitting'):>4.0f}"
            f"/{any_record['rank']['maps']}"
        )
        for record in records:
            rank = record["rank"]
            cohomology = record["cohomology"]
            channel = record["channel"]
            print(
                f"{'':>5}  seed {record['seed']}: "
                f"rank {rank['draining_effective_rank']:.4g} "
                f"(was {rank['draining_effective_rank_opening']:.4g}, "
                f"{rank['draining_edge_share']:.0%} draining)  "
                f"H0 {cohomology['dim_h0']} H1 {cohomology['dim_h1']} "
                f"rank d {cohomology['rank_delta']}  "
                f"cond in {channel['rim-to-apex']['conduction']['median']:.3g} "
                f"out {channel['apex-to-rim']['conduction']['median']:.3g}  "
                f"[{record['seconds']:.0f}s]"
            )

    print("\n   per-map effective rank across the whole fleet, closing window")
    print(f"{'lam':>5}  {'p05':>8} {'p25':>8} {'median':>8} {'p75':>8} {'p95':>8}")
    for lam in sorted(by_lam):
        records = by_lam[lam]
        cells = " ".join(
            f"{median(records, 'rank', 'closing_all_maps', q):>8.4g}"
            for q in ("p05", "p25", "median", "p75", "p95")
        )
        print(f"{lam:>5g}  {cells}")

    print("\n   conduction ratio, stated twice; ADR-0026's bar is >= 1")
    print(
        f"{'lam':>5}  {'direction':>12}  "
        f"{'p05':>9} {'p25':>9} {'median':>9} {'p75':>9} {'p95':>9}"
    )
    for lam in sorted(by_lam):
        for direction in ("rim-to-apex", "apex-to-rim"):
            cells = " ".join(
                f"{median(by_lam[lam], 'channel', direction, 'conduction', q):>9.3g}"
                for q in ("p05", "p25", "median", "p75", "p95")
            )
            print(f"{lam:>5g}  {direction:>12}  {cells}")

    print("\n   chi check: dim H0 - dim H1 at every lam, which must equal chi")
    for lam in sorted(by_lam):
        differences = {
            r["cohomology"]["dim_h0"] - r["cohomology"]["dim_h1"] for r in by_lam[lam]
        }
        print(f"{lam:>5g}  {sorted(differences)}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
