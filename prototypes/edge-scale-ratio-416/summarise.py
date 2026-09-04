"""The reading: where the edge scale ratio sits, and whether it drifts (#416).

    python prototypes/edge-scale-ratio-416/summarise.py --ticks 30000

Five tables, in the order #416's *What to report* asks for them.

1. **The distribution against the band**, split by how many of an edge's ends
   are free. Interior edges have both ends banded and an admissible ratio of
   `ρ² = 4`; boundary-incident edges have one end pinned at the exact gauge and
   an admissible ratio of `ρ = 2`. **Never pooled** — the two populations have
   different denominators and one occupancy figure over both would be a number
   about neither.
2. **Drift**, the discriminating reading: the same spread at every checkpoint on
   the ladder, so a ratio that steps early and holds is distinguishable from one
   still climbing at the horizon.
3. **By edge kind** — interior, sensory, motor, drive.
4. **By position on the channel**, `min(d(u, rim), d(v, rim))` — per edge, keyed
   by a distance to the rim and never by the dome's imposed level (#181).
5. **Flatness**, `σ_min/σ_max` per map, carried because the identification of
   the Frobenius ratio with `σ_u/σ_v` is exact under #411's flat spectrum and
   approximate otherwise. This is the column that says which of the two the
   reading is standing on.

**Spreads are over `|log₂ ratio|`,** the direction-free magnitude. A ratio and
its reciprocal are the same mismatch read from the two ends, so a median taken
over directed ratios would sit near 1 no matter how large the mismatches are.
`|log₂|` of 1 is a factor of 2; the interior band's `ρ² = 4` is 2 in these
units, and the boundary band's `ρ = 2` is 1.

**The seed row is a spread over runs, not the graph-wide average #127 forbids.**
Every within-run reduction here is already over a population of edges sliced by
a property of the graph.
"""

import argparse
import json
import pathlib
import statistics
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent


def load(directory: pathlib.Path, ticks: int | None) -> list[dict]:
    """Records at one horizon. Lengths select rather than pool.

    Drift is the whole question, so a 30,000-tick and a 100,000-tick record in
    the same row would answer it by averaging it away.
    """
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(directory.glob("416-*.json"))
    ]
    if ticks is not None:
        records = [r for r in records if r["ticks"] == ticks]
    records.sort(key=lambda r: r["seed"])
    return records


def magnitude(ratio: np.ndarray) -> np.ndarray:
    """`|log₂ ratio|` — a factor of 2 reads 1, a factor of 4 reads 2."""
    values = np.abs(np.log2(np.asarray(ratio, dtype=float)))
    return values[np.isfinite(values)]


def spread(values: np.ndarray) -> tuple[float, ...]:
    if values.size == 0:
        return (float("nan"),) * 5
    return tuple(float(v) for v in np.percentile(values, (50, 75, 95, 99, 100)))


def frame(record: dict, tick: int | None = None) -> dict:
    if tick is None:
        return record["frames"][-1]
    for item in record["frames"]:
        if item["tick"] == tick:
            return item
    raise KeyError(tick)


def masks(record: dict) -> dict[str, np.ndarray]:
    geometry = record["geometry"]
    free = np.array([row["free_ends"] for row in geometry])
    return {
        "both ends free (rho^2 = 4)": free == 2,
        "one end pinned (rho = 2)": free == 1,
    }


#: How close to the band face counts as *on* it. The maps are stored in float32
#: and the projection clamps there, so a map the projection has just set to
#: exactly `ρ` reads `2.0000002` once the norm is retaken in float64 — a
#: measured excess of 2.5e-7 at 30,000 ticks, which is round-off and not a
#: breach. Anything at or inside this is reported as **at the face**; only a
#: value beyond it would be the projection failing, and none is.
FACE = 1e-5


def occupancy(values: np.ndarray, admissible: float) -> float:
    """Share of the edge's own admissible band the ratio is using, at the p95."""
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values, 95) / np.log2(admissible))


def at_face(values: np.ndarray, admissible: float) -> tuple[float, float]:
    """(share sitting on the band face, share genuinely beyond it).

    Separated because they mean opposite things: sitting on the face is the
    gauge **binding**, and sitting beyond it would be the gauge **failing**.
    Reporting them as one number would let round-off read as a violation.
    """
    if values.size == 0:
        return float("nan"), float("nan")
    edge = np.log2(admissible)
    return (
        float((values >= edge - FACE).mean()),
        float((values > edge + FACE).mean()),
    )


def table(title: str, header: str, rows: list[str]) -> None:
    print(f"\n   {title}")
    print(header)
    print("   " + "-" * (len(header) - 3))
    for row in rows:
        print(row)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directory", nargs="?", default=str(HERE))
    parser.add_argument("--ticks", type=int, default=None)
    args = parser.parse_args(argv)
    records = load(pathlib.Path(args.directory), args.ticks)
    if not records:
        print(f"no records in {args.directory}; run read.py first")
        return 1

    first = records[0]
    lengths = sorted({r["ticks"] for r in records})
    print(
        f"\n== the edge scale ratio against ADR-0010's band (#416) ==\n"
        f"   real dome, {len(first['geometry'])} edges, rho = {first['rho']}, "
        f"split {first['split']}\n"
        f"   seeds {[r['seed'] for r in records]} at {lengths} ticks\n"
        f"   spreads are |log2 ratio|: 1 is a factor of 2, 2 is a factor of 4\n"
    )

    # 1. the distribution against the band
    rows = []
    header = (
        f"{'population':>28}  {'seed':>4} {'edges':>6}  "
        f"{'median':>8} {'p75':>8} {'p95':>8} {'p99':>8} {'max':>8}  {'band':>6} "
        f"{'p95/band':>9} {'on face':>8} {'beyond':>7}"
    )
    for name, mask in masks(records[0]).items():
        for record in records:
            admissible = float(
                np.array([r["admissible"] for r in record["geometry"]])[mask][0]
            )
            values = magnitude(np.array(frame(record)["norm_ratio"])[mask])
            p50, p75, p95, p99, top = spread(values)
            face, beyond = at_face(values, admissible)
            rows.append(
                f"{name:>28}  {record['seed']:>4} {int(mask.sum()):>6}  "
                f"{p50:>8.4g} {p75:>8.4g} {p95:>8.4g} {p99:>8.4g} {top:>8.4g}  "
                f"{admissible:>6.3g} {occupancy(values, admissible):>9.3f} "
                f"{face:>7.1%} {beyond:>6.1%}"
            )
    table("1. where the ratio sits, against each population's own band", header, rows)

    # 2. drift
    rows = []
    ladder = [f["tick"] for f in records[0]["frames"]]
    header = (
        f"{'population':>28}  {'tick':>7}  "
        + " ".join(f"{'s' + str(r['seed']):>8}" for r in records)
        + f"  {'median':>8} {'p95':>8}"
    )
    for name, mask in masks(records[0]).items():
        for tick in ladder:
            per_seed = []
            pooled_p95 = []
            for record in records:
                try:
                    values = magnitude(np.array(frame(record, tick)["norm_ratio"])[mask])
                except KeyError:
                    per_seed.append(float("nan"))
                    pooled_p95.append(float("nan"))
                    continue
                p50, _p75, p95, _p99, _top = spread(values)
                per_seed.append(p50)
                pooled_p95.append(p95)
            clean = [v for v in per_seed if v == v]
            clean95 = [v for v in pooled_p95 if v == v]
            rows.append(
                f"{name:>28}  {tick:>7}  "
                + " ".join(f"{v:>8.4g}" for v in per_seed)
                + f"  {statistics.median(clean) if clean else float('nan'):>8.4g}"
                f" {statistics.median(clean95) if clean95 else float('nan'):>8.4g}"
            )
    table(
        "2. does it drift with training? the same spread at every checkpoint",
        header,
        rows,
    )

    # 3. by edge kind
    rows = []
    header = (
        f"{'kind':>10} {'m':>3} {'edges':>6}  {'band':>6}  "
        + " ".join(f"{'s' + str(r['seed']):>8}" for r in records)
        + f"  {'p95':>8}"
    )
    kinds = sorted({row["kind"] for row in records[0]["geometry"]})
    for kind in kinds:
        mask = np.array([row["kind"] == kind for row in records[0]["geometry"]])
        meta = [r for r in records[0]["geometry"] if r["kind"] == kind][0]
        medians, p95s = [], []
        for record in records:
            values = magnitude(np.array(frame(record)["norm_ratio"])[mask])
            p50, _p75, p95, _p99, _top = spread(values)
            medians.append(p50)
            p95s.append(p95)
        rows.append(
            f"{kind:>10} {meta['m']:>3} {int(mask.sum()):>6}  "
            f"{meta['admissible']:>6.3g}  "
            + " ".join(f"{v:>8.4g}" for v in medians)
            + f"  {statistics.median(p95s):>8.4g}"
        )
    table("3. by edge kind", header, rows)

    # 4. by position on the channel
    rows = []
    depths = sorted({row["depth"] for row in records[0]["geometry"]})
    header = (
        f"{'d(edge,rim)':>11} {'edges':>6}  "
        + " ".join(f"{'s' + str(r['seed']):>8}" for r in records)
        + f"  {'p95':>8}  {'kinds':>28}"
    )
    for depth in depths:
        mask = np.array([row["depth"] == depth for row in records[0]["geometry"]])
        present = sorted(
            {r["kind"] for r, keep in zip(records[0]["geometry"], mask) if keep}
        )
        medians, p95s = [], []
        for record in records:
            values = magnitude(np.array(frame(record)["norm_ratio"])[mask])
            p50, _p75, p95, _p99, _top = spread(values)
            medians.append(p50)
            p95s.append(p95)
        rows.append(
            f"{depth:>11} {int(mask.sum()):>6}  "
            + " ".join(f"{v:>8.4g}" for v in medians)
            + f"  {statistics.median(p95s):>8.4g}  {','.join(present):>28}"
        )
    table(
        "4. by position on the channel, per edge and never per level (#181)",
        header,
        rows,
    )

    # correlation of the ratio magnitude with depth, per seed
    #
    # **Stated twice, and the pooled figure is the misleading one.** Over all
    # edges the correlation is dominated by the rim: every boundary-incident
    # edge sits at depth 0 and at its band face, so a strong negative number
    # falls out of the *populations* being different rather than of any gradient
    # along the channel. The interior-only figure is the one that answers "does
    # the ratio correlate with position", because within it every edge has the
    # same band and the same pair of free ends.
    print("\n   rank correlation of |log2 ratio| with d(edge, rim), per seed")
    depth_of = np.array([row["depth"] for row in records[0]["geometry"]])
    interior = np.array([row["free_ends"] == 2 for row in records[0]["geometry"]])
    for record in records:
        values = np.abs(np.log2(np.array(frame(record)["norm_ratio"], dtype=float)))
        line = []
        for label, subset in (("all edges", np.ones_like(interior)), ("interior only", interior)):
            keep = np.isfinite(values) & subset
            order_v = np.argsort(np.argsort(values[keep]))
            order_d = np.argsort(np.argsort(depth_of[keep]))
            correlation = float(np.corrcoef(order_v, order_d)[0, 1])
            line.append(f"{label} {correlation:+.3f} (n={int(keep.sum())})")
        print(f"      seed {record['seed']}: " + ",  ".join(line))

    # 5. flatness
    rows = []
    header = (
        f"{'seed':>6}  {'p05':>8} {'p25':>8} {'median':>8} {'p75':>8} {'p95':>8}"
    )
    for record in records:
        flat = np.array(frame(record)["flatness"], dtype=float)
        flat = flat[np.isfinite(flat)]
        cells = " ".join(
            f"{np.percentile(flat, q):>8.4g}" for q in (5, 25, 50, 75, 95)
        )
        rows.append(f"{record['seed']:>6}  {cells}")
    table(
        "5. flatness, sigma_min/sigma_max per map -- 1 is #411's target",
        header,
        rows,
    )

    # the map norms themselves, because the band's upper face is claimed to bind
    rows = []
    header = f"{'seed':>6}  {'pop':>28}  {'min':>8} {'median':>8} {'max':>8} {'at rho':>8}"
    for record in records:
        norms = np.array(frame(record)["map_norms"], dtype=float)
        pinned = []
        free = []
        for row in record["geometry"]:
            pinned.append(row["u_pinned"])
            pinned.append(row["v_pinned"])
        pinned = np.array(pinned)
        free = ~pinned
        for label, mask in (
            ("banded maps", free),
            ("pinned maps (exact gauge)", pinned),
        ):
            values = norms[mask]
            riding = float((values > record["rho"] - 1e-6).mean())
            rows.append(
                f"{record['seed']:>6}  {label:>28}  "
                f"{values.min():>8.4g} {np.median(values):>8.4g} "
                f"{values.max():>8.4g} {riding:>7.1%}"
            )
    table(
        "6. the map norms themselves: does the larger end ride rho, as ADR-0010 says?",
        header,
        rows,
    )
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
