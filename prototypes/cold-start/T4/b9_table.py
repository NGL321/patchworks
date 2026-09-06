"""B9 (#551): the three-column table -- `c = 1` in circuit against the two things it is not.

Column 1 is this ticket's arm. Column 2 is #547's comparator, `c = 2` in circuit
on the same rig, same surface, same seed. Column 3 is the **re-projection** --
`sweep_c`'s `c = 1` read taken off the `c = 2` surface, which is the number the
map currently carries for `c = 1` and the one this ticket exists to replace.

Usage::

    python prototypes/cold-start/T4/b9_table.py
"""

from __future__ import annotations

import json
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def load(name: str) -> dict | None:
    for candidate in (_HERE / name, _HERE / name.replace(".json", ".inflight.json")):
        if candidate.exists():
            return json.loads(candidate.read_text())
    return None


def excess(x: float) -> str:
    """The excess over one, which is the quantity the whole map is arguing about."""
    return f"{x - 1:.3e}"


def table(condition: str) -> list[str]:
    circuit = load(f"551-c1-{condition}-seed42-100000.json")
    ref = load(f"537-{condition}-seed42-100000.json")
    if circuit is None or ref is None:
        return [f"### {condition}: missing arm ({circuit is None=}, {ref is None=})"]

    by_ticks = {cp["ticks"]: cp for cp in ref["checkpoints"]}
    rows = [
        f"### {condition}",
        "",
        "| ticks | c=1 in circuit | c=2 in circuit | c=1 re-projected | in-circuit p90 | c=2 p90 | ER max | cap max | at cap | h max | non-finite |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cp in circuit["checkpoints"]:
        t = cp["ticks"]
        a = cp["angles"]
        r = by_ticks.get(t)
        r_med = excess(r["angles"]["composed_er"]["median"]) if r else "--"
        r_p90 = f"{r['angles']['composed_er']['p90']:.5f}" if r else "--"
        reproj = "--"
        if r:
            hit = [x for x in r["sweep_c"] if x["c"] == 1]
            if hit:
                reproj = excess(hit[0]["er_median"])
        s = cp["stability"]
        nonfinite = s["h_nonfinite_cells"] + s["e_nonfinite_cells"] + s["maps_nonfinite"]
        rows.append(
            f"| {t} | {excess(a['composed_er']['median'])} | {r_med} | {reproj} "
            f"| {a['composed_er']['p90']:.5f} | {r_p90} | {a['composed_er']['max']:.4f} "
            f"| {cp['cap']['ratio_max']:.3f} | {cp['cap']['cells_at_cap']} "
            f"| {s['h_norm']['max']:.4g} | {nonfinite} |"
        )
    rows.append("")
    return rows


def main() -> None:
    out = ["# B9 (#551): `c = 1` in circuit, projection and gain, to horizon", ""]
    for name in ("551-c1-baseline-seed42-100000.json", "551-c1-winner-seed42-100000.json"):
        rec = load(name)
        if rec:
            out += [
                f"`{name}` -- gain moved at **{rec['gain']['cells_gain_moved']}** cells, "
                f"ratios {rec['gain']['gain_ratio_vs_c2']}, counts {rec['gain']['overlap_counts_histogram']} "
                f"(at `c = 2`: {rec['gain']['overlap_counts_histogram_at_c2']}).",
                "",
            ]
            break
    for condition in ("baseline", "winner"):
        out += table(condition)
    print("\n".join(out))


if __name__ == "__main__":
    main()
