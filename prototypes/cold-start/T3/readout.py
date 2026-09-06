"""T3's readout: the canonical table on the full dome, and Done-when (1) and (2).

Reads `524-{condition}-seed{42,43,44}-100000.json` beside this file (with
``--inflight``, the `.inflight.json` files too, for a look at runs still
going). Prints markdown.

Conventions, T1's and stated once. `rho` is the **used** operator's (T0's
comparator), raw beside it. A class figure is the median over the class's
cells, per seed; the published number is the **mean of the per-seed medians**
and the **spread** is their standard deviation across seeds. Horizons are never
pooled (#178) -- 100k is the reading and 30k is printed beside it. Every class
is published, not only the ones a clause names (T2's cost-of-learning: a guard
chosen for one intervention does not transfer to another, and at `T_b = 5000`
its guard passed a condition that took vision from 0.977 to 0.779).

Done-when, read exactly as the map writes it:

1. apex `rho(K)` -- median over the 8 L7 cells, mean over seeds, spread
   published -- is **not below** core L3-L6's;
2. with induced activity annealed to zero (here: identically zero throughout,
   the frozen world), arm travel per window stays **> 0** and composed
   rim-to-apex effective rank is **> 1.5**.

Only the falsifying end of each clause is a verdict; a pass is a magnitude with
a spread (#481's idiom).

**Done-when (3), ADR-0026's conduction ratio, is not read here.** It is the
map's clause for [T4](https://github.com/NGL321/patchworks/issues/525), which
runs on this same full dome and hands the reading to #127 as the first on a
world that varies. T2's paired-counterfactual fork
(`prototypes/cold-start/T2/run.py::reach_fork`) is the instrument and is
reusable as-is; a second reading taken here would sit on the same surface with
no consumer. Named so the table's one gap is a scoping call on the record
rather than an omission.

Usage::

    python prototypes/cold-start/T3/readout.py [--inflight]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# The table is written in the map's idiom -- `ρ`, `±`, `−` -- and this box's
# console defaults to cp1252, which cannot encode any of them.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
CONDITIONS = ("baseline", "winner")
BASELINE = "baseline"
LABEL = {
    "baseline": "baseline: ρ=1 off, c=1.0",
    "winner": "T1's winner: ρ=1 on, c=0.1",
}
CLASSES = ("apex", "core", "vision", "soma")
CLASS_LABEL = {
    "apex": "apex (core, drive-adjacent; L7, 8)",
    "core": "core L3–L6 (52)",
    "vision": "vision L1 (64)",
    "soma": "somatomotor L1 boundary-adjacent (6)",
}
HORIZONS = (30_000, 100_000)
#: Done-when (2)'s bar on composed rim-to-apex transport.
RANK_BAR = 1.5


def load(inflight: bool) -> dict[str, dict[int, dict]]:
    out: dict[str, dict[int, dict]] = {c: {} for c in CONDITIONS}
    for cond in CONDITIONS:
        patterns = [f"524-{cond}-seed*-100000.json"]
        if inflight:
            patterns.append(f"524-{cond}-seed*-100000.inflight.json")
        for pat in patterns:
            for path in sorted(HERE.glob(pat)):
                d = json.loads(path.read_text())
                d["_inflight"] = path.name.endswith(".inflight.json")
                out[cond].setdefault(int(d["seed"]), d)
    return out


def at(d: dict, tick: int) -> dict | None:
    for cp in d["checkpoints"]:
        if cp["ticks"] == tick:
            return cp
    return None


def arr(cp: dict, key: str) -> np.ndarray:
    return np.asarray(cp["per_cell"][key], dtype=float)


def fmt(x, p: int = 3) -> str:
    return "—" if x is None or not np.isfinite(x) else f"{x:.{p}f}"


def spread(vals: list[float]) -> tuple[float, float]:
    """(mean, std across seeds). The spread is the precision of the claim."""
    if not vals:
        return float("nan"), float("nan")
    a = np.asarray(vals, dtype=float)
    return float(a.mean()), float(a.std(ddof=0))


def per_seed(runs: dict[int, dict], tick: int, f) -> list[float]:
    """`f(record, checkpoint)` at `tick`, one value per seed that reached it."""
    vals = []
    for s in sorted(runs):
        cp = at(runs[s], tick)
        if cp is not None:
            v = f(runs[s], cp)
            if v is not None and np.isfinite(v):
                vals.append(float(v))
    return vals


def class_median(cls: str, key: str = "rho_used"):
    def f(d, cp):
        idx = d["groups"][cls]
        return float(np.median(arr(cp, key)[idx])) if idx else float("nan")
    return f


def class_mean(cls: str, key: str):
    def f(d, cp):
        idx = d["groups"][cls]
        return float(np.mean(arr(cp, key)[idx])) if idx else float("nan")
    return f


def used_modes(cls: str):
    def f(d, cp):
        idx = d["groups"][cls]
        m = np.asarray(cp["used"]["per_cell"]["modes_retaining"], dtype=float)
        return float(np.median(m[idx])) if idx else float("nan")
    return f


def composed_median(d, cp):
    return float(np.median(np.asarray(cp["composed"]["effective_rank"], dtype=float)))


def composed_max(d, cp):
    return float(np.max(np.asarray(cp["composed"]["effective_rank"], dtype=float)))


def travel_window(d, cp):
    return float(cp["travel_window"])


def travel_per_tick(d, cp):
    return float(cp["travel_per_tick"])


def drive_relative(d, cp):
    return float(np.median(np.asarray(cp["drive_edges"]["relative"], dtype=float)))


def dead_cells(d, cp):
    return float((np.asarray(cp["used"]["per_cell"]["modes_retaining"]) == 0).sum())


def line(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def header(cols: list[str]) -> list[str]:
    return [line(cols), line(["---"] * len(cols))]


def stat_cell(runs, tick, f, p: int = 3) -> str:
    vals = per_seed(runs, tick, f)
    if not vals:
        return "—"
    m, s = spread(vals)
    return f"{m:.{p}f} ± {s:.{p}f}" if len(vals) > 1 else f"{m:.{p}f} (1 seed)"


def section_retention(data: dict) -> list[str]:
    out = ["## Retention: `ρ(K)` and `modes_retaining`, every class", ""]
    for tick in HORIZONS:
        out += [f"### {tick:,} ticks", ""]
        out += header(["class", *[f"{LABEL[c]} — ρ used" for c in CONDITIONS],
                       *[f"{LABEL[c]} — modes" for c in CONDITIONS]])
        for cls in CLASSES:
            row = [CLASS_LABEL[cls]]
            row += [stat_cell(data[c], tick, class_median(cls)) for c in CONDITIONS]
            row += [stat_cell(data[c], tick, used_modes(cls), 1) for c in CONDITIONS]
            out.append(line(row))
        out.append("")
        # The clause-(1) quantity itself, and the gap it is about.
        out += header(["reading", *[LABEL[c] for c in CONDITIONS]])
        gap = lambda d, cp: class_median("core")(d, cp) - class_median("apex")(d, cp)  # noqa: E731
        out.append(line(["core L3–L6 − apex gap", *[stat_cell(data[c], tick, gap) for c in CONDITIONS]]))
        out.append(line(["ρ raw, apex", *[stat_cell(data[c], tick, class_median("apex", "rho_raw")) for c in CONDITIONS]]))
        out.append(line(["dead cells (modes = 0)", *[stat_cell(data[c], tick, dead_cells, 1) for c in CONDITIONS]]))
        out.append("")
    return out


def section_excitation(data: dict) -> list[str]:
    out = ["## Per-cell excitation rank (participation ratio of the evidence stream)", ""]
    for tick in HORIZONS:
        out += [f"### {tick:,} ticks", ""]
        out += header(["class", *[f"{LABEL[c]} — PR" for c in CONDITIONS],
                       *[f"{LABEL[c]} — PR centred" for c in CONDITIONS]])
        for cls in CLASSES:
            row = [CLASS_LABEL[cls]]
            row += [stat_cell(data[c], tick, class_median(cls, "pr_total")) for c in CONDITIONS]
            row += [stat_cell(data[c], tick, class_median(cls, "pr_total_centred")) for c in CONDITIONS]
            out.append(line(row))
        out.append("")
    return out


def section_transport(data: dict) -> list[str]:
    out = [
        "## Done-when (2): composed rim-to-apex effective rank, and arm travel",
        "",
        "Seven hops on the full dome (#436/#497's construction: one chain per sensorimotor rim",
        "cell, the graph's own shortest edge path to an apex cell, composed, participation ratio",
        f"of the spectrum). The bar is **> {RANK_BAR}**. Induced activity is identically zero in",
        "both arms, so *annealed to zero* is the condition throughout, not a phase boundary.",
        "",
    ]
    any_run = next((r for c in CONDITIONS for r in data[c].values()), None)
    if any_run is not None:
        con = any_run["composed_at_construction"]
        out += ["Composed rank **at construction**, before a tick is run, by rim kind:", ""]
        out += header(["rim kind", "chains", "median", "p10", "p90"])
        for kind in sorted(con):
            k = con[kind]
            out.append(line([kind, str(k["chains"]), fmt(k["effective_rank_median"]),
                             fmt(k["effective_rank_p10"]), fmt(k["effective_rank_p90"])]))
        out.append("")
    out += header(["reading", *[LABEL[c] for c in CONDITIONS]])
    for tick in HORIZONS:
        out.append(line([f"composed ER, median over chains @ {tick:,}",
                         *[stat_cell(data[c], tick, composed_median) for c in CONDITIONS]]))
    for tick in HORIZONS:
        out.append(line([f"composed ER, max over chains @ {tick:,}",
                         *[stat_cell(data[c], tick, composed_max) for c in CONDITIONS]]))
    for tick in HORIZONS:
        out.append(line([f"arm travel this window @ {tick:,}",
                         *[stat_cell(data[c], tick, travel_window, 4) for c in CONDITIONS]]))
    for tick in HORIZONS:
        out.append(line([f"arm travel per tick @ {tick:,}",
                         *[stat_cell(data[c], tick, travel_per_tick, 8) for c in CONDITIONS]]))
    out.append("")
    return out


def section_edges(data: dict) -> list[str]:
    out = [
        "## Drive edges, and disagreement energy beside per-edge effective rank",
        "",
        "#488's relative form, `|d| / (|a| + |b|)`: a nulled edge reads 0 and a fixed 2x scale",
        "mismatch reads 1/3. This is what the `ρ = 1` arm is for, read on this surface rather",
        "than assumed from the shallow one (where T1 found #488's 1/3 was **not** present, 0.01–0.02).",
        "",
    ]
    out += header(["reading", *[LABEL[c] for c in CONDITIONS]])
    for tick in HORIZONS:
        out.append(line([f"drive-edge relative disagreement, median @ {tick:,}",
                         *[stat_cell(data[c], tick, drive_relative, 4) for c in CONDITIONS]]))
    out.append("")
    # Per-stratum disagreement energy and map effective rank, at the horizon.
    for tick in HORIZONS:
        strata = None
        for c in CONDITIONS:
            for d in data[c].values():
                cp = at(d, tick)
                if cp is not None:
                    strata = sorted(cp["edges"]["by_stratum"])
                    break
            if strata:
                break
        if not strata:
            continue
        out += [f"### Per-edge, by stratum, at {tick:,} ticks", ""]
        out += header(["stratum", "m", *[f"{LABEL[c]} — dis. energy" for c in CONDITIONS],
                       *[f"{LABEL[c]} — map ER" for c in CONDITIONS],
                       *[f"{LABEL[c]} — PR centred" for c in CONDITIONS]])
        for st in strata:
            def g(key, s=st):
                return lambda d, cp: cp["edges"]["by_stratum"].get(s, {}).get(key, float("nan"))
            m = None
            for c in CONDITIONS:
                for d in data[c].values():
                    cp = at(d, tick)
                    if cp is not None and st in cp["edges"]["by_stratum"]:
                        m = cp["edges"]["by_stratum"][st]["m"]
                        break
                if m is not None:
                    break
            row = [st, str(m)]
            row += [stat_cell(data[c], tick, g("disagreement_energy_median"), 6) for c in CONDITIONS]
            row += [stat_cell(data[c], tick, g("map_effective_rank_median")) for c in CONDITIONS]
            row += [stat_cell(data[c], tick, g("pr_centred_median")) for c in CONDITIONS]
            out.append(line(row))
        out.append("")
    return out


def section_verdict(data: dict) -> list[str]:
    """The two clauses, read at the deepest horizon reached by both arms."""
    tick = 0
    for t in HORIZONS:
        if all(per_seed(data[c], t, composed_median) for c in CONDITIONS):
            tick = t
    out = ["## The two clauses", ""]
    if not tick:
        return out + ["*No horizon reached by both arms yet.*", ""]
    out += [f"Read at **{tick:,} ticks**, the deepest horizon both arms reached (#178: horizons are never pooled).", ""]
    for cond in CONDITIONS:
        runs = data[cond]
        apex = per_seed(runs, tick, class_median("apex"))
        core = per_seed(runs, tick, class_median("core"))
        am, asd = spread(apex)
        cm, csd = spread(core)
        # "Not below": falsified only when apex sits below core beyond the larger spread.
        margin = max(asd, csd)
        one_holds = am >= cm - margin
        trav = per_seed(runs, tick, travel_window)
        tm, tsd = spread(trav)
        comp = per_seed(runs, tick, composed_median)
        km, ksd = spread(comp)
        cmax = per_seed(runs, tick, composed_max)
        xm, _xsd = spread(cmax)
        out += [
            f"### {LABEL[cond]} ({len(apex)} seeds)",
            "",
            f"- **(1)** apex ρ **{am:.3f} ± {asd:.3f}** against core L3–L6 **{cm:.3f} ± {csd:.3f}** "
            f"(gap {cm - am:+.3f}, larger spread {margin:.3f}) → **{'holds' if one_holds else 'FAILS'}**",
            f"- **(2a)** arm travel this window **{tm:.4f} ± {tsd:.4f}** → "
            f"**{'holds (> 0)' if tm > 0 else 'FAILS (= 0)'}**",
            f"- **(2b)** composed rim-to-apex ER **{km:.3f} ± {ksd:.3f}** (max over chains {xm:.3f}) "
            f"against the bar {RANK_BAR} → **{'holds' if km > RANK_BAR else 'FAILS'}**",
            "",
        ]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inflight", action="store_true", help="include runs still going")
    args = p.parse_args()
    data = load(args.inflight)

    out = [
        "# T3 (#524): the canonical table on the full dome",
        "",
        "**Surface.** Full dome (`DEFAULT_SPEC`, forward normalisation, `interior_m = 3`,",
        "`boundary_m = 4`), `map/cold-start` with `main` merged, frozen world, no induced",
        "activity at any point. 100k ticks with 30k printed beside it; seeds 42/43/44.",
        "Aggregation: a class figure is the median over its cells per seed; published as the",
        "mean of the per-seed medians ± their standard deviation across seeds.",
        "",
    ]
    for cond in CONDITIONS:
        seeds = sorted(data[cond])
        flags = [f"{s}{'*' if data[cond][s].get('_inflight') else ''}" for s in seeds]
        deepest = []
        for s in seeds:
            deepest.append(str(data[cond][s]["checkpoints"][-1]["ticks"]) if data[cond][s]["checkpoints"] else "0")
        out.append(f"- **{LABEL[cond]}** — seeds {', '.join(flags) or '(none)'}; deepest ticks {', '.join(deepest)}")
    out += ["", "`*` = still in flight.", ""]

    out += section_verdict(data)
    out += section_retention(data)
    out += section_excitation(data)
    out += section_transport(data)
    out += section_edges(data)
    out += [
        "## Not read here",
        "",
        "**ADR-0026's conduction ratio, inbound and outbound.** It is Done-when (3), the map's",
        "clause for [T4](https://github.com/NGL321/patchworks/issues/525) — which runs on this",
        "same full dome and hands the reading to #127 as the first on a world that varies. T2's",
        "paired-counterfactual fork (`prototypes/cold-start/T2/run.py::reach_fork`) is the",
        "instrument and is reusable as it stands. A scoping call, named rather than silent.",
        "",
    ]
    print("\n".join(out))


if __name__ == "__main__":
    main()
