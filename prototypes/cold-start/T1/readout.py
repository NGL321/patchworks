"""T1's readout: the 2x2's pre-registered reads, and which branch-table rows fired.

Reads `521-{condition}-seed{42,43,44}-30000.json` beside this file (with
``--inflight``, the `.inflight.json` files too, for a look at runs still
going). Prints markdown.

Conventions, stated once. `rho` is the **used** operator's (T0's comparator),
raw beside it. A class figure is the median over the class's cells, per seed;
the published number is the mean of the per-seed medians and the **spread** is
their standard deviation across seeds. The winner test compares a cell's mean
apex `rho` to baseline's, against the larger of the two spreads, at the
deepest horizon reached (30k; #178's rule, 20k published beside it). The
retention guard is the same comparison, per class, downward.

Usage::

    python prototypes/cold-start/T1/readout.py [--inflight]
"""

from __future__ import annotations

import argparse
import sys
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CONDITIONS = ("rho1off-c1", "rho1on-c1", "rho1off-c0.1", "rho1on-c0.1")
BASELINE = "rho1off-c1"
LABEL = {
    "rho1off-c1": "baseline: ρ=1 off, c=1.0",
    "rho1on-c1": "ρ=1 on, c=1.0",
    "rho1off-c0.1": "ρ=1 off, c=0.1",
    "rho1on-c0.1": "ρ=1 on, c=0.1",
}
CLASSES = ("apex", "core", "vision", "soma")
CLASS_LABEL = {
    "apex": "apex (core, drive-adjacent; 8)",
    "core": "core (one level; 16)",
    "vision": "vision L1 (64)",
    "soma": "somatomotor L1 boundary-adjacent (6)",
}
HORIZONS = (5000, 10000, 20000, 30000)


def load(inflight: bool) -> dict[str, dict[int, dict]]:
    out: dict[str, dict[int, dict]] = {c: {} for c in CONDITIONS}
    for cond in CONDITIONS:
        patterns = [f"521-{cond}-seed*-30000.json"]
        if inflight:
            patterns.append(f"521-{cond}-seed*-30000.inflight.json")
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


def fmt(x: float, p: int = 3) -> str:
    return "—" if x is None or not np.isfinite(x) else f"{x:.{p}f}"


def class_median(d: dict, cp: dict, cls: str, key: str = "rho_used") -> float:
    idx = d["groups"][cls]
    return float(np.median(arr(cp, key)[idx]))


def class_stat(runs: dict[int, dict], tick: int, cls: str, key: str = "rho_used") -> tuple[float, float, list[float]]:
    """(mean of per-seed medians, std across seeds, the per-seed medians)."""
    vals = []
    for s in sorted(runs):
        cp = at(runs[s], tick)
        if cp is not None:
            vals.append(class_median(runs[s], cp, cls, key))
    if not vals:
        return float("nan"), float("nan"), []
    v = np.asarray(vals)
    return float(v.mean()), float(v.std()), vals


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inflight", action="store_true")
    args = p.parse_args()
    data = load(args.inflight)
    have = {c: sorted(data[c]) for c in CONDITIONS}
    any_run = next((data[c][s] for c in CONDITIONS for s in have[c]), None)
    if any_run is None:
        raise SystemExit("no T1 runs found")
    surf = any_run["surface"]
    spec = any_run["dome_spec"]
    print(f"# T1 readout — surface `{surf['describe']}` on `{surf['branch']}`")
    print(f"band: {surf['band']}; (interior_m, boundary_m) = ({surf['interior_m']}, {surf['boundary_m']}); "
          f"shallow dome vision_sides {spec['vision_sides']}, somatomotor {spec['somatomotor_sizes']}, core {spec['core_sizes']}, "
          f"core_degree {spec['core_degree']}, apex_degree {spec['apex_degree']}; {spec['cells']} cells / {spec['edges']} edges; "
          f"frozen world, 30k ticks.\n")
    print("| condition | seeds present | complete | elapsed min/seed |")
    print("|---|---|---|---|")
    for c in CONDITIONS:
        runs = data[c]
        done = [s for s in have[c] if not runs[s]["_inflight"] and at(runs[s], 30000) is not None]
        print(f"| {LABEL[c]} | {have[c]} | {done} | {[round(runs[s].get('elapsed_minutes', float('nan')), 1) for s in have[c]]} |")
    print()

    # ---- the canonical per-class rho table --------------------------------
    print("## ρ(K) by class and condition — used operator, mean of per-seed medians ± spread [per seed]\n")
    for tick in HORIZONS:
        print(f"**tick {tick}**\n")
        print("| class | " + " | ".join(LABEL[c] for c in CONDITIONS) + " |")
        print("|---|" + "---|" * len(CONDITIONS))
        for cls in CLASSES:
            cells = []
            for c in CONDITIONS:
                m, sd, per = class_stat(data[c], tick, cls)
                cells.append(f"**{fmt(m)}** ± {fmt(sd)} {[round(x, 3) for x in per]}" if per else "—")
            print(f"| {CLASS_LABEL[cls]} | " + " | ".join(cells) + " |")
        print()
    print("Raw operator beside it (the stored parameter, rescaled into the band by forward normalisation), apex and core, tick 30k:\n")
    print("| class | " + " | ".join(LABEL[c] for c in CONDITIONS) + " |")
    print("|---|" + "---|" * len(CONDITIONS))
    for cls in ("apex", "core"):
        cells = []
        for c in CONDITIONS:
            m, sd, per = class_stat(data[c], 30000, cls, "rho_raw")
            cells.append(f"{fmt(m)} ± {fmt(sd)}" if per else "—")
        print(f"| {CLASS_LABEL[cls]} | " + " | ".join(cells) + " |")
    print()

    # ---- apex trajectory ----------------------------------------------------
    print("## Apex ρ(K) along the ladder — mean of per-seed medians\n")
    ladder = [cp["ticks"] for cp in any_run["checkpoints"]]
    ladder = [t for t in (1000, 2000, 5000, 10000, 20000, 30000) if t in ladder or t == 30000]
    print("| condition | " + " | ".join(str(t) for t in ladder) + " |")
    print("|---|" + "---|" * len(ladder))
    for c in CONDITIONS:
        row = []
        for t in ladder:
            m, sd, per = class_stat(data[c], t, "apex")
            row.append(fmt(m) if per else "—")
        print(f"| {LABEL[c]} | " + " | ".join(row) + " |")
    print("\nCore ρ(K) along the same ladder:\n")
    print("| condition | " + " | ".join(str(t) for t in ladder) + " |")
    print("|---|" + "---|" * len(ladder))
    for c in CONDITIONS:
        row = []
        for t in ladder:
            m, sd, per = class_stat(data[c], t, "core")
            row.append(fmt(m) if per else "—")
        print(f"| {LABEL[c]} | " + " | ".join(row) + " |")
    print()

    # ---- the winner test -----------------------------------------------------
    print("## The winner test — apex ρ against baseline's, beyond the seed spread\n")
    verdict = {}
    for tick in (20000, 30000):
        bm, bsd, bper = class_stat(data[BASELINE], tick, "apex")
        print(f"**tick {tick}** — baseline apex {fmt(bm)} ± {fmt(bsd)}\n")
        print("| condition | apex ρ | Δ vs baseline | spread (max of the two) | exceeds |")
        print("|---|---|---|---|---|")
        winners = []
        for c in CONDITIONS:
            if c == BASELINE:
                continue
            m, sd, per = class_stat(data[c], tick, "apex")
            if not per or not bper:
                print(f"| {LABEL[c]} | — | — | — | — |")
                continue
            spread = max(sd, bsd)
            delta = m - bm
            exceeds = delta > spread
            if exceeds:
                winners.append((m, c))
            print(f"| {LABEL[c]} | {fmt(m)} | {delta:+.3f} | {fmt(spread)} | **{exceeds}** |")
        verdict[tick] = max(winners)[1] if winners else None
        print()
        if winners:
            print(f"→ at {tick}: winner **{LABEL[verdict[tick]]}** (highest apex ρ among the cells beyond spread: {[(LABEL[c], round(m, 3)) for m, c in sorted(winners, reverse=True)]})\n")
        else:
            print(f"→ at {tick}: **no cell exceeds the spread; baseline is the winner.**\n")

    # ---- apex vs core --------------------------------------------------------
    print("## Apex against core, per cell of the 2x2 (branch row 3 — a magnitude, never a pass of Done-when (1))\n")
    print("| condition | tick | apex | core | core − apex | apex ≥ core |")
    print("|---|---|---|---|---|---|")
    for c in CONDITIONS:
        for tick in (20000, 30000):
            am, asd, aper = class_stat(data[c], tick, "apex")
            cm, csd, cper = class_stat(data[c], tick, "core")
            if not aper or not cper:
                continue
            print(f"| {LABEL[c]} | {tick} | {fmt(am)} ± {fmt(asd)} | {fmt(cm)} ± {fmt(csd)} | {cm - am:+.3f} | {'**yes**' if am >= cm else 'no'} |")
    print()

    # ---- dead cells ------------------------------------------------------------
    print("## Dead cells (`modes_retaining == 0`, used operator) by class, per seed\n")
    print("| condition | tick | " + " | ".join(CLASSES) + " |")
    print("|---|---|" + "---|" * len(CLASSES))
    for c in CONDITIONS:
        for tick in (20000, 30000):
            row = []
            for cls in CLASSES:
                counts = []
                for s in have[c]:
                    cp = at(data[c][s], tick)
                    if cp is None:
                        continue
                    mr = np.asarray(cp["used"]["per_cell"]["modes_retaining"])
                    counts.append(int((mr[data[c][s]["groups"][cls]] == 0).sum()))
                row.append(str(counts) if counts else "—")
            print(f"| {LABEL[c]} | {tick} | " + " | ".join(row) + " |")
    print("\n`modes_retaining` median per class at 30k (used operator; out of 12):\n")
    print("| condition | " + " | ".join(CLASSES) + " |")
    print("|---|" + "---|" * len(CLASSES))
    for c in CONDITIONS:
        row = []
        for cls in CLASSES:
            vals = []
            for s in have[c]:
                cp = at(data[c][s], 30000)
                if cp is None:
                    continue
                mr = np.asarray(cp["used"]["per_cell"]["modes_retaining"])
                vals.append(float(np.median(mr[data[c][s]["groups"][cls]])))
            row.append(f"{np.mean(vals):.1f} {[round(v,1) for v in vals]}" if vals else "—")
        print(f"| {LABEL[c]} | " + " | ".join(row) + " |")
    print()

    # ---- retention guard --------------------------------------------------------
    print("## Retention guard — no class below the frozen baseline beyond spread (tick 30k, 20k beside)\n")
    print("| condition | tick | " + " | ".join(f"{cls}: Δ / spread / ok" for cls in CLASSES) + " | passes |")
    print("|---|---|" + "---|" * (len(CLASSES) + 1))
    guard = {}
    for c in CONDITIONS:
        if c == BASELINE:
            continue
        for tick in (20000, 30000):
            row, ok_all, any_data = [], True, False
            for cls in CLASSES:
                m, sd, per = class_stat(data[c], tick, cls)
                bm, bsd, bper = class_stat(data[BASELINE], tick, cls)
                if not per or not bper:
                    row.append("—")
                    continue
                any_data = True
                spread = max(sd, bsd)
                delta = m - bm
                ok = delta >= -spread
                ok_all &= ok
                row.append(f"{delta:+.3f} / {fmt(spread)} / {'ok' if ok else '**FAIL**'}")
            if any_data:
                guard[(c, tick)] = ok_all
                print(f"| {LABEL[c]} | {tick} | " + " | ".join(row) + f" | {'**yes**' if ok_all else '**no**'} |")
    print()

    # ---- c = 0.1: the rate ratio -------------------------------------------------
    print("## `c = 0.1`: slows the collapse, stops it? (branch row 4 — the rate ratio)\n")
    print("Collapse is the fall in apex median ρ from 5k to 30k, per seed; the rate ratio is `c = 0.1`'s mean fall over `c = 1.0`'s within the same ρ=1 arm. *Still falling* is apex ρ at 30k below 20k by more than the seed spread.\n")
    print("| arm | c | fall 5k→30k (mean [per seed]) | rate ratio (c=0.1 / c=1.0) | 20k→30k Δ | still falling at 30k |")
    print("|---|---|---|---|---|---|")
    for arm, pair in (("ρ=1 off", ("rho1off-c1", "rho1off-c0.1")), ("ρ=1 on", ("rho1on-c1", "rho1on-c0.1"))):
        falls = {}
        for c in pair:
            per = []
            for s in have[c]:
                a, b = at(data[c][s], 5000), at(data[c][s], 30000)
                if a is None or b is None:
                    continue
                per.append(class_median(data[c][s], a, "apex") - class_median(data[c][s], b, "apex"))
            falls[c] = per
        for c in pair:
            per = falls[c]
            m20, sd20, p20 = class_stat(data[c], 20000, "apex")
            m30, sd30, p30 = class_stat(data[c], 30000, "apex")
            ratio = (np.mean(falls[pair[1]]) / np.mean(falls[pair[0]])) if falls[pair[0]] and falls[pair[1]] and np.mean(falls[pair[0]]) != 0 else float("nan")
            still = (m20 - m30) > max(sd20, sd30) if p20 and p30 else None
            print(f"| {arm} | {c.split('-c')[1]} | {fmt(np.mean(per)) if per else '—'} {[round(x,3) for x in per]} | {fmt(ratio) if c == pair[1] else ''} | {fmt(m30 - m20, 3) if p20 and p30 else '—'} | {'' if still is None else ('**yes**' if still else 'no')} |")
    print()

    # ---- the drive edges --------------------------------------------------------
    print("## The drive edges — what the ρ=1 flag does to #488's standing disagreement\n")
    print("Median over the 8 drive edges of the relative disagreement `|d| / (|a| + |b|)` between the two ends' broadcasts (#488's 1/3 is a fixed 2x scale mismatch), and the apex-side map's Frobenius norm; per seed.\n")
    print("| condition | tick | relative disagreement | apex-side map norm | |d| (absolute) |")
    print("|---|---|---|---|---|")
    for c in CONDITIONS:
        for tick in (5000, 20000, 30000):
            rel, nrm, absd = [], [], []
            for s in have[c]:
                cp = at(data[c][s], tick)
                if cp is None:
                    continue
                de = cp["drive_edges"]
                rel.append(float(np.median(de["relative"])))
                nrm.append(float(np.median(de["apex_map_norm"])))
                absd.append(float(np.median(np.abs(de["disagreement"]))))
            if rel:
                print(f"| {LABEL[c]} | {tick} | {[round(x,3) for x in rel]} | {[round(x,3) for x in nrm]} | {[f'{x:.2e}' for x in absd]} |")
    print()

    # ---- mechanism trace ----------------------------------------------------------
    print("## Mechanism trace at the apex, tick 30k — T0's reads under each condition (median over the 8 apex cells, per seed)\n")
    print("| condition | ‖ē‖ | ē direction stability | ē share along the drive lane | P1 |cos(v₁, h̄)| | ∇K PR |")
    print("|---|---|---|---|---|---|")
    for c in CONDITIONS:
        rows = {k: [] for k in ("p3_ebar_norm", "p3_direction_stability", "p4_ebar_share_drive", "p1_cos_v1_hbar", "grad_pr")}
        for s in have[c]:
            cp = at(data[c][s], 30000)
            if cp is None:
                continue
            idx = data[c][s]["groups"]["apex"]
            for k in rows:
                rows[k].append(float(np.nanmedian(arr(cp, k)[idx])))
        if rows["p3_ebar_norm"]:
            print(f"| {LABEL[c]} | {[f'{x:.1e}' for x in rows['p3_ebar_norm']]} | {[round(x,3) for x in rows['p3_direction_stability']]} | {[round(x,3) for x in rows['p4_ebar_share_drive']]} | {[round(x,3) for x in rows['p1_cos_v1_hbar']]} | {[round(x,2) for x in rows['grad_pr']]} |")
    print("\nCore, the same reads:\n")
    print("| condition | ‖ē‖ | ē direction stability | P1 |cos(v₁, h̄)| |")
    print("|---|---|---|---|")
    for c in CONDITIONS:
        rows = {k: [] for k in ("p3_ebar_norm", "p3_direction_stability", "p1_cos_v1_hbar")}
        for s in have[c]:
            cp = at(data[c][s], 30000)
            if cp is None:
                continue
            idx = data[c][s]["groups"]["core"]
            for k in rows:
                rows[k].append(float(np.nanmedian(arr(cp, k)[idx])))
        if rows["p3_ebar_norm"]:
            print(f"| {LABEL[c]} | {[f'{x:.1e}' for x in rows['p3_ebar_norm']]} | {[round(x,3) for x in rows['p3_direction_stability']]} | {[round(x,3) for x in rows['p1_cos_v1_hbar']]} |")
    print()

    # ---- travel ---------------------------------------------------------------------
    print("## Arm travel per tick (frozen world: the arm under the untrained command), last window before 30k\n")
    print("| condition | travel per tick, per seed |")
    print("|---|---|")
    for c in CONDITIONS:
        vals = []
        for s in have[c]:
            cp = at(data[c][s], 30000)
            if cp is not None:
                vals.append(cp["travel_per_tick"])
        if vals:
            print(f"| {LABEL[c]} | {[f'{x:.1e}' for x in vals]} |")
    print()

    # ---- branch table -----------------------------------------------------------------
    print("## Branch table — which rows fired (verdict horizon 30k)\n")
    w = verdict.get(30000)
    print("| row | reading | consequence |")
    print("|---|---|---|")
    if w is not None:
        g = guard.get((w, 30000))
        print(f"| a winner exists | **fired**: {LABEL[w]} (guard {'passes' if g else '**fails**'}) | T2b runs on it |")
        print("| no cell exceeds spread | did not fire | — |")
    else:
        print("| a winner exists | did not fire | — |")
        print("| no cell exceeds spread | **fired** | baseline is the winner; #526's `@when` fires (ledger row, no action) |")
    any_ge = False
    for c in CONDITIONS:
        am, _, aper = class_stat(data[c], 30000, "apex")
        cm, _, cper = class_stat(data[c], 30000, "core")
        if aper and cper and am >= cm:
            any_ge = True
    print(f"| apex ≥ core on some cell | {'**fired**' if any_ge else 'did not fire'} | {'recorded as a magnitude; T3 replicates' if any_ge else '—'} |")


if __name__ == "__main__":
    main()
