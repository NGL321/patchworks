"""T2's readout: the four pre-registered reads, and which branch-table rows fired.

Reads `522-{condition}-tb{T_b}-seed{42,43,44}-post30000.json` beside this file
(with ``--inflight``, runs still going too) and T1's frozen baseline from
`../T1/`. Prints markdown.

**Conventions, stated once, because the ticket states none.**

* A **class figure** is the median over the class's cells, per seed; the
  published number is the mean of the per-seed medians and the **spread** is
  their standard deviation across seeds. T1's convention, inherited verbatim.
* **Beyond spread** is a difference exceeding the larger of the two arms'
  spreads. T1's convention, inherited.
* The **verdict horizon** is the post-phase `+30k` checkpoint, with `+20k`
  beside it (#178's deepest-reached rule); *during the phase* is the deepest
  in-phase checkpoint at which the supply was still on.
* **Reach** is reported as a **magnitude** -- the peak paired private deviation
  over the arithmetic floor at the same tick -- not as a predicate. #481's
  idiom, and on a three-hop dome the on/off form does not discriminate.
* **Travel > 0** is read against T1's frozen baseline's own last-window travel
  per tick rather than against literal zero: the frozen baseline's arm sits at
  its stops and still registers 1e-5 to 1e-3 per tick, so literal zero is not
  the question the branch row is asking.
* **Priming** is the composed rim-to-apex effective rank over the **patch**
  chains (256 of them, the only stratum that is genuinely many; the other three
  are reported beside it and never averaged in, per #181).

Usage::

    python prototypes/cold-start/T2/readout.py [--inflight] [--tb 5000]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
T1 = HERE.parent / "T1"
#: Z is a control, not one of the ticket's four, and is listed last everywhere
#: so no table reads as if the sweep had five members.
CONDITIONS = ("A", "B", "C", "D", "Z")
SWEEP = ("A", "B", "C", "D")
TBS = (5000, 20000)
POST = 30000
LABEL = {
    "A": "A — motor, small, shuffled (today's babble)",
    "B": "B — sensory, large, ordered (the wave)",
    "C": "C — sensory, large, shuffled",
    "D": "D — sensory, small, ordered",
    "Z": "**Z — no supply at all (control)**",
}
#: The control each condition's priming is read against: the same energy with
#: the structure removed (B vs C) and the same structure at the other wall (D vs A).
CONTROL = {"B": "C", "D": "A"}
CLASSES = ("apex", "core", "vision", "soma")
CLASS_LABEL = {
    "apex": "apex (core, drive-adjacent; 8)",
    "core": "core (one level; 16)",
    "vision": "vision L1 (64)",
    "soma": "somatomotor L1 boundary-adjacent (6)",
}


def load(inflight: bool) -> dict:
    out: dict = {(c, tb): {} for c in CONDITIONS for tb in TBS}
    for c in CONDITIONS:
        for tb in TBS:
            patterns = [f"522-{c}-tb{tb}-seed*-post{POST}.json"]
            if inflight:
                patterns.append(f"522-{c}-tb{tb}-seed*-post{POST}.inflight.json")
            for pat in patterns:
                for path in sorted(HERE.glob(pat)):
                    d = json.loads(path.read_text())
                    d["_inflight"] = path.name.endswith(".inflight.json")
                    out[(c, tb)].setdefault(int(d["seed"]), d)
    return out


def load_t1() -> dict[int, dict]:
    out = {}
    for path in sorted(T1.glob("521-rho1off-c1-seed*-30000.json")):
        d = json.loads(path.read_text())
        out[int(d["seed"])] = d
    return out


def at(d: dict, ticks: int) -> dict | None:
    for cp in d["checkpoints"]:
        if cp["ticks"] == ticks:
            return cp
    return None


def post(d: dict, since: int) -> dict | None:
    """The post-phase checkpoint `since` ticks after the phase ended."""
    return at(d, d["tb"] + since)


def last_on(d: dict) -> dict | None:
    """The deepest in-phase checkpoint at which the supply was still on (the reach fork)."""
    live = [cp for cp in d["checkpoints"] if cp["phase"] == "induced" and cp["reach"]]
    return live[-1] if live else None


def during(d: dict) -> dict | None:
    """The deepest in-phase checkpoint strictly before the phase ends.

    Equal to :func:`last_on`'s tick for every condition of the sweep -- the
    anneal reaches zero exactly at `T_b`, so the last checkpoint carrying a live
    supply is the last one before it -- and defined for the zero-supply control
    too, which has no fork to hang a checkpoint off. That is what lets the rank
    table put Z in the same row space as the four.
    """
    live = [cp for cp in d["checkpoints"] if cp["phase"] == "induced" and cp["ticks"] < d["tb"]]
    return live[-1] if live else None


def fmt(x, p: int = 3) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:.{p}f}"


def sci(x, p: int = 2) -> str:
    return "—" if x is None or not np.isfinite(x) else f"{x:.{p}e}"


def stat(values: list[float]) -> tuple[float, float, list[float]]:
    """(mean of per-seed values, std across seeds, the per-seed values)."""
    if not values:
        return float("nan"), float("nan"), []
    v = np.asarray(values, dtype=float)
    return float(v.mean()), float(v.std()), [float(x) for x in v]


def class_medians(runs: dict[int, dict], pick, cls: str, key: str = "rho_used") -> tuple[float, float, list[float]]:
    vals = []
    for s in sorted(runs):
        d = runs[s]
        cp = pick(d)
        if cp is None:
            continue
        idx = d["groups"][cls]
        vals.append(float(np.median(np.asarray(cp["per_cell"][key], dtype=float)[idx])))
    return stat(vals)


def composed_medians(runs: dict[int, dict], pick, kind: str = "patch") -> tuple[float, float, list[float]]:
    vals = []
    for s in sorted(runs):
        cp = pick(runs[s])
        if cp is None:
            continue
        by = cp["composed"]["by_kind"].get(kind)
        if by:
            vals.append(float(by["effective_rank_median"]))
    return stat(vals)


def travel_medians(runs: dict[int, dict], pick) -> tuple[float, float, list[float]]:
    vals = []
    for s in sorted(runs):
        cp = pick(runs[s])
        if cp is not None:
            vals.append(float(cp["travel_per_tick"]))
    return stat(vals)


def stratum_medians(runs: dict[int, dict], pick, stratum: str, key: str) -> tuple[float, float, list[float]]:
    vals = []
    for s in sorted(runs):
        cp = pick(runs[s])
        if cp is None:
            continue
        by = cp["edges"]["by_stratum"].get(stratum)
        if by:
            vals.append(float(by[key]))
    return stat(vals)


def reach_medians(runs: dict[int, dict], cls: str, key: str = "ratio_private_median") -> tuple[float, float, list[float]]:
    vals = []
    for s in sorted(runs):
        cp = last_on(runs[s])
        if cp is None:
            continue
        vals.append(float(cp["reach"]["by_class"][cls][key]))
    return stat(vals)


def beyond(a: tuple, b: tuple) -> tuple[float, float, bool]:
    """(delta, spread, a exceeds b beyond spread)."""
    if not a[2] or not b[2]:
        return float("nan"), float("nan"), False
    spread = max(a[1], b[1])
    delta = a[0] - b[0]
    return delta, spread, bool(delta > spread)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inflight", action="store_true")
    p.add_argument("--tb", type=int, nargs="+", default=list(TBS))
    args = p.parse_args()
    data = load(args.inflight)
    t1 = load_t1()
    tbs = [tb for tb in args.tb]
    any_run = next((data[(c, tb)][s] for c in CONDITIONS for tb in tbs for s in data[(c, tb)]), None)
    if any_run is None:
        raise SystemExit("no T2 runs found")

    surf, spec, sup = any_run["surface"], any_run["dome_spec"], any_run["supply"]
    print(f"# T2 readout — surface `{surf['describe']}` on `{surf['branch']}`")
    print(
        f"band: {surf['band']}; (interior_m, boundary_m) = ({surf['interior_m']}, {surf['boundary_m']}); "
        f"shallow dome core_degree {spec['core_degree']}, {spec['cells']} cells / {spec['edges']} edges; "
        f"**baseline build** (ρ=1 off, c=1). Induced phase T_b, annealed `{sup['anneal']}`, then {POST} ticks "
        f"with the world arranged and the supply off.\n"
    )
    print("## Runs\n")
    print("| condition | T_b | seeds | complete | min/seed | A₀ |")
    print("|---|---|---|---|---|---|")
    for tb in tbs:
        for c in CONDITIONS:
            runs = data[(c, tb)]
            have = sorted(runs)
            done = [s for s in have if not runs[s]["_inflight"] and post(runs[s], POST) is not None]
            a0 = runs[have[0]]["supply"]["a0_peak"] if have else float("nan")
            mins = [round(runs[s].get("elapsed_minutes", float("nan")), 1) for s in have]
            print(f"| {LABEL[c]} | {tb} | {have} | {done} | {mins} | {fmt(a0, 4)} |")
    print()
    if any_run["supply"]["wall"] == "sensory" or True:
        cal = None
        for c in ("B", "C", "D"):
            for tb in tbs:
                for s in data[(c, tb)]:
                    cal = data[(c, tb)][s]["supply"].get("calibration")
                    break
        loops = any_run["loops"]
        print(
            f"**Derived constants.** Babble correlation time τ = median `world_loop(c)` over the six L1 somatomotor "
            f"cells = **{loops['tau_motor']}** ticks ({loops['soma_l1_world_loops']}), so φ = exp(−1/τ); the front "
            f"advances one patch per **{loops['tau_vision']}** ticks, the L1 vision cells' median `world_loop(c)` "
            f"({loops['vision_l1_world_loops_hist']}). "
        )
        if cal:
            print(
                f"The sensory wall's **small** amplitude is calibrated, not chosen: motor babble at the bound moves "
                f"the render with RMS **{cal['babble_render_rms']:.4f}** stalk units, and the front at peak 1 has "
                f"time-averaged RMS {cal['front_rms_at_peak_one']:.4f}, so D's peak is "
                f"**{cal['small_peak_derived']:.4f}** against B and C's 1.0 — D vs A is a structure comparison at "
                f"matched sensory energy.\n"
            )

    for tb in tbs:
        runs_at = {c: data[(c, tb)] for c in CONDITIONS}
        if not any(runs_at[c] for c in CONDITIONS):
            continue
        print(f"\n---\n\n# T_b = {tb}\n")

        # ---- reach ------------------------------------------------------------
        print("## Reach — ADR-0026's paired counterfactual, rules off, at the deepest in-phase checkpoint with the supply on\n")
        print("Peak paired private-feature deviation over `eps_f32·‖state‖` at the same tick, median over the class's cells; "
              "mean of per-seed medians ± spread. **A ratio above 1 is a deviation that arrived**; the magnitude is the reading.\n")
        print("| condition | A at the fork | " + " | ".join(CLASS_LABEL[c] for c in CLASSES) + " |")
        print("|---|---|" + "---|" * len(CLASSES))
        for c in CONDITIONS:
            if not runs_at[c]:
                continue
            amps = [last_on(runs_at[c][s])["reach"]["amplitude_at_fork"] for s in sorted(runs_at[c]) if last_on(runs_at[c][s])]
            cells = []
            for cls in CLASSES:
                m, sd, per = reach_medians(runs_at[c], cls)
                cells.append(f"**{m:.3g}** ± {sd:.2g}" if per else "—")
            print(f"| {LABEL[c]} | {fmt(np.mean(amps), 4) if amps else '—'} | " + " | ".join(cells) + " |")
        print("\nFraction of a class's cells whose peak cleared the floor, and the apex's peak in absolute terms:\n")
        print("| condition | apex above-floor fraction | apex peak ‖Δ‖ | apex floor | apex peak tick (of 64) |")
        print("|---|---|---|---|---|")
        for c in CONDITIONS:
            if not runs_at[c]:
                continue
            f_, _, fper = reach_medians(runs_at[c], "apex", "above_floor_fraction")
            pk, _, pper = reach_medians(runs_at[c], "apex", "peak_private_median")
            fl, _, _ = reach_medians(runs_at[c], "apex", "floor_median")
            tk, _, _ = reach_medians(runs_at[c], "apex", "peak_tick_median")
            if fper:
                print(f"| {LABEL[c]} | {fmt(f_)} | {sci(pk)} | {sci(fl)} | {fmt(tk, 1)} |")
        print()

        # ---- rank -------------------------------------------------------------
        print("## Rank — per-edge excitation rank at the deepest interior edges (core–apex, m_e = 3), during the phase\n")
        print("Median over the 32 core–apex edges' two ends, both forms published (ledger row 1: the uncentred form is "
              "DC-dominated on this surface and the centred one is reported beside it, never in place of it).\n")
        print("| condition | uncentred PR | centred PR | against m_e | disagreement energy | map effective rank |")
        print("|---|---|---|---|---|---|")
        for c in CONDITIONS:
            if not runs_at[c]:
                continue
            u = stratum_medians(runs_at[c], during, "core-apex", "pr_uncentred_median")
            cm = stratum_medians(runs_at[c], during, "core-apex", "pr_centred_median")
            de = stratum_medians(runs_at[c], during, "core-apex", "disagreement_energy_median")
            me = stratum_medians(runs_at[c], during, "core-apex", "map_effective_rank_median")
            if u[2]:
                print(f"| {LABEL[c]} | {fmt(u[0])} ± {fmt(u[1])} | **{fmt(cm[0])}** ± {fmt(cm[1])} | 3 | {sci(de[0])} | {fmt(me[0])} |")
        print("\nThe same, at the shallower strata, centred (so the taper by depth is visible):\n")
        strata = ("sensory-rim", "L1-L1", "L1-core", "core-core", "core-apex", "drive")
        print("| condition | " + " | ".join(strata) + " |")
        print("|---|" + "---|" * len(strata))
        for c in CONDITIONS:
            if not runs_at[c]:
                continue
            row = []
            for st in strata:
                m, sd, per = stratum_medians(runs_at[c], during, st, "pr_centred_median")
                row.append(f"{fmt(m)} ± {fmt(sd)}" if per else "—")
            print(f"| {LABEL[c]} | " + " | ".join(row) + " |")
        print()

        # ---- priming ----------------------------------------------------------
        print("## Priming — composed rim-to-apex effective rank at +30k, world arranged, supply off\n")
        print("Median over the 256 **patch** chains (the graph's own shortest edge path from each rim cell to an apex "
              "cell, composed); the other rim strata beside it, never averaged in (#181). The pre-registered comparison "
              "is each ordered condition against its shuffled control: **B vs C** and **D vs A**.\n")
        print("| condition | at construction | at end of phase | +20k | **+30k** | spread |")
        print("|---|---|---|---|---|---|")
        for c in CONDITIONS:
            if not runs_at[c]:
                continue
            con = [runs_at[c][s]["composed_at_construction"]["patch"]["effective_rank_median"] for s in sorted(runs_at[c])]
            eop = composed_medians(runs_at[c], lambda d: at(d, d["tb"]))
            p20 = composed_medians(runs_at[c], lambda d: post(d, 20000))
            p30 = composed_medians(runs_at[c], lambda d: post(d, POST))
            print(f"| {LABEL[c]} | {fmt(np.mean(con), 4)} | {fmt(eop[0], 4)} | {fmt(p20[0], 4)} | **{fmt(p30[0], 4)}** | {fmt(p30[1], 4)} |")
        print("\n**The pre-registered contrasts**, at +30k:\n")
        print("| ordered | control | ordered ER | control ER | Δ | spread | beats control |")
        print("|---|---|---|---|---|---|---|")
        primed = {}
        for c, ctrl in CONTROL.items():
            a = composed_medians(runs_at[c], lambda d: post(d, POST))
            b = composed_medians(runs_at[ctrl], lambda d: post(d, POST))
            delta, spread, ok = beyond(a, b)
            primed[c] = ok
            print(f"| {LABEL[c]} | {LABEL[ctrl]} | {fmt(a[0], 4)} | {fmt(b[0], 4)} | {delta:+.4f} | {fmt(spread, 4)} | **{ok}** |")
        print("\n**Against the zero-supply control**, which is what says whether *anything* was laid down "
              "rather than which member laid down most:\n")
        print("| condition | ER at +30k | Z at +30k | Δ vs Z | spread | above Z |")
        print("|---|---|---|---|---|---|")
        z = composed_medians(runs_at["Z"], lambda d: post(d, POST))
        for c in SWEEP:
            if not runs_at[c]:
                continue
            a = composed_medians(runs_at[c], lambda d: post(d, POST))
            delta, spread, ok = beyond(a, z)
            print(f"| {LABEL[c]} | {fmt(a[0], 4)} | {fmt(z[0], 4)} | {delta:+.4f} | {fmt(spread, 4)} | **{ok}** |")
        print("\nThe other rim strata at +30k, for the record:\n")
        kinds = ("patch", "proprioceptive", "touch", "actuator")
        print("| condition | " + " | ".join(kinds) + " |")
        print("|---|" + "---|" * len(kinds))
        for c in CONDITIONS:
            if not runs_at[c]:
                continue
            row = []
            for kind in kinds:
                m, sd, per = composed_medians(runs_at[c], lambda d: post(d, POST), kind)
                row.append(f"{fmt(m, 4)} ± {fmt(sd, 4)}" if per else "—")
            print(f"| {LABEL[c]} | " + " | ".join(row) + " |")
        print()

        # ---- travel -----------------------------------------------------------
        base_travel = stat([at(t1[s], 30000)["travel_per_tick"] for s in sorted(t1) if at(t1[s], 30000)])
        print("## Travel — arm travel per tick under the graph's own command, post-phase windows\n")
        print(f"T1's frozen baseline reads **{sci(base_travel[0])}** per tick in its last window (the arm at its stops), "
              "which is what *travel > 0* is read against rather than literal zero.\n")
        print("| condition | +1k | +5k | +10k | +20k | **+30k** | spread at +30k | above the frozen baseline |")
        print("|---|---|---|---|---|---|---|---|")
        travelled = {}
        for c in CONDITIONS:
            if not runs_at[c]:
                continue
            row = []
            for since in (1000, 5000, 10000, 20000, POST):
                m, sd, per = travel_medians(runs_at[c], lambda d, k=since: post(d, k))
                row.append(sci(m) if per else "—")
            last = travel_medians(runs_at[c], lambda d: post(d, POST))
            delta, spread, ok = beyond(last, base_travel)
            travelled[c] = (ok, last[0])
            print(f"| {LABEL[c]} | " + " | ".join(row) + f" | {sci(last[1])} | **{ok}** |")
        print()

        # ---- retention guard --------------------------------------------------
        print("## Retention guard — apex and soma ρ(K) at +30k, not below T1's frozen baseline beyond spread\n")
        print("T1's baseline is 30k ticks from construction; a T2 run at +30k has run "
              f"{tb + POST} in total, so the comparator is matched on **post-phase** ticks and not on total ticks. "
              "Stated rather than corrected: no frozen run exists at the longer horizon.\n")
        print("| condition | " + " | ".join(f"{CLASS_LABEL[c]}" for c in CLASSES) + " | guard |")
        print("|---|" + "---|" * (len(CLASSES) + 1))
        base = {cls: class_medians(t1, lambda d: at(d, 30000), cls) for cls in CLASSES}
        print("| **T1 frozen baseline @30k** | " + " | ".join(f"{fmt(base[cls][0])} ± {fmt(base[cls][1])}" for cls in CLASSES) + " | — |")
        guard = {}
        for c in CONDITIONS:
            if not runs_at[c]:
                continue
            cells, ok_all, seen_any = [], True, False
            for cls in CLASSES:
                m = class_medians(runs_at[c], lambda d: post(d, POST), cls)
                if not m[2]:
                    cells.append("—")
                    continue
                seen_any = True
                spread = max(m[1], base[cls][1])
                delta = m[0] - base[cls][0]
                ok = delta >= -spread
                if cls in ("apex", "soma"):
                    ok_all &= ok
                cells.append(f"{fmt(m[0])} ± {fmt(m[1])} ({delta:+.3f}{'' if ok else ' **FAIL**'})")
            if seen_any:
                guard[c] = ok_all
            print(f"| {LABEL[c]} | " + " | ".join(cells) + f" | {('**passes**' if ok_all else '**FAILS**') if seen_any else '—'} |")
        print("\nApex ρ(K) through the run, so the phase's own effect is visible beside the post-phase state:\n")
        print("| condition | end of phase | +1k | +5k | +10k | +20k | +30k |")
        print("|---|---|---|---|---|---|---|")
        for c in CONDITIONS:
            if not runs_at[c]:
                continue
            row = [fmt(class_medians(runs_at[c], lambda d: at(d, d["tb"]), "apex")[0])]
            for since in (1000, 5000, 10000, 20000, POST):
                row.append(fmt(class_medians(runs_at[c], lambda d, k=since: post(d, k), "apex")[0]))
            print(f"| {LABEL[c]} | " + " | ".join(row) + " |")
        print()

        # ---- mechanism ---------------------------------------------------------
        print("## Mechanism at the apex (T0's reads, +30k) — ledger row 1's coherence variable\n")
        print("| condition | ‖ē‖ | ē direction stability | ē share along the drive lane | dead cells |")
        print("|---|---|---|---|---|")
        for c in CONDITIONS:
            if not runs_at[c]:
                continue
            e = class_medians(runs_at[c], lambda d: post(d, POST), "apex", "p3_ebar_norm")
            st = class_medians(runs_at[c], lambda d: post(d, POST), "apex", "p3_direction_stability")
            dr = class_medians(runs_at[c], lambda d: post(d, POST), "apex", "p4_ebar_share_drive")
            dead = []
            for s in sorted(runs_at[c]):
                cp = post(runs_at[c][s], POST)
                if cp:
                    dead.append(int((np.asarray(cp["used"]["per_cell"]["modes_retaining"]) == 0).sum()))
            if e[2]:
                print(f"| {LABEL[c]} | {sci(e[0])} | {fmt(st[0])} | {fmt(dr[0])} | {dead} |")
        print()

        # ---- branch table -------------------------------------------------------
        print("## Branch table — which rows fired\n")
        passes = [c for c in CONTROL if primed.get(c) and travelled.get(c, (False, 0))[0]]
        moving = [c for c in SWEEP if travelled.get(c, (False, 0))[0]]
        print("| row | reading | consequence |")
        print("|---|---|---|")
        if passes:
            best = max(passes, key=lambda c: (composed_medians(runs_at[c], lambda d: post(d, POST))[0], travelled[c][1]))
            ok_guard = [c for c in passes if guard.get(c)]
            winner = best if guard.get(best) else (max(ok_guard, key=lambda c: composed_medians(runs_at[c], lambda d: post(d, POST))[0]) if ok_guard else None)
            print(f"| a condition passes priming beyond its control **and** travels | **fired**: {passes} | T2b runs {winner or 'nothing — no passing condition clears the guard (ledger row)'} |")
        elif moving:
            best = max(moving, key=lambda c: travelled[c][1])
            print(f"| none passes priming, some travels | **fired**: {moving} | T2b runs {best} (best travel); **ledger row: priming not shown** |")
            print("| a condition passes priming and travels | did not fire | — |")
        else:
            print("| none travels post-phase | **fired** | wave 3 skipped; T3 replicates T1's winner alone; **ledger row: curiosity drive owed** |")
        bc = beyond(composed_medians(runs_at["B"], lambda d: post(d, POST)), composed_medians(runs_at["C"], lambda d: post(d, POST)))
        ca = beyond(composed_medians(runs_at["C"], lambda d: post(d, POST)), composed_medians(runs_at["A"], lambda d: post(d, POST)))
        b_approx_c = abs(bc[0]) <= bc[1] if np.isfinite(bc[0]) else False
        print(f"| B beats C beyond spread | {'**fired**' if bc[2] else 'did not fire'} (Δ {bc[0]:+.4f}, spread {fmt(bc[1], 4)}) | "
              f"{'structure matters: the wave pattern is fixed as T3s default' if bc[2] else '—'} |")
        print(f"| C beats A beyond spread and B ≈ C | {'**fired**' if (ca[2] and b_approx_c) else 'did not fire'} "
              f"(C−A {ca[0]:+.4f} / {fmt(ca[1], 4)}; B−C {bc[0]:+.4f}) | "
              f"{'amplitude is what mattered: the structure axis is dropped from T3' if (ca[2] and b_approx_c) else '—'} |")
        failed = [c for c in SWEEP if c in guard and not guard[c]]
        print(f"| retention guard fails on the winning condition | {'**fired**' if failed else 'did not fire'} "
              f"({'fails: ' + str(failed) if failed else 'every condition passes'}) | "
              f"{'the winner is the best condition that passes' if failed else '—'} |")
        print()


if __name__ == "__main__":
    main()
