"""T0's readout: P1-P5 from the run JSONs, and which branch-table rows fired.

Reads `518-baseline-real-train-seed{42,43,44}-20000.json` beside this file and
#496's baseline JSONs for the pre-#466 column of P5. Prints markdown.

Usage::

    python prototypes/cold-start/T0/readout.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
P496 = HERE.parents[1] / "exogenous-variation-496"
CHANCE = 1.0 / np.sqrt(12.0)
GROUP_ORDER = ("apex", "core", "vision", "soma")
GROUP_LABEL = {
    "apex": "apex L7 drive-adjacent (8)",
    "core": "core L3–L6 (52)",
    "vision": "vision L1 (64)",
    "soma": "somatomotor L1 boundary-adjacent (6)",
}


def load(pattern: str, where: Path) -> dict[int, dict]:
    out = {}
    for path in sorted(where.glob(pattern)):
        d = json.loads(path.read_text())
        out[int(d["seed"])] = d
    return out


def groups_496(d: dict) -> dict[str, list[int]]:
    c = d["context"]
    out = {"soma": [], "vision": [], "apex": [], "core": []}
    for i, (col, lv, dr, bd) in enumerate(
        zip(c["columns"], c["levels"], c["drive_adjacent"], c["boundary_adjacent"])
    ):
        if col == "somatomotor" and lv == 1 and bd > 0:
            out["soma"].append(i)
        elif col == "vision" and lv == 1:
            out["vision"].append(i)
        elif col == "core" and dr:
            out["apex"].append(i)
        elif col == "core" and 3 <= lv <= 6:
            out["core"].append(i)
    return out


def at(d: dict, tick: int) -> dict:
    for c in d["checkpoints"]:
        if c["ticks"] == tick:
            return c
    raise KeyError(tick)


def arr(cp: dict, key: str) -> np.ndarray:
    return np.asarray(cp["per_cell"][key], dtype=float)


def fmt(x: float, p: int = 3) -> str:
    return "nan" if not np.isfinite(x) else f"{x:.{p}f}"


def r2(y: np.ndarray, X: np.ndarray) -> float:
    X1 = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    resid = y - X1 @ beta
    return 1.0 - resid @ resid / ((y - y.mean()) @ (y - y.mean()))


def main() -> None:
    runs = load("518-baseline-real-train-seed*-20000.json", HERE)
    base = load("496-baseline-real-train-seed*-100000.json", P496)
    if not runs:
        sys.exit("no completed T0 runs found")
    seeds = sorted(runs)
    d0 = runs[seeds[0]]
    g = {k: np.array(v) for k, v in d0["groups"].items()}
    ctx = d0["context"]
    columns = np.array(ctx["columns"])
    drive_adj = np.array(ctx["drive_adjacent"])
    print(f"# T0 readout — seeds {seeds}, surface `{d0['surface']['describe']}` on `{d0['surface']['branch']}`")
    print(f"band: {d0['surface']['band']}; (interior_m, boundary_m) = ({d0['surface']['interior_m']}, {d0['surface']['boundary_m']})")
    print(f"elapsed per seed: {[round(runs[s]['elapsed_minutes'], 1) for s in seeds]} min\n")

    # ---- P1 --------------------------------------------------------------
    print("## P1 alignment: |cos| between top right-singular vector of K_used and h̄")
    print(f"chance `1/√12 = {CHANCE:.3f}`. Mean ± std across the group's cells; per seed.\n")
    print("| tick | group | " + " | ".join(f"seed {s}" for s in seeds) + " |")
    print("|---|---|" + "---|" * len(seeds))
    p1_verdict = {}
    for tick in (2000, 5000, 10000, 20000):
        for grp in ("apex", "core", "vision"):
            cells = []
            for s in seeds:
                v = arr(at(runs[s], tick), "p1_cos_v1_hbar")[g[grp]]
                cells.append(f"{v.mean():.3f} ± {v.std():.3f}")
            print(f"| {tick} | {grp} | " + " | ".join(cells) + " |")
        if tick == 20000:
            for s in seeds:
                a = arr(at(runs[s], tick), "p1_cos_v1_hbar")
                apex, core = a[g["apex"]], a[g["core"]]
                p1_verdict[s] = {
                    "apex_mean": apex.mean(), "apex_std": apex.std(), "core_mean": core.mean(), "core_std": core.std(),
                    "holds": (apex.mean() - CHANCE) > max(apex.std(), core.std()),
                }
    print()
    for s in seeds:
        v = p1_verdict[s]
        print(f"- seed {s} @ 20k: apex {v['apex_mean']:.3f} (std {v['apex_std']:.3f}) vs chance {CHANCE:.3f}; core {v['core_mean']:.3f} (std {v['core_std']:.3f}) → **P1 {'holds' if v['holds'] else 'fails'}** (excess over chance {v['apex_mean']-CHANCE:+.3f} against spread {max(v['apex_std'], v['core_std']):.3f})")
    print("\nContext, unregistered — the gradient's own shape at the checkpoint tick (`∇K` participation ratio; |cos| of its top right-singular vector with that tick's `h` and with `h̄`):\n")
    print("| tick | group | ∇K PR | cos(∇K v1, h) | cos(∇K v1, h̄) | cos(K_used u1, h̄) | cos(top eigvec, h̄) |")
    print("|---|---|---|---|---|---|---|")
    for tick in (5000, 20000):
        for grp in ("apex", "core", "vision"):
            cp = at(runs[seeds[0]], tick)
            row = [np.nanmedian(arr(cp, k)[g[grp]]) for k in ("grad_pr", "grad_cos_v1_h", "grad_cos_v1_hbar", "cos_u1_hbar", "cos_eigvec_hbar")]
            print(f"| {tick} | {grp} | " + " | ".join(fmt(x) for x in row) + " |")
    print(f"\n(seed {seeds[0]}, medians over the group.)\n")

    # ---- P2 --------------------------------------------------------------
    print("## P2 excitation rank vs column identity: R² on log ρ")
    print("`ρ` is the used operator's; column identity is #477's `apex + somatomotor` (two dummies), and the full 4-way column one-hot beside it. Excitation rank is the window's uncentred participation ratio unless marked.\n")
    designs = {
        "column (#477: apex + soma)": lambda cp: np.column_stack([drive_adj, columns == "somatomotor"]).astype(float),
        "column (4-way one-hot)": lambda cp: np.column_stack([drive_adj, columns == "somatomotor", columns == "vision"]).astype(float),
        "PR total": lambda cp: arr(cp, "pr_total")[:, None],
        "log PR total": lambda cp: np.log(arr(cp, "pr_total"))[:, None],
        "PR total, centred": lambda cp: arr(cp, "pr_total_centred")[:, None],
        "log PR total, centred": lambda cp: np.log(np.maximum(arr(cp, "pr_total_centred"), 1e-9))[:, None],
        "PR exposed": lambda cp: arr(cp, "pr_exposed")[:, None],
        "PR private": lambda cp: arr(cp, "pr_private")[:, None],
        "PR interior (drive removed)": lambda cp: arr(cp, "pr_interior")[:, None],
        "log PR total + log PR total centred": lambda cp: np.column_stack([np.log(arr(cp, "pr_total")), np.log(np.maximum(arr(cp, "pr_total_centred"), 1e-9))]),
    }
    for tick in (5000, 20000):
        print(f"**tick {tick}**\n")
        print("| design | " + " | ".join(f"seed {s}" for s in seeds) + " |")
        print("|---|" + "---|" * len(seeds))
        for name, fn in designs.items():
            vals = []
            for s in seeds:
                cp = at(runs[s], tick)
                y = np.log(arr(cp, "rho_used"))
                vals.append(fmt(r2(y, fn(cp))))
            print(f"| {name} | " + " | ".join(vals) + " |")
        print()
    print("Context, unregistered — the mechanism's own variables on the same regression (R² on log ρ_used at 20k): the direction stability of `ē` and of `h` across consecutive windows, and the persistent fraction `‖ē‖ / RMS‖e‖`.\n")
    mech = {
        "ē direction stability": lambda cp: np.nan_to_num(arr(cp, "p3_direction_stability"))[:, None],
        "h direction stability": lambda cp: np.nan_to_num(arr(cp, "h_direction_stability"))[:, None],
        "log ‖ē‖": lambda cp: np.log(arr(cp, "p3_ebar_norm"))[:, None],
        "log (‖ē‖ / RMS‖e‖)": lambda cp: np.log(arr(cp, "p3_ebar_norm") / arr(cp, "e_rms"))[:, None],
        "ē stability + log ‖ē‖": lambda cp: np.column_stack([np.nan_to_num(arr(cp, "p3_direction_stability")), np.log(arr(cp, "p3_ebar_norm"))]),
        "column (#477) + ē stability": lambda cp: np.column_stack([drive_adj, columns == "somatomotor", np.nan_to_num(arr(cp, "p3_direction_stability"))]).astype(float),
    }
    print("| design | " + " | ".join(f"seed {s}" for s in seeds) + " |")
    print("|---|" + "---|" * len(seeds))
    for name, fn in mech.items():
        vals = []
        for s in seeds:
            cp = at(runs[s], 20000)
            vals.append(fmt(r2(np.log(arr(cp, "rho_used")), fn(cp))))
        print(f"| {name} | " + " | ".join(vals) + " |")
    print()
    print("Excitation rank per group (uncentred / centred), medians, seed 42, tick 20k:\n")
    print("| group | PR total | PR total centred | PR private | PR exposed | PR interior | energy share private / interior / drive | variance share private / interior / drive |")
    print("|---|---|---|---|---|---|---|---|")
    cp = at(runs[seeds[0]], 20000)
    for grp in GROUP_ORDER:
        idx = g[grp]
        m = lambda k: np.nanmedian(arr(cp, k)[idx])
        print(f"| {grp} | {m('pr_total'):.2f} | {m('pr_total_centred'):.2f} | {m('pr_private'):.2f} | {m('pr_exposed'):.2f} | {m('pr_interior'):.2f} | {m('energy_share_private'):.3f} / {m('energy_share_interior'):.3f} / {m('energy_share_drive'):.3f} | {m('variance_share_private'):.3f} / {m('variance_share_interior'):.3f} / {m('variance_share_drive'):.3f} |")
    print()
    print("Separation at 20k, seed 42: the 8 apex cells' PR range against core's:")
    for grp in ("apex", "core", "vision", "soma"):
        v = arr(cp, "pr_total")[g[grp]]
        vc = arr(cp, "pr_total_centred")[g[grp]]
        print(f"- {grp}: uncentred [{v.min():.2f}, {v.max():.2f}] median {np.median(v):.2f}; centred [{vc.min():.2f}, {vc.max():.2f}] median {np.median(vc):.2f}")
    print()

    # ---- P3 --------------------------------------------------------------
    print("## P3 persistent error: ‖ē‖ and direction stability")
    print("`‖ē‖` is the norm of the window-mean error vector; RMS is the per-tick error norm's RMS over the window; stability is cos between consecutive 1,000-tick window means. Medians over the group; per seed.\n")
    print("| tick | group | " + " | ".join(f"seed {s}: ‖ē‖ / RMS / stability" for s in seeds) + " |")
    print("|---|---|" + "---|" * len(seeds))
    for tick in (2000, 5000, 10000, 20000):
        for grp in ("apex", "core", "vision", "soma"):
            cells = []
            for s in seeds:
                cp = at(runs[s], tick)
                idx = g[grp]
                cells.append(f"{np.median(arr(cp,'p3_ebar_norm')[idx]):.2e} / {np.median(arr(cp,'e_rms')[idx]):.2e} / {fmt(np.nanmedian(arr(cp,'p3_direction_stability')[idx]))}")
            print(f"| {tick} | {grp} | " + " | ".join(cells) + " |")
    print()
    for s in seeds:
        cp = at(runs[s], 20000)
        a_norm = np.median(arr(cp, "p3_ebar_norm")[g["apex"]])
        a_stab = np.nanmedian(arr(cp, "p3_direction_stability")[g["apex"]])
        v_norm = np.median(arr(cp, "p3_ebar_norm")[g["vision"]])
        v_stab = np.nanmedian(arr(cp, "p3_direction_stability")[g["vision"]])
        holds = (1e-3 <= a_norm <= 1e-2) and a_stab > 0.9 and v_norm < a_norm and v_stab < a_stab
        print(f"- seed {s} @ 20k: apex ‖ē‖ {a_norm:.2e} (in 1e-3–1e-2: {1e-3 <= a_norm <= 1e-2}), stability {a_stab:.3f} (> 0.9: {a_stab > 0.9}); vision ‖ē‖ {v_norm:.2e}, stability {v_stab:.3f} → **P3 {'holds' if holds else 'fails as stated'}**")
    print("\nApex direction stability along the run (seed 42, cos between block means ending at t and t−1000):\n")
    blk = runs[seeds[0]]["blocks"]
    apex_idx = g["apex"]
    # Recompute stability from the npz block means for every block.
    npz = np.load(HERE / f"518-baseline-real-train-seed{seeds[0]}-20000.npz")
    me = npz["block_mean_e"]  # [blocks, cells, n]
    ends = npz["block_end_ticks"]
    line = []
    for b in range(1, len(ends)):
        a, p = me[b][apex_idx], me[b - 1][apex_idx]
        cs = np.abs((a * p).sum(-1)) / np.maximum(np.linalg.norm(a, axis=-1) * np.linalg.norm(p, axis=-1), 1e-30)
        line.append(f"{ends[b]}: {np.median(cs):.3f}")
    print(", ".join(line) + "\n")
    print("Apex ‖ē‖ per block (median over 8 cells): " + ", ".join(f"{ends[b]}: {np.median(np.linalg.norm(me[b][apex_idx], axis=-1)):.2e}" for b in range(len(ends))) + "\n")

    # ---- P4 --------------------------------------------------------------
    print("## P4 decomposition at the apex")
    print("Share of `‖ē‖²` in each piece of the apex stalk: private block (19 columns), exposed block (13), the drive lane's own direction (the unit row of the apex-side drive map, learned), and the span of the four interior maps' rows. Share of the evidence stream's energy (uncentred) and variance (centred) in private / interior / drive over the window. Medians over the 8 apex cells; per seed.\n")
    print("| tick | seed | ē: private | ē: exposed | ē: drive lane | ē: interior rowspace | evidence energy priv/int/drive | evidence variance priv/int/drive |")
    print("|---|---|---|---|---|---|---|---|")
    p4 = {}
    for tick in (5000, 20000):
        for s in seeds:
            cp = at(runs[s], tick)
            idx = g["apex"]
            m = lambda k: np.nanmedian(arr(cp, k)[idx])
            print(f"| {tick} | {s} | {m('p4_ebar_share_private'):.3f} | {m('p4_ebar_share_exposed'):.3f} | {m('p4_ebar_share_drive'):.3f} | {m('p4_ebar_share_interior_rowspace'):.3f} | {m('energy_share_private'):.3f} / {m('energy_share_interior'):.3f} / {m('energy_share_drive'):.3f} | {m('variance_share_private'):.3f} / {m('variance_share_interior'):.3f} / {m('variance_share_drive'):.3f} |")
            if tick == 20000:
                p4[s] = (m("p4_ebar_share_drive"), m("p4_ebar_share_private"), m("p4_ebar_share_interior_rowspace"))
    print()
    print("The exposed block is 13-dimensional and the four interior maps' rows span up to 12 of it, so *interior rowspace* is not a complement of the drive lane and is reported as context only. The comparator is the drive lane against **everything else in the exposed block** (its orthogonal complement, `1 − drive − private`):\n")
    for s in seeds:
        dr, pr, ir = p4[s]
        rest = 1.0 - dr - pr
        largest = dr > max(pr, rest)
        print(f"- seed {s} @ 20k: drive lane {dr:.3f} vs private {pr:.3f} vs rest of exposed {rest:.3f} → drive lane carries the largest share: **{largest}**")
    print("\nPer apex cell at 20k, seed 42 (ē share drive / private / interior-rowspace; ‖ē‖; ρ_used; PR total):\n")
    cp = at(runs[seeds[0]], 20000)
    for i in g["apex"]:
        print(f"- cell row {i}: {arr(cp,'p4_ebar_share_drive')[i]:.3f} / {arr(cp,'p4_ebar_share_private')[i]:.3f} / {arr(cp,'p4_ebar_share_interior_rowspace')[i]:.3f}; ‖ē‖ {arr(cp,'p3_ebar_norm')[i]:.2e}; ρ {arr(cp,'rho_used')[i]:.3f}; PR {arr(cp,'pr_total')[i]:.2f}")
    print()

    # ---- P5 --------------------------------------------------------------
    print("## P5 re-baseline: group ρ(K) at 5k and 20k, both surfaces named")
    print("Median over the group's cells per seed, then mean over seeds (per-seed in brackets). *This surface*: forward normalisation (#466 / PR #513) + (3,4); `ρ` of the **used** operator, raw beside it. *#496 surface*: post-step projection band + (3,4), `main` at `10183c9`, `ρ` of the stored (projected) `K`.\n")
    print("| tick | group | this surface, ρ(used) | this surface, ρ(raw) | #496 surface (pre-#466) |")
    print("|---|---|---|---|---|")
    p5 = {}
    bseeds = sorted(base)
    for tick in (5000, 20000):
        for grp in GROUP_ORDER:
            used = [np.median(arr(at(runs[s], tick), "rho_used")[g[grp]]) for s in seeds]
            raw = [np.median(arr(at(runs[s], tick), "rho_raw")[g[grp]]) for s in seeds]
            old = []
            for s in bseeds:
                gb = groups_496(base[s])
                old.append(np.median(np.asarray(at(base[s], tick)["per_cell"]["rho_K"])[gb[grp]]))
            p5[(tick, grp)] = (np.mean(used), used)
            print(f"| {tick} | {GROUP_LABEL[grp]} | **{np.mean(used):.3f}** [{', '.join(f'{x:.3f}' for x in used)}] | {np.mean(raw):.3f} [{', '.join(f'{x:.3f}' for x in raw)}] | {np.mean(old):.3f} [{', '.join(f'{x:.3f}' for x in old)}] |")
    print()
    for tick in (5000, 20000):
        a, per_a = p5[(tick, "apex")]
        c, per_c = p5[(tick, "core")]
        below = [x < y for x, y in zip(per_a, per_c)]
        print(f"- @ {tick}: apex {a:.3f} vs core {c:.3f}; apex below core on {sum(below)}/{len(below)} seeds (gap {c - a:+.3f})")
    print("\nDead cells (`modes_retaining == 0`, used operator) at 20k, per seed: " + ", ".join(str(int((np.asarray(at(runs[s], 20000)['used']['per_cell']['modes_retaining']) == 0).sum())) for s in seeds))
    print("Apex `modes_retaining` (used) at 20k, per seed, median: " + ", ".join(f"{np.median(np.asarray(at(runs[s], 20000)['used']['per_cell']['modes_retaining'])[g['apex']]):.1f}" for s in seeds))
    print("Travel per tick at 20k window, per seed: " + ", ".join(f"{at(runs[s], 20000)['travel_per_tick']:.2e}" for s in seeds))
    print()

    # ---- verdict --------------------------------------------------------
    print("## Branch table")
    p1_all = all(p1_verdict[s]["holds"] for s in seeds)
    p1_42 = p1_verdict[seeds[0]]["holds"]
    p4_42 = p4[seeds[0]][0] > max(p4[seeds[0]][1], 1.0 - p4[seeds[0]][0] - p4[seeds[0]][1])
    a20, per_a = p5[(20000, "apex")]
    c20, per_c = p5[(20000, "core")]
    surprise = not all(x < y for x, y in zip(per_a, per_c))
    print(f"- P1 (seed 42, the pre-registered read): **{'holds' if p1_42 else 'fails'}**; on all seeds: {p1_all}")
    print(f"- P4 drive lane carries the largest share of ē (seed 42): **{p4_42}**")
    print(f"- P5 apex below core on this surface at 20k: **{not surprise}** ({'no surprise' if not surprise else 'SURPRISE — ledger row'})")


if __name__ == "__main__":
    main()
