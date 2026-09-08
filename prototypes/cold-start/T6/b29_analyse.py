"""T6 ([B29](#585)): the tables the readout quotes, off the saved JSON.

`b29_holonomy.py` writes one record per arm and `b29_theory.py` one for the algebra.
This reads them back and prints the five tables #585 asks for, so no number in
`READOUT-585.md` is retyped by hand.

Usage::

    python prototypes/cold-start/T6/b29_analyse.py
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

_HERE = Path(__file__).resolve().parent

ARM_ORDER = ["shipped", "reserve_p8", "reserve_p16", "reserve_p24"]


def _load_arms(pattern: str) -> dict:
    out = {}
    for path in sorted(glob.glob(str(_HERE / pattern))):
        d = json.load(open(path, encoding="utf-8"))
        out[d["arm"]] = d
    return out


#: The last checkpoint at which the sandbox is still moving. `PlanarPushSandbox` motion
#: collapses between 1,000 and 2,000 ticks on every cold-start arm and stays collapsed
#: (#572's `world_std_*` on four arm-seeds; #518's own published `travel_window` records
#: the same shape). Every reading past this is taken against a world with no exogenous
#: variation, so the trajectory table below marks the boundary rather than quoting the
#: horizon alone. **Nothing in this readout's conclusions turns on the frozen half** --
#: the trajectory tables are printed so that can be checked rather than asserted.
MOVING_WORLD_TICKS = 2_000


def _last(record: dict) -> dict:
    return record["checkpoints"][-1]


def _at(record: dict, ticks: int) -> dict | None:
    for entry in record["checkpoints"]:
        if entry["ticks"] == ticks:
            return entry
    return None


def _fmt(x, spec="{:.4f}"):
    return "--" if x is None else spec.format(x)


def _load_null_widths(path: Path | None = None) -> dict | None:
    path = path or (_HERE / "585-nulls-by-width.json")
    if not path.exists():
        return None
    return json.load(open(path, encoding="utf-8"))


def _one_dim_positive_share(entry: dict) -> float | None:
    """Share of the rank-1 loops whose carried direction returns as itself, not negated.

    `identification` on a 1x1 holonomy is `|sign - 1| / sqrt(2)`: exactly 0 for `+1` and
    `sqrt(2)` for `-1`. So thresholding at 0.5 recovers the sign, and its mean is the
    share. The null is a coin at 0.5.
    """
    holo = entry["bases"]["full"]["holonomy"]
    ident = holo["per_cycle"]["identification"]
    ones = [ident[i] for i, m in enumerate(holo["m_base"]) if m == 1]
    if not ones:
        return None
    return float(sum(1 for x in ones if x < 0.5) / len(ones))


def table_census(arms: dict) -> None:
    print("\n## 0. The lane census -- a construction fact, before any map is drawn\n")
    print("| arm | p | interior edges | width-1 (lateral) | wider (cross-level) | "
          "full-basis cycles | capped at rank 1 | wide-basis cycles |")
    print("|---|---|---|---|---|---|---|---|")
    for arm in ARM_ORDER:
        if arm not in arms:
            continue
        c = arms[arm]["lane_census"]
        print(
            f"| `{arm}` | {arms[arm]['reserve_p']} | {c['interior_edges']} | "
            f"{c['narrow_edges']} | {c['wide_edges']} | "
            f"{c['bases']['full']['cycles']} | {c['bases']['full']['capped_at_one']} | "
            f"{c['bases']['wide']['cycles']} |"
        )
    any_arm = next(iter(arms.values()))
    c = any_arm["lane_census"]
    print(
        f"\nEvery width-1 edge is lateral: **{c['narrow_all_lateral']}**. "
        f"Every wider edge is cross-level: **{c['wide_all_cross_level']}**."
    )


def table_flatness(arms: dict) -> None:
    print("\n## 1. How far from flat -- the wide basis at the horizon, against its own nulls\n")
    print("| arm | p | identification | flat null | rewired null | sigma_max | "
          "flatness | channel return | flat null |")
    print("|---|---|---|---|---|---|---|---|---|")
    for arm in ARM_ORDER:
        if arm not in arms:
            continue
        rec = arms[arm]
        w = _last(rec)["bases"]["wide"]["holonomy"]
        nulls = rec["nulls_at_horizon"]["wide"]
        print(
            f"| `{arm}` | {rec['reserve_p']} | "
            f"{_fmt(w['identification']['median'])} | "
            f"{_fmt(nulls['flat']['identification']['median'])} | "
            f"{_fmt(nulls['rewired']['identification']['median'])} | "
            f"{_fmt(w['sigma_max']['median'], '{:.3e}')} | "
            f"{_fmt(w['flatness']['median'], '{:.3e}')} | "
            f"{_fmt(w['channel_return']['median'])} | "
            f"{_fmt(nulls['flat']['channel_return']['median'])} |"
        )

    print("\n### The 1-dimensional loops (full basis, rank capped at one by a lateral)\n")
    print(
        "A 1x1 holonomy carries one direction, so the only path-independence question it\n"
        "poses is **the sign**: does the carried direction come back as itself or as its\n"
        "negative? `identification` is 0 for a positive return and `sqrt(2)` for a\n"
        "negative one, so the share returning positive is the whole distribution -- and\n"
        "the null is a coin, at 0.5. `flatness` and `channel_return` are identically 1\n"
        "here whatever the surface does, and are not quoted.\n"
    )
    nulls_by_width = _load_null_widths()
    print("| arm | cycles | share returning positive | Haar null share | at 2,000 ticks |")
    print("|---|---|---|---|---|")
    for arm in ARM_ORDER:
        if arm not in arms:
            continue
        rec = arms[arm]
        share_h = _one_dim_positive_share(_last(rec))
        moving = _at(rec, MOVING_WORLD_TICKS)
        share_m = _one_dim_positive_share(moving) if moving else None
        null = None
        if nulls_by_width and arm in nulls_by_width.get("arms", {}):
            null = nulls_by_width["arms"][arm]["bases"]["full"].get("one_dim_positive_share")
        n_one = sum(1 for m in _last(rec)["bases"]["full"]["holonomy"]["m_base"] if m == 1)
        print(
            f"| `{arm}` | {n_one} | {_fmt(share_h, '{:.3f}')} | {_fmt(null, '{:.3f}')} | "
            f"{_fmt(share_m, '{:.3f}')} |"
        )


def table_abelian(arms: dict) -> None:
    print("\n## 2. Is what remains abelian? -- wide basis, at the horizon\n")
    print("| arm | p | pairs | commutator (polar) | flat null | commutator (operator) | "
          "shared-direction invariance | direction d_eff |")
    print("|---|---|---|---|---|---|---|---|")
    for arm in ARM_ORDER:
        if arm not in arms:
            continue
        rec = arms[arm]
        a = _last(rec)["bases"]["wide"]["abelian"]
        nulls = rec["nulls_at_horizon"]["wide"]
        print(
            f"| `{arm}` | {rec['reserve_p']} | {a['wide']['pairs']} | "
            f"{_fmt(a['wide']['comm_polar']['median'])} | "
            f"{_fmt(nulls['flat']['comm_polar'].get('median'))} | "
            f"{_fmt(a['wide']['comm_op']['median'])} | "
            f"{_fmt(a['wide']['invariance']['median'])} | "
            f"{_fmt(a['direction_d_eff']['median'], '{:.3f}')} |"
        )


def table_link(arms: dict) -> None:
    print("\n## 3. Does holonomy predict composed rank? -- Spearman, wide basis\n")
    print("| arm | p | n chains | ER vs identification | ER vs flatness | "
          "ER vs channel return | apex d_eff vs identification | edge ER vs commutator |")
    print("|---|---|---|---|---|---|---|---|")
    for arm in ARM_ORDER:
        if arm not in arms:
            continue
        rec = arms[arm]
        link = _last(rec)["bases"]["wide"]["link"]
        print(
            f"| `{arm}` | {rec['reserve_p']} | {link['chains']} | "
            f"{_fmt(link['chain_er_vs_identification']['rho'], '{:+.3f}')} | "
            f"{_fmt(link['chain_er_vs_flatness']['rho'], '{:+.3f}')} | "
            f"{_fmt(link['chain_er_vs_channel_return']['rho'], '{:+.3f}')} | "
            f"{_fmt(link['apex_d_eff_vs_identification']['rho'], '{:+.3f}')} | "
            f"{_fmt(link['edge_er_vs_comm_polar']['rho'], '{:+.3f}')} |"
        )

    print("\n### The same correlation across the trajectory -- is it stable?\n")
    print("| arm | " + " | ".join(str(c["ticks"]) for c in next(iter(arms.values()))["checkpoints"]) + " |")
    print("|---" * (1 + len(next(iter(arms.values()))["checkpoints"])) + "|")
    for arm in ARM_ORDER:
        if arm not in arms:
            continue
        cells = [
            _fmt(c["bases"]["wide"]["link"]["chain_er_vs_identification"]["rho"], "{:+.3f}")
            for c in arms[arm]["checkpoints"]
        ]
        print(f"| `{arm}` | " + " | ".join(cells) + " |")


def table_p(arms: dict) -> None:
    print("\n## 4. Does `p` move it? -- horizon values against each arm's own chance\n")
    print("| arm | p | identification / flat null | channel return / flat null | "
          "commutator / flat null | per-chain ER | joint d_eff |")
    print("|---|---|---|---|---|---|---|")
    for arm in ARM_ORDER:
        if arm not in arms:
            continue
        rec = arms[arm]
        last = _last(rec)
        w = last["bases"]["wide"]["holonomy"]
        nulls = rec["nulls_at_horizon"]["wide"]
        a = last["bases"]["wide"]["abelian"]["wide"]

        def ratio(x, y):
            if x is None or y in (None, 0):
                return None
            return x / y

        print(
            f"| `{arm}` | {rec['reserve_p']} | "
            f"{_fmt(ratio(w['identification']['median'], nulls['flat']['identification']['median']), '{:.3f}')} | "
            f"{_fmt(ratio(w['channel_return']['median'], nulls['flat']['channel_return']['median']), '{:.3f}')} | "
            f"{_fmt(ratio(a['comm_polar']['median'], nulls['flat']['comm_polar'].get('median')), '{:.3f}')} | "
            f"{_fmt(last['joint']['per_chain_er']['median'])} | "
            f"{_fmt(last['joint']['joint']['d_eff_median'], '{:.3f}')} |"
        )


def table_tension(arms: dict) -> None:
    print("\n## 5. The tension -- does the loop contract, and does training change it?\n")
    print("| arm | sigma_max at construction | at horizon | flat null | "
          "flatness at construction | at horizon |")
    print("|---|---|---|---|---|---|")
    for arm in ARM_ORDER:
        if arm not in arms:
            continue
        rec = arms[arm]
        c = rec["at_construction"]["bases"]["wide"]["holonomy"]
        h = _last(rec)["bases"]["wide"]["holonomy"]
        nulls = rec["nulls_at_horizon"]["wide"]
        print(
            f"| `{arm}` | {_fmt(c['sigma_max']['median'], '{:.3e}')} | "
            f"{_fmt(h['sigma_max']['median'], '{:.3e}')} | "
            f"{_fmt(nulls['flat']['sigma_max']['median'], '{:.3e}')} | "
            f"{_fmt(c['flatness']['median'], '{:.3e}')} | "
            f"{_fmt(h['flatness']['median'], '{:.3e}')} |"
        )

    print("\n### `sigma_max` by cycle length -- contraction is geometric in the hops\n")
    arm = next((a for a in ARM_ORDER if a in arms), None)
    if arm is None:
        return
    by_len = _last(arms[arm])["bases"]["wide"]["holonomy"]["by_length"]
    print(f"`{arm}` at the horizon:\n")
    print("| cycle length | cycles | sigma_max median | identification median |")
    print("|---|---|---|---|")
    for length, row in sorted(by_len.items(), key=lambda kv: int(kv[0])):
        print(
            f"| {length} | {row['cycles']} | {_fmt(row['sigma_max']['median'], '{:.3e}')} | "
            f"{_fmt(row['identification']['median'])} |"
        )


def table_trajectory(arms: dict) -> None:
    """Every column across the ladder, with the moving-world boundary marked.

    The sandbox stops moving between 1,000 and 2,000 ticks, so the columns to the right
    of `2000` are taken on a frozen world. They are printed, not dropped: what matters
    is whether a conclusion needs them, and none of #585's does.
    """
    print(
        f"\n## The trajectory, with the world's stall marked"
        f" (motion collapses by {MOVING_WORLD_TICKS} ticks)\n"
    )
    ref = next(iter(arms.values()))
    ticks = [c["ticks"] for c in ref["checkpoints"]]
    header = " | ".join(
        f"{t}" + (" *" if t > MOVING_WORLD_TICKS else "") for t in ticks
    )
    for column, path, spec in (
        ("identification", ("holonomy", "identification"), "{:.4f}"),
        ("channel return", ("holonomy", "channel_return"), "{:.4f}"),
        ("commutator (polar)", ("abelian", "comm_polar"), "{:.4f}"),
        ("shared-direction invariance", ("abelian", "invariance"), "{:.4f}"),
    ):
        print(f"\n**{column}** (wide basis; `*` = frozen world)\n")
        print(f"| arm | construction | {header} |")
        print("|---" * (2 + len(ticks)) + "|")
        for arm in ARM_ORDER:
            if arm not in arms:
                continue
            rec = arms[arm]

            def pick(entry):
                node = entry["bases"]["wide"][path[0]]
                if path[0] == "abelian":
                    node = node["wide"]
                return _fmt(node[path[1]].get("median"), spec)

            cells = [pick(rec["at_construction"])] + [
                pick(c) for c in rec["checkpoints"]
            ]
            print(f"| `{arm}` | " + " | ".join(cells) + " |")

    print("\n**share of the 1-D lateral loops returning positive** (null 0.5; `*` = frozen world)\n")
    print(f"| arm | construction | {header} |")
    print("|---" * (2 + len(ticks)) + "|")
    for arm in ARM_ORDER:
        if arm not in arms:
            continue
        rec = arms[arm]
        cells = [_fmt(_one_dim_positive_share(rec["at_construction"]), "{:.3f}")] + [
            _fmt(_one_dim_positive_share(c), "{:.3f}") for c in rec["checkpoints"]
        ]
        print(f"| `{arm}` | " + " | ".join(cells) + " |")


def table_theory(path: Path) -> None:
    if not path.exists():
        return
    d = json.load(open(path, encoding="utf-8"))
    print("\n## The ladder, checked on the algebra rather than inherited\n")
    lad = d["ladder"]
    print("| d | trivial holonomy: dim fixed | abelian (generic torus) | non-abelian (generic) |")
    print("|---|---|---|---|")
    for row in lad["rows"]:
        print(
            f"| {row['d']} | {row['trivial_fixed_dim']} | "
            f"{row['abelian_fixed_dim_median']:.0f} (max {row['abelian_fixed_dim_max']}) | "
            f"{row['non_abelian_fixed_dim_median']:.0f} (max {row['non_abelian_fixed_dim_max']}) |"
        )
    b = d["rank_vs_flatness"]
    print(
        f"\nFlat by construction: composed ER **{b['flat_by_construction']['composed_er']:.3f}**"
        f" at identification {b['flat_by_construction']['identification']:.2e};"
        f" generic lanes: composed ER **{b['generic_lanes']['composed_er']:.3f}**"
        f" at identification {b['generic_lanes']['identification']:.3f}"
        f" ({b['hops']} hops, m = {b['m']}, n = {b['n']})."
    )
    a = d["flat_iff_equal"]
    print(
        f"\nEquality case is exactly the identity: **{a['equality_case_is_identity']}**."
        f" The subspace angle moves `sigma_max` and leaves `identification` alone:"
        f" **{a['theta_moves_identification']}**. The frame angle moves"
        f" `identification` and leaves `sigma_max` alone: **{a['phi_moves_sigma_max']}**."
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pattern", default="585-holonomy-*-baseline-seed42-20000.json")
    p.add_argument("--theory", type=Path, default=_HERE / "585-theory.json")
    args = p.parse_args()

    arms = _load_arms(args.pattern)
    if not arms:
        raise SystemExit(f"no records match {args.pattern}")
    print(f"# B29 (#585) tables -- {len(arms)} arms: {', '.join(sorted(arms))}")
    table_census(arms)
    table_flatness(arms)
    table_abelian(arms)
    table_link(arms)
    table_p(arms)
    table_tension(arms)
    table_trajectory(arms)
    table_theory(args.theory)


if __name__ == "__main__":
    main()
