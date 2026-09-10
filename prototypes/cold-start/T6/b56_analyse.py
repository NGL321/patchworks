"""B56 (#628): read the four arms against the null, jointly and never one column alone.

Three things this does that the run itself deliberately does not:

1. **Stamps the stall horizon per run.** [B38 (#599)](https://github.com/NGL321/patchworks/issues/599)
   measured the horizon varying **13x between seeds of one arm**, so a constant
   would be inherited rather than read. Each run carries MuJoCo's own `std_max`
   at every rung; the horizon is the last rung where the world was still moving
   by this run's own stamps, and every later rung is marked `*`.
2. **Scores jointly.** [B48 (#615)](https://github.com/NGL321/patchworks/issues/615):
   a candidate must show **agreement rising while differentiation stays
   nonzero**, and the null it must fail is B42's flat bundle at
   `channel_return` 1.0000 / differentiation 0.0000. `verdict` applies that rule
   rather than leaving it to prose.
3. **Answers the ceded question.** B54 Q5 was ceded on the argument that
   retention is temporal on `K` and holonomy spatial on the transport operator,
   so the two do not compete. `rho(used)` and `tau` are read against `baseline`
   at the same rung; a term that moves them has fought the prediction term.

Usage::

    PYTHONPATH=src python prototypes/cold-start/T6/b56_analyse.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent

SEED, TICKS = 42, 2000
ARMS = ("baseline", "channel", "hinge", "unbounded")

#: B42's flat bundle, `605-stagger-seed42.json` stagger 0. Perfect on clause 1,
#: zero on differentiation: the null clause 2 exists to fail.
NULL = {"channel_return": 1.0, "differentiation": 0.0, "identification": 0.0}

#: What "the world is still moving" means, in MuJoCo's own units. B33 read the
#: stall as a ~1,800x fall in `std_max`; anything at or below this is the
#: motionless side of it on every run recorded so far.
MOVING = 1e-2


def load(arm: str) -> dict | None:
    p = _HERE / f"628-channel-{arm}-seed{SEED}-{TICKS}.json"
    if not p.exists():
        p = _HERE / f"628-channel-{arm}-seed{SEED}-{TICKS}.inflight.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def horizon(record: dict) -> int:
    """The last rung this run's own motion stamps say the world was still moving.

    Inherited from nothing. A rung with no stamp (`available` false, which is
    tick 0's case) cannot extend the horizon and does not end it either.
    """
    last = 0
    for c in record["checkpoints"]:
        mo = c.get("motion") or {}
        if mo.get("available") and mo.get("std_max", 0.0) > MOVING:
            last = c["ticks"]
    return last


def row(c: dict) -> dict:
    w = c["holonomy_wide"]["subsets"]["wide"]
    fw = c["holonomy_full"]["subsets"]["wide"]
    d = c["differentiation"]
    e = c["exposure"]
    tr = (c.get("floor") or {}).get("tracking", {})
    ret = c["retention"]
    cal = c.get("calibration", {})
    return {
        "ticks": c["ticks"],
        "chan_wide": w["channel_return"]["median"],
        "ident_wide": w["identification"]["median"],
        "sigma_wide": w["sigma_max"]["median"],
        "chan_full": fw["channel_return"]["median"],
        "ident_full": fw["identification"]["median"],
        "differentiation": d["audience_differentiation"],
        "kv_rank": e["k_v_rank_median"],
        "kv_participation": e["k_v_participation_median"],
        "corr_learned": tr.get("corr_learned", float("nan")),
        "corr_haar": tr.get("corr_haar", float("nan")),
        "rho_used": ret["rho_used"]["median"],
        "tau": ret["tau_from_rho_used"]["median"],
        "floor_participation": (c.get("representation_floor") or {}).get(
            "participation_loss", float("nan")
        ),
        "gamma_on_surrogate": (cal.get("identification") or {}).get(
            "gamma_on_surrogate_scale", float("nan")
        ),
        "ident_surrogate": (cal.get("identification") or {}).get(
            "surrogate_median", float("nan")
        ),
        "std_max": (c.get("motion") or {}).get("std_max", float("nan")),
    }


def table(arm: str, record: dict) -> list[dict]:
    h = horizon(record)
    rows = [row(c) for c in record["checkpoints"]]
    for r in rows:
        r["past_stall"] = r["ticks"] > h
    print(f"\n=== {arm}  (this run's live horizon: {h} ticks, stamped not inherited) ===")
    print(
        f"{'ticks':>6} {'chan(w)':>8} {'ident(w)':>9} {'sigma(w)':>10} "
        f"{'chan(f)':>8} {'ident(f)':>9} {'diff':>7} {'kv':>5} {'corr':>7} "
        f"{'haar':>7} {'rho(u)':>7} {'tau':>7} {'world':>9}"
    )
    for r in rows:
        print(
            f"{r['ticks']:>5}{'*' if r['past_stall'] else ' '} "
            f"{r['chan_wide']:>8.4f} {r['ident_wide']:>9.4f} {r['sigma_wide']:>10.2e} "
            f"{r['chan_full']:>8.4f} {r['ident_full']:>9.4f} "
            f"{r['differentiation']:>7.4f} {r['kv_participation']:>5.1f} "
            f"{r['corr_learned']:>+7.3f} {r['corr_haar']:>+7.3f} "
            f"{r['rho_used']:>7.4f} {r['tau']:>7.1f} {r['std_max']:>9.2e}"
        )
    print("  * past this run's own stall stamp -- read against a motionless world")
    return rows


def verdict(arm: str, rows: list[dict], base: list[dict]) -> dict:
    """B48's joint rule, applied rather than narrated.

    *Agreement rising while differentiation stays nonzero*, against `baseline`
    at the **same rung**, and separately at the last live rung and at the
    horizon. An arm that raises `channel_return` while differentiation falls has
    scored nothing, however good the first column looks.
    """
    live = [r for r in rows if not r["past_stall"]]
    live_base = [r for r in base if not r["past_stall"]]
    last_live = live[-1] if live else rows[0]
    lb = live_base[-1] if live_base else base[0]
    end, eb = rows[-1], base[-1]

    def block(mine, theirs, label):
        d_chan = mine["chan_wide"] - theirs["chan_wide"]
        d_diff = mine["differentiation"] - theirs["differentiation"]
        return {
            "at": label,
            "ticks": mine["ticks"],
            "chan_wide": mine["chan_wide"],
            "chan_wide_vs_baseline": d_chan,
            "ident_wide": mine["ident_wide"],
            "differentiation": mine["differentiation"],
            "differentiation_vs_baseline": d_diff,
            "kv_participation_vs_baseline": mine["kv_participation"] - theirs["kv_participation"],
            "corr_learned_vs_baseline": mine["corr_learned"] - theirs["corr_learned"],
            # The ceded question, made a number.
            "rho_used_vs_baseline": mine["rho_used"] - theirs["rho_used"],
            "tau_vs_baseline": mine["tau"] - theirs["tau"],
            # B48: agreement up AND differentiation not given away.
            "scores": bool(d_chan > 0 and mine["differentiation"] > 0 and d_diff >= 0),
            "buys_agreement_out_of_differentiation": bool(d_chan > 0 and d_diff < 0),
            # ...and neither of those means anything if it is still under the null.
            "beats_null_on_channel": bool(mine["chan_wide"] >= NULL["channel_return"]),
        }

    return {
        "arm": arm,
        "last_live": block(last_live, lb, "last live rung"),
        "horizon": block(end, eb, "horizon (past stall)"),
    }


def main() -> None:
    records = {}
    for arm in ARMS:
        r = load(arm)
        if r is None:
            print(f"[B56] {arm}: no record yet", file=sys.stderr)
            continue
        records[arm] = r
    if "baseline" not in records:
        print("[B56] baseline is the comparator; nothing to say without it", file=sys.stderr)
        return

    tables = {arm: table(arm, rec) for arm, rec in records.items()}
    base = tables["baseline"]

    print("\n=== B48's joint rule, against baseline at the same rung ===")
    print(
        f"{'arm':<11} {'rung':>6} {'chan(w)':>8} {'d chan':>8} {'diff':>7} {'d diff':>8} "
        f"{'d kv':>6} {'d corr':>7} {'d rho':>8} {'d tau':>7}  scores"
    )
    out = {"ticket": 628, "seed": SEED, "ticks": TICKS, "null": NULL, "arms": {}}
    for arm, rows in tables.items():
        v = verdict(arm, rows, base)
        out["arms"][arm] = {
            "live_horizon": horizon(records[arm]),
            "rows": rows,
            "verdict": v,
        }
        for key in ("last_live", "horizon"):
            b = v[key]
            print(
                f"{arm:<11} {b['ticks']:>6} {b['chan_wide']:>8.4f} "
                f"{b['chan_wide_vs_baseline']:>+8.4f} {b['differentiation']:>7.4f} "
                f"{b['differentiation_vs_baseline']:>+8.4f} "
                f"{b['kv_participation_vs_baseline']:>+6.2f} "
                f"{b['corr_learned_vs_baseline']:>+7.3f} "
                f"{b['rho_used_vs_baseline']:>+8.5f} {b['tau_vs_baseline']:>+7.2f}  "
                f"{'YES' if b['scores'] else 'no'}"
                + ("  (bought out of differentiation)" if b["buys_agreement_out_of_differentiation"] else "")
            )

    print(
        f"\nnull to beat (B42 flat bundle, construction): channel_return "
        f"{NULL['channel_return']:.4f} at differentiation {NULL['differentiation']:.4f}"
    )
    print(
        "clause 1 alone is satisfied *perfectly* by that null, which is why "
        "`channel` is an arm and not an assumption."
    )

    # GAMMA on the surrogate's own scale, which is the thing the hinge actually
    # thresholds. Reported per arm because it moves with the surface.
    print("\n=== GAMMA re-anchored: measured vs surrogate identification (wide/local cycles) ===")
    print(f"{'arm':<11} {'rung':>6} {'ident meas':>11} {'ident surr':>11} {'GAMMA on surr':>14}")
    for arm, rows in tables.items():
        for r in rows[-1:]:
            print(
                f"{arm:<11} {r['ticks']:>6} {r['ident_wide']:>11.4f} "
                f"{r['ident_surrogate']:>11.4f} {r['gamma_on_surrogate']:>14.4f}"
            )

    path = _HERE / "628-analysis.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nwrote {path.name}")


if __name__ == "__main__":
    main()
