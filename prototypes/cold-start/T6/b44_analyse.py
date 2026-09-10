"""B44 (#610): the paired read across the five transport arms, and `ρ(K)`'s track.

Medians of medians cannot settle this. Every arm ran the **same five trials** —
same trained surface, same sources, same rotation, same `world_loop(c)` — so the
comparison that carries information is **per cell, per trial, paired**, and that
is what this takes. The trial spread within an arm is reported beside it, because
a difference smaller than that spread is not a difference.

`τ̂`'s censoring is carried through rather than filtered out: a censored `τ̂` is a
**lower bound**, so an arm with more censored cells has its `τ̂` *under*-reported,
and the direction of that bias is stated wherever it could change a verdict.

Usage::

    PYTHONPATH=src python prototypes/cold-start/T6/b44_analyse.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
ARMS = ("control", "flat_unit", "haar_unit", "flat_banded", "haar_banded")
SEED, LEARN = 42, 2000


def load(arm: str) -> dict:
    return json.loads(
        (_HERE / f"610-tau-{arm}-seed{SEED}-{LEARN}.json").read_text(encoding="utf-8")
    )


def main() -> None:
    records = {a: load(a) for a in ARMS}
    rho = json.loads(
        (_HERE / f"610-rho-seed{SEED}-{LEARN}.json").read_text(encoding="utf-8")
    )
    out: dict = {"issue": 610, "seed": SEED, "learn": LEARN}

    # -- the arm table -------------------------------------------------------
    print("\n=== the five arms: what each one's transport actually reads ===")
    print(
        f"{'arm':<13} {'ident':>7} {'chan':>7} {'cyc sigma':>10} "
        f"{'bottleneck':>11} {'tau med':>8} {'ratio med':>10} {'cens':>5} {'read':>5}"
    )
    table = []
    for arm in ARMS:
        r = records[arm]
        cyc = r["cycle_surface"]
        tr = r["trials_out"]
        taus = np.array([t["per_cell"]["tau_median"] for t in tr], dtype=float)
        rats = np.array([t["per_cell"]["ratio_median"] for t in tr], dtype=float)
        cens = np.array([t["per_cell"]["censored_cells"] for t in tr], dtype=float)
        rd = np.array([t["readable_cells"] for t in tr], dtype=float)
        bn = np.array([t["bottleneck"] for t in tr], dtype=float)
        row = {
            "arm": arm,
            "identification": cyc["identification"],
            "channel_return": cyc["channel_return"],
            "cycle_sigma_max": cyc["sigma_max"],
            "bottleneck_median": float(np.median(bn)),
            "tau_median": float(np.median(taus)),
            "tau_median_spread": [float(taus.min()), float(taus.max())],
            "ratio_median": float(np.median(rats)),
            "censored_median": float(np.median(cens)),
            "readable_median": float(np.median(rd)),
        }
        table.append(row)
        print(
            f"{arm:<13} {cyc['identification']:>7.4f} {cyc['channel_return']:>7.4f} "
            f"{cyc['sigma_max']:>10.3e} {row['bottleneck_median']:>11.3e} "
            f"{row['tau_median']:>8.1f} {row['ratio_median']:>10.4f} "
            f"{row['censored_median']:>5.0f} {row['readable_median']:>5.0f}"
        )
    out["arms"] = table

    # -- the paired read ------------------------------------------------------
    #
    # Paired per cell, per trial, against `control`. The cells are in
    # `detectability.population` order in every arm and the trials are matched by
    # index, so `tau_per_cell[i]` means the same cell under the same perturbation
    # in every arm — which is the only comparison the arm medians cannot fake.
    print("\n=== paired per-cell tau-hat, trial-matched, vs control ===")
    base = np.array(
        [t["per_cell"]["tau_per_cell"] for t in records["control"]["trials_out"]],
        dtype=float,
    )
    print(
        f"{'arm':<13} {'d(tau) med':>11} {'d(tau) mean':>12} {'IQR':>17} "
        f"{'longer':>7} {'shorter':>8} {'same':>6} {'ratio med':>10}"
    )
    paired = []
    for arm in ARMS:
        arr = np.array(
            [t["per_cell"]["tau_per_cell"] for t in records[arm]["trials_out"]],
            dtype=float,
        )
        d = (arr - base).ravel()
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = np.where(base.ravel() > 0, arr.ravel() / np.maximum(base.ravel(), 1e-30), np.nan)
        f = frac[np.isfinite(frac)]
        row = {
            "arm": arm,
            "paired_samples": int(d.size),
            "delta_tau_median": float(np.median(d)),
            "delta_tau_mean": float(d.mean()),
            "delta_tau_q25": float(np.percentile(d, 25)),
            "delta_tau_q75": float(np.percentile(d, 75)),
            "cells_longer": int((d > 0).sum()),
            "cells_shorter": int((d < 0).sum()),
            "cells_same": int((d == 0).sum()),
            "tau_ratio_median": float(np.median(f)) if f.size else float("nan"),
        }
        paired.append(row)
        print(
            f"{arm:<13} {row['delta_tau_median']:>11.1f} {row['delta_tau_mean']:>12.3f} "
            f"[{row['delta_tau_q25']:>6.1f},{row['delta_tau_q75']:>6.1f}]   "
            f"{row['cells_longer']:>7} {row['cells_shorter']:>8} {row['cells_same']:>6} "
            f"{row['tau_ratio_median']:>10.3f}"
        )
    out["paired_vs_control"] = paired

    # The null this has to beat: control's own trial-to-trial spread, paired the
    # same way. If a between-arm shift is inside it, the arm did not move `τ̂`.
    within = []
    for i in range(base.shape[0]):
        for j in range(i + 1, base.shape[0]):
            within.append(base[j] - base[i])
    w = np.concatenate(within)
    # -- the same read, on #224's gate only -----------------------------------
    #
    # A cell whose `1/e` crossing sits under float32's granularity has a `τ̂` that
    # is arithmetic noise, and 85-95% of cells read that way on this map. So the
    # paired read is taken again over the cells readable in **both** arms of a
    # pair — the only cells where a difference in `τ̂` is a difference in the
    # architecture rather than in the noise.
    print("\n=== the same paired read, restricted to #224's gate (both arms) ===")
    gate_base = np.array(
        [t["per_cell"]["readable_per_cell"] for t in records["control"]["trials_out"]],
        dtype=bool,
    )
    print(
        f"{'arm':<13} {'n cells':>8} {'d(tau) med':>11} {'d(tau) mean':>12} "
        f"{'longer':>7} {'shorter':>8} {'ratio med':>10}"
    )
    gated = []
    for arm in ARMS:
        arr = np.array(
            [t["per_cell"]["tau_per_cell"] for t in records[arm]["trials_out"]],
            dtype=float,
        )
        g = np.array(
            [t["per_cell"]["readable_per_cell"] for t in records[arm]["trials_out"]],
            dtype=bool,
        )
        both = gate_base & g
        d = (arr - base)[both]
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = arr[both] / np.maximum(base[both], 1e-30)
        f = frac[np.isfinite(frac) & (base[both] > 0)]
        row = {
            "arm": arm,
            "cells_readable_in_both": int(both.sum()),
            "delta_tau_median": float(np.median(d)) if d.size else float("nan"),
            "delta_tau_mean": float(d.mean()) if d.size else float("nan"),
            "cells_longer": int((d > 0).sum()),
            "cells_shorter": int((d < 0).sum()),
            "tau_ratio_median": float(np.median(f)) if f.size else float("nan"),
        }
        gated.append(row)
        print(
            f"{arm:<13} {row['cells_readable_in_both']:>8} "
            f"{row['delta_tau_median']:>11.1f} {row['delta_tau_mean']:>12.3f} "
            f"{row['cells_longer']:>7} {row['cells_shorter']:>8} "
            f"{row['tau_ratio_median']:>10.3f}"
        )
    out["paired_vs_control_gated"] = gated

    out["control_within_arm_null"] = {
        "paired_samples": int(w.size),
        "abs_delta_median": float(np.median(np.abs(w))),
        "abs_delta_q75": float(np.percentile(np.abs(w), 75)),
        "abs_delta_q90": float(np.percentile(np.abs(w), 90)),
    }
    n = out["control_within_arm_null"]
    print(
        f"\ncontrol's own trial-to-trial null: |d(tau)| median {n['abs_delta_median']:.1f}, "
        f"q75 {n['abs_delta_q75']:.1f}, q90 {n['abs_delta_q90']:.1f} "
        f"({n['paired_samples']} paired samples)"
    )

    # -- the 2x2: is it the return, or is it the gain? ------------------------
    #
    # `flat_*` has an exact O(1) return, `haar_*` a chance one, at matched cycle
    # gain within each pair. A `τ̂` that tracks the return moves down the columns;
    # one that tracks gain moves across the rows.
    def med(arm: str) -> float:
        return float(
            np.median(
                np.array(
                    [t["per_cell"]["tau_median"] for t in records[arm]["trials_out"]]
                )
            )
        )

    out["two_by_two"] = {
        "exact_return_unit_gain": med("flat_unit"),
        "exact_return_banded_gain": med("flat_banded"),
        "chance_return_unit_gain": med("haar_unit"),
        "chance_return_banded_gain": med("haar_banded"),
        "control": med("control"),
        "note": (
            "haar_unit and haar_banded read identically on the cycle surface "
            "(ident 1.0088, chan 0.1903, sigma 1.102e-07): project()'s band is a "
            "no-op on maps that are already isometries, so the two are one arm "
            "wearing two names and are reported as such."
        ),
    }

    # -- reading 2 -----------------------------------------------------------
    print("\n=== rho(K) across the training horizon ===")
    print(f"{'tick':>6} {'rho(K) med':>11} {'rho(K) max':>11} {'rho(used) med':>14} {'sigma(used) med':>16} {'tau(rho used)':>14}")
    track = []
    for tick in sorted((int(k) for k in rho["checkpoints"]), key=int):
        c = rho["checkpoints"][str(tick)]
        row = {
            "tick": tick,
            "rho_raw_median": c["rho_raw"]["median"],
            "rho_raw_max": c["rho_raw"]["max"],
            "rho_used_median": c["rho_used"]["median"],
            "sigma_used_median": c["sigma_used"]["median"],
            "tau_median": c["tau_from_rho_used"]["median"],
            "cells_rho_used_at_or_above_one": c["tau_from_rho_used"][
                "cells_at_or_above_one"
            ],
        }
        track.append(row)
        print(
            f"{tick:>6} {row['rho_raw_median']:>11.6f} {row['rho_raw_max']:>11.6f} "
            f"{row['rho_used_median']:>14.6f} {row['sigma_used_median']:>16.6f} "
            f"{row['tau_median']:>14.2f}"
        )
    out["rho_track"] = track
    out["rho_movement"] = rho["rho_movement"]
    out["motion"] = rho.get("motion")
    m = rho["rho_movement"]
    print(
        f"\nrho(K) at construction {m['at_construction']:.6f} (uniform: "
        f"{m['construction_is_uniform']}); |delta| median {m['abs_delta_median']:.4f}, "
        f"max {m['abs_delta_max']:.4f}; {m['cells_moved_above_1e_3']} of {m['cells']} "
        f"cells moved > 1e-3"
    )

    path = _HERE / f"610-analysis-seed{SEED}-{LEARN}.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nwrote {path.name}")


if __name__ == "__main__":
    main()
