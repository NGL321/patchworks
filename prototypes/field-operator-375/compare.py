"""Before/after for #375 §2's gate, across the spectral floor (#434).

    python prototypes/field-operator-375/compare.py [after.json] [before.json]

`before` defaults to the pre-floor reading and `after` to the floored one, both
of which sit next to this file. The two are the *same* rig, seed, dome and
ladder run on two surfaces -- the maps before #434's floor and after it -- which
is what makes the columns comparable and is the only comparison intended here.
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
AFTER = HERE / "375-real-train-seed42-2000-floored.json"
BEFORE = HERE / "375-real-train-seed42-2000.json"

after = json.load(open(sys.argv[1] if len(sys.argv) > 1 else AFTER))
before = json.load(open(sys.argv[2] if len(sys.argv) > 2 else BEFORE))


def idx(d):
    return {r["ticks"]: r for r in d["rows"]}


B, A = idx(before), idx(after)
ticks = [t for t in sorted(A) if t in B]


def dep(h):
    for k in ("departure_over_frobenius", "departure_over_norm", "relative_departure"):
        if k in h:
            return h[k]
    return float("nan")


print("== non-normality (ADR-0023 instrument): field vs uncoupled ==")
print(
    f"{'tick':>5} | {'nnField B':>9} {'nnField A':>9} | {'nnUnc B':>9} {'nnUnc A':>9} |"
    f" {'gap B':>10} {'gap A':>10} | {'coup% B':>7} {'coup% A':>7}"
)
for t in ticks:
    b, a = B[t], A[t]
    fb, fa = b["non_normality"]["field"], a["non_normality"]["field"]
    ub, ua = b["non_normality"]["uncoupled"], a["non_normality"]["uncoupled"]
    print(
        f"{t:>5} | {fb:>9.5f} {fa:>9.5f} | {ub:>9.5f} {ua:>9.5f} |"
        f" {fb - ub:>+10.2e} {fa - ua:>+10.2e} |"
        f" {b['coupling_share_of_norm'] * 100:>7.2f} {a['coupling_share_of_norm'] * 100:>7.2f}"
    )

print()
print("== henrici departure and spectral radius ==")
print(
    f"{'tick':>5} | {'hen f B':>8} {'hen f A':>8} | {'hen u B':>8} {'hen u A':>8} |"
    f" {'rho f B':>8} {'rho f A':>8} | {'rho u B':>8} {'rho u A':>8}"
)
for t in ticks:
    b, a = B[t], A[t]
    hb, ha = b.get("henrici", {}).get("field"), a.get("henrici", {}).get("field")
    hub, hua = b.get("henrici", {}).get("uncoupled"), a.get("henrici", {}).get("uncoupled")
    if not hb or not ha:
        continue
    print(
        f"{t:>5} | {dep(hb):>8.4f} {dep(ha):>8.4f} | {dep(hub):>8.4f} {dep(hua):>8.4f} |"
        f" {hb['spectral_radius']:>8.4f} {ha['spectral_radius']:>8.4f} |"
        f" {hub['spectral_radius']:>8.4f} {hua['spectral_radius']:>8.4f}"
    )

print()
print("== per-cell uncoupled block non-normality ==")
for t in ticks:
    pb = B[t]["non_normality"]["per_cell_uncoupled"]
    pa = A[t]["non_normality"]["per_cell_uncoupled"]
    print(
        f"tick {t}: median B={pb['median']:.4f} A={pa['median']:.4f}"
        f"  min B={pb['min']:.4f} A={pa['min']:.4f}"
        f"  max B={pb['max']:.4f} A={pa['max']:.4f}"
    )

print()
print("== peak transient amplification ||M^t||/rho^t ==")
for t in ticks:
    def peak(r, key):
        amp, rho = r["amplification"][key], r["amplification"]["rho_powers"]
        return max(x / y for x, y in zip(amp, rho))

    print(
        f"tick {t}: field B={peak(B[t], 'field'):.3f} A={peak(A[t], 'field'):.3f}"
        f" | uncoupled B={peak(B[t], 'uncoupled'):.3f} A={peak(A[t], 'uncoupled'):.3f}"
    )

print()
print("== checks (after) ==")
for t in ticks:
    c = A[t]["checks"]
    print(
        f"tick {t}: broadcast med={c['broadcast_identity']['median_relative_error']:.4g}"
        f" max={c['broadcast_identity']['max_relative_error']:.4g}"
        f" | one_tick med={c['one_tick']['median_relative_error']:.4g}"
    )
