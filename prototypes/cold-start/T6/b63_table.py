"""B63 (#637): the cross-sections of `637-analysis.json`, as printed tables.

Separate from the instrument so the reading can be re-tabled without re-running
the estimator. Every number here is a **lower bound** on `I(P; Delta)` in bits
(the decoder is a function of `Delta`, so the data-processing inequality gives
the direction), and every one is quoted beside its **matched scramble null**.

    PYTHONPATH=src python prototypes/cold-start/T6/b63_table.py
"""

from __future__ import annotations

import json
from pathlib import Path

_HERE = Path(__file__).resolve().parent
ARMS = ("untrained", "trained", "flat")
KINDS = ("patch", "proprioceptive", "touch")


def rows(blob: dict, key: str):
    for kind in KINDS:
        block = blob["profile"]["strata"].get(kind)
        if not block:
            continue
        r = block["readings"].get(key)
        if r and not r.get("insufficient"):
            yield kind, block, r


def main() -> None:
    d = json.loads((_HERE / "637-analysis.json").read_text(encoding="utf-8"))
    arms = d["arms"]

    print("\n== 1. The terminus, the alphabet, and #224's gate ==\n")
    a = arms["trained"]
    print(f"terminus: actuator cell #{a['terminus']['actuator_cell']}, node stalk "
          f"{a['terminus']['stalk']}, commanded (readable) block "
          f"{a['terminus']['commanded_width']}")
    for kind in KINDS:
        al = a["profile"]["strata"][kind]["alphabet"]
        print(f"  {kind:<16} dim {al['dim']:>6}  ({al['cells']} cells x stalk "
              f"{al['width']})  orthonormal k=8: {al['orthonormal']}  "
              f"max|cos| {al['max_abs_cos']:.4f}")
    print()
    print(f"  {'arm/stratum':<28}{'|Delta| median':>16}{'/||state||':>13}"
          f"{'clears f32':>12}{'clears f64':>12}")
    for arm in ARMS:
        raw = json.loads(
            (_HERE / f"637-{arm}-seed{d['seed']}-{d['learn']}.json")
            .read_text(encoding="utf-8")
        )["trials"]
        for kind in KINDS:
            t = [r for r in raw if r["kind"] == kind]
            pn = [r["peak_norm"] for r in t]
            ratio = [r["peak_norm"] / r["state_norm_after_hold"] for r in t]
            import statistics as st
            print(f"  {arm + '/' + kind:<28}{st.median(pn):>16.3e}"
                  f"{st.median(ratio):>13.3e}"
                  f"{sum(r['clears_f32_floor'] for r in t):>7}/{len(t):<4}"
                  f"{sum(r['clears_f64_floor'] for r in t):>7}/{len(t):<4}")
    det = arms["trained"]["determinism"]
    print(f"\n  determinism (B43 §3): re-run Delta bit-identical = "
          f"{det['bit_identical']}, max|d| = {det['max_abs_delta']}")
    print("  -> I(P; Delta | C) = log k by construction; the conditioned form is "
          "vacuous, as B43 §3 argued. Not estimated: verified.")

    print("\n== 2. The headline profile: peak feature, k=8, nearest-centroid ==\n")
    h = f"{'arm':<11}{'stratum':<16}{'acc':>7}{'MI':>8}{'null':>8}{'excess':>9}{'p':>8}{'ceil':>7}"
    print(h)
    print("-" * len(h))
    for arm in ARMS:
        for kind, block, r in rows(arms[arm], "peak|k8|all|centroid"):
            print(f"{arm:<11}{kind:<16}{r['accuracy']:>7.3f}{r['mi_bits']:>8.3f}"
                  f"{r['null_mean_bits']:>8.3f}{r['excess_bits']:>9.3f}"
                  f"{r['p_value']:>8.4f}{r['ceiling_bits']:>7.3f}")

    print("\n== 3. The k-ladder (nested labels, so it is free) ==\n")
    h = f"{'arm':<11}{'stratum':<16}" + "".join(f"{'k=' + str(k):>18}" for k in (2, 4, 8))
    print(h)
    print("-" * len(h))
    for arm in ARMS:
        for kind in KINDS:
            block = arms[arm]["profile"]["strata"][kind]
            cells = []
            for k in (2, 4, 8):
                r = block["readings"].get(f"peak|k{k}|all|centroid")
                cells.append("      --          " if not r or r.get("insufficient")
                             else f"{r['mi_bits']:>8.3f}/{r['ceiling_bits']:<9.3f}")
            print(f"{arm:<11}{kind:<16}" + "".join(cells))
    print("  (MI / ceiling log2 k, both in bits)")

    print("\n== 4. The configuration ladder -- the sweep IS the noise model ==\n")
    h = f"{'arm':<11}{'stratum':<16}" + "".join(f"{'C=' + str(c):>10}" for c in (4, 8, 16, 24))
    print(h)
    print("-" * len(h))
    for arm in ARMS:
        for kind in KINDS:
            lad = arms[arm]["profile"]["strata"][kind]["config_ladder"]
            cells = []
            for c in (4, 8, 16, 24):
                r = lad.get(str(c))
                cells.append("        --" if not r or r.get("insufficient")
                             else f"{r['mi_bits']:>10.3f}")
            print(f"{arm:<11}{kind:<16}" + "".join(cells))
    print("  (MI bits; saturation toward log k = 3.000 as C falls is the "
          "conditioned form leaking back in)")

    print("\n== 5. Decoder and feature, on the trained arm ==\n")
    h = f"{'variant':<28}{'acc':>7}{'MI':>8}{'null':>8}{'excess':>9}{'p':>8}"
    print(h)
    print("-" * len(h))
    for kind in KINDS:
        for key in ("peak|k8|all|centroid", "peak|k8|all|nn1",
                    "trace|k8|all|centroid", "peak|k8|gated|centroid"):
            r = arms["trained"]["profile"]["strata"][kind]["readings"].get(key)
            if not r or r.get("insufficient"):
                continue
            label = f"{kind[:6]}/{key.replace('|', '/')}"
            print(f"{label:<28}{r['accuracy']:>7.3f}{r['mi_bits']:>8.3f}"
                  f"{r['null_mean_bits']:>8.3f}{r['excess_bits']:>9.3f}"
                  f"{r['p_value']:>8.4f}")

    print("\n== 6. The three references B43 §5 requires ==\n")
    for kind in KINDS:
        line = [f"  {kind:<16} ceiling log2 8 = 3.000 |"]
        for arm in ARMS:
            r = arms[arm]["profile"]["strata"][kind]["readings"].get(
                "peak|k8|all|centroid")
            line.append(f" {arm} {r['mi_bits']:.3f}" if r else f" {arm} --")
        print("".join(line))

    lanes = _HERE / "637-lanes-trained.json"
    if lanes.exists():
        L = json.loads(lanes.read_text(encoding="utf-8"))
        print("\n== 7. The per-edge profile -- a LANE object (B49), trained arm ==\n")
        print("  NOT comparable with anything above: sections 2-6 are on node")
        print("  stalks, this is on the lane. No number crosses (B49, #616).\n")
        h = (f"{'stratum':<16}{'edges':>7}{'median':>9}{'null':>8}{'excess':>9}"
             f"{'max':>8}{'p<=.05':>12}{'min':>7}")
        print(h)
        print("-" * len(h))
        for kind in KINDS:
            s = L["strata"][kind]
            print(f"{kind:<16}{s['carried_edges']:>7}{s['mi_median_bits']:>9.3f}"
                  f"{s['null_median_bits']:>8.3f}{s['excess_median_bits']:>9.3f}"
                  f"{s['mi_max_bits']:>8.3f}"
                  f"{str(s['edges_significant_p05']) + '/' + str(s['carried_edges']):>12}"
                  f"{s['minutes']:>7.1f}")
        print("  (bits; `min` is the estimator's own cost, which is ask 5's "
              "budget call answered)")
    print()


if __name__ == "__main__":
    main()
