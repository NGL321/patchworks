"""B66 (#642): the alphabet ladder, read off B63's existing files.

B63 (#637) built the sensorimotor dependence instrument and read it on three
arms. Its alphabet labels are NESTED, so k' = 2 and k' = 4 are already recorded
alongside the headline k = 8 -- no re-run, no new trials.

This script answers the one question the user put to B66: does the flat
bundle's margin over the trained surface depend on how hard the discrimination
is? If the margin shrinks as the alphabet grows, the flat bundle's advantage is
a small-task artifact. If it holds or grows, that explanation is weakened.

Input is `637-analysis.json` on branch `worktree-b63-dependence-instrument-637`
(the cold-start rig lives on unmerged branches). Pass its path, or fetch it:

    git show origin/<b63-branch>:prototypes/cold-start/T6/637-analysis.json > 637-analysis.json
    python b66_ladder.py 637-analysis.json

Every reading here is `traditional` mutual information (B64, #638), on NODE
STALKS at the world-read boundary (B49, #616) -- no number below is comparable
with a lane reading.
"""

import json
import sys

ARMS = ("untrained", "trained", "flat")
STRATA = ("patch", "proprioceptive", "touch")
KS = (2, 4, 8)


def reading(doc, arm, stratum, key):
    return doc["arms"][arm]["profile"]["strata"][stratum]["readings"].get(key)


def ladder(doc, reduction, decoder):
    """One (reduction, decoder) cross-section of the ladder."""
    rows = []
    for stratum in STRATA:
        for k in KS:
            key = "%s|k%d|all|%s" % (reduction, k, decoder)
            bits = {}
            for arm in ARMS:
                r = reading(doc, arm, stratum, key)
                bits[arm] = r["mi_bits"] if r else None
            ceiling = reading(doc, "trained", stratum, key)["ceiling_bits"]
            margin = bits["flat"] - bits["trained"]
            rows.append(
                {
                    "stratum": stratum,
                    "k": k,
                    "ceiling_bits": ceiling,
                    "untrained_bits": bits["untrained"],
                    "trained_bits": bits["trained"],
                    "flat_bits": bits["flat"],
                    "margin_bits": margin,
                    # the ceiling is log k and moves with k, so the margin in
                    # bits is not comparable across rungs; this is.
                    "margin_share_of_ceiling": margin / ceiling,
                    "trained_saturated": abs(bits["trained"] - ceiling) < 1e-9,
                    "flat_saturated": abs(bits["flat"] - ceiling) < 1e-9,
                }
            )
    return rows


def render(rows, title):
    print("\n=== %s ===" % title)
    print(
        "%-15s %-3s %-7s %-10s %-9s %-8s %-9s %-8s %s"
        % (
            "stratum",
            "k",
            "ceil",
            "untrained",
            "trained",
            "flat",
            "margin",
            "share",
            "sat",
        )
    )
    for r in rows:
        sat = "".join(
            (
                "T" if r["trained_saturated"] else "-",
                "F" if r["flat_saturated"] else "-",
            )
        )
        print(
            "%-15s %-3d %-7.3f %-10.3f %-9.3f %-8.3f %+-9.3f %+-8.3f %s"
            % (
                r["stratum"],
                r["k"],
                r["ceiling_bits"],
                r["untrained_bits"],
                r["trained_bits"],
                r["flat_bits"],
                r["margin_bits"],
                r["margin_share_of_ceiling"],
                sat,
            )
        )


def main(path):
    doc = json.load(open(path, encoding="utf-8"))
    assert doc["traditional_or_cohomological"] == "traditional", "B64 (#638)"
    out = {}
    for reduction in ("peak", "trace"):
        for decoder in ("centroid", "nn1"):
            title = "%s | %s" % (reduction, decoder)
            rows = ladder(doc, reduction, decoder)
            out[title] = rows
            render(rows, title)

    print("\n--- patch, the only stratum with headroom ---")
    for reduction in ("peak", "trace"):
        for decoder in ("centroid", "nn1"):
            rows = [
                r
                for r in out["%s | %s" % (reduction, decoder)]
                if r["stratum"] == "patch"
            ]
            share = ["%+.3f" % r["margin_share_of_ceiling"] for r in rows]
            bits = ["%+.3f" % r["margin_bits"] for r in rows]
            print(
                "%-16s margin bits k2/k4/k8 %s   share of ceiling %s"
                % ("%s|%s" % (reduction, decoder), " ".join(bits), " ".join(share))
            )

    print("\n--- saturation, all arms, every rung ---")
    for stratum in STRATA:
        sat = [
            r["trained_saturated"]
            for r in out["peak | centroid"]
            if r["stratum"] == stratum
        ]
        print(
            "%-15s trained at ceiling on k=2,4,8: %s"
            % (stratum, ", ".join(str(s) for s in sat))
        )
    return out


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "637-analysis.json")
