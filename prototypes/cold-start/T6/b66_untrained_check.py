"""B66 (#642): does the untrained arm still FAIL under every reduction?

Q1's ruling is that dependence is a gate, not a ranking, and that the gate has
teeth because the untrained surface fails it. B63 established that on the
`peak` reduction. The `trace` reduction reads far more bits on every arm, so
the gate's teeth have to be re-checked there or the ruling is only about one
reduction. Reads B63's `637-analysis.json`; no re-run.
"""

import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "637-analysis.json"
doc = json.load(open(path, encoding="utf-8"))

print(
    "%-15s %-16s %-8s %-9s %-9s %-8s %s"
    % ("stratum", "reduction", "mi", "null_mean", "null_q95", "excess", "p")
)
for arm in ("untrained", "trained", "flat"):
    print("\n-- %s --" % arm)
    for stratum in ("patch", "proprioceptive", "touch"):
        for reduction in ("peak", "trace"):
            for decoder in ("centroid", "nn1"):
                key = "%s|k8|all|%s" % (reduction, decoder)
                r = doc["arms"][arm]["profile"]["strata"][stratum]["readings"][key]
                print(
                    "%-15s %-16s %-8.3f %-9.3f %-9.3f %-8.3f %.4f"
                    % (
                        stratum,
                        "%s|%s" % (reduction, decoder),
                        r["mi_bits"],
                        r["null_mean_bits"],
                        r["null_q95_bits"],
                        r["excess_bits"],
                        r["p_value"],
                    )
                )
