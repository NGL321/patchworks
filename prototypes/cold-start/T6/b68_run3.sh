#!/bin/sh
# B68 (#645), stage 3: sizing the non-reproducibility, and the control it needs.
#
# Stage 0 found that two *identical* invocations of the `transport` arm inside
# one process do not agree -- A vs B part by |dER| 2.3e-02 at tick 100 and
# 2.5e-01 at tick 200 -- while the run carrying the matched-generic null (C) is
# **bit-identical** to A. So the null is exonerated and the arm itself does not
# reproduce.
#
# Two things follow, and both are measurements rather than arguments.
#
# **Is the rule the amplifier or the messenger?** `--mode frozen` runs the same
# A/B/C triple with no rule at all. If the frozen arm reproduces exactly, the
# world stream is pinned and `TransportRule` is amplifying something below the
# last bit; if it does not, the stream itself is loose and every dynamical
# reading on this map inherits it.
#
# **Does the horizon separation survive the spread?** Item 1's verdict is that
# `TransportRule` settles at ER ~2.99 against the frozen arm's flat ~2.78 -- a
# separation of ~0.2, measured once. The spread at tick 200 was 0.25, larger
# than that, though that rung sits in the middle of the transient where the arm
# is climbing fastest. Whether the arm is still that loose at 20,000 is not
# knowable from a 200-tick check, so the replicate is run at the horizon on the
# arm that carries the claim.
set -e
cd "$(dirname "$0")/../../.."
export PYTHONPATH=src
T6=prototypes/cold-start/T6

python $T6/b68_repro.py --mode frozen > $T6/645-repro-frozen.log 2>&1

python $T6/b62_frozen.py --mode transport --ticks 20000 --no-generic \
  --out $T6/645-transport-rep2-baseline-seed42-20000.json \
  > $T6/645-transport-rep2-20k.log 2>&1
