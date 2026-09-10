#!/bin/sh
# B68 (#645), stages 2-4. Strictly sequential, per B62's low-memory note.
#
# **Stage 2 -- the both-on arm on *this* surface.** B62's table put its two solo
# arms (read on `f51667d`) beside B57's both-on baseline (read on `c925866`),
# and the two surfaces do not agree at construction: uncentered traffic ER reads
# 2.7798 on one and 2.7656 on the other, same condition and same seed. That gap
# is small against the effects item 3 is about, but item 3's whole question is
# whether the pair's outcome is predictable from the solos, and an
# out-of-hull verdict must not be resting on a surface difference. #455's rule
# is name the surface and inherit no figure, so the fourth arm is re-run here.
#
# **Stage 3 -- the stall stamp, per mode.** B38's stamp is per run and both
# rules are reported; see b68_stall.py.
#
# **Stage 4 -- item 2's mechanism reading**, on the arm under test and on both
# controls.
set -e
cd "$(dirname "$0")/../../.."
export PYTHONPATH=src
T6=prototypes/cold-start/T6

# **Stage 0, and it runs first because it can invalidate everything after it.**
# The re-run of B62's `transport` arm did not reproduce B62's stored record on
# byte-identical code and the same seed. Either `--no-generic` perturbs the run
# or the arm is not deterministic; b68_repro.py separates the two, and if it is
# the second then B62's attribution and every non-paired arm comparison on this
# map is resting on a difference a re-run could produce by itself.
python $T6/b68_repro.py > $T6/645-repro.log 2>&1

python $T6/b62_frozen.py --mode both --ticks 20000 --no-generic \
  --out $T6/645-both-baseline-seed42-20000.json > $T6/645-both-20k.log 2>&1

python $T6/b68_stall.py --ticks 20000 > $T6/645-stall.log 2>&1

python $T6/b68_mechanism.py --mode bias --ticks 2000 > $T6/645-mech-bias.log 2>&1

python $T6/b68_mechanism.py --mode frozen --ticks 2000 > $T6/645-mech-frozen.log 2>&1

python $T6/b68_mechanism.py --mode transport --ticks 2000 > $T6/645-mech-transport.log 2>&1

python $T6/b68_mechanism.py --mode both --ticks 2000 > $T6/645-mech-both.log 2>&1
