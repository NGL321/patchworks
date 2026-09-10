#!/bin/sh
# B62 (#635): the stages, strictly one at a time.
#
# **Sequential is not a preference.** Each stage holds `delta_P` [3861, 4800]
# and `G` [6654, 4800] in float64, a [2000, 4800] traffic buffer and an SVD
# workspace. Running item 1's arm, item 3's pencil and item 2's first arm
# together tripped this box's low-memory guard and killed two of the three; the
# frozen arm's record survived only because every checkpoint is written as it
# lands. Do not overlap these.
#
# Item 1 runs the full 20k ladder because its question is whether the rank
# decays *eventually* with nothing learning. Item 2's arms stop at 5,000: the
# rules-on collapse is complete by 1,000 (B57 reads 2.9825 at 500 and 1.0967 at
# 1,000) and settled by 2,000, so 5,000 brackets it with two rungs to spare and
# the attribution does not need the long horizon.
set -e
cd "$(dirname "$0")/../../.."
export PYTHONPATH=src
T6=prototypes/cold-start/T6

python $T6/b62_frozen.py --mode bias --ticks 5000 > $T6/635-bias.log 2>&1

python $T6/b62_frozen.py --mode transport --ticks 5000 > $T6/635-transport.log 2>&1

# Item 1's long horizon, retried last because it is the one already answered at
# 5,000. Its through-5,000 record is kept beside it under its own name so a
# second kill cannot cost what is already established.
python $T6/b62_frozen.py --mode frozen --ticks 20000 \
  --out $T6/635-frozen-baseline-seed42-20000.json > $T6/635-frozen-20k.log 2>&1
