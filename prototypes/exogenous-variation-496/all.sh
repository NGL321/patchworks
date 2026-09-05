#!/bin/sh
# #496's nine runs, sequentially and in the order that keeps a killed job useful.
#
# Sequential is not a preference: parallel 100k seeds trip this box's low-memory
# guard. Each run writes its JSON after every checkpoint, so an interruption
# costs the tail of one run rather than the run.
#
# `baseline` goes first because it supplies every denominator and both ceilings
# the two readouts need, and because on the (3, 4) surface #474 left behind it
# is the first reading of the collapse that exists at all.
set -e
cd "$(dirname "$0")/../.."
export PYTHONPATH=src
for condition in baseline arm drive; do
  python prototypes/exogenous-variation-496/run.py \
    --condition "$condition" --seeds 42 43 44 --ticks 100000
done
echo "ALL NINE RUNS COMPLETE"
