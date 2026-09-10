#!/bin/sh
# B68 (#645) item 1: the horizon on the split, on B62's own rig, unmodified.
#
# B62 stopped both split arms at 5,000 because the rules-on collapse is settled
# by 2,000. The antagonism, though, is a claim about two *trajectories*:
# `TransportRule` alone peaked at ER 3.4018 at 1,000 and had drifted to 3.0024
# by 5,000 with no measured floor. #645 pre-registers the falsifier -- if that
# drift reaches the both-on arm's ~1.01 by 20,000, the antagonism is a transient
# of the first 5,000 ticks and the ticket's premise is gone.
#
# 20,000 is B57's and B62's own horizon; the frozen arm is already read there
# (2.7798 -> 2.7817), so all three arms end on the same rung.
#
# Strictly sequential, per B62's own note: each stage holds delta_P, G, a
# [2000, 4800] traffic buffer and an SVD workspace, and overlapping them trips
# this box's low-memory guard. Every checkpoint is written as it lands.
#
# `--no-generic` is added to both. B57 established the matched-generic excess
# N_gen = 0.0000 at every level, arm, checkpoint and redraw, and item 1 asks
# about ER, q_w and the joint columns, none of which the null enters. B48's
# joint rule is unaffected: `b57.joint` reports agreement, audience
# differentiation and exposure on every row regardless.
set -e
cd "$(dirname "$0")/../../.."
export PYTHONPATH=src
T6=prototypes/cold-start/T6

python $T6/b62_frozen.py --mode transport --ticks 20000 --no-generic \
  --out $T6/645-transport-baseline-seed42-20000.json > $T6/645-transport-20k.log 2>&1

python $T6/b62_frozen.py --mode bias --ticks 20000 --no-generic \
  --out $T6/645-bias-baseline-seed42-20000.json > $T6/645-bias-20k.log 2>&1
