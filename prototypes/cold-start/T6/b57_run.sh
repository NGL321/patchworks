#!/bin/sh
# B57 (#629): the acceptance check, then the three arms, sequentially.
#
# Sequential rather than parallel: parallel long runs on this box trip the
# low-memory guard, and every checkpoint is written as it lands, so a kill costs
# the checkpoint in flight and nothing before it.
#
# Every reading is taken twice -- on B52's uncentered second moment and on the
# centred covariance B53's advisory on #629 asks for -- off the same window, so
# neither costs a run.
set -e
export PYTHONPATH=src
B57=prototypes/cold-start/T6/b57_agreement.py
python "$B57" --stage accept --condition baseline
python "$B57" --stage trained --condition baseline --ticks 20000
python "$B57" --stage trained --condition winner --ticks 20000
python "$B57" --stage trained --condition flat --ticks 20000
