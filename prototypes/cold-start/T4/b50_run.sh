#!/bin/sh
# B50 (#618): both T3 arms to 20k, sequentially. Long runs get killed if run in
# parallel, and every checkpoint is written as it lands.
set -e
export PYTHONPATH=src
python prototypes/cold-start/T4/b50_spectra.py --stage trained --condition baseline --seeds 42 --ticks 20000
python prototypes/cold-start/T4/b50_spectra.py --stage trained --condition winner --seeds 42 --ticks 20000
echo "[B50] both arms done"
