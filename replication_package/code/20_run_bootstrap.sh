#!/bin/bash
# Runs every bootstrap chunk for card 20 (each call stays under three minutes), then the summary.
cd "$(dirname "$0")/.."
rm -f output/tables/20_boot_rounds.csv
for seeds in 0:250 250:500; do python3 -W ignore code/20_hurdle_and_bootstrap.py --stage boot_hurdle --seeds $seeds; done
for spec in re fe; do for sch in country year both; do
  python3 -W ignore code/20_hurdle_and_bootstrap.py --stage boot_scheme --spec $spec --scheme $sch --seeds 0:500
done; done
python3 -W ignore code/20_hurdle_and_bootstrap.py --stage summary
