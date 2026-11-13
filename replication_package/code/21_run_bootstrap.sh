#!/bin/bash
# Runs every bootstrap chunk for card 21, one specification and model per call
# (each call takes about 90 seconds; the sandbox stops a call after three minutes), then the summary.
cd "$(dirname "$0")/.."
rm -f output/tables/21_boot_rounds.csv
SPECS=("+ liberal democracy (V-Dem)" "+ democracy + KOF" "+ all four (V-Dem, KOF, DPI, Muslim share)" \
       "+ Gründler-Krieger ML democracy (to 2019)" "others: OECD members (that year)" "others: non-OECD (that year)" \
       "others: democracies (V-Dem regime 2-3)" "others: non-democracies (V-Dem regime 0-1)")
for spec in "${SPECS[@]}"; do
  for m in re_year fe; do
    python3 -W ignore code/21_referee1_controls_and_democracy.py --stage boot --spec "$spec" --model $m --seeds 0:500
  done
done
python3 -W ignore code/21_referee1_controls_and_democracy.py --stage summary
