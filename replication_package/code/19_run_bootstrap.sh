#!/bin/bash
# Runs every bootstrap chunk for card 19, 250 seeds per call (each call stays under three minutes).
cd "$(dirname "$0")/.."
rm -f output/tables/19_boot_rounds.csv
for spec in pub_re pub_fe pub_row3_fe cor_re cor_fe cor_row3_re cor_row3_fe; do
  for seeds in 0:250 250:500; do python3 -W ignore code/19_corrected_tables.py --stage boot --data published --spec $spec --seeds $seeds; done
done
for data in corrected_A corrected_B; do
  for spec in cor_re cor_fe cor_row3_re cor_row3_fe; do
    for seeds in 0:250 250:500; do python3 -W ignore code/19_corrected_tables.py --stage boot --data $data --spec $spec --seeds $seeds; done
  done
done
python3 -W ignore code/19_corrected_tables.py --stage summary
