#!/bin/bash
# Bootstrap chunks for card 22, one specification, variant and model per call; then the summary.
cd "$(dirname "$0")/.."
rm -f output/tables/22_boot_rounds.csv
S1="combined: EPR + first use + democracy (1981-2014)"
S2="+ EPR + democracy (1981-2023)"
for v in "as is" "without South Africa" "without nuclear variable" "without both"; do for m in re_year fe; do
  python3 -W ignore code/22_referee2_package.py --stage boot --spec "$S1" --variant "$v" --model $m --seeds 0:500
done; done
for v in "as is" "without South Africa"; do for m in re_year fe; do
  python3 -W ignore code/22_referee2_package.py --stage boot --spec "$S2" --variant "$v" --model $m --seeds 0:500
done; done
python3 -W ignore code/22_referee2_package.py --stage summary
