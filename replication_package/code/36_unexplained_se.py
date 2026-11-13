"""
36_unexplained_se.py

Card 36 (29 October 2026): a bootstrap standard error for the UNEXPLAINED part
of Table 3, so that the manuscript does not have to assume it equals the
explained part's standard error (proposal 31 made that assumption with a note).

How it works
  The bootstrap of card 19 stored, for every round, only the explained part
  (output/tables/19_boot_rounds.csv). The overall gap of a round depends only on
  which countries were drawn, not on any regression, and the draws are fixed by
  the round's seed (numpy default_rng(seed), as in code/19). This script
  reproduces every draw with the same function (c19.draw_countries), computes
  the round's overall gap (others' mean share minus Israel's), joins it with the
  stored explained part, and takes unexplained = overall - explained per round.
  No regression is re-estimated and no stored number is changed.

Inputs (read-only): the inputs of code/19_corrected_tables.py and
  output/tables/19_boot_rounds.csv (500 seeds per data version and specification)
Outputs: output/tables/36_unexplained_se.csv, 36_run_log.txt
Run from the project root:  python3 code/36_unexplained_se.py   (about 1 minute)
Requires: python 3.10+, numpy, pandas. No network access.
"""

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
c19 = importlib.import_module("19_corrected_tables")
OUT = ROOT / "output" / "tables"
LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(str(s))


def main():
    rounds = pd.read_csv(OUT / "19_boot_rounds.csv").drop_duplicates(["data", "spec", "seed"])
    allc = c19.rebuild_counts()
    panels = c19.build_panels(allc)
    res = []
    for (data_key, spec_key), g in rounds.groupby(["data", "spec"], sort=False):
        key, samp, model, V, label = [x for x in c19.SPECS if x[0] == spec_key][0]
        D = c19.samples_of(panels[data_key])[samp].reset_index(drop=True)
        israel_cid = int(D.loc[D.Israel_dummy == 1, "cid"].iloc[0])
        groups = {c: idx for c, idx in D.groupby("cid").indices.items() if c != israel_cid}
        isr_mean = float(D.loc[D.Israel_dummy == 1, "share"].mean())
        # The same draw as c19.draw_countries (one rng.choice over the same sorted country
        # keys), but the others' mean share is computed from per-country sums and counts
        # instead of concatenating the rows, which is what makes this quick.
        cs = list(groups.keys())
        ssum = D.groupby("cid")["share"].sum()
        scnt = D.groupby("cid")["share"].count()
        overall = {}
        for s in g.seed:
            C = np.random.default_rng(int(s)).choice(cs, len(cs))
            overall[int(s)] = float(ssum.loc[C].sum() / scnt.loc[C].sum()) - isr_mean
        if data_key == "published" and spec_key == "pub_re":   # check against the slow method, seed 0
            S = c19.draw_countries(D, np.random.default_rng(0), groups)
            slow = float(S.loc[S.Israel_dummy == 0, "share"].mean()) - isr_mean
            assert abs(slow - overall[0]) < 1e-8, (slow, overall[0])
            say(f"check: fast and slow overall agree for seed 0 ({slow:.6f})")
        g = g.assign(overall=g.seed.map(overall))
        ok = g[g.explained.notna()]
        unexpl = ok.overall - ok.explained
        pt_e, pt_o, _ = c19.explained(D, model, V)
        pt_u = pt_o - pt_e
        res.append(dict(data=data_key, spec=spec_key, label=label,
                        explained=round(pt_e, 4), overall=round(pt_o, 4), unexplained=round(pt_u, 4),
                        se_explained=round(float(ok.explained.std(ddof=1)), 4),
                        se_overall=round(float(g.overall.std(ddof=1)), 4),
                        se_unexplained=round(float(unexpl.std(ddof=1)), 4),
                        ratio_unexplained=round(pt_u / float(unexpl.std(ddof=1)), 2),
                        pct2_5_unexplained=round(float(np.percentile(unexpl, 2.5)), 4),
                        pct97_5_unexplained=round(float(np.percentile(unexpl, 97.5)), 4),
                        rounds_ok=int(len(ok)), rounds_failed=int(g.explained.isna().sum())))
        say(str(res[-1]))
    t = pd.DataFrame(res)
    t.to_csv(OUT / "36_unexplained_se.csv", index=False)
    pd.set_option("display.width", 250)
    say("\n" + t.to_string(index=False))
    (OUT / "36_run_log.txt").write_text("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
