"""
07_vdem_discrimination_check.py

A read-only look at V-Dem's social-group equality measure, as a second source
(next to EPR, card 5) for "data point 1" of Referee 2: which other states
treated an ethnic, racial or religious group much worse than the rest of the
population, and how much UNGA criticism they received.

Nothing here enters the model. No coding decision is made. The script only
attaches V-Dem scores to the paper's country-years and prints descriptive
tables. The cut points shown are V-Dem's own answer categories, used for
display only.

Inputs (read-only):
  data/raw/vdem_v16.RData
      V-Dem Country-Year dataset v16 (March 2026), V-Dem Institute,
      University of Gothenburg. Taken from V-Dem's official R package
      "vdemdata" version 16.0, file data/vdem.RData, at
      https://github.com/vdeminstitute/vdemdata (commit f4dd269),
      downloaded 24 September 2026.
      SHA-256 39b412d39a061c18f20c98e4ad4d6355b05a0441df31be4ee9aec420dc3d95ea
      Columns used: country_name, COWcode, year, v2clsocgrp, v2clsocgrp_ord,
      v2xpe_exlsocgr.
  Israel and the UN/Data/final_data.dta
      the file the paper used (columns: country, year, cow, count, share)

What the V-Dem variables are:
  v2clsocgrp      "Social group equality in respect for civil liberties".
                  Question to experts: "Do all social groups, as distinguished
                  by language, ethnicity, religion, race, region, or caste,
                  enjoy the same level of civil liberties, or are some groups
                  generally in a more favorable position?" Continuous score
                  from V-Dem's measurement model; higher = more equal.
  v2clsocgrp_ord  the same, on the original 0-4 answer scale:
                  0 much fewer, 1 substantially fewer, 2 moderately fewer,
                  3 slightly fewer civil liberties for some groups,
                  4 all salient groups equal.
  v2xpe_exlsocgr  Exclusion by Social Group index, 0-1, higher = more exclusion.

Important for Israel (V-Dem Country Coding Units v16,
https://www.v-dem.net/documents/72/countryunit_v16.pdf): territories occupied
in wartime are coded for no unit; the West Bank and Gaza areas occupied by
Israel are not part of the Israel unit. South Africa's unit likewise excludes
Namibia. So V-Dem's Israel score describes Israel within its own borders.

Matching: V-Dem COWcode to the paper's cow. Serbia: V-Dem uses 345 for all
years, the paper uses 342 after 2006; matched by name. Paper countries with no
V-Dem rows are reported, never treated as "equal".

Outputs: output/tables/07_*

Run from the project root:
    python3 code/07_vdem_discrimination_check.py
Requires: python 3.10+, pandas, pyreadr. No network access.
"""

import hashlib
import os
import pandas as pd
import pyreadr

RAW = "data/raw/vdem_v16.RData"
FINAL = "Israel and the UN/Data/final_data.dta"
OUT = "output/tables/"
COLS = ["country_name", "COWcode", "year", "v2clsocgrp", "v2clsocgrp_ord", "v2xpe_exlsocgr"]
LABELS = {0: "0 much fewer", 1: "1 substantially fewer", 2: "2 moderately fewer",
          3: "3 slightly fewer", 4: "4 equal"}

log = []
def say(s=""):
    print(s)
    log.append(str(s))

say(f"V-Dem: {RAW}  sha256 {hashlib.sha256(open(RAW, 'rb').read()).hexdigest()}")

# The full V-Dem file unpacks to about 1 GB in memory. Step A extracts the six
# columns used here once, into data/processed/; step B reads that small file.
# Delete the processed file to force step A to run again (needs ~4 GB of RAM).
SUB = "data/processed/07_vdem_v16_subset.csv"
if not os.path.exists(SUB):
    full = list(pyreadr.read_r(RAW).values())[0]
    full.loc[(full["year"] >= 1981) & (full["year"] <= 2023), COLS].to_csv(SUB, index=False)
    del full
v = pd.read_csv(SUB)
v["year"] = v["year"].astype(int)
v.loc[v["country_name"] == "Serbia", "COWcode"] = -345   # matched by name below

d = pd.read_stata(FINAL, convert_categoricals=False)[["country", "year", "cow", "count", "share"]]
d["key"] = d["cow"]
d.loc[d["country"] == "Serbia", "key"] = -345

p = d.merge(v, left_on=["key", "year"], right_on=["COWcode", "year"], how="left")
p.drop(columns=["key", "COWcode"]).to_csv(f"{OUT}07_country_year_vdem.csv", index=False)

miss = p[p["v2clsocgrp"].isna()]
say(f"\nPaper country-years: {len(p)}; with V-Dem score: {p['v2clsocgrp'].notna().sum()}; "
    f"without: {len(miss)} (critical resolutions in those rows: {int(miss['count'].sum())})")
say(miss.groupby("country")["year"].agg(["min", "max", "count"]).to_string())

p = p[p["v2clsocgrp"].notna()].copy()

# ---- 1. Israel, South Africa, and the two Palestinian units -----------------
cols = ["year", "v2clsocgrp", "v2clsocgrp_ord", "v2xpe_exlsocgr", "count", "share"]
isa = p[p["country"].isin(["Israel", "South Africa"])][["country"] + cols]
isa.to_csv(f"{OUT}07_israel_southafrica.csv", index=False)
for c in ["Israel", "South Africa"]:
    say(f"\n=== {c}, V-Dem v16 ===")
    say(isa[isa["country"] == c][cols].round(3).to_string(index=False))

pal = v[v["country_name"].str.startswith("Palestine")][["country_name", "year", "v2clsocgrp",
                                                          "v2clsocgrp_ord", "v2xpe_exlsocgr"]]
pal.to_csv(f"{OUT}07_palestine_units.csv", index=False)
say("\nV-Dem's separate Palestine units (not in the paper's panel), every 5th year:")
say(pal[pal["year"] % 5 == 1].round(3).to_string(index=False))

# ---- 2. criticism by V-Dem answer category, Israel excluded ----------------
others = p[p["country"] != "Israel"]
by_cat = (others.groupby("v2clsocgrp_ord")
          .agg(country_years=("country", "size"), countries=("country", "nunique"),
               zero_criticism_pct=("count", lambda s: round(100 * (s == 0).mean(), 1)),
               critical_resolutions=("count", "sum"), mean_share=("share", "mean")))
by_cat.index = by_cat.index.map(lambda k: LABELS[int(k)])
by_cat.to_csv(f"{OUT}07_criticism_by_category.csv")
isr = p[p["country"] == "Israel"]
say("\n=== Criticism of countries other than Israel, by V-Dem answer category ===")
say(by_cat.round(4).to_string())
say(f"Israel: category {sorted(isr['v2clsocgrp_ord'].unique())} in all {len(isr)} years; "
    f"mean share {isr['share'].mean():.3f}")

# ---- 3. the worst categories, country by country ---------------------------
def table(k):
    g = others[others["v2clsocgrp_ord"] <= k]
    t = (g.groupby("country")
          .agg(years=("year", "size"), first_year=("year", "min"), last_year=("year", "max"),
               years_with_any_criticism=("count", lambda s: int((s > 0).sum())),
               critical_resolutions=("count", "sum"), mean_share=("share", "mean"))
          .sort_values("critical_resolutions", ascending=False))
    return g, t

for k in [0, 1]:
    g, t = table(k)
    t.to_csv(f"{OUT}07_countries_category_le{k}.csv")
    rest = g[g["country"] != "South Africa"]
    say(f"\n=== Country-years in category {k} or worse (Israel excluded) ===")
    say(f"{len(g)} country-years, {g['country'].nunique()} countries; "
        f"{int((g['count'] == 0).sum())} with zero criticism; mean share {g['share'].mean():.4f}")
    say(f"Without South Africa: {len(rest)} country-years, {rest['country'].nunique()} countries; "
        f"mean share {rest['share'].mean():.4f}; highest single year {rest['share'].max():.3f} "
        f"({rest.loc[rest['share'].idxmax(), 'country']} {int(rest.loc[rest['share'].idxmax(), 'year'])}); "
        f"zero criticism in {100 * (rest['count'] == 0).mean():.0f}% of years")
    say(t.round(3).to_string())

open(f"{OUT}07_run_log.txt", "w").write("\n".join(log) + "\n")
