"""
12_build_reading_set.py

Data point 2 for Referee 2: which critical resolutions to read and classify.
Builds the list of resolution-country rows that are read by hand (Claude) and
classified in data/processed/14_classification_input.tsv.

Scope (Matanel, 24 Sep 2026): the comparator states from all three routes,
plus every Israel-critical resolution.
  Route EPR    : country-years EPR Core 2021 codes as having at least one
                 DISCRIMINATED group (2021 values copied to 2022-2023, as
                 decided for card 6). Israel excluded (read in full anyway).
  Route V-Dem  : country-years in V-Dem v16 v2clsocgrp category 0
                 ("much fewer"), as in card 7.
  Route cases  : violator-years in data/processed/08_case_list.csv (core and
                 contested lists).
  Israel       : all critical resolutions naming Israel, 1981-2023.
South Africa is included through all three routes (it is the referee's
explicit comparison).

Resolution-country rows are rebuilt exactly as in code/09_keyword_content.py
(same files, same steps; 2,200 rows). The unit is the paper's unit: a row is
one resolution (or, for 1981-1993, one Mesquita resolution part) labelled by
GPT as criticising one country.

Inputs (read-only):
  Israel and the UN/Data/GPT_Counts/MesquitaGPT.csv
  Israel and the UN/Data/full_text_data2.csv
  output/tables/05_country_year_epr.csv     (built by code/05)
  output/tables/07_country_year_vdem.csv    (built by code/07)
  data/processed/08_case_list.csv           (built by code/08)
Outputs:
  data/processed/12_reading_set.csv   one row per resolution-country, with text
  output/tables/12_reading_set_summary.csv
Run from the project root: python3 code/12_build_reading_set.py
"""

import hashlib
import numpy as np
import pandas as pd

D = "Israel and the UN/Data/"
NAME_FIX = {  # identical to code/09 (from Annual_Country_Resolutions_MV.ipynb)
    "Russian Federation": "Russia", "Islamic Republic of Iran": "Iran",
    "Republic of Korea": "South Korea",
    "Democratic People’s Republic of Korea": "Democratic People's Republic of Korea",
    "Republic of Moldova": "Moldova", "United States of America": "United States",
    "Syrian Arab Republic": "Syria",
    "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
    "Taliban": "Afghanistan", "Democratic Peoples Republic of Korea": "North Korea",
    "Federal Republic of Yugoslavia": "Serbia", "Serbia and Montenegro": "Serbia",
    "Federal Republic of Yugoslavia (Serbia and Montenegro)": "Serbia",
    "Côte d'Ivoire": "Cote d'Ivoire", "Côte d’Ivoire": "Cote d'Ivoire",
    "South Africa and Namibia": "South Africa", "Libya Arab Jamahiriya": "Libya",
}

# ---- 1. rebuild the critical resolution-country rows (as code/09) ----------
v = pd.read_csv(D + "full_text_data2.csv", usecols=["unres", "year", "text", "country"])
v = v.rename(columns={"unres": "res_id"}).assign(source="Voelsen")
m = pd.read_csv(D + "GPT_Counts/MesquitaGPT.csv", usecols=["res_id2", "session_reg", "content", "country"])
m = (m.rename(columns={"res_id2": "res_id", "content": "text"})
      .assign(year=1945 + m["session_reg"], source="Mesquita").drop(columns="session_reg"))
df = pd.concat([v, m], ignore_index=True)
df["country"] = df["country"].fillna("None").str.strip()
df = df[~df["country"].str.contains("Understood", case=False, na=False)]
df["gpt_label"] = df["country"]
df["country"] = df["country"].str.split(", ")
df = df.explode("country")
df["country"] = df["country"].str.strip().replace(NAME_FIX).replace("None", np.nan)
crit = df[df["country"].notna()].copy()
crit["year"] = crit["year"].astype(int)
crit = crit[(crit["year"] >= 1981) & (crit["year"] <= 2023)].reset_index(drop=True)
crit["text"] = crit["text"].fillna("").str.replace(r"\s+", " ", regex=True)
print(f"Critical resolution-country rows 1981-2023: {len(crit)} (code/09 has 2,200)")

# ---- 2. route membership ----------------------------------------------------
epr = pd.read_csv("output/tables/05_country_year_epr.csv")
# copy 2021 values to 2022-2023 (card 6 decision)
e21 = epr[epr["year"] == 2021][["country", "epr2021_any_disc"]].rename(columns={"epr2021_any_disc": "e21"})
epr = epr.merge(e21, on="country", how="left")
epr["epr_disc"] = np.where(epr["year"] >= 2022, epr["e21"], epr["epr2021_any_disc"])
epr_cy = set(map(tuple, epr.loc[epr["epr_disc"] == 1, ["country", "year"]].values))

vd = pd.read_csv("output/tables/07_country_year_vdem.csv")
vdem_cy = set(map(tuple, vd.loc[vd["v2clsocgrp_ord"] == 0, ["country", "year"]].values))

cases = pd.read_csv("data/processed/08_case_list.csv")
# case list uses COW; map COW to the paper's country names via final_data rows
cow2name = epr.drop_duplicates("cow").set_index("cow")["country"].to_dict()
case_cy = {}
for _, r in cases.iterrows():
    name = cow2name.get(r["cow"])
    for y in range(int(r["start"]), int(r["end"]) + 1):
        case_cy.setdefault((name, y), []).append(r["case_id"])

# GPT names that are spelled differently in final_data / the country-year tables.
# Used only to decide route membership; the GPT label itself is kept unchanged.
PAPER_NAME = {"United States": "United States of America",
              "Democratic Republic of the Congo": "Congo, Democratic Republic of",
              "Zaire": "Congo, Democratic Republic of",
              "Bosnia and Herzegovina": "Bosnia Herzegovenia",
              "Democratic People's Republic of Korea": "Korea, Democratic People's Republic of",
              "Libyan Arab Jamahiriya": "Libya"}
crit["paper_country"] = crit["country"].replace(PAPER_NAME)
key = list(zip(crit["paper_country"], crit["year"]))
crit["route_epr"] = [int(k in epr_cy and k[0] != "Israel") for k in key]
crit["route_vdem"] = [int(k in vdem_cy and k[0] != "Israel") for k in key]
crit["case_ids"] = [";".join(c for c in case_cy.get(k, []) if not c.startswith("israel")) for k in key]
crit["route_cases"] = (crit["case_ids"] != "").astype(int)
crit["israel"] = (crit["country"] == "Israel").astype(int)

# names in the resolution file that never match the country-year tables
known = set(epr["country"])
unmatched = sorted(set(crit["paper_country"]) - known)
print("GPT country names not found in final_data country list:", unmatched)

read = crit[(crit[["route_epr", "route_vdem", "route_cases", "israel"]].sum(axis=1) > 0)].copy()
read["text_md5"] = read["text"].map(lambda t: hashlib.md5(t.encode()).hexdigest()[:12])
read["row_id"] = [f"R{i:04d}" for i in range(len(read))]
cols = ["row_id", "res_id", "year", "country", "paper_country", "gpt_label", "source", "israel", "route_epr",
        "route_vdem", "route_cases", "case_ids", "text_md5", "text"]
read[cols].to_csv("data/processed/12_reading_set.csv", index=False)

summ = []
for name, mask in [("Israel", read["israel"] == 1), ("EPR route", read["route_epr"] == 1),
                   ("V-Dem route", read["route_vdem"] == 1), ("Case route", read["route_cases"] == 1),
                   ("South Africa (any route)", read["country"] == "South Africa"),
                   ("All rows to read", read.index == read.index)]:
    s = read[mask]
    summ.append({"set": name, "rows": len(s), "countries": s["country"].nunique(),
                 "distinct_texts": s["text_md5"].nunique()})
summ = pd.DataFrame(summ)
summ.to_csv("output/tables/12_reading_set_summary.csv", index=False)
print(summ.to_string(index=False))
print("\nNon-Israel rows by country:")
print(read[read["israel"] == 0].groupby("country").size().sort_values(ascending=False).to_string())
