"""
05_epr_discrimination_check.py

Task card 5: a read-only look at the Ethnic Power Relations (EPR) data, to see
whether it can serve as "data point 1" for Referee 2 (other states committing a
comparable violation, here state discrimination against an ethnic group).

Nothing here enters the model. No coding decision is made. The script only
reshapes EPR to country-years, attaches the criticism counts the paper already
uses, and prints descriptive tables.

Inputs (read-only):
  data/raw/EPR-2021.csv
      EPR Core 2021, ETH Zurich, https://icr.ethz.ch/data/epr/core/
      downloaded by Matanel on 24 September 2026.
      SHA-256 98e2d56431f55355d5d69deaa28c9063a49e859722bbb6d777a541ee831bcfe0
  data/raw/EPR-2023.csv
      EPR 2023 (beta), ETH Zurich, https://icr.ethz.ch/data/epr/core/beta.html
      downloaded by Matanel on 24 September 2026.
      SHA-256 8b4637484fb0963e6fc72306139deab655145589c9c851f312e8b231ca66cbd7
  Israel and the UN/Data/final_data.dta
      the file the paper used (columns: country, year, cow, count, share)

EPR columns used: gwid, statename, from, to, group, size, status.
"Discriminated" means status == "DISCRIMINATED" exactly, as coded by EPR.

Matching EPR to the paper's countries: by Gleditsch-Ward code (gwid) against
the paper's COW code (cow), which agree for every state in the period except
Serbia after 2006 (GW 340, paper uses 342/345), handled by name. EPR has no
rows for 20 small states in the paper (microstates and small island states);
they are reported as "not in EPR", never as "no discrimination".

Outputs: output/tables/05_*

Run from the project root:
    python3 code/05_epr_discrimination_check.py
Requires: python 3.10+, pandas. No network access.
"""

import hashlib
import pandas as pd

RAW = {"2021": "data/raw/EPR-2021.csv", "2023": "data/raw/EPR-2023.csv"}
FINAL = "Israel and the UN/Data/final_data.dta"
OUT = "output/tables/"
Y0, Y1 = 1981, 2023

log = []
def say(s=""):
    print(s)
    log.append(str(s))

for v, f in RAW.items():
    say(f"EPR {v}: {f}  sha256 {hashlib.sha256(open(f,'rb').read()).hexdigest()}")

# ---- the paper's panel -------------------------------------------------------
d = pd.read_stata(FINAL, convert_categoricals=False)[["country", "year", "cow", "count", "share"]]
code_to_country = d[["cow", "country"]].drop_duplicates().set_index("cow")["country"].to_dict()

def epr_country_years(path):
    """Expand EPR group periods to one row per group-year, 1981-2023."""
    e = pd.read_csv(path)
    e = e[(e["to"] >= Y0) & (e["from"] <= Y1)].copy()
    rows = []
    for r in e.itertuples(index=False):
        for y in range(max(r._2, Y0), min(r.to, Y1) + 1):   # r._2 is column "from"
            rows.append((r.gwid, r.statename, y, r.group, r.size, r.status))
    g = pd.DataFrame(rows, columns=["gwid", "statename", "year", "group", "size", "status"])
    g["country"] = g["gwid"].map(code_to_country)
    g.loc[g["gwid"] == 340, "country"] = "Serbia"
    return g

groups = {v: epr_country_years(p) for v, p in RAW.items()}

# ---- 1. Israel and South Africa, group by group ----------------------------
for v, g in groups.items():
    x = g[g["country"].isin(["Israel", "South Africa"])]
    # collapse consecutive identical years back into periods for readability
    x = (x.sort_values(["country", "group", "year"])
           .assign(key=lambda t: t["status"] + "|" + t["size"].astype(str))
           .assign(block=lambda t: (t.groupby(["country", "group"])["key"]
                                     .transform(lambda s: (s != s.shift()).cumsum())))
           .groupby(["country", "group", "block", "status", "size"], as_index=False)["year"]
           .agg(["min", "max"]).rename(columns={"min": "from", "max": "to"})
           .drop(columns="block").sort_values(["country", "from", "group"]))
    x.to_csv(f"{OUT}05_israel_southafrica_epr{v}.csv", index=False)
    say(f"\n=== Israel and South Africa in EPR {v}, {Y0}-{Y1} ===")
    say(x.to_string(index=False))

# ---- 2. country-year indicators (descriptive, two readings side by side) ----
def country_year(g, v):
    disc = g[g["status"] == "DISCRIMINATED"]
    cy = (g.groupby(["country", "year"]).size().rename(f"epr{v}_groups").to_frame()
            .join(disc.groupby(["country", "year"]).size().rename(f"epr{v}_n_disc"))
            .join(disc.groupby(["country", "year"])["size"].sum().rename(f"epr{v}_disc_popshare"))
            .join(disc.groupby(["country", "year"])["group"].apply(lambda s: "; ".join(sorted(s)))
                  .rename(f"epr{v}_disc_groups")))
    cy[f"epr{v}_n_disc"] = cy[f"epr{v}_n_disc"].fillna(0).astype(int)
    cy[f"epr{v}_disc_popshare"] = cy[f"epr{v}_disc_popshare"].fillna(0.0)
    cy[f"epr{v}_any_disc"] = (cy[f"epr{v}_n_disc"] > 0).astype(int)
    return cy.reset_index()

panel = d.copy()
for v, g in groups.items():
    panel = panel.merge(country_year(g[g["country"].notna()], v), on=["country", "year"], how="left")
panel.to_csv(f"{OUT}05_country_year_epr.csv", index=False)

not_in = sorted(set(d["country"]) - set(groups["2023"]["country"].dropna()))
say(f"\nPaper countries with no EPR rows at all ({len(not_in)}): {', '.join(not_in)}")
say(f"EPR states not in the paper's panel: "
    f"{sorted(groups['2023'].loc[groups['2023']['country'].isna(), 'statename'].unique())}")

# ---- 3. the two versions compared, 1981-2021 --------------------------------
both = panel[panel["year"] <= 2021].dropna(subset=["epr2021_any_disc", "epr2023_any_disc"])
diff = both[both["epr2021_any_disc"] != both["epr2023_any_disc"]]
say(f"\nCountry-years 1981-2021 where the two versions disagree on 'any discriminated group': {len(diff)}")
if len(diff):
    say(diff.groupby("country")["year"].agg(["min", "max", "count"]).to_string())
diff[["country", "year", "epr2021_any_disc", "epr2023_any_disc",
      "epr2021_disc_groups", "epr2023_disc_groups"]].to_csv(f"{OUT}05_version_differences.csv", index=False)

# ---- 4. who else discriminates, and how much criticism they got -------------
V = "2023"   # covers 1981-2023; the version comparison above shows how much this matters
p = panel[panel[f"epr{V}_any_disc"].notna()].copy()
disc = p[p[f"epr{V}_any_disc"] == 1]
others = disc[disc["country"] != "Israel"]

say(f"\n=== Using EPR {V}, {Y0}-{Y1} ===")
say(f"Country-years in both the paper's panel and EPR: {len(p)}")
say(f"Country-years with at least one discriminated group: {len(disc)} "
    f"(Israel {int((disc['country']=='Israel').sum())}, others {len(others)})")
say(f"Countries other than Israel with a discriminated group in at least one year: {others['country'].nunique()}")
say(f"Of the {len(others)} other discriminating country-years, "
    f"{int((others['count']==0).sum())} ({(others['count']==0).mean():.0%}) had zero critical resolutions")

yr = (p.assign(disc=p[f"epr{V}_any_disc"])
        .groupby("year").agg(n_countries=("country", "size"),
                             n_discriminating=("disc", "sum")))
yr.to_csv(f"{OUT}05_discriminating_countries_per_year.csv")
say("\nNumber of discriminating countries by year (every 5th year shown; full list in CSV):")
say(yr.iloc[::5].to_string())

tot_other_crit = p.loc[p["country"] != "Israel", "count"].sum()
say(f"\nAll critical resolutions against countries other than Israel (in EPR countries): {int(tot_other_crit)}")
say(f"  of which against a country in a year it was coded as discriminating: "
    f"{int(others['count'].sum())} ({others['count'].sum()/tot_other_crit:.0%})")

per_country = (others.groupby("country")
               .agg(first_year=("year", "min"), last_year=("year", "max"),
                    years_discriminating=("year", "size"),
                    years_with_any_criticism=("count", lambda s: int((s > 0).sum())),
                    critical_resolutions=("count", "sum"),
                    mean_share=("share", "mean"),
                    mean_disc_popshare=(f"epr{V}_disc_popshare", "mean"))
               .sort_values("critical_resolutions", ascending=False))
per_country.to_csv(f"{OUT}05_discriminating_countries_criticism.csv")
say("\nOther discriminating countries, ranked by critical resolutions in their discriminating years (top 25):")
say(per_country.head(25).round(3).to_string())

isr = p[p["country"] == "Israel"]
say(f"\nIsrael: discriminating in {int(isr[f'epr{V}_any_disc'].sum())} of {len(isr)} years; "
    f"critical resolutions {int(isr['count'].sum())}; mean share {isr['share'].mean():.3f}; "
    f"mean discriminated population share {isr[f'epr{V}_disc_popshare'].mean():.2f}")
say(f"Other discriminating country-years: mean share {others['share'].mean():.4f}; "
    f"median share {others['share'].median():.4f}")

# one illustrative cut, not a coding choice: discriminated groups at least as large,
# relative to population, as EPR's figure for Israel
cut = isr[f"epr{V}_disc_popshare"].min()
big = others[others[f"epr{V}_disc_popshare"] >= cut]
say(f"\nIllustration only: other country-years whose discriminated groups make up at least "
    f"{cut:.2f} of the population (Israel's lowest EPR figure in the period): {len(big)} "
    f"country-years, {big['country'].nunique()} countries, "
    f"{int((big['count']==0).sum())} with zero critical resolutions")
say(big.groupby("country").agg(years=("year", "size"), first=("year", "min"), last=("year", "max"),
                               critical_resolutions=("count", "sum"),
                               mean_share=("share", "mean"),
                               disc_popshare=(f"epr{V}_disc_popshare", "mean"))
       .sort_values("critical_resolutions", ascending=False).round(3).to_string())

open(f"{OUT}05_run_log.txt", "w").write("\n".join(log) + "\n")
