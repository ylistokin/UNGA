"""
14_classification_summary.py

Turns the hand classification of GPT-flagged critical resolutions into row-level
codes and summary tables.

Inputs (all produced earlier in this project):
  data/processed/12_reading_set.csv          one row per (resolution, criticised country)
                                             for Israel and the comparator countries
  data/processed/13_families.csv             row_id -> family (same country, same title)
  data/processed/14_classification_input.tsv the hand codes. A key is either a family
                                             (F###, applies to every row in the family)
                                             or a row (R####, overrides its family)

Code meanings (written down in docs/14_content_of_criticism.md):
  crit  Y = the resolution criticises the state
        M = the resolution concerns a situation involving the state in one of the
            categories, but does not accuse the state (for example assistance to refugees)
        N = not a criticism of the state at all (GPT false positive)
  cats  R racial or ethnic discrimination, apartheid
        C international crimes (war crimes, crimes against humanity, genocide, grave
          breaches, massacres, ethnic cleansing, apartheid called a crime against humanity);
          (broad reading: also rows where large-scale killing of civilians is attributed
          to the state without a crime label; the strict column requires a label in the
          text)
        F use of force against another state (aggression, invasion, armed attack)
        O occupation, annexation, settlements, denial of self-determination
        N nuclear weapons and the NPT
        T torture, slavery, forced labour
  tags  R_weak = discrimination named but not called racial or ethnic (dropped from
          the strict R column)
  other 1 = the criticism also (or only) concerns something outside these categories

Outputs:
  data/processed/14_row_codes.csv
  data/processed/14_country_year_content.csv
  output/tables/14_groups.csv
  output/tables/14_countries.csv
  output/tables/14_israel_southafrica_1981_1993.csv
  output/tables/14_israel_periods.csv
"""
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"
T = ROOT / "output" / "tables"
T.mkdir(parents=True, exist_ok=True)

CATS = ["R", "C", "F", "O", "N", "T"]
CRIME_LABEL = re.compile(
    r"war crime|crimes? against humanity|genocid|grave breach|massacre|ethnic cleansing",
    re.I,
)

rs = pd.read_csv(P / "12_reading_set.csv")
fam = pd.read_csv(P / "13_families.csv")[["row_id", "family"]]
ci = pd.read_csv(P / "14_classification_input.tsv", sep="\t", dtype=str,
                 keep_default_na=False)
assert not ci.key.duplicated().any(), "duplicate keys in the classification input"

codes = ci.set_index("key").to_dict("index")
df = rs.merge(fam, on="row_id", how="left")
missing = df[df.family.isna()]
assert missing.empty, f"rows without a family: {missing.row_id.tolist()[:5]}"


def effective(row):
    c = codes.get(row.row_id) or codes.get(row.family)
    if c is None:
        raise KeyError(f"no code for {row.row_id} / {row.family}")
    level = "row" if row.row_id in codes else "family"
    return pd.Series({"crit": c["crit"], "cats": c["cats"], "other": int(c["other"] or 0),
                      "quote": c["quote"], "note": c["note"], "tags": c.get("tags", ""),
                      "code_level": level})


df = pd.concat([df, df.apply(effective, axis=1)], axis=1)

for k in CATS:
    df[f"cat_{k}"] = df.cats.str.split(";").apply(lambda xs, k=k: k in xs).astype(int)
df["accused"] = (df.crit == "Y").astype(int)
df["not_accused"] = (df.crit == "M").astype(int)
df["false_pos"] = (df.crit == "N").astype(int)

# stricter readings, used as robustness columns
# strict C: a crime label appears in the text. The standard Geneva Convention phrase
# "penal sanctions, grave breaches and responsibilities of the High Contracting Parties"
# (a list of treaty articles, not an accusation) is removed before searching.
GENERIC = re.compile(r"penal\s+sanctio\s*ns,?\s+grave\s+breaches", re.I)
label_text = df.text.fillna("").str.replace(GENERIC, " ", regex=True)
df["cat_C_strict"] = ((df.cat_C == 1) & label_text.str.contains(CRIME_LABEL)).astype(int)
# strict R: drops rows tagged R_weak (discrimination named, but not called racial or
# ethnic and not apartheid)
df["cat_R_strict"] = ((df.cat_R == 1) & (df.tags != "R_weak")).astype(int)

# a criticism counts in a category only when the state is accused (crit == Y)
for k in CATS + ["C_strict", "R_strict"]:
    df[f"acc_{k}"] = df.accused * df[f"cat_{k}"]
df["acc_any_referee"] = df[[f"acc_{k}" for k in CATS]].max(axis=1)
df["acc_other_only"] = df.accused * (1 - df.acc_any_referee)

keep = ["row_id", "res_id", "year", "country", "paper_country", "gpt_label", "source",
        "route_epr", "route_vdem", "route_cases", "israel", "family", "code_level",
        "crit", "cats", "other", "quote", "note", "tags", "accused", "not_accused", "false_pos"] + \
       [f"cat_{k}" for k in CATS + ["C_strict", "R_strict"]] + \
       [f"acc_{k}" for k in CATS + ["C_strict", "R_strict", "any_referee", "other_only"]]
df[keep].to_csv(P / "14_row_codes.csv", index=False)

ACC = [f"acc_{k}" for k in CATS] + ["acc_C_strict", "acc_R_strict", "acc_any_referee",
                                    "acc_other_only"]


def summarise(g):
    n = len(g)
    out = {"gpt_rows": n, "countries": g.country.nunique(),
           "accused": int(g.accused.sum()), "not_accused": int(g.not_accused.sum()),
           "false_pos": int(g.false_pos.sum()),
           "share_accused": round(g.accused.mean(), 3) if n else float("nan")}
    for c in ACC:
        out[c] = int(g[c].sum())
    return pd.Series(out)


sa = df.country.eq("South Africa")
groups = {
    "Israel": df.israel == 1,
    "South Africa": sa,
    "EPR route (excl. South Africa)": (df.route_epr == 1) & ~sa,
    "V-Dem route (excl. South Africa)": (df.route_vdem == 1) & ~sa,
    "Case route (excl. South Africa)": (df.route_cases == 1) & ~sa,
    "All comparators (excl. South Africa)": (df.israel == 0) & ~sa,
    "All comparators (incl. South Africa)": df.israel == 0,
}
gt = pd.DataFrame({k: summarise(df[m]) for k, m in groups.items()}).T
gt.index.name = "group"
gt.to_csv(T / "14_groups.csv")

ct = pd.DataFrame({c: summarise(g) for c, g in df.groupby("country")}).T
ct.index.name = "country"
span = df.groupby("country").year.agg(["min", "max"])
routes = df.groupby("country")[["route_epr", "route_vdem", "route_cases", "israel"]].max()
ct = ct.join(span).join(routes).sort_values("gpt_rows", ascending=False)
ct.to_csv(T / "14_countries.csv")

# country-year counts, the unit the decomposition needs (Engel's point)
cy = df.groupby(["country", "paper_country", "year"]).agg(
    gpt_rows=("row_id", "size"), accused=("accused", "sum"),
    **{c: (c, "sum") for c in ACC}).reset_index()
cy.to_csv(P / "14_country_year_content.csv", index=False)

# Israel against South Africa in the years both appear (1981-1993)
w = df[df.year.between(1981, 1993) & (df.israel.eq(1) | sa)]
w = w.assign(who=w.israel.map({1: "Israel", 0: "South Africa"}))
isa = pd.DataFrame({k: summarise(g) for k, g in w.groupby("who")}).T
isa["years"] = 13
for c in ["gpt_rows", "accused"] + ACC:
    isa[f"{c}_per_year"] = (isa[c] / 13).round(1)
isa.to_csv(T / "14_israel_southafrica_1981_1993.csv")

# Israel by period
isr = df[df.israel == 1].copy()
isr["period"] = pd.cut(isr.year, [1980, 1993, 2008, 2023],
                       labels=["1981-1993", "1994-2008", "2009-2023"])
ip = pd.DataFrame({k: summarise(g) for k, g in isr.groupby("period", observed=True)}).T
ip.to_csv(T / "14_israel_periods.csv")

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
print(gt.to_string())
print()
print(isa.to_string())
print()
print(ip.to_string())
print()
print(ct[["gpt_rows", "accused", "false_pos", "not_accused", "acc_R", "acc_R_strict",
          "acc_C", "acc_C_strict", "acc_F", "acc_O", "acc_N", "acc_T", "min", "max"]]
      .to_string())
