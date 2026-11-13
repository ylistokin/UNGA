"""
08_grave_violations_comparison.py

Card 8 (the task log calls it "Card 6"). Referee 2 asks whether Israel is
condemned more because its violations are graver. This script takes a
hand-built list of states that, by an international court finding or broad
scholarly agreement, committed grave violations of international law, and
counts how much UNGA criticism each received in its violation years. It sets
this beside Israel.

Nothing here enters the model. It is a side-by-side descriptive table.

The case list is written out below (CASES), so the script is the single record
of what was coded and why. The selection rule and every source are in
docs/08_grave_violations_case_list.md. Decisions taken by Matanel on
24 Sep 2026:
  - Basis: international, hybrid or regional court or tribunal findings
    (verdicts and judgments, including ICJ advisory opinions), or broad
    scholarly agreement. NOT: UN political organs, national courts, truth
    commissions, ICC arrest warrants or confirmed charges.
  - The Eritrea-Ethiopia Claims Commission counts as an international tribunal.
  - Israel coded strictly: counted as an apartheid violator in all window
    years, 1981-2023.
  - South Africa apartheid dated to 1994 (research convention).
  - Srebrenica removed. Philippines removed (charges only, no verdict).
  - Contested cases kept as a separate check.
Many case dates rest on memory and are marked "memory" in the memo; they
must be verified before any number here is used in the paper.

Inputs (read-only):
  Israel and the UN/Data/final_data.dta
      the file the paper used. Columns used: country, year, cow, count, share.
      count = number of critical UNGA resolutions naming the country that year.
      share = the paper's dependent variable, that country's share of the
      year's critical resolutions.

Choices made for display only (not decisions, flagged in the memo):
  - The Soviet Union 1981-1991 is recorded in final_data.dta under Russia
    (cow 365). The Afghanistan case therefore reads the Russia rows.
  - Two comparisons with Israel are shown for each case: Israel in the SAME
    calendar years as the case (because the volume of criticism changes from
    year to year), and Israel over its own violation years in that category.
  - Kosovo 1999 (contested) is not computed: which NATO states to count is a
    judgment not yet taken.
  - Iraq 2003 (contested) is computed for the United States and the United
    Kingdom only.

Outputs:
  data/processed/08_case_list.csv        the case list as data
  output/tables/08_case_comparison.csv   one row per case
  output/tables/08_state_summary.csv     one row per state, core cases only,
                                         union of its violation years
  output/tables/08_category_summary.csv  per category: Israel vs the others
  output/tables/08_topic_check_1981_1993.csv  part 2, see below
  output/tables/08_run_log.txt           everything printed

Part 2, a check on the zeros (1981-1993 only):
  Some violators get zero critical resolutions because the General Assembly
  criticised the conduct without naming the state (for example "the immediate
  withdrawal of the foreign troops from Afghanistan"). The GPT label records
  only a named target, so such resolutions do not count against the violator.
  Part 2 counts, for the 1981-1993 cases, resolutions whose TOPIC matches the
  situation, whether or not the violator is named, and shows the GPT label
  each received. Input: Israel and the UN/Data/GPT_Counts/MesquitaGPT.csv
  (columns used: res_id2, date_p, topic, content, country). The keywords are
  my choice, for display only. 1994-2023 cases are not checked here.

Run from the project root:
    python3 code/08_grave_violations_comparison.py
Requires: python 3.10+, pandas. No network access.
"""

import io
import os
import sys
import pandas as pd

FINAL = "Israel and the UN/Data/final_data.dta"
OUT_T = "output/tables"
OUT_P = "data/processed"
ISRAEL_COW = 666

# case_id, list, category, state, cow, start, end, basis
CASES = [
    # ---- core list ----
    ("rwanda_genocide", "core", "genocide / crimes against humanity", "Rwanda", 517, 1994, 1994,
     "ICTR judgments"),
    ("sudan_darfur", "core", "genocide / crimes against humanity", "Sudan", 625, 2003, 2004,
     "ICC verdict, Abd-Al-Rahman, Oct 2025 (crimes against humanity)"),
    ("usa_nicaragua", "core", "use of force", "United States of America", 2, 1981, 1986,
     "ICJ Nicaragua v. US, 1986"),
    ("iraq_iran", "core", "use of force", "Iraq", 645, 1981, 1988,
     "literature (war began 1980)"),
    ("iraq_kuwait", "core", "use of force", "Iraq", 645, 1990, 1991,
     "literature"),
    ("eritrea_badme", "core", "use of force", "Eritrea", 531, 1998, 1998,
     "Eritrea-Ethiopia Claims Commission, 2005"),
    ("uganda_drc", "core", "use of force", "Uganda", 500, 1998, 2003,
     "ICJ Armed Activities, 2005"),
    ("russia_ukraine", "core", "use of force", "Russia", 365, 2014, 2023,
     "ECtHR 2024 and 2025; literature on the force question"),
    ("israel_opt", "core", "occupation / annexation", "Israel", ISRAEL_COW, 1981, 2023,
     "ICJ advisory opinions 2004 and 2024"),
    ("morocco_wsahara", "core", "occupation / annexation", "Morocco", 600, 1981, 2023,
     "ICJ advisory opinion 1975; CJEU 2024"),
    ("turkey_ncyprus", "core", "occupation / annexation", "Turkey", 640, 1981, 2023,
     "ECtHR Loizidou 1996, Cyprus v. Turkey 2001"),
    ("indonesia_etimor", "core", "occupation / annexation", "Indonesia", 850, 1981, 1999,
     "literature; ICJ East Timor 1995"),
    ("southafrica_namibia", "core", "occupation / annexation", "South Africa", 560, 1981, 1990,
     "ICJ Namibia advisory opinion 1971"),
    ("ussr_afghanistan", "core", "occupation / annexation", "Russia", 365, 1981, 1989,
     "literature (Soviet Union, recorded under Russia)"),
    ("russia_crimea", "core", "occupation / annexation", "Russia", 365, 2014, 2023,
     "ECtHR Ukraine v. Russia (re Crimea), 2024"),
    ("southafrica_apartheid", "core", "apartheid", "South Africa", 560, 1981, 1994,
     "universal"),
    ("israel_apartheid", "core", "apartheid", "Israel", ISRAEL_COW, 1981, 2023,
     "Matanel's decision (strict coding); ICJ 2024 CERD art. 3"),
    # ---- contested list (separate check) ----
    ("iraq_anfal", "contested", "genocide / crimes against humanity", "Iraq", 645, 1987, 1988,
     "national court only; needs a named scholarly source"),
    ("guatemala_maya", "contested", "genocide / crimes against humanity", "Guatemala", 90, 1981, 1983,
     "truth commission only; needs a named scholarly source"),
    ("syria_detention", "contested", "genocide / crimes against humanity", "Syria", 652, 2011, 2023,
     "national court only; needs a named scholarly source"),
    ("myanmar_rohingya", "contested", "genocide / crimes against humanity", "Myanmar", 775, 2016, 2017,
     "ICJ judgment pending"),
    ("usa_iraq2003", "contested", "use of force", "United States of America", 2, 2003, 2003,
     "scholarly opinion divided"),
    ("uk_iraq2003", "contested", "use of force", "United Kingdom", 200, 2003, 2003,
     "scholarly opinion divided"),
    ("armenia_karabakh", "contested", "occupation / annexation", "Armenia", 371, 1992, 2020,
     "ECtHR Chiragov 2015; framing disputed"),
    ("vietnam_cambodia", "contested", "occupation / annexation", "Vietnam", 816, 1981, 1989,
     "scholarly opinion divided"),
]

log = io.StringIO()


def out(*a):
    print(*a)
    print(*a, file=log)


def main():
    os.makedirs(OUT_T, exist_ok=True)
    os.makedirs(OUT_P, exist_ok=True)

    d = pd.read_stata(FINAL, convert_categoricals=False)
    d = d[["country", "year", "cow", "count", "share"]].copy()
    d["year"] = d["year"].astype(int)
    d["cow"] = d["cow"].astype(int)

    isr = d[d["cow"] == ISRAEL_COW].set_index("year")
    out("Check against CLAUDE.md Table 1:")
    out(f"  Israel years {len(isr)} (expected 43), mean count {isr['count'].mean():.1f} (19.4),"
        f" mean share {isr['share'].mean():.3f} (0.411)")

    cases = pd.DataFrame(CASES, columns=["case_id", "list", "category", "state", "cow",
                                         "start", "end", "basis"])
    cases.to_csv(f"{OUT_P}/08_case_list.csv", index=False)

    rows = []
    for c in cases.itertuples():
        s = d[(d["cow"] == c.cow) & d["year"].between(c.start, c.end)].set_index("year")
        yrs = list(range(c.start, c.end + 1))
        missing = [y for y in yrs if y not in s.index]
        names = sorted(s["country"].unique())
        i_same = isr.loc[[y for y in yrs if y in isr.index]]
        rows.append({
            "case_id": c.case_id, "list": c.list, "category": c.category, "state": c.state,
            "years": f"{c.start}-{c.end}", "n_years": len(s),
            "missing_years": ",".join(map(str, missing)),
            "name_in_data": "; ".join(names),
            "total_critical_res": int(s["count"].sum()),
            "mean_count_per_year": round(s["count"].mean(), 2),
            "mean_share": round(s["share"].mean(), 3),
            "max_share": round(s["share"].max(), 3),
            "years_zero_criticism": int((s["count"] == 0).sum()),
            "israel_same_years_mean_count": round(i_same["count"].mean(), 2),
            "israel_same_years_mean_share": round(i_same["share"].mean(), 3),
            "basis": c.basis,
        })
    res = pd.DataFrame(rows)
    res.to_csv(f"{OUT_T}/08_case_comparison.csv", index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    show = ["case_id", "list", "years", "n_years", "total_critical_res", "mean_count_per_year",
            "mean_share", "years_zero_criticism", "israel_same_years_mean_count",
            "israel_same_years_mean_share"]
    for lst in ["core", "contested"]:
        out(f"\n=== {lst} list ===")
        out(res[res["list"] == lst][show].to_string(index=False))
    miss = res[res["missing_years"] != ""]
    if len(miss):
        out("\nCases with years missing from final_data.dta:")
        out(miss[["case_id", "years", "missing_years"]].to_string(index=False))

    # per state, core only, union of violation years
    core = cases[cases["list"] == "core"]
    srows = []
    for cow, g in core.groupby("cow"):
        yrs = sorted({y for r in g.itertuples() for y in range(r.start, r.end + 1)})
        s = d[(d["cow"] == cow) & d["year"].isin(yrs)]
        srows.append({
            "state": g["state"].iloc[0], "cases": "; ".join(g["case_id"]),
            "n_years": len(s), "first": min(yrs), "last": max(yrs),
            "total_critical_res": int(s["count"].sum()),
            "mean_count_per_year": round(s["count"].mean(), 2),
            "mean_share": round(s["share"].mean(), 3),
            "years_zero_criticism": int((s["count"] == 0).sum()),
        })
    st = pd.DataFrame(srows).sort_values("mean_share", ascending=False)
    st.to_csv(f"{OUT_T}/08_state_summary.csv", index=False)
    out("\n=== core list, per state (union of violation years) ===")
    out(st.to_string(index=False))

    # per category: Israel vs the other states' country-years
    crows = []
    for (lst, cat), g in cases.groupby(["list", "category"]):
        others = g[g["cow"] != ISRAEL_COW]
        cy = set()
        for r in others.itertuples():
            cy |= {(r.cow, y) for y in range(r.start, r.end + 1)}
        o = d[[(a, b) in cy for a, b in zip(d["cow"], d["year"])]]
        ig = g[g["cow"] == ISRAEL_COW]
        iy = {y for r in ig.itertuples() for y in range(r.start, r.end + 1)}
        i = isr.loc[sorted(iy)] if iy else isr.iloc[0:0]
        crows.append({
            "list": lst, "category": cat,
            "other_states": len(others["cow"].unique()),
            "other_country_years": len(o),
            "other_mean_share": round(o["share"].mean(), 3) if len(o) else None,
            "other_mean_count": round(o["count"].mean(), 2) if len(o) else None,
            "israel_in_category": "yes" if len(ig) else "no",
            "israel_mean_share": round(i["share"].mean(), 3) if len(i) else None,
            "israel_mean_count": round(i["count"].mean(), 2) if len(i) else None,
        })
    ct = pd.DataFrame(crows)
    ct.to_csv(f"{OUT_T}/08_category_summary.csv", index=False)
    out("\n=== per category ===")
    out(ct.to_string(index=False))

    topic_check(isr)

    with open(f"{OUT_T}/08_run_log.txt", "w", encoding="utf-8") as f:
        f.write(log.getvalue())


MESQ = "Israel and the UN/Data/GPT_Counts/MesquitaGPT.csv"
# case_id, topic keyword (regex), words that would name the violator (regex)
TOPICS = [
    ("ussr_afghanistan", r"situation in afghanistan", r"soviet|u\.s\.s\.r"),
    ("vietnam_cambodia", r"kampuchea", r"viet ?nam"),
    ("turkey_ncyprus", r"cyprus", r"turkey|turkish"),
    ("indonesia_etimor", r"east timor", r"indonesia"),
    ("iraq_iran", r"iran and iraq|between iran|iraq and iran", r"\biraq"),
    ("morocco_wsahara", r"western sahara", r"morocco"),
    ("southafrica_namibia", r"namibia", r"south africa"),
]


def topic_check(isr):
    m = pd.read_csv(MESQ)
    m["year"] = pd.to_datetime(m["date_p"]).dt.year
    cases = {c[0]: c for c in CASES}
    rows = []
    for cid, pat, name_pat in TOPICS:
        c = cases[cid]
        yrs = range(c[5], min(c[6], 1993) + 1)
        s = m[m["topic"].str.contains(pat, case=False, na=False) & m["year"].isin(yrs)]
        labels = s["country"].fillna("None").value_counts().to_dict()
        rows.append({
            "case_id": cid, "years": f"{min(yrs)}-{max(yrs)}",
            "topic_resolutions": len(s),
            "topic_res_per_year": round(len(s) / len(yrs), 2),
            "text_names_violator": int(s["content"].str.contains(name_pat, case=False, na=False).sum()),
            "gpt_labels": "; ".join(f"{k}: {v}" for k, v in labels.items()),
            "israel_named_per_year_same_years": round(isr.loc[list(yrs), "count"].mean(), 2),
        })
    r = pd.DataFrame(rows)
    r.to_csv(f"{OUT_T}/08_topic_check_1981_1993.csv", index=False)
    out("\n=== part 2: resolutions on the situation, named or not (1981-1993) ===")
    out(r.to_string(index=False))


if __name__ == "__main__":
    if not os.path.exists(FINAL):
        sys.exit("Run from the project root (folder containing 'Israel and the UN').")
    main()
