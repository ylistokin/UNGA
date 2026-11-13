"""
22_referee2_package.py

Task card 22 (28 October 2026): the Referee 2 package in its final form, on the
corrected data (count variant A, memo 19), following Matanel's eight decisions
of 28 October 2026 (task log, "Decisions by Matanel for card 22"):

  Part A. Combined model. The decomposition with the three gravity-of-violation
          variables together: EPR discrimination (any DISCRIMINATED group,
          yes/no), first use of force (Brown's rule on MID 5.0, low threshold,
          yes/no), and V-Dem liberal democracy. Because the first-use variable
          ends in 2014, the combined model is run on 1981-2014; a version
          without first use is run on 1981-2023. Every specification is shown
          four ways: as is, without South Africa's rows, without the nuclear
          variable, and without both (decision 5).
  Part B. Israel against South Africa: a figure of the two countries' annual
          share of criticism 1981-2023, and a table for 1981-1993 (resolutions
          per year, share, and what the accusations were about, strict and
          broad readings side by side; decision 2).
  Part C. The grave-violation case list with the four corrections of memo 15
          (decision 4), recomputed on the corrected counts.
  Part D. The content table: what Israel, South Africa and the comparison
          states are accused of, strict and broad (decision 2).
  Part E. Criticism without naming the violator (Afghanistan, Kampuchea,
          Cyprus): the numbers behind the proposed paragraph (memo 08
          section 6.2, memo 15 section 2), on the corrected counts.

Inputs (read-only)
  code/19_corrected_tables.py, code/21_referee1_controls_and_democracy.py
      (the corrected panel with V-Dem and the other controls)
  data/raw/EPR-2021.csv                       EPR Core 2021 (memo 05/06 citation)
  data/processed/10_first_use_country_year.csv  first use of force (memo 10)
  data/processed/14_row_codes.csv             content codes (memo 14)
  data/processed/14_country_year_content.csv
  output/tables/08_topic_check_1981_1993.csv  unnamed-criticism topic check (memo 08)

Outputs: output/tables/22_* and output/figures/22_israel_southafrica_shares.png
Run from the project root:
    python3 code/22_referee2_package.py                      (points, figure, tables)
    bash code/22_run_bootstrap.sh                            (bootstrap chunks)
    python3 code/22_referee2_package.py --stage summary
Requires: python 3.10+, numpy, pandas, matplotlib. No network access.
"""

import argparse
import importlib
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
c19 = importlib.import_module("19_corrected_tables")
c21 = importlib.import_module("21_referee1_controls_and_democracy")

RAW, PROC, OUT = ROOT / "data" / "raw", ROOT / "data" / "processed", ROOT / "output" / "tables"
FIG = ROOT / "output" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
V8 = c19.V8
LABEL = dict(c21.LABEL, epr_disc="EPR discrimination (yes/no)", fu_low_any="First use of force (yes/no)")
LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(str(s))


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def epr_country_year():
    """EPR 2021 to one row per country-year (memo 06's rule): 1 if any group is
    DISCRIMINATED; 2021 values copied to 2022-2023."""
    e = pd.read_csv(RAW / "EPR-2021.csv")
    e = e[(e["to"] >= 1981) & (e["from"] <= 2021)]
    rows = []
    for r in e.itertuples(index=False):
        for y in range(max(getattr(r, "_2"), 1981), min(r.to, 2021) + 1):
            rows.append((r.gwid, y, r.status == "DISCRIMINATED"))
    g = pd.DataFrame(rows, columns=["gwid", "year", "disc"])
    cy = g.groupby(["gwid", "year"]).disc.max().astype(float).rename("epr_disc").reset_index()
    last = cy[cy.year == 2021]
    return pd.concat([cy] + [last.assign(year=y) for y in (2022, 2023)], ignore_index=True)


def build():
    p = c21.build()
    p["gwid"] = p["cow"].astype(int).replace({255: 260, 679: 678})
    p.loc[p["country"] == "Serbia", "gwid"] = np.where(p.loc[p["country"] == "Serbia", "year"] >= 2006, 340, 345)
    p = p.merge(epr_country_year(), on=["gwid", "year"], how="left")
    fu = pd.read_csv(PROC / "10_first_use_country_year.csv")[["cow", "year", "fu_low_any"]]
    fu["year"] = fu["year"].astype(int)
    p = p.merge(fu, on=["cow", "year"], how="left")
    p.loc[p.year > 2014, "fu_low_any"] = np.nan
    p["is_sa"] = (p["country"] == "South Africa").astype(int)
    return p


# ---------------------------------------------------------------------------
# Part A. Combined model, four ways
# ---------------------------------------------------------------------------
SPECS = {
    "baseline 1981-2023": ([], 2023),
    "+ EPR discrimination": (["epr_disc"], 2023),
    "+ liberal democracy": (["libdem"], 2023),
    "+ EPR + democracy (1981-2023)": (["epr_disc", "libdem"], 2023),
    "baseline 1981-2014": ([], 2014),
    "+ first use of force (1981-2014)": (["fu_low_any"], 2014),
    "combined: EPR + first use + democracy (1981-2014)": (["epr_disc", "fu_low_any", "libdem"], 2014),
}
VARIANTS = {"as is": (True, True), "without South Africa": (False, True),
            "without nuclear variable": (True, False), "without both": (False, False)}


def spec_data(p, spec, variant):
    extra, last_year = SPECS[spec]
    keep_sa, keep_nuc = VARIANTS[variant]
    V = [v for v in V8 + extra if keep_nuc or v != "NuclearPowerOutsideNPT"]
    D = p[p.year <= last_year]
    if not keep_sa:
        D = D[D.is_sa == 0]
    D = D[D.Israel_dummy.eq(1) | D[V + ["share"]].notna().all(axis=1)]
    return D, V


def part_a(p):
    rows, crows = [], []
    for spec in SPECS:
        for variant in VARIANTS:
            D, V = spec_data(p, spec, variant)
            for model, ml in [("re_year", "RE"), ("fe", "FE")]:
                e, o, c = c19.explained(D, model, V)
                rows.append(dict(spec=spec, variant=variant, model=ml, explained=round(e, 4), overall=round(o, 4),
                                 explained_pct=round(100 * e / o, 1), n_other_rows=int((D.Israel_dummy == 0).sum())))
                crows.append(dict(spec=spec, variant=variant, model=ml, **{LABEL.get(v, v): round(c[v], 4) for v in V}, total=round(e, 4)))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "22_combined_points.csv", index=False)
    pd.DataFrame(crows).fillna("").to_csv(OUT / "22_combined_contributions.csv", index=False)
    say("\n=== Part A. Combined model, point estimates ===")
    say(t.to_string(index=False))
    # the coefficients of the gravity variables among the others, combined model
    D, V = spec_data(p, "combined: EPR + first use + democracy (1981-2014)", "as is")
    o = D[D.Israel_dummy == 0]
    from panel_tools import random_effects, fixed_effects
    re = random_effects(o, "share", V, "cid", time="year")
    fe = fixed_effects(o, "share", [v for v in V if v != "PermanentUNSCMember"], "cid", time="year")
    crow = []
    for v in V:
        k = 1000 if v == "battle_fat" else 1
        crow.append(dict(variable=LABEL.get(v, v), RE=f"{k * re.params[v]:.4f}{re.stars()[v]} ({k * re.se[v]:.4f})",
                         FE=(f"{k * fe.params[v]:.4f}{fe.stars()[v]} ({k * fe.se[v]:.4f})" if v in fe.params.index else "absorbed")))
    crow.append(dict(variable="N", RE=re.nobs, FE=fe.nobs))
    t2 = pd.DataFrame(crow)
    t2.to_csv(OUT / "22_combined_coefficients.csv", index=False)
    say("\nCoefficients, combined model 1981-2014 (others only):")
    say(t2.to_string(index=False))
    # means of the gravity variables
    m = []
    for v in ["epr_disc", "fu_low_any", "libdem", "NuclearPowerOutsideNPT", "TerritoryOccupied"]:
        m.append(dict(variable=LABEL.get(v, v), israel=round(D[D.Israel_dummy == 1][v].mean(), 3), others=round(o[v].mean(), 3)))
    say(pd.DataFrame(m).to_string(index=False))


# ---------------------------------------------------------------------------
# Part B. Israel against South Africa
# ---------------------------------------------------------------------------
def part_b(p):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    isr = p[p.country == "Israel"].set_index("year").sort_index()
    sa = p[p.country == "South Africa"].set_index("year").sort_index()
    others = p[(p.country != "Israel") & (p.country != "South Africa")]
    top = others.groupby("year").share.max()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(isr.index, isr.share, color="black", lw=2, label="Israel")
    ax.plot(sa.index, sa.share, color="grey", lw=2, ls="--", label="South Africa")
    ax.plot(top.index, top.values, color="lightgrey", lw=1.5, ls=":", label="Highest other country in that year")
    ax.axvline(1994, color="lightgrey", lw=0.8)
    ax.set_ylabel("Share of the year's critical UNGA resolutions")
    ax.set_xlabel("Year")
    ax.set_ylim(0, 0.7)
    ax.legend(frameon=False)
    ax.set_title("Israel and South Africa, 1981-2023")
    fig.tight_layout()
    fig.savefig(FIG / "22_israel_southafrica_shares.png", dpi=200)
    pd.DataFrame({"year": isr.index, "israel_share": isr.share.values, "israel_count": isr["count"].values,
                  "south_africa_share": sa.share.reindex(isr.index).values, "south_africa_count": sa["count"].reindex(isr.index).values,
                  "highest_other_share": top.reindex(isr.index).values}).round(4).to_csv(OUT / "22_israel_southafrica_by_year.csv", index=False)
    # 1981-1993 table with content
    cyc = pd.read_csv(PROC / "14_country_year_content.csv")
    rows = []
    for name in ["Israel", "South Africa"]:
        d = p[(p.country == name) & (p.year <= 1993)]
        c = cyc[(cyc.country == name) & (cyc.year <= 1993)]
        n = len(d)
        rows.append({
            "country": name, "years": "1981-1993",
            "critical resolutions per year (corrected counts)": round(d["count"].mean(), 1),
            "mean share": round(d.share.mean(), 3),
            "resolutions read per year (memo 14)": round(c.gpt_rows.sum() / n, 1),
            "accusing the state, per year": round(c.accused.sum() / n, 1),
            "racial discrimination or apartheid, strict": round(c.acc_R_strict.sum() / n, 1),
            "racial discrimination or apartheid, broad": round(c.acc_R.sum() / n, 1),
            "occupation, annexation, self-determination": round(c.acc_O.sum() / n, 1),
            "use of force against another state": round(c.acc_F.sum() / n, 1),
            "international crimes, strict": round(c.acc_C_strict.sum() / n, 1),
            "international crimes, broad": round(c.acc_C.sum() / n, 1),
            "nuclear": round(c.acc_N.sum() / n, 1), "torture, slavery": round(c.acc_T.sum() / n, 1),
            "human rights score (mean)": round(d.human_rights_score.mean(), 1),
            "war fatalities (mean)": round(d.battle_fat.mean(), 0),
        })
    t = pd.DataFrame(rows).set_index("country").T
    t.to_csv(OUT / "22_israel_southafrica_1981_1993.csv")
    say("\n=== Part B. Israel and South Africa, 1981-1993 ===")
    say(t.to_string())
    say(f"Figure saved: output/figures/22_israel_southafrica_shares.png. Peak shares: Israel {isr.share.max():.3f} ({isr.share.idxmax()}), "
        f"South Africa {sa.share.max():.3f} ({sa.share.idxmax()}). South Africa 1981-1993 mean share {sa.loc[1981:1993].share.mean():.3f}, "
        f"Israel same years {isr.loc[1981:1993].share.mean():.3f}; South Africa 1994-2023 mean {sa.loc[1994:].share.mean():.4f}.")


# ---------------------------------------------------------------------------
# Part C. Case list with the four corrections
# ---------------------------------------------------------------------------
CASES = [
    # case_id, list, category, state, cow, start, end, basis
    ("rwanda_genocide", "core", "genocide / crimes against humanity", "Rwanda", 517, 1994, 1994, "ICTR judgments"),
    ("sudan_darfur", "core", "genocide / crimes against humanity", "Sudan", 625, 2003, 2004, "ICC verdict, Abd-Al-Rahman, Oct 2025 (crimes against humanity)"),
    ("iraq_anfal", "core", "genocide / crimes against humanity", "Iraq", 645, 1988, 1988, "Trahan 2009 (moved from contested; year corrected to 1988)"),
    ("guatemala_maya", "core", "genocide / crimes against humanity", "Guatemala", 90, 1981, 1983, "Brett 2016 (moved from contested)"),
    ("usa_nicaragua", "core", "use of force", "United States of America", 2, 1981, 1985, "ICJ Nicaragua v. US, 1986 (fact window 1981-1985)"),
    ("iraq_kuwait", "core", "use of force", "Iraq", 645, 1990, 1991, "O'Connell 1991"),
    ("eritrea_badme", "core", "use of force", "Eritrea", 531, 1998, 1998, "Eritrea-Ethiopia Claims Commission, 2005"),
    ("uganda_drc", "core", "use of force", "Uganda", 500, 1998, 2003, "ICJ Armed Activities, 2005"),
    ("russia_ukraine", "core", "use of force", "Russia", 365, 2014, 2023, "ECtHR 2024 and 2025; literature on the force question"),
    ("israel_opt", "core", "occupation / annexation", "Israel", 666, 1981, 2023, "ICJ advisory opinions 2004 and 2024"),
    ("morocco_wsahara", "core", "occupation / annexation", "Morocco", 600, 1981, 2023, "ICJ advisory opinion 1975; CJEU 2024"),
    ("turkey_ncyprus", "core", "occupation / annexation", "Turkey", 640, 1981, 2023, "ECtHR Loizidou 1996, Cyprus v. Turkey 2001"),
    ("indonesia_etimor", "core", "occupation / annexation", "Indonesia", 850, 1981, 1999, "Clark 1980; ICJ East Timor 1995"),
    ("southafrica_namibia", "core", "occupation / annexation", "South Africa", 560, 1981, 1990, "ICJ Namibia advisory opinion 1971"),
    ("ussr_afghanistan", "core", "occupation / annexation", "Russia", 365, 1981, 1989, "Terry 2011 (Soviet Union, recorded under Russia)"),
    ("russia_crimea", "core", "occupation / annexation", "Russia", 365, 2014, 2023, "ECtHR Ukraine v. Russia (re Crimea), 2024"),
    ("southafrica_apartheid", "core", "apartheid", "South Africa", 560, 1981, 1994, "Dubow 2014; universal"),
    ("israel_apartheid", "core", "apartheid", "Israel", 666, 1981, 2023, "Matanel's decision (strict coding); ICJ 2024 CERD art. 3"),
    ("iraq_iran", "contested", "use of force", "Iraq", 645, 1981, 1988, "no scholarly source found (moved from core)"),
    ("syria_detention", "contested", "genocide / crimes against humanity", "Syria", 652, 2011, 2023, "national court only"),
    ("myanmar_rohingya", "contested", "genocide / crimes against humanity", "Myanmar", 775, 2016, 2017, "ICJ judgment pending"),
    ("usa_iraq2003", "contested", "use of force", "United States of America", 2, 2003, 2003, "Murphy 2004: scholarly opinion divided"),
    ("uk_iraq2003", "contested", "use of force", "United Kingdom", 200, 2003, 2003, "scholarly opinion divided"),
    ("armenia_karabakh", "contested", "occupation / annexation", "Armenia", 371, 1992, 2020, "ECtHR Chiragov 2015; framing disputed"),
    ("vietnam_cambodia", "contested", "occupation / annexation", "Vietnam", 816, 1981, 1989, "scholarly opinion divided"),
]


def part_c(p):
    cases = pd.DataFrame(CASES, columns=["case_id", "list", "category", "state", "cow", "start", "end", "basis"])
    cases.to_csv(PROC / "22_case_list.csv", index=False)
    isr = p[p.country == "Israel"].set_index("year")
    rows = []
    for c in cases.itertuples():
        s = p[(p.cow == c.cow) & p.year.between(c.start, c.end)].set_index("year")
        yrs = [y for y in range(c.start, c.end + 1) if y in isr.index]
        rows.append(dict(case_id=c.case_id, list=c.list, category=c.category, state=c.state, years=f"{c.start}-{c.end}",
                         n_years=len(s), total_critical_res=int(s["count"].sum()), mean_count_per_year=round(s["count"].mean(), 2),
                         mean_share=round(s.share.mean(), 3), years_zero_criticism=int((s["count"] == 0).sum()),
                         israel_same_years_mean_count=round(isr.loc[yrs, "count"].mean(), 2),
                         israel_same_years_mean_share=round(isr.loc[yrs, "share"].mean(), 3), basis=c.basis))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "22_case_comparison.csv", index=False)
    say("\n=== Part C. Case list (corrected dates, corrected counts) ===")
    say(t.drop(columns=["basis", "category"]).to_string(index=False))


# ---------------------------------------------------------------------------
# Part D. Content table, strict and broad
# ---------------------------------------------------------------------------
def part_d():
    r = pd.read_csv(PROC / "14_row_codes.csv")
    r["group"] = np.where(r.country == "Israel", "Israel", np.where(r.country == "South Africa", "South Africa", "Comparison states"))
    rows = []
    for g, d in r.groupby("group"):
        acc = d[d.accused == 1]
        n = len(acc)
        row = dict(group=g, gpt_rows=len(d), accused=n, share_accused=round(n / len(d), 3), gpt_errors=int(d.false_pos.sum()))
        for k, lab in [("acc_R_strict", "racial/apartheid, strict"), ("acc_R", "racial/apartheid, broad"),
                       ("acc_O", "occupation"), ("acc_F", "use of force"), ("acc_C_strict", "crimes, strict"),
                       ("acc_C", "crimes, broad"), ("acc_N", "nuclear"), ("acc_T", "torture/slavery"), ("acc_other_only", "other only")]:
            row[lab] = f"{int(d[k].sum())} ({100 * d[k].sum() / n:.0f}%)"
        rows.append(row)
    t = pd.DataFrame(rows).set_index("group").loc[["Israel", "South Africa", "Comparison states"]]
    t.to_csv(OUT / "22_content_strict_broad.csv")
    say("\n=== Part D. What the accusations are about (share of accusing resolutions) ===")
    say(t.to_string())
    # Israel's broad-R rows
    w = r[(r.country == "Israel") & (r.acc_R == 1)]
    say(f"Israel rows counted as racial/apartheid under the broad reading: {len(w)}, years {sorted(w.year.unique())}")
    w[["res_id", "year", "quote"]].to_csv(OUT / "22_israel_broad_R_rows.csv", index=False)


# ---------------------------------------------------------------------------
# Part E. Unnamed criticism
# ---------------------------------------------------------------------------
def part_e(p):
    t = pd.read_csv(OUT / "08_topic_check_1981_1993.csv")
    say("\n=== Part E. Criticism without naming the violator, 1981-1993 (memo 08 topic check) ===")
    say(t.to_string(index=False))
    for name, cow in [("Afghanistan (USSR)", 365), ("Vietnam", 816), ("Turkey", 640), ("Israel", 666)]:
        s = p[(p.cow == cow) & (p.year <= 1993)]
        say(f"  named critical resolutions per year 1981-1993, {name}: {s['count'].mean():.2f}")


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
def rounds(p, spec, variant, model, seeds):
    D, V = spec_data(p, spec, variant)
    D = D.reset_index(drop=True)
    isr = int(D.loc[D.Israel_dummy == 1, "cid"].iloc[0])
    groups = {c: idx for c, idx in D.groupby("cid").indices.items() if c != isr}
    out = []
    for s in seeds:
        S = c19.draw_countries(D, np.random.default_rng(s), groups)
        try:
            out.append(dict(spec=spec, variant=variant, model=model, seed=s, explained=c19.explained(S, model, V)[0], failed=""))
        except Exception as ex:
            out.append(dict(spec=spec, variant=variant, model=model, seed=s, explained=np.nan, failed=str(ex)))
    out = pd.DataFrame(out)
    path = OUT / "22_boot_rounds.csv"
    out.to_csv(path, mode="a", header=not path.exists(), index=False)
    say(f"{spec} | {variant} | {model}: ok {out.explained.notna().sum()}, failed {out.explained.isna().sum()}")


def summary(p):
    r = pd.read_csv(OUT / "22_boot_rounds.csv").drop_duplicates(["spec", "variant", "model", "seed"])
    res, fails = [], []
    for (spec, variant, model), g in r.groupby(["spec", "variant", "model"]):
        D, V = spec_data(p, spec, variant)
        pt, od, _ = c19.explained(D, model, V)
        ok = g.explained.dropna().values
        res.append(dict(spec=spec, variant=variant, model=("RE" if model == "re_year" else "FE"), explained=round(pt, 4), overall=round(od, 4),
                        explained_pct=round(100 * pt / od, 1), boot_se=round(ok.std(ddof=1), 4), ratio=round(pt / ok.std(ddof=1), 2),
                        rounds_ok=len(ok), rounds_failed=int(g.explained.isna().sum())))
        for reason, n in Counter(g.failed.dropna()).items():
            if reason:
                fails.append(dict(spec=spec, variant=variant, model=model, reason=reason, rounds=n))
    t = pd.DataFrame(res)
    t.to_csv(OUT / "22_bootstrap.csv", index=False)
    pd.DataFrame(fails).to_csv(OUT / "22_bootstrap_failures.csv", index=False)
    say("\n=== Bootstrap (500 rounds, countries resampled, Israel fixed) ===")
    say(t.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="points", choices=["points", "boot", "summary"])
    ap.add_argument("--spec", default="combined: EPR + first use + democracy (1981-2014)")
    ap.add_argument("--variant", default="as is")
    ap.add_argument("--model", default="re_year")
    ap.add_argument("--seeds", default="0:500")
    a = ap.parse_args()
    say("Card 22: Referee 2 package. Run " + time.strftime("%Y-%m-%d %H:%M"))
    p = build()
    if a.stage == "points":
        part_a(p)
        part_b(p)
        part_c(p)
        part_d()
        part_e(p)
        (OUT / "22_run_log_points.txt").write_text("\n".join(LOG))
    elif a.stage == "boot":
        lo, hi = [int(x) for x in a.seeds.split(":")]
        rounds(p, a.spec, a.variant, a.model, list(range(lo, hi)))
    else:
        summary(p)
        (OUT / "22_run_log_summary.txt").write_text("\n".join(LOG))
