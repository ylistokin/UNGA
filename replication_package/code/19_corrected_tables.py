"""
19_corrected_tables.py

Task card 19 (28 October 2026): Tables 1, 2 and 3 of the paper with the
technical corrections Yair approved on 30 September 2026, and the two count
corrections found in September (memo 01 question 6, memo 09 side finding).

Corrections to the METHOD (Yair's answers in the 30 Sep 2026 email):
  M1. Year effects are really included in the random effects model (the
      published notebook asked for them but the software ignored them).
  M2. Permanent Security Council membership enters the decomposition (it was
      in Table 2 but not in Table 3). In the fixed effects model it cannot be
      estimated because it never changes within a country, so it is omitted
      there, exactly as Stata omitted it from Table 2 column 2.
  M3. Row 3 of Table 3 is computed with one model throughout (both the
      random effects version, as its label says, and the fixed effects
      version, as it was computed, are reported; Yair chooses).
  M4. Bootstrap rounds that cannot be estimated are counted and reported,
      not skipped silently.
  M3a. Row 3's sample is the sample of Table 2 column 4 (every country-year
      with a positive share: 543 rows, 61 countries), as the paper's text
      says. Yair's notebook also dropped countries with a single positive
      year (527 rows, 45 countries). With that extra filter a random effects
      model with year dummies cannot be estimated (45 countries, 50
      regressors in the between step), which is one more reason the
      published row 3 ended up being a fixed effects number.
  M5. The F1 check against UN Watch uses UN Watch's own "Condemnatory" flag.

Corrections to the COUNTS (the data behind the share variable):
  C1. Multi-country GPT answers in 1981-1993 (14 resolutions, for example
      "South Africa, Namibia", "Iran, Iraq") were dropped for both countries
      by the original counting notebook, because its splitting step grouped
      by a column that is empty for those years. For 1994-2023 the same
      answers were split and counted. Variant A splits them the same way for
      all years. Variant B also drops the label "Namibia" from the "South
      Africa, Namibia" answers, because Namibia is the occupied territory in
      those resolutions, not the criticised state (memo 14 read them). Which
      variant the paper uses is Matanel's decision; both are reported.
  C2. UNGA session 48 runs from September 1993 to September 1994. The
      1981-1993 corpus (Mesquita) covers the whole session and dates it 1993;
      the 1994-2023 corpus (Voelsen) also contains the 50 session-48
      resolutions adopted in 1994. Two of them were labelled critical in both
      corpora (48/27B Haiti, 48/258B South Africa) and therefore counted
      twice, once in 1993 and once in 1994. The 1994 copies are dropped.

Inputs (read-only)
  Israel and the UN/Data/final_data.dta                    the paper's panel
  Israel and the UN/Data/full_text_data2.csv               GPT labels 1994-2024
  Israel and the UN/Data/GPT_Counts/MesquitaGPT.csv        GPT labels 1981-1993
  Israel and the UN/Data/GPT_Counts/GPT_resolution_counts.csv  the paper's counts
  code/panel_tools.py                                       estimators (numpy only)

Outputs
  data/processed/19_counts_corrected.csv      corrected counts, variants A and B
  output/tables/19_count_changes.csv          every country-year whose count changes
  output/tables/19_table1_*.csv               Table 1, published data and corrected
  output/tables/19_table2_*.csv               Table 2, published data and corrected
  output/tables/19_table3_points.csv          Table 3 point estimates, all variants
  output/tables/19_table3_contributions.csv   explained part split by variable
  output/tables/19_boot_rounds.csv            one line per bootstrap round (appended by each chunk)
  output/tables/19_table3_bootstrap.csv       bootstrap SEs and failed rounds
  output/tables/19_table3_bootstrap_failures.csv   why rounds failed
  output/tables/19_f1_unwatch.csv             F1 and confusion matrix, both ways
  output/tables/19_run_log_*.txt              everything printed

Run from the project root:
    python3 code/19_corrected_tables.py                       (tables and point estimates, 1 minute)
    bash code/19_run_bootstrap.sh                             (all bootstrap chunks, about 20 minutes)
    python3 code/19_corrected_tables.py --stage summary       (collects the rounds into the SE table)
  A single chunk: python3 code/19_corrected_tables.py --stage boot --data corrected_A --spec cor_re --seeds 0:250

Requires: python 3.10+, numpy, pandas. No network access, no other packages.
"""

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from panel_tools import fixed_effects, random_effects, between_effects, ols, _design  # noqa: E402

DATA = ROOT / "Israel and the UN" / "Data"
OUT = ROOT / "output" / "tables"
PROC = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)
PROC.mkdir(parents=True, exist_ok=True)

LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(str(s))


# Variables. V7 is the published decomposition set; V8 adds the Security Council.
V7 = ["battle_fat", "TerritoryOccupied", "human_rights_score", "western_affiliate",
      "NuclearPowerOutsideNPT", "log_gdppc", "log_population"]
V8 = V7 + ["PermanentUNSCMember"]
LABEL = {"battle_fat": "War fatalities", "TerritoryOccupied": "Occupying country",
         "human_rights_score": "Human rights score", "western_affiliate": "US military ally",
         "NuclearPowerOutsideNPT": "Nuclear outside NPT", "log_gdppc": "GDP per capita (log)",
         "log_population": "Population (log)", "PermanentUNSCMember": "Permanent SC member"}

# ---------------------------------------------------------------------------
# Part 0. Counts: rebuild the paper's resolution-country list and apply C1, C2
# ---------------------------------------------------------------------------
NAME_FIX = {  # from Annual_Country_Resolutions_MV.ipynb (the notebook that made the counts)
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
# counts-file name -> final_data name (from 01_make_variables_8123.do)
STATA_FIX = {"Bosnia and Herzegovina": "Bosnia Herzegovenia", "Côte d'Ivoire": "Cote d'Ivoire",
             "Democratic People's Republic of Korea": "Korea, Democratic People's Republic of",
             "Democratic Republic of the Congo": "Congo, Democratic Republic of",
             "Estonia and Latvia": "Estonia", "Libyan Arab Jamahiriya": "Libya",
             "United States": "United States of America"}
SESSION48_DUPES = ["R/48/27B", "R/48/258B"]


def rebuild_counts():
    v = pd.read_csv(DATA / "full_text_data2.csv", usecols=["unres", "year", "country"])
    v = v.rename(columns={"unres": "res_id"}).assign(source="Voelsen")
    m = pd.read_csv(DATA / "GPT_Counts" / "MesquitaGPT.csv", usecols=["res_id2", "session_reg", "country"])
    m = (m.rename(columns={"res_id2": "res_id"}).assign(year=1945 + m["session_reg"], source="Mesquita")
          .drop(columns="session_reg"))
    df = pd.concat([v, m], ignore_index=True)
    df["country"] = df["country"].fillna("None").str.strip()
    df = df[~df["country"].str.contains("Understood", case=False, na=False)]
    df["multi"] = df["country"].str.contains(", ")
    df["country"] = df["country"].str.split(", ")
    df = df.explode("country")
    df["country"] = df["country"].str.strip().replace(NAME_FIX).replace("None", np.nan)
    crit = df[df["country"].notna()].copy()
    crit["year"] = crit["year"].astype(int)
    crit = crit[(crit.year >= 1981) & (crit.year <= 2023)]

    # The paper's counts: reproduce the notebook's drop of 1981-1993 multi-country answers.
    published = crit[~(crit.multi & (crit.source == "Mesquita"))]
    chk = pd.read_csv(DATA / "GPT_Counts" / "GPT_resolution_counts.csv")
    chk["year"] = chk["year"].astype(int)
    chk = chk[(chk.year >= 1981) & (chk.year <= 2023)]
    mine = published.groupby(["country", "year"]).size().rename("n").reset_index()
    cmp_ = chk.merge(mine, on=["country", "year"], how="outer").fillna(0)
    say(f"Check: rebuilt published counts differ from GPT_resolution_counts.csv in "
        f"{int((cmp_['count'] != cmp_['n']).sum())} country-years (should be 0).")

    # C2: session-48 duplicates in the Voelsen 1994 rows
    dup = crit[(crit.source == "Voelsen") & crit.res_id.isin(SESSION48_DUPES)]
    say(f"C2: session-48 resolutions counted in both corpora: {len(dup)} rows "
        f"({', '.join(dup.res_id + ' ' + dup.country)}). The 1994 copies are dropped.")
    crit = crit[~((crit.source == "Voelsen") & crit.res_id.isin(SESSION48_DUPES))]

    # C1 variant A: split all multi-country answers (the 1994-2023 rule applied to 1981-1993)
    A = crit.groupby(["country", "year"]).size().rename("count_A")
    # C1 variant B: as A, but "Namibia" in a "South Africa, Namibia" answer is not a criticised state
    namibia_victim = crit.multi & (crit.source == "Mesquita") & (crit.country == "Namibia")
    B = crit[~namibia_victim].groupby(["country", "year"]).size().rename("count_B")
    P = published.groupby(["country", "year"]).size().rename("count_published")
    allc = pd.concat([P, A, B], axis=1).fillna(0).astype(int).reset_index()
    allc.to_csv(PROC / "19_counts_corrected.csv", index=False)
    ch = allc[(allc.count_A != allc.count_published) | (allc.count_B != allc.count_published)]
    ch.to_csv(OUT / "19_count_changes.csv", index=False)
    say(f"C1: multi-country answers in 1981-1993: {int(crit[crit.multi & (crit.source == 'Mesquita')].res_id.nunique())} "
        f"resolutions. Country-years whose count changes under A: {int((allc.count_A != allc.count_published).sum())}, "
        f"under B: {int((allc.count_B != allc.count_published).sum())}.")
    say(ch.to_string(index=False))
    return allc


def build_panels(allc):
    d = pd.read_stata(DATA / "final_data.dta")
    d["cid"] = d["country_num"].cat.codes
    d["battle_fat_k"] = d["battle_fat"] / 1000
    panels = {"published": d}
    c = allc.copy()
    c["country"] = c["country"].replace(STATA_FIX)
    for var in ["A", "B"]:
        p = d.merge(c[["country", "year", f"count_{var}"]], on=["country", "year"], how="left")
        p[f"count_{var}"] = p[f"count_{var}"].fillna(0)
        # rows whose published count differs from the counts file are name mismatches; keep published
        p["count_new"] = np.where(p[f"count_{var}"] >= 0, p[f"count_{var}"], p["count"])
        # sanity: the published count should be recovered where nothing changed
        lost = c[(c.country.isin(d.country.unique()) == False) & (c[f"count_{var}"] != c.count_published)]
        if len(lost):
            say(f"WARNING variant {var}: changed counts for names not in final_data: {lost.country.unique().tolist()}")
        p["count"] = p["count_new"]
        p["total_count"] = p.groupby("year")["count"].transform("sum")
        p["share"] = p["count"] / p["total_count"]
        panels[f"corrected_{var}"] = p.drop(columns=[f"count_{var}", "count_new"])
    # report the Israel effect
    for k, p in panels.items():
        isr = p[p.Israel_dummy == 1]
        say(f"{k}: Israel mean count {isr['count'].mean():.2f}, mean share {isr['share'].mean():.4f}; "
            f"others mean share {p[p.Israel_dummy == 0]['share'].mean():.4f}")
    return panels


# ---------------------------------------------------------------------------
# Part 1. Table 1
# ---------------------------------------------------------------------------
T1 = [("count", "Critical UNGA resolutions"), ("share", "Share of UNGA criticism"),
      ("total_intensity", "War intensity"), ("battle_fat", "War fatalities"),
      ("TerritoryOccupied", "Occupying country"), ("human_rights_score", "Human rights score"),
      ("NuclearPowerOutsideNPT", "Nuclear outside treaty"), ("OECDMember", "OECD member"),
      ("western_affiliate", "US military ally"), ("population", "Population (1000s)"),
      ("gdp_pc", "GDP per capita (current $)")]  # the same columns as 02_sumstat.do


def table1(p, tag):
    """Means on the rows with no missing Table 1 variable (7,721 others, 43 Israel),
    which is what Stata's `mean` command did for the published table."""
    cols = {}
    p = p.dropna(subset=[v for v, _ in T1])
    for grp, name in [(1, "Israel"), (0, "Other countries")]:
        s = p[p.Israel_dummy == grp]
        cols[name + " mean"] = [s[v].mean() if v in s else np.nan for v, _ in T1]
        cols[name + " sd"] = [s[v].std() if v in s else np.nan for v, _ in T1]
    t = pd.DataFrame(cols, index=[lab for _, lab in T1])
    nonmiss = p.dropna(subset=V8 + ["share"])
    t.loc["Observations"] = [int((nonmiss.Israel_dummy == 1).sum()), np.nan, int((nonmiss.Israel_dummy == 0).sum()), np.nan]
    t.round(3).to_csv(OUT / f"19_table1_{tag}.csv")
    say(f"\n=== Table 1 ({tag}) ===")
    say(t.round(3).to_string())


# ---------------------------------------------------------------------------
# Part 2. Table 2
# ---------------------------------------------------------------------------
def table2(p, tag):
    X = ["battle_fat_k", "TerritoryOccupied", "human_rights_score", "western_affiliate",
         "NuclearPowerOutsideNPT", "PermanentUNSCMember", "log_population", "log_gdppc"]
    o = p[p.Israel_dummy == 0].dropna(subset=X + ["share"])
    pos = o[(o.share > 0) & (o.share < 1)]
    isr = p[p.Israel_dummy == 1].dropna(subset=["battle_fat_k", "human_rights_score", "share"])
    fits = {
        "RE (ex. Israel)": random_effects(o, "share", X, "cid", time="year"),
        "FE (ex. Israel)": fixed_effects(o, "share", [x for x in X if x != "PermanentUNSCMember"], "cid", time="year"),
        "BE (ex. Israel)": between_effects(o, "share", X, "cid"),
        "RE (positive share)": random_effects(pos, "share", X, "cid", time="year"),
    }
    Xi, ni = _design(isr, ["battle_fat_k", "human_rights_score"])
    fits["Israel only"] = ols(isr["share"].values, Xi, ni, robust=True)
    rows = []
    for x in X:
        r1, r2 = {"variable": x}, {"variable": ""}
        for k, f in fits.items():
            if x in f.params.index:
                r1[k] = f"{f.params[x]:.4f}{f.stars()[x]}"
                r2[k] = f"({f.se[x]:.4f})"
            else:
                r1[k] = r2[k] = ""
        rows += [r1, r2]
    rows.append({"variable": "Observations", **{k: f.nobs for k, f in fits.items()}})
    rows.append({"variable": "R-squared (FE within / BE / Israel)",
                 "FE (ex. Israel)": f"{fits['FE (ex. Israel)'].extra['r2_within']:.3f}",
                 "BE (ex. Israel)": f"{fits['BE (ex. Israel)'].extra['r2']:.3f}"})
    t = pd.DataFrame(rows).fillna("")
    t.to_csv(OUT / f"19_table2_{tag}.csv", index=False)
    say(f"\n=== Table 2 ({tag}) ===")
    say(t.to_string(index=False))


# ---------------------------------------------------------------------------
# Part 3. Table 3: point estimates and contributions
# ---------------------------------------------------------------------------
def positive_sample(p, notebook_filter=False):
    """Row 3 sample. The paper says row 3 uses the coefficients of Table 2 column 4,
    whose sample is every country-year with a positive share (543 rows, 61 countries).
    Yair's notebook additionally dropped countries with only one positive year
    (527 rows, 45 countries); that filter is kept only for the reproduction rows."""
    pos = p[p.share > 0]
    if notebook_filter:
        pos = pos.groupby("cid").filter(lambda g: len(g) >= 2)
    return pos


def fit_model(D, model, V):
    """Fit the others-only regression. Returns params or raises if a variable cannot be estimated."""
    A = D[D.Israel_dummy == 0].dropna(subset=V + ["share"])
    if model == "fe":
        Vfe = [v for v in V if v != "PermanentUNSCMember"]
        w = A[Vfe] - A.groupby("cid")[Vfe].transform("mean")
        bad = [v for v in Vfe if (w[v].abs() < 1e-12).all()]
        if bad:
            raise ValueError("no within variation: " + ",".join(bad))
        f = fixed_effects(A, "share", Vfe, "cid", time="year", cluster=False)
        b = f.params.reindex(V).fillna(0.0)  # PSC absorbed: contributes nothing
        return b
    bad = [v for v in V if A[v].nunique() <= 1]
    if bad:
        raise ValueError("no variation: " + ",".join(bad))
    f = random_effects(A, "share", V, "cid", time=("year" if model == "re_year" else None), cluster=False)
    return f.params.reindex(V)


def explained(D, model, V):
    A = D[D.Israel_dummy == 0]
    B = D[D.Israel_dummy == 1]
    b = fit_model(D, model, V)
    contrib = (A[V].mean() - B[V].mean()) * b
    return float(contrib.sum()), float(A.share.mean() - B.share.mean()), contrib


SPECS = [
    # key, sample, model, variables, label
    ("pub_re", "all", "re", V7, "Row 1 as published: RE, no year effects, 7 variables"),
    ("pub_fe", "all", "fe", V7, "Row 2 as published: FE with year effects, 7 variables"),
    ("pub_row3_fe", "pos_nb", "fe", V7, "Row 3 as published (computed with FE, notebook sample), 7 variables"),
    ("cor_re", "all", "re_year", V8, "Row 1 corrected: RE with year effects + Security Council"),
    ("cor_fe", "all", "fe", V8, "Row 2 corrected: FE with year effects (Security Council absorbed)"),
    ("cor_row3_re", "pos", "re_year", V8, "Row 3 corrected: RE with year effects + Security Council (Table 2 col. 4 sample)"),
    ("cor_row3_fe", "pos", "fe", V8, "Row 3 corrected: FE with year effects (Table 2 col. 4 sample)"),
]


def samples_of(p):
    return {"all": p, "pos": positive_sample(p), "pos_nb": positive_sample(p, notebook_filter=True)}


def table3_points(panels):
    rows, crows = [], []
    for pk, p in panels.items():
        samples = samples_of(p)
        for key, samp, model, V, label in SPECS:
            e, o, c = explained(samples[samp], model, V)
            rows.append(dict(data=pk, spec=key, label=label, explained=round(e, 4), overall=round(o, 4),
                             explained_pct=round(100 * e / o, 1)))
            crows.append(dict(data=pk, spec=key, **{LABEL[v]: round(c[v], 4) for v in V}, total=round(e, 4)))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "19_table3_points.csv", index=False)
    c = pd.DataFrame(crows).fillna("")
    c.to_csv(OUT / "19_table3_contributions.csv", index=False)
    say("\n=== Table 3 point estimates (all data versions and specifications) ===")
    say(t.to_string(index=False))
    say("\n=== Explained part split by variable ===")
    say(c.to_string(index=False))


# ---------------------------------------------------------------------------
# Part 4. Bootstrap: countries resampled with replacement, Israel held fixed
# ---------------------------------------------------------------------------
def draw_countries(D, rng, groups):
    cs = list(groups.keys())
    C = rng.choice(cs, len(cs))
    parts, k = [], 0
    for c in C:
        part = D.iloc[groups[c]].copy()
        part["cid"] = k  # a country drawn twice becomes two separate units
        parts.append(part)
        k += 1
    isr = D[D.Israel_dummy == 1].copy()
    isr["cid"] = k
    return pd.concat(parts + [isr], ignore_index=True)


def bootstrap_rounds(panels, data_key, spec_key, seeds):
    """Run the given seeds for one data version and one specification and append
    one line per round to output/tables/19_boot_rounds.csv (value, or the reason
    the round could not be estimated). Split into chunks because the sandbox
    used for this card stops any command after three minutes."""
    p = panels[data_key]
    samples = samples_of(p)
    key, samp, model, V, label = [x for x in SPECS if x[0] == spec_key][0]
    D = samples[samp].reset_index(drop=True)
    israel_cid = int(D.loc[D.Israel_dummy == 1, "cid"].iloc[0])
    groups = {c: idx for c, idx in D.groupby("cid").indices.items() if c != israel_cid}
    path = OUT / "19_boot_rounds.csv"
    rows = []
    for s in seeds:
        rng = np.random.default_rng(s)
        S = draw_countries(D, rng, groups)
        try:
            rows.append(dict(data=data_key, spec=spec_key, seed=s, explained=explained(S, model, V)[0], failed=""))
        except Exception as ex:
            rows.append(dict(data=data_key, spec=spec_key, seed=s, explained=np.nan, failed=str(ex)))
    out = pd.DataFrame(rows)
    out.to_csv(path, mode="a", header=not path.exists(), index=False)
    say(f"{data_key} {spec_key}: seeds {seeds[0]}-{seeds[-1]}, ok {out.explained.notna().sum()}, failed {out.explained.isna().sum()}")


def bootstrap_summary(panels):
    """Collect every round in 19_boot_rounds.csv into the Table 3 bootstrap table."""
    r = pd.read_csv(OUT / "19_boot_rounds.csv").drop_duplicates(["data", "spec", "seed"])
    res, fails = [], []
    for (dk, sk), g in r.groupby(["data", "spec"]):
        key, samp, model, V, label = [x for x in SPECS if x[0] == sk][0]
        pt, od, _ = explained(samples_of(panels[dk])[samp], model, V)
        ok = g.explained.dropna().values
        se = ok.std(ddof=1)
        res.append(dict(data=dk, spec=sk, label=label, explained=round(pt, 4), overall=round(od, 4),
                        boot_se=round(se, 4), ratio_to_se=round(pt / se, 2),
                        pct2_5=round(np.percentile(ok, 2.5), 4), pct97_5=round(np.percentile(ok, 97.5), 4),
                        rounds_ok=len(ok), rounds_failed=int(g.explained.isna().sum())))
        for reason, n in Counter(g.failed.dropna()).items():
            if reason:
                fails.append(dict(data=dk, spec=sk, reason=reason, rounds=n))
    t = pd.DataFrame(res)
    t.to_csv(OUT / "19_table3_bootstrap.csv", index=False)
    pd.DataFrame(fails).to_csv(OUT / "19_table3_bootstrap_failures.csv", index=False)
    say("\n=== Bootstrap (500 rounds, countries resampled with replacement, Israel fixed) ===")
    say(t.drop(columns="label").to_string(index=False))
    if fails:
        say("\nWhy rounds failed:")
        say(pd.DataFrame(fails).to_string(index=False))


# ---------------------------------------------------------------------------
# Part 5. F1 against UN Watch, with and without the Condemnatory flag
# ---------------------------------------------------------------------------
def f1_check():
    df = pd.read_csv(DATA / "full_text_data2.csv", usecols=["unres", "year", "country", "Country Concerned", "Condemnatory"])
    df["country"] = df["country"].fillna("None").str.strip()
    df["Country Concerned"] = df["Country Concerned"].fillna("None").str.strip()
    df = df[~df["country"].str.contains("Understood", case=False, na=False)]
    # the F1 notebook's name fixes
    df["country"] = df["country"].replace({"Russian Federation": "Russia", "Islamic Republic of Iran": "Iran",
                                           "Republic of Korea": "South Korea",
                                           "Democratic People's Republic of Korea": "North Korea",
                                           "Republic of Moldova": "Moldova", "United States of America": "United States",
                                           "Syrian Arab Republic": "Syria",
                                           "United Kingdom of Great Britain and Northern Ireland": "United Kingdom"})
    df["cond"] = df["Condemnatory"].fillna("").str.strip().str.lower()
    rows = []
    for win_label, win in [("2015-2024 as in the notebook", df.year >= 2015), ("2015-2023 as the paper states", (df.year >= 2015) & (df.year <= 2023))]:
        f = df[win]
        pred = (f.country == "Israel").astype(int)
        for truth_label, truth in [("UN Watch Country Concerned only (published)", (f["Country Concerned"] == "Israel").astype(int)),
                                   ("UN Watch Country Concerned AND Condemnatory = Yes", ((f["Country Concerned"] == "Israel") & (f.cond == "yes")).astype(int))]:
            tp = int(((pred == 1) & (truth == 1)).sum()); fp = int(((pred == 1) & (truth == 0)).sum())
            fn = int(((pred == 0) & (truth == 1)).sum()); tn = int(((pred == 0) & (truth == 0)).sum())
            prec = tp / (tp + fp); rec = tp / (tp + fn); f1 = 2 * prec * rec / (prec + rec)
            rows.append(dict(window=win_label, truth=truth_label, resolutions=len(f), unwatch_israel=int(truth.sum()),
                             TP=tp, FP=fp, FN=fn, TN=tn, precision=round(prec, 3), recall=round(rec, 3), F1=round(f1, 3)))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "19_f1_unwatch.csv", index=False)
    say("\n=== F1 against UN Watch ===")
    say(t.to_string(index=False))
    f = df[df.year >= 2015]
    flagged = f[(f["Country Concerned"] == "Israel") & (f.cond != "yes")]
    say(f"UN Watch rows with Country Concerned = Israel but Condemnatory not 'Yes': {len(flagged)}; "
        f"GPT says Israel in {int((flagged.country == 'Israel').sum())} of them.")
    say(flagged[["unres", "year", "country", "Condemnatory"]].to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="tables", choices=["tables", "boot", "summary"])
    ap.add_argument("--data", default="published")
    ap.add_argument("--spec", default="cor_re")
    ap.add_argument("--seeds", default="0:500", help="seed range a:b for one bootstrap chunk")
    a = ap.parse_args()
    say("Card 19: corrected Tables 1-3. Run " + time.strftime("%Y-%m-%d %H:%M"))
    allc = rebuild_counts()
    panels = build_panels(allc)
    if a.stage == "tables":
        for k, p in panels.items():
            table1(p, k)
        for k, p in panels.items():
            table2(p, k)
        table3_points(panels)
        f1_check()
        (OUT / "19_run_log_tables.txt").write_text("\n".join(LOG))
    elif a.stage == "boot":
        lo, hi = [int(x) for x in a.seeds.split(":")]
        bootstrap_rounds(panels, a.data, a.spec, list(range(lo, hi)))
    else:
        bootstrap_summary(panels)
        (OUT / "19_run_log_bootstrap.txt").write_text("\n".join(LOG))
