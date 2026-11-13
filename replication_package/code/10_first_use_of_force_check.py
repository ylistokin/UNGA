"""
10_first_use_of_force_check.py

Task card 10: a read-only look at a "first use of force" variable built from
the Militarized Interstate Disputes data, following the rules in
  Davis Brown, "Measuring first use of force: methods, results, and
  implications" (2019) 6(1) Journal on the Use of Force and International
  Law 83-112, doi 10.1080/20531702.2019.1575124.
Purpose: see whether it can serve as "data point 1" for Referee 2's Article
2(4) category (other states that used force first). Nothing enters the model.
Matanel decided (24 September 2026) that partial coverage, 1981-2014, is
acceptable.

Input (read-only, read directly from the zip, nothing extracted):
  data/raw/MID-5-Data-and-Supporting-Materials.zip
      Correlates of War, Militarized Interstate Disputes v5.0,
      https://correlatesofwar.org/data-sets/mids/, downloaded by Matanel on
      24 September 2026. SHA-256 in 10_run_log.txt.
      File used: "MIDB 5.0.csv" (one row per state per dispute).
      Columns used: dispnum, stabb, ccode, styear, endyear, sidea, orig, hiact.
  Israel and the UN/Data/final_data.dta (columns country, year, cow, count, share)

Brown's rules as applied here (section 3 of the article):
  - "Use of force" = highest action (hiact) of 12 (border violation) or more.
  - Directed pairs: every Side A state with every Side B state in the same
    dispute, when their participation overlaps in time. A pair starts in the
    later of the two states' start years (a joiner enters in the year it joins).
  - State 1 is a first user of force against state 2, in the pair's start year, if
      (1) state 1 is on Side A and its hiact is 12 or more, or
      (2) state 1 is on Side B, state 2 (Side A) only threatened or showed
          force (hiact 11 or less), and state 1's hiact is 12 or more.
  - Severity thresholds: "low" = the pair's higher hiact is 12 or more (Brown's
    init12); "high" = 16 (attack) or more (Brown's init16). Both are shown.
  - Disputes Brown excludes as authorised by the Security Council (his note 40):
    3551, 4016, 4264, 4265, 4283. Dispute 4265 does not exist in MID 5.0; this
    is reported, not fixed.
  - Iraq 2003 (dispute 4273). Brown corrects this dispute by hand so that each
    state's responsibility for the 2003 invasion is recorded correctly, but the
    article does not give his corrected values. Matanel decided (24 September
    2026) to correct it as the author did. Implemented here as follows, which
    is our reading of Brown's description, not his exact values: within
    dispute 4273 the only first uses of force kept are the invading states
    (United States, United Kingdom, Australia, Poland, each with highest action
    20, "begin interstate war") against Iraq, all dated 2003. Dropped: the
    United States and United Kingdom dated 1997 (the dispute's start year) and
    Iraq as a first user against the states on the other side (1997-2003),
    which arises only because Iraq's highest action, 20, comes from defending
    against the 2003 invasion. Saudi Arabia 2000 (its own highest action 15,
    seizure) is kept. The unmodified result is printed first for comparison.
  - hiact is each state's highest action over the whole dispute, as in the
    MID participant file; this is how the EUGene tool Brown cites builds pairs.

Nothing here is a coding decision for the paper. How to turn pairs into one
number per country-year is left open; two readings are printed side by side.

Outputs: output/tables/10_*, data/processed/10_first_use_country_year.csv
Run from the project root:  python3 code/10_first_use_of_force_check.py
Requires: python 3.10+, pandas. No network access.
"""

import hashlib
import zipfile
import pandas as pd

ZIP = "data/raw/MID-5-Data-and-Supporting-Materials.zip"
FINAL = "Israel and the UN/Data/final_data.dta"
OUT = "output/tables/"
Y0, Y1 = 1981, 2014
UNSC = [3551, 4016, 4264, 4265, 4283]

log = []
def say(s=""):
    print(s)
    log.append(str(s))

say(f"MID zip SHA-256 {hashlib.sha256(open(ZIP, 'rb').read()).hexdigest()}")
b = pd.read_csv(zipfile.ZipFile(ZIP).open("MIDB 5.0.csv"))
b = b[["dispnum", "stabb", "ccode", "styear", "endyear", "sidea", "orig", "hiact"]]
say(f"MIDB rows {len(b)}, disputes {b.dispnum.nunique()}, years {b.styear.min()}-{b.endyear.max()}")

missing = [d for d in UNSC if d not in set(b.dispnum)]
say(f"Security Council authorised disputes excluded (Brown note 40): {UNSC}; not found in MID 5.0: {missing}")
b = b[~b.dispnum.isin(UNSC)]

# ---- directed pairs ----------------------------------------------------------
A = b[b.sidea == 1]
B = b[b.sidea == 0]
pairs = []
for s1, s2 in [(A, B), (B, A)]:
    m = s1.merge(s2, on="dispnum", suffixes=("_1", "_2"))
    m = m[(m.styear_1 <= m.endyear_2) & (m.styear_2 <= m.endyear_1)]
    pairs.append(m)
p = pd.concat(pairs, ignore_index=True)
p["year"] = p[["styear_1", "styear_2"]].max(axis=1)
p["pair_hiact"] = p[["hiact_1", "hiact_2"]].max(axis=1)
cond1 = (p.sidea_1 == 1) & (p.hiact_1 >= 12)
cond2 = (p.sidea_1 == 0) & (p.hiact_2 <= 11) & (p.hiact_1 >= 12)
p["first_use"] = (cond1 | cond2).astype(int)
p["first_use_low"] = p.first_use * (p.pair_hiact >= 12)
p["first_use_high"] = p.first_use * (p.pair_hiact >= 16)
fu = p[(p.first_use == 1) & p.year.between(Y0, Y1)].copy()
say(f"\nDirected pairs with a first use of force, {Y0}-{Y1}: {len(fu)} "
    f"(high threshold: {int(fu.first_use_high.sum())})")

# ---- Iraq 2003 (dispute 4273): unmodified, then corrected ---------------------
say("\nDispute 4273 (Iraq, 1997-2003), unmodified rule:")
say(fu[fu.dispnum == 4273][["year", "stabb_1", "stabb_2", "hiact_1", "hiact_2"]].to_string(index=False))
INVADERS = ["USA", "UKG", "AUL", "POL"]
k = fu.dispnum == 4273
inv = k & fu.stabb_1.isin(INVADERS) & (fu.stabb_2 == "IRQ")
keep_saudi = k & (fu.stabb_1 == "SAU")
fu.loc[inv, "year"] = 2003
fu = fu[~k | inv | keep_saudi]
fu[["dispnum", "year", "stabb_1", "ccode_1", "stabb_2", "ccode_2", "sidea_1", "orig_1",
    "hiact_1", "hiact_2", "first_use_low", "first_use_high"]].sort_values(["year", "stabb_1"]) \
    .to_csv(OUT + "10_first_use_pairs.csv", index=False)
say("Dispute 4273 after the correction:")
say(fu[fu.dispnum == 4273][["year", "stabb_1", "stabb_2", "hiact_1", "hiact_2"]].to_string(index=False))

# ---- country-year, two readings ----------------------------------------------
cy = (fu.groupby(["ccode_1", "year"])
        .agg(fu_low_any=("first_use_low", "max"), fu_high_any=("first_use_high", "max"),
             fu_low_targets=("ccode_2", lambda s: s[fu.loc[s.index, "first_use_low"] == 1].nunique()),
             targets=("stabb_2", lambda s: ",".join(sorted(set(s)))))
        .reset_index().rename(columns={"ccode_1": "cow"}))

d = pd.read_stata(FINAL, convert_categoricals=False)[["country", "year", "cow", "count", "share"]]
d = d[d.year.between(Y0, Y1)]
panel = d.merge(cy, on=["cow", "year"], how="left")
for c in ["fu_low_any", "fu_high_any", "fu_low_targets"]:
    panel[c] = panel[c].fillna(0).astype(int)
unmatched = cy.merge(d[["cow", "year"]], on=["cow", "year"], how="left", indicator=True)
unmatched = unmatched[unmatched._merge == "left_only"]
say(f"\nFirst-use country-years not matched to the paper's panel: {len(unmatched)}")
if len(unmatched):
    say(unmatched[["cow", "year", "targets"]].to_string(index=False))
panel.to_csv(OUT + "10_country_year_first_use.csv", index=False)
# the file the decomposition (code/11_first_use_in_decomposition.py) reads
panel[["country", "year", "cow", "fu_low_any", "fu_high_any", "fu_low_targets"]] \
    .to_csv("data/processed/10_first_use_country_year.csv", index=False)

# ---- Israel and South Africa -------------------------------------------------
for name in ["Israel", "South Africa"]:
    x = panel[(panel.country == name) & (panel.fu_low_any == 1)]
    say(f"\n{name}: first use of force (low threshold) in {len(x)} of "
        f"{int((panel.country == name).sum())} years {Y0}-{Y1}; high threshold in "
        f"{int(panel.loc[panel.country == name, 'fu_high_any'].sum())} years")
    say(x[["year", "fu_low_any", "fu_high_any", "targets", "count", "share"]].to_string(index=False))

# ---- Israel: years with and without a first use of force --------------------
isr_all = panel[panel.country == "Israel"]
for thr in ["low", "high"]:
    g = isr_all.groupby(f"fu_{thr}_any").agg(years=("year", "size"), mean_count=("count", "mean"),
                                             mean_share=("share", "mean"))
    say(f"\nIsrael {Y0}-{Y1}, threshold {thr}: criticism in years with (1) and without (0) a first use of force")
    say(g.round(3).to_string())

# ---- everyone else -------------------------------------------------------------
for thr in ["low", "high"]:
    col = f"fu_{thr}_any"
    o = panel[(panel[col] == 1) & (panel.country != "Israel")]
    isr = panel[(panel[col] == 1) & (panel.country == "Israel")]
    say(f"\n=== Threshold {thr} ({'hiact 12+' if thr == 'low' else 'hiact 16+'}) ===")
    say(f"Other countries with a first use of force in at least one year: {o.country.nunique()}; "
        f"country-years: {len(o)}; with zero critical resolutions: {int((o['count'] == 0).sum())} "
        f"({(o['count'] == 0).mean():.0%})")
    say(f"Mean share of criticism, other first-use country-years: {o.share.mean():.4f} "
        f"(median {o.share.median():.4f}); Israel's first-use years: {isr.share.mean():.3f}")
    pc = (o.groupby("country").agg(first_use_years=("year", "size"),
                                   years_criticised=("count", lambda s: int((s > 0).sum())),
                                   critical_resolutions=("count", "sum"),
                                   mean_share=("share", "mean"))
            .sort_values(["first_use_years", "critical_resolutions"], ascending=False))
    pc.to_csv(OUT + f"10_first_users_{thr}.csv")
    say("Most frequent first users (top 20):")
    say(pc.head(20).round(3).to_string())
    say("Most criticised first users (top 10):")
    say(pc.sort_values("critical_resolutions", ascending=False).head(10).round(3).to_string())

open(OUT + "10_run_log.txt", "w").write("\n".join(log) + "\n")
