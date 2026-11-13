"""
21_referee1_controls_and_democracy.py

Task card 21 (28 October 2026). Two referee requests that share the same data:

Referee 1
  (a) Add country-level controls: political institutions (democracy),
      globalisation (KOF index), government ideology (DPI execrlc) and the
      share of the predominant religion (Muslim-majority in particular).
  (b) Heterogeneity: OECD against non-OECD countries.
Referee 2
  (c) Are democracies held to a higher standard?

Democracy measure (Matanel's decision, 28 October 2026): V-Dem is the main
measure, because it covers the whole period and is the measure already used
in the project; the Gründler and Krieger machine learning index that Referee
1 named is run as a robustness check (it ends in 2019).

New variables and their sources (rule 7)
  libdem      V-Dem v16 (Varieties of Democracy, vdemdata R package 16.0,
              GitHub commit f4dd269, downloaded by Matanel 24 Sep 2026 as
              data/raw/vdem_v16.RData, SHA-256 39b412d3...95ea), column
              v2x_libdem (liberal democracy index, 0 to 1). Also v2x_polyarchy
              and v2x_regime (Regimes of the World: 0 closed autocracy,
              1 electoral autocracy, 2 electoral democracy, 3 liberal
              democracy). Read with code/rdata_reader.py; the subset is saved
              as data/processed/21_vdem_v16_democracy_subset.csv and checked
              against the September pyreadr extract (identical).
  mldi        Gründler and Krieger, Machine Learning Democracy Index, updated
              version (186 countries, 1919-2019), file data/raw/ML indices.xlsx
              downloaded by Matanel 28 Oct 2026, column C_MLDI (continuous,
              0 to 1); D_MLDI is the dichotomous version.
  kofgi       KOF Globalisation Index 2025 (Gygli et al. 2019), file
              data/raw/KOFGI_2025_public.xlsx downloaded by Matanel 28 Oct 2026
              from kof.ethz.ch, column KOFGI (overall index, 0 to 100,
              1970-2023).
  exec_left, exec_right
              Database of Political Institutions 2023 (Inter-American
              Development Bank, DOI 10.60966/8tx1xs4i, published 11 Mar 2026),
              zip downloaded by Matanel 28 Oct 2026 and unzipped to
              data/raw/dpi2023/, file "DPI 2023 CSV Version.csv", column
              execrlc (orientation of the chief executive's party: Left,
              Center, Right, 0 = no information, -999 = missing). Coded as
              two dummies, exec_left and exec_right; Center and "no
              information" are the reference group; -999 is missing. The
              reference-group choice is mine (Claude) and can be changed.
              Matched by ISO3 code (column ifs).
  muslim_share, muslim_majority
              Correlates of War World Religion Data v1.1 (Maoz and Henderson
              2013), file data/raw/WRP_national.csv downloaded by Matanel
              28 Oct 2026, column islmgenpct (share of Muslims, all branches),
              observed every five years 1945-2010. Interpolated linearly
              between observation years and carried forward from 2010 to
              2011-2023 (Matanel's decision D4, 28 Oct 2026). Matched by COW
              code (column state). muslim_majority = 1 if the share exceeds
              one half.

Matching: V-Dem by COW code (with the three fixes memo 06 used: Germany
255 to 260, Yemen 679 to 678, Serbia 340 to 345); KOF and the ML index by
ISO3 code (final_data's CountryCode), with KOF's own codes for Congo DR
(ZAR), Romania (ROM) and Timor-Leste (TMP).

Base data: the corrected panel, count variant A (memo 19). Estimators:
code/panel_tools.py. Decomposition and bootstrap: as in code/19_*.

Outputs: output/tables/21_coverage.csv, 21_table2_controls.csv,
  21_decomp_controls.csv, 21_contributions_controls.csv, 21_oecd_split.csv,
  21_democracy_coefficients.csv, 21_democracy_decomp.csv,
  21_democracy_interactions.csv, 21_boot_rounds.csv, 21_bootstrap.csv,
  21_run_log_*.txt; data/processed/21_panel_with_controls.csv

Run from the project root:
    python3 code/21_referee1_controls_and_democracy.py              (point estimates)
    bash code/21_run_bootstrap.sh                                   (bootstrap chunks)
    python3 code/21_referee1_controls_and_democracy.py --stage summary

Requires: python 3.10+, numpy, pandas, openpyxl. No network access.
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
from panel_tools import random_effects, fixed_effects  # noqa: E402

RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
OUT = ROOT / "output" / "tables"
DPI_FILE = "dpi2023/DPI 2023 (CSV Version)/DPI 2023 CSV Version.csv"   # unzipped from the IDB zip
RELIGION_FILE = "WRP_national.csv"   # Correlates of War World Religion Data v1.1, national file

V8 = c19.V8
LABEL = dict(c19.LABEL, libdem="Liberal democracy (V-Dem)", mldi="Democracy (Gründler-Krieger ML)",
             kofgi="KOF globalisation", exec_left="Left executive (DPI)", exec_right="Right executive (DPI)",
             muslim_share="Muslim share (WRP)", muslim_majority="Muslim majority (WRP)")
LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(str(s))


# ---------------------------------------------------------------------------
# Build the panel with the new variables
# ---------------------------------------------------------------------------
def load_vdem():
    sub = PROC / "21_vdem_v16_democracy_subset.csv"
    if not sub.exists():
        from rdata_reader import read_rdata
        objs = read_rdata(RAW / "vdem_v16.RData",
                          keep=["country_name", "country_text_id", "COWcode", "year", "v2x_libdem", "v2x_polyarchy",
                                "v2x_regime", "v2clsocgrp", "v2clsocgrp_ord", "v2xpe_exlsocgr"])
        list(objs.values())[0].to_csv(sub, index=False)
    v = pd.read_csv(sub)
    v = v[(v.year >= 1981) & (v.year <= 2023) & v.COWcode.notna()]
    v = v.rename(columns={"COWcode": "cow_v", "v2x_libdem": "libdem", "v2x_polyarchy": "polyarchy", "v2x_regime": "regime"})
    v["cow_v"] = v["cow_v"].astype(int)
    v["year"] = v["year"].astype(int)
    return v[["cow_v", "year", "libdem", "polyarchy", "regime"]]


def build():
    allc = c19.rebuild_counts()
    p = c19.build_panels(allc)["corrected_A"].copy()
    p["year"] = p["year"].astype(int)
    # V-Dem and the World Religion Project both use COW codes, as final_data does
    # (Germany 260 to 1990 and 255 after; Yemen 678 to 1990 and 679 after), so the
    # codes are used as they are. Serbia: final_data uses 342/345, V-Dem and WRP 345.
    # Fixed 29 Oct 2026: the earlier version applied the Gleditsch-Ward recode used for
    # EPR (255->260, 679->678), which lost V-Dem values for Germany 1991-2023 and Yemen
    # 1991-2023 and gave them West German / North Yemeni religion shares (memo 25).
    p["cow_v"] = p["cow"].astype(int)
    p.loc[p["country"] == "Serbia", "cow_v"] = 345
    p = p.merge(load_vdem(), on=["cow_v", "year"], how="left")
    # Gründler-Krieger by ISO3
    ml = pd.read_excel(RAW / "ML indices.xlsx")[["ISO", "Year", "C_MLDI", "D_MLDI"]]
    ml = ml.rename(columns={"ISO": "CountryCode", "Year": "year", "C_MLDI": "mldi", "D_MLDI": "mldi_dich"})
    p = p.merge(ml, on=["CountryCode", "year"], how="left")
    # KOF by ISO3 with KOF's own codes
    k = pd.read_excel(RAW / "KOFGI_2025_public.xlsx")[["code", "year", "KOFGI"]].rename(columns={"KOFGI": "kofgi"})
    k["CountryCode"] = k["code"].replace({"ZAR": "COD", "ROM": "ROU", "TMP": "TLS"})
    p = p.merge(k[["CountryCode", "year", "kofgi"]], on=["CountryCode", "year"], how="left")
    # DPI 2023: executive party orientation
    dpi = pd.read_csv(RAW / DPI_FILE, low_memory=False, usecols=["countryname", "ifs", "year", "execrlc"])
    dpi["year"] = pd.to_datetime(dpi["year"]).dt.year
    dpi = dpi[dpi.ifs.astype(str).str.len() == 3].rename(columns={"ifs": "CountryCode"})
    dpi["execrlc"] = dpi["execrlc"].astype(str).str.strip()
    dpi = dpi[dpi.execrlc != "-999"]
    dpi["exec_left"] = (dpi.execrlc == "Left").astype(float)
    dpi["exec_right"] = (dpi.execrlc == "Right").astype(float)
    dpi = dpi.drop_duplicates(["CountryCode", "year"])
    p = p.merge(dpi[["CountryCode", "year", "exec_left", "exec_right", "execrlc"]], on=["CountryCode", "year"], how="left")
    # World Religion Data: Muslim share, five-year points interpolated and carried forward
    w = pd.read_csv(RAW / RELIGION_FILE, usecols=["year", "state", "islmgenpct"])
    grid = pd.MultiIndex.from_product([sorted(w.state.unique()), range(1981, 2024)], names=["cow_v", "year"]).to_frame(index=False)
    w = w.rename(columns={"state": "cow_v"})
    grid = grid.merge(w, on=["cow_v", "year"], how="left")
    grid["muslim_share"] = grid.groupby("cow_v")["islmgenpct"].transform(lambda s: s.interpolate(limit_direction="both"))
    grid["muslim_majority"] = (grid.muslim_share > 0.5).astype(float).where(grid.muslim_share.notna())
    p = p.merge(grid[["cow_v", "year", "muslim_share", "muslim_majority"]], on=["cow_v", "year"], how="left")
    p["democracy"] = (p["regime"] >= 2).astype(float).where(p["regime"].notna())
    p.to_csv(PROC / "21_panel_with_controls.csv", index=False)
    return p


def coverage(p):
    rows = []
    base = p.dropna(subset=V8 + ["share"])
    for v in ["libdem", "polyarchy", "regime", "mldi", "kofgi", "exec_left", "muslim_share"]:
        have = base[v].notna()
        miss = base[~have].groupby("country").size().sort_values(ascending=False)
        rows.append(dict(variable=v, rows_with_value=int(have.sum()), rows_missing=int((~have).sum()),
                         israel_rows=int(base[(base.Israel_dummy == 1) & have].shape[0]),
                         years=f"{int(base[have].year.min())}-{int(base[have].year.max())}",
                         countries_fully_missing=int((miss == miss.max()).sum()) if len(miss) else 0,
                         top_missing=", ".join(f"{c} ({n})" for c, n in miss.head(6).items())))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "21_coverage.csv", index=False)
    say("\n=== Coverage of the new variables on the 7,721-row regression sample ===")
    say(t.to_string(index=False))
    isr = p[p.Israel_dummy == 1]
    say(f"Israel: libdem mean {isr.libdem.mean():.3f}, regime values {sorted(isr.regime.dropna().unique())}, "
        f"mldi mean {isr.mldi.mean():.3f}, kofgi mean {isr.kofgi.mean():.1f}, exec_left {isr.exec_left.mean():.2f}, "
        f"exec_right {isr.exec_right.mean():.2f}, muslim share {isr.muslim_share.mean():.3f}")
    o = p[p.Israel_dummy == 0]
    say(f"Others: libdem mean {o.libdem.mean():.3f}, share democracies (regime>=2) {o.democracy.mean():.3f}, "
        f"mldi mean {o.mldi.mean():.3f}, kofgi mean {o.kofgi.mean():.1f}, exec_left {o.exec_left.mean():.2f}, "
        f"exec_right {o.exec_right.mean():.2f}, muslim share {o.muslim_share.mean():.3f}, muslim majority {o.muslim_majority.mean():.3f}")
    say(f"DPI execrlc values on the sample: {base.execrlc.value_counts(dropna=False).to_dict()}")


# ---------------------------------------------------------------------------
# Specifications
# ---------------------------------------------------------------------------
CONTROL_SETS = {
    "baseline": [],
    "+ liberal democracy (V-Dem)": ["libdem"],
    "+ KOF globalisation": ["kofgi"],
    "+ executive ideology (DPI)": ["exec_left", "exec_right"],
    "+ Muslim share (WRP)": ["muslim_share"],
    "+ Muslim majority (WRP)": ["muslim_majority"],
    "+ all four (V-Dem, KOF, DPI, Muslim share)": ["libdem", "kofgi", "exec_left", "exec_right", "muslim_share"],
    "+ democracy + KOF": ["libdem", "kofgi"],
    "+ Gründler-Krieger ML democracy (to 2019)": ["mldi"],
}


def table2_controls(p):
    rows = []
    for name, extra in CONTROL_SETS.items():
        V = V8 + extra
        o = p[p.Israel_dummy == 0].dropna(subset=V + ["share"])
        re = random_effects(o, "share", V, "cid", time="year")
        fe = fixed_effects(o, "share", [v for v in V if v != "PermanentUNSCMember"], "cid", time="year")
        for v in V:
            k = 1000 if v == "battle_fat" else 1  # show fatalities per 1,000 deaths, as in Table 2
            rows.append(dict(spec=name, variable=LABEL.get(v, v) + (" (1000s)" if k == 1000 else ""),
                             RE=f"{k * re.params[v]:.4f}{re.stars()[v]} ({k * re.se[v]:.4f})",
                             FE=(f"{k * fe.params[v]:.4f}{fe.stars()[v]} ({k * fe.se[v]:.4f})" if v in fe.params.index else "absorbed")))
        rows.append(dict(spec=name, variable="N", RE=re.nobs, FE=fe.nobs))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "21_table2_controls.csv", index=False)
    say("\n=== Table 2 with Referee 1 controls (RE and FE, year effects, clustered SE) ===")
    say(t.to_string(index=False))


def group_data(p, extra, mask):
    """The rows for one specification: Israel plus the others in the group, and the
    variable list after dropping any variable that never varies among those others
    (for example no OECD member has nuclear weapons outside the NPT). Dropped
    variables are reported; Israel's value on them then counts as unexplained."""
    V = V8 + extra
    D = p if mask is None else p[(p.Israel_dummy == 1) | mask(p)]
    D = D[D.Israel_dummy.eq(1) | D[V + ["share"]].notna().all(axis=1)]
    others = D[D.Israel_dummy == 0]
    dropped = {"re_year": [v for v in V if others[v].nunique() <= 1]}
    w = others[V] - others.groupby("cid")[V].transform("mean")
    dropped["fe"] = [v for v in V if (w[v].abs() < 1e-12).all() and v != "PermanentUNSCMember"]
    Vs = {m: [v for v in V if v not in d] for m, d in dropped.items()}
    return D, Vs, dropped


def decomp_table(p, sets, fname, label_col="spec", restrict=None):
    """Point estimates and contributions for a dict name -> (extra vars, others filter)."""
    rows, crows = [], []
    for name, (extra, mask) in sets.items():
        D, Vs, dropped_all = group_data(p, extra, mask)
        for model, mlabel in [("re_year", "RE"), ("fe", "FE")]:
            V, dropped = Vs[model], dropped_all[model]
            e, o, c = c19.explained(D, model, V)
            rows.append({label_col: name, "model": mlabel, "explained": round(e, 4), "overall": round(o, 4),
                         "explained_pct": round(100 * e / o, 1),
                         "n_other_rows": int((D.Israel_dummy == 0).sum()), "n_other_countries": int(D[D.Israel_dummy == 0].cid.nunique()),
                         "dropped_no_variation": ", ".join(LABEL.get(v, v) for v in dropped)})
            crows.append({label_col: name, "model": mlabel, **{LABEL.get(v, v): round(c[v], 4) for v in V}, "total": round(e, 4)})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / f"21_{fname}.csv", index=False)
    pd.DataFrame(crows).fillna("").to_csv(OUT / f"21_contributions_{fname}.csv", index=False)
    say(f"\n=== Decomposition: {fname} ===")
    say(t.to_string(index=False))
    say(pd.DataFrame(crows).fillna("").to_string(index=False))
    return t


def controls_sets():
    return {k: (v, None) for k, v in CONTROL_SETS.items()}


def oecd_sets():
    return {"others: OECD members (that year)": ([], lambda d: d.OECDMember == 1),
            "others: non-OECD (that year)": ([], lambda d: d.OECDMember == 0),
            "others: OECD, + democracy + KOF": (["libdem", "kofgi"], lambda d: d.OECDMember == 1),
            "others: non-OECD, + democracy + KOF": (["libdem", "kofgi"], lambda d: d.OECDMember == 0)}


def democracy_sets():
    return {"others: democracies (V-Dem regime 2-3)": ([], lambda d: d.democracy == 1),
            "others: non-democracies (V-Dem regime 0-1)": ([], lambda d: d.democracy == 0),
            "others: democracies, + KOF": (["kofgi"], lambda d: d.democracy == 1),
            "others: non-democracies, + KOF": (["kofgi"], lambda d: d.democracy == 0)}


def democracy_coefficients(p):
    """Are the behaviour variables punished more among democracies? Two ways: separate
    regressions by regime group, and a pooled model with democracy interactions."""
    rows = []
    for name, mask in [("democracies", p.democracy == 1), ("non-democracies", p.democracy == 0), ("all others", p.democracy.notna())]:
        o = p[(p.Israel_dummy == 0) & mask].dropna(subset=V8 + ["share"])
        re = random_effects(o, "share", V8, "cid", time="year")
        fe = fixed_effects(o, "share", [v for v in V8 if v != "PermanentUNSCMember"], "cid", time="year")
        # A variable that never changes within any country in this group cannot be estimated
        # by fixed effects (the estimate is 0 by construction and its standard error is a
        # rounding artefact: 0.0000 on one machine, nan on another). Added 29 Oct 2026 after
        # the external re-run (memo 25); it changes only how such a cell is labelled.
        no_within = {v for v in V8 if (o.groupby("cid")[v].transform(lambda s: s - s.mean()).abs().max() < 1e-12)}
        for v in V8:
            k = 1000 if v == "battle_fat" else 1
            if v not in fe.params.index:
                fe_cell = "absorbed"
            elif v in no_within:
                fe_cell = "not identified (no change within any country)"
            else:
                fe_cell = f"{k * fe.params[v]:.4f}{fe.stars()[v]} ({k * fe.se[v]:.4f})"
            rows.append(dict(group=name, variable=LABEL[v] + (" (1000s)" if k == 1000 else ""),
                             RE=f"{k * re.params[v]:.4f}{re.stars()[v]} ({k * re.se[v]:.4f})",
                             FE=fe_cell))
        rows.append(dict(group=name, variable="N / countries", RE=f"{re.nobs} / {o.cid.nunique()}", FE=fe.nobs))
        rows.append(dict(group=name, variable="mean share", RE=f"{o.share.mean():.4f}", FE=""))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "21_democracy_coefficients.csv", index=False)
    say("\n=== Coefficients by regime group (others only) ===")
    say(t.to_string(index=False))
    # pooled interactions
    o = p[(p.Israel_dummy == 0)].dropna(subset=V8 + ["share", "democracy"]).copy()
    inter = []
    for v in ["human_rights_score", "battle_fat", "TerritoryOccupied", "NuclearPowerOutsideNPT"]:
        o[f"dem_x_{v}"] = o["democracy"] * o[v]
        inter.append(f"dem_x_{v}")
    V = V8 + ["democracy"] + inter
    re = random_effects(o, "share", V, "cid", time="year")
    fe = fixed_effects(o, "share", [v for v in V if v != "PermanentUNSCMember"], "cid", time="year")
    rows = []
    for v in ["democracy"] + inter + ["human_rights_score", "battle_fat", "TerritoryOccupied", "NuclearPowerOutsideNPT"]:
        rows.append(dict(variable=v, RE=f"{re.params[v]:.5f}{re.stars()[v]} ({re.se[v]:.5f})",
                         FE=(f"{fe.params[v]:.5f}{fe.stars()[v]} ({fe.se[v]:.5f})" if v in fe.params.index else "absorbed")))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "21_democracy_interactions.csv", index=False)
    say("\n=== Pooled model with democracy interactions (others only) ===")
    say(t.to_string(index=False))


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
BOOT_SPECS = {}


def all_boot_specs():
    s = {}
    for name, (extra, mask) in {**controls_sets(), **oecd_sets(), **democracy_sets()}.items():
        if name == "baseline":
            continue
        s[name] = (extra, mask)
    return s


def rounds(p, name, model, seeds):
    extra, mask = all_boot_specs()[name]
    D, Vs, _ = group_data(p, extra, mask)
    V = Vs[model]
    D = D.reset_index(drop=True)
    isr = int(D.loc[D.Israel_dummy == 1, "cid"].iloc[0])
    groups = {c: idx for c, idx in D.groupby("cid").indices.items() if c != isr}
    path = OUT / "21_boot_rounds.csv"
    out = []
    for s in seeds:
        S = c19.draw_countries(D, np.random.default_rng(s), groups)
        try:
            out.append(dict(spec=name, model=model, seed=s, explained=c19.explained(S, model, V)[0], failed=""))
        except Exception as ex:
            out.append(dict(spec=name, model=model, seed=s, explained=np.nan, failed=str(ex)))
    out = pd.DataFrame(out)
    out.to_csv(path, mode="a", header=not path.exists(), index=False)
    say(f"{name} {model} seeds {seeds[0]}-{seeds[-1]}: ok {out.explained.notna().sum()}, failed {out.explained.isna().sum()}")


def summary(p):
    r = pd.read_csv(OUT / "21_boot_rounds.csv").drop_duplicates(["spec", "model", "seed"])
    specs = all_boot_specs()
    res, fails = [], []
    for (name, model), g in r.groupby(["spec", "model"]):
        extra, mask = specs[name]
        D, Vs, _ = group_data(p, extra, mask)
        V = Vs[model]
        pt, od, _ = c19.explained(D, model, V)
        ok = g.explained.dropna().values
        res.append(dict(spec=name, model=("RE" if model == "re_year" else "FE"), explained=round(pt, 4), overall=round(od, 4),
                        explained_pct=round(100 * pt / od, 1), boot_se=round(ok.std(ddof=1), 4), ratio=round(pt / ok.std(ddof=1), 2),
                        pct2_5=round(np.percentile(ok, 2.5), 4), pct97_5=round(np.percentile(ok, 97.5), 4),
                        rounds_ok=len(ok), rounds_failed=int(g.explained.isna().sum())))
        for reason, n in Counter(g.failed.dropna()).items():
            if reason:
                fails.append(dict(spec=name, model=model, reason=reason, rounds=n))
    t = pd.DataFrame(res)
    t.to_csv(OUT / "21_bootstrap.csv", index=False)
    pd.DataFrame(fails).to_csv(OUT / "21_bootstrap_failures.csv", index=False)
    say("\n=== Bootstrap (500 rounds, countries resampled, Israel fixed) ===")
    say(t.to_string(index=False))
    if fails:
        say(pd.DataFrame(fails).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="points", choices=["points", "boot", "summary", "list"])
    ap.add_argument("--spec", default="")
    ap.add_argument("--model", default="re_year")
    ap.add_argument("--seeds", default="0:500")
    a = ap.parse_args()
    say("Card 21: Referee 1 controls, OECD split, democracy test. Run " + time.strftime("%Y-%m-%d %H:%M"))
    p = build()
    if a.stage == "list":
        for k in all_boot_specs():
            print(k)
    elif a.stage == "points":
        coverage(p)
        table2_controls(p)
        decomp_table(p, controls_sets(), "decomp_controls")
        decomp_table(p, oecd_sets(), "oecd_split")
        democracy_coefficients(p)
        decomp_table(p, democracy_sets(), "democracy_decomp")
        (OUT / "21_run_log_points.txt").write_text("\n".join(LOG))
    elif a.stage == "boot":
        lo, hi = [int(x) for x in a.seeds.split(":")]
        rounds(p, a.spec, a.model, list(range(lo, hi)))
    else:
        summary(p)
        (OUT / "21_run_log_summary.txt").write_text("\n".join(LOG))
