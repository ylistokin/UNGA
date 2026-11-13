"""
28_deepseek_decomposition.py

Task card 24, part 2 (29 October 2026): what happens to the paper's numbers
when the country-year counts come from DeepSeek V4 Pro's labels instead of
GPT-4o mini's. Runs in Claude's sandbox on the labels Matanel produced with
code/24_deepseek_reclassify.py.

Steps
  1. Turn DeepSeek's answers into counts the same way the paper's pipeline turns
     GPT's answers into counts (memo 19, variant A): split multi-country
     answers, harmonise names to final_data's names, drop the two session-48
     duplicates, count per country-year 1981-2023.
  2. Replace `count` and `share` in the corrected panel (variant A) with the
     DeepSeek counts and shares.
  3. Table 1 lines for Israel and others; Table 3 rows 1 and 2 (corrected
     method) with and without South Africa and the nuclear variable; 500-round
     country bootstrap for the two main rows.
  4. Where the two classifiers disagree about Israel: what those resolutions
     are (titles), and what Israel's share would be if the resolutions that
     merely concern Israel were excluded.

Inputs: data/processed/24_deepseek_labels.csv, Israel and the UN/Data/final_data.dta,
  the raw label files (through code/19_corrected_tables.py), the title columns of
  full_text_data2.csv and MesquitaGPT.csv
Outputs: data/processed/28_deepseek_counts.csv, data/processed/28_panel_deepseek.csv,
  output/tables/28_table1.csv, 28_table3_points.csv, 28_contributions.csv,
  28_boot_rounds.csv, 28_bootstrap.csv, 28_israel_label_diff_titles.csv, 28_run_log_*.txt
Run: python3 code/28_deepseek_decomposition.py                (points)
     python3 code/28_deepseek_decomposition.py --stage boot --spec <re_year|fe> --variant <as is|without South Africa> --seeds 0:500
     python3 code/28_deepseek_decomposition.py --stage summary
Requires: python 3.10+, numpy, pandas. No network access.
"""

import argparse
import importlib
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
c19 = importlib.import_module("19_corrected_tables")
DATA, PROC, OUT = ROOT / "Israel and the UN" / "Data", ROOT / "data" / "processed", ROOT / "output" / "tables"
V8 = c19.V8
LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(str(s))


# names as DeepSeek writes them -> names in final_data.dta
PROTECT = ["Bosnia and Herzegovina", "Antigua and Barbuda", "Trinidad and Tobago", "Sao Tome and Principe",
           "Serbia and Montenegro", "Saint Kitts and Nevis", "Saint Vincent and the Grenadines",
           "Yugoslavia (Serbia and Montenegro)", "Federal Republic of Yugoslavia (Serbia and Montenegro)",
           "United Kingdom of Great Britain and Northern Ireland"]
TO_FINAL = {
    "Democratic People’s Republic of Korea": "Korea, Democratic People's Republic of",
    "Democratic People's Republic of Korea": "Korea, Democratic People's Republic of",
    "the Democratic People’s Republic of Korea": "Korea, Democratic People's Republic of",
    "North Korea": "Korea, Democratic People's Republic of", "DPRK": "Korea, Democratic People's Republic of",
    "United Kingdom of Great Britain and Northern Ireland": "United Kingdom", "UK": "United Kingdom",
    "United States": "United States of America", "USA": "United States of America",
    "Syrian Arab Republic": "Syria", "Russian Federation": "Russia", "Soviet Union": "Russia", "USSR": "Russia",
    "Federal Republic of Yugoslavia (Serbia and Montenegro)": "Serbia", "Federal Republic of Yugoslavia": "Serbia",
    "Yugoslavia": "Serbia", "Yugoslavia (Serbia and Montenegro)": "Serbia", "Serbia and Montenegro": "Serbia",
    "Democratic Republic of the Congo": "Congo, Democratic Republic of", "Zaire": "Congo, Democratic Republic of",
    "Viet Nam": "Vietnam", "Bosnia and Herzegovina": "Bosnia Herzegovenia",
    "Libyan Arab Jamahiriya": "Libya", "Islamic Republic of Iran": "Iran", "Côte d’Ivoire": "Cote d'Ivoire",
    "Côte d'Ivoire": "Cote d'Ivoire", "the Comoros": "Comoros", "Republic of Korea": "Korea, Republic of",
    "South Korea": "Korea, Republic of", "Republic of Moldova": "Moldova", "Taliban": "Afghanistan",
    "Southern Rhodesia": "Zimbabwe", "Southern Rhodesia (Zimbabwe)": "Zimbabwe", "Burma": "Myanmar",
}
NOT_STATES = {"International Telecommunication Union", "International Labour Organisation", "The resolution condemns the Bosnian Serb party"}


def split_label(s):
    s = str(s).strip().strip('."\'')
    if s.lower() in ("none", "nan", ""):
        return []
    # a few answers include a sentence; keep only the part after the last colon
    if ":" in s:
        s = s.split(":")[-1]
    for i, p in enumerate(PROTECT):
        s = s.replace(p, f"@{i}@")
    parts = re.split(r",\s*|\s+and\s+|\s*/\s*|;\s*|\n+", s)
    out = []
    for p in parts:
        for i, pp in enumerate(PROTECT):
            p = p.replace(f"@{i}@", pp)
        p = re.sub(r"^(and|the)\s+", "", p.strip()).strip(". ")
        if p and p.lower() != "none" and p not in NOT_STATES:
            out.append(TO_FINAL.get(p, p))
    return list(dict.fromkeys(out))


def deepseek_counts(final_names):
    t = pd.read_csv(PROC / "24_deepseek_labels.csv")
    t = t[t.year.between(1981, 2023)]
    t = t[~((t.source == "Voelsen") & t.res_id.isin(c19.SESSION48_DUPES))]
    rows = []
    for r in t.itertuples():
        for c in split_label(r.deepseek_label):
            rows.append((c, int(r.year), r.uid))
    lab = pd.DataFrame(rows, columns=["country", "year", "uid"])
    unmatched = lab[~lab.country.isin(final_names)].country.value_counts()
    say(f"DeepSeek labels: {len(lab)} resolution-country rows; names not in final_data (dropped): "
        f"{int(unmatched.sum())} rows, {len(unmatched)} names: {unmatched.head(10).to_dict()}")
    lab = lab[lab.country.isin(final_names)]
    cnt = lab.groupby(["country", "year"]).size().rename("count_ds").reset_index()
    cnt.to_csv(PROC / "28_deepseek_counts.csv", index=False)
    return cnt


def build():
    allc = c19.rebuild_counts()
    p = c19.build_panels(allc)["corrected_A"].copy()
    p["year"] = p["year"].astype(int)
    cnt = deepseek_counts(set(p.country.unique()))
    p = p.merge(cnt, on=["country", "year"], how="left")
    p["count_gpt"], p["share_gpt"] = p["count"], p["share"]
    p["count"] = p["count_ds"].fillna(0)
    p["total_count"] = p.groupby("year")["count"].transform("sum")
    p["share"] = p["count"] / p["total_count"]
    p["is_sa"] = (p.country == "South Africa").astype(int)
    p.to_csv(PROC / "28_panel_deepseek.csv", index=False)
    return p


def table1(p):
    rows = []
    base = p.dropna(subset=V8 + ["share"])
    for g, name in [(1, "Israel"), (0, "Other countries")]:
        s = base[base.Israel_dummy == g]
        rows.append(dict(group=name, gpt_count=round(s.count_gpt.mean(), 2), deepseek_count=round(s["count"].mean(), 2),
                         gpt_share=round(s.share_gpt.mean(), 4), deepseek_share=round(s.share.mean(), 4), rows=len(s)))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "28_table1.csv", index=False)
    say("\n=== Table 1 lines, GPT (variant A) against DeepSeek ===")
    say(t.to_string(index=False))
    isr = p[p.Israel_dummy == 1]
    say(f"Israel annual share: correlation between the two classifiers {np.corrcoef(isr.share_gpt, isr.share)[0, 1]:.3f}; "
        f"mean absolute difference {np.abs(isr.share_gpt - isr.share).mean():.3f}")


VARIANTS = {"as is": (True, True), "without South Africa": (False, True),
            "without nuclear variable": (True, False), "without both": (False, False)}


def spec_data(p, variant):
    keep_sa, keep_nuc = VARIANTS[variant]
    V = [v for v in V8 if keep_nuc or v != "NuclearPowerOutsideNPT"]
    D = p if keep_sa else p[p.is_sa == 0]
    D = D[D.Israel_dummy.eq(1) | D[V + ["share"]].notna().all(axis=1)]
    return D, V


def points(p):
    rows, crows = [], []
    for variant in VARIANTS:
        D, V = spec_data(p, variant)
        for model, ml in [("re_year", "RE"), ("fe", "FE")]:
            e, o, c = c19.explained(D, model, V)
            rows.append(dict(counts="DeepSeek", variant=variant, model=ml, explained=round(e, 4), overall=round(o, 4), explained_pct=round(100 * e / o, 1)))
            crows.append(dict(variant=variant, model=ml, **{c19.LABEL[v]: round(c[v], 4) for v in V}, total=round(e, 4)))
    # the GPT reference on the same rows
    q = p.copy()
    q["share"] = q["share_gpt"]
    for variant in VARIANTS:
        D, V = spec_data(q, variant)
        for model, ml in [("re_year", "RE"), ("fe", "FE")]:
            e, o, _ = c19.explained(D, model, V)
            rows.append(dict(counts="GPT (variant A)", variant=variant, model=ml, explained=round(e, 4), overall=round(o, 4), explained_pct=round(100 * e / o, 1)))
    t = pd.DataFrame(rows).sort_values(["variant", "model", "counts"])
    t.to_csv(OUT / "28_table3_points.csv", index=False)
    pd.DataFrame(crows).fillna("").to_csv(OUT / "28_contributions.csv", index=False)
    say("\n=== Table 3 rows 1-2 with DeepSeek counts against GPT counts ===")
    say(t.to_string(index=False))


def israel_diff(p):
    t = pd.read_csv(PROC / "24_deepseek_labels.csv")
    t["gpt_isr"] = t.gpt_label.fillna("").str.contains("Israel")
    t["ds_isr"] = t.deepseek_label.fillna("").str.contains("Israel")
    v = pd.read_csv(DATA / "full_text_data2.csv", usecols=["unres", "title_x"]).drop_duplicates("unres")
    m = pd.read_csv(DATA / "GPT_Counts" / "MesquitaGPT.csv", usecols=["res_id2", "topic"])
    titles = dict(zip("Voelsen|" + v.unres.astype(str), v.title_x))
    titles.update(dict(zip("Mesquita|" + m.res_id2.astype(str), m.topic)))
    d = t[t.gpt_isr != t.ds_isr].copy()
    d["title"] = d.uid.map(titles).fillna("")
    d["title_key"] = d.title.str.lower().str.replace(r"\d+", "", regex=True).str.strip()
    d[["uid", "year", "gpt_label", "deepseek_label", "title"]].to_csv(OUT / "28_israel_label_diff_titles.csv", index=False)
    say(f"\nResolutions where only one classifier names Israel: {len(d)} (DeepSeek only {int((d.ds_isr & ~d.gpt_isr).sum())}, GPT only {int((d.gpt_isr & ~d.ds_isr).sum())})")
    say("Most common titles among the DeepSeek-only ones:")
    say(d[d.ds_isr].title_key.value_counts().head(12).to_string())


def rounds(p, spec, variant, seeds):
    D, V = spec_data(p, variant)
    D = D.reset_index(drop=True)
    isr = int(D.loc[D.Israel_dummy == 1, "cid"].iloc[0])
    groups = {c: idx for c, idx in D.groupby("cid").indices.items() if c != isr}
    out = []
    for s in seeds:
        S = c19.draw_countries(D, np.random.default_rng(s), groups)
        try:
            out.append(dict(spec=spec, variant=variant, seed=s, explained=c19.explained(S, spec, V)[0], failed=""))
        except Exception as ex:
            out.append(dict(spec=spec, variant=variant, seed=s, explained=np.nan, failed=str(ex)))
    out = pd.DataFrame(out)
    path = OUT / "28_boot_rounds.csv"
    out.to_csv(path, mode="a", header=not path.exists(), index=False)
    say(f"{spec} | {variant}: ok {out.explained.notna().sum()}, failed {out.explained.isna().sum()}")


def summary(p):
    r = pd.read_csv(OUT / "28_boot_rounds.csv").drop_duplicates(["spec", "variant", "seed"])
    res = []
    for (spec, variant), g in r.groupby(["spec", "variant"]):
        D, V = spec_data(p, variant)
        pt, od, _ = c19.explained(D, spec, V)
        ok = g.explained.dropna().values
        res.append(dict(spec=("RE" if spec == "re_year" else "FE"), variant=variant, explained=round(pt, 4), overall=round(od, 4),
                        explained_pct=round(100 * pt / od, 1), boot_se=round(ok.std(ddof=1), 4), ratio=round(pt / ok.std(ddof=1), 2),
                        rounds_ok=len(ok), rounds_failed=int(g.explained.isna().sum())))
    t = pd.DataFrame(res)
    t.to_csv(OUT / "28_bootstrap.csv", index=False)
    say("\n=== Bootstrap with DeepSeek counts (500 rounds, countries, Israel fixed) ===")
    say(t.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="points", choices=["points", "boot", "summary"])
    ap.add_argument("--spec", default="re_year")
    ap.add_argument("--variant", default="as is")
    ap.add_argument("--seeds", default="0:500")
    a = ap.parse_args()
    say("Card 24 part 2: DeepSeek counts in the decomposition. Run " + time.strftime("%Y-%m-%d %H:%M"))
    p = build()
    if a.stage == "points":
        table1(p)
        points(p)
        israel_diff(p)
        (OUT / "28_run_log_points.txt").write_text("\n".join(LOG))
    elif a.stage == "boot":
        lo, hi = [int(x) for x in a.seeds.split(":")]
        rounds(p, a.spec, a.variant, list(range(lo, hi)))
    else:
        summary(p)
        (OUT / "28_run_log_summary.txt").write_text("\n".join(LOG))
