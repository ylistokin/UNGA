"""
13_reading_packets.py

Reading aid for the hand classification (data point 2). It does not classify
anything. For each resolution-country row in data/processed/12_reading_set.csv
it prepares what the reader (Claude) reads:

  * the resolution title (Voelsen: title_x; Mesquita: topic);
  * every clause (text split at semicolons and numbered paragraphs) that names
    the criticised state, or contains one of the screening terms below.

Rows are grouped into "families": same criticised state and same title (for
example the yearly "Jerusalem" resolution on Israel). The first row of each
family (earliest year) is shown with all its matching clauses. Every other row
of the family is shown too, but only with the clauses that do not already
appear in the family's first row, so that changes from year to year are read
and nothing is skipped silently. Every row appears in the packets.

Screening terms (a reading aid only; the classification is made by reading):
  1 racial discrimination / apartheid : apartheid, racis(m/t), racial, segregation
  2 international crimes : genocide, crimes against humanity, war crime(s),
    grave breach(es), ethnic cleansing, extrajudicial, summary execution(s),
    massacre(s), mass killing(s)
  3 use of force (art. 2(4)) : aggression, invasion, invade(d), use of force,
    armed attack, territorial integrity, withdrawal of (foreign) troops/forces
  4 occupation / annexation / self-determination : occupation, occupied,
    settlement(s), annex(ation), self-determination, colonial
  5 nuclear / NPT : nuclear, non-proliferation
  6 other ius cogens (torture, slavery) : torture, slavery, forced labour, enslave
  Condemnation verbs : condemn(s), deplore(s), demand(s), censure(s)

Inputs: data/processed/12_reading_set.csv, full_text_data2.csv (title_x),
        MesquitaGPT.csv (topic)
Outputs: data/processed/13_families.csv (row_id -> family id and title)
         data/processed/13_screen_flags.csv (term flags per row)
         reading packets in the directory given as argument (not part of the
         deliverable; regenerated on each run)
Run: python3 code/13_reading_packets.py <packet_dir>
"""

import re
import sys
from pathlib import Path
import pandas as pd

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "packets")
OUT.mkdir(parents=True, exist_ok=True)
# characters of matching clauses shown: first row of a family / other rows
# (Israel and South Africa have many near-identical yearly resolutions, so a
# smaller cap is used for their later rows; see memo 10)
CAP_FIRST, CAP_OTHER, CAP_OTHER_BIG = 3500, 1200, 600
D = "Israel and the UN/Data/"

r = pd.read_csv("data/processed/12_reading_set.csv")
v = pd.read_csv(D + "full_text_data2.csv", usecols=["unres", "title_x"]).drop_duplicates("unres")
m = pd.read_csv(D + "GPT_Counts/MesquitaGPT.csv", usecols=["res_id2", "topic"]).drop_duplicates("res_id2")
titles = dict(zip(v["unres"], v["title_x"]))
titles.update(dict(zip(m["res_id2"], m["topic"])))
r["title"] = r["res_id"].map(titles).fillna("")
r["fam_key"] = r["title"].str.lower().str.replace(r"[^a-z ]", "", regex=True).str.split().str.join(" ")

# names and adjectives used to find clauses about the criticised state
NAMES = {
    "Israel": r"israel", "South Africa": r"south\s*africa|pretoria|racist r[eé]gime",
    "United States": r"united\s*states|american", "United Kingdom": r"united\s*kingdom|british",
    "Russia": r"russia|soviet", "Iraq": r"iraq", "Iran": r"iran", "Syria": r"syria",
    "Myanmar": r"myanmar|burm", "Sudan": r"sudan", "Serbia": r"serbia|yugoslavia|belgrade",
    "Afghanistan": r"afghan|taliban", "Somalia": r"somali", "Morocco": r"morocc",
    "Guatemala": r"guatemala", "Cambodia": r"cambodia|kampuchea", "Ethiopia": r"ethiopia",
    "El Salvador": r"salvador", "Chile": r"chile", "Namibia": r"namibia", "Rwanda": r"rwanda",
    "Indonesia": r"indonesia|timor", "Ukraine": r"ukrain", "Liberia": r"liberia",
    "Nigeria": r"nigeria", "Uganda": r"uganda", "Pakistan": r"pakistan", "Argentina": r"argentin|malvinas|falkland",
    "Angola": r"angola", "Belarus": r"belarus", "Equatorial Guinea": r"equatorial\s*guinea",
    "Haiti": r"haiti", "Colombia": r"colombia", "Armenia": r"armenia|karabakh",
    "Bangladesh": r"bangladesh", "Central African Republic": r"central\s*african", "Turkey": r"turk",
    "France": r"france|french", "Eritrea": r"eritrea", "Democratic Republic of the Congo": r"congo",
    "Zaire": r"zaire|congo", "Vietnam": r"viet", "Bosnia and Herzegovina": r"bosnia",
}
SCREEN = {
    "s1_racial": r"apartheid|racis[mt]|racial|segregation",
    "s2_crimes": r"genocid|crimes?\s*against\s*humanity|war\s*crimes?|grave\s*breach|ethnic\s*cleansing|extrajudicial|summary\s*execution|massacre|mass\s*killing",
    "s3_force": r"aggression|invasion|invade|use\s*of\s*force|armed\s*attack|territorial\s*integrity|withdrawal\s*of\s*(all\s*)?(the\s*)?(foreign\s*)?(troops|forces)",
    "s4_occupation": r"occupation|occupied|settlements?\b|annex(ation|ed)|self\s*-?\s*determination|colonial",
    "s5_nuclear": r"nuclear|non\s*-?\s*proliferation",
    "s6_torture_slavery": r"tortur|slavery|forced\s*labou?r|enslav",
    "verb_condemn": r"condemn|deplor|demand|censur",
}
ANY = "|".join(f"(?:{p})" for p in SCREEN.values())

def clauses(text):
    t = re.sub(r"\s+", " ", str(text))
    parts = re.split(r";|\s(?=\d{1,2}\.\s+[A-Za-z])|(?<=\.)\s(?=[A-Z][a-z]+s\s)", t)
    return [p.strip() for p in parts if len(p.strip()) > 15]

def norm(c):
    return re.sub(r"[^a-z]", "", c.lower())[:200]

flags = []
fam_rows = []
r = r.sort_values(["country", "fam_key", "year", "row_id"]).reset_index(drop=True)
r["family"] = r.groupby(["country", "fam_key"], sort=False).ngroup().map(lambda i: f"F{i:03d}")
packets = {}
for fam, g in r.groupby("family", sort=False):
    country = g["country"].iloc[0]
    name_pat = NAMES.get(country, re.escape(country.lower()))
    seen = set()
    lines = [f"\n######## {fam} | {country} | {g['title'].iloc[0][:200]} | rows: {len(g)} | years: {g['year'].min()}-{g['year'].max()}"]
    for k, (_, row) in enumerate(g.iterrows()):
        text = str(row["text"])
        f = {"row_id": row["row_id"]}
        for s, p in SCREEN.items():
            f[s] = int(bool(re.search(p, re.sub(r"-\s*", "", text), flags=re.I)))
        f["names_state"] = int(bool(re.search(name_pat, text, flags=re.I)))
        flags.append(f)
        fam_rows.append({"row_id": row["row_id"], "family": fam, "title": row["title"]})
        cl = [c for c in clauses(text) if re.search(name_pat, c, re.I) or re.search(ANY, re.sub(r"-\s*", "", c), re.I)]
        new = []
        for c in cl:
            n = norm(c)
            if n not in seen:
                seen.add(n)
                new.append(c[:420])
        cap = CAP_FIRST if k == 0 else (CAP_OTHER_BIG if country in ("Israel", "South Africa") else CAP_OTHER)
        body, total = [], 0
        for c in new:
            if total + len(c) > cap:
                body.append(f"[... {len(new) - len(body)} more clauses cut]")
                break
            body.append("- " + c)
            total += len(c)
        hdr = (f"== {row['row_id']} {row['year']} {row['res_id']} gpt='{row['gpt_label']}' "
               f"named={f['names_state']} screen=" +
               "".join(str(f[s]) for s in list(SCREEN)[:6]) + f" len={len(text)}")
        if k == 0:
            lines.append(hdr + "  [FIRST OF FAMILY]")
        else:
            lines.append(hdr + ("  [no new clauses]" if not new else ""))
        lines.extend(body)
    packets.setdefault(country, []).append("\n".join(lines))

pd.DataFrame(fam_rows).to_csv("data/processed/13_families.csv", index=False)
pd.DataFrame(flags).to_csv("data/processed/13_screen_flags.csv", index=False)
for country, blocks in packets.items():
    fn = OUT / (re.sub(r"[^A-Za-z]", "_", country) + ".txt")
    fn.write_text("\n".join(blocks))
sizes = {p.name: p.stat().st_size for p in OUT.glob("*.txt")}
print(f"families: {r['family'].nunique()}, rows: {len(r)}, packet chars: {sum(sizes.values())}")
for k, s in sorted(sizes.items(), key=lambda x: -x[1])[:15]:
    print(k, s)
