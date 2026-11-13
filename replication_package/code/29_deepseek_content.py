"""
29_deepseek_content.py

Task card 29 (29 October 2026): a second, independent reading of the content
classification of memo 14. DeepSeek V4 Pro reads each of the 1,682
resolution-country pairs that Claude classified in September and answers the
same two questions with the same codebook: does the resolution accuse the
state (Y / M / N), and what is the accusation about (R, C, F, O, N, T, other,
with the strict and broad readings of R and C). The answers are then compared
with Claude's codes row by row.

THIS SCRIPT RUNS ON MATANEL'S COMPUTER (double-click code/29_run_deepseek_content.command).
It uses the key saved by card 24 in ~/.deepseek_key. Thinking mode is ON
(the model reasons before answering), because the task is a legal reading
rather than a one-word label; cost about 5 to 10 dollars for all 1,682 pairs.

Inputs: data/processed/12_reading_set.csv (row_id, res_id, year, country, text),
        data/processed/13_families.csv (titles), data/processed/14_row_codes.csv (Claude's codes)
Outputs: data/processed/29_deepseek_content_raw.jsonl (resumable), 29_deepseek_content.csv
         output/tables/29_agreement.csv, 29_agreement_by_group.csv, 29_disagreements.csv, 29_run_log.txt
Run: python3 code/29_deepseek_content.py --sample 20 | python3 code/29_deepseek_content.py | --stage summary
"""

import argparse
import concurrent.futures as cf
import json
import os
import re
import ssl
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

try:
    import certifi
    SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except Exception:
    SSL_CTX = ssl.create_default_context()

ROOT = Path(__file__).resolve().parents[1]
PROC, OUT = ROOT / "data" / "processed", ROOT / "output" / "tables"
RAW = PROC / "29_deepseek_content_raw.jsonl"
MODEL, URL = "deepseek-v4-pro", "https://api.deepseek.com/chat/completions"

PROMPT = """You are an expert in public international law reading resolutions of the UN General Assembly.

Below is the full text of one General Assembly resolution. An automatic classifier labelled it as criticising the state "{state}". Answer two questions about that state only.

QUESTION 1. Does the resolution accuse {state}?
- "Y": the resolution criticises, condemns, deplores, demands something of, or otherwise accuses {state} (or its government, forces or authorities) by name.
- "M": the resolution concerns a situation involving {state} but does not accuse it (for example assistance to refugees, financing of a peacekeeping force, {state} named only as a party to talks, or criticism of unnamed "foreign forces").
- "N": the resolution is not a criticism of {state} at all; the classifier picked the wrong country (for example a relief resolution for the state, or the state is the victim of the conduct condemned).

QUESTION 2. Only if the answer to question 1 is "Y": what is {state} accused of? Choose every category that applies.
- "R": racial or ethnic discrimination, or apartheid. Set "R_explicit" to true if the resolution uses the words racial, racist, ethnic or apartheid about {state}'s conduct, false if it speaks of discrimination without naming a racial or ethnic basis.
- "C": international crimes: war crimes, crimes against humanity, genocide, grave breaches of the Geneva Conventions, massacre, ethnic cleansing. Set "C_label" to true if such a label appears in the text about {state}'s conduct, false if the resolution only attributes large-scale civilian deaths or "excessive use of force" to {state} without such a label. A standard phrase that merely lists treaty articles ("penal sanctions, grave breaches and responsibilities") is not a label.
- "F": use of force against another state: aggression, invasion, armed attack. Force against the state's own population or inside an occupied territory is not F. A general principle ("inadmissibility of the acquisition of territory by force") is not F unless {state}'s own act is named.
- "O": occupation, annexation, settlements, or denial of a people's right to self-determination.
- "N": nuclear weapons, the Non-Proliferation Treaty, nuclear safeguards.
- "T": torture, slavery, forced labour.
- "other": the accusation is about something else (for example the costs of a UN force, an oil slick, an embargo).
Conventions: crimes by private actors (a settler, a militia) are not C for the state. Religious discrimination is not R.

Answer with a single JSON object and nothing else, in this form:
{{"accused": "Y", "categories": ["O", "F"], "R_explicit": false, "C_label": false, "quote": "the clause that best supports your answer, at most 40 words"}}
If the answer to question 1 is "M" or "N", give "categories": [] .

RESOLUTION {res_id} ({year}), title: {title}

{text}"""


def api_key():
    k = os.environ.get("DEEPSEEK_API_KEY")
    if k:
        return k.strip()
    f = Path.home() / ".deepseek_key"
    if not f.exists():
        sys.exit("No API key in ~/.deepseek_key")
    return f.read_text().strip()


def load_set():
    r = pd.read_csv(PROC / "12_reading_set.csv", usecols=["row_id", "res_id", "year", "country", "text"])
    fam = pd.read_csv(PROC / "13_families.csv")[["row_id", "title"]]
    r = r.merge(fam, on="row_id", how="left")
    r["title"] = r["title"].fillna("")
    r["text"] = r["text"].fillna("")
    return r


def ask(key, row, retries=6):
    body = {"model": MODEL,
            "messages": [{"role": "user", "content": PROMPT.format(state=row.country, res_id=row.res_id, year=int(row.year),
                                                                   title=row.title[:200], text=row.text[:120000])}],
            "max_tokens": 16000,   # reasoning tokens count against this limit; 4,000 was too few for some texts
            "response_format": {"type": "json_object"}}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    last = ""
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=300, context=SSL_CTX) as r:
                out = json.loads(r.read())
            msg = out["choices"][0]["message"]["content"]
            u = out.get("usage", {})
            return dict(answer=msg, prompt_tokens=u.get("prompt_tokens"), completion_tokens=u.get("completion_tokens"),
                        reasoning_tokens=(u.get("completion_tokens_details") or {}).get("reasoning_tokens"), error="")
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}: {e.read().decode(errors='ignore')[:300]}"
            if e.code == 429 or e.code >= 500:
                time.sleep(2 ** attempt)
                continue
            return dict(answer=None, error=last)
        except Exception as e:
            last = repr(e)
            if "CERTIFICATE_VERIFY_FAILED" in last:
                return dict(answer=None, error="SSL certificate problem: " + last)
            time.sleep(2 ** attempt)
    return dict(answer=None, error=last)


def run(sample, workers):
    key = api_key()
    df = load_set()
    done = set()
    if RAW.exists():
        for line in open(RAW):
            try:
                rec = json.loads(line)
                p = parse(rec.get("answer"))
                if p and p["ds_crit"]:          # an empty or unparseable answer is redone
                    done.add(rec["row_id"])
            except Exception:
                pass
    todo = df[~df.row_id.isin(done)]
    if sample:
        todo = todo.sample(min(sample, len(todo)), random_state=29)
    print(f"reading set {len(df)} pairs; already done {len(done)}; to do now {len(todo)}", flush=True)
    t0 = time.time()
    n_ok = n_err = tok_in = tok_out = 0
    errors = {}
    lock = threading.Lock()

    def work(row):
        r = ask(key, row)
        return dict(row_id=row.row_id, res_id=row.res_id, year=int(row.year), country=row.country, **r,
                    ts=time.strftime("%Y-%m-%d %H:%M:%S"))

    with open(RAW, "a") as f, cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(work, row): row.row_id for row in todo.itertuples(index=False)}
        for i, fut in enumerate(cf.as_completed(futs), 1):
            rec = fut.result()
            with lock:
                if rec["answer"] is None:
                    n_err += 1
                    errors[rec["error"][:160]] = errors.get(rec["error"][:160], 0) + 1
                    if "HTTP 401" in rec["error"] or "HTTP 402" in rec["error"]:
                        print("STOPPING:", rec["error"], flush=True)
                        for other in futs:
                            other.cancel()
                        break
                else:
                    n_ok += 1
                    tok_in += rec["prompt_tokens"] or 0
                    tok_out += rec["completion_tokens"] or 0
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                if i % 50 == 0 or i == len(futs):
                    cost = (tok_in * 1.32 + tok_out * 3.96) / 1e6
                    rate = i / (time.time() - t0)
                    print(f"{i}/{len(futs)} done, {n_err} errors, {tok_in/1e6:.2f}M in, {tok_out/1e6:.2f}M out, "
                          f"cost so far at most ${cost:.2f}, about {(len(futs) - i) / rate / 60:.0f} min left", flush=True)
    print(f"finished: {n_ok} answered, {n_err} errors, in {(time.time() - t0)/60:.1f} min")
    for msg, n in sorted(errors.items(), key=lambda x: -x[1])[:5]:
        print(f"  {n} x {msg}")
    tidy()


def parse(answer):
    try:
        j = json.loads(answer)
    except Exception:
        m = re.search(r"\{.*\}", str(answer), re.S)
        if not m:
            return None
        try:
            j = json.loads(m.group(0))
        except Exception:
            return None
    acc = str(j.get("accused", "")).strip().upper()[:1]
    cats = [str(c).strip().upper() for c in (j.get("categories") or [])]
    cats = [("OTHER" if c.startswith("OTH") else c[:1]) for c in cats]
    return dict(ds_crit=acc if acc in "YMN" else "",
                ds_cats=";".join(sorted(set(c for c in cats if c in ("R", "C", "F", "O", "N", "T")))),
                ds_other=int("OTHER" in cats), ds_R_explicit=bool(j.get("R_explicit")), ds_C_label=bool(j.get("C_label")),
                ds_quote=str(j.get("quote", ""))[:300])


def tidy():
    rows = [json.loads(l) for l in open(RAW)] if RAW.exists() else []
    out, seen = [], set()
    for r in reversed(rows):                 # keep the latest usable answer per row
        p = parse(r["answer"]) or {}
        if not p.get("ds_crit") or r["row_id"] in seen:
            continue
        seen.add(r["row_id"])
        out.append({k: r.get(k) for k in ("row_id", "res_id", "year", "country", "prompt_tokens", "completion_tokens", "reasoning_tokens")} | p)
    t = pd.DataFrame(out)
    if len(t):
        t.to_csv(PROC / "29_deepseek_content.csv", index=False)
    return t


def summary():
    d = tidy()
    c = pd.read_csv(PROC / "14_row_codes.csv")
    m = c.merge(d, on="row_id", suffixes=("", "_ds"))
    m = m[m.ds_crit.notna() & (m.ds_crit != "")]
    log = []

    def say(s=""):
        print(s); log.append(str(s))
    say(f"Pairs with both readings: {len(m)} of {len(c)}")
    m["group"] = ["Israel" if x == "Israel" else ("South Africa" if x == "South Africa" else "Comparison states") for x in m.country]
    m["agree_crit"] = (m.crit == m.ds_crit)
    m["accused_both"] = (m.crit == "Y") & (m.ds_crit == "Y")
    for k in ["R", "C", "F", "O", "N", "T"]:
        m[f"cl_{k}"] = m[f"cat_{k}"].fillna(0).astype(int)
        m[f"ds_{k}"] = m.ds_cats.fillna("").str.contains(k).astype(int)
        m[f"agree_{k}"] = (m[f"cl_{k}"] == m[f"ds_{k}"])
    m["ds_R_strict"] = ((m.ds_R == 1) & m.ds_R_explicit).astype(int)
    m["ds_C_strict"] = ((m.ds_C == 1) & m.ds_C_label).astype(int)
    rows = []
    for g, s in [("all", m)] + [(g, m[m.group == g]) for g in ["Israel", "South Africa", "Comparison states"]]:
        a = s[s.accused_both]
        rows.append(dict(group=g, pairs=len(s), agree_accused=round(s.agree_crit.mean(), 3),
                         claude_Y=int((s.crit == "Y").sum()), deepseek_Y=int((s.ds_crit == "Y").sum()),
                         both_Y=len(a), **{f"agree_{k}": round(a[f"agree_{k}"].mean(), 3) for k in ["R", "C", "F", "O", "N", "T"]},
                         claude_R=int(a.cl_R.sum()), deepseek_R=int(a.ds_R.sum()), claude_R_strict=int(a.cat_R_strict.fillna(0).sum()),
                         deepseek_R_strict=int(a.ds_R_strict.sum()), claude_O=int(a.cl_O.sum()), deepseek_O=int(a.ds_O.sum()),
                         claude_C=int(a.cl_C.sum()), deepseek_C=int(a.ds_C.sum()), claude_F=int(a.cl_F.sum()), deepseek_F=int(a.ds_F.sum())))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "29_agreement_by_group.csv", index=False)
    say(t.T.to_string())
    ct = pd.crosstab(m.crit, m.ds_crit)
    say("\nAccused? Claude (rows) against DeepSeek (columns):")
    say(ct.to_string())
    dis = m[~m.agree_crit | (m.accused_both & ~(m.agree_R & m.agree_C & m.agree_F & m.agree_O & m.agree_N & m.agree_T))]
    dis[["row_id", "res_id", "year", "country", "crit", "ds_crit", "cats", "ds_cats", "quote", "ds_quote"]].to_csv(OUT / "29_disagreements.csv", index=False)
    say(f"\nDisagreements (any code): {len(dis)} of {len(m)} -> output/tables/29_disagreements.csv")
    say(f"Tokens: {d.prompt_tokens.sum()/1e6:.2f}M in, {d.completion_tokens.sum()/1e6:.2f}M out (reasoning {d.reasoning_tokens.fillna(0).sum()/1e6:.2f}M)")
    (OUT / "29_run_log.txt").write_text("\n".join(log))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="run", choices=["run", "summary", "diagnose"])
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.stage == "run":
        run(a.sample, a.workers)
    elif a.stage == "diagnose":
        df = load_set()
        row = df[df.country == "Israel"].iloc[0]
        r = ask(api_key(), row, retries=1)
        print(json.dumps(r, ensure_ascii=False, indent=1)[:1500])
        print("parsed:", parse(r["answer"]) if r["answer"] else None)
    else:
        summary()
