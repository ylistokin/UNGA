"""
24_deepseek_reclassify.py

Task card 24 (29 October 2026): re-classify every UNGA resolution in the
corpus with a Chinese model (DeepSeek V4 Pro), using the same prompt that
produced the paper's GPT-4o-mini labels, so that the two classifiers can be
compared (Yair's request of 30 September 2026; also Engel's question whether
GPT and UN Watch could share a pro-Israel bias).

THIS SCRIPT RUNS ON MATANEL'S COMPUTER, not in Claude's sandbox (which has no
network access). It needs Python 3.9+ with pandas; nothing else (HTTP is done
with the standard library).

Setup (once)
  1. Put the DeepSeek API key, and nothing else, in a file in your home folder:
         ~/.deepseek_key
     (on a Mac: a file called .deepseek_key in /Users/<you>/). The project
     folder is synced to Google Drive, so the key must not be inside it.
  2. Make sure the balance on platform.deepseek.com is topped up.

Run (from the project folder, i.e. the folder that contains CLAUDE.md)
  python3 code/24_deepseek_reclassify.py --sample 30        test run on 30 resolutions, prints cost
  python3 code/24_deepseek_reclassify.py                    full run, 14,095 resolutions, a few hours
  python3 code/24_deepseek_reclassify.py --stage summary    agreement with GPT, writes output/tables/24_*

The full run can be stopped at any time (Ctrl+C) and restarted with the same
command; it continues where it stopped, because every answer is appended to
data/processed/24_deepseek_raw.jsonl as soon as it arrives and already
answered resolutions are skipped.

What is sent to the model
  Model: deepseek-v4-pro (DeepSeek-V4-Pro-0813 at the time of writing), thinking
  mode off, temperature 0, max 50 output tokens. One request per resolution.
  Prompt: the exact prompt of Yair's notebooks (GPT_Voelsen.ipynb,
  GPT_on_Mesquita.ipynb), which says "international relations, law, and
  political science" (the paper's text says "macroeconomics"; the code is
  what ran, memo 01). Differences from the GPT run that cannot be avoided:
  GPT-4o-mini ran at temperature 0.5 with max_tokens 2500; here temperature 0
  so the run is reproducible.

Inputs (read-only)
  Israel and the UN/Data/full_text_data2.csv      (1994-2024, columns unres, year, text, country = GPT label)
  Israel and the UN/Data/GPT_Counts/MesquitaGPT.csv  (1981-1993, columns res_id2, session_reg, content, country)
Outputs
  data/processed/24_deepseek_raw.jsonl           one line per resolution: id, source, answer, tokens, time
  data/processed/24_deepseek_labels.csv          tidy table: id, source, year, gpt_label, deepseek_label
  output/tables/24_agreement_*.csv, 24_run_log.txt   (summary stage)
"""

import argparse
import concurrent.futures as cf
import json
import os
import sys
import time
import ssl
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

# SSL: a fresh python.org install on a Mac often lacks root certificates, which
# makes every HTTPS request fail. Use the certifi bundle if it is installed.
try:
    import certifi
    SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except Exception:
    SSL_CTX = ssl.create_default_context()

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Israel and the UN" / "Data"
PROC = ROOT / "data" / "processed"
OUT = ROOT / "output" / "tables"
RAW = PROC / "24_deepseek_raw.jsonl"
MODEL = "deepseek-v4-pro"
URL = "https://api.deepseek.com/chat/completions"
PRICE_IN, PRICE_OUT = 1.32, 3.96   # USD per 1M tokens, peak rate (off-peak is half); for the cost print only

PROMPT = ("You are an AI assistant with expertise in international relations, law, and political science.\n"
          "Your task is to analyze the text of resolutions issued by the UN General Assembly\n"
          "and identify if a specific country is being condemned or criticized by the UN resolution.\n"
          "For each General Assembly Resolution I provide, please return only the name of the specific country being criticized.\n"
          "If no specific country is being criticized, return \"None\": {text}")


def api_key():
    k = os.environ.get("DEEPSEEK_API_KEY")
    if k:
        return k.strip()
    f = Path.home() / ".deepseek_key"
    if not f.exists():
        sys.exit("No API key: put it in ~/.deepseek_key or set DEEPSEEK_API_KEY")
    return f.read_text().strip()


def load_corpus():
    v = pd.read_csv(DATA / "full_text_data2.csv", usecols=["unres", "year", "text", "country"])
    v = v.rename(columns={"unres": "res_id", "country": "gpt_label"}).assign(source="Voelsen")
    m = pd.read_csv(DATA / "GPT_Counts" / "MesquitaGPT.csv", usecols=["res_id2", "session_reg", "content", "country"])
    m = (m.rename(columns={"res_id2": "res_id", "content": "text", "country": "gpt_label"})
          .assign(year=1945 + m["session_reg"], source="Mesquita").drop(columns="session_reg"))
    df = pd.concat([v, m], ignore_index=True)
    df["year"] = df["year"].astype("Int64")
    df["text"] = df["text"].fillna("")
    df["uid"] = df["source"] + "|" + df["res_id"].astype(str)
    return df


def ask(key, text, retries=6):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT.format(text=text)}],
        "temperature": 0,
        "max_tokens": 50,
        "thinking": {"type": "disabled"},
    }
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    last = ""
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=120, context=SSL_CTX) as r:
                out = json.loads(r.read())
            msg = out["choices"][0]["message"]["content"]
            u = out.get("usage", {})
            return dict(answer=msg, prompt_tokens=u.get("prompt_tokens"), completion_tokens=u.get("completion_tokens"),
                        cache_hit=u.get("prompt_cache_hit_tokens"), model=out.get("model"), error="")
        except urllib.error.HTTPError as e:
            code = e.code
            detail = e.read().decode(errors="ignore")[:300]
            last = f"HTTP {code}: {detail}"
            if code == 429 or code >= 500:    # rate limit or server trouble: wait and retry
                time.sleep(2 ** attempt)
                continue
            return dict(answer=None, error=last)   # any other 4xx will not fix itself
        except Exception as e:                # timeouts, connection resets, SSL problems
            last = repr(e)
            if "CERTIFICATE_VERIFY_FAILED" in last:
                return dict(answer=None, error="SSL certificate problem: " + last)
            time.sleep(2 ** attempt)
    return dict(answer=None, error=last)


def run(sample, workers):
    key = api_key()
    df = load_corpus()
    df = df[df.text.str.len() > 0]
    done = set()
    if RAW.exists():
        with open(RAW) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["uid"])
                except Exception:
                    pass
    todo = df[~df.uid.isin(done)]
    if sample:
        todo = todo.sample(min(sample, len(todo)), random_state=24)
    print(f"corpus {len(df)} resolutions with text; already done {len(done)}; to do now {len(todo)}", flush=True)
    t0 = time.time()
    n_ok = n_err = tok_in = tok_out = 0
    errors = {}
    lock = __import__("threading").Lock()

    def work(row):
        r = ask(key, row.text)
        rec = dict(uid=row.uid, res_id=row.res_id, source=row.source, year=int(row.year) if pd.notna(row.year) else None,
                   gpt_label=(None if pd.isna(row.gpt_label) else row.gpt_label), **r, ts=time.strftime("%Y-%m-%d %H:%M:%S"))
        return rec

    with open(RAW, "a") as f, cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(work, row): row.uid for row in todo.itertuples(index=False)}
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
                if i % 100 == 0 or i == len(futs):
                    cost = (tok_in * PRICE_IN + tok_out * PRICE_OUT) / 1e6
                    rate = i / (time.time() - t0)
                    left = (len(futs) - i) / rate / 60 if rate else 0
                    print(f"{i}/{len(futs)} done, {n_err} errors, {tok_in/1e6:.2f}M input tokens, "
                          f"cost so far at most ${cost:.2f} (peak rate), about {left:.0f} min left", flush=True)
    print(f"finished: {n_ok} answered, {n_err} errors, in {(time.time() - t0)/60:.1f} min")
    if errors:
        print("Error messages (how many times each):")
        for msg, n in sorted(errors.items(), key=lambda x: -x[1])[:5]:
            print(f"  {n} x {msg}")
    tidy()


def tidy():
    rows = [json.loads(l) for l in open(RAW)] if RAW.exists() else []
    t = pd.DataFrame(rows)
    if len(t):
        t["deepseek_label"] = t["answer"].astype(str).str.strip()
        t[["uid", "res_id", "source", "year", "gpt_label", "deepseek_label", "prompt_tokens", "completion_tokens"]].to_csv(
            PROC / "24_deepseek_labels.csv", index=False)
    return t


NAME_FIX = {"Russian Federation": "Russia", "Islamic Republic of Iran": "Iran", "Republic of Korea": "South Korea",
            "Democratic People's Republic of Korea": "North Korea", "Democratic People’s Republic of Korea": "North Korea",
            "Republic of Moldova": "Moldova", "United States of America": "United States", "Syrian Arab Republic": "Syria",
            "United Kingdom of Great Britain and Northern Ireland": "United Kingdom", "Taliban": "Afghanistan",
            "Federal Republic of Yugoslavia": "Serbia", "Serbia and Montenegro": "Serbia",
            "Federal Republic of Yugoslavia (Serbia and Montenegro)": "Serbia", "Côte d'Ivoire": "Cote d'Ivoire",
            "South Africa and Namibia": "South Africa", "Libya Arab Jamahiriya": "Libya", "Libyan Arab Jamahiriya": "Libya",
            "State of Palestine": "Palestine", "DPRK": "North Korea", "USA": "United States", "UK": "United Kingdom"}


def norm(label):
    if label is None or (isinstance(label, float) and pd.isna(label)):
        return "None"
    s = str(label).strip().strip('."\'').strip()
    if s.lower().startswith("understood") or s == "":
        return "NA"
    if s.lower() in ("none", "none.", "no specific country", "no country"):
        return "None"
    parts = [NAME_FIX.get(p.strip(), p.strip()) for p in s.split(",")]
    return ", ".join(sorted(set(parts)))


def summary():
    t = tidy()
    if not len(t):
        sys.exit("no answers yet")
    t["gpt"] = t.gpt_label.map(norm)
    t["ds"] = t.deepseek_label.map(norm)
    t = t[(t.gpt != "NA")]
    log = []

    def say(s=""):
        print(s); log.append(str(s))
    say(f"Resolutions with both labels: {len(t)} (of 14,095 in the corpus)")
    say(f"Exact agreement on the label: {(t.gpt == t.ds).mean():.3f}")
    both_none = (t.gpt == "None") & (t.ds == "None")
    both_crit = (t.gpt != "None") & (t.ds != "None")
    say(f"Both say no country criticised: {both_none.mean():.3f}; both name a country: {both_crit.mean():.3f}; "
        f"GPT names a country and DeepSeek does not: {((t.gpt != 'None') & (t.ds == 'None')).mean():.3f}; "
        f"the reverse: {((t.gpt == 'None') & (t.ds != 'None')).mean():.3f}")
    say(f"Among resolutions where both name a country, same country: {(t[both_crit].gpt == t[both_crit].ds).mean():.3f}")
    # Israel specifically
    gi, di = t.gpt.str.contains("Israel"), t.ds.str.contains("Israel")
    tp, fp, fn = int((gi & di).sum()), int((~gi & di).sum()), int((gi & ~di).sum())
    say(f"Israel: GPT {int(gi.sum())}, DeepSeek {int(di.sum())}, both {tp}, DeepSeek only {fp}, GPT only {fn}; "
        f"F1 of DeepSeek against GPT as reference {2*tp/(2*tp+fp+fn):.3f}")
    # per-year counts of Israel and of all critical resolutions, by both
    y = t.groupby("year").agg(n=("uid", "size"), gpt_crit=("gpt", lambda s: (s != "None").sum()),
                              ds_crit=("ds", lambda s: (s != "None").sum()),
                              gpt_israel=("gpt", lambda s: s.str.contains("Israel").sum()),
                              ds_israel=("ds", lambda s: s.str.contains("Israel").sum()))
    y["gpt_israel_share"] = (y.gpt_israel / y.gpt_crit).round(3)
    y["ds_israel_share"] = (y.ds_israel / y.ds_crit).round(3)
    y.to_csv(OUT / "24_agreement_by_year.csv")
    say("\nIsrael's share of critical resolutions by year, GPT vs DeepSeek:")
    say(y[["gpt_crit", "ds_crit", "gpt_israel", "ds_israel", "gpt_israel_share", "ds_israel_share"]].to_string())
    say(f"\nMean Israel share 1981-2023: GPT {y.loc[1981:2023].gpt_israel_share.mean():.3f}, DeepSeek {y.loc[1981:2023].ds_israel_share.mean():.3f}")
    # country totals
    def counts(col):
        s = t[t[col] != "None"][col].str.split(", ").explode()
        return s.value_counts()
    c = pd.concat([counts("gpt").rename("gpt"), counts("ds").rename("deepseek")], axis=1).fillna(0).astype(int)
    c["diff"] = c.deepseek - c.gpt
    c = c.sort_values("gpt", ascending=False)
    c.to_csv(OUT / "24_agreement_by_country.csv")
    say("\nTop 25 countries by number of critical resolutions, GPT vs DeepSeek:")
    say(c.head(25).to_string())
    dis = t[t.gpt != t.ds][["uid", "year", "gpt", "ds"]]
    dis.to_csv(OUT / "24_disagreements.csv", index=False)
    say(f"\nDisagreements saved: {len(dis)} rows -> output/tables/24_disagreements.csv")
    cost = (t.prompt_tokens.sum() * 0.66 + t.completion_tokens.sum() * 1.98) / 1e6
    say(f"Tokens: {t.prompt_tokens.sum()/1e6:.1f}M in, {t.completion_tokens.sum()/1e6:.2f}M out; cost between ${cost:.1f} (off-peak) and ${2*cost:.1f} (peak)")
    (OUT / "24_run_log.txt").write_text("\n".join(log))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="run", choices=["run", "summary", "diagnose"])
    ap.add_argument("--sample", type=int, default=0, help="test run on N random resolutions")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    PROC.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    if a.stage == "run":
        run(a.sample, a.workers)
    elif a.stage == "diagnose":
        # one short request, no retries, full error text
        r = ask(api_key(), "The General Assembly, Condemns the continued occupation of X by the armed forces of Y.", retries=1)
        print(json.dumps(r, ensure_ascii=False, indent=1))
    else:
        summary()
