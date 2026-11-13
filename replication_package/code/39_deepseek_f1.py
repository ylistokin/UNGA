"""
39_deepseek_f1.py

Card 39 (5 November 2026): Yair asked (reply of early November 2026) what the
F1 score is for the agreement between DeepSeek and GPT-4o mini, noting that
raw agreement (94 percent) is inflated because most resolutions criticize no
country. This script reports agreement measures that ignore the easy "None /
None" cases or correct for chance:

  1. Israel: F1 of DeepSeek against GPT-4o mini (as in card 24) and Cohen's kappa.
  2. Any country: F1 and kappa for "the resolution criticizes some country".
  3. Country level: micro F1 over (resolution, named country) pairs, so a
     resolution counts as agreement only if the same country is named.
  4. The same F1 for the countries most often named.

F1 is symmetric, so it does not matter which classifier is the reference.

Labels are normalized with norm() from code/24_deepseek_reclassify.py, exactly
as in the card 24 summary, and resolutions with an unusable GPT label ("NA")
are dropped, as there. Nothing is re-run against any model.

Inputs:  data/processed/24_deepseek_labels.csv (written by code/24, stage summary)
Outputs: output/tables/39_f1_scores.csv, output/tables/39_run_log.txt
Run:     python3 code/39_deepseek_f1.py      (from the project root)
Requires: python 3.10+, pandas
"""

import importlib
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
c24 = importlib.import_module("24_deepseek_reclassify")

PROC, OUT = ROOT / "data" / "processed", ROOT / "output" / "tables"
LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(str(s))


def f1(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan")


def kappa(a, b):
    """Cohen's kappa for two boolean series."""
    po = (a == b).mean()
    pa, pb = a.mean(), b.mean()
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe)


def binary_row(name, g, d):
    tp, fp, fn = int((g & d).sum()), int((~g & d).sum()), int((g & ~d).sum())
    tn = int((~g & ~d).sum())
    return dict(measure=name, gpt_positive=int(g.sum()), deepseek_positive=int(d.sum()), both=tp,
                deepseek_only=fp, gpt_only=fn, neither=tn, raw_agreement=round((g == d).mean(), 4),
                f1=round(f1(tp, fp, fn), 4), kappa=round(kappa(g, d), 4))


def main():
    t = pd.read_csv(PROC / "24_deepseek_labels.csv")
    t["gpt"] = t.gpt_label.map(c24.norm)
    t["ds"] = t.deepseek_label.map(c24.norm)
    t = t[t.gpt != "NA"].copy()
    say(f"Resolutions with both labels: {len(t)}")
    say(f"Raw agreement on the exact label (as in card 24): {(t.gpt == t.ds).mean():.3f}")

    rows = []
    gi, di = t.gpt.str.contains("Israel"), t.ds.str.contains("Israel")
    rows.append(binary_row("Israel criticized", gi, di))
    ga, da = t.gpt.ne("None"), t.ds.ne("None")
    rows.append(binary_row("Any country criticized", ga, da))

    # country level: (resolution, country) pairs
    def sets(col):
        return t[col].map(lambda s: set() if s == "None" else {p.strip() for p in s.split(",")})
    gs, ds = sets("gpt"), sets("ds")
    tp = sum(len(a & b) for a, b in zip(gs, ds))
    fp = sum(len(b - a) for a, b in zip(gs, ds))
    fn = sum(len(a - b) for a, b in zip(gs, ds))
    rows.append(dict(measure="Same country named (resolution-country pairs)", gpt_positive=tp + fn,
                     deepseek_positive=tp + fp, both=tp, deepseek_only=fp, gpt_only=fn,
                     f1=round(f1(tp, fp, fn), 4)))

    # most often named countries
    counts = pd.Series([c for s in gs for c in s]).value_counts()
    for c in [c for c in counts.index[:8] if c != "Israel"]:
        g = gs.map(lambda s: c in s)
        d = ds.map(lambda s: c in s)
        rows.append(binary_row(f"{c} criticized", g, d))

    res = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / "39_f1_scores.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        say(res.to_string(index=False))
    LOG_F = OUT / "39_run_log.txt"
    LOG_F.write_text("\n".join(LOG) + "\n", encoding="utf-8")
    say(f"\nwritten {OUT / '39_f1_scores.csv'} and {LOG_F}")


if __name__ == "__main__":
    main()
