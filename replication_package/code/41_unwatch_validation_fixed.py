"""
41_unwatch_validation_fixed.py

Task card 41 (8 November 2026): audit of every number in the revised manuscript.
This script recomputes the validation of the GPT classification against UN Watch
(Section III.A of the paper, Figure 1, the Israel F1 score, the multiclass F1 score,
and the count of resolutions UN Watch codes as condemning a country).

Why it is needed. The validation in F1_score.ipynb (and in code/19_corrected_tables.py,
which copied it) reads UN Watch's codes from the "Country Concerned" and
"Condemnatory" columns that were merged into full_text_data2.csv. That merge
missed every emergency special session resolution, because UN Watch writes
"A/ES-10/21" and "A/RES/ES-11/1" while the Voelsen database writes "R/ES/10/21"
and "R/ES/11/1". Ten resolutions UN Watch codes as condemning a country (four on
Israel, six on Russia) therefore entered the validation as "condemns no country".
Here UN Watch's own file is merged on a normalised symbol instead.

Choices (stated, not hidden):
  * Window 2015-2023, as the paper states.
  * Truth = UN Watch "Condemnatory" = Yes (as decided on 30 Sep 2026 / card 19).
  * A resolution UN Watch codes "Israel, United States" (A/ES-10/19) counts as
    condemning Israel in the Israel task (rule "contains"); the published rule
    (exact match) is also reported.
  * The multiclass F1 follows F1_score.ipynb: sklearn-style weighted F1 over all
    labels including "None", with the notebook's country-name fixes.

Inputs:  Israel and the UN/Data/full_text_data2.csv
         Israel and the UN/Data/UN Watch GA Resolutions 2015-August 2024.xlsx
Outputs: output/tables/41_unwatch_validation.csv
         output/tables/41_unwatch_unmatched.csv
         output/figures/41_figure1_confusion_matrix.png
         output/tables/41_run_log.txt
Run:     python3 code/41_unwatch_validation_fixed.py   (from the project root)
Requires: python 3.10+, pandas, openpyxl, numpy, matplotlib. No network access.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Israel and the UN" / "Data"
OUT = ROOT / "output" / "tables"
FIG = ROOT / "output" / "figures" / "41_figure1_confusion_matrix.png"
LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(str(s))


def norm(sym):
    """A/RES/70/173, A/ES-10/21, R/ES/10/21 -> 70/173, ES/10/21."""
    s = str(sym).strip().replace(" ", "")
    s = re.sub(r"^(A/RES/|A/|R/)", "", s)
    return re.sub(r"^ES-(\d+)", r"ES/\1", s)


NAME_FIX = {"Russian Federation": "Russia", "Islamic Republic of Iran": "Iran", "Republic of Korea": "South Korea",
            "Democratic People's Republic of Korea": "North Korea", "Republic of Moldova": "Moldova",
            "United States of America": "United States", "Syrian Arab Republic": "Syria",
            "United Kingdom of Great Britain and Northern Ireland": "United Kingdom"}


def weighted_f1(truth, pred):
    """F1 per label, weighted by the label's count in truth (sklearn average='weighted')."""
    total = 0.0
    for lab in truth.unique():
        tp = int(((truth == lab) & (pred == lab)).sum())
        fp = int(((truth != lab) & (pred == lab)).sum())
        fn = int(((truth == lab) & (pred != lab)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        total += (2 * p * r / (p + r) if p + r else 0.0) * int((truth == lab).sum())
    return total / len(truth)


def binary(truth, pred):
    tp = int(((pred == 1) & (truth == 1)).sum()); fp = int(((pred == 1) & (truth == 0)).sum())
    fn = int(((pred == 0) & (truth == 1)).sum()); tn = int(((pred == 0) & (truth == 0)).sum())
    return dict(TP=tp, FP=fp, FN=fn, TN=tn, F1=2 * tp / (2 * tp + fp + fn))


def main():
    db = pd.read_csv(DATA / "full_text_data2.csv", usecols=["unres", "year", "country", "Country Concerned", "Condemnatory"],
                     low_memory=False)
    db["country"] = db["country"].fillna("None").str.strip()
    db = db[~db["country"].str.contains("Understood", case=False, na=False)].copy()
    db["country"] = db["country"].replace(NAME_FIX)
    db["k"] = db["unres"].map(norm)
    w = db[(db.year >= 2015) & (db.year <= 2023)].copy()
    say(f"Voelsen database, 2015-2023: {len(w)} resolutions ({w.k.nunique()} distinct symbols)")

    uw = pd.read_excel(DATA / "UN Watch GA Resolutions 2015-August 2024.xlsx")
    uw["cc"] = uw["Country Concerned"].astype(str).str.strip()
    uw["cond"] = uw["Condemnatory"].astype(str).str.strip().str.lower()
    uw["k"] = uw["Symbol"].map(norm)
    u = uw[uw.Year <= 2023]
    say(f"UN Watch file, 2015-2023: {len(u)} resolutions listed, {int((u.cond == 'yes').sum())} coded condemnatory "
        f"({int(((u.cond == 'yes') & u.cc.str.contains('Israel')).sum())} naming Israel)")

    miss = u[~u.k.isin(set(w.k))]
    say(f"UN Watch resolutions not in the Voelsen database: {len(miss)}: "
        + "; ".join(f"{r.Symbol} ({r.cc}, condemnatory={r.cond})" for r in miss.itertuples()))
    old_cc = w.set_index("k")["Country Concerned"]
    merged = u.merge(w[["k", "unres"]], on="k", how="inner")
    lost = merged[merged.k.map(old_cc).isna()]
    say(f"In the database but missing from its merged 'Country Concerned' column: {len(lost)}: "
        + "; ".join(f"{r.Symbol} ({r.cc})" for r in lost.itertuples()))
    lost.assign(reason="symbol format: emergency special session")[["Year", "Symbol", "unres", "cc", "cond", "reason"]] \
        .to_csv(OUT / "41_unwatch_unmatched.csv", index=False)

    cond = u[u.cond == "yes"].set_index("k")["cc"]
    w["uw_cond"] = w.k.map(cond).fillna("None")
    w["uw_any"] = w.k.map(u.set_index("k")["cc"]).fillna("None")
    say(f"Resolutions in the database that UN Watch codes condemnatory: {int((w.uw_cond != 'None').sum())}")

    rows = []
    pred = (w.country == "Israel").astype(int)
    old_truth = ((w["Country Concerned"].fillna("None").str.strip() == "Israel")
                 & (w["Condemnatory"].fillna("").astype(str).str.strip().str.lower() == "yes")).astype(int)
    for label, truth in [("as in the manuscript (merged columns, Condemnatory = Yes, exact 'Israel')", old_truth),
                         ("fixed merge, Condemnatory = Yes, exact 'Israel'", (w.uw_cond == "Israel").astype(int)),
                         ("fixed merge, Condemnatory = Yes, names Israel (incl. 'Israel, United States')",
                          w.uw_cond.str.contains("Israel").astype(int))]:
        b = binary(truth, pred)
        rows.append(dict(task="Israel", truth=label, unwatch_condemning=int(truth.sum()), **b))
    old_multi = np.where(w["Condemnatory"].fillna("").astype(str).str.strip().str.lower() == "yes",
                         w["Country Concerned"].fillna("None").str.strip(), "None")
    for label, truth in [("multiclass, as in the manuscript's data with Condemnatory = Yes", pd.Series(old_multi, index=w.index)),
                         ("multiclass, merged columns, no Condemnatory flag (the published .94)",
                          w["Country Concerned"].fillna("None").str.strip()),
                         ("multiclass, fixed merge, Condemnatory = Yes", w.uw_cond)]:
        rows.append(dict(task="all countries (weighted F1)", truth=label, F1=weighted_f1(truth, w.country)))
    t = pd.DataFrame(rows)
    t["F1"] = t["F1"].round(4)
    t.to_csv(OUT / "41_unwatch_validation.csv", index=False)
    say(t.to_string(index=False))

    r = t[t.truth.str.startswith("fixed merge, Condemnatory = Yes, names Israel")].iloc[0]
    draw(int(r.TN), int(r.FP), int(r.FN), int(r.TP))
    say(f"Figure 1 drawn from the 'names Israel' row -> {FIG.relative_to(ROOT)}")
    (OUT / "41_run_log.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")


def draw(tn, fp, fn, tp):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    im = ax.imshow(cm, cmap="Oranges")
    fig.colorbar(im, ax=ax)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "#7f2704")
    labels = ["Israel Not Condemned", "Israel Condemned"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("ChatGPT Predicted Classifications")
    ax.set_ylabel("UN Watch (True) Classifications")
    ax.set_title("Figure 1: Confusion Matrix for 2015-2023 UNGA Resolution\nIsrael Condemnation Classification")
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
