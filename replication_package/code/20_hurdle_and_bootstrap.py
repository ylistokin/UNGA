"""
20_hurdle_and_bootstrap.py

Task card 20 (28 October 2026), two parts, both for the editor (Engel):

Part A. A hurdle model of UNGA criticism, as a robustness check next to Table 3.
  Stage 1: was the country criticised at all in that year (share > 0)? A logit
           with the eight explanatory variables and year dummies, estimated on
           the non-Israel rows. Pooled (no country effects), because a country
           that is never criticised would drop out of a fixed effects logit.
  Stage 2: given that it was criticised, what share did it get? The random
           effects model of Table 2 column 4 (year effects, Security Council).
  Decomposition: the two stages together predict, for any set of
  characteristics, an expected share = probability of criticism x expected
  share if criticised. The explained part of the gap is the average predicted
  share of the other countries minus Israel's average predicted share, both
  computed with the other countries' coefficients. What is left of the actual
  gap is unexplained. This is the usual way to decompose with a non-linear
  model; it cannot be split variable by variable the way Table 3 is, so the
  script instead reports how much the explained part changes when one
  variable at a time is set to the others' average for Israel.
  Standard errors: 500 bootstrap rounds, countries resampled with
  replacement, Israel held fixed, both stages re-estimated each round.
  Spec approved by Matanel on 28 October 2026 (decision D13).

Part B. What the bootstrap resamples (Engel's question), on the corrected data.
  For rows 1 and 2 of Table 3 (corrected method, count variant A): standard
  errors from resampling countries only (the paper), years only, and countries
  and years together. Memo 02 did this on the published data.

Inputs: the same as code/19_corrected_tables.py (the script imports it and
rebuilds the corrected panel, count variant A, from the raw count files and
final_data.dta). Variant A is used because it is the mechanical fix; if
Matanel chooses variant B the script takes --variant B.

Outputs: output/tables/20_hurdle_points.csv, 20_hurdle_logit.csv,
  20_boot_rounds.csv (appended by chunks), 20_hurdle_bootstrap.csv,
  20_scheme_bootstrap.csv, 20_run_log_*.txt

Run from the project root:
    python3 code/20_hurdle_and_bootstrap.py                      (point estimates, 1 minute)
    bash code/20_run_bootstrap.sh                                (all chunks, about 25 minutes)
    python3 code/20_hurdle_and_bootstrap.py --stage summary

Requires: python 3.10+, numpy, pandas. No network access.
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
import importlib  # noqa: E402

c19 = importlib.import_module("19_corrected_tables")
from panel_tools import random_effects, fixed_effects, _dummies  # noqa: E402

OUT = ROOT / "output" / "tables"
V8 = c19.V8
LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(str(s))


# ---------------------------------------------------------------------------
# Logit by Newton's method with step halving (no scipy in the sandbox)
# ---------------------------------------------------------------------------
def logit_fit(X, y, max_iter=100, tol=1e-8, ridge=1e-8):
    """Returns (beta, converged). A tiny ridge keeps the Hessian invertible when
    a dummy separates the outcome; such cases are reported, not hidden."""
    n, k = X.shape
    b = np.zeros(k)
    ll_old = -np.inf
    conv = False
    for it in range(max_iter):
        eta = X @ b
        p = 1 / (1 + np.exp(-eta))
        ll = np.sum(y * np.log(p + 1e-300) + (1 - y) * np.log(1 - p + 1e-300))
        g = X.T @ (y - p) - ridge * b
        W = p * (1 - p)
        H = (X * W[:, None]).T @ X + ridge * np.eye(k)
        step = np.linalg.solve(H, g)
        # step halving so the likelihood never falls
        t = 1.0
        while t > 1e-4:
            bn = b + t * step
            pn = 1 / (1 + np.exp(-(X @ bn)))
            lln = np.sum(y * np.log(pn + 1e-300) + (1 - y) * np.log(1 - pn + 1e-300))
            if lln >= ll - 1e-10:
                break
            t /= 2
        b = bn
        if abs(lln - ll_old) < tol * (1 + abs(lln)):
            conv = True
            break
        ll_old = lln
    return b, conv


def design(df, V, years):
    X = df[V].astype(float).values
    D = np.column_stack([(df["year"].values == y).astype(float) for y in years[1:]])
    return np.column_stack([X, D, np.ones(len(df))])


def hurdle(D, V):
    """Fit both stages on the non-Israel rows and decompose. Returns a dict."""
    A = D[D.Israel_dummy == 0].dropna(subset=V + ["share"])
    B = D[D.Israel_dummy == 1].dropna(subset=V + ["share"])
    years = sorted(D.year.unique())
    XA, XB = design(A, V, years), design(B, V, years)
    yA = (A.share > 0).astype(float).values
    b1, conv = logit_fit(XA, yA)
    pA, pB = 1 / (1 + np.exp(-(XA @ b1))), 1 / (1 + np.exp(-(XB @ b1)))
    P = A[A.share > 0]
    f2 = random_effects(P, "share", V, "cid", time="year", cluster=False)
    names = list(V) + [f"yr_{y}" for y in years[1:]] + ["const"]
    # the RE design's dummy names come from _dummies; align by position
    yr_cols = [c for c in f2.params.index if c.startswith("yr_")]
    if yr_cols != [f"yr_{y}" for y in years[1:]]:
        raise ValueError("stage 2 year dummies do not line up with the panel's years")
    b2 = np.r_[f2.params[V].values, f2.params[yr_cols].values, f2.params["const"]]
    mA, mB = XA @ b2, XB @ b2
    yhatA, yhatB = (pA * mA).mean(), (pB * mB).mean()
    out = dict(overall=float(A.share.mean() - B.share.mean()), explained=float(yhatA - yhatB),
               stage1_prob_others=float(pA.mean()), stage1_prob_israel=float(pB.mean()),
               stage1_actual_others=float(yA.mean()), stage1_actual_israel=float((B.share > 0).mean()),
               stage2_pred_others_if_pos=float(mA[yA == 1].mean()), stage2_pred_israel=float(mB.mean()),
               stage2_actual_others_if_pos=float(P.share.mean()), stage2_actual_israel=float(B.share.mean()),
               fit_check_others=float(yhatA - A.share.mean()), logit_converged=bool(conv))
    # one variable at a time: give Israel the others' average on that variable
    for v in V:
        XB2 = XB.copy()
        j = V.index(v)
        XB2[:, j] = A[v].mean()
        pB2, mB2 = 1 / (1 + np.exp(-(XB2 @ b1))), XB2 @ b2
        out[f"explained_if_{v}_at_others_mean"] = float(yhatA - (pB2 * mB2).mean())
    out["_b1"] = b1
    out["_names"] = names
    return out


# ---------------------------------------------------------------------------
# Bootstrap draws (countries, years, both), as in memo 02
# ---------------------------------------------------------------------------
def draw(D, scheme, rng):
    D = D.reset_index(drop=True)
    isr = int(D.loc[D.Israel_dummy == 1, "cid"].iloc[0])
    cs = [c for c in D.cid.unique() if c != isr]
    ys = sorted(D.year.unique())
    C = rng.choice(cs, len(cs)) if scheme in ("country", "both") else cs
    Y = rng.choice(ys, len(ys)) if scheme in ("year", "both") else ys
    g = D.groupby(["cid", "year"]).indices
    rows, eid, tid = [], [], []
    for k, c in enumerate(list(C) + [isr]):
        for j, y in enumerate(Y):
            idx = g.get((c, y))
            if idx is not None:
                rows.append(idx[0]); eid.append(k); tid.append(ys[j] if scheme == "country" else j)
    S = D.iloc[rows].copy()
    S["cid"] = eid
    S["year"] = tid  # a year drawn twice becomes two separate "years"
    return S


def rounds_hurdle(D, V, seeds):
    path = OUT / "20_boot_rounds.csv"
    rows = []
    for s in seeds:
        S = draw(D, "country", np.random.default_rng(s))
        try:
            o = hurdle(S, V)
            rows.append(dict(part="hurdle", scheme="country", spec="hurdle", seed=s, explained=o["explained"],
                             stage1=o["stage1_prob_others"] - o["stage1_prob_israel"],
                             stage2=o["stage2_pred_others_if_pos"] - o["stage2_pred_israel"],
                             converged=o["logit_converged"], failed=""))
        except Exception as ex:
            rows.append(dict(part="hurdle", scheme="country", spec="hurdle", seed=s, explained=np.nan,
                             stage1=np.nan, stage2=np.nan, converged=False, failed=str(ex)))
    out = pd.DataFrame(rows)
    out.to_csv(path, mode="a", header=not path.exists(), index=False)
    say(f"hurdle seeds {seeds[0]}-{seeds[-1]}: ok {out.explained.notna().sum()}, not converged {(~out.converged.astype(bool)).sum()}")


def rounds_scheme(D, V, spec, scheme, seeds):
    path = OUT / "20_boot_rounds.csv"
    model = "re_year" if spec == "re" else "fe"
    rows = []
    for s in seeds:
        S = draw(D, scheme, np.random.default_rng(s))
        try:
            rows.append(dict(part="scheme", scheme=scheme, spec=spec, seed=s, explained=c19.explained(S, model, V)[0],
                             stage1=np.nan, stage2=np.nan, converged=True, failed=""))
        except Exception as ex:
            rows.append(dict(part="scheme", scheme=scheme, spec=spec, seed=s, explained=np.nan,
                             stage1=np.nan, stage2=np.nan, converged=True, failed=str(ex)))
    out = pd.DataFrame(rows)
    out.to_csv(path, mode="a", header=not path.exists(), index=False)
    say(f"{spec} {scheme} seeds {seeds[0]}-{seeds[-1]}: ok {out.explained.notna().sum()}, failed {out.explained.isna().sum()}")


def summary(D, V):
    r = pd.read_csv(OUT / "20_boot_rounds.csv").drop_duplicates(["part", "scheme", "spec", "seed"])
    # hurdle
    h = r[r.part == "hurdle"]
    pt = hurdle(D, V)
    rows = []
    for k, col in [("explained", "explained"), ("stage 1: probability of any criticism", "stage1"),
                   ("stage 2: share if criticised", "stage2")]:
        ok = h[col].dropna().values
        p = {"explained": pt["explained"], "stage1": pt["stage1_prob_others"] - pt["stage1_prob_israel"],
             "stage2": pt["stage2_pred_others_if_pos"] - pt["stage2_pred_israel"]}[col]
        rows.append(dict(quantity=k, point=round(p, 4), boot_se=round(ok.std(ddof=1), 4), ratio=round(p / ok.std(ddof=1), 2),
                         pct2_5=round(np.percentile(ok, 2.5), 4), pct97_5=round(np.percentile(ok, 97.5), 4),
                         rounds_ok=len(ok), rounds_failed=int(h[col].isna().sum()),
                         rounds_logit_not_converged=int((~h.converged.astype(bool)).sum())))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "20_hurdle_bootstrap.csv", index=False)
    say("\n=== Hurdle model, bootstrap (500 rounds, countries) ===")
    say(t.to_string(index=False))
    # schemes
    sc = r[r.part == "scheme"]
    rows = []
    for (spec, scheme), g in sc.groupby(["spec", "scheme"]):
        model = "re_year" if spec == "re" else "fe"
        p, od, _ = c19.explained(D, model, V)
        ok = g.explained.dropna().values
        rows.append(dict(row=("Row 1 RE" if spec == "re" else "Row 2 FE"), resample=scheme, explained=round(p, 4),
                         overall=round(od, 4), boot_se=round(ok.std(ddof=1), 4), ratio=round(p / ok.std(ddof=1), 2),
                         pct2_5=round(np.percentile(ok, 2.5), 4), pct97_5=round(np.percentile(ok, 97.5), 4),
                         rounds_ok=len(ok), rounds_failed=int(g.explained.isna().sum())))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "20_scheme_bootstrap.csv", index=False)
    say("\n=== What is resampled: countries, years, or both (corrected data, variant A) ===")
    say(t.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="points", choices=["points", "boot_hurdle", "boot_scheme", "summary"])
    ap.add_argument("--variant", default="A")
    ap.add_argument("--spec", default="re")
    ap.add_argument("--scheme", default="country")
    ap.add_argument("--seeds", default="0:500")
    a = ap.parse_args()
    say("Card 20: hurdle model and bootstrap schemes. Run " + time.strftime("%Y-%m-%d %H:%M"))
    allc = c19.rebuild_counts()
    panels = c19.build_panels(allc)
    D = panels[f"corrected_{a.variant}"]
    lo, hi = [int(x) for x in a.seeds.split(":")]
    if a.stage == "points":
        pt = hurdle(D, V8)
        b1, names = pt.pop("_b1"), pt.pop("_names")
        say("\n=== Hurdle model, point estimates (corrected data, variant %s) ===" % a.variant)
        for k, v in pt.items():
            say(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
        pd.DataFrame([pt]).T.rename(columns={0: "value"}).to_csv(OUT / "20_hurdle_points.csv")
        lg = pd.DataFrame({"variable": names, "logit_coef": b1})
        lg[lg.variable.isin(V8 + ["const"])].to_csv(OUT / "20_hurdle_logit.csv", index=False)
        say("\nStage 1 logit coefficients (log-odds of any criticism):")
        say(lg[lg.variable.isin(V8 + ["const"])].round(4).to_string(index=False))
        (OUT / "20_run_log_points.txt").write_text("\n".join(LOG))
    elif a.stage == "boot_hurdle":
        rounds_hurdle(D, V8, list(range(lo, hi)))
    elif a.stage == "boot_scheme":
        rounds_scheme(D, V8, a.spec, a.scheme, list(range(lo, hi)))
    else:
        summary(D, V8)
        (OUT / "20_run_log_summary.txt").write_text("\n".join(LOG))
