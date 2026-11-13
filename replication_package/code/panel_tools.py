"""
panel_tools.py

Small, self-contained panel regression helpers written with numpy and pandas
only. Written on 28 October 2026 because the sandbox used for the revision
has no network access and the linearmodels package used by the September
scripts cannot be installed there. Every estimator here is checked against
the paper's published numbers in code/19_corrected_tables.py (Table 2 from
Stata xtreg, Table 3 from Yair's linearmodels notebook).

Estimators
  ols(y, X, cluster)            pooled OLS with cluster-robust standard errors
  fixed_effects(df, y, X, ...)  one-way entity effects (within transformation),
                                optionally with year dummies (two-way)
  random_effects(df, y, X, ...) Swamy-Arora random effects (feasible GLS), the
                                same recipe as Stata's xtreg, re and
                                linearmodels' RandomEffects
  between_effects(df, y, X)     OLS on entity means

Conventions
  df must contain an entity column and a time column. Rows with a missing
  value in y or X are dropped by the caller.
  Cluster-robust standard errors use the finite-sample factor
  G/(G-1) * (N-1)/(N-K), as Stata does.

Requires: python 3.10+, numpy, pandas. No network access.
"""

import numpy as np
import pandas as pd


class Fit:
    """Holds a fitted model: params, se, and a few scalars."""

    def __init__(self, names, beta, se, nobs, resid=None, extra=None):
        self.params = pd.Series(beta, index=names)
        self.se = pd.Series(se, index=names)
        self.tstat = self.params / self.se
        self.nobs = nobs
        self.resid = resid
        self.extra = extra or {}

    def stars(self):
        from math import erf, sqrt
        out = {}
        for k, t in self.tstat.items():
            p = 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))
            out[k] = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
        return pd.Series(out)

    def table(self):
        return pd.DataFrame({"coef": self.params, "se": self.se, "t": self.tstat, "stars": self.stars()})


def _cluster_cov(X, u, groups, df_adjust=True):
    """Cluster-robust covariance (Liang-Zeger) with Stata's small-sample factor."""
    N, K = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    G = len(np.unique(groups))
    meat = np.zeros((K, K))
    order = np.argsort(groups, kind="stable")
    Xs, us, gs = X[order], u[order], groups[order]
    bounds = np.flatnonzero(np.r_[True, gs[1:] != gs[:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        s = Xs[a:b].T @ us[a:b]
        meat += np.outer(s, s)
    V = XtX_inv @ meat @ XtX_inv
    if df_adjust:
        V *= (G / (G - 1)) * ((N - 1) / (N - K))
    return V


def ols(y, X, names, cluster=None, robust=False):
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    u = y - X @ beta
    N, K = X.shape
    if cluster is not None:
        V = _cluster_cov(X, u, np.asarray(cluster))
    elif robust:
        XtX_inv = np.linalg.pinv(X.T @ X)
        V = XtX_inv @ (X.T @ (X * (u ** 2)[:, None])) @ XtX_inv * N / (N - K)
    else:
        V = np.linalg.pinv(X.T @ X) * (u @ u) / (N - K)
    return Fit(names, beta, np.sqrt(np.diag(V)), N, resid=u)


def _dummies(s, drop_first=True):
    d = pd.get_dummies(s.astype(str), prefix="yr", dtype=float)
    d = d.reindex(sorted(d.columns, key=lambda c: int(c.split("_")[1])), axis=1)
    return d.iloc[:, 1:] if drop_first else d


def _design(df, xvars, time_dummies=None, const=True):
    X = df[xvars].astype(float)
    names = list(xvars)
    if time_dummies is not None:
        D = _dummies(df[time_dummies])
        X = pd.concat([X, D], axis=1)
        names += list(D.columns)
    if const:
        X["const"] = 1.0
        names.append("const")
    return X.values, names


def fixed_effects(df, y, xvars, entity, time=None, cluster=True):
    """Within estimator. Entity effects absorbed by demeaning; year dummies added
    explicitly if `time` is given (this equals two-way absorption exactly).
    Standard errors clustered by entity. The degrees of freedom count the
    absorbed entity effects, as Stata's xtreg, fe does."""
    d = df.copy()
    Xdf = d[xvars].astype(float)
    names = list(xvars)
    if time is not None:
        D = _dummies(d[time])
        Xdf = pd.concat([Xdf, D], axis=1)
        names += list(D.columns)
    g = d[entity]
    Xw = Xdf.values - Xdf.groupby(g.values).transform("mean").values
    yw = d[y].astype(float).values
    yw = yw - pd.Series(yw).groupby(g.values).transform("mean").values
    beta = np.linalg.lstsq(Xw, yw, rcond=None)[0]
    u = yw - Xw @ beta
    N, K = Xw.shape
    n_ent = g.nunique()
    if cluster:
        V = _cluster_cov(Xw, u, g.values, df_adjust=False)
        V *= (n_ent / (n_ent - 1)) * ((N - 1) / (N - K))
    else:
        V = np.linalg.pinv(Xw.T @ Xw) * (u @ u) / (N - n_ent - K)
    sigma2_e = (u @ u) / (N - n_ent - K)
    # within R-squared
    r2_within = 1 - (u @ u) / (yw @ yw)
    return Fit(names, beta, np.sqrt(np.diag(V)), N, resid=u,
               extra=dict(sigma2_e=sigma2_e, r2_within=r2_within, n_entities=n_ent))


def between_effects(df, y, xvars, entity):
    m = df.groupby(entity)[[y] + list(xvars)].mean()
    X, names = _design(m, xvars)
    f = ols(m[y].values, X, names)
    f.extra["r2"] = 1 - (f.resid @ f.resid) / np.sum((m[y].values - m[y].mean()) ** 2)
    return f


def random_effects(df, y, xvars, entity, time=None, cluster=True):
    """Swamy-Arora feasible GLS.
    sigma2_e from the within regression; sigma2_u from the between regression
    (on entity means, including the constant and the same regressors), with
    T_bar the harmonic mean of the entity lengths. theta_i = 1 - sqrt(sigma2_e /
    (T_i sigma2_u + sigma2_e)). Quasi-demeaned OLS with cluster-robust SEs."""
    d = df.copy()
    X_full, names = _design(d, xvars, time_dummies=time, const=True)
    Xdf = pd.DataFrame(X_full, columns=names, index=d.index)
    yv = d[y].astype(float).values
    g = d[entity].values
    # within residual variance
    fe = fixed_effects(d, y, xvars, entity, time=time, cluster=False)
    sigma2_e = fe.extra["sigma2_e"]
    # between regression on entity means
    M = Xdf.groupby(g).mean()
    ym = pd.Series(yv, index=d.index).groupby(g).mean().values
    Mv = M.values
    bb = np.linalg.lstsq(Mv, ym, rcond=None)[0]
    ub = ym - Mv @ bb
    n_ent = len(M)
    K_b = np.linalg.matrix_rank(Mv)
    if n_ent <= K_b:
        raise ValueError(f"random effects: between regression has no degrees of freedom "
                         f"({n_ent} entities, {K_b} regressors incl. dummies)")
    s2_b = (ub @ ub) / (n_ent - K_b)
    T = pd.Series(1.0, index=d.index).groupby(g).size().values
    T_bar = len(T) / np.sum(1.0 / T)
    sigma2_u = max(0.0, s2_b - sigma2_e / T_bar)
    # quasi-demeaning
    Tmap = pd.Series(T, index=M.index)
    Ti = Tmap.loc[g].values
    theta = 1 - np.sqrt(sigma2_e / (Ti * sigma2_u + sigma2_e))
    Xm = Xdf.groupby(g).transform("mean").values
    ymean = pd.Series(yv, index=d.index).groupby(g).transform("mean").values
    Xq = X_full - theta[:, None] * Xm
    yq = yv - theta * ymean
    beta = np.linalg.lstsq(Xq, yq, rcond=None)[0]
    u = yq - Xq @ beta
    N, K = Xq.shape
    if cluster:
        V = _cluster_cov(Xq, u, g)
    else:
        V = np.linalg.pinv(Xq.T @ Xq) * (u @ u) / (N - K)
    # overall R-squared as Stata reports for re: squared correlation of y and Xb (untransformed)
    yhat = X_full @ beta
    r2_overall = np.corrcoef(yv, yhat)[0, 1] ** 2
    return Fit(names, beta, np.sqrt(np.diag(V)), N, resid=u,
               extra=dict(sigma2_e=sigma2_e, sigma2_u=sigma2_u, theta_mean=float(theta.mean()),
                          r2_overall=r2_overall, n_entities=n_ent))
