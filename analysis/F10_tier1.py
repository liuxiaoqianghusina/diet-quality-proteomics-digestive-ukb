"""Tier-1 robustness analyses for the signature -> outcome associations (independent signature sample).
(1) Extended adjustment for gallstone-related confounders: HRT, oral contraceptives, parity, cholesterol-lowering,
    antihypertensive and insulin treatment, diabetes, weight change (M1x = M1 + these; M3x = M3 + these).
(2) Alternative gallstone definitions: first occurrence (main), first occurrence or cholecystectomy (OPCS J18),
    hospital-diagnosed (inpatient K80 only), cholecystectomy only.
(3) Dose-response: quintiles (HR, incidence per 1,000 person-years, 10-year cumulative incidence, P for trend)
    and restricted cubic splines (4 df; P for non-linearity by likelihood-ratio test).
(4) Missing covariates: complete-case analysis and multiple imputation (5 imputations, Rubin's rules); E-values.
"""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lifelines import CoxPHFitter, KaplanMeierFitter
from patsy import dmatrix
from scipy import stats
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

from common import (CACHE, CENSOR, EXPOSURES, PC, RES, base_covariates, cox, follow_up_end, model_matrix,
                    outcome_defs)

SIGS = [f"sig_{e}" for e in EXPOSURES]
GS = ["Gallstones", "GS_or_chole", "GS_hosp", "Cholecystectomy"]
PRIMARY = ["IBD", "Crohn", "UC", "Gallstones", "Acute_pancreatitis", "AP_nonbiliary"]


def num(s):
    return pd.to_numeric(s, errors="coerce")


def has_code(s, code):
    return s.fillna("").astype(str).str.contains(rf"(?:^|\D){code}(?:\D|$)")


def extended(d):
    """Gallstone-related confounders; women-only items set to 0 for men (absorbed by the sex term)."""
    X = pd.DataFrame(index=d.index)
    women = d["sex"] == 0
    for name, col in [("hrt", "p2814_i0"), ("ocp", "p2784_i0")]:
        v = num(d[col]).where(lambda s: s.isin([0, 1]))
        X[name] = np.where(women, v.fillna(-1), 0)
    births = num(d["p2734_i0"]).where(lambda s: s >= 0).clip(upper=3)
    X["parity"] = np.where(women, births.fillna(-1), 0)
    med = d["p6153_i0"].where(women, d["p6177_i0"])
    X["chol_med"], X["bp_med"], X["insulin"] = [has_code(med, c).astype(float) for c in (1, 2, 3)]
    X["diabetes"] = num(d["p2443_i0"]).where(lambda s: s.isin([0, 1])).fillna(-1)
    X["weight_change"] = num(d["p2306_i0"]).where(lambda s: s.isin([0, 2, 3])).fillna(-1)
    out = pd.DataFrame(index=d.index)
    for c in ["hrt", "ocp", "parity", "diabetes", "weight_change"]:
        out = out.join(pd.get_dummies(X[c].astype(int), prefix=c, drop_first=True, dtype=float))
    return out.join(X[["chol_med", "bp_med", "insulin"]])


def hospital_k80(eids):
    o = pd.read_pickle(PC / "outcomes.pkl")
    date_cols = sorted([c for c in o.columns if c.startswith("p41280_a")], key=lambda x: int(x.split("_a")[1]))
    codes = o["p41270"].fillna("").str.split("|").explode().to_frame("code")
    codes["pos"] = codes.groupby(level=0).cumcount()
    hit = codes[codes["code"].str.startswith("K80")]
    vals = [o.at[i, date_cols[p]] if p < len(date_cols) else None for i, p in zip(hit.index, hit["pos"])]
    s = pd.Series(pd.to_datetime(vals, errors="coerce"), index=hit.index).groupby(level=0).min()
    return pd.Series(s.reindex(o.index).to_numpy(), index=o["eid"].astype(int).to_numpy()).reindex(eids).to_numpy()


def evalue(hr, lo, hi):
    def e(r):
        r = 1 / r if r < 1 else r
        return r + np.sqrt(r * (r - 1))
    bound = lo if hr >= 1 else hi
    return e(hr), (1.0 if (lo <= 1 <= hi) else e(bound))


if __name__ == "__main__":
    person = pd.read_pickle(CACHE / "person.pkl").merge(pd.read_pickle(CACHE / "extra_tier1.pkl"), on="eid", how="left")
    sig = pd.read_pickle(CACHE / "signatures.pkl")
    d = person.merge(sig, on="eid", how="inner")
    d = d[(d.consortium == 0) & ~d.train].reset_index(drop=True)
    d["Cholecystectomy"] = d["cholecystectomy_date"]
    d["GS_or_chole"] = d[["Gallstones", "cholecystectomy_date"]].min(axis=1)
    d["GS_hosp"] = pd.to_datetime(hospital_k80(d.eid))
    print("n =", len(d))
    Xext = extended(d)
    Z = {"M1": model_matrix(d, "M1"), "M3": model_matrix(d, "M3")}
    Z["M1x"] = Z["M1"].join(Xext)
    Z["M3x"] = Z["M3"].join(Xext)
    outs = PRIMARY + GS[1:]
    defs = outcome_defs(d, outs)

    # ------------------------------------------------ (1)-(2) extended adjustment and gallstone definitions
    jobs = [(m, o, x) for m in ["M1", "M1x", "M3x"] for o in outs for x in SIGS]
    res = Parallel(n_jobs=6)(delayed(cox)(d[x].to_numpy(), Z[m].to_numpy(), *defs[o], d["baseline_date"]) for m, o, x in jobs)
    r1 = pd.DataFrame([{"model": m, "outcome": o, "x": x, **v} for (m, o, x), v in zip(jobs, res)])
    ev = r1.apply(lambda s: evalue(s.hr, s.lci, s.uci), axis=1)
    r1["evalue_point"], r1["evalue_ci"] = [v[0] for v in ev], [v[1] for v in ev]
    r1.to_csv(RES / "tier1_extended_adjustment.csv", index=False)
    fmt = lambda s: f"{s.hr:.2f} ({s.lci:.2f}-{s.uci:.2f})"
    for m in ["M1", "M1x", "M3x"]:
        t = r1[r1.model == m].assign(c=lambda x: x.apply(fmt, axis=1)).pivot(index="outcome", columns="x", values="c")
        evn = r1[r1.model == m].groupby("outcome")["events"].first()
        print(f"\n== {m}\n" + t.loc[outs].join(evn).to_string())

    # ------------------------------------------------ (3) dose-response
    Zx = Z["M1x"].to_numpy()
    rows, curves = [], []
    for o in ["Gallstones", "IBD", "Acute_pancreatitis"]:
        elig, evt, stop = defs[o]
        T = (pd.to_datetime(stop) - d["baseline_date"]).dt.days / 365.25
        for x in SIGS:
            q = pd.qcut(d[x], 5, labels=False) + 1
            med = d.groupby(q)[x].median()
            f = pd.DataFrame(Zx, index=d.index).add_prefix("z")
            f = f.join(pd.get_dummies(q, prefix="q", drop_first=True, dtype=float))
            f["T"], f["E"] = T, evt.astype(int)
            keep = elig & (f["T"] > 0)
            ff = f[keep].loc[:, lambda t: (t.std() > 0) | t.columns.isin(["T", "E"])]
            s = CoxPHFitter(penalizer=1e-4).fit(ff, "T", "E").summary
            ft = f[[c for c in f.columns if not c.startswith("q_")]].assign(trend=q.map(med))[keep]
            ft = ft.loc[:, lambda t: (t.std() > 0) | t.columns.isin(["T", "E"])]
            p_trend = CoxPHFitter(penalizer=1e-4).fit(ft, "T", "E").summary.loc["trend", "p"]
            for k in range(1, 6):
                m_ = keep & (q == k)
                pyrs, nev = T[m_].sum(), int(evt[m_].sum())
                km = KaplanMeierFitter().fit(T[m_], evt[m_].astype(int))
                ci10 = 1 - km.predict(10)
                row = {"outcome": o, "x": x, "quintile": k, "median_sig": med[k], "events": nev, "person_years": pyrs,
                       "rate_per_1000py": 1000 * nev / pyrs, "cuminc_10y_%": 100 * ci10, "p_trend": p_trend}
                if k > 1:
                    row.update(hr=np.exp(s.loc[f"q_{k}", "coef"]), lci=np.exp(s.loc[f"q_{k}", "coef lower 95%"]),
                               uci=np.exp(s.loc[f"q_{k}", "coef upper 95%"]))
                else:
                    row.update(hr=1.0, lci=np.nan, uci=np.nan)
                rows.append(row)
            # restricted cubic spline (natural cubic regression spline, 4 df) vs linear
            B = dmatrix("cr(v, df=4) - 1", {"v": d[x]}, return_type="dataframe")
            design = B.design_info
            B.columns = [f"s{i}" for i in range(B.shape[1])]
            fs = pd.DataFrame(Zx, index=d.index).add_prefix("z").join(B)
            fs["T"], fs["E"] = T, evt.astype(int)
            fs = fs[keep].loc[:, lambda t: (t.std() > 0) | t.columns.isin(["T", "E"])]
            sp = CoxPHFitter(penalizer=1e-4).fit(fs, "T", "E")
            fl = pd.DataFrame(Zx, index=d.index).add_prefix("z").assign(lin=d[x])
            fl["T"], fl["E"] = T, evt.astype(int)
            fl = fl[keep].loc[:, lambda t: (t.std() > 0) | t.columns.isin(["T", "E"])]
            ln = CoxPHFitter(penalizer=1e-4).fit(fl, "T", "E")
            lr = 2 * (sp.log_likelihood_ - ln.log_likelihood_)
            p_nl = stats.chi2.sf(lr, B.shape[1] - 1)
            sc = [c for c in B.columns if c in fs.columns]
            grid = np.linspace(d[x].quantile(0.01), d[x].quantile(0.99), 60)
            Bg = pd.DataFrame(np.asarray(dmatrix(design, {"v": np.r_[grid, d[x].median()]})), columns=B.columns)
            diff = (Bg.iloc[:-1][sc] - Bg.iloc[-1][sc]).to_numpy()
            beta, cov = sp.params_[sc].to_numpy(), sp.variance_matrix_.loc[sc, sc].to_numpy()
            lh, se = diff @ beta, np.sqrt(np.einsum("ij,jk,ik->i", diff, cov, diff))
            curves.append(pd.DataFrame({"outcome": o, "x": x, "value": grid, "hr": np.exp(lh), "lci": np.exp(lh - 1.96 * se),
                                        "uci": np.exp(lh + 1.96 * se), "p_nonlinear": p_nl}))
    dr = pd.DataFrame(rows)
    dr.to_csv(RES / "tier1_dose_response.csv", index=False)
    pd.concat(curves).to_csv(RES / "tier1_spline_curves.csv", index=False)
    print("\n== quintiles (M1x)")
    print(dr.assign(c=dr.apply(lambda s: f"{s.hr:.2f}" + ("" if np.isnan(s.lci) else f" ({s.lci:.2f}-{s.uci:.2f})"), axis=1))
          .pivot_table(index=["outcome", "x"], columns="quintile", values="c", aggfunc="first").to_string())
    print(dr.groupby(["outcome", "x"])[["p_trend"]].first().to_string())
    print(pd.concat(curves).groupby(["outcome", "x"]).p_nonlinear.first().round(3).to_string())
    print(dr[dr.quintile.isin([1, 5])].pivot_table(index=["outcome", "x"], columns="quintile",
                                                     values=["rate_per_1000py", "cuminc_10y_%"]).round(2).to_string())

    # ------------------------------------------------ (4) complete-case and multiple imputation
    raw = {"townsend": d.townsend, "bmi": d.bmi, "met_min": d.met_min, "fasting_h": d.fasting_h, "cysc": d.cysc,
           "creat": d.creat, "crp": d.crp, "smoking": d.smoking, "alcohol_freq": d.alcohol_freq, "degree": d.degree,
           "white": d.white}
    miss = pd.DataFrame(raw).isna()
    print("\nmissing (%):", (100 * miss.mean()).round(1).to_dict(), "complete cases:", int((~miss.any(axis=1)).sum()))
    cc = ~miss.any(axis=1)
    dcc = d[cc].reset_index(drop=True)
    Zcc = model_matrix(dcc, "M1").join(extended(dcc)).to_numpy()
    defs_cc = outcome_defs(dcc, PRIMARY)
    jobs = [(o, x) for o in PRIMARY for x in SIGS]
    res = Parallel(n_jobs=6)(delayed(cox)(dcc[x].to_numpy(), Zcc, *defs_cc[o], dcc["baseline_date"]) for o, x in jobs)
    rcc = pd.DataFrame([{"analysis": "complete case", "outcome": o, "x": x, **v} for (o, x), v in zip(jobs, res)])

    imp_cols = list(raw) + ["age", "sex", "waist", "alt", "ggt", "hba1c", "tg"]
    base_mat = d[imp_cols].astype(float)
    pooled = []
    fits = {k: [] for k in jobs}
    for m in range(5):
        imp = IterativeImputer(sample_posterior=True, random_state=m, max_iter=10)
        filled = pd.DataFrame(imp.fit_transform(base_mat), columns=imp_cols)
        di = d.copy()
        for c in raw:
            v = filled[c]
            if c in ("smoking", "alcohol_freq", "degree", "white"):
                lo, hi = {"smoking": (0, 2), "alcohol_freq": (1, 6), "degree": (0, 1), "white": (0, 1)}[c]
                v = v.round().clip(lo, hi)
            if c in ("cysc", "creat", "crp", "met_min"):
                v = v.clip(lower=0.001)
            di[c] = v.to_numpy()
        Zi = model_matrix(di, "M1").join(extended(di)).to_numpy()
        res = Parallel(n_jobs=6)(delayed(cox)(di[x].to_numpy(), Zi, *defs[o], di["baseline_date"]) for o, x in jobs)
        for k, v in zip(jobs, res):
            fits[k].append(v)
    for (o, x), fl in fits.items():
        b = np.array([f["logHR"] for f in fl]); w = np.array([f["se"] ** 2 for f in fl])
        qbar, ubar, bvar = b.mean(), w.mean(), b.var(ddof=1)
        tvar = ubar + (1 + 1 / len(b)) * bvar
        se = np.sqrt(tvar)
        pooled.append({"analysis": "multiple imputation", "outcome": o, "x": x, "n": fl[0]["n"], "events": fl[0]["events"],
                       "hr": np.exp(qbar), "lci": np.exp(qbar - 1.96 * se), "uci": np.exp(qbar + 1.96 * se),
                       "p": 2 * stats.norm.sf(abs(qbar / se))})
    rmi = pd.concat([rcc, pd.DataFrame(pooled)])
    rmi.to_csv(RES / "tier1_missing_data.csv", index=False)
    print("\n== missing-data analyses (M1x)")
    print(rmi.assign(c=rmi.apply(fmt, axis=1)).pivot_table(index=["analysis", "outcome"], columns="x", values="c",
                                                           aggfunc="first").to_string())
    print("\n== E-values (M1x)")
    print(r1[(r1.model == "M1x") & r1.outcome.isin(PRIMARY)][["outcome", "x", "hr", "evalue_point", "evalue_ci"]]
          .round(2).to_string(index=False))
