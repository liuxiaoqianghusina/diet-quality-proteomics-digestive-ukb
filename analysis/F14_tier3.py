"""Tier-3 analyses.
(10) Subgroups: sex, age (<60 / >=60 y), BMI (<25 / 25-<30 / >=30); HRs within strata (M1x) and P for interaction
     (Wald test of signature x subgroup product terms in the whole independent sample).
(11) Prediction of gallstone disease and IBD: Harrell's C of conventional risk factors with and without each signature,
     5-fold cross-validated linear predictors; 95% CI of the C-index increment from 500 bootstrap resamples.
(12) Isocaloric substitution (self-reported diet cohort): energy from NOVA 1 and from all other foods entered together
     (UPF omitted), so the NOVA 1 term estimates replacing 10% of energy from UPF with NOVA 1 foods.
(13) Exports data for Fine-Gray competing-risk models (death as competing event; F14b_finegray.R).
"""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from scipy import stats
from sklearn.model_selection import KFold

from common import CACHE, EXPOSURES, RES, cox, follow_up_end, model_matrix, outcome_defs
import importlib
F10 = importlib.import_module("F10_tier1")

SIGS = [f"sig_{e}" for e in EXPOSURES]
PRIMARY = ["Gallstones", "IBD", "Crohn", "UC", "Acute_pancreatitis", "AP_nonbiliary"]


def frame(d, Z, x, elig, ev, stop, extra=None):
    T = (pd.to_datetime(stop) - d["baseline_date"]).dt.days / 365.25
    f = pd.DataFrame(np.asarray(Z), index=d.index).add_prefix("z")
    f["x"] = d[x].to_numpy()
    if extra is not None:
        f = f.join(extra)
    f["T"], f["E"] = T, ev.astype(int)
    f = f[elig & (f["T"] > 0)]
    return f.loc[:, (f.std() > 0) | f.columns.isin(["T", "E"])]


def load_independent():
    person = pd.read_pickle(CACHE / "person.pkl").merge(pd.read_pickle(CACHE / "extra_tier1.pkl"), on="eid", how="left")
    sig = pd.read_pickle(CACHE / "signatures.pkl")
    d = person.merge(sig, on="eid", how="inner")
    return d[(d.consortium == 0) & ~d.train].reset_index(drop=True)


if __name__ == "__main__":
    d = load_independent()
    Zx = model_matrix(d, "M1").join(F10.extended(d))
    defs = outcome_defs(d, PRIMARY)

    # ---------------------------------------------------------------- (10) subgroups
    groups = {"Sex": pd.Series(np.where(d.sex == 1, "Men", "Women")),
              "Age": pd.Series(np.where(d.age >= 60, ">=60 years", "<60 years")),
              "BMI": pd.Series(pd.cut(d.bmi, [0, 25, 30, 100], right=False, labels=["<25", "25-<30", ">=30"]).astype(str))}
    rows = []
    for oc in ["Gallstones", "IBD", "Acute_pancreatitis"]:
        elig, ev, stop = defs[oc]
        for x in SIGS:
            for gname, g in groups.items():
                for level in sorted(g[g != "nan"].unique()):
                    m = (g == level).to_numpy()
                    dd = d[m]
                    r = cox(dd[x].to_numpy(), Zx[m].to_numpy(), elig[m], ev[m], stop[m], dd["baseline_date"])
                    rows.append({"outcome": oc, "x": x, "subgroup": gname, "level": level, **r})
                # interaction LR test
                ok = (g != "nan").to_numpy()
                dummies = pd.get_dummies(g[ok], drop_first=True, dtype=float)
                inter = dummies.mul(d.loc[ok, x].to_numpy(), axis=0).add_prefix("ix_")
                full = frame(d[ok], Zx[ok], x, elig[ok], ev[ok], stop[ok], dummies.add_prefix("g_").join(inter))
                fit = CoxPHFitter(penalizer=1e-4).fit(full, "T", "E")
                ix = [c for c in fit.params_.index if c.startswith("ix_")]
                b = fit.params_[ix].to_numpy(); V = fit.variance_matrix_.loc[ix, ix].to_numpy()
                p_int = stats.chi2.sf(float(b @ np.linalg.solve(V, b)), len(ix))     # Wald test of product terms
                for r in rows:
                    if r["outcome"] == oc and r["x"] == x and r["subgroup"] == gname:
                        r["p_interaction"] = p_int
    sg = pd.DataFrame(rows)
    sg.to_csv(RES / "tier3_subgroups.csv", index=False)
    print(sg.assign(c=sg.apply(lambda s: f"{s.hr:.2f} ({s.lci:.2f}-{s.uci:.2f})", axis=1))
          .pivot_table(index=["outcome", "subgroup", "level"], columns="x", values="c", aggfunc="first").to_string())
    print(sg.groupby(["outcome", "subgroup", "x"]).p_interaction.first().round(3).unstack().to_string())

    # ---------------------------------------------------------------- (11) prediction (5-fold CV C-index)
    conv = ["age", "sex", "white", "townsend", "bmi", "waist", "smoking", "alcohol_freq", "met_min", "tg", "hba1c"]
    C = pd.DataFrame(index=d.index)
    for c in conv:
        C[c] = d[c].fillna(d[c].median())
    C["log_met"] = np.log1p(C.pop("met_min"))
    C = C.join(F10.extended(d))
    kf = KFold(5, shuffle=True, random_state=1)
    pred_rows, boot_rows = [], []
    for oc in ["Gallstones", "IBD"]:
        elig, ev, stop = defs[oc]
        T = (pd.to_datetime(stop) - d["baseline_date"]).dt.days / 365.25
        keep = (elig & (T > 0)).to_numpy()
        base_X = C[keep].reset_index(drop=True)
        TT, EE = T[keep].to_numpy(), ev[keep].astype(int).to_numpy()
        sets = {"Conventional risk factors": base_X}
        for x in SIGS:
            sets[f"+ {x}"] = base_X.assign(sig=d.loc[keep, x].to_numpy())
        sets["+ all four signatures"] = base_X.join(d.loc[keep, SIGS].reset_index(drop=True))
        lp = {}
        for name, X in sets.items():
            out = np.zeros(len(X))
            for tr, te in kf.split(X):
                f = X.iloc[tr].assign(T=TT[tr], E=EE[tr])
                f = f.loc[:, (f.std() > 0) | f.columns.isin(["T", "E"])]
                m = CoxPHFitter(penalizer=1e-3).fit(f, "T", "E")
                out[te] = m.predict_partial_hazard(X.iloc[te][m.params_.index]).to_numpy().ravel()
            lp[name] = out
            pred_rows.append({"outcome": oc, "model": name, "C": concordance_index(TT, -out, EE)})
        rng = np.random.default_rng(1)
        base_lp = lp["Conventional risk factors"]
        for name in list(sets)[1:]:
            diffs = []
            for _ in range(500):
                i = rng.integers(0, len(TT), len(TT))
                diffs.append(concordance_index(TT[i], -lp[name][i], EE[i]) - concordance_index(TT[i], -base_lp[i], EE[i]))
            boot_rows.append({"outcome": oc, "model": name, "delta_C": np.mean(diffs),
                              "lci": np.percentile(diffs, 2.5), "uci": np.percentile(diffs, 97.5)})
    pr = pd.DataFrame(pred_rows).merge(pd.DataFrame(boot_rows), on=["outcome", "model"], how="left")
    pr.to_csv(RES / "tier3_prediction.csv", index=False)
    print("\n", pr.round(4).to_string(index=False))

    # ---------------------------------------------------------------- (12) isocaloric substitution
    person = pd.read_pickle(CACHE / "person.pkl").merge(pd.read_pickle(CACHE / "extra_tier1.pkl"), on="eid", how="left")
    dd = person[person.n_recalls >= 1].reset_index(drop=True)
    dd["nova1_10"] = dd["NOVA1_pct"] / 10
    dd["other_10"] = (100 - dd["UPF"] - dd["NOVA1_pct"]) / 10
    dd["upf_10"] = dd["UPF"] / 10
    Zd = model_matrix(dd, "M1", diet=True).join(F10.extended(dd))
    dd_defs = outcome_defs(dd, PRIMARY)
    sub = []
    for oc in PRIMARY:
        elig, ev, stop = dd_defs[oc]
        r1 = cox(dd["upf_10"].to_numpy(), Zd.to_numpy(), elig, ev, stop, dd["baseline_date"])
        sub.append({"outcome": oc, "analysis": "UPF, per 10% of energy", **r1})
        Zs = Zd.assign(other_10=dd["other_10"].to_numpy())
        r2 = cox(dd["nova1_10"].to_numpy(), Zs.to_numpy(), elig, ev, stop, dd["baseline_date"])
        sub.append({"outcome": oc, "analysis": "Replacing 10% of energy from UPF with NOVA 1 foods", **r2})
    sb = pd.DataFrame(sub)
    sb.to_csv(RES / "tier3_substitution.csv", index=False)
    print("\n", sb[["outcome", "analysis", "events", "n", "hr", "lci", "uci", "p"]].round(4).to_string(index=False))

    # ---------------------------------------------------------------- (13) export for Fine-Gray
    end = follow_up_end(d)
    out = pd.DataFrame(Zx.to_numpy(), columns=[f"z{i}" for i in range(Zx.shape[1])])
    out = out.loc[:, out.std() > 0]
    for x in SIGS:
        out[x] = d[x].to_numpy()
    for oc in PRIMARY:
        elig, ev, stop = defs[oc]
        T = (pd.to_datetime(stop) - d["baseline_date"]).dt.days / 365.25
        died = d["death_date"].notna() & (d["death_date"] <= stop) & ~ev.astype(bool) & (stop == d["death_date"])
        out[f"T_{oc}"] = T.to_numpy()
        out[f"S_{oc}"] = np.where(ev, 1, np.where(died, 2, 0))
        out[f"elig_{oc}"] = (elig & (T > 0)).astype(int).to_numpy()
    out.to_csv(CACHE / "finegray_input.csv", index=False)
    print("exported Fine-Gray input", out.shape)
