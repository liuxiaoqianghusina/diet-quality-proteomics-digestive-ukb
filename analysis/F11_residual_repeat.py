"""Items 7 and 8.
(7) Diet-specific (residual) signatures: each diet exposure is residualised (OLS fitted in the training set) on age, sex,
    ethnicity, Townsend, education, smoking, alcohol, physical activity, BMI, waist, ALT, GGT, HbA1c, triglycerides,
    cystatin C, creatinine, CRP, energy, number of recalls, fasting time and centre; an elastic net is trained on the
    residual, evaluated in the test set and related to outcomes in the independent sample (M1, M3).
(8) Repeat proteomics at the imaging visits (olink2 = instance 2, olink3 = instance 3): within-person correlation of
    intestinal proteins and of 'reduced' signatures retrained on the proteins measured at the repeat visits.
"""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.linear_model import ElasticNetCV

from common import (CACHE, EXPOSURES, NEGATIVE, OLINK, PRIMARY, RES, add_kidney_crp, add_metabolic, base_covariates,
                    cox, model_matrix, outcome_defs)

OUTS = PRIMARY + ["MASLD", "Cirrhosis"] + NEGATIVE
EPI = ["guca2a", "ddc", "fabp2", "fabp6", "cdh17", "dsg2"]
person = pd.read_pickle(CACHE / "person.pkl")
sig = pd.read_pickle(CACHE / "signatures.pkl")
P = pd.read_pickle(CACHE / "olink_qc_plateadj.pkl")
o = person[person.olink_qc].set_index("eid").loc[P.index].reset_index().merge(sig, on="eid", how="left")
train, test = o.train.to_numpy(), o.test.to_numpy()
X = P.to_numpy()
ind = ((o.consortium == 0) & ~o.train).to_numpy()


def fit_signature(y, cols_mask=None, ok=None):
    Xm = X if cols_mask is None else X[:, cols_mask]
    ok = train & np.isfinite(y) if ok is None else ok
    en = ElasticNetCV(l1_ratio=0.5, alphas=20, cv=5, max_iter=5000, n_jobs=-1, random_state=1).fit(Xm[ok], y[ok])
    pred = en.predict(Xm)
    return (pred - pred.mean()) / pred.std(), en


def cox_block(d, xs, label):
    defs = outcome_defs(d, OUTS)
    Zs = {m: model_matrix(d, m).to_numpy() for m in ("M1", "M3")}
    jobs = [(m, oc, x) for m in Zs for oc in OUTS for x in xs]
    res = Parallel(n_jobs=6)(delayed(cox)(d[x].to_numpy(), Zs[m], *defs[oc], d["baseline_date"]) for m, oc, x in jobs)
    return pd.DataFrame([{"analysis": label, "model": m, "outcome": oc, "x": x, **v} for (m, oc, x), v in zip(jobs, res)])


# ================================================================== (7) residual signatures
C = add_metabolic(add_kidney_crp(base_covariates(o, diet=False), o), o)
for c in ["energy_kj", "n_recalls"]:
    C[c] = o[c].fillna(o[c].median())
Cm = np.column_stack([np.ones(len(o)), C.to_numpy()])
perf = []
for e in EXPOSURES:
    y = o[e].to_numpy()
    okt = train & np.isfinite(y)
    beta = np.linalg.lstsq(Cm[okt], y[okt], rcond=None)[0]
    resid = y - Cm @ beta
    o[f"rsig_{e}"], en = fit_signature(resid)
    tt = test & np.isfinite(resid)
    r_res = np.corrcoef(o.loc[tt, f"rsig_{e}"], resid[tt])[0, 1]
    perf.append({"signature": e, "n_proteins": int((en.coef_ != 0).sum()), "r_test_residual_diet": r_res,
                 "r_with_original_signature": np.corrcoef(o.loc[ind, f"rsig_{e}"], o.loc[ind, f"sig_{e}"])[0, 1],
                 "r_with_BMI_signature": np.corrcoef(o.loc[ind, f"rsig_{e}"], o.loc[ind, "sig_BMI"])[0, 1],
                 "r2_diet_explained_by_covariates": 1 - np.var(resid[okt]) / np.var(y[okt])})
    pd.Series(en.coef_, index=P.columns).loc[lambda s: s != 0].to_csv(RES / f"residual_signature_weights_{e}.csv")
    print(perf[-1], flush=True)
pd.DataFrame(perf).to_csv(RES / "residual_signature_performance.csv", index=False)
d_ind = o[ind].reset_index(drop=True)
r7 = cox_block(d_ind, [f"rsig_{e}" for e in EXPOSURES], "diet-specific residual signature")
r7.to_csv(RES / "residual_signature_cox.csv", index=False)
fmt = lambda s: f"{s.hr:.2f} ({s.lci:.2f}-{s.uci:.2f}){'*' if (s.lci > 1 or s.uci < 1) else ''}"
for m in ("M1", "M3"):
    print(f"\n== residual signatures {m}\n" + r7[r7.model == m].assign(c=lambda t: t.apply(fmt, axis=1))
          .pivot(index="outcome", columns="x", values="c").loc[OUTS].to_string())

# ================================================================== (8) repeat proteomics
D = OLINK.parent
rep = {}
for inst, f in [(2, "olink_instance_2.csv"), (3, "olink_instance_3.csv")]:
    r = pd.read_csv(D / f).rename(columns={"Participant.ID": "eid"}).set_index("eid")
    rep[inst] = r
    print(f"instance {inst}: {r.shape[0]} participants, {r.shape[1]} proteins; overlap with baseline QC sample:",
          len(r.index.intersection(P.index)), flush=True)
common_prot = [p for p in P.columns if p in rep[2].columns and p in rep[3].columns]
mask = np.isin(P.columns, common_prot)
extra = pd.read_pickle(CACHE / "extra_tier1.pkl").set_index("eid")
stab = []
for e in EXPOSURES:
    o[f"red_{e}"], en = fit_signature(o[e].to_numpy(), cols_mask=mask)
    w = pd.Series(en.coef_, index=np.array(P.columns)[mask])
    tt = test & np.isfinite(o[e].to_numpy())
    stab.append({"signature": e, "type": "reduced signature (repeat-panel proteins)", "n_proteins": int((w != 0).sum()),
                 "r_test": np.corrcoef(o.loc[tt, f"red_{e}"], o.loc[tt, e])[0, 1],
                 "r_with_full_signature": np.corrcoef(o.loc[ind, f"red_{e}"], o.loc[ind, f"sig_{e}"])[0, 1]})
    for inst, r in rep.items():
        ids = r.index.intersection(P.index)
        Rz = r.loc[ids, common_prot]
        Rz = ((Rz - Rz.mean()) / Rz.std()).fillna(0)
        follow = Rz[w.index].to_numpy() @ w.to_numpy()
        base = o.set_index("eid").loc[ids, f"red_{e}"].to_numpy()
        yrs = (pd.to_datetime(extra.loc[ids, f"p53_i{inst}"]) - o.set_index("eid").loc[ids, "baseline_date"]).dt.days / 365.25
        stab[-1][f"spearman_baseline_vs_instance{inst}"] = stats.spearmanr(base, follow).correlation
        stab[-1][f"n_instance{inst}"] = len(ids)
        stab[-1][f"years_to_instance{inst}"] = float(yrs.median())
prot_rows = []
for p in EPI:
    row = {"signature": p.upper(), "type": "single protein"}
    for inst, r in rep.items():
        if p not in r.columns:
            continue
        ids = r.index.intersection(P.index)
        row[f"spearman_baseline_vs_instance{inst}"] = stats.spearmanr(P.loc[ids, p], r.loc[ids, p], nan_policy="omit").correlation
        row[f"n_instance{inst}"] = len(ids)
    prot_rows.append(row)
st = pd.concat([pd.DataFrame(stab), pd.DataFrame(prot_rows)])
st.to_csv(RES / "repeat_stability.csv", index=False)
print(st.round(3).to_string(index=False))
r8 = cox_block(o[ind].reset_index(drop=True), [f"red_{e}" for e in EXPOSURES], "reduced signature")
r8.to_csv(RES / "reduced_signature_cox.csv", index=False)
print("\n== reduced signatures M1\n" + r8[r8.model == "M1"].assign(c=lambda t: t.apply(fmt, axis=1))
      .pivot(index="outcome", columns="x", values="c").loc[PRIMARY].to_string())
