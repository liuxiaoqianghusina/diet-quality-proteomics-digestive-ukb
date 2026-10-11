"""Extreme sensitivity analyses for the NOVA classification.
'upper': every item originally flagged as uncertain is classified as UPF; 'lower': none of them is UPF.
For each scenario: UPF (% energy) recomputed (data_prep/04_nova_items.py <scenario>), UPF proteomic signature
retrained (same training set and elastic-net settings), signature -> outcomes in the independent sample (M1, M3), and
self-reported UPF -> outcomes in the self-reported diet cohort (M1).
"""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import ElasticNetCV

from common import CACHE, PC, PRIMARY, RES, cox, model_matrix, outcome_defs

person = pd.read_pickle(CACHE / "person.pkl")
sig = pd.read_pickle(CACHE / "signatures.pkl")
P = pd.read_pickle(CACHE / "olink_qc_plateadj.pkl")
rows, info = [], []
for sc in ["main", "upper", "lower"]:
    u = pd.read_pickle(PC / ("upf_nova.pkl" if sc == "main" else f"upf_nova_{sc}.pkl"))[["eid", "UPF_nova"]]
    p = person.drop(columns=["UPF"]).merge(u.rename(columns={"UPF_nova": "UPF"}), on="eid", how="left")
    o = p[p.olink_qc].set_index("eid").loc[P.index].reset_index().merge(sig[["eid", "train", "test"]], on="eid")
    y = o["UPF"].to_numpy()
    ok = o.train.to_numpy() & np.isfinite(y)
    en = ElasticNetCV(l1_ratio=0.5, alphas=20, cv=5, max_iter=5000, n_jobs=-1, random_state=1).fit(P.to_numpy()[ok], y[ok])
    pred = en.predict(P.to_numpy())
    o["sig_UPF_s"] = (pred - pred.mean()) / pred.std()
    tt = o.test.to_numpy() & np.isfinite(y)
    info.append({"scenario": sc, "mean_UPF_%E": p.loc[p.n_recalls >= 1, "UPF"].mean(),
                 "r_with_main_UPF": np.corrcoef(p.loc[p.n_recalls >= 1, "UPF"], person.loc[person.n_recalls >= 1, "UPF"])[0, 1],
                 "n_proteins": int((en.coef_ != 0).sum()), "r_test": np.corrcoef(pred[tt], y[tt])[0, 1]})
    print(info[-1], flush=True)
    ind = o[(o.consortium == 0) & ~o.train].reset_index(drop=True)
    defs = outcome_defs(ind, PRIMARY)
    for m in ("M1", "M3"):
        Z = model_matrix(ind.merge(sig[["eid", "sig_BMI"]], on="eid", how="left") if m == "M3" and "sig_BMI" not in ind
                         else ind, m).to_numpy()
        res = Parallel(n_jobs=6)(delayed(cox)(ind["sig_UPF_s"].to_numpy(), Z, *defs[oc], ind["baseline_date"]) for oc in PRIMARY)
        rows += [{"scenario": sc, "analysis": f"UPF signature, {m}", "outcome": oc, **r} for oc, r in zip(PRIMARY, res)]
    dd = p[p.n_recalls >= 1].reset_index(drop=True)
    x = ((dd.UPF - dd.UPF.mean()) / dd.UPF.std()).to_numpy()
    Zd = model_matrix(dd, "M1", diet=True).to_numpy()
    dd_defs = outcome_defs(dd, PRIMARY)
    res = Parallel(n_jobs=6)(delayed(cox)(x, Zd, *dd_defs[oc], dd["baseline_date"]) for oc in PRIMARY)
    rows += [{"scenario": sc, "analysis": "Self-reported UPF, M1", "outcome": oc, **r} for oc, r in zip(PRIMARY, res)]
r = pd.DataFrame(rows)
r.to_csv(RES / "nova_sensitivity.csv", index=False)
pd.DataFrame(info).to_csv(RES / "nova_sensitivity_info.csv", index=False)
print(r.assign(c=r.apply(lambda s: f"{s.hr:.2f} ({s.lci:.2f}-{s.uci:.2f})", axis=1))
      .pivot_table(index=["analysis", "outcome"], columns="scenario", values="c", aggfunc="first")[["main", "upper", "lower"]].to_string())
