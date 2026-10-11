"""Sensitivity analyses for the signature -> outcome associations (models M1 and M3).
S1 exclude events in the first 2 years of follow-up
S2 include UKB-PPP consortium-selected participants
S3 participants without any 24h recall (signature applied to people who never reported diet)
S4 diet scores among participants with >=2 valid recalls (diet -> outcome, M1)
"""
import pandas as pd
from joblib import Parallel, delayed

from common import CACHE, EXPOSURES, PRIMARY, POSITIVE, RES, cox, model_matrix, outcome_defs

OUTS = PRIMARY + POSITIVE
XS = [f"sig_{e}" for e in EXPOSURES]


def block(d, label, models=("M1", "M3"), xs=XS, diet=False, lag=0):
    defs = outcome_defs(d, OUTS, lag_years=lag)
    Zs = {m: model_matrix(d, m, diet=diet).to_numpy() for m in models}
    jobs = [(m, o, x) for m in models for o in OUTS for x in xs]
    res = Parallel(n_jobs=6)(delayed(cox)(d[x].to_numpy(), Zs[m], *defs[o], d["baseline_date"]) for m, o, x in jobs)
    return pd.DataFrame([{"analysis": label, "model": m, "outcome": o, "x": x, **v} for (m, o, x), v in zip(jobs, res)])


if __name__ == "__main__":
    person = pd.read_pickle(CACHE / "person.pkl")
    sig = pd.read_pickle(CACHE / "signatures.pkl")
    o = person.merge(sig, on="eid", how="inner")
    o = o[~o.train]
    main = o[o.consortium == 0].reset_index(drop=True)
    out = [block(main, "S1: exclude first 2 years", lag=2),
           block(o.reset_index(drop=True), "S2: include consortium-selected"),
           block(main[main.n_recalls.fillna(0) == 0].reset_index(drop=True), "S3: no 24h recall")]
    dd = person[person.n_recalls >= 2].reset_index(drop=True)
    for e in EXPOSURES:
        dd[f"diet_{e}"] = (dd[e] - dd[e].mean()) / dd[e].std()
    out.append(block(dd, "S4: diet, >=2 recalls", models=("M1",), xs=[f"diet_{e}" for e in EXPOSURES], diet=True))
    r = pd.concat(out)
    r.to_csv(RES / "cox_sensitivity.csv", index=False)

    # proportional-hazards check (Schoenfeld residuals, rank time) for the main M1 signature models
    import numpy as np
    from lifelines import CoxPHFitter
    from lifelines.statistics import proportional_hazard_test
    defs = outcome_defs(main, OUTS)
    Z = model_matrix(main, "M1").to_numpy()
    ph = []
    for oc in OUTS:
        elig, ev, stop = defs[oc]
        T = (pd.to_datetime(stop) - main["baseline_date"]).dt.days / 365.25
        for x in XS:
            f = pd.DataFrame(Z, index=main.index).add_prefix("z")
            f["x"], f["T"], f["E"] = main[x].to_numpy(), T, ev.astype(int)
            f = f[elig & (f["T"] > 0)]
            f = f.loc[:, (f.std() > 0) | f.columns.isin(["T", "E"])]
            cph = CoxPHFitter(penalizer=1e-4).fit(f, "T", "E")
            t = proportional_hazard_test(cph, f, time_transform="rank").summary.loc["x"]
            ph.append({"outcome": oc, "x": x, "ph_p": float(t["p"])})
    ph = pd.DataFrame(ph)
    ph.to_csv(RES / "ph_test.csv", index=False)
    print(ph.pivot(index="outcome", columns="x", values="ph_p").round(3).to_string())
    r["c"] = r.apply(lambda s: f"{s.hr:.2f} ({s.lci:.2f}-{s.uci:.2f})", axis=1)
    for a in r.analysis.unique():
        for m in r[r.analysis == a].model.unique():
            t = r[(r.analysis == a) & (r.model == m)]
            print(f"\n== {a} {m} (n={t.n.max()})\n" + t.pivot(index="outcome", columns="x", values="c").loc[OUTS].to_string())
