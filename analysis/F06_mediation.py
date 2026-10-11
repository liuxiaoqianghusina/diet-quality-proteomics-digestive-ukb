"""Single-protein mediation: diet -> intestinal epithelial protein -> outcome (product of coefficients).

Sample: diet-proteome sample (randomly selected, QC-passed Olink participants with >=1 valid 24h recall).
a: linear regression of protein (SD) on diet (SD), M1 covariates; b: Cox log HR of protein adjusted for diet + M1;
c: total Cox log HR of diet, M1. Indirect effect a*b with 95% CI by Monte Carlo (100,000 draws).
Proportion mediated reported only when the total effect has P < 0.05.
"""
import numpy as np
import pandas as pd

from common import CACHE, EXPOSURES, RES, cox, model_matrix, ols_many, outcome_defs

MEDIATORS = ["ddc", "guca2a", "fabp2", "fabp6", "cdh17"]
OUTCOMES = ["Gallstones", "IBD", "Acute_pancreatitis"]
rng = np.random.default_rng(1)

person = pd.read_pickle(CACHE / "person.pkl")
P = pd.read_pickle(CACHE / "olink_qc_plateadj.pkl")
d = person[person.olink_qc & (person.consortium == 0) & (person.n_recalls >= 1)]
d = d[d.eid.isin(P.index)].reset_index(drop=True)
Z = model_matrix(d, "M1", diet=True)
defs = outcome_defs(d, OUTCOMES)
rows = []
for e in EXPOSURES:
    x = ((d[e] - d[e].mean()) / d[e].std()).to_numpy()
    for oc in OUTCOMES:
        tot = cox(x, Z.to_numpy(), *defs[oc], d["baseline_date"])
        for m in MEDIATORS:
            med = P.loc[d.eid, m].to_numpy()
            a, a_se, a_p = [v[0] for v in ols_many(x, Z.to_numpy(), med[:, None])]
            Zb = Z.assign(diet=x).to_numpy()
            b = cox(med, Zb, *defs[oc], d["baseline_date"])
            draws = rng.normal(a, a_se, 100000) * rng.normal(b["logHR"], b["se"], 100000)
            ind, lo, hi = a * b["logHR"], *np.percentile(draws, [2.5, 97.5])
            prop = ind / tot["logHR"] if tot["p"] < 0.05 else np.nan
            rows.append({"exposure": e, "outcome": oc, "mediator": m, "n": tot["n"], "events": tot["events"],
                         "total_HR": tot["hr"], "total_lci": tot["lci"], "total_uci": tot["uci"], "total_p": tot["p"],
                         "a_beta": a, "a_p": a_p, "b_HR": b["hr"], "b_lci": b["lci"], "b_uci": b["uci"], "b_p": b["p"],
                         "indirect_HR": np.exp(ind), "indirect_lci": np.exp(lo), "indirect_uci": np.exp(hi),
                         "indirect_sig": (lo > 0) or (hi < 0), "prop_mediated": prop})
    print(e, "done", flush=True)
r = pd.DataFrame(rows)
r.to_csv(RES / "mediation.csv", index=False)
print(r[["exposure", "outcome", "mediator", "total_HR", "total_p", "a_beta", "b_HR", "b_p", "indirect_HR",
         "indirect_lci", "indirect_uci", "prop_mediated"]].round(4).to_string(index=False))
