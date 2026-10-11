"""Diet exposures -> plasma proteome (linear regression, one protein at a time).

Sample: QC-passed, randomly selected Olink participants with >=1 valid 24h recall.
Main model (M1): age, sex, ethnicity, Townsend, education, smoking, alcohol, physical activity, BMI, fasting time,
assessment centre, energy intake, number of recalls, log cystatin C, log creatinine, log CRP.
Sensitivity (M0): without kidney function and CRP. Exposures and proteins per SD.
"""
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from common import CACHE, EXPOSURES, RES, add_kidney_crp, base_covariates, ols_many

GUT = ["guca2a", "ddc", "fabp2", "fabp6", "cdh17", "lgals4", "dsg2", "tff2", "tff1", "tff3", "lcn2", "ocln", "lbp",
       "cd14", "reg3a", "reg4", "reg1a", "reg1b", "pigr", "muc13", "muc2", "agr2", "ccl25", "cdhr2", "fgf19", "gcg",
       "gip", "pyy", "fgf21", "stc1"]

person = pd.read_pickle(CACHE / "person.pkl")
P = pd.read_pickle(CACHE / "olink_qc_plateadj.pkl")
d = person[person.olink_qc & (person.consortium == 0) & (person.n_recalls >= 1)].set_index("eid")
d = d.loc[d.index.intersection(P.index)].reset_index()
Y = P.loc[d["eid"]].to_numpy()
print("n =", len(d), "proteins =", Y.shape[1])
Z1 = add_kidney_crp(base_covariates(d), d).to_numpy()
Z0 = base_covariates(d).to_numpy()

summary = []
for e in EXPOSURES:
    x = ((d[e] - d[e].mean()) / d[e].std()).to_numpy()
    for label, Z in [("M1", Z1), ("M0", Z0)]:
        b, se, p = ols_many(x, Z, Y)
        r = pd.DataFrame({"protein": P.columns, "beta": b, "se": se, "p": p})
        r["lci"], r["uci"] = r.beta - 1.96 * r.se, r.beta + 1.96 * r.se
        r["fdr"] = multipletests(r.p, method="fdr_bh")[1]
        r["bonf"] = np.minimum(r.p * len(r), 1)
        r = r.sort_values("p")
        r.to_csv(RES / f"diet_proteome_{e}_{label}.csv", index=False)
        summary.append({"exposure": e, "model": label, "n": len(d), "n_fdr": int((r.fdr < .05).sum()),
                        "n_bonf": int((r.bonf < .05).sum()), "n_bonf_pos": int(((r.bonf < .05) & (r.beta > 0)).sum()),
                        "n_bonf_neg": int(((r.bonf < .05) & (r.beta < 0)).sum()),
                        "top10": ", ".join(r.protein.str.upper().head(10))})
s = pd.DataFrame(summary)
s.to_csv(RES / "diet_proteome_summary.csv", index=False)
print(s.to_string(index=False))

# intestinal and gut-related proteins across exposures (main model)
rows = []
for e in EXPOSURES:
    r = pd.read_csv(RES / f"diet_proteome_{e}_M1.csv")
    rows.append(r[r.protein.isin(GUT)].assign(exposure=e))
g = pd.concat(rows)
g.to_csv(RES / "diet_gutproteins_M1.csv", index=False)
print(g.assign(c=g.apply(lambda s: f"{s.beta:+.3f}{'*' if s.bonf < .05 else ''}", axis=1))
      .pivot(index="protein", columns="exposure", values="c")[EXPOSURES].to_string())

# food-group drivers of selected proteins
targets = ["ddc", "guca2a", "fabp2", "fabp6", "cdh17", "dsg2", "lcn2", "fgf21"]
groups = [c for c in d.columns if c.startswith("g_")]
fr = []
for gname in groups:
    x = np.log1p(d[gname])
    x = ((x - x.mean()) / x.std()).to_numpy()
    b, se, p = ols_many(x, Z1, P.loc[d["eid"], targets].to_numpy())
    fr += [{"food_group": gname[2:], "protein": t, "beta": bb, "se": ss, "p": pp} for t, bb, ss, pp in zip(targets, b, se, p)]
pd.DataFrame(fr).to_csv(RES / "foodgroup_drivers.csv", index=False)
