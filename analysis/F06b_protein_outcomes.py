"""Intestinal epithelial proteins -> outcomes in the independent sample (randomly selected Olink participants outside the
signature training set), Models 1 and 3. Complements the b path of the mediation analysis with more events."""
import pandas as pd
from joblib import Parallel, delayed

from common import CACHE, RES, cox, model_matrix, outcome_defs

PROTS = ["guca2a", "ddc", "fabp2", "fabp6", "cdh17", "dsg2", "lgals4"]
OUTS = ["Gallstones", "IBD", "Crohn", "UC", "Acute_pancreatitis", "AP_nonbiliary"]

person = pd.read_pickle(CACHE / "person.pkl")
sig = pd.read_pickle(CACHE / "signatures.pkl")
P = pd.read_pickle(CACHE / "olink_qc_plateadj.pkl")
o = person.merge(sig, on="eid", how="inner")
o = o[(o.consortium == 0) & ~o.train].reset_index(drop=True)
for p in PROTS:
    o[f"prot_{p}"] = P.loc[o.eid, p].to_numpy()
defs = outcome_defs(o, OUTS)
Zs = {m: model_matrix(o, m).to_numpy() for m in ["M1", "M3"]}
jobs = [(m, oc, p) for m in Zs for oc in OUTS for p in PROTS]
res = Parallel(n_jobs=6)(delayed(cox)(o[f"prot_{p}"].to_numpy(), Zs[m], *defs[oc], o["baseline_date"]) for m, oc, p in jobs)
r = pd.DataFrame([{"model": m, "outcome": oc, "protein": p, **v} for (m, oc, p), v in zip(jobs, res)])
r.to_csv(RES / "cox_epithelial_proteins.csv", index=False)
r["c"] = r.apply(lambda s: f"{s.hr:.2f} ({s.lci:.2f}-{s.uci:.2f}){'*' if s.p < .05 else ''}", axis=1)
for m in ["M1", "M3"]:
    print(f"== {m}\n" + r[r.model == m].pivot(index="protein", columns="outcome", values="c")[OUTS].to_string())
