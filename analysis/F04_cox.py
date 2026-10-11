"""Cox models.
(a) proteomic signatures -> outcomes, randomly selected QC-passed Olink participants outside the training set; M1-M3.
(b) diet scores -> outcomes in all participants with >=1 valid 24h recall; M1 and M2.
(c) pre-specified check of the main hypothesis (rule fixed before the outcome analyses).
"""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from statsmodels.stats.multitest import multipletests

from common import (CACHE, EXPOSURES, NEGATIVE, POSITIVE, PRIMARY, RES, SECONDARY, cox, model_matrix, outcome_defs)

OUTS = PRIMARY + ["AP_biliary"] + POSITIVE + SECONDARY + NEGATIVE


def run(d, xs, models, diet, label):
    defs = outcome_defs(d, OUTS)
    Zs = {m: model_matrix(d, m, diet=diet).to_numpy() for m in models}
    jobs = [(m, o, x) for m in models for o in OUTS for x in xs if not (m == "M3" and x == "sig_BMI")]
    res = Parallel(n_jobs=6)(delayed(cox)(d[x].to_numpy(), Zs[m], *defs[o], d["baseline_date"]) for m, o, x in jobs)
    r = pd.DataFrame([{"model": m, "outcome": o, "x": x, **v} for (m, o, x), v in zip(jobs, res)])
    r["fdr"] = r.groupby(["model", "x"])["p"].transform(lambda p: multipletests(p, method="fdr_bh")[1])
    r.to_csv(RES / f"cox_{label}.csv", index=False)
    return r


if __name__ == "__main__":
    person = pd.read_pickle(CACHE / "person.pkl")
    sig = pd.read_pickle(CACHE / "signatures.pkl")
    o = person.merge(sig, on="eid", how="inner")
    o = o[(o.consortium == 0) & ~o.train].reset_index(drop=True)
    print("signature analysis n =", len(o))
    xs = [f"sig_{e}" for e in EXPOSURES] + ["sig_BMI"]
    rs = run(o, xs, ["M1", "M2", "M3"], diet=False, label="signatures")

    dd = person[person.n_recalls >= 1].reset_index(drop=True)
    for e in EXPOSURES:
        dd[f"diet_{e}"] = (dd[e] - dd[e].mean()) / dd[e].std()
    print("diet analysis n =", len(dd))
    rd = run(dd, [f"diet_{e}" for e in EXPOSURES], ["M1", "M2"], diet=True, label="diet")

    fmt = lambda s: f"{s.hr:.2f} ({s.lci:.2f}-{s.uci:.2f}){'*' if (s.lci > 1 or s.uci < 1) else ''}"
    for name, r in [("signatures", rs), ("diet", rd)]:
        for m in sorted(r.model.unique()):
            t = r[r.model == m].assign(c=lambda x: x.apply(fmt, axis=1)).pivot(index="outcome", columns="x", values="c")
            ev = r[(r.model == m)].groupby("outcome")["events"].first()
            print(f"\n== {name} {m}\n" + t.loc[OUTS].join(ev).to_string())

    m3 = rs[rs.model == "M3"].set_index(["outcome", "x"])
    harm, prot = ["sig_UPF", "sig_uPDI"], ["sig_hPDI", "sig_fibre"]
    count = lambda out: int(sum(m3.loc[(out, x), "lci"] > 1 for x in harm) + sum(m3.loc[(out, x), "uci"] < 1 for x in prot))
    key = {k: count(k) for k in ["IBD", "Gallstones", "Acute_pancreatitis"]}
    neg = {k: count(k) for k in NEGATIVE}
    ok = sum(v >= 2 for v in key.values()) >= 2 and all(v < 2 for v in neg.values())
    print("\nHypothesis check (M3):", key, "negative controls:", neg, "->", "HOLDS" if ok else "DOES NOT HOLD")
    pd.Series({"key": key, "neg": neg, "holds": ok}).to_json(RES / "hypothesis_check.json")
