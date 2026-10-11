"""Proteomic signatures of diet (and a competing BMI signature) by elastic net.

Training: random 70% of the diet-proteome sample (seed 2026); performance (Pearson r) in the remaining 30%.
Scores are computed for every QC-passed Olink participant and standardised; Cox analyses exclude the training set.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV

from common import CACHE, EXPOSURES, RES, SEED

person = pd.read_pickle(CACHE / "person.pkl")
P = pd.read_pickle(CACHE / "olink_qc_plateadj.pkl")
o = person[person.olink_qc].set_index("eid").loc[P.index].reset_index()
diet = ((o.consortium == 0) & (o.n_recalls >= 1)).to_numpy()
rng = np.random.default_rng(SEED)
train = diet & (rng.random(len(o)) < 0.7)
test = diet & ~train
X = P.to_numpy()

scores = pd.DataFrame({"eid": o["eid"], "train": train, "test": test})
perf = []
for e in EXPOSURES + ["BMI"]:
    col = "bmi" if e == "BMI" else e
    y = o[col].to_numpy()
    ok = train & np.isfinite(y)
    en = ElasticNetCV(l1_ratio=0.5, alphas=20, cv=5, max_iter=5000, n_jobs=-1, random_state=1)
    en.fit(X[ok], y[ok])
    pred = en.predict(X)
    tt = test & np.isfinite(y)
    r = np.corrcoef(pred[tt], y[tt])[0, 1]
    w = pd.Series(en.coef_, index=P.columns)
    w[w != 0].sort_values().to_csv(RES / f"signature_weights_{e}.csv", header=["weight"])
    scores[f"sig_{e}"] = (pred - pred.mean()) / pred.std()
    perf.append({"signature": e, "n_train": int(ok.sum()), "n_test": int(tt.sum()), "n_proteins": int((w != 0).sum()),
                 "alpha": en.alpha_, "r_test": r})
    print(perf[-1], flush=True)
scores.to_pickle(CACHE / "signatures.pkl")
pd.DataFrame(perf).to_csv(RES / "signature_performance.csv", index=False)
print(scores.loc[~scores.train, [f"sig_{e}" for e in EXPOSURES + ["BMI"]]].corr().round(2).to_string())
