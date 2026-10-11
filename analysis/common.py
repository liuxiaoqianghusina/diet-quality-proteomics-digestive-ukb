"""Shared paths, covariate builders and model helpers for the diet-proteome-digestive disease project."""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from scipy import stats

# Directories are configured with environment variables (see README):
#   DIET_PROJECT_DIR  this project: cache/, results/, figures/ and manuscript/ are created inside; MR inputs in data/mr/
#   UKB_EXTRACT_DIR   cache written by data_prep/ (diet, covariates, outcomes, biochemistry, item-level UPF)
#   UKB_DATA_DIR      UK Biobank dataset CSV extracts
#   UKB_OLINK_DIR     Olink NPX tables (olink_instance_0.csv, olink_instance_2.csv, olink_instance_3.csv)
PROJ = Path(os.environ.get("DIET_PROJECT_DIR", "."))
NOVA_DIR = Path(__file__).resolve().parent.parent / "data_prep"   # nova_classification.csv
PC = Path(os.environ.get("UKB_EXTRACT_DIR", "../ukb_extracts"))
UKB = Path(os.environ.get("UKB_DATA_DIR", "../ukb_data"))
CACHE = PROJ / "cache"
RES = PROJ / "results"
FIG = PROJ / "figures"
for d in (CACHE, RES, FIG):
    d.mkdir(parents=True, exist_ok=True)
OLINK = Path(os.environ.get("UKB_OLINK_DIR", UKB / "olink")) / "olink_instance_0.csv"
CENSOR = pd.Timestamp("2022-12-31")      # hospital inpatient records incomplete from 2023
SEED = 2026

EXPOSURES = ["UPF", "uPDI", "hPDI", "fibre"]
EXPO_LABEL = {"UPF": "Ultra-processed food (% energy)", "uPDI": "Unhealthful plant-based diet index",
              "hPDI": "Healthful plant-based diet index", "fibre": "Dietary fibre (energy-adjusted)",
              "BMI": "BMI"}
PRIMARY = ["IBD", "Crohn", "UC", "Gallstones", "Acute_pancreatitis", "AP_nonbiliary"]
POSITIVE = ["MASLD", "Cirrhosis"]
SECONDARY = ["GERD", "PUD", "Gastritis", "Diverticular", "IBS"]
NEGATIVE = ["NC_cataract", "NC_inguinal_hernia", "NC_hearing_loss", "NC_carpal_tunnel", "NC_varicose_veins"]
OUT_LABEL = {"IBD": "Inflammatory bowel disease", "Crohn": "Crohn's disease", "UC": "Ulcerative colitis",
             "Gallstones": "Gallstone disease", "Acute_pancreatitis": "Acute pancreatitis",
             "AP_nonbiliary": "Non-gallstone acute pancreatitis", "AP_biliary": "Gallstone-related acute pancreatitis",
             "MASLD": "MASLD", "Cirrhosis": "Liver fibrosis/cirrhosis",
             "GERD": "Gastro-oesophageal reflux disease", "PUD": "Peptic ulcer disease", "Gastritis": "Gastritis/duodenitis",
             "Diverticular": "Diverticular disease", "IBS": "Irritable bowel syndrome",
             "NC_cataract": "Cataract", "NC_inguinal_hernia": "Inguinal hernia", "NC_hearing_loss": "Hearing loss",
             "NC_carpal_tunnel": "Upper-limb mononeuropathy", "NC_varicose_veins": "Varicose veins"}
NEG_FIELDS = {"NC_cataract": ["p131164", "p131166"], "NC_inguinal_hernia": ["p131612"],
              "NC_hearing_loss": ["p131258", "p131260"], "NC_carpal_tunnel": ["p131074"],
              "NC_varicose_veins": ["p131402"]}
WINDOW = pd.Timedelta(days=90)   # gallstone diagnosis within 90 days after acute pancreatitis -> gallstone-related


def base_covariates(d, diet=True):
    """Age, sex, ethnicity, Townsend, education, smoking, alcohol, physical activity, BMI, fasting time,
    assessment centre (+ energy intake and number of recalls when diet=True)."""
    X = pd.DataFrame(index=d.index)
    cont = ["age", "townsend", "bmi", "fasting_h"] + (["energy_kj", "n_recalls"] if diet else [])
    for c in cont:
        X[c] = d[c].fillna(d[c].median())
    lm = np.log1p(d["met_min"])
    X["log_met"], X["met_missing"] = lm.fillna(lm.median()), d["met_min"].isna().astype(float)
    for c in ["sex", "white", "degree"]:
        X[c] = d[c].fillna(-1)
    for c in ["smoking", "alcohol_freq", "centre"]:
        X = X.join(pd.get_dummies(d[c].fillna(-1).astype(int), prefix=c, drop_first=True, dtype=float))
    return X.astype(float)


def add_kidney_crp(X, d):
    for c, n in [("cysc", "log_cysc"), ("creat", "log_creat"), ("crp", "log_crp")]:
        v = np.log(d[c].clip(lower=1e-3))
        X[n] = v.fillna(v.median()).to_numpy()
    return X


def add_metabolic(X, d):
    vals = {"waist": d["waist"], "log_alt": np.log(d["alt"].clip(lower=1)), "log_ggt": np.log(d["ggt"].clip(lower=1)),
            "hba1c": d["hba1c"], "log_tg": np.log(d["tg"].clip(lower=0.1))}
    for k, v in vals.items():
        X[k] = v.fillna(v.median()).to_numpy()
    return X


def model_matrix(d, model, diet=True):
    """M1 = base + kidney function + CRP (main); M2 = M1 + metabolic/liver markers; M3 = M2 + BMI proteomic signature."""
    X = add_kidney_crp(base_covariates(d, diet), d)
    if model in ("M2", "M3"):
        X = add_metabolic(X, d)
    if model == "M3":
        X["sig_BMI"] = d["sig_BMI"].to_numpy()
    return X


def ols_many(x, Z, Y):
    X = np.column_stack([np.ones(len(x)), x, Z])
    XtX_inv = np.linalg.pinv(X.T @ X)
    B = XtX_inv @ X.T @ Y
    resid = Y - X @ B
    dfree = X.shape[0] - np.linalg.matrix_rank(X)
    se = np.sqrt((resid ** 2).sum(axis=0) / dfree * XtX_inv[1, 1])
    p = 2 * stats.t.sf(np.abs(B[1] / se), dfree)
    return B[1], se, p


def follow_up_end(d):
    return pd.concat([d["death_date"], d["lost_date"]], axis=1).min(axis=1).clip(upper=CENSOR).fillna(CENSOR)


def outcome_defs(d, names, lag_years=0):
    """Returns {name: (eligible mask, event indicator, stop date)}; prevalent cases (date <= baseline) excluded.
    lag_years > 0 drops events (and person-time) in the first years after baseline."""
    base, end = d["baseline_date"], follow_up_end(d)
    out = {}
    for name in names:
        if name in ("AP_nonbiliary", "AP_biliary"):
            ap, gs = d["Acute_pancreatitis"], d["Gallstones"]
            prev_ap, prev_gs = ap.notna() & (ap <= base), gs.notna() & (gs <= base)
            inc = ap.notna() & (ap > base) & (ap <= end)
            biliary = inc & gs.notna() & (gs <= ap + WINDOW)
            if name == "AP_biliary":
                out[name] = (~prev_ap, biliary, ap.where(biliary, end))
            else:
                gs_first = gs.notna() & (gs > base) & (~inc | (gs < ap - WINDOW))
                stop = end.where(~gs_first, np.minimum(gs, end))
                ev = inc & ~biliary & (ap <= stop)
                out[name] = (~prev_ap & ~prev_gs, ev, ap.where(ev, stop))
            continue
        s = d[name]
        prev = s.notna() & (s <= base)
        inc = s.notna() & (s > base) & (s <= end)
        out[name] = (~prev, inc, s.where(inc, end))
    if lag_years:
        lag = pd.Timedelta(days=365.25 * lag_years)
        for k, (elig, ev, stop) in out.items():
            early = (stop - base) <= lag
            out[k] = (elig & ~(early & ev), ev & ~early, stop)
    return out


def cox(x, Z, elig, ev, stop, base, extra_terms=None):
    T = (pd.to_datetime(stop) - base).dt.days / 365.25
    f = pd.DataFrame(np.asarray(Z), index=elig.index).add_prefix("z")
    f["x"], f["T"], f["E"] = np.asarray(x), T, ev.astype(int)
    f = f[elig & (f["T"] > 0) & np.isfinite(f["x"])]
    f = f.loc[:, (f.std() > 0) | f.columns.isin(["T", "E"])]
    s = CoxPHFitter(penalizer=1e-4).fit(f, "T", "E").summary.loc["x"]
    return {"n": len(f), "events": int(f["E"].sum()), "hr": np.exp(s["coef"]), "lci": np.exp(s["coef lower 95%"]),
            "uci": np.exp(s["coef upper 95%"]), "p": s["p"], "logHR": s["coef"], "se": s["se(coef)"]}


def fmt_hr(r):
    return f"{r['hr']:.2f} ({r['lci']:.2f}–{r['uci']:.2f})"
