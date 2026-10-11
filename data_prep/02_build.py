"""Build diet exposures (hPDI, uPDI, PDI, UPF-approx, fibre), covariates and incident outcomes.

Diet: Oxford WebQ 24h recalls, UKB updated food groups (fields 26062-26155, g/day) and nutrients.
Each recall is kept if not flagged by field 100026 and energy is plausible
(500-3500 kcal women, 500-4200 kcal men); food groups are averaged over valid recalls.
Output: UKB_EXTRACT_DIR/analytic.pkl (one row per participant).
"""
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

C = Path(os.environ.get("UKB_EXTRACT_DIR", "../ukb_extracts"))

# ---------------------------------------------------------------- food-group mapping (Satija 18 groups, UKB 26000 fields)
PDI_GROUPS = {
    # healthy plant foods
    "whole_grains": ([26071, 26074, 26076, 26077, 26078, 26105, 26114], "healthy"),
    "fruits": ([26089, 26090, 26091, 26092, 26093, 26094], "healthy"),
    "vegetables": ([26065, 26098, 26115, 26123, 26125, 26143, 26146, 26147], "healthy"),
    "nuts": ([26106, 26107, 26108], "healthy"),
    "legumes": ([26101, 26136, 26137, 26145], "healthy"),
    "vegetable_oils": ([26110, 26111, 26112], "healthy"),
    "tea_coffee": ([26081, 26082, 26141, 26142], "healthy"),
    # less healthy plant foods
    "fruit_juices": ([26095], "unhealthy"),
    "refined_grains": ([26072, 26073, 26075, 26079, 26083, 26097, 26113], "unhealthy"),
    "potatoes": ([26118, 26119, 26120, 26134], "unhealthy"),
    "ssb": ([26127], "unhealthy"),
    "sweets_desserts": ([26064, 26068, 26080, 26085, 26086, 26140], "unhealthy"),
    # animal foods
    "animal_fat": ([26062, 26063, 26154], "animal"),
    "dairy": ([26084, 26087, 26096, 26099, 26102, 26103, 26131, 26133, 26150], "animal"),
    "eggs": ([26088], "animal"),
    "fish_seafood": ([26070, 26109, 26132, 26139, 26149], "animal"),
    "meat": ([26066, 26069, 26100, 26104, 26117, 26121, 26122], "animal"),
    "misc_animal": ([26116, 26129], "animal"),
}

# food groups dominated by NOVA group 4 items (approximation; item-level NOVA needed for the paper)
UPF_GROUPS = [26062, 26068, 26069, 26070, 26072, 26073, 26075, 26079, 26080, 26083, 26084, 26085, 26086,
              26087, 26106, 26111, 26112, 26116, 26122, 26124, 26126, 26127, 26128, 26129, 26130, 26134,
              26136, 26137, 26138, 26140, 26144, 26145]

# ---------------------------------------------------------------- diet
d = pd.read_pickle(C / "diet_raw.pkl")
pop = pd.read_pickle(C / "pop.pkl")
d = d.merge(pop[["eid", "p31"]], on="eid", how="left")

all_fields = sorted({f for g in PDI_GROUPS.values() for f in g[0]} | set(UPF_GROUPS) | {26000, 26002, 26017})
per_recall = []
for i in range(5):
    e = d[f"p26002_i{i}"]
    kcal = e / 4.184
    lo = 500
    hi = np.where(d["p31"] == 1, 4200, 3500)
    ok = e.notna() & d[f"p100026_i{i}"].isna() & (kcal >= lo) & (kcal <= hi)
    block = d[[f"p{f}_i{i}" for f in all_fields]].copy()
    block.columns = [f"f{f}" for f in all_fields]
    block[~ok] = np.nan
    block["eid"] = d["eid"]
    block["valid"] = ok.astype(int)
    per_recall.append(block)

stack = pd.concat(per_recall)
n_valid = stack.groupby("eid")["valid"].sum().rename("n_recalls")
mean = stack[stack["valid"] == 1].groupby("eid")[[f"f{f}" for f in all_fields]].mean()
diet = mean.join(n_valid, how="right")
diet = diet[diet["n_recalls"] >= 1].copy()

grp = pd.DataFrame(index=diet.index)
for g, (fields, _) in PDI_GROUPS.items():
    grp[g] = diet[[f"f{f}" for f in fields]].fillna(0).sum(axis=1)


def quint(s):
    # quintile score 1-5 by percentile rank (ties averaged, so non-consumers share the lowest score)
    return np.ceil(s.rank(pct=True, method="average") * 5).clip(1, 5)


q = grp.apply(quint)
rev = 6 - q
types = {g: t for g, (_, t) in PDI_GROUPS.items()}
H = [g for g, t in types.items() if t == "healthy"]
U = [g for g, t in types.items() if t == "unhealthy"]
A = [g for g, t in types.items() if t == "animal"]
out = pd.DataFrame(index=diet.index)
out["PDI"] = q[H + U].sum(axis=1) + rev[A].sum(axis=1)
out["hPDI"] = q[H].sum(axis=1) + rev[U + A].sum(axis=1)
out["uPDI"] = q[U].sum(axis=1) + rev[H + A].sum(axis=1)
out["UPF_pct"] = 100 * diet[[f"f{f}" for f in UPF_GROUPS]].fillna(0).sum(axis=1) / diet["f26000"]
out["energy_kj"] = diet["f26002"]
fib = diet["f26017"]
# residual-method energy adjustment for fibre
b = np.polyfit(out["energy_kj"], fib, 1)
out["fibre_adj"] = fib - (b[0] * out["energy_kj"] + b[1]) + fib.mean()
out["n_recalls"] = diet["n_recalls"]
out = out.join(grp.add_prefix("g_"))
out = out.reset_index().rename(columns={"index": "eid"})

# bowel habit fields (digestive health questionnaire) kept for later use
bowel = d[["eid", "p21033", "p21034", "p21040", "p21043", "p21044"]]
out = out.merge(bowel, on="eid", how="left")
del d, stack

# ---------------------------------------------------------------- covariates
base = pd.read_pickle(C / "baseline.pkl")
base["eid"] = base["eid"].astype(int)
cov = pop.merge(base, on="eid", how="outer")
num = lambda s: pd.to_numeric(s, errors="coerce")
cv = pd.DataFrame({"eid": cov["eid"]})
cv["baseline_date"] = pd.to_datetime(cov["p53_i0"], errors="coerce")
cv["age"] = num(cov["p21022"])
cv["sex"] = num(cov["p31"])
eth = num(cov["p21000_i0"])
cv["white"] = np.where(eth.isin([1, 1001, 1002, 1003]), 1, np.where(eth > 0, 0, np.nan))
cv["townsend"] = num(cov["p22189"])
cv["bmi"] = num(cov["p21001_i0"])
smk = num(cov["p20116_i0"])
cv["smoking"] = smk.where(smk >= 0)
alc = num(cov["p1558_i0"])
cv["alcohol_freq"] = alc.where(alc > 0)
cv["met"] = num(cov["p22040_i0"])
edu = cov["p6138_i0"].astype(str)
cv["degree"] = np.where(edu.str.contains(r"(^|\D)1(\D|$)"), 1, np.where(edu.str.contains(r"\d"), 0, np.nan))
cv["degree"] = np.where(edu.str.contains("-3"), np.nan, cv["degree"])
cv["fasting_h"] = num(cov["p74_i0"])
cv["centre"] = num(cov["p54_i0"])
cv["lost_date"] = pd.to_datetime(cov["p191"], errors="coerce")

# ---------------------------------------------------------------- outcomes
o = pd.read_pickle(C / "outcomes.pkl")
dt = lambda s: pd.to_datetime(s, errors="coerce")
FO = {  # first-occurrence date fields
    "GERD": ["p131584"], "PUD": ["p131590", "p131592", "p131594"], "Gastritis": ["p131598"],
    "Crohn": ["p131626"], "UC": ["p131628"], "Diverticular": ["p131636"], "IBS": ["p131638"],
    "Functional_bowel": ["p131640"], "Cirrhosis": ["p131666"], "Gallstones": ["p131674"],
    "Acute_pancreatitis": ["p131682"], "Depression": ["p130894", "p130896"], "Anxiety": ["p130906"],
    "Sleep_disorder": ["p131060"], "Migraine": ["p131052"],
}
ev = pd.DataFrame({"eid": o["eid"].astype(int)})
for k, cols in FO.items():
    ev[k] = pd.concat([dt(o[c]) for c in cols], axis=1).min(axis=1)
ev["IBD"] = ev[["Crohn", "UC"]].min(axis=1)
ev["Dementia"] = dt(o["p42018"])
ev["PD"] = dt(o["p42032"])
ev["Stroke"] = dt(o["p42006"])

# MASLD from hospital ICD-10 (K76.0 / K75.8) with matched dates
date_cols = sorted([c for c in o.columns if c.startswith("p41280_a")], key=lambda x: int(x.split("_a")[1]))
codes = o["p41270"].fillna("").str.split("|").explode().to_frame("code")
codes["pos"] = codes.groupby(level=0).cumcount()


def first_code_date(prefixes):
    hit = codes[codes["code"].str.startswith(prefixes)]
    vals = [o.at[i, date_cols[p]] if p < len(date_cols) else None for i, p in zip(hit.index, hit["pos"])]
    s = pd.Series(pd.to_datetime(vals, errors="coerce"), index=hit.index)
    return s.groupby(level=0).min().reindex(o.index)


ev["MASLD"] = first_code_date(("K760", "K758"))

# colorectal cancer from cancer registry
crc = pd.Series(pd.NaT, index=o.index)
for c in [c for c in o.columns if c.startswith("p40006_i")]:
    inst = c.split("_i")[1]
    m = o[c].fillna("").str.match(r"C1[89]|C20")
    dd = dt(o[f"p40005_i{inst}"]).where(m)
    crc = pd.concat([crc, dd], axis=1).min(axis=1)
ev["CRC"] = crc
ev["death_date"] = dt(o["p40000_i0"])

df = cv.merge(ev, on="eid", how="left")
df = df.merge(out, on="eid", how="left")
df.to_pickle(C / "analytic.pkl")
print(df.shape)
print(df[["PDI", "hPDI", "uPDI", "UPF_pct", "fibre_adj", "n_recalls"]].describe().T.to_string())
