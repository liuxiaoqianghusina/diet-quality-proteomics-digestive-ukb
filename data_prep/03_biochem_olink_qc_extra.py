"""Blood biochemistry, Olink sample metadata and further covariates/negative-control outcomes.

Outputs (pickle) go to UKB_EXTRACT_DIR:
  biochem.pkl   creatinine (30700), CRP (30710), cystatin C (30720), HbA1c (30750), ALT (30620), AST (30650)
  olink_qc.pkl  Olink sample metadata: proteins measured (30900), plate (30901), well (30902),
                UKB-PPP consortium-selected participant (30903)
  extra.pkl     waist circumference (48), GGT (30730), triglycerides (30870) and first-occurrence dates of the
                negative-control outcomes (cataract 131164/131166, inguinal hernia 131612, hearing loss 131258/131260,
                upper-limb mononeuropathy 131074, varicose veins of the lower limbs 131402)
"""
import os
from pathlib import Path

import pandas as pd

D = Path(os.environ.get("UKB_DATA_DIR", "../ukb_data"))
OUT = Path(os.environ.get("UKB_EXTRACT_DIR", "../ukb_extracts"))
OUT.mkdir(parents=True, exist_ok=True)
BIO = D / "UKB619657_biological_samples_2025Nov.csv"
NEG = {"NC_cataract": ["p131164", "p131166"], "NC_inguinal_hernia": ["p131612"],
       "NC_hearing_loss": ["p131258", "p131260"], "NC_carpal_tunnel": ["p131074"],
       "NC_varicose_veins": ["p131402"]}

b = pd.read_csv(BIO, usecols=["eid", "p30700_i0", "p30710_i0", "p30720_i0", "p30750_i0", "p30620_i0", "p30650_i0"])
b.to_pickle(OUT / "biochem.pkl")
print("biochem", b.shape)

q = pd.read_csv(BIO, usecols=["eid", "p30900_i0", "p30901_i0", "p30902_i0", "p30903_i0"])
q.to_pickle(OUT / "olink_qc.pkl")
print("olink_qc", q.shape)

w = pd.read_csv(D / "UKB619657_assessment_center_2025Nov.csv", usecols=["eid", "p48_i0"], dtype=str)
g = pd.read_csv(BIO, usecols=["eid", "p30730_i0", "p30870_i0"])
o = pd.read_csv(D / "UKB619657_health_outcomes_2025Nov.csv", usecols=["eid"] + [c for v in NEG.values() for c in v],
                dtype=str)
w["eid"] = w["eid"].astype(int)
o["eid"] = o["eid"].astype(int)
x = w.merge(g, on="eid", how="outer").merge(o, on="eid", how="outer")
x.to_pickle(OUT / "extra.pkl")
print("extra", x.shape)
