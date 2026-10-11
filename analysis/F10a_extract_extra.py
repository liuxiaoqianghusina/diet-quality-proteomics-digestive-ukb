"""Extra fields for the tier-1 analyses: gallstone-related confounders, repeat-visit dates and OPCS procedures.
HRT ever (2814), oral contraceptive ever (2784), live births (2734), medication for cholesterol/BP/diabetes/hormones
(women 6153, men 6177), doctor-diagnosed diabetes (2443), weight change in the past year (2306),
assessment dates at the imaging visits (53_i2, 53_i3); OPCS-4 procedures (41272) with dates (41282).
"""
import re

import pandas as pd

from common import CACHE, UKB

D = UKB
A_COLS = ["eid", "p2814_i0", "p2784_i0", "p2734_i0", "p6153_i0", "p6177_i0", "p2443_i0", "p2306_i0", "p53_i2", "p53_i3"]
a = pd.concat(pd.read_csv(D / "UKB619657_assessment_center_2025Nov.csv", usecols=A_COLS, dtype=str, chunksize=50000),
              ignore_index=True)
a["eid"] = a["eid"].astype(int)
print(a.head(3).to_string())
for c in ["p6153_i0", "p6177_i0"]:
    print(c, a[c].dropna().head(8).tolist())

with open(D / "UKB619657_health_outcomes_2025Nov.csv") as fh:
    hdr = [c.strip().strip('"') for c in fh.readline().rstrip("\n").split(",")]
o_cols = ["eid", "p41272"] + [c for c in hdr if re.match(r"p41282_a\d+$", c)]
o = pd.concat(pd.read_csv(D / "UKB619657_health_outcomes_2025Nov.csv", usecols=o_cols, dtype=str, chunksize=50000),
              ignore_index=True)
o["eid"] = o["eid"].astype(int)
date_cols = sorted([c for c in o.columns if c.startswith("p41282_a")], key=lambda x: int(x.split("_a")[1]))
codes = o["p41272"].fillna("").str.split("|").explode().to_frame("code")
codes["pos"] = codes.groupby(level=0).cumcount()
hit = codes[codes["code"].str.startswith("J18")]          # J18 excision of gall bladder
vals = [o.at[i, date_cols[p]] if p < len(date_cols) else None for i, p in zip(hit.index, hit["pos"])]
chole = pd.Series(pd.to_datetime(vals, errors="coerce"), index=hit.index).groupby(level=0).min()
a = a.merge(pd.DataFrame({"eid": o.loc[chole.index, "eid"].to_numpy(), "cholecystectomy_date": chole.to_numpy()}),
            on="eid", how="left")
a.to_pickle(CACHE / "extra_tier1.pkl")
print(a.shape, "cholecystectomy recorded:", a.cholecystectomy_date.notna().sum())
