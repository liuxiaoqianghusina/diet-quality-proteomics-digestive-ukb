"""Extract diet (24h recall), covariates and outcome dates from local UKB files.

Inputs: UK Biobank dataset CSV extracts in UKB_DATA_DIR (one CSV per UK Biobank Research Analysis Platform table).
Outputs (pickle) go to UKB_EXTRACT_DIR:
  diet_raw.pkl, pop.pkl, baseline.pkl, outcomes.pkl
"""
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd

D = Path(os.environ.get("UKB_DATA_DIR", "../ukb_data"))
OUT = Path(os.environ.get("UKB_EXTRACT_DIR", "../ukb_extracts"))
OUT.mkdir(parents=True, exist_ok=True)


def header(path):
    with open(path) as f:
        return [c.strip().strip('"') for c in f.readline().rstrip("\n").split(",")]


def read_cols(path, cols, name, dtype=None):
    t = time.time()
    hdr = header(path)
    missing = [c for c in cols if c not in hdr]
    if missing:
        print(f"[{name}] missing columns: {missing[:10]} ...", flush=True)
    use = [c for c in cols if c in hdr]
    parts = []
    for i, chunk in enumerate(pd.read_csv(path, usecols=use, chunksize=50000, dtype=dtype, low_memory=False)):
        parts.append(chunk)
        print(f"[{name}] chunk {i} rows={len(chunk)} {time.time()-t:.0f}s", flush=True)
    df = pd.concat(parts, ignore_index=True)
    df.to_pickle(OUT / f"{name}.pkl")
    print(f"[{name}] done {df.shape} in {time.time()-t:.0f}s", flush=True)


which = sys.argv[1:] or ["diet", "pop", "outcomes", "baseline"]

if "diet" in which:
    f = D / "UKB619657_online_followup_2025Nov.csv"
    hdr = header(f)
    cols = ["eid", "p20077"] + [c for c in hdr if re.match(r"p(26\d{3}|100026)_i[0-4]$", c)]
    cols += ["p21033", "p21034", "p21040", "p21043", "p21044"]
    read_cols(f, cols, "diet_raw")

if "pop" in which:
    f = D / "UKB619657_population_characteristics_2025Nov.csv"
    read_cols(f, ["eid", "p31", "p34", "p52", "p191", "p21022", "p22189"], "pop")

if "outcomes" in which:
    f = D / "UKB619657_health_outcomes_2025Nov.csv"
    hdr = header(f)
    fo = ["p131584", "p131590", "p131592", "p131594", "p131598", "p131626", "p131628", "p131636",
          "p131638", "p131640", "p131658", "p131666", "p131670", "p131674", "p131676", "p131682",
          "p131684", "p130836", "p130838", "p130840", "p130842", "p131036", "p131022", "p130894",
          "p130896", "p130906", "p131366", "p131368", "p131060", "p131052", "p131042"]
    cols = ["eid", "p40000_i0", "p42006", "p42018", "p42032", "p41270"] + fo
    cols += [c for c in hdr if re.match(r"p4000[56]_i\d+$", c) or re.match(r"p41280_a\d+$", c)]
    read_cols(f, cols, "outcomes", dtype=str)

if "baseline" in which:
    f = D / "UKB619657_assessment_center_2025Nov.csv"
    cols = ["eid", "p53_i0", "p54_i0", "p74_i0", "p21000_i0", "p21001_i0", "p21003_i0", "p20116_i0",
            "p1558_i0", "p22040_i0", "p6138_i0"]
    read_cols(f, cols, "baseline", dtype=str)
