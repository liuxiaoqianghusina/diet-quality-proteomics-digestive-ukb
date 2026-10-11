"""Person-level analysis table and quality-controlled, plate-adjusted Olink matrix.

Exposures: UPF (% energy, item-level NOVA, data_prep/04_nova_items.py), uPDI, hPDI, energy-adjusted fibre.
Olink QC: samples with >50% of proteins missing excluded; proteins with >20% missing excluded; remaining
missing values imputed with the protein mean; plate effects removed by subtracting plate means; z-scored.
UKB-PPP consortium-selected participants are flagged (excluded from the main analyses).
"""
import numpy as np
import pandas as pd

from common import CACHE, NEG_FIELDS, OLINK, PC

person = pd.read_pickle(PC / "analytic.pkl").rename(columns={"met": "met_min", "fibre_adj": "fibre",
                                                             "UPF_pct": "UPF_groupapprox"})
upf = pd.read_pickle(PC / "upf_nova.pkl").rename(columns={"UPF_nova": "UPF"})
person = person.merge(upf, on="eid", how="left")

bio = pd.read_pickle(PC / "biochem.pkl").rename(columns={
    "p30720_i0": "cysc", "p30700_i0": "creat", "p30710_i0": "crp", "p30750_i0": "hba1c", "p30620_i0": "alt",
    "p30650_i0": "ast"})
extra = pd.read_pickle(PC / "extra.pkl")
extra = extra.assign(waist=pd.to_numeric(extra["p48_i0"], errors="coerce")).rename(
    columns={"p30730_i0": "ggt", "p30870_i0": "tg"})
for k, cols in NEG_FIELDS.items():
    extra[k] = pd.concat([pd.to_datetime(extra[c], errors="coerce") for c in cols], axis=1).min(axis=1)
qc = pd.read_pickle(PC / "olink_qc.pkl").rename(columns={"p30900_i0": "n_prot_measured", "p30901_i0": "plate",
                                                          "p30902_i0": "well", "p30903_i0": "consortium"})
qc["consortium"] = qc["consortium"].fillna(0).astype(int)
person = (person.merge(bio, on="eid", how="left")
          .merge(extra[["eid", "waist", "ggt", "tg"] + list(NEG_FIELDS)], on="eid", how="left")
          .merge(qc, on="eid", how="left"))

# ---------------------------------------------------------------- Olink
ol = pd.read_csv(OLINK, dtype={c: "float32" for c in pd.read_csv(OLINK, nrows=0).columns if c != "Participant.ID"})
ol = ol.rename(columns={"Participant.ID": "eid"}).set_index("eid")
n_raw = len(ol)
ol = ol[ol.isna().mean(axis=1) <= 0.5]
n_sample_qc = len(ol)
keep = ol.columns[ol.isna().mean() <= 0.20]
ol = ol[keep]
ol = ol.fillna(ol.mean())
plate = qc.set_index("eid").reindex(ol.index)["plate"]
ol = ol - ol.groupby(plate.to_numpy()).transform("mean") + ol.mean()
ol = ((ol - ol.mean()) / ol.std(ddof=0)).astype("float32")
ol.to_pickle(CACHE / "olink_qc_plateadj.pkl")

person["olink_qc"] = person["eid"].isin(ol.index)
person.to_pickle(CACHE / "person.pkl")
flow = {"Olink participants (instance 0)": n_raw, "after sample QC (<=50% missing)": n_sample_qc,
        "proteins retained (<=20% missing)": len(keep),
        "consortium-selected among QC-passed": int(person.loc[person.olink_qc, "consortium"].sum()),
        "QC-passed, randomly selected": int((person.olink_qc & (person.consortium == 0)).sum()),
        "... with >=1 valid 24h recall": int((person.olink_qc & (person.consortium == 0) & (person.n_recalls >= 1)).sum()),
        "24h recall >=1 (whole cohort)": int((person.n_recalls >= 1).sum())}
pd.Series(flow).to_csv(CACHE.parent / "results" / "flow_counts.csv", header=["n"])
print(pd.Series(flow).to_string())
print(person[["UPF", "UPF_groupapprox", "uPDI", "hPDI", "fibre"]].describe().round(1).to_string())
print("corr UPF(NOVA) vs group approximation:", round(person[["UPF", "UPF_groupapprox"]].corr().iloc[0, 1], 3))
