"""Item-level NOVA classification of Oxford WebQ 24h-recall items and UPF (% energy) recalculation.

Classification principles follow Rauber et al. Eur J Nutr 2021 (UK Biobank, NOVA):
  - items lacking detail are assigned to the form most commonly consumed in the UK;
  - when processed and unprocessed forms are both common, the less processed group is chosen.
Items absent from this extract (spreads, sauces, dips, plant milks, cream) enter via the UKB
26000-series food-group weights.

UKB provides servings, not grams or energy, per item. Energy (kJ) per serving is estimated by
regressing recall-level total energy (field 26002) on servings of every item and the extra food groups
(no intercept, non-negative least squares) across all valid recalls.
UPF% energy = sum over NOVA-4 items of servings x kJ/serving / total energy x 100, per recall,
then averaged over valid recalls.
Usage: python3 04_nova_items.py [main|upper|lower]   (upper/lower: all uncertain items classified as UPF / as not UPF)
Outputs (UKB_EXTRACT_DIR): upf_nova.pkl (upf_nova_upper.pkl, upf_nova_lower.pkl), items_raw.pkl, nova_item_energy.csv
"""
import os
import re
import time
import functools
print = functools.partial(print, flush=True)
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

HERE = Path(__file__).resolve().parent
C = Path(os.environ.get("UKB_EXTRACT_DIR", "../ukb_extracts"))
D = Path(os.environ.get("UKB_DATA_DIR", "../ukb_data"))

# Item-level classification: nova_classification.csv (UKB field, item, NOVA group, rationale, uncertain flag, source and
# comparison with Rauber et al. 2021). Oxford WebQ items enter as servings; items absent from the WebQ extract
# (spreads, sauces, dips, plant-based milks, cream) enter through the 26000-series food groups (g/day).
NOVA_TABLE = pd.read_csv(HERE / "nova_classification.csv", keep_default_na=False)
_webq = NOVA_TABLE["Source"] == "Oxford WebQ item"
ITEMS = {int(r["UKB field"]): (r["Item"], int(r["NOVA group"])) for _, r in NOVA_TABLE[_webq].iterrows()}
GROUPS = {int(r["UKB field"]): (r["Item"], int(r["NOVA group"])) for _, r in NOVA_TABLE[~_webq].iterrows()}

# items present in the extract but only flags/modifiers, handled separately or ignored
IGNORED = {100240, 100360, 100390, 100470, 100510, 100580, 100760, 100920, 100940, 101300, 101310, 101350,
           101390, 101430, 101470, 101510, 101550, 102080, 102130, 102250, 102400, 102520, 102700, 102800,
           102930, 103000, 103120, 103130, 103140, 103250, 103310, 103980, 103990, 104040, 104400, 104660, 104670}


def servings(s):
    s = pd.to_numeric(s, errors="coerce")
    s = s.replace({555: 0.5, 444: 0.25, 111: 0.5, 200: 2, 300: 3, 400: 4, 500: 5, 600: 6})
    return s.where(s >= 0)


if __name__ == "__main__":
    import sys
    SCENARIO = sys.argv[1] if len(sys.argv) > 1 else "main"   # main | upper | lower (extreme sensitivity analyses)
    t0 = time.time()
    if SCENARIO in ("upper", "lower"):
        # all items originally flagged as uncertain are treated as UPF (upper) or as not UPF (lower)
        unc = set(NOVA_TABLE.loc[NOVA_TABLE["Uncertain"] == "Yes", "UKB field"])
        for d_ in (ITEMS, GROUPS):
            for f in list(d_):
                if f in unc:
                    g = d_[f][1]
                    new = 4 if SCENARIO == "upper" else (3 if g == 4 else g)
                    d_[f] = (d_[f][0], new)
        print(SCENARIO, "scenario:", len(unc), "uncertain items reassigned")
    print(NOVA_TABLE["NOVA group"].value_counts().sort_index().to_dict(), "uncertain:", (NOVA_TABLE["Uncertain"] == "Yes").sum())

    item_fields = sorted(ITEMS)
    cache = C / "items_raw.pkl"
    if cache.exists():
        raw = pd.read_pickle(cache)
    else:
        f = D / "UKB619657_online_followup_2025Nov.csv"
        cols = ["eid"] + [f"p{x}_i{i}" for x in item_fields for i in range(5)]
        with open(f) as fh:
            hdr = [c.strip().strip('"') for c in fh.readline().rstrip("\n").split(",")]
        cols = [c for c in cols if c in hdr]
        dt = {c: "float32" for c in cols if c != "eid"}
        raw = pd.concat(pd.read_csv(f, usecols=cols, dtype=dt, chunksize=50000), ignore_index=True)
        raw.to_pickle(cache)
    print("items extracted", raw.shape, f"{time.time()-t0:.0f}s")

    diet = pd.read_pickle(C / "diet_raw.pkl")
    pop = pd.read_pickle(C / "pop.pkl")[["eid", "p31"]]
    diet = diet.merge(pop, on="eid", how="left")
    raw = raw.set_index("eid").reindex(diet["eid"]).reset_index()

    # stack valid recalls
    X_list, y_list, ids, inst = [], [], [], []
    for i in range(5):
        e = diet[f"p26002_i{i}"]
        kcal = e / 4.184
        hi = np.where(diet["p31"] == 1, 4200, 3500)
        ok = (e.notna() & diet[f"p100026_i{i}"].isna() & (kcal >= 500) & (kcal <= hi)).to_numpy()
        Xi = pd.DataFrame({f"i{x}": servings(raw[f"p{x}_i{i}"]) if f"p{x}_i{i}" in raw else np.nan
                           for x in item_fields}).fillna(0)
        for g in GROUPS:
            Xi[f"g{g}"] = diet[f"p{g}_i{i}"].fillna(0).to_numpy()
        X_list.append(Xi[ok])
        y_list.append(e[ok].to_numpy())
        ids.append(diet["eid"].to_numpy()[ok])
        inst.append(np.full(ok.sum(), i))
    X = pd.concat(X_list, ignore_index=True)
    y = np.concatenate(y_list)
    eid = np.concatenate(ids)
    print("valid recalls:", len(y))

    # milk added to drinks: scale the yes/varied flag by number of cups of that drink
    pairs = {"i100260": "i100250", "i100280": "i100270", "i100320": "i100310", "i100350": "i100330",
             "i100460": "i100400", "i100480": "i100410", "i100370": None, "i100490": None}
    for flag, cups in pairs.items():
        if cups:
            X[flag] = X[flag].clip(upper=1) * X[cups]

    coef, _ = nnls(X.to_numpy(dtype=float), y)
    pred = X.to_numpy() @ coef
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    print(f"energy model R2 = {r2:.3f}")
    info = {**{f"i{k}": v for k, v in ITEMS.items()}, **{f"g{k}": v for k, v in GROUPS.items()}}
    est = pd.DataFrame({"feature": X.columns, "kJ_per_unit": coef,
                        "item": [info[c][0] for c in X.columns], "NOVA": [info[c][1] for c in X.columns],
                        "mean_units_per_recall": X.mean().to_numpy()})
    est["energy_share_%"] = 100 * est["kJ_per_unit"] * est["mean_units_per_recall"] / y.mean()
    if SCENARIO == "main":
        est.to_csv(C / "nova_item_energy.csv", index=False)
    print(est.sort_values("energy_share_%", ascending=False).head(25).round(2).to_string(index=False))

    # UPF % energy per recall (numerator from estimated item energies, denominator = recorded total energy)
    nova = est.set_index("feature")["NOVA"]
    contrib = X.to_numpy() * coef
    by_group = {g: contrib[:, (nova == g).to_numpy()].sum(axis=1) for g in [1, 2, 3, 4]}
    upf = 100 * by_group[4] / y
    rec = pd.DataFrame({"eid": eid, "UPF_nova": np.clip(upf, 0, 100),
                       "NOVA1_pct": 100 * by_group[1] / y, "NOVA3_pct": 100 * by_group[3] / y})
    per = rec.groupby("eid").mean().reset_index()
    per.to_pickle(C / ("upf_nova.pkl" if SCENARIO == "main" else f"upf_nova_{SCENARIO}.pkl"))
    print(per.describe().round(1).to_string())

    old = pd.read_pickle(C / "analytic.pkl")[["eid", "UPF_pct"]].merge(per, on="eid")
    print("corr with food-group UPF approximation:", round(old[["UPF_pct", "UPF_nova"]].corr().iloc[0, 1], 3))
